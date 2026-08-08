//! A sans-io driver for [`ExecutionCoordinator`].
//!
//! Mirrors the beacon and shard `CoordinatorSim` pattern, minus the
//! consensus: blocks arrive already committed, so there is no QC chaining,
//! view change, or round timeout to model. What is left is the part this
//! crate owns — tick composition, the tick chain, and wave resolution.
//!
//! # What it is for
//!
//! Control over *ordering*. The full simulator produces whatever interleaving
//! its network happens to produce; here a test states one. A tick's completion
//! can be held while later blocks commit, a wave's certificate can be placed
//! in a block of the test's choosing, and both can be varied while the
//! committed chain stays byte-identical. That is what makes the
//! schedule-invariance lane a real assertion rather than a restatement of
//! determinism: the committed chain is the input, local timing is the thing
//! being quantified over, and the tick outputs must not move.
//!
//! Execution is a stub, not the engine. It reads each transaction's declared
//! cells through [`TickChain::view_at`] exactly as the real handler does and
//! writes a value derived from what it read, so a tick's output depends on
//! its baseline — which is the only property the lane needs. The engine's own
//! fold is checked against the kernel's applied store on every batch it runs
//! (`BFT CRITICAL: VM fold diverged from the kernel apply`), so nothing here
//! is standing in for that.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use hyperscale_core::{Action, TickExecutionGroup, WaveExecutionResult};
use hyperscale_engine::ExecutedTx;
use hyperscale_engine::sharding::writes_root;
use hyperscale_execution::ExecutionCoordinator;
use hyperscale_execution::action_handlers::{
    ExecutionOutputs, accumulate_tick_output, split_execution_outputs,
};
use hyperscale_storage::{
    SubstateDatabase, SubstateStore, TickChain, TickOutput, VersionedStore,
    merge_writes_from_receipts,
};
use hyperscale_types::test_utils::{TestCommittee, certify, make_live_block};
use hyperscale_types::{
    Address, AggregateSignature, BeaconWitnessRoot, BlockHeight, ConsensusReceipt, EventRoot,
    ExecutionCertificate, ExecutionMetadata, ExecutionOutcome, FinalizedWave, GlobalReceipt,
    LocalKey, MerkleInclusionProof, ShardId, ShardTrie, SignerBitfield, StateRoot, StateWrites,
    StoredReceipt, SubstateKey, TopologySchedule, TopologySnapshot, Transaction, TxHash, TxOutcome,
    ValidatorId, Verifiable, Verified, WaveCertificate, WaveId, WeightedTimestamp,
    compute_global_receipt_root,
};

/// The shard a single-shard fixture runs on.
pub const SHARD: ShardId = ShardId::ROOT;

/// The local shard of a two-shard fixture. A declared prefix routes by its
/// leading bit, so `test_prefix(seed)` with `seed < 128` lands here and
/// `seed >= 128` lands on its sibling.
pub const LEFT: ShardId = ShardId::leaf(1, 0);

/// Milliseconds between synthesised block timestamps. Large enough that
/// nothing in the wave machinery reaches a deadline over a short run.
const BLOCK_INTERVAL_MS: u64 = 500;

/// When the driver releases a dispatched tick's completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// Complete each tick before the commit that dispatched it returns.
    Eager,
    /// Hold every completion until `n` further blocks have committed, so
    /// composition runs well ahead of execution. Ticks still complete in
    /// order — they are serial by construction — but the resolutions they
    /// gate are emitted later, which is the timing the lane quantifies over.
    Lagged(usize),
}

/// The settled base every tick reads through, versioned by height.
///
/// A tick reads it as of its own anchor, not as of whatever this replica
/// has persisted, so the harness has to keep the history: a settlement at
/// height 9 must be invisible to a read anchored at 8, exactly as it is
/// in the real store.
#[derive(Default)]
struct StubBase {
    /// Settled writes in commit order, each with the height that applied
    /// them.
    history: Mutex<Vec<(BlockHeight, StateWrites)>>,
}

impl StubBase {
    /// Land a settled write set at `height`.
    fn apply(&self, height: BlockHeight, writes: &StateWrites) {
        self.history
            .lock()
            .expect("base lock")
            .push((height, writes.clone()));
    }

    /// The cells as of `height`: every settled write at or below it, in
    /// commit order, last writer per cell.
    fn cells_at(&self, height: BlockHeight) -> HashMap<SubstateKey, Vec<u8>> {
        let mut cells = HashMap::new();
        for (applied, writes) in self.history.lock().expect("base lock").iter() {
            if *applied > height {
                break;
            }
            for (key, change) in &writes.cells {
                match change {
                    Some(value) => {
                        cells.insert(*key, value.clone());
                    }
                    None => {
                        cells.remove(key);
                    }
                }
            }
        }
        cells
    }
}

/// A snapshot of [`StubBase`] — cloned, so a fold cannot mutate through it.
struct StubSnapshot(HashMap<SubstateKey, Vec<u8>>);

impl SubstateDatabase for StubSnapshot {
    fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.0.get(&key).cloned()
    }
}

impl SubstateDatabase for StubBase {
    fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.cells_at(BlockHeight::new(u64::MAX)).get(&key).cloned()
    }
}

impl SubstateStore for StubBase {
    type Snapshot<'a> = StubSnapshot;

    fn snapshot(&self) -> Self::Snapshot<'_> {
        StubSnapshot(self.cells_at(BlockHeight::new(u64::MAX)))
    }
    fn jmt_height(&self) -> BlockHeight {
        BlockHeight::GENESIS
    }
    fn state_root(&self) -> StateRoot {
        StateRoot::ZERO
    }
    fn get_substate_at_height(
        &self,
        _key: SubstateKey,
        _block_height: BlockHeight,
    ) -> Option<Option<Vec<u8>>> {
        None
    }
    fn generate_merkle_proofs(
        &self,
        _keys: &[SubstateKey],
        _block_height: BlockHeight,
    ) -> Option<MerkleInclusionProof> {
        None
    }
}

impl VersionedStore for StubBase {
    fn snapshot_at(&self, height: BlockHeight) -> Self::Snapshot<'_> {
        StubSnapshot(self.cells_at(height))
    }
    fn substate_bytes_at(&self, _height: BlockHeight) -> Option<u64> {
        None
    }
}

/// A tick dispatched but not yet completed.
struct PendingBatch {
    tick: BlockHeight,
    groups: Vec<TickExecutionGroup>,
    /// Height at which the schedule releases it.
    release_at: BlockHeight,
}

/// The driver.
pub struct ExecutionSim {
    coord: ExecutionCoordinator,
    topology: TopologySchedule,
    snapshot: Arc<TopologySnapshot>,
    chain: Arc<TickChain<StubBase>>,
    schedule: Schedule,
    pending: VecDeque<PendingBatch>,
    height: BlockHeight,
    /// Every tick output produced, in production order — what the
    /// invariance lane compares.
    outputs: Vec<(BlockHeight, TickOutput)>,
    /// The receipts each wave's tick produced, so a test can settle a wave
    /// with what it actually executed rather than with a stand-in.
    receipts: BTreeMap<WaveId, Vec<StoredReceipt>>,
    /// The settled state every tick reads through, so the harness models
    /// the whole path: a committed certificate's receipts land here in
    /// commit order, exactly as `merge_writes_from_receipts` lands them in
    /// the JMT.
    base: Arc<StubBase>,
    local_shard: ShardId,
}

impl ExecutionSim {
    /// A single-shard driver over a four-validator committee, this node
    /// seated first. Every wave it composes is single-shard, so every
    /// write is determined at commit.
    #[must_use]
    pub fn new(schedule: Schedule) -> Self {
        Self::with_shards(schedule, 1, SHARD)
    }

    /// A driver over a `num_shards`-wide topology with `local_shard` as
    /// this node's seat. A transaction declaring prefixes on both sides of
    /// the partition composes a cross-shard wave, whose writes are
    /// provisional until its certificate commits.
    #[must_use]
    pub fn with_shards(schedule: Schedule, num_shards: u64, local_shard: ShardId) -> Self {
        let committee = TestCommittee::new(4, 42);
        let base = Arc::new(StubBase::default());
        let snapshot = Arc::new(committee.topology_snapshot(num_shards));
        Self {
            coord: ExecutionCoordinator::new(ValidatorId::new(0), local_shard),
            topology: TopologySchedule::single(Arc::clone(&snapshot)),
            snapshot,
            chain: Arc::new(TickChain::new(Arc::clone(&base))),
            schedule,
            pending: VecDeque::new(),
            height: BlockHeight::GENESIS,
            outputs: Vec::new(),
            receipts: BTreeMap::new(),
            base,
            local_shard,
        }
    }

    /// Commit a block carrying `txs` and `certificates`, then run whatever
    /// the schedule releases.
    pub fn commit(&mut self, txs: Vec<Transaction>, certificates: Vec<FinalizedWave>) {
        self.height = self.height.next();
        // Settlement: a committed certificate's receipts reach state here,
        // in commit order and last-writer-wins per cell — the same
        // projection `merge_writes_from_receipts` performs into the JMT.
        for fw in &certificates {
            self.base.apply(
                self.height,
                &merge_writes_from_receipts(&fw.settling_receipts()),
            );
        }
        let block = make_live_block(
            self.local_shard,
            self.height,
            self.height.inner() * BLOCK_INTERVAL_MS,
            ValidatorId::new(0),
            txs.into_iter().map(Arc::new).collect(),
            certificates
                .into_iter()
                .map(|fw| Arc::new(Verifiable::from(fw)))
                .collect(),
        );
        let certified = certify(block, self.height.inner() * BLOCK_INTERVAL_MS);
        let actions = self.coord.on_block_committed(&self.topology, &certified);
        self.absorb(actions);
        // Persistence follows the commit, which is when the chain evicts
        // the folds it believes the base now covers.
        self.chain.prune_persisted(self.height);
        self.release_due();
    }

    /// Complete every tick the schedule has released at the current height.
    fn release_due(&mut self) {
        while self
            .pending
            .front()
            .is_some_and(|batch| batch.release_at <= self.height)
        {
            let batch = self.pending.pop_front().expect("checked");
            let actions = self.run_batch(batch);
            self.absorb(actions);
        }
    }

    /// Drain every held completion regardless of schedule — what a test
    /// calls once it has finished committing and wants the pipeline empty.
    pub fn drain(&mut self) {
        while !self.pending.is_empty() {
            let batch = self.pending.pop_front().expect("checked");
            let actions = self.run_batch(batch);
            self.absorb(actions);
        }
    }

    /// Route the coordinator's actions: queue dispatches, apply tick-chain
    /// maintenance inline exactly as the shard thread does, ignore the rest
    /// (votes and broadcasts have no bearing on what is being measured).
    fn absorb(&mut self, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::ExecuteTransactions { tick, groups, .. } => {
                    let release_at = match self.schedule {
                        Schedule::Eager => self.height,
                        Schedule::Lagged(n) => BlockHeight::new(
                            self.height.inner() + u64::try_from(n).expect("lag fits"),
                        ),
                    };
                    self.pending.push_back(PendingBatch {
                        tick,
                        groups,
                        release_at,
                    });
                }
                Action::ResolveTickWaves { resolutions } => {
                    for (wave_id, resolution) in &resolutions {
                        self.chain.resolve(wave_id, resolution);
                    }
                }
                Action::ClearTickChain => self.chain.clear(),
                _ => {}
            }
        }
    }

    /// Execute one tick against its baseline and feed the result back.
    ///
    /// The same order the real handler uses: read through the previous
    /// tick's view, fold the output, append it, and only then notify — the
    /// coordinator dispatches the next tick on that notification and its
    /// baseline has to include this one.
    fn run_batch(&mut self, batch: PendingBatch) -> Vec<Action> {
        let PendingBatch { tick, groups, .. } = batch;
        let view = self
            .chain
            .view_at(BlockHeight::new(tick.inner().saturating_sub(1)));
        let snapshot = view.snapshot();
        let trie = self.snapshot.shard_trie();

        let mut output = TickOutput::default();
        let mut waves = Vec::with_capacity(groups.len());
        for group in &groups {
            let executed: Vec<ExecutedTx> = group
                .requests
                .iter()
                .map(|request| {
                    stub_execute(
                        &snapshot,
                        trie,
                        self.local_shard,
                        request.tx_hash,
                        &request.transaction,
                    )
                })
                .collect();
            accumulate_tick_output(&mut output, group, &executed);
            let ExecutionOutputs {
                outcomes,
                results,
                fee_receipts,
                attested_work,
            } = split_execution_outputs(executed);
            self.receipts
                .entry(group.wave_id.clone())
                .or_default()
                .extend(results.iter().cloned());
            waves.push(WaveExecutionResult {
                wave_id: group.wave_id.clone(),
                results,
                tx_outcomes: outcomes,
                fee_receipts,
                attested_work,
            });
        }

        self.outputs.push((tick, output.clone()));
        self.chain.append(tick, output);
        self.coord
            .on_execution_batch_completed(&self.topology, tick, waves)
    }

    /// Every tick output this run produced, in order.
    #[must_use]
    pub fn outputs(&self) -> &[(BlockHeight, TickOutput)] {
        &self.outputs
    }

    /// The readable value of `key` as of the tick chain's tip: settled
    /// state with every retained fold over it — what the next tick would
    /// execute against.
    #[must_use]
    pub fn read(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.chain.view_at(self.height).snapshot().substate(key)
    }

    /// The settled value of `key` — what committed certificates have put
    /// into state, with no unresolved fold over it.
    #[must_use]
    pub fn settled(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.base.substate(key)
    }

    /// Whether a block carrying `certificates` in this order would settle
    /// two cell-sharing waves out of the order they executed in — the
    /// pre-vote gate's question. No ancestor blocks here: the harness
    /// models one block at a time.
    #[must_use]
    pub fn settles_out_of_order(&self, certificates: &[WaveId]) -> Option<WaveId> {
        self.coord
            .certificates_settle_out_of_order(certificates, &HashSet::new())
    }

    /// The receipts `wave_id`'s tick produced.
    #[must_use]
    pub fn receipts_for(&self, wave_id: &WaveId) -> Vec<StoredReceipt> {
        self.receipts.get(wave_id).cloned().unwrap_or_default()
    }

    /// The wave a transaction was assigned to, if the coordinator still
    /// tracks it.
    #[must_use]
    pub fn wave_of(&self, tx_hash: TxHash) -> Option<WaveId> {
        self.coord.get_wave_assignment(tx_hash)
    }
}

/// The cell a declared owner prefix maps to.
#[must_use]
pub const fn cell_of(owner: [u8; 16]) -> SubstateKey {
    SubstateKey {
        owner: Address(owner),
        local: LocalKey([0; 16]),
    }
}

/// Decode a counter cell; an absent cell reads as zero.
#[must_use]
pub fn counter(bytes: Option<Vec<u8>>) -> u64 {
    bytes.map_or(0, |raw| {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&raw[..8]);
        u64::from_le_bytes(buf)
    })
}

/// Execute one transaction: increment every cell it declares exclusively.
///
/// Reads run through the tick view, so the receipt is a function of the
/// baseline — which is what makes a wrong baseline observable as a wrong
/// count rather than as nothing at all.
fn stub_execute(
    snapshot: &impl SubstateDatabase,
    trie: &ShardTrie,
    local_shard: ShardId,
    tx_hash: TxHash,
    tx: &Arc<Verified<Transaction>>,
) -> ExecutedTx {
    let mut cells = BTreeMap::new();
    for key in tx.admission_write_keys() {
        // Only the owning shard applies a cell, exactly as `Locality`
        // scopes the engine's fold.
        if trie.shard_for_prefix(key.owner()) != local_shard {
            continue;
        }
        let cell = cell_of(key.owner().0);
        let next = counter(snapshot.substate(cell)) + 1;
        cells.insert(cell, Some(next.to_le_bytes().to_vec()));
    }
    let writes = StateWrites { cells };
    let receipt_hash = GlobalReceipt::new(
        true,
        EventRoot::ZERO,
        BeaconWitnessRoot::ZERO,
        writes_root(&writes),
    )
    .receipt_hash();
    ExecutedTx::new(
        tx_hash,
        ConsensusReceipt::Succeeded {
            receipt_hash,
            writes,
            beacon_witness_events: Vec::new(),
            events: Vec::new(),
        },
        ExecutionMetadata::empty(),
    )
}

/// A committed `FinalizedWave` settling `wave_id`, accepting every member.
///
/// The harness places these in blocks of its own choosing, which is how a
/// test states a settlement order rather than observing one.
#[must_use]
pub fn settle(wave_id: &WaveId, receipts: &[StoredReceipt]) -> FinalizedWave {
    let outcomes: Vec<TxOutcome> = receipts
        .iter()
        .map(|receipt| {
            TxOutcome::new(
                receipt.tx_hash,
                ExecutionOutcome::Succeeded {
                    receipt_hash: receipt.consensus.receipt_hash(),
                },
            )
        })
        .collect();
    let ec = ExecutionCertificate::new(
        wave_id.clone(),
        WeightedTimestamp::from_millis(wave_id.block_height().inner() * BLOCK_INTERVAL_MS),
        compute_global_receipt_root(&outcomes),
        outcomes,
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::new(4),
    );
    let certificate = WaveCertificate::new(wave_id.clone(), vec![Arc::new(ec)]);
    FinalizedWave::new(Arc::new(certificate), receipts.to_vec())
}

/// A committed `FinalizedWave` whose counterpart refused every member.
///
/// The local shard completed its half and carries the receipts to prove
/// it; the counterpart's certificate reports failure for the same
/// transactions, so the wave as a whole decided against them. Two
/// certificates for one wave is the ordinary cross-shard shape — what the
/// combine exists to reconcile.
#[must_use]
pub fn settle_refused_by_counterpart(
    wave_id: &WaveId,
    counterpart: ShardId,
    receipts: &[StoredReceipt],
) -> FinalizedWave {
    let local = settle(wave_id, receipts);
    let refused: Vec<TxOutcome> = receipts
        .iter()
        .map(|receipt| TxOutcome::new(receipt.tx_hash, ExecutionOutcome::Failed))
        .collect();
    let remote_id = WaveId::new(
        counterpart,
        wave_id.block_height(),
        std::iter::once(wave_id.shard_id()).collect(),
    );
    let remote = ExecutionCertificate::new(
        remote_id,
        WeightedTimestamp::from_millis(wave_id.block_height().inner() * BLOCK_INTERVAL_MS),
        compute_global_receipt_root(&refused),
        refused,
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::new(4),
    );
    let certificate = WaveCertificate::new(
        wave_id.clone(),
        vec![
            Arc::new(local.execution_certificates()[0].as_unverified().clone()),
            Arc::new(remote),
        ],
    );
    FinalizedWave::new(Arc::new(certificate), receipts.to_vec())
}
