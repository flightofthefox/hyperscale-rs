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

use hyperscale_core::{Action, CrossShardExecutionRequest, TickBatchOutcome};
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
    Address, AggregateSignature, BeaconWitnessRoot, Block, BlockHeight, ConsensusReceipt,
    EventRoot, ExecutionCertificate, ExecutionMetadata, ExecutionOutcome, Finalization,
    GlobalReceipt, LocalKey, MerkleInclusionProof, Movement, ProvisionEntry, Provisions,
    RevealChain, SettledWrites, ShardId, ShardTrie, SignerBitfield, StateRoot, StateWrites,
    StoredReceipt, SubstateKey, TickId, TopologySchedule, TopologySnapshot, Transaction, TxHash,
    TxOutcome, ValidatorId, Verifiable, Verified, WeightedTimestamp, compute_global_receipt_root,
    read_amount,
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
    history: Mutex<Vec<(BlockHeight, SettledWrites)>>,
}

impl StubBase {
    /// Land a settled write set at `height`.
    fn apply(&self, height: BlockHeight, writes: &SettledWrites) {
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
            for (key, change) in writes.cells() {
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
    requests: Vec<CrossShardExecutionRequest>,
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
    receipts: BTreeMap<TickId, Vec<StoredReceipt>>,
    /// The charges each wave's tick held in reserve beside those
    /// receipts — what settles instead of them when the wave refuses.
    charges: BTreeMap<TickId, Vec<StoredReceipt>>,
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
            charges: BTreeMap::new(),
            base,
            local_shard,
        }
    }

    /// Commit a block carrying `txs` and `certificates`, then run whatever
    /// the schedule releases.
    pub fn commit(&mut self, txs: Vec<Transaction>, certificates: Vec<Finalization>) {
        self.height = self.height.next();
        // Settlement: a committed certificate's receipts reach state here,
        // in commit order and last-writer-wins per cell — the same
        // projection `merge_writes_from_receipts` performs into the JMT.
        for fw in &certificates {
            // Movements resolve against the state they land on, which is
            // whatever the certificates before this one already settled.
            let resolved = merge_writes_from_receipts(&fw.settling_receipts(), &mut |key| {
                self.base.substate(key)
            });
            self.base.apply(self.height, &resolved);
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

    /// Commit a counterpart's engagement for `tx_hashes`: the bundle a
    /// shard sends the payer because its own block committed the
    /// transaction. A payer's leg waits for this before executing, since
    /// the tick that runs it is the tick that attests it.
    pub fn engage(&mut self, from: ShardId, tx_hashes: &[TxHash]) {
        let bundle = Provisions::new(
            from,
            self.local_shard,
            self.height,
            WeightedTimestamp::from_millis(self.height.inner() * BLOCK_INTERVAL_MS),
            RevealChain::ZERO,
            MerkleInclusionProof::dummy(),
            tx_hashes
                .iter()
                .map(|h| ProvisionEntry::new(*h, vec![]))
                .collect(),
        );
        self.height = self.height.next();
        let block = match make_live_block(
            self.local_shard,
            self.height,
            self.height.inner() * BLOCK_INTERVAL_MS,
            ValidatorId::new(0),
            Vec::new(),
            Vec::new(),
        ) {
            Block::Live {
                header,
                transactions,
                certificates,
                witness_sources,
                ..
            } => Block::Live {
                header,
                transactions,
                certificates,
                provisions: Arc::new(vec![Arc::new(Verifiable::from(bundle))]),
                witness_sources,
            },
            sealed @ Block::Sealed { .. } => sealed,
        };
        let certified = certify(block, self.height.inner() * BLOCK_INTERVAL_MS);
        let actions = self.coord.on_block_committed(&self.topology, &certified);
        self.absorb(actions);
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
                Action::ExecuteTransactions { tick, requests, .. } => {
                    let release_at = match self.schedule {
                        Schedule::Eager => self.height,
                        Schedule::Lagged(n) => BlockHeight::new(
                            self.height.inner() + u64::try_from(n).expect("lag fits"),
                        ),
                    };
                    self.pending.push_back(PendingBatch {
                        tick,
                        requests,
                        release_at,
                    });
                }
                Action::ResolveTickWaves { resolutions } => {
                    for (tick_id, resolution) in &resolutions {
                        self.chain.resolve(tick_id, resolution);
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
        let PendingBatch { tick, requests, .. } = batch;
        let view = self
            .chain
            .view_at(BlockHeight::new(tick.inner().saturating_sub(1)));
        let snapshot = view.snapshot();
        let trie = self.snapshot.shard_trie();

        let tick_id = TickId::new(self.local_shard, tick);
        let executed: Vec<ExecutedTx> = requests
            .iter()
            .map(|request| {
                stub_execute(
                    &snapshot,
                    trie,
                    self.local_shard,
                    request.tx_hash,
                    &request.transaction,
                    request.reaches_beyond,
                )
            })
            .collect();
        let mut output = TickOutput::default();
        accumulate_tick_output(&mut output, &requests, &executed);
        let ExecutionOutputs {
            outcomes,
            results,
            fee_receipts,
            attested_work,
        } = split_execution_outputs(executed);
        self.receipts
            .entry(tick_id)
            .or_default()
            .extend(results.iter().cloned());
        self.charges
            .entry(tick_id)
            .or_default()
            .extend(fee_receipts.iter().cloned());
        let outcome = TickBatchOutcome {
            tick_id,
            results,
            tx_outcomes: outcomes,
            fee_receipts,
            attested_work,
        };

        self.outputs.push((tick, output.clone()));
        self.chain.append(tick, output);
        self.coord
            .on_execution_batch_completed(&self.topology, tick, outcome)
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
    pub fn settles_out_of_order(&self, certificates: &[TickId]) -> Option<TickId> {
        self.coord
            .certificates_settle_out_of_order(certificates, &HashSet::new())
    }

    /// The receipts `tick_id`'s tick produced.
    #[must_use]
    pub fn receipts_for(&self, tick_id: &TickId) -> Vec<StoredReceipt> {
        self.receipts.get(tick_id).cloned().unwrap_or_default()
    }

    /// The charges `tick_id`'s tick held in reserve.
    #[must_use]
    pub fn charges_for(&self, tick_id: &TickId) -> Vec<StoredReceipt> {
        self.charges.get(tick_id).cloned().unwrap_or_default()
    }

    /// The wave a transaction was assigned to, if the coordinator still
    /// tracks it.
    #[must_use]
    pub fn wave_of(&self, tx_hash: TxHash) -> Option<TickId> {
        self.coord.tick_assignment_for(tx_hash)
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

/// The amount cell a declared owner prefix credits, beside the counter
/// cell it writes.
///
/// A receipt says two kinds of thing and the pair has to be exercised
/// together: an exclusive write states the value it left, and a
/// commutative access states only what it moved. The second is what every
/// fee burn and every payment actually carries, and it is the one that
/// does not survive being dropped from a fold or applied twice.
#[must_use]
pub const fn vault_of(owner: [u8; 16]) -> SubstateKey {
    SubstateKey {
        owner: Address(owner),
        local: LocalKey([1; 16]),
    }
}

/// What each transaction credits to the vault of every cell it writes —
/// one per write, so the vault must always read exactly the counter.
pub const CREDIT: u128 = 1;

/// The cell a leg's abort charge reaches, and the amount it carries.
///
/// A cross-shard leg that completes here still owes a floor if the wave
/// refuses it, and that charge rides its own receipt — held in reserve
/// beside the effects, settled only if the effects are not. Separate
/// from [`vault_of`] so a test can read what was charged without
/// unpicking it from what was moved.
#[must_use]
pub const fn charge_of(owner: [u8; 16]) -> SubstateKey {
    SubstateKey {
        owner: Address(owner),
        local: LocalKey([2; 16]),
    }
}

/// What an abort charge carries.
pub const FLOOR: u128 = 7;

/// Decode a counter cell; an absent cell reads as zero.
#[must_use]
pub fn counter(bytes: Option<Vec<u8>>) -> u64 {
    bytes.map_or(0, |raw| {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&raw[..8]);
        u64::from_le_bytes(buf)
    })
}

/// Decode an amount cell; an absent cell reads as zero.
#[must_use]
pub fn amount(bytes: Option<Vec<u8>>) -> u128 {
    bytes.map_or(0, |raw| read_amount(&raw).expect("an amount cell"))
}

/// Execute one transaction: increment every cell it declares exclusively,
/// and credit the same owner's vault.
///
/// Reads run through the tick view, so the counter is a function of the
/// baseline — which is what makes a wrong baseline observable as a wrong
/// count rather than as nothing at all. The credit reads nothing, so it
/// is the opposite probe: it states what it moved, and any fold that
/// drops it or applies it twice shows up as a vault that disagrees with
/// the counter beside it.
fn stub_execute(
    snapshot: &impl SubstateDatabase,
    trie: &ShardTrie,
    local_shard: ShardId,
    tx_hash: TxHash,
    tx: &Arc<Verified<Transaction>>,
    abortable: bool,
) -> ExecutedTx {
    let mut cells = BTreeMap::new();
    let mut movements: BTreeMap<SubstateKey, Movement> = BTreeMap::new();
    // The payer this shard would charge: the first owner it holds.
    let mut charged: Option<[u8; 16]> = None;
    for key in tx.admission_write_keys() {
        // Only the owning shard applies a cell, exactly as `Locality`
        // scopes the engine's fold.
        if trie.shard_for_prefix(key.owner()) != local_shard {
            continue;
        }
        let cell = cell_of(key.owner().0);
        let next = counter(snapshot.substate(cell)) + 1;
        cells.insert(cell, Some(next.to_le_bytes().to_vec()));
        charged.get_or_insert_with(|| key.owner().0);
        let credit = movements.entry(vault_of(key.owner().0)).or_default();
        *credit = credit.then(Movement {
            credit: CREDIT,
            debit: 0,
        });
    }
    let writes = StateWrites { cells, movements };
    let receipt_hash = GlobalReceipt::new(
        true,
        EventRoot::ZERO,
        BeaconWitnessRoot::ZERO,
        writes_root(&writes),
    )
    .receipt_hash();
    let mut executed = ExecutedTx::new(
        tx_hash,
        ConsensusReceipt::Succeeded {
            receipt_hash,
            writes,
            beacon_witness_events: Vec::new(),
            events: Vec::new(),
        },
        ExecutionMetadata::empty(),
    );
    // A leg a wave can still discard carries its charge beside its
    // effects, exactly as the engine builds one for a cross-shard
    // member. Which of the two settles is the wave's decision, not this
    // shard's.
    if abortable {
        executed.fee_receipt = charged.map(stub_charge);
    }
    executed
}

/// The receipt a refused leg settles: the abort floor and nothing else.
fn stub_charge(owner: [u8; 16]) -> ConsensusReceipt {
    let mut writes = StateWrites::default();
    writes.movements.insert(
        charge_of(owner),
        Movement {
            credit: FLOOR,
            debit: 0,
        },
    );
    let receipt_hash = GlobalReceipt::new(
        true,
        EventRoot::ZERO,
        BeaconWitnessRoot::ZERO,
        writes_root(&writes),
    )
    .receipt_hash();
    ConsensusReceipt::Succeeded {
        receipt_hash,
        writes,
        beacon_witness_events: Vec::new(),
        events: Vec::new(),
    }
}

/// A committed `Finalization` settling `tick_id`, accepting every member.
///
/// The harness places these in blocks of its own choosing, which is how a
/// test states a settlement order rather than observing one.
#[must_use]
pub fn settle(tick_id: &TickId, receipts: &[StoredReceipt]) -> Finalization {
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
        *tick_id,
        WeightedTimestamp::from_millis(tick_id.block_height().inner() * BLOCK_INTERVAL_MS),
        compute_global_receipt_root(&outcomes),
        outcomes,
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::new(4),
    );
    Finalization::new(*tick_id, vec![Arc::new(ec)], receipts.to_vec())
}

/// A committed `Finalization` whose counterpart refused every member.
///
/// The local shard completed its half and carries the receipts to prove
/// it; the counterpart's certificate reports failure for the same
/// transactions, so the wave as a whole decided against them. Two
/// certificates for one wave is the ordinary cross-shard shape — what the
/// combine exists to reconcile.
#[must_use]
pub fn settle_refused_by_counterpart(
    tick_id: &TickId,
    counterpart: ShardId,
    receipts: &[StoredReceipt],
    charges: &[StoredReceipt],
) -> Finalization {
    // The local certificate reports what this shard did and names the
    // charge it holds against a refusal, which is what its own outcomes
    // carry. The stored receipts are the charges, because that is the
    // side of each outcome the wave's verdict selects.
    let outcomes: Vec<TxOutcome> = receipts
        .iter()
        .zip(charges)
        .map(|(receipt, charge)| {
            TxOutcome::with_fee(
                receipt.tx_hash,
                ExecutionOutcome::Succeeded {
                    receipt_hash: receipt.consensus.receipt_hash(),
                },
                charge.consensus.receipt_hash(),
                0,
            )
        })
        .collect();
    let local = ExecutionCertificate::new(
        *tick_id,
        WeightedTimestamp::from_millis(tick_id.block_height().inner() * BLOCK_INTERVAL_MS),
        compute_global_receipt_root(&outcomes),
        outcomes,
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::new(4),
    );
    let refused: Vec<TxOutcome> = receipts
        .iter()
        .map(|receipt| TxOutcome::new(receipt.tx_hash, ExecutionOutcome::Failed))
        .collect();
    let remote_id = TickId::new(counterpart, tick_id.block_height());
    let remote = ExecutionCertificate::new(
        remote_id,
        WeightedTimestamp::from_millis(tick_id.block_height().inner() * BLOCK_INTERVAL_MS),
        compute_global_receipt_root(&refused),
        refused,
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::new(4),
    );
    Finalization::new(
        *tick_id,
        vec![Arc::new(local), Arc::new(remote)],
        charges.to_vec(),
    )
}
