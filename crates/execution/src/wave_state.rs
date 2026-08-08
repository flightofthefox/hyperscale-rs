//! Per-wave execution state.
//!
//! One `WaveState` owns an in-flight wave from block commit through
//! finalization: per-tx execution progress, local vote generation, and
//! cross-shard EC collection all live here.
//!
//! ## Wave lifecycle
//!
//! 1. **Created** in `ExecutionCoordinator::on_block_committed` when waves are assigned
//!    for a newly committed block. At creation, each tx's already-received
//!    provisions are folded in — if every tx is fully provisioned at that point
//!    (single-shard waves are trivially so), `all_provisioned_at` is set to the
//!    block's own height.
//! 2. **Waits for provisions** until every tx has all required remote shards'
//!    state. Each `mark_provisioned(tx, at_height)` call that completes the
//!    final missing tx sets `all_provisioned_at = Some(at_height)` and returns
//!    `true` — the caller uses that as the dispatch trigger.
//! 3. **Executes atomically** — one `ExecuteTransactions` /
//!    `ExecuteCrossShardTransactions` action per wave. Results land via
//!    `record_execution_result`.
//! 4. **Votes** once all results present (or at the `wave_start_ts +
//!    WAVE_TIMEOUT` deadline if still not provisioned — entire wave aborts).
//! 5. **Collects ECs** from all participating shards via
//!    `add_execution_certificate`. When every tx is covered (or aborted, which
//!    is terminal-covered), the wave is complete and ready for finalization.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use hyperscale_core::{CrossShardExecutionRequest, TickExecutionGroup};
use hyperscale_types::{
    BlockHash, BlockHeight, DeclaredKey, ExecutionCertificate, ExecutionOutcome, FinalizedWave,
    GlobalReceiptRoot, Mode, RevealChain, ShardId, StoredReceipt, Transaction, TransactionDecision,
    TxHash, TxOutcome, Verifiable, Verified, WAVE_TIMEOUT, WaveCertificate, WaveId,
    WeightedTimestamp, compute_global_receipt_root,
};

use crate::provisional::ProvisionalCells;
use crate::provisioning::ProvisioningTracker;

/// A wave whose local execution disagreed with the quorum's.
///
/// The receipt root the validator voted against the one its committee
/// certified: direct proof that this node's execution produced different
/// writes from the same committed chain.
#[derive(Debug, Clone)]
pub struct Divergence {
    /// The wave whose roots disagreed.
    pub wave_id: WaveId,
    /// The block whose commit created it.
    pub block_hash: BlockHash,
    /// The root this validator voted.
    pub local_root: GlobalReceiptRoot,
    /// The root its committee certified.
    pub ec_root: GlobalReceiptRoot,
}

/// Age at which a still-alive wave emits a single diagnostic warning.
///
/// Under the two-stage lifecycle (windows gate admission, the wave/execution
/// timeout owns termination), every tx is supposed to terminate with a
/// `WaveCertificate` — success via vote aggregation or abort via the
/// deterministic all-abort fallback. The threshold is set past
/// `WAVE_TIMEOUT` so waves resolving via the normal abort path
/// (including cross-shard cert gossip) pass silently. If a wave reaches
/// this age, the post-inclusion termination guarantee has failed and the
/// dump is invariant-violation diagnostics, not routine load noise.
pub const WAVE_OVERDUE_WARN: Duration = Duration::from_secs(WAVE_TIMEOUT.as_secs() * 2);

/// Per-wave state across the entire execution lifecycle.
#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)] // independent lifecycle flags, not config knobs
pub struct WaveState {
    // ── Identity ────────────────────────────────────────────────────────
    wave_id: WaveId,
    block_hash: BlockHash,

    // ── Tx layout (in block order) ──────────────────────────────────────
    tx_hashes: Vec<TxHash>,
    /// Participating shards per tx — the shards whose ECs must cover each tx
    /// for completion. Always includes local shard; cross-shard txs include
    /// remote shards too.
    participating_shards: HashMap<TxHash, BTreeSet<ShardId>>,
    /// O(1) membership check (mirrors `tx_hashes`).
    tx_hash_set: HashSet<TxHash>,
    /// Transactions owned by the wave, used to build execution requests at
    /// dispatch time.
    transactions: HashMap<TxHash, Arc<Verified<Transaction>>>,

    /// BFT-authenticated weighted timestamp of the wave-starting block.
    /// Anchor for wave-level wall-clock timeouts (wave abort, vote anchor
    /// in the timeout path).
    wave_start_ts: WeightedTimestamp,
    /// The wave-starting block's reveal chain — the randomness anchor for
    /// every transaction this shard committed in that block.
    wave_start_reveal: RevealChain,

    // ── Provisioning phase ──────────────────────────────────────────────
    /// Txs whose required remote-shard provisions have all arrived.
    provisioned_txs: HashSet<TxHash>,
    /// Per-tx earliest ready timestamp. `all_provisioned_at` is
    /// the max across this map — deterministic regardless of call order.
    provisioned_tx_ts: HashMap<TxHash, WeightedTimestamp>,
    /// The weighted timestamp at which every tx in the wave became
    /// ready. `None` until `provisioned_txs` is full.
    all_provisioned_at: Option<WeightedTimestamp>,

    // ── Engagement coverage (payer-shard legs) ───────────────────────
    /// Per-tx, the counterpart shards whose engagement echo this shard —
    /// the transaction's fee payer — still waits for before voting.
    /// Recorded at wave creation and drained as echoes commit; empty for
    /// every other wave, which votes on execution alone.
    engagement_pending: HashMap<TxHash, BTreeSet<ShardId>>,
    /// The weighted timestamp past which the wave votes without full
    /// engagement coverage, aborting whatever is still uncovered: the
    /// latest member's validity end plus the echo margin. `None` when
    /// nothing in the wave is engagement-gated.
    engagement_deadline: Option<WeightedTimestamp>,
    /// Per-tx fee receipts the engine built alongside the execution
    /// receipt, for the cross-shard transactions this shard pays for.
    /// An abort settles one of these: the transaction's own effects are
    /// discarded, the payer's floor is not.
    fee_receipts: HashMap<TxHash, StoredReceipt>,
    /// What this shard attests it did per transaction, carried from
    /// execution onto the outcomes it votes.
    attested_work: HashMap<TxHash, u64>,
    /// Members that have joined a tick. A wave usually joins one whole,
    /// but a member whose declared cells another wave holds provisionally
    /// waits for a later tick, so membership is tracked per transaction.
    dispatched: HashSet<TxHash>,

    // ── Local execution outputs ─────────────────────────────────────────
    /// Execution results from the engine (per-tx). Non-abort outcomes only.
    execution_results: HashMap<TxHash, ExecutionOutcome>,
    /// Local receipts from the engine, one per executed tx. Drained into the
    /// `FinalizedWave` at finalization via `take_receipt`. Scoping these to
    /// the wave (rather than a process-wide cache) prevents a receipt from a
    /// locally-executed tx from leaking into a `FinalizedWave` whose EC later
    /// attests that tx as `Aborted` — the `ExtraReceipt` race.
    execution_receipts: HashMap<TxHash, StoredReceipt>,
    /// Explicit aborts from `ConflictDetector`. Distinct from remote-reported
    /// aborts in `tracker_aborted` — these are local pre-vote decisions.
    explicit_aborts: HashSet<TxHash>,
    /// Whether the local vote has been emitted (`build_vote_data` called once).
    voted: bool,
    /// `global_receipt_root` carried on this validator's own emitted vote.
    /// Set by `build_vote_data`. Reconciled against `admitted_local_ec_root`
    /// to detect divergence (see `reconcile_local_ec_decision`).
    local_vote_global_receipt_root: Option<GlobalReceiptRoot>,
    /// `global_receipt_root` from the admitted local EC. Set by
    /// `add_execution_certificate` for the `is_local` arm. May arrive
    /// before the local vote (cross-shard race where peers aggregate the
    /// EC before this validator finishes executing).
    admitted_local_ec_root: Option<GlobalReceiptRoot>,
    /// Set when the admitted local EC's `global_receipt_root` disagreed
    /// with `local_vote_global_receipt_root`. Bars the wave from
    /// finalizing locally so divergent receipts cannot enter the
    /// `finalized` store, propagate via `cert_bloom`, or be re-served on
    /// sync — which matters for the window before the coordinator
    /// escalates, not as an outcome. There is no recovery from here: a
    /// wrong tick output is the baseline every later tick reads.
    locally_divergent: bool,
    /// The mismatch behind the latch, until the coordinator reports it.
    divergence: Option<Divergence>,
    /// Whether the local EC has been added to `execution_certificates`.
    /// Gates wave completion: `is_complete` requires the local EC to be
    /// present. Independent of the canonical-root reconciliation —
    /// `locally_divergent` carries the divergence verdict separately.
    local_ec_emitted: bool,
    /// Latches `log_if_overdue`: fires once per wave after crossing the
    /// `WAVE_OVERDUE_WARN` threshold. Under ts-based ages we can't rely on
    /// exact equality (commits can skip over any given ms value).
    overdue_warned: bool,

    // ── Cross-shard EC collection ───────────────────────────────────────
    /// Per-tx, which shards have reported via an EC.
    covered_shards: HashMap<TxHash, BTreeSet<ShardId>>,
    /// Per-tx, whether any shard's EC reported abort. Terminal — an aborted tx
    /// doesn't require further remote coverage.
    tracker_aborted: HashSet<TxHash>,
    /// Per-tx, whether any shard's EC reported a non-success outcome.
    tx_has_failure: HashSet<TxHash>,
    /// All collected ECs (local + remote).
    execution_certificates: Vec<Arc<Verified<ExecutionCertificate>>>,
    /// Deduplication of received ECs by `wave_id`. At most one valid EC
    /// exists per `wave_id` (signature verification upstream ensures this),
    /// so `wave_id` is a content-equivalent identity for dedup.
    seen_ec_wave_ids: HashSet<WaveId>,
}

impl WaveState {
    /// Create a new wave state.
    ///
    /// `txs` is in block order. Each entry is `(transaction, participating_shards)`.
    /// `single_shard` indicates whether this is a single-shard wave (`remote_shards` empty);
    /// if so, `all_provisioned_at` / `all_provisioned_at` are set to the
    /// wave-starting block's height/timestamp immediately.
    #[must_use]
    pub fn new(
        wave_id: WaveId,
        block_hash: BlockHash,
        wave_start_ts: WeightedTimestamp,
        wave_start_reveal: RevealChain,
        txs: Vec<(Arc<Verifiable<Transaction>>, BTreeSet<ShardId>)>,
        single_shard: bool,
    ) -> Self {
        let mut tx_hashes: Vec<TxHash> = Vec::with_capacity(txs.len());
        let mut transactions: HashMap<TxHash, Arc<Verified<Transaction>>> =
            HashMap::with_capacity(txs.len());
        let mut participating_shards: HashMap<TxHash, BTreeSet<ShardId>> =
            HashMap::with_capacity(txs.len());
        let mut covered_shards: HashMap<TxHash, BTreeSet<ShardId>> =
            HashMap::with_capacity(txs.len());

        for (tx, shards) in txs {
            let h = tx.hash();
            tx_hashes.push(h);
            // Block-container entries decoded from the wire land as
            // `Unverified`; lift via `from_persisted` under the same
            // BFT-transitive trust that gates the containing block. Honest
            // live-consensus blocks already carry `Verified` entries (the
            // `.into_verified()` arm short-circuits without re-validating).
            let verified: Arc<Verified<Transaction>> = match (*tx).clone().into_verified() {
                Ok(v) => Arc::new(v),
                Err(raw) => Arc::new(Verified::<Transaction>::from_persisted(raw)),
            };
            transactions.insert(h, verified);
            participating_shards.insert(h, shards);
            covered_shards.insert(h, BTreeSet::new());
        }

        let tx_hash_set: HashSet<TxHash> = tx_hashes.iter().copied().collect();

        // Single-shard waves are trivially provisioned at creation.
        let (provisioned_txs, provisioned_tx_ts, all_provisioned_at) = if single_shard {
            let ts_map: HashMap<TxHash, WeightedTimestamp> =
                tx_hashes.iter().map(|h| (*h, wave_start_ts)).collect();
            (tx_hash_set.clone(), ts_map, Some(wave_start_ts))
        } else {
            (HashSet::new(), HashMap::new(), None)
        };

        Self {
            wave_id,
            block_hash,
            wave_start_ts,
            wave_start_reveal,
            tx_hashes,
            participating_shards,
            tx_hash_set,
            transactions,
            provisioned_txs,
            provisioned_tx_ts,
            all_provisioned_at,
            engagement_pending: HashMap::new(),
            engagement_deadline: None,
            fee_receipts: HashMap::new(),
            attested_work: HashMap::new(),
            dispatched: HashSet::new(),
            execution_results: HashMap::new(),
            execution_receipts: HashMap::new(),
            explicit_aborts: HashSet::new(),
            voted: false,
            local_vote_global_receipt_root: None,
            admitted_local_ec_root: None,
            locally_divergent: false,
            divergence: None,
            local_ec_emitted: false,
            overdue_warned: false,
            covered_shards,
            tracker_aborted: HashSet::new(),
            tx_has_failure: HashSet::new(),
            execution_certificates: Vec::new(),
            seen_ec_wave_ids: HashSet::new(),
        }
    }

    // ── Identity getters ────────────────────────────────────────────────

    /// The wave's identity ([`WaveId`]).
    #[must_use]
    pub const fn wave_id(&self) -> &WaveId {
        &self.wave_id
    }

    /// Hash of the wave-starting block.
    #[must_use]
    pub const fn block_hash(&self) -> BlockHash {
        self.block_hash
    }

    /// Height of the wave-starting block (mirrors `wave_id.block_height`).
    #[must_use]
    pub const fn block_height(&self) -> BlockHeight {
        self.wave_id.block_height()
    }

    /// Transaction hashes in this wave, in block order.
    #[must_use]
    pub fn tx_hashes(&self) -> &[TxHash] {
        &self.tx_hashes
    }

    // ── Provisioning ────────────────────────────────────────────────────

    /// Whether this wave has reached full provisioning.
    #[must_use]
    pub const fn is_fully_provisioned(&self) -> bool {
        self.all_provisioned_at.is_some()
    }

    /// Whether every member has joined a tick.
    #[must_use]
    pub fn fully_dispatched(&self) -> bool {
        self.tx_hashes
            .iter()
            .all(|tx_hash| self.dispatched.contains(tx_hash))
    }

    /// How this wave's members declare they will reach each cell.
    ///
    /// What a later tick has to be compatible with while the wave is
    /// unresolved: its legs' effects are provisional, and whether that
    /// stops a candidate depends on the modes on both sides. Declared
    /// rather than actual, so the claim stands from the moment the wave
    /// joins a tick rather than from when its batch comes back.
    #[must_use]
    pub fn declared_mutations(&self) -> Vec<(DeclaredKey, Mode)> {
        self.transactions
            .values()
            .flat_map(|tx| tx.routing().declared_modes.clone())
            .collect()
    }

    // ── Engagement coverage ─────────────────────────────────────────────

    /// Record that this shard, as `tx_hash`'s fee payer, waits for
    /// `counterparts` to echo their engagement before voting. Called at
    /// wave creation for each payer-local cross-shard transaction;
    /// `validity_end` is the signed window end bounding the wait.
    pub fn record_engagement_wait(
        &mut self,
        tx_hash: TxHash,
        counterparts: BTreeSet<ShardId>,
        validity_end: WeightedTimestamp,
    ) {
        if counterparts.is_empty() {
            return;
        }
        self.engagement_pending.insert(tx_hash, counterparts);
        let deadline = validity_end.plus(WAVE_TIMEOUT);
        self.engagement_deadline = Some(
            self.engagement_deadline
                .map_or(deadline, |current| current.max(deadline)),
        );
    }

    /// Drain engagement coverage from committed provisions: a bundle from
    /// a counterpart names the transaction only because that shard's
    /// block committed it, so absorption is the engagement evidence.
    ///
    /// Runs regardless of dispatch — the payer's leg dispatches on its own
    /// requirement long before the echoes it votes on arrive.
    pub fn absorb_engagement_evidence(&mut self, provisioning: &ProvisioningTracker) {
        self.engagement_pending.retain(|tx_hash, pending| {
            pending.retain(|shard| !provisioning.has_received_from(*tx_hash, *shard));
            !pending.is_empty()
        });
    }

    /// Whether every engagement echo the wave waits for has committed.
    #[must_use]
    pub fn engagement_covered(&self) -> bool {
        self.engagement_pending.is_empty()
    }

    /// Whether the local EC has been fed into this wave (via
    /// `add_execution_certificate` with `ec.wave_id() == &self.wave_id`).
    #[must_use]
    pub const fn local_ec_emitted(&self) -> bool {
        self.local_ec_emitted
    }

    /// Build this wave's [`TickExecutionGroup`] for the tick being
    /// composed and record which members joined it.
    ///
    /// Returns `None` (without mutating) when the wave isn't fully
    /// provisioned, when every member has already joined a tick or is
    /// pre-aborted, when a cross-shard tx is missing its provisions, or
    /// when every remaining member is waiting on a cell `blocked` holds
    /// provisionally. Pairing the build with the bookkeeping is what
    /// keeps a member out of two ticks.
    ///
    /// Provisions still gate at wave granularity; only the cell wait is
    /// per member, because that is the granularity the hazard has — one
    /// transaction's declared cell, not its wave's.
    pub fn tick_group_if_ready(
        &mut self,
        provisioning: &ProvisioningTracker,
        blocked: &ProvisionalCells,
    ) -> Option<TickExecutionGroup> {
        if !self.is_fully_provisioned() {
            return None;
        }
        let group = self.build_tick_group(provisioning, blocked)?;
        for request in &group.requests {
            self.dispatched.insert(request.tx_hash);
        }
        Some(group)
    }

    /// Pre-aborted txs are excluded — they produce no state change, so there's
    /// no reason to execute them. So are members already in a tick, and
    /// members whose declared cells are provisionally held.
    fn build_tick_group(
        &self,
        provisioning: &ProvisioningTracker,
        blocked: &ProvisionalCells,
    ) -> Option<TickExecutionGroup> {
        let mut requests: Vec<CrossShardExecutionRequest> =
            Vec::with_capacity(self.tx_hashes.len());
        for &tx_hash in &self.tx_hashes {
            if self.is_tx_explicitly_aborted(tx_hash) || self.dispatched.contains(&tx_hash) {
                continue;
            }
            let tx = self.transactions.get(&tx_hash)?;
            if !blocked.is_empty() && blocked.blocks(&tx.routing().declared_modes) {
                continue;
            }
            if self.wave_id.is_zero() {
                // Single-shard member: no provisions, the committing
                // block's own anchors.
                requests.push(CrossShardExecutionRequest {
                    tx_hash,
                    transaction: Arc::clone(tx),
                    provisions: Vec::new(),
                    clock: self.wave_start_ts,
                    randomness: self.wave_start_reveal,
                });
                continue;
            }
            // An absent entry is the dependency-free leg: the tx recorded
            // an empty requirement, so nothing ever arrived to store. A
            // tx with real requirements always has entries here — the
            // fully-provisioned gate holds required ⊆ received, and
            // absorption populates both together.
            let provisions = provisioning
                .provisions_for(tx_hash)
                .map(<[_]>::to_vec)
                .unwrap_or_default();
            // The transaction environment: a remote-payer leg executes
            // under the anchor its payer bundle carried — the payer shard
            // sits in `required`, so the fully-provisioned gate guarantees
            // the bundle was absorbed. A payer-local leg anchors on this
            // wave's own committing block.
            let anchor = provisioning.payer_anchor(tx_hash);
            requests.push(CrossShardExecutionRequest {
                tx_hash,
                transaction: Arc::clone(tx),
                provisions,
                clock: anchor.map_or(self.wave_start_ts, |a| a.clock),
                randomness: anchor.map_or(self.wave_start_reveal, |a| a.randomness),
            });
        }
        if requests.is_empty() {
            return None;
        }
        Some(TickExecutionGroup {
            wave_id: self.wave_id.clone(),
            requests,
        })
    }

    /// Mark a single tx as provisioned. Keeps the earliest `at` per tx
    /// so the wave's transition timestamp is a pure function of the event
    /// set.
    ///
    /// Returns `true` iff this call transitioned the wave from "partial" to
    /// "all provisioned" — the caller uses that signal to emit the single
    /// per-wave execution dispatch action.
    fn mark_tx_provisioned(&mut self, tx_hash: TxHash, at: WeightedTimestamp) -> bool {
        if !self.tx_hash_set.contains(&tx_hash) {
            return false;
        }

        self.provisioned_tx_ts
            .entry(tx_hash)
            .and_modify(|t| *t = (*t).min(at))
            .or_insert(at);

        let is_new = self.provisioned_txs.insert(tx_hash);

        if is_new
            && self.all_provisioned_at.is_none()
            && self.provisioned_txs.len() == self.tx_hashes.len()
        {
            let max_ts = self.provisioned_tx_ts.values().copied().max().unwrap_or(at);
            self.all_provisioned_at = Some(max_ts);
            true
        } else {
            false
        }
    }

    /// Mark every tx the tracker reports fully-provisioned at `at`. Cheap
    /// no-op for txs already marked. Drives the partial → fully-provisioned
    /// transition from both wave creation (when prior batches arrived
    /// before the wave existed) and per-batch absorption (when this
    /// commit's provisions land).
    pub fn absorb_ready_provisions(
        &mut self,
        provisioning: &ProvisioningTracker,
        at: WeightedTimestamp,
    ) {
        let tx_hashes: Vec<TxHash> = self.tx_hashes.clone();
        for tx_hash in tx_hashes {
            if provisioning.is_fully_provisioned(tx_hash) {
                self.mark_tx_provisioned(tx_hash, at);
            }
        }
    }

    // ── Local execution bookkeeping ─────────────────────────────────────

    /// Record an execution outcome from the engine. First-write-wins.
    /// Returns `true` if the wave now has an outcome (execution result or
    /// explicit abort) for every tx.
    pub fn record_execution_result(&mut self, tx_hash: TxHash, outcome: ExecutionOutcome) -> bool {
        if !self.tx_hash_set.contains(&tx_hash) {
            return false;
        }
        self.execution_results.entry(tx_hash).or_insert(outcome);
        self.has_outcome_for_every_tx()
    }

    /// Record a local receipt from the engine. First-write-wins.
    ///
    /// Paired with `record_execution_result`: both flow from the same
    /// `ExecutionBatchCompleted` event and are scoped to this wave. Receipts
    /// for txs not in the wave are silently dropped.
    pub fn record_receipt(&mut self, receipt: StoredReceipt) {
        if !self.tx_hash_set.contains(&receipt.tx_hash) {
            return;
        }
        self.execution_receipts
            .entry(receipt.tx_hash)
            .or_insert(receipt);
    }

    /// Record what this shard attested it did for a transaction.
    pub fn record_attested_work(&mut self, tx_hash: TxHash, work: u64) {
        if !self.tx_hash_set.contains(&tx_hash) {
            return;
        }
        self.attested_work.insert(tx_hash, work);
    }

    /// Record the fee receipt the engine built beside a transaction's
    /// execution receipt: what the payer owes if the wave aborts it.
    pub fn record_fee_receipt(&mut self, receipt: StoredReceipt) {
        if !self.tx_hash_set.contains(&receipt.tx_hash) {
            return;
        }
        self.fee_receipts.entry(receipt.tx_hash).or_insert(receipt);
    }

    /// Number of receipts currently held by this wave. Exposed for memory
    /// stats; receipts drain at finalization.
    #[must_use]
    pub fn receipt_count(&self) -> usize {
        self.execution_receipts.len()
    }

    /// Take the receipt for a tx, removing it from the wave. Used internally
    /// by [`Self::into_finalized`] to drain receipts in canonical order.
    fn take_receipt(&mut self, tx_hash: TxHash) -> Option<StoredReceipt> {
        self.execution_receipts.remove(&tx_hash)
    }

    /// Record an explicit abort from `ConflictDetector`. Keeps the earliest
    /// commit height if called multiple times for the same tx.
    /// Returns `true` if the wave now has an outcome (execution result or
    /// explicit abort) for every tx.
    ///
    /// No-op once the wave has dispatched: a dispatched wave is committed to
    /// executing what it started with, and mid-flight conflict aborts would
    /// introduce non-determinism across validators (the conflict batch lands
    /// at slightly different wall-clock offsets from `ExecutionBatchCompleted`
    /// on each node). Conflict detection's purpose — deadlock avoidance — is
    /// served by the pre-dispatch path only.
    ///
    /// Also marks the tx as provisioned — an aborted tx has a determinate
    /// outcome, so the wave shouldn't block waiting for provisions that will
    /// never arrive. Without this, a single aborted tx forced the wave into
    /// the timeout branch, which then marked every tx Aborted (including the
    /// ones that executed successfully).
    pub fn record_abort(&mut self, tx_hash: TxHash, committed_at: WeightedTimestamp) -> bool {
        if self.dispatched.contains(&tx_hash) || !self.tx_hash_set.contains(&tx_hash) {
            return false;
        }
        self.explicit_aborts.insert(tx_hash);
        self.mark_tx_provisioned(tx_hash, committed_at);
        self.has_outcome_for_every_tx()
    }

    /// Abort members that have waited past the wave's deadline for a cell
    /// another wave holds provisionally.
    ///
    /// A member kept out of ticks has no outcome, and a fully provisioned
    /// wave has no deadline of its own, so a wait can otherwise outlive
    /// whatever it waits for. It can even outlive itself: two shards can
    /// each hold the cell the other's counterpart needs, and neither sees
    /// the whole ring. The bound is the one an unprovisioned wave already
    /// has — the deadline is a backstop, not a schedule, and a wait that
    /// reaches it was never going to clear.
    ///
    /// Returns whether the wave now has an outcome for every member.
    pub fn abort_members_blocked_past_deadline(
        &mut self,
        blocked: &ProvisionalCells,
        now: WeightedTimestamp,
    ) -> bool {
        if blocked.is_empty() || now < self.wave_start_ts.plus(WAVE_TIMEOUT) {
            return false;
        }
        let stuck: Vec<TxHash> = self
            .tx_hashes
            .iter()
            .copied()
            .filter(|tx_hash| {
                !self.dispatched.contains(tx_hash)
                    && !self.explicit_aborts.contains(tx_hash)
                    && self
                        .transactions
                        .get(tx_hash)
                        .is_some_and(|tx| blocked.blocks(&tx.routing().declared_modes))
            })
            .collect();
        let mut complete = false;
        for tx_hash in stuck {
            tracing::warn!(
                wave = %self.wave_id,
                tx_hash = ?tx_hash,
                "Aborting a member that waited past the deadline for a provisionally held cell"
            );
            complete = self.record_abort(tx_hash, now);
        }
        complete
    }

    /// True if each tx has either an execution result or an explicit abort.
    fn has_outcome_for_every_tx(&self) -> bool {
        self.tx_hashes
            .iter()
            .all(|h| self.execution_results.contains_key(h) || self.explicit_aborts.contains(h))
    }

    /// True if, for every non-aborted outcome in the local EC, this validator
    /// has produced a matching local receipt. Aborted outcomes need no receipt.
    ///
    /// Gates [`Self::is_complete`] so `finalize_wave` can't produce a
    /// [`FinalizedWave`] that fails
    /// [`FinalizedWave::validate_receipts_against_ec`]. The check mirrors that
    /// invariant: a receipt is needed exactly for the outcomes the EC attests
    /// as `Executed`. When this validator's local abort decision disagrees
    /// with the quorum's EC (e.g. its conflict detector aborted a tx peers
    /// executed), the gate blocks here rather than synthesizing a
    /// `FinalizedWave` with missing receipts. Recovery flows through the
    /// existing peer-fetch path.
    ///
    /// Returns false if the local EC hasn't arrived yet; `local_ec_emitted`
    /// is checked separately by [`Self::is_complete`] for the same reason.
    ///
    /// [`FinalizedWave`]: hyperscale_types::FinalizedWave
    /// [`FinalizedWave::validate_receipts_against_ec`]:
    ///     hyperscale_types::FinalizedWave::validate_receipts_against_ec
    fn has_local_receipts_for_non_aborted(&self) -> bool {
        let Some(local_ec) = self
            .execution_certificates
            .iter()
            .find(|ec| ec.wave_id() == &self.wave_id)
        else {
            return false;
        };
        local_ec.tx_outcomes().iter().all(|outcome| {
            outcome.is_aborted() || self.execution_receipts.contains_key(&outcome.tx_hash())
        })
    }

    // ── Vote emission ───────────────────────────────────────────────────

    /// Vote anchor timestamp: the wave-starting block's BFT-authenticated
    /// weighted timestamp.
    ///
    /// This rides the vote payload and the EC canonical hash, and
    /// [`VoteTracker`] groups votes by it, so every validator must derive the
    /// same value from the same wave or agreeing votes never aggregate. Only
    /// committed chain content carries that guarantee: the wave block's
    /// timestamp is identical everywhere, whereas when a wave became
    /// provisioned and which of its transactions a conflict detector aborted
    /// are local observations that differ across a committee.
    ///
    /// Unconditional for the same reason. A per-branch anchor would diverge
    /// even if each branch's value were sound on its own, because the branch
    /// itself is locally determined — one validator's `all_provisioned_at` is
    /// set by an abort its peers have not recorded yet.
    ///
    /// Already in the past when the vote is built, so the committee it
    /// resolves is available at once; an anchor ahead of the committed clock
    /// would defer every certificate until the schedule reached it.
    ///
    /// [`VoteTracker`]: crate::vote_tracker::VoteTracker
    #[must_use]
    pub const fn vote_anchor_ts(&self) -> WeightedTimestamp {
        self.wave_start_ts
    }

    /// Whether the local vote can be emitted at the given committed timestamp.
    ///
    /// Three conditions, all read off committed chain content so every
    /// member of the committee evaluates them identically:
    /// - Fully provisioned: need `committed_ts >= all_provisioned_at`
    ///   AND every tx has an execution result or explicit abort.
    /// - Not provisioned: wait until `committed_ts >= wave_start_ts +
    ///   WAVE_TIMEOUT`. Upon timeout, every tx in the wave is implicitly
    ///   aborted.
    /// - Engagement-gated (the payer shard's cross-shard legs): every
    ///   counterpart's engagement echo has committed, or the wave's
    ///   deadline passed — whichever comes first. The wave speaks once,
    ///   so a counterpart engaging at the edge of its window cannot
    ///   contradict this shard's verdict: its success EC loses worst-wins
    ///   to the abort, on every participant.
    #[must_use]
    pub fn can_emit_vote(&self, committed_ts: WeightedTimestamp) -> bool {
        if self.voted {
            return false;
        }
        if !self.engagement_settled(committed_ts) {
            return false;
        }
        self.all_provisioned_at.map_or_else(
            || committed_ts >= self.wave_start_ts.plus(WAVE_TIMEOUT),
            |provisioned_at| committed_ts >= provisioned_at && self.has_outcome_for_every_tx(),
        )
    }

    /// Whether engagement no longer blocks the vote: fully covered, or
    /// past the deadline for waiting.
    fn engagement_settled(&self, committed_ts: WeightedTimestamp) -> bool {
        self.engagement_pending.is_empty()
            || self
                .engagement_deadline
                .is_some_and(|deadline| committed_ts >= deadline)
    }

    /// Build vote payload at the target anchor, consuming the one-shot vote.
    ///
    /// Returns `(vote_anchor_ts, global_receipt_root, tx_outcomes)`.
    /// Returns `None` if `can_emit_vote` is false.
    ///
    /// In the timeout-abort branch (`all_provisioned_at = None`), every
    /// tx gets an `ExecutionOutcome::Aborted`. In the provisioned branch,
    /// each tx's outcome is its explicit abort (if any) or execution
    /// result — except a transaction whose counterparts never echoed
    /// their engagement before the deadline, which aborts here however
    /// its own execution went.
    ///
    /// # Panics
    ///
    /// Panics if `can_emit_vote` says yes for the provisioned branch but a tx
    /// is missing its execution result. The `has_outcome_for_every_tx` gate
    /// guards against this; the panic would indicate a bug in the gating logic.
    pub fn build_vote_data(
        &mut self,
        committed_ts: WeightedTimestamp,
    ) -> Option<(WeightedTimestamp, GlobalReceiptRoot, Vec<TxOutcome>)> {
        if !self.can_emit_vote(committed_ts) {
            return None;
        }

        let target = self.vote_anchor_ts();
        let timed_out = self.all_provisioned_at.is_none();

        let outcomes: Vec<TxOutcome> = self
            .tx_hashes
            .iter()
            .map(|tx_hash| {
                let outcome = if timed_out
                    || self.explicit_aborts.contains(tx_hash)
                    || self.engagement_pending.contains_key(tx_hash)
                {
                    ExecutionOutcome::Aborted
                } else {
                    // Safe: has_outcome_for_every_tx() ensured presence
                    self.execution_results
                        .get(tx_hash)
                        .cloned()
                        .expect("execution result must be present under provisioned branch")
                };
                // An outcome that applies no effects still settles the
                // payer's charge when the engine built one — an abort
                // whose effects were discarded, or a failure that produced
                // none.
                let work = self.attested_work.get(tx_hash).copied().unwrap_or(0);
                // What the transaction reserved when its block committed
                // it, carried so the settling block can release exactly
                // that. Derived from the transaction, which this wave
                // still holds; a member swept before it voted reserved
                // nothing this wave can return.
                let reserved = self.transactions.get(tx_hash).map_or(0, |tx| tx.work());
                match (&outcome, self.fee_receipts.get(tx_hash)) {
                    (ExecutionOutcome::Aborted | ExecutionOutcome::Failed, Some(fee)) => {
                        TxOutcome::with_fee(
                            *tx_hash,
                            outcome.clone(),
                            fee.consensus.receipt_hash(),
                            work,
                        )
                    }
                    _ => TxOutcome::attesting(*tx_hash, outcome.clone(), work),
                }
                .reserving(reserved)
            })
            .collect();

        let root = compute_global_receipt_root(&outcomes);
        self.voted = true;
        self.local_vote_global_receipt_root = Some(root);
        self.reconcile_local_ec_root();
        Some((target, root, outcomes))
    }

    // ── Cross-shard EC collection ───────────────────────────────────────

    /// Feed an EC into the wave. Handles dedup (by canonical hash), updates
    /// per-tx coverage, and tracks aborts/failures. For our own local EC
    /// (`ec.wave_id() == &self.wave_id`), records the admitted root and runs
    /// `reconcile_local_ec_decision` — which compares against the local
    /// vote when both are known. The local EC may arrive before the local
    /// vote in cross-shard waves where peers aggregate the EC before this
    /// validator finishes executing; the reconciliation runs again from
    /// `build_vote_data` once the local vote lands.
    ///
    /// Returns `true` if the wave is now complete (ready for `finalize_wave`).
    pub fn add_execution_certificate(&mut self, ec: Arc<Verified<ExecutionCertificate>>) -> bool {
        if !self.seen_ec_wave_ids.insert(ec.wave_id().clone()) {
            return self.is_complete();
        }

        let shard = ec.shard_id();
        let is_local = ec.wave_id() == &self.wave_id;

        for outcome in ec.tx_outcomes() {
            if let Some(covered) = self.covered_shards.get_mut(&outcome.tx_hash()) {
                covered.insert(shard);
                if outcome.is_aborted() {
                    self.tracker_aborted.insert(outcome.tx_hash());
                }
                if !matches!(outcome.outcome(), ExecutionOutcome::Succeeded { .. }) {
                    self.tx_has_failure.insert(outcome.tx_hash());
                }
            }
        }

        if is_local {
            self.admitted_local_ec_root = Some(ec.global_receipt_root());
            self.local_ec_emitted = true;
            self.reconcile_local_ec_root();
        }

        self.execution_certificates.push(ec);

        self.is_complete()
    }

    /// Compare `local_vote_global_receipt_root` against
    /// `admitted_local_ec_root` once both are known. Run from both sites
    /// that can supply the second half of the pair: `build_vote_data`
    /// (when the EC arrived first) and `add_execution_certificate`
    /// (when the vote arrived first).
    ///
    /// `global_receipt_root` commits to each tx's
    /// [`ConsensusReceipt::receipt_hash`](hyperscale_types::ConsensusReceipt::receipt_hash)
    /// (which folds in `writes_root` derived from `database_updates`),
    /// so a root mismatch is direct proof that local execution produced
    /// different writes than the quorum. Latches `locally_divergent` so
    /// the coordinator can escalate: under chaining a wrong tick output
    /// is the baseline every later tick reads, so the mismatch is not one
    /// wave's problem to sit out.
    fn reconcile_local_ec_root(&mut self) {
        if self.locally_divergent {
            return;
        }
        let (Some(local_root), Some(ec_root)) = (
            self.local_vote_global_receipt_root,
            self.admitted_local_ec_root,
        ) else {
            return;
        };
        if local_root != ec_root {
            self.locally_divergent = true;
            self.divergence = Some(Divergence {
                wave_id: self.wave_id.clone(),
                block_hash: self.block_hash,
                local_root,
                ec_root,
            });
        }
    }

    /// Take the divergence this wave detected, if it has not been
    /// reported yet. One report per wave: the latch stays set.
    pub const fn take_divergence(&mut self) -> Option<Divergence> {
        self.divergence.take()
    }

    /// Whether the admitted local EC's `global_receipt_root` disagreed
    /// with the validator's own vote. Callers (e.g. wave pruning) use
    /// this to skip recovery paths that assume local receipts are
    /// canonical, in the window before the coordinator escalates.
    #[must_use]
    pub const fn is_locally_divergent(&self) -> bool {
        self.locally_divergent
    }

    /// Whether the wave is complete: local EC present, every non-aborted
    /// tx has a local execution result on this validator, and every tx
    /// either aborted (terminal) or covered by every participating shard.
    ///
    /// The local-receipt gate prevents the race where a cross-shard wave's
    /// local EC arrives (aggregated from other validators' votes) before
    /// this validator's engine finishes executing — without it,
    /// `finalize_wave` silently drops the pending txs' receipt slots and
    /// produces a divergent `FinalizedWave`.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        if !self.local_ec_emitted {
            return false;
        }
        if self.locally_divergent {
            return false;
        }
        if !self.has_local_receipts_for_non_aborted() {
            return false;
        }
        for tx_hash in &self.tx_hashes {
            if self.tracker_aborted.contains(tx_hash) {
                continue;
            }
            let Some(expected) = self.participating_shards.get(tx_hash) else {
                return false;
            };
            let Some(covered) = self.covered_shards.get(tx_hash) else {
                return false;
            };
            if !expected.is_subset(covered) {
                return false;
            }
        }
        true
    }

    /// Whether any non-aborted tx still lacks an execution certificate from
    /// `shard`. The counterpart abort sweep reads this once a past-terminal
    /// partner shard's settled set is fully ingested: a wave that still
    /// lacks the partner's coverage can never gain it, so it is doomed and
    /// its transactions abort.
    #[must_use]
    pub fn lacks_coverage_from(&self, shard: ShardId) -> bool {
        self.tx_hashes.iter().any(|tx_hash| {
            !self.tracker_aborted.contains(tx_hash)
                && self
                    .covered_shards
                    .get(tx_hash)
                    .is_some_and(|covered| !covered.contains(&shard))
        })
    }

    /// Whether a tx was aborted before dispatch (pre-dispatch reverse-conflict).
    /// Used by dispatch to skip executing txs the wave has already decided to
    /// abort.
    fn is_tx_explicitly_aborted(&self, tx_hash: TxHash) -> bool {
        self.explicit_aborts.contains(&tx_hash)
    }

    /// Emit a `warn!` log exactly once, when the wave reaches
    /// `WAVE_OVERDUE_WARN` of age without completing. A firing here is an
    /// invariant violation under the two-stage lifecycle — every tx is
    /// supposed to terminate with a `WaveCertificate` — so the dump
    /// captures enough state to diagnose where the post-inclusion
    /// termination guarantee broke (provisioning / dispatch / voting /
    /// EC collection). Latched at the first crossing of the threshold so
    /// it fires once per stuck wave, not once per surviving commit.
    pub fn log_if_overdue(&mut self, committed_ts: WeightedTimestamp) {
        if self.overdue_warned {
            return;
        }
        let age = committed_ts.elapsed_since(self.wave_start_ts);
        if age < WAVE_OVERDUE_WARN {
            return;
        }
        self.overdue_warned = true;

        let total = self.tx_hashes.len();
        let provisioned = self.provisioned_txs.len();

        let mut missing_coverage: Vec<String> = Vec::new();
        for tx_hash in &self.tx_hashes {
            if self.tracker_aborted.contains(tx_hash) {
                continue;
            }
            let expected = self
                .participating_shards
                .get(tx_hash)
                .cloned()
                .unwrap_or_default();
            let covered = self
                .covered_shards
                .get(tx_hash)
                .cloned()
                .unwrap_or_default();
            let missing: BTreeSet<ShardId> = expected.difference(&covered).copied().collect();
            if !missing.is_empty() {
                let missing_list: Vec<String> =
                    missing.iter().map(|s| s.inner().to_string()).collect();
                missing_coverage.push(format!("{:?}→[{}]", tx_hash, missing_list.join(",")));
            }
        }

        let local_receipts_ready = self.has_local_receipts_for_non_aborted();

        tracing::warn!(
            wave = %self.wave_id,
            block_hash = ?self.block_hash,
            block_height = self.wave_id.block_height().inner(),
            wave_start_ts = self.wave_start_ts.as_millis(),
            committed_ts = committed_ts.as_millis(),
            age_ms = u64::try_from(age.as_millis()).unwrap_or(u64::MAX),
            timeout_ms = u64::try_from(WAVE_TIMEOUT.as_millis()).unwrap_or(u64::MAX),
            num_txs = total,
            provisioned = format!("{}/{}", provisioned, total),
            all_provisioned_at = ?self.all_provisioned_at.map(WeightedTimestamp::as_millis),
            dispatched = self.dispatched.len(),
            voted = self.voted,
            local_ec_emitted = self.local_ec_emitted,
            local_receipts_ready,
            execution_results = self.execution_results.len(),
            explicit_aborts = self.explicit_aborts.len(),
            tracker_aborted = self.tracker_aborted.len(),
            ecs_collected = self.execution_certificates.len(),
            is_complete = self.is_complete(),
            missing_coverage = missing_coverage.join(" "),
            "Wave overdue: alive past execution timeout without completing"
        );
    }

    /// Build the final `WaveCertificate`. Local EC is always included;
    /// a remote EC is included when it covers a tx this wave still needs a
    /// verdict on, or when it is the EC carrying that tx's abort.
    /// Deterministic order: `(shard_id, wave_id)`.
    ///
    /// The second clause is what keeps the two sides of a settlement in
    /// agreement. `tracker_aborted` is fed by the very ECs being filtered
    /// here — a remote abort lands as coverage *and* as an entry in that
    /// set — so pruning on `tracker_aborted` alone discards the only
    /// artifact carrying that verdict. Every downstream reader derives the
    /// outcome from the certificate and nothing else
    /// ([`FinalizedWave::tx_decisions`]), so what that drops is not merely
    /// redundant: the local EC's success stands unopposed and this shard
    /// commits an accept against the counterparty's abort. An abort the
    /// local EC reports itself needs no such corroboration, which is why a
    /// wave both sides aborted still keeps only the one certificate.
    ///
    /// Callers should invoke only when `is_complete()` is true.
    #[must_use]
    pub fn create_wave_certificate(&self) -> WaveCertificate {
        // What the local EC says on its own. A tx it already reports as
        // aborted needs no remote to corroborate it.
        let locally_aborted: HashSet<TxHash> = self
            .execution_certificates
            .iter()
            .find(|ec| ec.wave_id() == &self.wave_id)
            .map(|ec| {
                ec.tx_outcomes()
                    .iter()
                    .filter(|outcome| outcome.is_aborted())
                    .map(TxOutcome::tx_hash)
                    .collect()
            })
            .unwrap_or_default();

        let required_remote_wave_ids: HashSet<WaveId> = self
            .execution_certificates
            .iter()
            .filter(|ec| ec.wave_id() != &self.wave_id)
            .filter(|ec| {
                ec.tx_outcomes().iter().any(|outcome| {
                    let tx_hash = outcome.tx_hash();
                    if !self.participating_shards.contains_key(&tx_hash) {
                        return false;
                    }
                    // Still awaiting a verdict, or holding the only one that
                    // says abort.
                    !self.tracker_aborted.contains(&tx_hash)
                        || (outcome.is_aborted() && !locally_aborted.contains(&tx_hash))
                })
            })
            .map(|ec| ec.wave_id().clone())
            .collect();

        let mut ecs: Vec<Verified<ExecutionCertificate>> = self
            .execution_certificates
            .iter()
            .filter(|ec| {
                ec.wave_id() == &self.wave_id || required_remote_wave_ids.contains(ec.wave_id())
            })
            .map(|verified| (**verified).clone())
            .collect();

        ecs.sort_by(|a, b| (&a.shard_id(), a.wave_id()).cmp(&(&b.shard_id(), b.wave_id())));

        WaveCertificate::from_verified_ecs(self.wave_id.clone(), ecs)
    }

    /// Consume the wave and produce its terminal [`FinalizedWave`].
    ///
    /// Builds the [`WaveCertificate`] and drains a stored receipt for each
    /// non-aborted outcome in the local EC, in canonical order. Aborted
    /// outcomes contribute no receipt; stray receipts for aborted txs (e.g.
    /// local execution finished before the aggregated EC attested
    /// `Aborted`) drop with the wave. Mirrors `FinalizedWave::reconstruct`,
    /// matching the invariant peers enforce via
    /// `validate_receipts_against_ec` at ingress.
    ///
    /// Should only be called when [`Self::is_complete`] is true; that gate
    /// guarantees both the local EC's presence and a receipt for every
    /// non-aborted outcome. A missing receipt under those conditions is an
    /// invariant violation, logged but not fatal so the canonical
    /// `FinalizedWave` admitted via block sync can still recover the node.
    ///
    /// # Panics
    ///
    /// Panics if the constructed [`WaveCertificate`] doesn't carry the
    /// local EC. `is_complete` requires `local_ec_emitted`, so that ECs
    /// presence in `execution_certificates` — and thus in the WC — is
    /// guaranteed at the legitimate call site.
    #[must_use]
    pub fn into_finalized(mut self) -> FinalizedWave {
        let wc = self.create_wave_certificate();
        let local_ec = wc
            .execution_certificates()
            .iter()
            .find(|ec| ec.wave_id() == wc.wave_id())
            .expect("WaveCertificate invariant: local EC must be present")
            .clone();
        let mut receipts: Vec<StoredReceipt> = Vec::with_capacity(local_ec.tx_outcomes().len());
        for outcome in local_ec.tx_outcomes() {
            if outcome.is_aborted() {
                // The transaction's own effects are discarded; a fee it
                // settles is not.
                if outcome.fee_receipt().is_some()
                    && let Some(fee) = self.fee_receipts.remove(&outcome.tx_hash())
                {
                    receipts.push(fee);
                }
                continue;
            }
            // A failure that settles a charge stores that receipt instead
            // of its own — the `Failed` receipt carries nothing, and the
            // pairing stays one receipt per outcome either way.
            if outcome.fee_receipt().is_some()
                && let Some(fee) = self.fee_receipts.remove(&outcome.tx_hash())
            {
                self.take_receipt(outcome.tx_hash());
                receipts.push(fee);
                continue;
            }
            if let Some(receipt) = self.take_receipt(outcome.tx_hash()) {
                receipts.push(receipt);
            } else {
                tracing::error!(
                    wave = %self.wave_id,
                    tx_hash = ?outcome.tx_hash(),
                    "into_finalized: non-aborted tx is missing its stored receipt \
                     (is_complete gate bypassed)"
                );
            }
        }
        FinalizedWave::new(Arc::new(wc), receipts)
    }

    /// Per-tx terminal decisions derived from collected ECs.
    /// Priority: Aborted > Reject > Accept.
    #[must_use]
    pub fn tx_decisions(&self) -> Vec<(TxHash, TransactionDecision)> {
        self.tx_hashes
            .iter()
            .map(|tx_hash| {
                let decision = if self.tracker_aborted.contains(tx_hash) {
                    TransactionDecision::Aborted
                } else if self.tx_has_failure.contains(tx_hash) {
                    TransactionDecision::Reject
                } else {
                    TransactionDecision::Accept
                };
                (*tx_hash, decision)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::test_utils::{test_prefix, test_transaction_with_prefixes};
    use hyperscale_types::{
        AggregateSignature, ConsensusReceipt, GlobalReceiptHash, Hash, MerkleInclusionProof,
        ProvisionEntry, Provisions, RevealChain, SignerBitfield, StateWrites, SubstateEntry,
        tx_outcome_leaf,
    };

    use super::*;

    const WAVE_START: BlockHeight = BlockHeight::new(10);

    fn make_tx(seed: u8) -> Arc<Verifiable<Transaction>> {
        Arc::new(Verifiable::from(test_transaction_with_prefixes(
            &[seed, seed + 1, seed + 2],
            &[test_prefix(seed)],
            &[test_prefix(seed + 50)],
        )))
    }

    /// Tests use synthetic timestamps proportional to block heights so the
    /// block-height intuition in assertions maps cleanly to ts space.
    const TEST_BLOCK_INTERVAL_MS: u64 = 500;

    fn ts_for(height: BlockHeight) -> WeightedTimestamp {
        WeightedTimestamp::from_millis(height.inner() * TEST_BLOCK_INTERVAL_MS)
    }

    fn make_single_shard_wave(n: usize) -> WaveState {
        let txs: Vec<(Arc<Verifiable<Transaction>>, BTreeSet<ShardId>)> = (0..n)
            .map(|i| {
                (
                    make_tx(u8::try_from(i).unwrap_or(u8::MAX)),
                    BTreeSet::from([ShardId::leaf(1, 0)]),
                )
            })
            .collect();
        WaveState::new(
            WaveId::new(ShardId::leaf(1, 0), WAVE_START, BTreeSet::new()),
            BlockHash::from_raw(Hash::from_bytes(b"block")),
            ts_for(WAVE_START),
            RevealChain::ZERO,
            txs,
            true,
        )
    }

    fn make_cross_shard_wave(n: usize) -> WaveState {
        let shards = BTreeSet::from([ShardId::leaf(1, 0), ShardId::leaf(1, 1)]);
        let txs: Vec<(Arc<Verifiable<Transaction>>, BTreeSet<ShardId>)> = (0..n)
            .map(|i| (make_tx(u8::try_from(i).unwrap_or(u8::MAX)), shards.clone()))
            .collect();
        WaveState::new(
            WaveId::new(
                ShardId::leaf(1, 0),
                WAVE_START,
                BTreeSet::from([ShardId::leaf(1, 1)]),
            ),
            BlockHash::from_raw(Hash::from_bytes(b"block")),
            ts_for(WAVE_START),
            wave_start_reveal(),
            txs,
            false,
        )
    }

    /// The wave-starting block's own reveal chain — the anchor every leg
    /// falls back to when no payer bundle names another.
    fn wave_start_reveal() -> RevealChain {
        RevealChain::from_raw(Hash::from_bytes(b"wave start reveal"))
    }

    fn executed(success: bool) -> ExecutionOutcome {
        if success {
            ExecutionOutcome::Succeeded {
                receipt_hash: GlobalReceiptHash::from_raw(Hash::from_bytes(b"r")),
            }
        } else {
            ExecutionOutcome::Failed
        }
    }

    /// Record a result + matching receipt, as the production path does
    /// via `on_execution_batch_completed`. Tests that need execution to
    /// look "real" should use this rather than `record_execution_result`
    /// alone — the `is_complete` gate keys off `execution_receipts`.
    fn record_executed(w: &mut WaveState, tx_hash: TxHash, success: bool) {
        w.record_execution_result(tx_hash, executed(success));
        w.record_receipt(StoredReceipt {
            tx_hash,
            consensus: Arc::new(if success {
                ConsensusReceipt::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                    #[allow(clippy::default_trait_access)]
                    writes: Default::default(),
                    beacon_witness_events: Vec::new(),
                    events: Vec::new(),
                }
            } else {
                ConsensusReceipt::Failed
            }),
            metadata: None,
        });
    }

    fn make_ec(
        wave_id: &WaveId,
        ec_shard: ShardId,
        tx_hashes: &[TxHash],
        success: bool,
    ) -> Arc<Verified<ExecutionCertificate>> {
        let outcomes: Vec<TxOutcome> = tx_hashes
            .iter()
            .map(|h| {
                TxOutcome::new(
                    *h,
                    if success {
                        executed(true)
                    } else {
                        ExecutionOutcome::Aborted
                    },
                )
            })
            .collect();
        let ec_wave_id = WaveId::new(
            ec_shard,
            wave_id.block_height(),
            wave_id.remote_shards().iter().copied().collect(),
        );
        Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
            ec_wave_id,
            WeightedTimestamp::from_millis(wave_id.block_height().inner() + 1),
            GlobalReceiptRoot::from_raw(Hash::from_bytes(b"global_receipt_root")),
            outcomes,
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        )))
    }

    /// An EC over exactly `outcomes`, as this shard's own local EC.
    fn make_ec_from(
        wave_id: &WaveId,
        outcomes: Vec<TxOutcome>,
    ) -> Arc<Verified<ExecutionCertificate>> {
        Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
            wave_id.clone(),
            WeightedTimestamp::from_millis(wave_id.block_height().inner() + 1),
            compute_global_receipt_root(&outcomes),
            outcomes,
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        )))
    }

    /// A fee receipt the engine would have built for `tx`: one debit and
    /// nothing else.
    fn fee_receipt_for(tx: TxHash) -> StoredReceipt {
        StoredReceipt::synced(
            tx,
            Arc::new(ConsensusReceipt::Succeeded {
                receipt_hash: GlobalReceiptHash::from_raw(Hash::from_bytes(b"fee-receipt")),
                writes: StateWrites::default(),
                beacon_witness_events: Vec::new(),
                events: Vec::new(),
            }),
        )
    }

    #[test]
    fn single_shard_is_provisioned_on_creation() {
        let w = make_single_shard_wave(2);
        assert!(w.is_fully_provisioned());
        assert_eq!(w.all_provisioned_at, Some(ts_for(WAVE_START)));
    }

    #[test]
    fn cross_shard_not_provisioned_on_creation() {
        let w = make_cross_shard_wave(2);
        assert!(!w.is_fully_provisioned());
        assert_eq!(w.all_provisioned_at, None);
    }

    #[test]
    fn mark_tx_provisioned_transitions_exactly_once() {
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        assert!(!w.mark_tx_provisioned(h0, ts_for(WAVE_START + 1)));
        assert!(!w.is_fully_provisioned());
        assert!(w.mark_tx_provisioned(h1, ts_for(WAVE_START + 2)));
        assert!(w.is_fully_provisioned());
        assert_eq!(w.all_provisioned_at, Some(ts_for(WAVE_START + 2)));

        // Idempotent: repeat calls don't retransition.
        assert!(!w.mark_tx_provisioned(h1, ts_for(WAVE_START + 3)));
    }

    #[test]
    fn can_emit_vote_requires_results_when_provisioned() {
        let mut w = make_single_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        // No results yet.
        assert!(!w.can_emit_vote(ts_for(WAVE_START)));

        w.record_execution_result(h0, executed(true));
        assert!(!w.can_emit_vote(ts_for(WAVE_START)));

        w.record_execution_result(h1, executed(true));
        assert!(w.can_emit_vote(ts_for(WAVE_START)));
    }

    #[test]
    fn timeout_abort_without_provisions() {
        let mut w = make_cross_shard_wave(2);
        let wave_start_ts = ts_for(WAVE_START);
        let at_timeout = wave_start_ts.plus(WAVE_TIMEOUT);
        let just_before = WeightedTimestamp::from_millis(at_timeout.as_millis() - 1);

        // Not yet at timeout.
        assert!(!w.can_emit_vote(just_before));

        // At timeout — all txs implicitly abort.
        assert!(w.can_emit_vote(at_timeout));

        let (anchor, _root, outcomes) = w.build_vote_data(at_timeout).unwrap();
        assert_eq!(anchor, wave_start_ts, "the anchor is the wave block");
        assert_eq!(outcomes.len(), 2);
        assert!(
            outcomes
                .iter()
                .all(|o| matches!(o.outcome(), ExecutionOutcome::Aborted))
        );
    }

    /// Two validators recording the same conflict abort at different local
    /// commit timestamps must still vote under one anchor.
    ///
    /// The anchor keys vote aggregation, so a value derived from when a
    /// validator happened to observe the abort splits agreeing votes into
    /// groups that each fall short of quorum — the votes are unanimous on the
    /// outcome and no certificate ever forms.
    #[test]
    fn conflict_abort_timing_does_not_move_the_vote_anchor() {
        let early = ts_for(BlockHeight::new(WAVE_START.inner() + 12));
        let late = ts_for(BlockHeight::new(WAVE_START.inner() + 36));

        let mut seen_early = make_cross_shard_wave(1);
        let h = seen_early.tx_hashes()[0];
        seen_early.record_abort(h, early);
        let (anchor_early, root_early, _) = seen_early
            .build_vote_data(early)
            .expect("an explicit abort is an outcome, so the wave votes");

        let mut seen_late = make_cross_shard_wave(1);
        let h = seen_late.tx_hashes()[0];
        seen_late.record_abort(h, late);
        let (anchor_late, root_late, _) = seen_late
            .build_vote_data(late)
            .expect("an explicit abort is an outcome, so the wave votes");

        assert_eq!(anchor_early, ts_for(WAVE_START));
        assert_eq!(anchor_late, ts_for(WAVE_START));
        assert_eq!(
            root_early, root_late,
            "same outcome, so the votes must also agree on the receipt root"
        );
    }

    #[test]
    fn vote_exactly_once() {
        let mut w = make_single_shard_wave(1);
        let h0 = w.tx_hashes()[0];
        w.record_execution_result(h0, executed(true));

        assert!(w.build_vote_data(ts_for(WAVE_START)).is_some());
        // Already voted; can't again.
        assert!(!w.can_emit_vote(ts_for(WAVE_START + 100)));
        assert!(w.build_vote_data(ts_for(WAVE_START + 100)).is_none());
    }

    #[test]
    fn explicit_abort_produces_abort_outcome() {
        let mut w = make_single_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        w.record_execution_result(h0, executed(true));
        w.record_abort(h1, ts_for(WAVE_START + 3));

        let (_, _, outcomes) = w.build_vote_data(ts_for(WAVE_START + 3)).unwrap();
        assert!(matches!(
            outcomes[0].outcome(),
            ExecutionOutcome::Succeeded { .. } | ExecutionOutcome::Failed
        ));
        assert!(matches!(outcomes[1].outcome(), ExecutionOutcome::Aborted));
    }

    #[test]
    fn abort_marks_tx_as_aborted() {
        let mut w = make_single_shard_wave(1);
        let h0 = w.tx_hashes()[0];
        w.record_abort(h0, ts_for(BlockHeight::new(20)));
        assert!(w.explicit_aborts.contains(&h0));
        // Idempotent: calling again doesn't clear or duplicate.
        w.record_abort(h0, ts_for(BlockHeight::new(15)));
        assert!(w.explicit_aborts.contains(&h0));
        assert_eq!(w.explicit_aborts.len(), 1);
    }

    #[test]
    fn cross_shard_wave_requires_local_and_remote_ec() {
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        // Fully provision and execute locally.
        w.mark_tx_provisioned(h0, ts_for(WAVE_START + 1));
        w.mark_tx_provisioned(h1, ts_for(WAVE_START + 1));
        record_executed(&mut w, h0, true);
        record_executed(&mut w, h1, true);

        // Remote-only EC doesn't complete.
        let ec_remote = make_ec(w.wave_id(), ShardId::leaf(1, 1), &[h0, h1], true);
        assert!(!w.add_execution_certificate(ec_remote));
        assert!(!w.is_complete());

        // Add local EC — now complete.
        let ec_local = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[h0, h1], true);
        assert!(w.add_execution_certificate(ec_local));
        assert!(w.is_complete());
    }

    #[test]
    fn a_remote_abort_survives_into_the_wave_certificate() {
        // A remote shard's abort is only ever carried by that shard's EC, and
        // the certificate is the whole of what a committed block keeps: every
        // downstream reader derives the outcome from it alone. Pruning the EC
        // that carried the abort leaves the local EC's success unopposed, and
        // the two shards commit opposite outcomes for the same transaction.
        let mut w = make_cross_shard_wave(1);
        let tx = w.tx_hashes()[0];
        w.mark_tx_provisioned(tx, ts_for(WAVE_START + 1));
        record_executed(&mut w, tx, true);

        // This shard executed it successfully; the counterparty aborted.
        let local = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[tx], true);
        let remote = make_ec(w.wave_id(), ShardId::leaf(1, 1), &[tx], false);
        w.add_execution_certificate(local);
        assert!(w.add_execution_certificate(remote), "wave completes");

        let wc = w.create_wave_certificate();
        let signers: BTreeSet<ShardId> = wc
            .execution_certificates()
            .iter()
            .map(|ec| ec.wave_id().shard_id())
            .collect();
        assert_eq!(
            signers,
            BTreeSet::from([ShardId::leaf(1, 0), ShardId::leaf(1, 1)]),
            "every participant's verdict must reach the certificate",
        );

        let decisions = FinalizedWave::new(Arc::new(wc), vec![]).tx_decisions();
        assert_eq!(
            decisions,
            vec![(tx, TransactionDecision::Aborted)],
            "one participant aborting aborts the transaction everywhere",
        );
    }

    #[test]
    fn is_complete_false_when_local_ec_arrives_before_engine_results() {
        // The race the gate is designed to catch: a cross-shard wave's
        // local EC is aggregated from *other* validators' votes while this
        // validator's engine is still running. Coverage looks good but
        // there are no local receipts yet — finalizing here would produce
        // a `FinalizedWave` with missing receipts. Gate must hold until
        // the engine catches up.
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        w.mark_tx_provisioned(h0, ts_for(WAVE_START + 1));
        w.mark_tx_provisioned(h1, ts_for(WAVE_START + 1));

        // Remote EC lands first (other shard was fast).
        let ec_remote = make_ec(w.wave_id(), ShardId::leaf(1, 1), &[h0, h1], true);
        w.add_execution_certificate(ec_remote);

        // Local EC lands — built from the other three committee members'
        // votes without this validator contributing. Coverage is complete
        // but no local engine result yet.
        let ec_local = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[h0, h1], true);
        w.add_execution_certificate(ec_local);
        assert!(
            !w.is_complete(),
            "wave must not be complete before local engine results arrive"
        );

        // Engine finishes — first result not yet enough.
        record_executed(&mut w, h0, true);
        assert!(!w.is_complete());

        // Second result — now fully resolvable.
        record_executed(&mut w, h1, true);
        assert!(w.is_complete());
    }

    #[test]
    fn is_complete_when_ec_attests_abort_without_local_result() {
        // Symmetric to the race fix: if the local EC marks a tx aborted,
        // that tx legitimately has no local receipt. The gate must not
        // stall on such txs — `tracker_aborted` covers for them.
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        w.mark_tx_provisioned(h0, ts_for(WAVE_START + 1));
        w.mark_tx_provisioned(h1, ts_for(WAVE_START + 1));

        // Local EC attests both txs aborted. No execution results needed.
        let ec_local = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[h0, h1], false);
        w.add_execution_certificate(ec_local);
        assert!(
            w.is_complete(),
            "all-aborted wave resolves without local engine results"
        );
    }

    #[test]
    fn is_complete_false_when_explicit_abort_disagrees_with_local_ec() {
        // This validator's conflict detector aborted h0 locally (e.g. because
        // it was behind on commits and saw different prior provisions). The
        // quorum executed h0 and aggregated a local EC attesting Executed.
        // Without a receipt-vs-EC gate, finalize_wave would build a
        // FinalizedWave missing h0's receipt — which later fails
        // `validate_receipts_against_ec` on any peer. The gate must block
        // and let the existing peer-fetch path recover.
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        // h0: local explicit abort. h1: executed, receipt recorded.
        w.record_abort(h0, ts_for(WAVE_START + 1));
        record_executed(&mut w, h1, true);

        // Local EC disagrees: attests BOTH executed.
        let ec_local = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[h0, h1], true);
        w.add_execution_certificate(ec_local);

        assert!(
            !w.is_complete(),
            "must not finalize when local abort disagrees with quorum's EC"
        );
    }

    #[test]
    fn locally_divergent_when_vote_root_disagrees_with_admitted_ec() {
        // Local vote computed root R1 (from this validator's database_updates).
        // Quorum aggregated EC with root R2 (other validators' agreed root).
        // R1 != R2 ⇒ this validator's `ConsensusReceipt::Succeeded.database_updates`
        // differs from canonical. Wave must not finalize locally; the
        // canonical `FinalizedWave` is recovered later via block-sync.
        let mut w = make_single_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        record_executed(&mut w, h0, true);
        record_executed(&mut w, h1, true);

        // Emit local vote — captures local_vote_global_receipt_root.
        let (_, local_root, _) = w.build_vote_data(ts_for(WAVE_START + 2)).unwrap();

        // Build a local EC with a DIFFERENT global_receipt_root.
        let outcomes: Vec<TxOutcome> = w
            .tx_hashes()
            .iter()
            .map(|h| TxOutcome::new(*h, executed(true)))
            .collect();
        let divergent_root = GlobalReceiptRoot::from_raw(Hash::from_bytes(b"divergent"));
        assert_ne!(local_root, divergent_root);
        let ec_local = Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
            w.wave_id().clone(),
            WeightedTimestamp::from_millis(WAVE_START.inner() + 1),
            divergent_root,
            outcomes,
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        )));

        w.add_execution_certificate(ec_local);

        assert!(w.is_locally_divergent());
        assert!(!w.is_complete());
    }

    #[test]
    fn not_locally_divergent_when_vote_root_matches_admitted_ec() {
        // Happy path: local execution agrees with quorum. Vote and EC
        // carry the same root, wave finalizes normally.
        let mut w = make_single_shard_wave(1);
        let h0 = w.tx_hashes()[0];

        record_executed(&mut w, h0, true);
        let (_, local_root, outcomes) = w.build_vote_data(ts_for(WAVE_START + 2)).unwrap();

        let ec_local = Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
            w.wave_id().clone(),
            WeightedTimestamp::from_millis(WAVE_START.inner() + 1),
            local_root,
            outcomes,
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        )));
        w.add_execution_certificate(ec_local);

        assert!(!w.is_locally_divergent());
        assert!(w.is_complete());
    }

    #[test]
    fn divergence_caught_when_ec_arrives_before_vote() {
        // Cross-shard race: local EC aggregated and admitted before this
        // validator emits its vote. Reconciliation runs again from
        // `build_vote_data` and catches the mismatch then.
        let mut w = make_single_shard_wave(1);
        let h0 = w.tx_hashes()[0];

        record_executed(&mut w, h0, true);

        // EC arrives first with a root we will NOT match locally.
        let ec_root = GlobalReceiptRoot::from_raw(Hash::from_bytes(b"ec"));
        let ec_local = Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
            w.wave_id().clone(),
            WeightedTimestamp::from_millis(WAVE_START.inner() + 1),
            ec_root,
            vec![TxOutcome::new(h0, executed(true))],
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        )));
        w.add_execution_certificate(ec_local);
        // Vote not yet emitted — divergence undetectable, wave still
        // appears completable.
        assert!(!w.is_locally_divergent());

        // Local vote produces a different root.
        let (_, local_root, _) = w.build_vote_data(ts_for(WAVE_START + 2)).unwrap();
        assert_ne!(local_root, ec_root);

        // Reconciliation now flags divergence.
        assert!(w.is_locally_divergent());
        assert!(!w.is_complete());
    }

    #[test]
    fn aborted_tx_does_not_require_remote_coverage() {
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        // Local EC marks both aborted; tracker.aborted covers h0, h1.
        let ec_local = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[h0, h1], false);
        assert!(w.add_execution_certificate(ec_local));
        // Complete despite remote never sending a matching EC.
        assert!(w.is_complete());
    }

    #[test]
    fn record_abort_is_noop_once_dispatched() {
        // Once a wave has dispatched, mid-flight conflict aborts must not
        // mutate the wave — doing so would introduce receipt-level
        // non-determinism across validators (the conflict batch lands at
        // slightly different offsets from ExecutionBatchCompleted on each
        // node). The fix guards `record_abort` on `dispatched`.
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        w.mark_tx_provisioned(h0, ts_for(WAVE_START + 1));
        w.mark_tx_provisioned(h1, ts_for(WAVE_START + 1));
        let mut provisioning = ProvisioningTracker::new();
        provisioning.seed_provisions(h0, vec![Arc::new(Vec::new())]);
        provisioning.seed_provisions(h1, vec![Arc::new(Vec::new())]);
        assert!(
            w.tick_group_if_ready(&provisioning, &ProvisionalCells::default())
                .is_some()
        );
        assert!(w.fully_dispatched());

        assert!(!w.record_abort(h0, ts_for(WAVE_START + 2)));
        w.record_execution_result(h0, executed(true));
        w.record_execution_result(h1, executed(true));
        // If record_abort had mutated, h0's outcome would have flipped to Aborted.
        let (_, _, outcomes) = w.build_vote_data(ts_for(WAVE_START + 2)).unwrap();
        assert!(matches!(
            outcomes[0].outcome(),
            ExecutionOutcome::Succeeded { .. }
        ));
    }

    #[test]
    fn duplicate_ec_ignored() {
        let mut w = make_cross_shard_wave(1);
        let h0 = w.tx_hashes()[0];
        let ec1 = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[h0], true);
        let ec2 = Arc::clone(&ec1);
        w.add_execution_certificate(ec1);
        let before = w.execution_certificates.len();
        w.add_execution_certificate(ec2);
        assert_eq!(w.execution_certificates.len(), before);
    }

    #[test]
    fn wave_certificate_excludes_remote_covering_only_aborts() {
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];

        // Both sides all-abort.
        let ec_local = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[h0, h1], false);
        let ec_remote = make_ec(w.wave_id(), ShardId::leaf(1, 1), &[h0, h1], false);
        w.add_execution_certificate(ec_local);
        w.add_execution_certificate(ec_remote);

        let wc = w.create_wave_certificate();
        assert_eq!(wc.execution_certificates().len(), 1);
        assert_eq!(
            wc.execution_certificates()[0].wave_id().shard_id(),
            ShardId::leaf(1, 0)
        );
    }

    #[test]
    fn tx_decisions_priority() {
        let mut w = make_cross_shard_wave(3);
        let h0 = w.tx_hashes()[0];
        let h1 = w.tx_hashes()[1];
        let h2 = w.tx_hashes()[2];

        // h0: executed success; h1: abort from remote; h2: failure (non-success exec)
        let ec_local_mixed = make_ec(w.wave_id(), ShardId::leaf(1, 0), &[h0, h1, h2], true);
        w.add_execution_certificate(ec_local_mixed);

        // Remote aborts h1, succeeds h0, h2
        let mut outcomes = vec![
            TxOutcome::new(h0, executed(true)),
            TxOutcome::new(h1, ExecutionOutcome::Aborted),
            TxOutcome::new(h2, executed(false)),
        ];
        let ec_wave_id = WaveId::new(
            ShardId::leaf(1, 1),
            w.wave_id().block_height(),
            w.wave_id().remote_shards().iter().copied().collect(),
        );
        let ec_remote = Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
            ec_wave_id,
            WeightedTimestamp::from_millis(w.wave_id().block_height().inner() + 1),
            GlobalReceiptRoot::from_raw(Hash::from_bytes(b"gr")),
            std::mem::take(&mut outcomes),
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        )));
        w.add_execution_certificate(ec_remote);

        let decisions: HashMap<TxHash, TransactionDecision> =
            w.tx_decisions().into_iter().collect();
        assert_eq!(decisions[&h1], TransactionDecision::Aborted);
        assert_eq!(decisions[&h2], TransactionDecision::Reject);
        assert_eq!(decisions[&h0], TransactionDecision::Accept);
    }

    // ─── tick_group_if_ready ────────────────────────────────────────────

    #[test]
    fn tick_group_single_shard_carries_all_members_and_flips_flag() {
        let mut w = make_single_shard_wave(2);
        let provisioning = ProvisioningTracker::new();
        let group = w
            .tick_group_if_ready(&provisioning, &ProvisionalCells::default())
            .expect("group");
        assert!(group.wave_id.is_zero());
        assert_eq!(group.requests.len(), 2);
        // Single-shard members carry no provisions and the committing
        // block's own anchors.
        for request in &group.requests {
            assert!(request.provisions.is_empty());
            assert_eq!(request.clock, ts_for(WAVE_START));
            assert_eq!(request.randomness, RevealChain::ZERO);
        }
        assert!(w.fully_dispatched());
        assert!(
            w.tick_group_if_ready(&provisioning, &ProvisionalCells::default())
                .is_none()
        );
    }

    /// A member declaring a cell an unresolved wave holds provisionally
    /// stays out of the tick: the value it would read is one nothing may
    /// depend on. It joins the first tick composed after the claim
    /// clears.
    #[test]
    fn a_member_waits_while_its_declared_cell_is_held_provisionally() {
        let mut w = make_single_shard_wave(1);
        let provisioning = ProvisioningTracker::new();

        let mut blocked = ProvisionalCells::default();
        blocked.claim(&w.declared_mutations());
        assert!(
            w.tick_group_if_ready(&provisioning, &blocked).is_none(),
            "a member whose cell is held provisionally must not join a tick"
        );
        assert!(!w.fully_dispatched());

        let group = w
            .tick_group_if_ready(&provisioning, &ProvisionalCells::default())
            .expect("the member joins once the claim clears");
        assert_eq!(group.requests.len(), 1);
        assert!(w.fully_dispatched());
    }

    /// The wait is per member, not per wave: one blocked transaction
    /// must not hold back the rest of its block's batch.
    #[test]
    fn only_the_blocked_member_waits() {
        let mut w = make_single_shard_wave(2);
        let provisioning = ProvisioningTracker::new();

        // `make_tx(seed)` declares `test_prefix(seed + 50)` exclusively,
        // so claiming the first member's prefix names it alone.
        let mut blocked = ProvisionalCells::default();
        blocked.claim(&[(DeclaredKey::prefix(test_prefix(50)), Mode::Write)]);

        let group = w
            .tick_group_if_ready(&provisioning, &blocked)
            .expect("the unblocked member joins");
        assert_eq!(group.requests.len(), 1);
        assert_eq!(group.requests[0].tx_hash, w.tx_hashes()[1]);
        assert!(!w.fully_dispatched(), "the blocked member is still owed");

        let later = w
            .tick_group_if_ready(&provisioning, &ProvisionalCells::default())
            .expect("the blocked member joins a later tick");
        assert_eq!(later.requests.len(), 1);
        assert_eq!(later.requests[0].tx_hash, w.tx_hashes()[0]);
        assert!(w.fully_dispatched());
    }

    /// A wait has to be able to end. Two shards can each hold the cell
    /// the other's counterpart needs, so a member excluded from every
    /// tick would otherwise never get an outcome and its wave would never
    /// vote — the deadline is what breaks the ring.
    #[test]
    fn a_member_blocked_past_the_deadline_aborts() {
        let mut w = make_single_shard_wave(1);
        let provisioning = ProvisioningTracker::new();
        let mut blocked = ProvisionalCells::default();
        blocked.claim(&w.declared_mutations());

        assert!(w.tick_group_if_ready(&provisioning, &blocked).is_none());

        // Short of the deadline the member keeps waiting.
        let deadline = ts_for(WAVE_START).plus(WAVE_TIMEOUT);
        assert!(!w.abort_members_blocked_past_deadline(&blocked, ts_for(WAVE_START)));
        assert!(!w.has_outcome_for_every_tx());

        assert!(
            w.abort_members_blocked_past_deadline(&blocked, deadline),
            "the wave has an outcome for every member once the wait aborts"
        );
        assert!(
            w.tick_group_if_ready(&provisioning, &ProvisionalCells::default())
                .is_none(),
            "an aborted member never executes, even once the cell frees"
        );
    }

    #[test]
    fn dispatch_if_ready_cross_shard_dispatches_the_dependency_free_leg() {
        // A provisioned tx with no stored entries is the leg that never
        // needed any (an empty requirement): it dispatches with empty
        // provisions rather than waiting forever.
        let mut w = make_cross_shard_wave(1);
        let h0 = w.tx_hashes()[0];
        w.mark_tx_provisioned(h0, ts_for(WAVE_START + 1));

        let provisioning = ProvisioningTracker::new();
        let group = w
            .tick_group_if_ready(&provisioning, &ProvisionalCells::default())
            .expect("group");
        assert_eq!(group.requests.len(), 1);
        assert!(group.requests[0].provisions.is_empty());
        assert!(w.fully_dispatched());
    }

    #[test]
    fn dispatch_if_ready_cross_shard_succeeds_with_all_provisions() {
        let mut w = make_cross_shard_wave(1);
        let h0 = w.tx_hashes()[0];
        w.mark_tx_provisioned(h0, ts_for(WAVE_START + 1));

        let mut provisioning = ProvisioningTracker::new();
        provisioning.seed_provisions(h0, vec![Arc::new(Vec::<SubstateEntry>::new())]);

        let group = w
            .tick_group_if_ready(&provisioning, &ProvisionalCells::default())
            .expect("group");
        assert_eq!(group.requests.len(), 1);
        assert_eq!(group.requests[0].tx_hash, h0);
        assert!(w.fully_dispatched());
    }

    // ─── engagement coverage ────────────────────────────────────────────

    /// A committed bundle from `source` naming `tx` — the engagement echo
    /// a counterpart's committing block owes the payer.
    fn echo_from(source: ShardId, tx: TxHash) -> Verified<Provisions> {
        Verified::new_unchecked_for_test(Provisions::new(
            source,
            ShardId::leaf(1, 0),
            BlockHeight::new(3),
            ts_for(WAVE_START + 1),
            RevealChain::ZERO,
            MerkleInclusionProof::dummy(),
            vec![ProvisionEntry::new(tx, vec![])],
        ))
    }

    /// A cross-shard wave whose single transaction this shard pays for,
    /// executed and provisioned, waiting only on the counterpart's echo.
    fn payer_wave_awaiting_echo(validity_end: WeightedTimestamp) -> (WaveState, TxHash) {
        let mut w = make_cross_shard_wave(1);
        let tx = w.tx_hashes()[0];
        w.record_engagement_wait(tx, BTreeSet::from([ShardId::leaf(1, 1)]), validity_end);
        w.mark_tx_provisioned(tx, ts_for(WAVE_START + 1));
        record_executed(&mut w, tx, true);
        (w, tx)
    }

    #[test]
    fn the_payer_vote_waits_for_the_counterparts_echo() {
        let validity_end = ts_for(WAVE_START + 20);
        let (mut w, tx) = payer_wave_awaiting_echo(validity_end);
        let now = ts_for(WAVE_START + 2);

        // Executed and provisioned, but nothing has echoed: no vote.
        assert!(!w.can_emit_vote(now));

        // A bundle from the counterpart is its commitment of the
        // transaction — engagement, and the coverage the vote waits for.
        let mut provisioning = ProvisioningTracker::new();
        provisioning.absorb_provisions(&echo_from(ShardId::leaf(1, 1), tx));
        w.absorb_engagement_evidence(&provisioning);

        assert!(w.engagement_covered());
        assert!(w.can_emit_vote(now));
        let (_, _, outcomes) = w.build_vote_data(now).expect("vote");
        assert!(matches!(
            outcomes[0].outcome(),
            ExecutionOutcome::Succeeded { .. }
        ));
    }

    #[test]
    fn an_unechoed_engagement_aborts_at_the_deadline() {
        let validity_end = ts_for(WAVE_START + 20);
        let (mut w, _) = payer_wave_awaiting_echo(validity_end);

        // The window has closed but the echo margin has not elapsed: the
        // wave still waits rather than forgoing a late engagement.
        assert!(!w.can_emit_vote(validity_end));

        // Past the margin the wave votes anyway, aborting the transaction
        // no counterpart ever engaged — however its own execution went.
        let deadline = validity_end.plus(WAVE_TIMEOUT);
        assert!(w.can_emit_vote(deadline));
        let (_, _, outcomes) = w.build_vote_data(deadline).expect("vote");
        assert_eq!(outcomes[0].outcome(), &ExecutionOutcome::Aborted);
        assert_eq!(outcomes[0].fee_receipt(), None, "no fee receipt to settle");
    }

    #[test]
    fn an_abort_settles_the_payers_fee_receipt_in_place_of_its_effects() {
        // The engine builds a fee receipt beside the execution receipt for
        // a leg this shard pays for. The deadline abort discards the
        // transaction's effects and settles that fee instead — so the
        // finalized wave carries the fee receipt, not the execution one.
        let validity_end = ts_for(WAVE_START + 20);
        let (mut w, tx) = payer_wave_awaiting_echo(validity_end);
        let fee = fee_receipt_for(tx);
        let fee_hash = fee.consensus.receipt_hash();
        w.record_fee_receipt(fee);

        let deadline = validity_end.plus(WAVE_TIMEOUT);
        let (_, _, outcomes) = w.build_vote_data(deadline).expect("vote");
        assert_eq!(outcomes[0].outcome(), &ExecutionOutcome::Aborted);
        assert_eq!(outcomes[0].fee_receipt(), Some(fee_hash));

        // The leaf covers the settled receipt: an aggregator swapping it
        // for another would not reproduce the signed root.
        let bare = TxOutcome::new(tx, ExecutionOutcome::Aborted);
        assert_ne!(tx_outcome_leaf(&outcomes[0]), tx_outcome_leaf(&bare));

        let wave_id = w.wave_id().clone();
        w.add_execution_certificate(make_ec_from(&wave_id, outcomes));
        let finalized = w.into_finalized();
        assert_eq!(finalized.receipts().len(), 1);
        assert_eq!(finalized.receipts()[0].tx_hash, tx);
        assert_eq!(finalized.receipts()[0].consensus.receipt_hash(), fee_hash);
        finalized
            .validate_receipts_against_ec()
            .expect("a settled fee receipt is what the EC named");
    }

    #[test]
    fn a_wave_without_payer_legs_is_never_engagement_gated() {
        // The counterpart's own wave votes on execution alone: it waits
        // for nobody's echo, or the two shards would wait on each other.
        let mut w = make_cross_shard_wave(1);
        let tx = w.tx_hashes()[0];
        w.mark_tx_provisioned(tx, ts_for(WAVE_START + 1));
        record_executed(&mut w, tx, true);

        assert!(w.engagement_covered());
        assert!(w.can_emit_vote(ts_for(WAVE_START + 2)));
    }

    #[test]
    fn dispatch_environment_prefers_the_payer_bundles_anchor() {
        // A remote-payer leg executes under the clock and draw its
        // payer bundle carried; every other leg anchors on this wave's
        // own committing block.
        let mut w = make_cross_shard_wave(2);
        let remote_payer_tx = w.tx_hashes()[0];
        let local_anchor_tx = w.tx_hashes()[1];
        w.mark_tx_provisioned(remote_payer_tx, ts_for(WAVE_START + 1));
        w.mark_tx_provisioned(local_anchor_tx, ts_for(WAVE_START + 1));

        let payer_shard = ShardId::leaf(1, 1);
        let payer_clock = WeightedTimestamp::from_millis(77_777);
        let payer_reveal = RevealChain::from_raw(Hash::from_bytes(b"payer block reveal"));
        let mut provisioning = ProvisioningTracker::new();
        provisioning.record_payer_shard(remote_payer_tx, payer_shard);
        provisioning.absorb_provisions(&Verified::new_unchecked_for_test(Provisions::new(
            payer_shard,
            ShardId::leaf(1, 0),
            BlockHeight::new(3),
            payer_clock,
            payer_reveal,
            MerkleInclusionProof::dummy(),
            vec![ProvisionEntry::new(remote_payer_tx, vec![])],
        )));

        let group = w
            .tick_group_if_ready(&provisioning, &ProvisionalCells::default())
            .expect("group");
        let request_for = |hash: TxHash| {
            group
                .requests
                .iter()
                .find(|r| r.tx_hash == hash)
                .expect("request present")
        };
        assert_eq!(request_for(remote_payer_tx).clock, payer_clock);
        assert_eq!(request_for(remote_payer_tx).randomness, payer_reveal);
        assert_eq!(request_for(local_anchor_tx).clock, ts_for(WAVE_START));
        assert_eq!(request_for(local_anchor_tx).randomness, wave_start_reveal());
    }

    #[test]
    fn tick_group_skips_pre_aborted_txs() {
        let mut w = make_single_shard_wave(2);
        let aborted = w.tx_hashes()[0];
        w.record_abort(aborted, ts_for(WAVE_START));

        let provisioning = ProvisioningTracker::new();
        let group = w
            .tick_group_if_ready(&provisioning, &ProvisionalCells::default())
            .expect("group");
        assert_eq!(group.requests.len(), 1);
        assert_ne!(group.requests[0].tx_hash, aborted);
    }

    #[test]
    fn tick_group_is_none_when_all_txs_aborted() {
        let mut w = make_single_shard_wave(1);
        let aborted = w.tx_hashes()[0];
        w.record_abort(aborted, ts_for(WAVE_START));

        let provisioning = ProvisioningTracker::new();
        assert!(
            w.tick_group_if_ready(&provisioning, &ProvisionalCells::default())
                .is_none()
        );
        assert!(!w.fully_dispatched());
    }

    #[test]
    fn tick_group_is_none_when_not_fully_provisioned() {
        let mut w = make_cross_shard_wave(2);
        let h0 = w.tx_hashes()[0];
        w.mark_tx_provisioned(h0, ts_for(WAVE_START + 1));

        let provisioning = ProvisioningTracker::new();
        assert!(
            w.tick_group_if_ready(&provisioning, &ProvisionalCells::default())
                .is_none()
        );
        assert!(!w.fully_dispatched());
    }
}
