//! Per-tick attestation state.
//!
//! One `TickState` owns what a tick attests, from the moment its batch
//! is composed to the moment its finalization is handed off: the local
//! execution results, the vote this validator casts over them, and the
//! participating shards' certificates that complete each member's
//! coverage.
//!
//! ## What a tick holds
//!
//! Exactly the transactions that executed in it, plus the ones it
//! abandons. Both are decided at composition — a member that could not
//! reach its outcome in this tick is not in it, it is still waiting in
//! [`TickCandidates`](crate::candidates::TickCandidates) — so a tick has
//! an outcome for every member the moment its batch returns, and no
//! member of it waits on another.
//!
//! ## Lifecycle
//!
//! 1. **Composed** at a block commit, from the candidates that could
//!    execute there. The block is the tick's own: its hash, its weighted
//!    timestamp, and the committee seated at it.
//! 2. **Executes** as one batch. Results land via
//!    `record_execution_result`.
//! 3. **Votes** once every dispatched member has a result. The vote is
//!    one-shot.
//! 4. **Collects certificates** from all participating shards via
//!    `add_execution_certificate`. When every member is covered (or
//!    aborted, which is terminal-covered), the tick is complete and ready
//!    for finalization.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use hyperscale_types::{
    BlockHash, BlockHeight, ExecutionCertificate, ExecutionOutcome, Finalization,
    GlobalReceiptRoot, MAX_FINALIZATION_DELAY, Settles, ShardId, StoredReceipt, TickId,
    TransactionDecision, TxHash, TxOutcome, Verified, WeightedTimestamp,
    compute_global_receipt_root, refused_transactions, settles,
};

/// A tick whose local execution disagreed with the quorum's.
///
/// The receipt root the validator voted against the one its committee
/// certified: direct proof that this node's execution produced different
/// writes from the same committed chain.
#[derive(Debug, Clone)]
pub struct Divergence {
    /// The tick whose roots disagreed.
    pub tick_id: TickId,
    /// The hash of the block whose commit composed it.
    pub block_hash: BlockHash,
    /// The root this validator voted.
    pub local_root: GlobalReceiptRoot,
    /// The root its committee certified.
    pub ec_root: GlobalReceiptRoot,
}

/// Age at which a still-unresolved tick emits a single diagnostic warning.
///
/// Every committed transaction is supposed to reach a certificate — its
/// tick's, or the later tick that abandons it at its deadline. The
/// threshold sits past the deadline so a tick resolving through either
/// path passes silently. A firing means the post-inclusion termination
/// guarantee has failed, so the dump is invariant-violation diagnostics
/// rather than routine load noise.
pub const TICK_OVERDUE_WARN: Duration = Duration::from_secs(MAX_FINALIZATION_DELAY.as_secs() * 2);

/// Per-tick state from composition through finalization.
#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)] // independent lifecycle flags, not config knobs
pub struct TickState {
    // ── Identity ────────────────────────────────────────────────────────
    tick_id: TickId,
    block_hash: BlockHash,
    /// The tick's own block's BFT-authenticated weighted timestamp: the
    /// vote anchor, and so the committee that attests.
    tick_ts: WeightedTimestamp,

    // ── Membership (in composition order) ───────────────────────────────
    tx_hashes: Vec<TxHash>,
    /// O(1) membership check (mirrors `tx_hashes`).
    tx_hash_set: HashSet<TxHash>,
    /// Participating shards per member — the shards whose certificates
    /// must cover it for completion. Always includes the local shard.
    participating_shards: HashMap<TxHash, BTreeSet<ShardId>>,
    /// Per-member, the reservation its committing block took against the
    /// drain. Carried apart from the transaction because an abandoned
    /// member has no body here and still has to release exactly what was
    /// taken.
    reserved_work: HashMap<TxHash, u64>,
    /// Members dispatched to the engine and still owing a result. Empty
    /// is what makes the tick votable.
    awaiting_results: HashSet<TxHash>,
    /// Members this tick attests `Aborted` whatever their execution said:
    /// a payer's leg whose counterparts never engaged, and the ones
    /// abandoned past their own deadline.
    aborted: HashSet<TxHash>,
    /// Members abandoned rather than executed. A subset of `aborted`,
    /// kept apart because it is what tells a later commit this tick is
    /// already speaking for them.
    abandoned: HashSet<TxHash>,

    // ── Local execution outputs ─────────────────────────────────────────
    /// Execution results from the engine, per member.
    execution_results: HashMap<TxHash, ExecutionOutcome>,
    /// Local receipts from the engine, one per executed member. Drained
    /// into the `Finalization` at finalization via `take_receipt`. Scoping
    /// these to the tick (rather than a process-wide cache) prevents a
    /// receipt from a locally-executed transaction from leaking into a
    /// `Finalization` whose certificate attests it `Aborted` — the
    /// `ExtraReceipt` race.
    execution_receipts: HashMap<TxHash, StoredReceipt>,
    /// Per-member fee receipts the engine built alongside the execution
    /// receipt, for the cross-shard transactions this shard pays for.
    /// An abort settles one of these: the transaction's own effects are
    /// discarded, the payer's floor is not.
    fee_receipts: HashMap<TxHash, StoredReceipt>,
    /// What this shard attests it did per member, carried from execution
    /// onto the outcomes it votes.
    attested_work: HashMap<TxHash, u64>,
    /// Whether the local vote has been emitted (`build_vote_data` called once).
    voted: bool,
    /// `global_receipt_root` carried on this validator's own emitted vote.
    /// Reconciled against `admitted_local_ec_root` to detect divergence.
    local_vote_global_receipt_root: Option<GlobalReceiptRoot>,
    /// `global_receipt_root` from the admitted local certificate. May
    /// arrive before the local vote, when peers aggregate it before this
    /// validator's engine finishes.
    admitted_local_ec_root: Option<GlobalReceiptRoot>,
    /// Set when the admitted local certificate's `global_receipt_root`
    /// disagreed with `local_vote_global_receipt_root`. Bars the tick from
    /// finalizing locally so divergent receipts cannot enter the
    /// `finalized` store, propagate via `cert_bloom`, or be re-served on
    /// sync — which matters for the window before the coordinator
    /// escalates, not as an outcome. There is no recovery from here: a
    /// wrong tick output is the baseline every later tick reads.
    locally_divergent: bool,
    /// The mismatch behind the latch, until the coordinator reports it.
    divergence: Option<Divergence>,
    /// Whether the determined half's finalization has been handed off.
    determined_emitted: bool,
    /// Whether the legs half's finalization has been handed off.
    legs_emitted: bool,
    /// Members a committed finalization has resolved. The tick's
    /// certificate and tracking state drop once every member is here —
    /// which is later than the halves being emitted, since a handed-off
    /// finalization still has to reach a block.
    settled: HashSet<TxHash>,
    /// Whether the local certificate has been added to
    /// `execution_certificates`. Gates completion. Independent of the
    /// canonical-root reconciliation — `locally_divergent` carries the
    /// divergence verdict separately.
    local_ec_emitted: bool,
    /// Latches `log_if_overdue`: fires once per tick after crossing the
    /// `TICK_OVERDUE_WARN` threshold. Under ts-based ages we can't rely on
    /// exact equality (commits can skip over any given ms value).
    overdue_warned: bool,

    // ── Cross-shard certificate collection ──────────────────────────────
    /// Per-member, which shards have reported via a certificate.
    covered_shards: HashMap<TxHash, BTreeSet<ShardId>>,
    /// Per-member, whether any shard's certificate reported abort.
    /// Terminal — an aborted transaction needs no further coverage.
    tracker_aborted: HashSet<TxHash>,
    /// Per-member, whether any shard's certificate reported a non-success
    /// outcome.
    tx_has_failure: HashSet<TxHash>,
    /// All collected certificates (local + remote).
    execution_certificates: Vec<Arc<Verified<ExecutionCertificate>>>,
}

impl TickState {
    /// An empty tick at `tick_id`, anchored on the block whose commit
    /// composed it. Members are admitted by
    /// [`admit`](Self::admit) and [`abandon`](Self::abandon).
    #[must_use]
    pub fn new(tick_id: TickId, block_hash: BlockHash, tick_ts: WeightedTimestamp) -> Self {
        Self {
            tick_id,
            block_hash,
            tick_ts,
            tx_hashes: Vec::new(),
            tx_hash_set: HashSet::new(),
            participating_shards: HashMap::new(),
            reserved_work: HashMap::new(),
            awaiting_results: HashSet::new(),
            aborted: HashSet::new(),
            abandoned: HashSet::new(),
            execution_results: HashMap::new(),
            execution_receipts: HashMap::new(),
            fee_receipts: HashMap::new(),
            attested_work: HashMap::new(),
            voted: false,
            local_vote_global_receipt_root: None,
            admitted_local_ec_root: None,
            locally_divergent: false,
            divergence: None,
            determined_emitted: false,
            legs_emitted: false,
            settled: HashSet::new(),
            local_ec_emitted: false,
            overdue_warned: false,
            covered_shards: HashMap::new(),
            tracker_aborted: HashSet::new(),
            tx_has_failure: HashSet::new(),
            execution_certificates: Vec::new(),
        }
    }

    /// Admit a member this tick's batch executes.
    ///
    /// `forced_abort` marks the one whose verdict is decided before the
    /// result comes back — the payer's leg whose counterparts never
    /// engaged. It still executes, because the charge its abort settles
    /// is what that execution builds.
    pub fn admit(
        &mut self,
        tx_hash: TxHash,
        participating: BTreeSet<ShardId>,
        reserved_work: u64,
        forced_abort: bool,
    ) {
        if !self.enrol(tx_hash, participating, reserved_work) {
            return;
        }
        self.awaiting_results.insert(tx_hash);
        if forced_abort {
            self.aborted.insert(tx_hash);
        }
    }

    /// Admit a transaction this shard can no longer finalize, to be
    /// attested `Aborted` without executing.
    ///
    /// It joins with its outcome already decided and no body: the ledger
    /// names it by hash and by the work its committing block reserved,
    /// which is all an abort has to state. The only shard whose
    /// certificate it waits for is this one — an abort is dominant, so no
    /// counterpart can contradict it.
    pub fn abandon(&mut self, tx_hash: TxHash, reserved_work: u64) {
        let participating = BTreeSet::from([self.tick_id.shard_id()]);
        if !self.enrol(tx_hash, participating, reserved_work) {
            return;
        }
        self.aborted.insert(tx_hash);
        self.abandoned.insert(tx_hash);
    }

    /// Shared membership bookkeeping. Returns whether the hash was new —
    /// a member the tick already holds keeps the terms it joined under.
    fn enrol(
        &mut self,
        tx_hash: TxHash,
        participating: BTreeSet<ShardId>,
        reserved_work: u64,
    ) -> bool {
        if !self.tx_hash_set.insert(tx_hash) {
            return false;
        }
        self.tx_hashes.push(tx_hash);
        self.participating_shards.insert(tx_hash, participating);
        self.reserved_work.insert(tx_hash, reserved_work);
        self.covered_shards.insert(tx_hash, BTreeSet::new());
        true
    }

    /// Whether this tick is still going to attest `tx_hash`: its own
    /// certificate has not formed yet, or the verdict that certificate
    /// carries is the abandonment.
    ///
    /// The negative case is the one that matters — a tick whose local
    /// certificate exists has said everything it will say, so a member of
    /// it still awaiting a counterpart is stranded rather than in hand.
    #[must_use]
    pub fn will_attest(&self, tx_hash: TxHash) -> bool {
        self.tx_hash_set.contains(&tx_hash)
            && (!self.local_ec_emitted || self.abandoned.contains(&tx_hash))
    }

    // ── Identity getters ────────────────────────────────────────────────

    /// The tick's identity.
    #[must_use]
    pub const fn tick_id(&self) -> &TickId {
        &self.tick_id
    }

    /// Hash of the block whose commit composed this tick.
    #[must_use]
    pub const fn block_hash(&self) -> BlockHash {
        self.block_hash
    }

    /// Height of that block (mirrors `tick_id.block_height`).
    #[must_use]
    pub const fn block_height(&self) -> BlockHeight {
        self.tick_id.block_height()
    }

    /// Members, in composition order.
    #[must_use]
    pub fn tx_hashes(&self) -> &[TxHash] {
        &self.tx_hashes
    }

    /// Whether the tick holds no members at all.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tx_hashes.is_empty()
    }

    /// The subset of [`Self::tx_hashes`] that reaches beyond this shard —
    /// the ones whose settlement waits on a counterpart.
    pub fn cross_shard_tx_hashes(&self) -> impl Iterator<Item = TxHash> + '_ {
        self.tx_hashes
            .iter()
            .copied()
            .filter(|&tx_hash| self.reaches_beyond(tx_hash))
    }

    /// Whether the local certificate has been fed into this tick (via
    /// `add_execution_certificate` with `ec.tick_id() == &self.tick_id`).
    #[must_use]
    pub const fn local_ec_emitted(&self) -> bool {
        self.local_ec_emitted
    }

    /// Whether `tx_hash` reaches beyond this shard.
    ///
    /// A member with a participant other than this shard executes
    /// provisionally and stays abortable on that participant's verdict;
    /// one without does neither.
    fn reaches_beyond(&self, tx_hash: TxHash) -> bool {
        let local = self.tick_id.shard_id();
        self.participating_shards
            .get(&tx_hash)
            .is_some_and(|shards| shards.iter().any(|&s| s != local))
    }

    /// The shards other than this one that the tick's members name as
    /// participants — who its certificate is owed to.
    #[must_use]
    pub fn counterpart_shards(&self) -> Vec<ShardId> {
        let local = self.tick_id.shard_id();
        self.participating_shards
            .values()
            .flatten()
            .copied()
            .filter(|&s| s != local)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// The tick's members `shard` is a participant in — what this tick is
    /// waiting on that shard for.
    pub fn txs_awaiting(&self, shard: ShardId) -> impl Iterator<Item = TxHash> + '_ {
        self.tx_hashes.iter().copied().filter(move |tx_hash| {
            self.participating_shards
                .get(tx_hash)
                .is_some_and(|shards| shards.contains(&shard))
        })
    }

    // ── Local execution bookkeeping ─────────────────────────────────────

    /// Record an execution outcome from the engine. First-write-wins.
    pub fn record_execution_result(&mut self, tx_hash: TxHash, outcome: ExecutionOutcome) {
        if !self.tx_hash_set.contains(&tx_hash) {
            return;
        }
        self.execution_results.entry(tx_hash).or_insert(outcome);
        self.awaiting_results.remove(&tx_hash);
    }

    /// Record a local receipt from the engine. First-write-wins.
    ///
    /// Paired with `record_execution_result`: both flow from the same
    /// `ExecutionBatchCompleted` event and are scoped to this tick.
    /// Receipts for transactions not in it are silently dropped.
    pub fn record_receipt(&mut self, receipt: StoredReceipt) {
        if !self.tx_hash_set.contains(&receipt.tx_hash) {
            return;
        }
        self.execution_receipts
            .entry(receipt.tx_hash)
            .or_insert(receipt);
    }

    /// Record what this shard attested it did for a member.
    pub fn record_attested_work(&mut self, tx_hash: TxHash, work: u64) {
        if !self.tx_hash_set.contains(&tx_hash) {
            return;
        }
        self.attested_work.insert(tx_hash, work);
    }

    /// Record the fee receipt the engine built beside a member's
    /// execution receipt: what the payer owes if the transaction aborts.
    pub fn record_fee_receipt(&mut self, receipt: StoredReceipt) {
        if !self.tx_hash_set.contains(&receipt.tx_hash) {
            return;
        }
        self.fee_receipts.entry(receipt.tx_hash).or_insert(receipt);
    }

    /// Number of receipts currently held. Exposed for memory stats;
    /// receipts drain at finalization.
    #[must_use]
    pub fn receipt_count(&self) -> usize {
        self.execution_receipts.len()
    }

    /// Take the receipt for a member, removing it. Used internally by
    /// [`Self::into_finalization`] to drain receipts in canonical order.
    fn take_receipt(&mut self, tx_hash: TxHash) -> Option<StoredReceipt> {
        self.execution_receipts.remove(&tx_hash)
    }

    /// Whether this validator holds a local receipt for every member of
    /// `members` the local certificate attests as executed. Aborted
    /// outcomes need no receipt.
    ///
    /// Gates each half so it can't produce a [`Finalization`] that fails
    /// [`Finalization::validate_receipts_against_ec`]. The check mirrors
    /// that invariant: a receipt is needed exactly for the outcomes the
    /// certificate attests as `Executed`. When this validator's local
    /// decision disagrees with the quorum's, the gate blocks here rather
    /// than synthesizing a `Finalization` with missing receipts. Recovery
    /// flows through the existing peer-fetch path.
    ///
    /// **Scoped to one half, because each half drains its own receipts.**
    /// A tick-wide check would pass before the first half settles and
    /// fail afterwards, wedging the second on receipts the first
    /// legitimately consumed.
    ///
    /// Returns false if the local certificate hasn't arrived yet;
    /// `local_ec_emitted` is checked separately for the same reason.
    ///
    /// [`Finalization`]: hyperscale_types::Finalization
    /// [`Finalization::validate_receipts_against_ec`]:
    ///     hyperscale_types::Finalization::validate_receipts_against_ec
    fn has_local_receipts_for(&self, members: &[TxHash]) -> bool {
        let Some(local_ec) = self.local_certificate() else {
            return false;
        };
        local_ec
            .tx_outcomes()
            .iter()
            .filter(|outcome| members.contains(&outcome.tx_hash()))
            .all(|outcome| {
                outcome.is_aborted() || self.execution_receipts.contains_key(&outcome.tx_hash())
            })
    }

    /// This tick's own certificate, in the copy that carries the whole
    /// batch.
    ///
    /// The store can hold more than one copy of a tick — a broadcast and
    /// a narrower fetch answer both land, and the retention rule keeps
    /// whichever covers something new — so the first match is not
    /// necessarily the complete one. Projecting a half needs sibling
    /// nodes from the whole receipt tree, which only the complete copy
    /// carries, and a shard sends its own peers exactly that.
    fn local_certificate(&self) -> Option<&Arc<Verified<ExecutionCertificate>>> {
        self.execution_certificates
            .iter()
            .filter(|ec| ec.tick_id() == &self.tick_id)
            .find(|ec| ec.is_complete())
    }

    // ── Vote emission ───────────────────────────────────────────────────

    /// Vote anchor timestamp: the tick's own block's BFT-authenticated
    /// weighted timestamp.
    ///
    /// This rides the vote payload and the certificate's canonical hash,
    /// and [`VoteTracker`] groups votes by it, so every validator must
    /// derive the same value or agreeing votes never aggregate. Only
    /// committed chain content carries that guarantee, and the block that
    /// composed the tick is exactly that — identical on every replica,
    /// and already in the past when the vote is built, so the committee it
    /// resolves is available at once.
    ///
    /// It is also the right committee on the merits: the one seated at the
    /// tick is the one that holds the state and ran the batch, which is
    /// not always the one that committed the transactions.
    ///
    /// [`VoteTracker`]: crate::vote_tracker::VoteTracker
    #[must_use]
    pub const fn vote_anchor_ts(&self) -> WeightedTimestamp {
        self.tick_ts
    }

    /// Whether the local vote can be emitted.
    ///
    /// One condition, because composition established the rest: every
    /// member the batch was given has come back. A member that could not
    /// reach its outcome here was never admitted, so nothing in a tick
    /// waits on anything outside it.
    #[must_use]
    pub fn can_emit_vote(&self) -> bool {
        !self.voted && self.awaiting_results.is_empty()
    }

    /// Build the vote payload, consuming the one-shot vote.
    ///
    /// Returns `(vote_anchor_ts, global_receipt_root, tx_outcomes)`, or
    /// `None` if [`Self::can_emit_vote`] is false.
    ///
    /// # Panics
    ///
    /// Panics if a member has neither a decided abort nor an execution
    /// result. `can_emit_vote` guards against this; a panic here would
    /// indicate a bug in the gating.
    pub fn build_vote_data(
        &mut self,
    ) -> Option<(WeightedTimestamp, GlobalReceiptRoot, Vec<TxOutcome>)> {
        if !self.can_emit_vote() {
            return None;
        }

        let outcomes: Vec<TxOutcome> = self
            .tx_hashes
            .iter()
            .map(|tx_hash| {
                let outcome = if self.aborted.contains(tx_hash) {
                    ExecutionOutcome::Aborted
                } else {
                    self.execution_results
                        .get(tx_hash)
                        .cloned()
                        .expect("a votable tick holds a result for every member it ran")
                };
                // The charge this shard holds against the transaction, if
                // the engine built one. Named whatever the local outcome
                // was, because the local outcome is not what decides
                // whether it is owed: a leg that completed here still
                // owes the floor if a counterpart refuses it, and a
                // charge nobody named is a charge nothing can settle.
                let work = self.attested_work.get(tx_hash).copied().unwrap_or(0);
                // What the transaction reserved when its block committed
                // it, carried so the settling block can release exactly
                // that. A member the tick could not price would release
                // less than its block took, and the drain keeps the
                // difference for as long as the chain runs — so the
                // figure is required, not defaulted.
                let reserved = *self
                    .reserved_work
                    .get(tx_hash)
                    .expect("a tick prices every member it names");
                let charge = self
                    .fee_receipts
                    .get(tx_hash)
                    .map(|fee| fee.consensus.receipt_hash());
                match charge {
                    Some(fee) => TxOutcome::with_fee(*tx_hash, outcome, fee, work),
                    None => TxOutcome::attesting(*tx_hash, outcome, work),
                }
                .reserving(reserved)
            })
            .collect();

        let root = compute_global_receipt_root(&outcomes);
        self.voted = true;
        self.local_vote_global_receipt_root = Some(root);
        self.reconcile_local_ec_root();
        Some((self.tick_ts, root, outcomes))
    }

    // ── Cross-shard certificate collection ──────────────────────────────

    /// Feed a certificate into the tick: update per-member coverage, track
    /// aborts and failures, and keep it. For our own local certificate
    /// (`ec.tick_id() == &self.tick_id`), records the admitted root and
    /// reconciles against the local vote when both are known. The local
    /// certificate may arrive before the local vote when peers aggregate
    /// it before this validator's engine finishes; the reconciliation runs
    /// again from `build_vote_data` once the local vote lands.
    ///
    /// **A certificate is kept exactly when it covers something this tick
    /// does not already have covered.** A certificate carries the outcomes
    /// naming its holder, so two copies of one tick can differ — the
    /// broadcast a shard sends and a narrower one a fetch answered with —
    /// and dropping the second because its tick is familiar would leave
    /// this tick believing a transaction covered by an outcome it does not
    /// hold. That is not a missing optimisation but a hole: the outcome it
    /// would have dropped could be the counterpart's abort, and settling
    /// its sibling without it moves value one-sidedly.
    ///
    /// The same test is the bound. A peer can synthesise arbitrarily many
    /// valid narrower copies of one tick, so keeping every distinct copy
    /// would be unbounded; keeping only those that cover something new
    /// caps the collection at one certificate per transaction.
    ///
    /// Readiness is the caller's next question — a certificate can make
    /// either half settleable, or neither.
    pub fn add_execution_certificate(&mut self, ec: Arc<Verified<ExecutionCertificate>>) {
        let shard = ec.shard_id();
        let is_local = ec.tick_id() == &self.tick_id;

        let covers_something_new = ec.tx_outcomes().iter().any(|outcome| {
            self.covered_shards
                .get(&outcome.tx_hash())
                .is_some_and(|covered| !covered.contains(&shard))
        });
        // An empty tick's own certificate covers nothing yet still has to
        // land: both halves gate on having emitted it.
        let first_local = is_local && !self.local_ec_emitted;
        if !covers_something_new && !first_local {
            return;
        }

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
    }

    /// Compare `local_vote_global_receipt_root` against
    /// `admitted_local_ec_root` once both are known. Run from both sites
    /// that can supply the second half of the pair: `build_vote_data` and
    /// `add_execution_certificate`.
    fn reconcile_local_ec_root(&mut self) {
        let (Some(local), Some(admitted)) = (
            self.local_vote_global_receipt_root,
            self.admitted_local_ec_root,
        ) else {
            return;
        };
        if local == admitted || self.locally_divergent {
            return;
        }
        self.locally_divergent = true;
        self.divergence = Some(Divergence {
            tick_id: self.tick_id,
            block_hash: self.block_hash,
            local_root: local,
            ec_root: admitted,
        });
    }

    /// Take the latched divergence, if any, for the coordinator to report.
    pub const fn take_divergence(&mut self) -> Option<Divergence> {
        self.divergence.take()
    }

    /// Whether this validator's execution disagreed with its committee's.
    #[must_use]
    pub const fn is_locally_divergent(&self) -> bool {
        self.locally_divergent
    }

    /// The members whose settlement needs no shard but this one — the
    /// single-shard transactions plus the ones this tick abandons. Their
    /// outcome is decided by the tick's own certificate, so nothing a
    /// counterpart does can hold them.
    #[must_use]
    pub fn determined_members(&self) -> Vec<TxHash> {
        self.tx_hashes
            .iter()
            .copied()
            .filter(|&tx_hash| !self.reaches_beyond(tx_hash))
            .collect()
    }

    /// The members whose settlement waits on a counterpart.
    #[must_use]
    pub fn leg_members(&self) -> Vec<TxHash> {
        self.cross_shard_tx_hashes().collect()
    }

    /// Conditions both halves share: the certificate exists and this
    /// validator agreed with it. Receipts are checked per half, since
    /// each drains its own.
    const fn attestable(&self) -> bool {
        self.local_ec_emitted && !self.locally_divergent
    }

    /// Whether `tx_hash` has the verdict its settlement needs: an abort
    /// anywhere is terminal, and anything else waits for every
    /// participating shard to have certified it.
    fn is_covered(&self, tx_hash: TxHash) -> bool {
        if self.tracker_aborted.contains(&tx_hash) {
            return true;
        }
        match (
            self.participating_shards.get(&tx_hash),
            self.covered_shards.get(&tx_hash),
        ) {
            (Some(expected), Some(covered)) => expected.is_subset(covered),
            _ => false,
        }
    }

    /// Whether the determined half can settle now.
    ///
    /// It needs no coverage beyond the tick's own certificate, which is
    /// the whole point: a leg waiting on a counterpart that never comes
    /// cannot hold these members, and the settlement frontier they
    /// advance therefore cannot wedge.
    #[must_use]
    pub fn determined_ready(&self) -> bool {
        let members = self.determined_members();
        !self.determined_emitted
            && self.attestable()
            && !members.is_empty()
            && self.has_local_receipts_for(&members)
    }

    /// Whether the legs half can settle now: every leg has its verdict,
    /// and the determined half has already gone.
    ///
    /// The ordering is not bookkeeping. A determined member and a leg of
    /// one tick are scheduled sequentially inside a single batch, so each
    /// receipt states an absolute computed at its own point in the fold;
    /// settling them out of that order reverts one. The provisional-cell
    /// rule does not cover the pair — it governs across ticks, and these
    /// are one tick.
    #[must_use]
    pub fn legs_ready(&self) -> bool {
        let legs = self.leg_members();
        !self.legs_emitted
            && self.attestable()
            && !legs.is_empty()
            && self.determined_emitted_or_absent()
            && self.has_local_receipts_for(&legs)
            && legs.iter().all(|&tx_hash| self.is_covered(tx_hash))
    }

    /// Whether the determined half is out of the way — emitted, or never
    /// existed because the tick ran nothing but legs.
    fn determined_emitted_or_absent(&self) -> bool {
        self.determined_emitted || self.determined_members().is_empty()
    }

    /// Whether this tick has said everything it will say.
    #[must_use]
    pub fn has_spoken(&self) -> bool {
        self.determined_emitted_or_absent() && (self.legs_emitted || self.leg_members().is_empty())
    }

    /// Record that a committed finalization resolved these members.
    /// Returns whether every member of the tick is now settled, which is
    /// when its certificate and tracking state can be dropped.
    pub fn record_settled(&mut self, tx_hashes: impl IntoIterator<Item = TxHash>) -> bool {
        for tx_hash in tx_hashes {
            if self.tx_hash_set.contains(&tx_hash) {
                self.settled.insert(tx_hash);
            }
        }
        self.tx_hashes
            .iter()
            .all(|tx_hash| self.settled.contains(tx_hash))
    }

    /// Emit a `warn!` log exactly once, when the tick reaches
    /// `TICK_OVERDUE_WARN` of age without completing. A firing is an
    /// invariant violation — every member is supposed to terminate with a
    /// `Finalization` — so the dump captures enough state to diagnose
    /// where the post-inclusion termination guarantee broke. Latched at
    /// the first crossing so it fires once per stuck tick, not once per
    /// surviving commit.
    pub fn log_if_overdue(&mut self, committed_ts: WeightedTimestamp) {
        if self.overdue_warned {
            return;
        }
        let age = committed_ts.elapsed_since(self.tick_ts);
        if age < TICK_OVERDUE_WARN {
            return;
        }
        self.overdue_warned = true;

        let total = self.tx_hashes.len();

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

        let local_receipts_ready = self.has_local_receipts_for(&self.tx_hashes);

        tracing::warn!(
            tick = %self.tick_id,
            block_hash = ?self.block_hash,
            block_height = self.tick_id.block_height().inner(),
            tick_ts = self.tick_ts.as_millis(),
            committed_ts = committed_ts.as_millis(),
            age_ms = u64::try_from(age.as_millis()).unwrap_or(u64::MAX),
            num_txs = total,
            awaiting_results = self.awaiting_results.len(),
            voted = self.voted,
            local_ec_emitted = self.local_ec_emitted,
            local_receipts_ready,
            execution_results = self.execution_results.len(),
            aborted = self.aborted.len(),
            tracker_aborted = self.tracker_aborted.len(),
            ecs_collected = self.execution_certificates.len(),
            determined_ready = self.determined_ready(),
            legs_ready = self.legs_ready(),
            determined_emitted = self.determined_emitted,
            legs_emitted = self.legs_emitted,
            missing_coverage = missing_coverage.join(" "),
            "Tick overdue: unresolved past the deadline that bounds every member"
        );
    }

    /// Build one half's attestation: the tick's own certificate
    /// projected to `members`, plus the counterpart certificates those
    /// members still need a verdict from.
    ///
    /// **The projection is what makes each half account for itself.**
    /// `tx_hashes`, `tx_count` and `declared_work` all read the local
    /// certificate, so projecting it to this half's members is what makes
    /// the half name exactly its own transactions and release exactly
    /// their reservations — no new field, and the drain term stays
    /// readable off the block. Both projections verify under the same
    /// signed root and signature as the copy they came from.
    ///
    /// A remote certificate is included when it covers a member this half
    /// still needs a verdict on, or when it is the certificate carrying
    /// that member's abort. That second clause is what keeps the two
    /// sides of a settlement in agreement. `tracker_aborted` is fed by
    /// the very certificates being filtered here — a remote abort lands
    /// as coverage *and* as an entry in that set — so pruning on
    /// `tracker_aborted` alone discards the only artifact carrying that
    /// verdict. Every downstream reader derives the outcome from the
    /// certificate and nothing else ([`Finalization::tx_decisions`]), so
    /// what that drops is not merely redundant: the local certificate's
    /// success stands unopposed and this shard commits an accept against
    /// the counterparty's abort.
    ///
    /// Returns `None` when this half has no members, or when the local
    /// certificate has not landed yet.
    #[must_use]
    fn attestation_for(&self, members: &HashSet<TxHash>) -> Option<Finalization> {
        let local = self.local_certificate()?;
        // What the local certificate says on its own. A member it already
        // reports as aborted needs no remote to corroborate it.
        let locally_aborted: HashSet<TxHash> = local
            .tx_outcomes()
            .iter()
            .filter(|outcome| outcome.is_aborted())
            .map(TxOutcome::tx_hash)
            .collect();

        let projected = local.project_to(members)?;

        let required_remote_tick_ids: HashSet<TickId> = self
            .execution_certificates
            .iter()
            .filter(|ec| ec.tick_id() != &self.tick_id)
            .filter(|ec| {
                ec.tx_outcomes().iter().any(|outcome| {
                    let tx_hash = outcome.tx_hash();
                    if !members.contains(&tx_hash) {
                        return false;
                    }
                    // Still awaiting a verdict, or holding the only one that
                    // says abort.
                    !self.tracker_aborted.contains(&tx_hash)
                        || (outcome.is_aborted() && !locally_aborted.contains(&tx_hash))
                })
            })
            .map(|ec| *ec.tick_id())
            .collect();

        let mut ecs: Vec<Verified<ExecutionCertificate>> = std::iter::once(projected)
            .chain(
                self.execution_certificates
                    .iter()
                    .filter(|ec| required_remote_tick_ids.contains(ec.tick_id()))
                    .map(|verified| (**verified).clone()),
            )
            .collect();
        ecs.sort_by(|a, b| (&a.shard_id(), a.tick_id()).cmp(&(&b.shard_id(), b.tick_id())));

        Some(Finalization::from_verified_ecs(self.tick_id, ecs))
    }

    /// Drain one stored receipt per outcome of `attestation` that settles
    /// anything, in the certificate's canonical order.
    ///
    /// Which side of an outcome settles is [`settles`]'s question, read
    /// against the whole certificate rather than against this shard's own
    /// verdict: a leg that completed here and was refused by a counterpart
    /// settles its charge, not its effects. Peers re-derive the same rule
    /// through `validate_receipts_against_ec` at ingress.
    fn with_drained_receipts(&mut self, attestation: Finalization) -> Finalization {
        let local_ec = attestation
            .execution_certificates()
            .iter()
            .find(|ec| ec.tick_id() == attestation.tick_id())
            .expect("a half carries the projection of its own tick's certificate")
            .clone();
        let refused = refused_transactions(attestation.execution_certificates());
        let mut receipts: Vec<StoredReceipt> = Vec::with_capacity(local_ec.tx_outcomes().len());
        for outcome in local_ec.tx_outcomes() {
            let drained = match settles(outcome, &refused) {
                // The charge stands in for whatever the transaction did:
                // a `Failed` receipt carries nothing, and a completed
                // leg's effects are discarded by the refusal. Either way
                // the pairing stays one receipt per outcome.
                Settles::Charge(_) => {
                    self.take_receipt(outcome.tx_hash());
                    self.fee_receipts.remove(&outcome.tx_hash())
                }
                Settles::Effects(_) | Settles::Failure => self.take_receipt(outcome.tx_hash()),
                Settles::Nothing => continue,
            };
            if let Some(receipt) = drained {
                receipts.push(receipt);
            } else {
                tracing::error!(
                    tick = %self.tick_id,
                    tx_hash = ?outcome.tx_hash(),
                    "an outcome that settles something is missing its stored receipt \
                     (readiness gate bypassed)"
                );
            }
        }
        attestation.with_receipts(receipts)
    }

    /// Take the determined half's finalization, if it is ready.
    ///
    /// Settles on the tick's own certificate alone, so a leg waiting on a
    /// counterpart cannot hold it — which is what keeps the settlement
    /// frontier advancing whatever a counterpart does.
    pub fn take_determined_finalization(&mut self) -> Option<Finalization> {
        if !self.determined_ready() {
            return None;
        }
        let members: HashSet<TxHash> = self.determined_members().into_iter().collect();
        let attestation = self.attestation_for(&members)?;
        self.determined_emitted = true;
        Some(self.with_drained_receipts(attestation))
    }

    /// Take the legs half's finalization, if it is ready.
    pub fn take_legs_finalization(&mut self) -> Option<Finalization> {
        if !self.legs_ready() {
            return None;
        }
        let members: HashSet<TxHash> = self.leg_members().into_iter().collect();
        let attestation = self.attestation_for(&members)?;
        self.legs_emitted = true;
        Some(self.with_drained_receipts(attestation))
    }

    /// Per-member terminal decisions derived from collected certificates.
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
    use hyperscale_types::{
        AggregateSignature, ConsensusReceipt, GlobalReceiptHash, Hash, SignerBitfield,
    };

    use super::*;

    fn tx(seed: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[seed; 32]))
    }

    fn receipt(tx_hash: TxHash) -> StoredReceipt {
        StoredReceipt {
            tx_hash,
            consensus: Arc::new(ConsensusReceipt::Succeeded {
                receipt_hash: GlobalReceiptHash::ZERO,
                #[allow(clippy::default_trait_access)]
                writes: Default::default(),
                beacon_witness_events: Vec::new(),
                events: Vec::new(),
            }),
            metadata: None,
        }
    }

    fn shard(index: u64) -> ShardId {
        ShardId::leaf(1, index)
    }

    /// A tick at `shard(0)` running one single-shard member and one leg
    /// bound to `shard(1)`, both executed and both holding a receipt.
    fn mixed_tick() -> (TickState, TxHash, TxHash) {
        let local = shard(0);
        let (determined, leg) = (tx(1), tx(2));
        let mut tick = TickState::new(
            TickId::new(local, BlockHeight::new(1)),
            BlockHash::ZERO,
            WeightedTimestamp::from_millis(1_000),
        );
        tick.admit(determined, BTreeSet::from([local]), 10, false);
        tick.admit(leg, BTreeSet::from([local, shard(1)]), 20, false);
        for tx_hash in [determined, leg] {
            tick.record_execution_result(
                tx_hash,
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                },
            );
            tick.record_receipt(receipt(tx_hash));
        }
        let (_, root, outcomes) = tick.build_vote_data().expect("every member came back");
        tick.add_execution_certificate(Arc::new(Verified::new_unchecked_for_test(
            ExecutionCertificate::new(
                *tick.tick_id(),
                tick.vote_anchor_ts(),
                root,
                outcomes,
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            ),
        )));
        (tick, determined, leg)
    }

    /// The milestone's whole point: a leg whose counterpart never
    /// certifies does not hold its tick-mates. The determined half
    /// settles on the tick's own certificate, names only its own member,
    /// and releases only that member's reservation.
    #[test]
    fn a_stranded_leg_does_not_hold_its_tick_mates() {
        let (mut tick, determined, leg) = mixed_tick();

        assert!(tick.determined_ready());
        assert!(!tick.legs_ready(), "the counterpart has not certified");

        let half = tick
            .take_determined_finalization()
            .expect("settles on the tick's own certificate");
        assert_eq!(
            half.tx_hashes().collect::<Vec<_>>(),
            vec![determined],
            "the half names its own members and no others",
        );
        assert_eq!(
            half.declared_work(),
            10,
            "and releases only what those members reserved",
        );
        assert!(!tick.has_spoken(), "the leg is still owed a verdict");

        // The counterpart still never certifies.
        assert!(!tick.legs_ready());
        assert!(tick.take_legs_finalization().is_none());
        assert!(!tick.determined_ready(), "the determined half is one-shot");
        let _ = leg;
    }

    /// Once the counterpart certifies, the legs half follows and the tick
    /// has said everything it will say.
    #[test]
    fn the_legs_half_follows_its_counterpart() {
        let (mut tick, _determined, leg) = mixed_tick();
        tick.take_determined_finalization().expect("ready");

        tick.add_execution_certificate(Arc::new(Verified::new_unchecked_for_test(
            ExecutionCertificate::new(
                TickId::new(shard(1), BlockHeight::new(4)),
                WeightedTimestamp::from_millis(1_000),
                GlobalReceiptRoot::ZERO,
                vec![TxOutcome::new(
                    leg,
                    ExecutionOutcome::Succeeded {
                        receipt_hash: GlobalReceiptHash::ZERO,
                    },
                )],
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            ),
        )));

        assert!(tick.legs_ready());
        let half = tick.take_legs_finalization().expect("covered");
        assert_eq!(half.tx_hashes().collect::<Vec<_>>(), vec![leg]);
        assert_eq!(half.declared_work(), 20);
        assert!(tick.has_spoken());
    }

    /// A tick with nothing but single-shard members has no legs half at
    /// all, so it is done the moment its own certificate lands.
    #[test]
    fn a_purely_local_tick_has_one_half() {
        let local = shard(0);
        let mut tick = TickState::new(
            TickId::new(local, BlockHeight::new(1)),
            BlockHash::ZERO,
            WeightedTimestamp::from_millis(1_000),
        );
        tick.admit(tx(1), BTreeSet::from([local]), 10, false);
        tick.record_execution_result(
            tx(1),
            ExecutionOutcome::Succeeded {
                receipt_hash: GlobalReceiptHash::ZERO,
            },
        );
        tick.record_receipt(receipt(tx(1)));
        let (_, root, outcomes) = tick.build_vote_data().expect("came back");
        tick.add_execution_certificate(Arc::new(Verified::new_unchecked_for_test(
            ExecutionCertificate::new(
                *tick.tick_id(),
                tick.vote_anchor_ts(),
                root,
                outcomes,
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            ),
        )));

        assert!(tick.take_determined_finalization().is_some());
        assert!(tick.leg_members().is_empty());
        assert!(tick.has_spoken());
    }
}
