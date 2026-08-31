//! In-flight proposal correlation and the variant taxonomy of proposal
//! kinds.
//!
//! Proposal building is asynchronous: the coordinator emits a
//! `BuildProposal` action, the runner executes it on a worker thread, and
//! the result comes back as a `ProposalBuilt` event. Between those two
//! moments, another `try_propose` could fire (e.g. a repeated
//! proposal-retry latch) or the round could advance. [`ProposalTracker`]
//! holds the `(height, round)` of the in-flight build so the
//! `ProposalBuilt` handler can detect stale callbacks and the `try_propose`
//! path can back off while a build is already pending.
//!
//! [`ProposalKind`] names the three shapes of proposal the coordinator can
//! emit — full content, empty fallback, empty sync — so a single
//! build-and-dispatch helper can drive them uniformly.

use std::collections::HashSet;
use std::sync::Arc;

use hyperscale_core::{Action, FeeDemand};
use hyperscale_types::{
    BeaconWitnessLeafCount, BlockHash, BlockHeight, Epoch, Finalization, Hash, LocalTimestamp,
    ProposerTimestamp, ProvisionHash, Provisions, ReadySignal, ReshapeTrigger, RevealChain, Round,
    ShardId, TerminalVerdict, TopologySchedule, TopologySnapshot, Transaction, TxHash, ValidatorId,
    Verifiable, Verified, WeightedTimestamp, sweep_admits_block,
};
use tracing::debug;

use crate::chain_view::ChainView;
use crate::commit_dedup::CommitDedupIndex;
use crate::precut::Precut;
use crate::verification::VerificationPipeline;

/// Variant-specific content for a proposal build.
///
/// The coordinator uses this to pass proposal-kind-specific inputs
/// (payload, timestamp source, `is_fallback` flag, logging label) to its
/// unified build-and-dispatch helper.
#[derive(Debug)]
pub enum ProposalKind {
    /// Normal proposal with a filtered payload and a real-clock timestamp.
    Normal {
        transactions: Vec<Arc<Verified<Transaction>>>,
        finalizations: Vec<Arc<Verifiable<Finalization>>>,
        provisions: Vec<Arc<Verifiable<Provisions>>>,
        terminal_verdicts: Vec<TerminalVerdict>,
    },
    /// View-change fallback: empty payload, parent's weighted timestamp
    /// (prevents Byzantine proposers from manipulating consensus time on
    /// timeout), `is_fallback = true`.
    Fallback,
    /// Syncing proposer: empty payload, normal timestamp. Proposer is
    /// online with an accurate clock but can't execute transactions.
    Sync,
}

#[derive(Debug, Clone)]
pub struct PendingProposal {
    pub height: BlockHeight,
    pub round: Round,
}

pub struct ProposalTracker {
    pending: Option<PendingProposal>,
    /// Slot for a proposal that `dispatch_or_defer` could not dispatch
    /// because the parent JMT tree wasn't available yet. Consulted from
    /// `can_propose` so repeated proposal-retry / QC-formed events for
    /// the same `(height, round)` don't spin through `assemble_build_action`
    /// while we're blocked waiting on `VerificationPipeline` to unblock.
    deferred: Option<PendingProposal>,
}

/// Result of correlating a `ProposalBuilt` callback against the tracker.
#[derive(Debug)]
pub enum TakeResult {
    /// The callback matches the in-flight build; the slot has been cleared.
    Matched,
    /// No build was in flight when the callback arrived.
    NotPending,
    /// A build was in flight but for different coordinates. The slot is
    /// preserved so the matching callback can still be handled later.
    Mismatch { expected: PendingProposal },
}

impl ProposalTracker {
    pub const fn new() -> Self {
        Self {
            pending: None,
            deferred: None,
        }
    }

    /// Record a new in-flight build. A successful dispatch also invalidates
    /// any deferred slot for a prior attempt.
    pub const fn start(&mut self, height: BlockHeight, round: Round) {
        self.pending = Some(PendingProposal { height, round });
        self.deferred = None;
    }

    /// Read the in-flight build, if any.
    pub const fn pending(&self) -> Option<&PendingProposal> {
        self.pending.as_ref()
    }

    /// Drop both the in-flight and deferred slots. Called on round advance
    /// so a stale build completing later is discarded by the next
    /// `take_matching` and the deferred slot doesn't gate the new round's
    /// `(height, round)` target.
    pub const fn clear(&mut self) {
        self.pending = None;
        self.deferred = None;
    }

    /// Record that a build for `(height, round)` could not dispatch because
    /// the parent JMT tree wasn't available. Consulted by `can_propose` to
    /// suppress re-entry until `clear_deferred` fires.
    pub const fn mark_deferred(&mut self, height: BlockHeight, round: Round) {
        self.deferred = Some(PendingProposal { height, round });
    }

    /// Read the deferred slot, if any.
    pub const fn deferred(&self) -> Option<&PendingProposal> {
        self.deferred.as_ref()
    }

    /// Drop the deferred slot. Called when the verification pipeline signals
    /// that the awaited parent tree has landed, so the next `try_propose`
    /// actually re-dispatches.
    pub const fn clear_deferred(&mut self) {
        self.deferred = None;
    }

    /// Consume the in-flight build iff its `(height, round)` matches.
    pub fn take_matching(&mut self, height: BlockHeight, round: Round) -> TakeResult {
        match self.pending.take() {
            None => TakeResult::NotPending,
            Some(p) if p.height == height && p.round == round => TakeResult::Matched,
            Some(p) => {
                let expected = p.clone();
                self.pending = Some(p);
                TakeResult::Mismatch { expected }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Payload selection
// ═══════════════════════════════════════════════════════════════════════════

/// Filter ready transactions for proposal inclusion. Drops, in this order:
///
/// 1. Txs already in the QC chain (ancestors in the two-chain window) or the
///    retention-backed committed-tx cache (historically-committed hashes that
///    survive past mempool eviction — critical after sync).
/// 2. Txs whose `validity_range` is malformed against `validity_anchor`, or
///    whose half-open range does not contain `validity_anchor`.
/// 3. Txs whose window opened before `chain_origin_wt` — the
///    predecessor's, on a reshape successor — unless `precut` holds an
///    absence proof for it against every predecessor's committed set.
///    This is the same question voters ask during block verification, so
///    filtering here keeps a proposer from offering what its own voters
///    would defer on or refuse.
/// 4. Txs naming a package this window cannot run — the same question
///    `validate_packages_usable` asks, for the same reason.
/// 5. Txs that would carry the block past the cap on sweepable cells
///    one block may create — the question `validate_sweepable_creation`
///    asks. A transaction that does not fit is skipped rather than
///    ending the selection, so a large composition never starves the
///    small ones behind it.
///
/// Logs the dedup and expiry counts when non-zero.
pub fn select_transactions(
    ready_txs: &[Arc<Verified<Transaction>>],
    qc_chain_tx_hashes: &HashSet<TxHash>,
    dedup_index: &CommitDedupIndex,
    validity_anchor: WeightedTimestamp,
    chain_origin_wt: WeightedTimestamp,
    precut: &Precut,
    topology_snapshot: &TopologySnapshot,
) -> Vec<Arc<Verified<Transaction>>> {
    let before = ready_txs.len();
    let mut deduped = 0;
    let mut expired = 0;
    let mut predates = 0;
    let mut unrunnable = 0;
    let mut oversweeping = 0;
    let mut sweepable = 0usize;
    let filtered: Vec<_> = ready_txs
        .iter()
        .filter(|tx| {
            let h = tx.hash();
            if qc_chain_tx_hashes.contains(&h) || dedup_index.contains_tx(&h) {
                deduped += 1;
                return false;
            }
            if !tx.validity_range().is_well_formed(validity_anchor)
                || !tx.validity_range().contains(validity_anchor)
            {
                expired += 1;
                return false;
            }
            // Opened before this chain did, so it belongs to the
            // predecessor that ran before the cut. Offerable only where
            // every predecessor proved it absent from its committed set;
            // anything else a voter defers on or refuses. Zero for a
            // chain born at network genesis.
            if tx.validity_range().start_timestamp_inclusive < chain_origin_wt
                && !precut.admissible(&h)
            {
                predates += 1;
                return false;
            }
            // Not published, or not for long enough that every voter is
            // sure to hold its code yet.
            if topology_snapshot.unusable_package_of(tx).is_some() {
                unrunnable += 1;
                return false;
            }
            let with_this = sweepable.saturating_add(tx.sweepable_writes() as usize);
            if !sweep_admits_block(with_this) {
                oversweeping += 1;
                return false;
            }
            sweepable = with_this;
            true
        })
        .cloned()
        .collect();
    if deduped > 0 || expired > 0 || predates > 0 || unrunnable > 0 || oversweeping > 0 {
        debug!(
            deduped,
            expired,
            predates,
            unrunnable,
            oversweeping,
            before,
            after = filtered.len(),
            "Filtered proposal candidates"
        );
    }
    filtered
}

/// Select finalizations for inclusion: drop those whose tick or whose
/// transactions the QC chain or the retention window has already
/// resolved, and cap the total finalized-tx count at the
/// `max_finalized_txs` limit. Returns `(ticks, total_tx_count)`.
///
/// The per-transaction half mirrors `validate_no_duplicate_resolutions`,
/// so a proposer never offers a second verdict its own voters refuse —
/// which is reachable without any misbehaviour: a settlement and an
/// abandonment for one transaction are different ticks, and only the
/// transaction they name says they are the same verdict twice.
///
/// Order is the caller's and is preserved. It arrives in the order the
/// ticks executed, which is the order their receipts have to settle in —
/// two ticks writing one cell each carry an absolute computed from their
/// own baseline, and settlement is last writer per cell, so the later
/// execution must land last. Re-sorting here by kickoff height would
/// invert exactly the pairs that matter: a tick held back from its own
/// block's tick executes after a later-numbered one it shares a cell
/// with. Order stays deterministic because the caller's is, which is what
/// verifiers flattening receipts into JMT `work_items` in manifest order
/// need.
///
/// Truncation is a suffix for the same reason: dropping the tail cannot
/// leave a tick ahead of a predecessor it should follow.
pub fn select_finalizations(
    finalizations: Vec<Arc<Verifiable<Finalization>>>,
    qc_chain_resolved_txs: &HashSet<TxHash>,
    dedup_index: &CommitDedupIndex,
    parent_settled_frontier: BlockHeight,
    max_finalized_txs: usize,
    chain_origin_wt: WeightedTimestamp,
) -> (Vec<Arc<Verifiable<Finalization>>>, usize) {
    let mut finalized_tx_count = 0usize;
    let mut resolved_here: HashSet<TxHash> = HashSet::new();
    let mut frontier = parent_settled_frontier;
    let ticks_to_propose: Vec<_> = finalizations
        .into_iter()
        .filter(|fw| {
            // Anchored before this chain began, so it resolves
            // transactions this chain never committed and every voter
            // refuses it. Zero for a chain born at network genesis.
            if fw.local_ec().vote_anchor_ts() < chain_origin_wt {
                return false;
            }
            // The settlement frontier, proposer-side: a determined half
            // at or below it would be refused by every voter, and one
            // offered out of order would settle an older absolute over a
            // newer one. The store hands them over in tick order, so this
            // drops only what a gap in that order would have made
            // unofferable anyway.
            let fw_ref = fw.as_unverified();
            if fw_ref.is_determined() {
                let tick = fw_ref.tick_id().block_height();
                if tick <= frontier {
                    return false;
                }
                frontier = tick;
            }
            true
        })
        .filter(|fw| {
            let unresolved = fw.tx_hashes().all(|tx_hash| {
                !resolved_here.contains(&tx_hash)
                    && !qc_chain_resolved_txs.contains(&tx_hash)
                    && !dedup_index.contains_resolved_tx(&tx_hash)
            });
            if unresolved {
                resolved_here.extend(fw.tx_hashes());
            }
            unresolved
        })
        .take_while(|fw| {
            let new_total = finalized_tx_count.saturating_add(fw.tx_count());
            if new_total <= max_finalized_txs {
                finalized_tx_count = new_total;
                true
            } else {
                false
            }
        })
        .collect();
    (ticks_to_propose, finalized_tx_count)
}

/// Drop boundary records whose evidence has stopped answering at the
/// clock the vote will read.
///
/// A record claims a departed shard left transactions unsettled, and the
/// vote checks that against the shard's settled set — which stops being
/// readable at its terminal-evidence expiry, past which the fence refuses
/// the claim outright. The composing side holds the set against the
/// *committed* frontier while the vote reads the block's own `anchor_wt`,
/// which runs ahead of it, so the two can disagree by up to the pipeline's
/// depth: without this the proposer offers a record every voter refuses,
/// and because a chain that commits nothing never advances the frontier
/// that would retire the set, the next proposal carries it again.
pub fn select_terminal_verdicts(
    verdicts: Vec<TerminalVerdict>,
    topology_schedule: &TopologySchedule,
    anchor_wt: WeightedTimestamp,
) -> Vec<TerminalVerdict> {
    verdicts
        .into_iter()
        .filter(|verdict| topology_schedule.terminal_evidence_readable(verdict.shard(), anchor_wt))
        .collect()
}

/// Drop cross-shard transactions whose payer bundle is neither among
/// the block's selected provisions nor committed within the retention
/// window — the proposer-side form of `validate_engagement`, applied
/// after provision selection so a capped-out bundle can never strand its
/// transaction in a self-rejecting proposal.
pub fn filter_engaged_transactions(
    topology_snapshot: &TopologySnapshot,
    local_shard: ShardId,
    transactions: Vec<Arc<Verified<Transaction>>>,
    provisions: &[Arc<Verifiable<Provisions>>],
    dedup_index: &CommitDedupIndex,
) -> Vec<Arc<Verified<Transaction>>> {
    transactions
        .into_iter()
        .filter(|tx| {
            if topology_snapshot.is_single_shard_transaction(tx.as_ref()) {
                return true;
            }
            let payer_shard = topology_snapshot
                .shard_trie()
                .shard_for_prefix(tx.body().fee_payer);
            if payer_shard == local_shard {
                return true;
            }
            let tx_hash = tx.hash();
            dedup_index.contains_provision_tx(payer_shard, tx_hash)
                || provisions.iter().any(|batch| {
                    batch.source_shard() == payer_shard
                        && batch
                            .transactions()
                            .iter()
                            .any(|entry| entry.tx_hash == tx_hash)
                })
        })
        .collect()
}

/// Select provisions for inclusion: drop those already in the QC
/// chain or committed within the retention window, then take from the FIFO
/// queue until the running tx-count total would exceed `max_provision_txs`.
/// Oldest batches go first so the queue drains monotonically; unselected
/// batches remain queued for the next proposal.
pub fn select_provisions(
    provisions: Vec<Arc<Verifiable<Provisions>>>,
    qc_chain_provision_hashes: &HashSet<ProvisionHash>,
    dedup_index: &CommitDedupIndex,
    max_provision_txs: usize,
) -> Vec<Arc<Verifiable<Provisions>>> {
    let mut running_tx_count = 0usize;
    provisions
        .into_iter()
        .filter(|b| {
            let h = b.hash();
            !qc_chain_provision_hashes.contains(&h) && !dedup_index.contains_provision(&h)
        })
        .take_while(|b| {
            let new_total = running_tx_count.saturating_add(b.transactions().len());
            if new_total <= max_provision_txs {
                running_tx_count = new_total;
                true
            } else {
                false
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Build + dispatch
// ═══════════════════════════════════════════════════════════════════════════

/// Outcome of assembling a proposal build action.
///
/// `assemble_build_action` returns this so the coordinator can decide what
/// mutations to apply (e.g. recording leader activity, starting the
/// tracker, deferring if the parent tree isn't ready).
pub struct BuildActionPlan {
    /// The `BuildProposal` action ready for dispatch.
    pub action: Action,
    /// Parent hash, forwarded to the tracker / verification pipeline.
    pub parent_block_hash: BlockHash,
    /// Parent block height, same rationale.
    pub parent_block_height: BlockHeight,
    /// Whether to record leader activity: `Fallback` / `Sync` count as
    /// proposer progress; `Normal` does not (it isn't progress until the
    /// QC forms).
    pub record_leader_activity: bool,
    /// Logging label for the "proposal built" info event.
    pub log_label: &'static str,
}

/// Assemble a `BuildProposal` action for the given `ProposalKind`.
///
/// Pure with respect to the coordinator — reads only from the chain view
/// and the supplied inputs, writes nothing. The caller applies the returned
/// mutations (leader activity, tracker.start, dispatch-or-defer).
///
/// `ready_signals` and the `beacon_witness_*` trio are pre-derived by
/// the coordinator (which owns the `BeaconWitnessAccumulator`, the
/// `ReadySignalPool`, and the window-resolved schedule entry) and
/// threaded through the action so `build_proposal` doesn't need to know
/// about the accumulator type.
#[allow(clippy::too_many_arguments)] // assemble fans a coordinator-side bundle into the action
pub fn assemble_build_action(
    me: ValidatorId,
    local_shard: ShardId,
    chain: &ChainView,
    height: BlockHeight,
    round: Round,
    now: LocalTimestamp,
    kind: ProposalKind,
    ready_signals: Vec<ReadySignal>,
    reshape_trigger: Option<ReshapeTrigger>,
    parent_witness_leaves: Vec<Hash>,
    beacon_witness_base: BeaconWitnessLeafCount,
    parent_reveal_chain: RevealChain,
    parent_committee_anchor_epoch: Epoch,
    committee_anchor_epoch: Epoch,
    carry_split_child_roots: bool,
    carry_terminal_roots: bool,
    settled_txs_window_floor: Option<WeightedTimestamp>,
    classification_topology_snapshot: Arc<TopologySnapshot>,
    fee_checks: Vec<FeeDemand>,
    fee_read_height: BlockHeight,
    substate_bytes: Option<u64>,
) -> BuildActionPlan {
    let (parent_block_hash, parent_qc) = chain.proposal_parent();
    let parent_block_height = parent_qc.height();
    let parent_state_root = chain.parent_state_root(parent_block_hash);
    let parent_in_flight = chain.parent_in_flight(parent_block_hash);
    let parent_settled_frontier = chain.parent_settled_frontier(parent_block_hash);
    let parent_sweep_frontier = chain.parent_sweep_frontier(parent_block_hash);
    let parent_load = chain.parent_load_checked(parent_block_hash);

    let (
        timestamp,
        is_fallback,
        transactions,
        finalizations,
        provisions,
        terminal_verdicts,
        log_label,
        record_leader_activity,
    ) = match kind {
        ProposalKind::Normal {
            transactions,
            finalizations,
            provisions,
            terminal_verdicts,
        } => (
            ProposerTimestamp::from_local(now),
            false,
            transactions,
            finalizations,
            provisions,
            terminal_verdicts,
            "Requesting block build for proposal",
            false,
        ),
        ProposalKind::Fallback => (
            ProposerTimestamp::from_millis(parent_qc.weighted_timestamp().as_millis()),
            true,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "Building fallback block (leader timeout)",
            true,
        ),
        ProposalKind::Sync => (
            ProposerTimestamp::from_local(now),
            false,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "Building sync block (syncing, empty payload)",
            true,
        ),
    };

    // The proposer's new BlockHeader will carry parent_qc in its wire
    // form; HBOR encoding is byte-identical between the raw and
    // verified shapes, but the field type is raw.
    let parent_qc_raw = parent_qc.into_inner();
    let action = Action::BuildProposal {
        shard_id: local_shard,
        proposer: me,
        height,
        round,
        parent_block_hash,
        parent_qc: parent_qc_raw,
        timestamp,
        is_fallback,
        parent_state_root,
        parent_block_height,
        transactions,
        finalizations,
        provisions,
        terminal_verdicts,
        fee_checks,
        fee_read_height,
        parent_in_flight,
        parent_settled_frontier,
        parent_sweep_frontier,
        parent_load,
        substate_bytes,
        ready_signals,
        reshape_trigger,
        parent_witness_leaves,
        beacon_witness_base,
        parent_reveal_chain,
        parent_committee_anchor_epoch,
        committee_anchor_epoch,
        carry_split_child_roots,
        carry_terminal_roots,
        settled_txs_window_floor,
        classification_topology_snapshot,
    };

    BuildActionPlan {
        action,
        parent_block_hash,
        parent_block_height,
        record_leader_activity,
        log_label,
    }
}

/// Dispatch a built proposal action, deferring instead if the parent's JMT
/// tree isn't available yet.
///
/// Even empty blocks need the parent root node: `noop_jmt_snapshot` copies
/// it to the new version so the overlay chain stays intact. Without that, a
/// child block's `VerifyStateRoot` hits `ParentVersionMissing`.
///
/// The one exception is `bridge_over_attested_parent` — the build-side
/// mirror of the verifier's recovery-bridge escape (see
/// `initiate_state_root_verification`): a tick-less block in the bridge
/// band whose parent is a sync-admitted, QC-attested certified block
/// dispatches without the parent tree. Its prepare applies no updates and
/// inherits the attested parent root; the commit pipeline releases store
/// persists strictly height contiguous, so by the time this block
/// persists its parent's tree is durable and the no-op root carry
/// completes. Without the escape a fully redrawn recovery committee
/// deadlocks: no member ever verified the halted tip through the live
/// pipeline, the tip's tree materializes only at commit, and that commit
/// needs the successor QC this very build produces.
///
/// When deferred, the verification pipeline unblocks and re-enters
/// `try_propose` via the proposal-retry latch when the parent tree lands.
pub fn dispatch_or_defer(
    tracker: &mut ProposalTracker,
    verification: &mut VerificationPipeline,
    plan: BuildActionPlan,
    block_height: BlockHeight,
    round: Round,
    bridge_over_attested_parent: bool,
) -> Vec<Action> {
    if bridge_over_attested_parent
        || verification.parent_tree_available(plan.parent_block_height, plan.parent_block_hash)
    {
        tracker.start(block_height, round);
        vec![plan.action]
    } else {
        verification.defer_proposal(plan.parent_block_hash, plan.parent_block_height);
        tracker.mark_deferred(block_height, round);
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use hyperscale_types::test_utils::{
        install_stub_protocol_statics, make_finalization, stub_abort_charge, stub_transaction,
        stub_transaction_binding, test_prefix, test_principal, test_transaction_running,
    };
    use hyperscale_types::{
        CommittedTxsRoot, Hash, MAX_FINALIZED_TX_PER_BLOCK, MAX_SUBINTENTS,
        MAX_SWEEPABLE_CREATED_PER_BLOCK, MAX_VALIDITY_RANGE, NetworkDefinition,
        PredecessorTerminal, TimestampRange, TransactionDecision, UnsettledTx, ValidatorSet,
    };

    use super::*;

    /// A boundary record is offered only while the vote can still accept
    /// it. The vote reads the block's own anchor, which runs ahead of the
    /// committed frontier the composing side holds its evidence against,
    /// so the anchor is what decides here too — against the same
    /// handoff-anchored evidence window the fence itself derives.
    #[test]
    fn select_terminal_verdicts_stops_at_the_evidence_expiry() {
        use std::collections::{BTreeMap, BTreeSet, HashMap};

        use hyperscale_types::{
            NetworkDefinition, ShardAnchor, StateRoot, TopologySchedule, ValidatorSet,
        };

        let departed = ShardId::leaf(1, 0);
        let survivor = ShardId::leaf(1, 1);
        let schedule = |handoff_complete: Option<Epoch>| {
            let mut boundaries = HashMap::new();
            boundaries.insert(
                departed,
                ShardAnchor {
                    state_root: StateRoot::ZERO,
                    block_hash: BlockHash::from_raw(Hash::from_bytes(b"terminal")),
                    height: BlockHeight::new(9),
                    weighted_timestamp: WeightedTimestamp::from_millis(2_000),
                    witness_base: BeaconWitnessLeafCount::ZERO,
                    terminal_roots: None,
                    handoff_complete,
                },
            );
            let snapshot = Arc::new(TopologySnapshot::from_explicit_committees(
                NetworkDefinition::simulator(),
                &ValidatorSet::new(Vec::new()),
                std::iter::once((survivor, Vec::new())).collect(),
                HashMap::new(),
                boundaries,
                HashMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeSet::new(),
            ));
            let mut sched = TopologySchedule::new(1_000, Epoch::new(0), Arc::clone(&snapshot));
            for epoch in 1..=20u64 {
                sched.insert(Epoch::new(epoch), Arc::clone(&snapshot));
            }
            sched.set_head(snapshot);
            sched
        };

        let record = TerminalVerdict::new(
            departed,
            WeightedTimestamp::from_millis(2_000),
            [UnsettledTx {
                tx_hash: TxHash::from(Hash::from_bytes(b"stranded")),
                deadline: WeightedTimestamp::from_millis(5_000),
                declared_work: 3,
                charge: stub_abort_charge(3),
            }],
        );
        let offered = |sched: &TopologySchedule, anchor: WeightedTimestamp| {
            select_terminal_verdicts(vec![record.clone()], sched, anchor).len()
        };

        let handoff = Epoch::new(4);
        let stamped = schedule(Some(handoff));
        let expiry = stamped.windows().handoff_evidence_expiry(handoff);
        assert_eq!(offered(&stamped, expiry), 1, "to the last instant of it");
        assert_eq!(
            offered(&stamped, expiry.plus(Duration::from_millis(1))),
            0,
            "past it every voter refuses the claim, so it must not be offered",
        );

        let open = schedule(None);
        assert_eq!(
            offered(&open, expiry.plus(Duration::from_millis(1))),
            1,
            "an unstamped handoff holds the window open however late the anchor",
        );
    }

    /// A proposer offers one verdict per transaction. A settlement and an
    /// abandonment for the same transaction are different ticks, so
    /// nothing about their identity separates them — the second is
    /// dropped because of the transaction it names, which is the rule the
    /// voters apply to the same list.
    #[test]
    fn select_finalizations_offers_one_verdict_per_transaction() {
        let tx_hash = TxHash::from(Hash::from_bytes(b"contested"));
        let settled: Arc<Verifiable<Finalization>> = Arc::new(
            make_finalization(BlockHeight::new(1), tx_hash, TransactionDecision::Accept).into(),
        );
        let abandoned: Arc<Verifiable<Finalization>> = Arc::new(
            make_finalization(BlockHeight::new(9), tx_hash, TransactionDecision::Aborted).into(),
        );
        assert_ne!(settled.tick_id(), abandoned.tick_id());

        let (selected, count) = select_finalizations(
            vec![Arc::clone(&settled), abandoned],
            &HashSet::new(),
            &CommitDedupIndex::new(),
            BlockHeight::GENESIS,
            MAX_FINALIZED_TX_PER_BLOCK,
            WeightedTimestamp::ZERO,
        );
        assert_eq!(selected.len(), 1, "the second verdict is dropped");
        assert_eq!(selected[0].tick_id(), settled.tick_id());
        assert_eq!(count, 1);
    }

    /// A transaction a committed block already resolved keeps its
    /// finalization out of the next proposal, whichever verdict each
    /// carried.
    #[test]
    fn select_finalizations_drops_what_the_retention_window_resolved() {
        let tx_hash = TxHash::from(Hash::from_bytes(b"already resolved"));
        let committed: Arc<Verifiable<Finalization>> = Arc::new(
            make_finalization(BlockHeight::new(1), tx_hash, TransactionDecision::Accept).into(),
        );
        let mut dedup_index = CommitDedupIndex::new();
        dedup_index.register_committed_certs(std::slice::from_ref(&committed));

        let abandoned: Arc<Verifiable<Finalization>> = Arc::new(
            make_finalization(BlockHeight::new(9), tx_hash, TransactionDecision::Aborted).into(),
        );
        let (selected, count) = select_finalizations(
            vec![abandoned],
            &HashSet::new(),
            &dedup_index,
            BlockHeight::GENESIS,
            MAX_FINALIZED_TX_PER_BLOCK,
            WeightedTimestamp::ZERO,
        );
        assert!(selected.is_empty());
        assert_eq!(count, 0);
    }

    #[test]
    fn start_records_pending_slot() {
        let mut tracker = ProposalTracker::new();
        assert!(tracker.pending().is_none());

        tracker.start(BlockHeight::new(5), Round::new(1));
        let p = tracker.pending().unwrap();
        assert_eq!(p.height, BlockHeight::new(5));
        assert_eq!(p.round, Round::new(1));
    }

    #[test]
    fn take_matching_clears_on_match() {
        let mut tracker = ProposalTracker::new();
        tracker.start(BlockHeight::new(5), Round::new(1));

        let result = tracker.take_matching(BlockHeight::new(5), Round::new(1));
        assert!(matches!(result, TakeResult::Matched));
        assert!(tracker.pending().is_none());
    }

    #[test]
    fn take_matching_preserves_slot_on_mismatch() {
        let mut tracker = ProposalTracker::new();
        tracker.start(BlockHeight::new(5), Round::new(1));

        let result = tracker.take_matching(BlockHeight::new(5), Round::new(2));
        match result {
            TakeResult::Mismatch { expected } => {
                assert_eq!(expected.height, BlockHeight::new(5));
                assert_eq!(expected.round, Round::new(1));
            }
            other => panic!("expected Mismatch, got {other:?}"),
        }
        assert!(
            tracker.pending().is_some(),
            "slot must be preserved on mismatch"
        );
    }

    #[test]
    fn take_matching_returns_not_pending_when_empty() {
        let mut tracker = ProposalTracker::new();
        assert!(matches!(
            tracker.take_matching(BlockHeight::new(5), Round::new(1)),
            TakeResult::NotPending
        ));
    }

    #[test]
    fn mark_deferred_records_slot_without_touching_pending() {
        let mut tracker = ProposalTracker::new();
        tracker.mark_deferred(BlockHeight::new(5), Round::new(1));

        let d = tracker.deferred().unwrap();
        assert_eq!(d.height, BlockHeight::new(5));
        assert_eq!(d.round, Round::new(1));
        assert!(tracker.pending().is_none());
    }

    #[test]
    fn start_clears_deferred() {
        let mut tracker = ProposalTracker::new();
        tracker.mark_deferred(BlockHeight::new(5), Round::new(1));
        tracker.start(BlockHeight::new(5), Round::new(1));

        assert!(tracker.deferred().is_none());
        assert!(tracker.pending().is_some());
    }

    #[test]
    fn clear_drops_both_slots() {
        let mut tracker = ProposalTracker::new();
        tracker.start(BlockHeight::new(5), Round::new(1));
        tracker.mark_deferred(BlockHeight::new(6), Round::new(2));
        tracker.clear();

        assert!(tracker.pending().is_none());
        assert!(tracker.deferred().is_none());
    }

    #[test]
    fn clear_deferred_leaves_pending_intact() {
        let mut tracker = ProposalTracker::new();
        tracker.start(BlockHeight::new(5), Round::new(1));
        tracker.mark_deferred(BlockHeight::new(6), Round::new(2));
        tracker.clear_deferred();

        assert!(tracker.deferred().is_none());
        assert!(tracker.pending().is_some());
    }

    // ─── select_transactions: validity-window filter ───────────────────

    fn ts(ms: u64) -> WeightedTimestamp {
        WeightedTimestamp::from_millis(ms)
    }

    fn tx_with_range(seed: u8, range: TimestampRange) -> Arc<Verified<Transaction>> {
        install_stub_protocol_statics();
        Arc::new(Verified::<Transaction>::from_persisted(stub_transaction(
            test_principal(seed),
            &[test_prefix(seed)],
            1_000,
            range,
        )))
    }

    fn empty_dedup_index() -> CommitDedupIndex {
        CommitDedupIndex::new()
    }

    /// A window whose registry lists nothing.
    ///
    /// It holds back everything that names a package and nothing that
    /// does not, which is what every case here but the package ones
    /// wants: their transactions run no code.
    fn window_listing_no_packages() -> TopologySnapshot {
        TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            ValidatorSet::new(Vec::new()),
        )
    }

    /// The same window with `packages` listed as runnable.
    fn window_listing(packages: &[Hash]) -> TopologySnapshot {
        window_listing_no_packages().with_usable_packages(packages.iter().copied().collect())
    }

    fn tx_running(seed: u8, packages: &[Hash]) -> Arc<Verified<Transaction>> {
        install_stub_protocol_statics();
        Arc::new(Verified::<Transaction>::from_persisted(
            test_transaction_running(seed, packages),
        ))
    }

    /// A successor of one chain that has answered nothing, so no pre-cut
    /// transaction is admissible. Cases anchored at
    /// `WeightedTimestamp::ZERO` never consult it, since nothing opens
    /// before an origin of zero.
    fn refuses_precut() -> Precut {
        Precut::succeeding(vec![PredecessorTerminal {
            shard: ShardId::leaf(1, 0),
            height: BlockHeight::new(9),
            block_hash: BlockHash::ZERO,
            committed_txs_root: CommittedTxsRoot::ZERO,
        }])
    }

    /// The same successor, with `tx_hash` proven absent from its
    /// predecessor's committed set.
    fn admits_precut(tx_hash: TxHash) -> Precut {
        let mut precut = refuses_precut();
        precut.record(precut.predecessors()[0].shard, tx_hash, true);
        precut
    }

    /// A transaction opening before the chain's origin is dropped while
    /// unresolved, and offered once every predecessor has proven it
    /// absent — the whole point of the committed set, seen from the
    /// proposer's side.
    #[test]
    fn select_transactions_offers_a_precut_tx_only_once_resolved_absent() {
        let cut = ts(10_000);
        let anchor = ts(10_500);
        // Opens before the cut, still valid at the anchor: exactly the
        // population the successor cannot judge on its own.
        let candidate = tx_with_range(9, TimestampRange::new(ts(9_000), ts(40_000)));
        let hash = candidate.hash();
        let txs = vec![candidate];

        let refused = select_transactions(
            &txs,
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            cut,
            &refuses_precut(),
            &window_listing_no_packages(),
        );
        assert!(
            refused.is_empty(),
            "an unresolved pre-cut transaction is not offered"
        );

        let admitted = select_transactions(
            &txs,
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            cut,
            &admits_precut(hash),
            &window_listing_no_packages(),
        );
        assert_eq!(
            admitted.len(),
            1,
            "proven absent from every predecessor, so this is its first commit"
        );
    }

    #[test]
    fn select_transactions_drops_expired_txs() {
        // Anchor in the future of the tx's range.
        let anchor = ts(100_000);
        let expired_range = TimestampRange::new(ts(0), ts(1_000));
        let valid_range = TimestampRange::new(anchor, anchor.plus(Duration::from_mins(1)));

        let txs = vec![
            tx_with_range(1, expired_range),
            tx_with_range(2, valid_range),
        ];

        let selected = select_transactions(
            &txs,
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing_no_packages(),
        );

        assert_eq!(selected.len(), 1, "only the in-range tx should survive");
        assert_eq!(selected[0].hash(), txs[1].hash());
    }

    #[test]
    fn select_transactions_drops_not_yet_valid_txs() {
        // Anchor sits before the tx's start.
        let anchor = ts(50);
        let future_range = TimestampRange::new(ts(1_000), ts(60_000));
        let txs = vec![tx_with_range(3, future_range)];

        let selected = select_transactions(
            &txs,
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing_no_packages(),
        );

        assert!(
            selected.is_empty(),
            "tx whose start is past anchor should be filtered"
        );
    }

    #[test]
    fn select_transactions_drops_malformed_ranges() {
        let anchor = ts(1_000);
        // Length over MAX_VALIDITY_RANGE.
        let too_wide = TimestampRange::new(
            ts(0),
            anchor.plus(MAX_VALIDITY_RANGE + Duration::from_secs(1)),
        );
        let txs = vec![tx_with_range(4, too_wide)];

        let selected = select_transactions(
            &txs,
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing_no_packages(),
        );

        assert!(selected.is_empty(), "malformed range should be filtered");
    }

    #[test]
    fn selection_skips_what_would_overrun_the_sweep_creation_cap() {
        install_stub_protocol_statics();
        let anchor = ts(1_000);
        let range = TimestampRange::new(ts(500), anchor.plus(Duration::from_mins(1)));
        let binding = |seed: u8, bound: usize| -> Arc<Verified<Transaction>> {
            Arc::new(Verified::<Transaction>::from_persisted(
                stub_transaction_binding(seed, bound, range),
            ))
        };
        // Fill the cap exactly, then offer one that cannot fit followed
        // by one that can. Skipping rather than stopping is what keeps a
        // large composition from starving the small ones behind it.
        let full = MAX_SWEEPABLE_CREATED_PER_BLOCK / MAX_SUBINTENTS;
        let mut txs: Vec<Arc<Verified<Transaction>>> = (0..full)
            .map(|i| binding(u8::try_from(i).expect("fewer than 256"), MAX_SUBINTENTS))
            .collect();
        let overflows = binding(u8::MAX, MAX_SUBINTENTS);
        let fits = binding(u8::MAX - 1, 0);
        txs.push(overflows.clone());
        txs.push(fits.clone());

        let selected = select_transactions(
            &txs,
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing_no_packages(),
        );

        assert_eq!(
            selected.len(),
            full + 1,
            "the one that cannot fit is skipped"
        );
        assert!(selected.iter().any(|tx| tx.hash() == fits.hash()));
        assert!(!selected.iter().any(|tx| tx.hash() == overflows.hash()));
    }

    #[test]
    fn select_transactions_drops_at_upper_bound_exclusive() {
        // Half-open: end_timestamp_exclusive == anchor must be filtered.
        let anchor = ts(1_000);
        let range = TimestampRange::new(ts(500), anchor); // [500, 1000)
        let txs = vec![tx_with_range(5, range)];

        let selected = select_transactions(
            &txs,
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing_no_packages(),
        );

        assert!(
            selected.is_empty(),
            "anchor == end_exclusive must be excluded (half-open)"
        );
    }

    #[test]
    fn select_transactions_keeps_at_lower_bound_inclusive() {
        // Half-open: start_timestamp_inclusive == anchor must be kept.
        let anchor = ts(1_000);
        let range = TimestampRange::new(anchor, anchor.plus(Duration::from_mins(1)));
        let txs = vec![tx_with_range(6, range)];

        let selected = select_transactions(
            &txs,
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing_no_packages(),
        );

        assert_eq!(selected.len(), 1, "anchor == start_inclusive must be kept");
    }

    #[test]
    fn select_transactions_dedup_short_circuits_validity_check() {
        // Tx in QC chain — should be dropped without consulting the
        // validity range. We pass an obviously-invalid range to confirm.
        let anchor = ts(100_000);
        let any_range = TimestampRange::new(ts(0), ts(1_000));
        let tx = tx_with_range(7, any_range);
        let mut chain = HashSet::new();
        chain.insert(tx.hash());

        let selected = select_transactions(
            &[tx],
            &chain,
            &empty_dedup_index(),
            anchor,
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing_no_packages(),
        );
        assert!(selected.is_empty());
    }

    /// A proposer offers only what its own voters would accept, and the
    /// package rule is one of the things they ask. A transaction naming a
    /// package the window does not list is left out; one naming a listed
    /// package goes in.
    ///
    /// Stated as the permission, so the two ways a package can fail to be
    /// runnable here — registered but still maturing, and never
    /// registered at all — are refused by the same test.
    #[test]
    fn a_tx_naming_a_package_the_window_does_not_list_is_left_out() {
        let listed = Hash::from_bytes(b"a package past its window");
        let unlisted = Hash::from_bytes(b"a package still inside one");
        let anchor = ts(1_000);

        let runnable = tx_running(1, &[listed]);
        let held = tx_running(2, &[unlisted]);
        let selected = select_transactions(
            &[Arc::clone(&runnable), held],
            &HashSet::new(),
            &empty_dedup_index(),
            anchor,
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing(&[listed]),
        );

        assert_eq!(
            selected.iter().map(|tx| tx.hash()).collect::<Vec<_>>(),
            vec![runnable.hash()],
            "only the transaction whose packages the window lists is offered"
        );
    }

    /// One unlisted package is enough, however many the transaction runs.
    #[test]
    fn one_unlisted_package_holds_back_a_tx_that_names_others_too() {
        let listed = Hash::from_bytes(b"code the window lists");
        let unlisted = Hash::from_bytes(b"code it does not");
        let selected = select_transactions(
            &[tx_running(3, &[listed, unlisted])],
            &HashSet::new(),
            &empty_dedup_index(),
            ts(1_000),
            WeightedTimestamp::ZERO,
            &refuses_precut(),
            &window_listing(&[listed]),
        );
        assert!(
            selected.is_empty(),
            "a transaction is offered only when every package it runs is listed"
        );
    }
}
