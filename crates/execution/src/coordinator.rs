//! Execution state machine.
//!
//! Drives transaction execution after blocks are committed. Transactions are
//! grouped into ticks (same provision dependency set within a block) and each
//! tick runs through its lifecycle inside a [`TickState`](crate::TickState).
//!
//! # Transaction Types
//!
//! - **Single-shard**: Dispatched immediately at block commit; local quorum
//!   votes produce an execution certificate.
//! - **Cross-shard**: Dispatched once the tick's provisions assemble, then
//!   voted and cross-shard-finalized.
//!
//! # Cross-Shard Atomic Execution Protocol
//!
//! ## Phase 1: State Provisioning
//! When a block commits with cross-shard transactions, the block proposer broadcasts
//! state provisions (with merkle inclusion proofs) to target shards. Provisions are
//! committed in blocks via `provision_root` — all validators have the same data.
//!
//! ## Phase 2: Tick-Atomic Execution
//! Once every tx in a tick is provisioned (or at block commit for single-shard
//! ticks), the whole tick dispatches atomically via `ExecuteTransactions`.
//!
//! ## Phase 3: Vote Aggregation
//! Validators send execution votes to the tick leader. When the leader collects
//! 2f+1 voting power agreeing on the same receipt hash, it aggregates an
//! execution certificate and broadcasts it to local peers and remote shards.
//!
//! ## Phase 5: Finalization
//! Validators collect shard execution proofs from all participating shards. When all
//! proofs are received, a `Finalization` is created.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::sync::Arc;

use hyperscale_core::{
    Action, CrossShardExecutionRequest, FetchAbandon, FetchRequest, ProtocolEvent, TickBatchOutcome,
};
use hyperscale_engine::legs::{Classified, Runs, Side};
use hyperscale_engine::{TickEnvironment, build_fee_receipt};
use hyperscale_metrics::{
    record_rebuilt_verdict_entry, record_reclaim_admitted, record_reclaim_probe_answered,
    record_unresolvable_tx,
};
use hyperscale_storage::{RecoveredState, TickResolution, committed_tx_cell_key};
use hyperscale_types::{
    AbandonmentRecord, Absence, Attempt, AwaitingTopologyBuffer, Block, BlockHash, BlockHeader,
    BlockHeight, BloomFilter, CertifiedBlock, ClaimProof, CounterpartClaim, CounterpartMirror,
    DeclaredKey, Derivation, ExecutionCertificate, ExecutionCertificateVerifyError,
    ExecutionOutcome, ExecutionVote, Finalization, FinalizationHash, FinalizationVerifyError,
    GlobalReceiptRoot, Hash, Inclusion, MAX_ABANDONMENT_RECORDS_PER_BLOCK,
    MAX_PROVISIONS_PER_BLOCK, MAX_UNSETTLED_PER_BLOCK, MerkleInclusionProof, Mode, Probed,
    ProvenAnchors, Provisions, Refusal, ScheduleLookup, SettledSetVerdict, SettledTxSet, ShardId,
    ShardTrie, StateAnchor, StateProofBundle, StoredReceipt, SubstateKey, TickId, TopologySchedule,
    TopologySnapshot, Transaction, TransactionDecision, TxClaim, TxHash, TxOutcome, TxResolution,
    UnsettledTx, ValidatorId, VerdictClaim, Verifiable, Verified, WeightedTimestamp,
    claim_readable_at, derive_block_transactions, lapse_probe_anchor, reclaim_probe_anchor,
    settled_set_verdict, tick_leader, tick_leader_at,
};
use hyperscale_vm_effects::CrossingCell;
use tracing::instrument;

use crate::candidates::{Admitted, TickCandidates};
use crate::early_arrivals::{EARLY_VOTE_RETENTION, EarlyArrivalBuffer};
use crate::exec_cert_store::ExecCertStore;
use crate::expected_certs::ExpectedCertTracker;
use crate::finalizations::FinalizationStore;
use crate::lookups::{
    assign_participants, build_provision_requests, committee_public_keys_for_shard,
    ec_has_shard_quorum_power, fetch_keys_covered, peers_excluding_self,
};
use crate::outbound_certs::OutboundExecutionCertificateTracker;
use crate::provisional::ProvisionalCells;
use crate::provisioning::{ProvisioningTracker, Requirement, divided_requirements};
use crate::tick_state::{Admission, Divergence, Membership, TickState};
use crate::ticks::{PendingVoteRetry, RetryEffect, TickRegistry};
use crate::unresolved::{
    Abandonable, Answer, Probe, Probeable, Reclaimable, Retirable, Unanswerable, UnresolvedTxs,
};
use crate::vote_tracker::VoteTracker;

/// One payer-side engagement wait: the transaction, the counterpart
/// shards whose echo its vote waits on, and the signed window end that
/// bounds the wait.
type EngagementWait = (TxHash, BTreeSet<ShardId>, WeightedTimestamp);

/// One transaction a committed block put in flight, as this shard sees
/// it: the shards party to it and the classification frozen at that
/// commit.
#[derive(Clone)]
struct CommittedMember {
    tx: Arc<Verifiable<Transaction>>,
    participating: BTreeSet<ShardId>,
    classified: Classified,
}

impl CommittedMember {
    /// Whether the transaction touches a shard besides this one — asked
    /// here, before a member of it exists, to pick the ones that need a
    /// cross-shard registration at all.
    fn reaches_beyond(&self, local: ShardId) -> bool {
        self.participating.iter().any(|&shard| shard != local)
    }
}

/// The committed block a tick is created against: its identity, and the
/// environment anchors the transactions it commits execute under.
#[derive(Clone, Copy, Debug)]
struct CommittingBlock {
    hash: BlockHash,
    height: BlockHeight,
    /// The block's parent-QC weighted timestamp.
    ts: WeightedTimestamp,
}

/// A tick with entries on the tick chain and no committed fate yet.
struct TickedBatch {
    /// What its abortable members declared they would mutate — the cells
    /// nothing may read until a counterpart's certificate says which
    /// side of each survives.
    ///
    /// Its other members claim nothing: their writes are determined the
    /// moment they execute and readable at once, so a later tick may
    /// fold over them freely. Empty for a tick that ran no member a
    /// counterpart can retract, which is then held only for its own
    /// fate.
    provisional_claims: Vec<(DeclaredKey, Mode)>,
    /// The legs those claims belong to. A tick's fate arrives in halves,
    /// and only the half carrying the legs releases their cells — so the
    /// entry has to know which members that is.
    legs: BTreeSet<TxHash>,
}

/// One composed-but-undispatched tick: the block's identity anchors plus
/// the members that joined at its commit.
struct PendingTick {
    tick: BlockHeight,
    tick_ts: WeightedTimestamp,
    env: TickEnvironment,
    requests: Vec<CrossShardExecutionRequest>,
}

impl PendingTick {
    /// Whether any member runs code drawn from `missing`.
    fn runs_any_of(&self, missing: &BTreeSet<Hash>) -> bool {
        self.requests.iter().any(|request| {
            request
                .transaction
                .as_ref()
                .is_some_and(|body| body.packages().iter().any(|p| missing.contains(p)))
        })
    }
}

/// Data returned when a tick is ready for voting.
///
/// The state machine produces this; the `io_loop` uses it to sign the execution vote
/// and broadcast (since the state machine doesn't hold the signing key).
#[derive(Debug)]
pub struct CompletionData {
    /// Block this tick belongs to; pairs with `tick_id` to identify the vote target.
    pub block_hash: BlockHash,
    /// Height of the tick-starting block.
    pub block_height: BlockHeight,
    /// BFT-authenticated weighted timestamp at which this tick's outcome is
    /// fixed. Included in the vote payload and the EC canonical hash, so all
    /// validators aggregate under the same identifier.
    pub vote_anchor_ts: WeightedTimestamp,
    /// Tick identifier; unique within `block_hash`.
    pub tick_id: TickId,
    /// Merkle root over per-tx outcome leaves (cross-shard agreement).
    pub global_receipt_root: GlobalReceiptRoot,
    /// Per-tx outcomes in tick order.
    pub tx_outcomes: Vec<TxOutcome>,
}

/// Execution memory statistics for monitoring collection sizes.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutionMemoryStats {
    /// Total receipts held across all in-flight ticks, awaiting finalization.
    pub tick_execution_receipts: usize,
    /// Finalizations cached in memory until their proposing block commits.
    pub finalizations: usize,
    /// In-flight tick states (created, not yet finalized or evicted).
    pub ticks: usize,
    /// Committed transactions still owed an outcome.
    pub unresolved_txs: usize,
    /// Per-tick vote trackers awaiting quorum.
    pub vote_trackers: usize,
    /// Buffered execution votes waiting for their tick to begin.
    pub early_votes: usize,
    /// Expected EC arrivals from remote shards we're awaiting.
    pub expected_exec_certs: usize,
    /// Verified provisions held per cross-shard tx.
    pub verified_provisions: usize,
    /// Distinct (tx, source-shard) requirements awaiting provisioning.
    pub required_provision_shards: usize,
    /// Distinct (tx, source-shard) provisions received so far.
    pub received_provision_shards: usize,
    /// Ticks whose local EC has been emitted.
    pub ticks_with_ec: usize,
    /// Vote retries scheduled for resend after rotation timeout.
    pub pending_vote_retries: usize,
    /// Active tx → tick assignments in the registry.
    pub tick_assignments: usize,
    /// Early tick attestations buffered before local routing.
    pub early_attestations: usize,
    /// Buffered ECs awaiting tx assignment routing.
    pub pending_routing: usize,
    /// Expected ECs that have already been fulfilled (kept for diagnostics).
    pub fulfilled_exec_certs: usize,
    /// Outbound ECs retained for re-broadcast to remote shards.
    pub outbound_certs: usize,
    /// Commit-proven remote source blocks within retention.
    pub proven_remote_blocks: usize,
    /// Cross-shard ECs deferred on their source block's commit proof. A
    /// sustained rise means a source shard certifies without proving
    /// commits — the fork/withholding signature.
    pub unproven_ecs: usize,
}

/// One counterpart cell a leg entry asks about: the shard holding it,
/// the cell, the anchor an answer is held to, and which question it is.
type CounterpartCell = (ShardId, SubstateKey, WeightedTimestamp, Probed);

/// Every cell `entry` asks a counterpart about, under `trie`: one core
/// shard's committed cell, each delivery's claim on the shard that was
/// to deliver it and on whatever shard holds the cell's prefix now, and
/// each core consumer's claim. The one enumeration the prober and the
/// commit-time fold both read, so what is asked and what is answered
/// are the same cells.
fn counterpart_cells(entry: &Probeable, trie: &ShardTrie) -> Vec<CounterpartCell> {
    let core = entry.core.iter().next().map(|&shard| {
        (
            shard,
            committed_tx_cell_key(shard, entry.tx_hash, entry.validity_end),
            entry.deadline,
            Probed::Core,
        )
    });
    let deliveries = entry.deliveries.iter().flat_map(|&(delivered_by, claim)| {
        BTreeSet::from([delivered_by, trie.shard_for_prefix(claim.owner)])
            .into_iter()
            .map(move |shard| {
                (
                    shard,
                    claim,
                    lapse_probe_anchor(entry.validity_end),
                    Probed::Delivery,
                )
            })
    });
    let claims = entry.claims.iter().flat_map(|&(shard, claim)| {
        BTreeSet::from([shard, trie.shard_for_prefix(claim.owner)])
            .into_iter()
            .map(move |shard| (shard, claim, entry.deadline, Probed::Claim))
    });
    core.into_iter().chain(deliveries).chain(claims).collect()
}

/// The name a housekeeping member over inherited records takes on the
/// chain that inherited them.
///
/// Derived from the transaction that issued the crossings and the record
/// cells being settled, so every replica at one frontier reaches the same
/// name and no two members ever share one. It is not the issuing
/// transaction's own hash: that transaction belongs to a chain that has
/// ended, this one never committed it, and a receipt naming it would be a
/// verdict this chain has no standing to reach.
fn inherited_member_name(issued_by: TxHash, records: &[SubstateKey]) -> TxHash {
    let keys: Vec<Vec<u8>> = records.iter().map(|key| key.to_bytes().to_vec()).collect();
    let mut parts: Vec<&[u8]> = vec![b"hyperscale.inherited.records", &issued_by.0.0];
    parts.extend(keys.iter().map(Vec::as_slice));
    TxHash::from(Hash::from_parts(&parts))
}

/// The earliest a member's provisioning entry may be swept, where the
/// member is a delivery.
///
/// A delivery is admissible to the delivery window's close and probed a
/// finalization delay past it, which outlives the retention horizon the
/// entry would otherwise take: a bundle landing after the sweep would
/// populate `present` against a requirement nobody records, and the
/// delivery would be abandoned at the close while the issuer reclaims a
/// crossing its deliverer could have taken. Every other member waits
/// inside the horizon and takes no floor.
fn delivery_floor(side: Side, validity_end: WeightedTimestamp) -> Option<WeightedTimestamp> {
    (side == Side::Delivering).then(|| lapse_probe_anchor(validity_end))
}

/// Whether a proposer offers `Claimed` records from the claims the
/// chain has proved.
///
/// Off, and waiting on two things that are not the offer's own. A
/// transfer sent `Departing` and included by the leaving shard reaches
/// `Completed(Aborted)` where it owes `Settled`, which is the shape the
/// train's fate map exists to refuse. And a retirement runs a member
/// whose membership decides nothing, which is never cleared from what a
/// proposer offers — clearing keys off resolution, and a member that
/// resolves no transaction gives it nothing to key on. Measured on a
/// shard composing them: the finalization is re-offered every block, the
/// settlement frontier never advances, and block production falls to the
/// view-change cadence.
///
/// The retirement machinery itself is live and pinned, and this is the
/// only caller that would exercise it — which is why the second defect
/// has never appeared in a simulation.
const CLAIMED_RECORDS_OFFERED: bool = false;

/// Execution state machine.
///
/// Handles transaction execution after blocks are committed.
pub struct ExecutionCoordinator {
    /// Finalizations ready for block inclusion, keyed by
    /// `TickId`. Terminal-state lookup surface for tick-id fetches,
    /// tx-membership queries, and proposal building. Held behind an
    /// `Arc` and shared across same-shard `ExecutionCoordinator`s so
    /// `IoLoop`'s sync-inventory bloom and elided-block rehydration read
    /// from one canonical store per shard rather than vnode-0's
    /// incidentally-convergent copy.
    finalized: Arc<FinalizationStore>,

    /// Current committed height for pruning stale entries.
    committed_height: BlockHeight,

    /// BFT-authenticated weighted timestamp of the last locally committed
    /// block. "Now" reference for timeouts that must be deterministic across
    /// validators and independent of block production rate.
    committed_ts: WeightedTimestamp,

    /// Anchor selecting the committee that governs the last locally committed
    /// block — the anchor its *parent* carried, since a block's committee
    /// keys on its parent. What tick and provision classification resolves
    /// against, so every replica groups a block's transactions exactly as the
    /// proposer built them and the verifier validated them.
    committed_committee_anchor_wt: WeightedTimestamp,

    // ═══════════════════════════════════════════════════════════════════════
    // Tick dispatch
    // ═══════════════════════════════════════════════════════════════════════
    /// Ticks composed at commit but not yet dispatched, in height order.
    /// Ticks execute serially — each output is the next tick's baseline —
    /// so the head dispatches only when no tick is in flight.
    pending_ticks: VecDeque<PendingTick>,

    /// Whether a dispatched tick's `ExecutionBatchCompleted` is still
    /// outstanding.
    tick_in_flight: bool,

    /// Packages the beacon registers that this node does not hold the
    /// bytes for, replaced wholesale at each beacon commit.
    ///
    /// A queued tick whose members run one of them waits at the dispatch
    /// head until the fetch lands. It cannot be consulted where the tick
    /// is composed: membership is what the committee's votes are cast
    /// over, so it has to be a function of committed chain state alone,
    /// and a node's holdings are not that. Dispatch is where the
    /// difference is local — the tick still answers for exactly the
    /// members it was composed with, whenever it runs.
    missing_packages: BTreeSet<Hash>,

    /// The highest tick whose output is on the tick chain. A resolution
    /// is emitted only once its tick's tick has appended, so a commit
    /// racing ahead of a queued tick cannot resolve a tick the chain has
    /// never seen.
    last_completed_tick: BlockHeight,

    /// Ticks with entries on the tick chain whose fate is still open. A
    /// tick absent from the map — never dispatched, or committed by a
    /// shard past the execution window — resolves nothing.
    ticked: BTreeMap<TickId, TickedBatch>,

    /// Committed transactions still owed an outcome, folded from the
    /// chain rather than read off live tick state — the only account of
    /// what this shard has in flight that can be rebuilt after losing
    /// that state.
    unresolved: UnresolvedTxs,

    /// The escrow records this shard inherited with a prefix, each still
    /// unresolved, by cell key.
    ///
    /// A merge successor's store arrives holding value its predecessors
    /// escrowed, and nothing else names it: its ledger begins empty, no
    /// body arrives with the leaves, and both children's chains have
    /// ended, so no counterpart record will ever be composed about them.
    /// What is left is the claim cell each record names — this shard's
    /// to read, since the merge gave it both children's prefixes — and
    /// an entry leaves here when a tick has taken it.
    inherited: BTreeMap<SubstateKey, CrossingCell>,

    /// The blocks a restart has to replay before this coordinator's
    /// account of what is in flight matches its peers'. Construction has
    /// no schedule to compose against, so they wait for
    /// [`on_committed_state_restored`](Self::on_committed_state_restored)
    /// and are empty from then on.
    replay_blocks: Vec<Verified<CertifiedBlock>>,

    /// Tick fates known but not yet emittable, each with the tick that
    /// carries its entries. Drained whenever a tick completes or a block
    /// commits.
    pending_tick_resolutions: Vec<(TickId, BlockHeight, TickResolution)>,

    // ═══════════════════════════════════════════════════════════════════════
    // Provisioning
    // ═══════════════════════════════════════════════════════════════════════
    /// Owns the verified-provision map, required/received remote-shard sets
    /// per tx, and the `ConflictDetector` used for bidirectional node-ID
    /// overlap detection. Wraps the detector as a field so conflict flows
    /// stay co-located with the provision state they reason about.
    provisioning: ProvisioningTracker,

    // ═══════════════════════════════════════════════════════════════════════
    // Per-tick execution
    // ═══════════════════════════════════════════════════════════════════════
    /// Committed transactions no tick has taken yet, each waiting on the
    /// provisions, engagement echoes or cells it needs to reach its
    /// outcome. Nothing here has said anything, so nothing here is owed.
    candidates: TickCandidates,

    /// Owns in-flight `TickState`s, their `VoteTracker`s, the
    /// certificate-dispatched gate, vote-retry bookkeeping, and the
    /// `tx_hash → TickId` reverse index. Every per-tick mutation the
    /// coordinator drives flows through this field.
    ticks: TickRegistry,

    // ═══════════════════════════════════════════════════════════════════════
    // Early arrivals (buffered until tracking starts at block commit)
    // ═══════════════════════════════════════════════════════════════════════
    /// Buffers execution votes and cross-shard ECs that arrived before the
    /// local tick was tracked. Drained on block commit (ECs) and on leader
    /// tracker creation (votes).
    early: EarlyArrivalBuffer,

    // ═══════════════════════════════════════════════════════════════════════
    // Expected Execution Certificate Tracking (Fallback Detection)
    // ═══════════════════════════════════════════════════════════════════════
    /// Tracks expected ECs from remote block headers and drives timeout-based
    /// fallback fetches when they don't arrive. Owns both the active-expectation
    /// set and the fulfilled-tombstone set used to guard against duplicate
    /// headers re-opening closed expectations.
    expected_certs: ExpectedCertTracker,

    // ═══════════════════════════════════════════════════════════════════════
    // Outbound EC Retention (target → source delivery guarantee)
    // ═══════════════════════════════════════════════════════════════════════
    /// Retains ECs the tick leader broadcast to remote shards and re-emits
    /// them on a deterministic interval until the tick finalizes locally
    /// (positive ACK signal) or the safety horizon elapses. Symmetric to
    /// `OutboundProvisionTracker` on the source side.
    outbound_certs: OutboundExecutionCertificateTracker,

    /// Aggregated local-shard execution certificates awaiting block commit.
    /// Held behind an `Arc` and shared with the `io_loop` so the inbound EC
    /// fetch handler can serve cross-shard fallback requests without taking
    /// a coordinator lock. Populated on local aggregation and on verifying
    /// a local-shard EC received via broadcast; evicted in
    /// `remove_finalization` once the containing block commits.
    exec_certs: Arc<ExecCertStore>,

    /// In-flight EC verifications, keyed by a content hash over the
    /// cached wire bytes. A flooding peer would otherwise re-trigger a
    /// dispatch on every byte-identical retransmit. Different aggregations
    /// of the same logical EC produce distinct wire bytes and so still
    /// dispatch — important when a first aggregation's signature is bad and
    /// a peer follows up with a valid one.
    pending_ec_verifications: HashSet<Hash>,

    /// In-flight `Finalization` verifications, keyed by `TickId`. The
    /// tick is content-addressed by id (one tick per `TickId`), so a second
    /// fetch arrival for the same tick can short-circuit the crypto pool.
    pending_finalization_verifications: HashSet<FinalizationHash>,

    // ═══════════════════════════════════════════════════════════════════════
    // Beacon-sync-lag buffers
    // ═══════════════════════════════════════════════════════════════════════
    /// Cross-shard ECs whose committee epoch this node's beacon hasn't reached,
    /// so `at(vote_anchor_ts)` can't resolve the signing committee. Keyed by the
    /// EC's own shard, bounded per shard (drop-oldest). Re-attempted on
    /// `BeaconBlockPersisted`. Pure catch-up: a buffered EC means *we* are
    /// behind, since under lookahead its committee is already globally fixed.
    awaiting_certs: AwaitingTopologyBuffer<Verifiable<ExecutionCertificate>>,

    /// Fetched `Finalization`s deferred for the same reason — a contained EC's
    /// committee epoch isn't in our schedule yet. Keyed by the tick's own shard;
    /// re-attempted on `BeaconBlockPersisted`.
    awaiting_finalizations: AwaitingTopologyBuffer<Arc<Verifiable<Finalization>>>,

    // ═══════════════════════════════════════════════════════════════════════
    // Commit-proof gate
    // ═══════════════════════════════════════════════════════════════════════
    /// What counterparts have said about the transactions legs here
    /// issued for, shared with the shard coordinator's vote fence: a
    /// core's refusal, a proved absence, a consumer's claim. This
    /// coordinator is the only writer, and the only one that says what
    /// to drop — the ledger below is what an entry there speaks for.
    ///
    /// One mirror, because the fence checks a record against exactly
    /// what was offered from, and two copies could answer differently.
    evidence: Arc<CounterpartMirror>,

    /// Commit-proven remote source blocks, shared with the shard
    /// coordinator, which owns the mirror and feeds it off
    /// `RemoteHeaderCommitted`.
    ///
    /// A cross-shard EC is consumable only against a proven source block
    /// — a bare QC certifies availability, and an f+1..2f corrupt
    /// committee can certify a sibling that never commits and export ECs
    /// computed from it. The same anchors are what a probe of a
    /// counterpart's committed set is taken against, and what this
    /// validator's vote fence holds a block's state proofs to: one
    /// mirror, so a bundle cannot pass the fence at an anchor no prober
    /// here would have chosen.
    proven_anchors: Arc<ProvenAnchors>,

    /// Cross-shard ECs racing ahead of their source block's commit proof,
    /// keyed by the EC's shard, bounded per shard (drop-oldest). Replayed
    /// when `RemoteHeaderCommitted` proves a block for the shard; entries
    /// still unproven re-buffer. Dropping an entry is safe — the expected
    /// tracker re-fetches on timeout.
    unproven_ecs: AwaitingTopologyBuffer<Verifiable<ExecutionCertificate>>,

    // ═══════════════════════════════════════════════════════════════════════
    // Split-boundary finalize gate
    // ═══════════════════════════════════════════════════════════════════════
    /// The proofs this validator's own fetches answered, each with the
    /// transactions whose probes it spoke to, held to offer in a block
    /// this validator proposes: a proof is committed content, folded by
    /// every replica at the same height, and the fetch is only how the
    /// proposer comes by the bytes. A bundle leaves when a block
    /// carries it, or when every transaction it answered for is gone.
    fetched: BTreeMap<StateProofBundle, BTreeSet<TxHash>>,

    /// Finalizations built but withheld because a contained EC names a
    /// shard that is scheduled to terminate, or past-terminal with its
    /// settled set not yet known (the gate's `Defer`). Re-checked on every
    /// commit and when a set is recorded; a tick leaves only on evidence —
    /// settled-set membership, the scheduled termination clearing, or the
    /// schedule evicting the shard — never on a clock. Keyed by `TickId`.
    gated_finalized: BTreeMap<FinalizationHash, Arc<Verifiable<Finalization>>>,

    /// This validator's identity.
    me: ValidatorId,

    /// This validator's home shard.
    local_shard: ShardId,
}

impl ExecutionCoordinator {
    /// Create a new execution state machine with its own fresh stores and a
    /// genesis commit frontier. For hosts running multiple same-shard
    /// validators, prefer [`Self::with_shared_stores`] to share one set of
    /// stores across every coordinator in the shard; for a recovered chain,
    /// it also seeds the frontier from storage.
    #[must_use]
    pub fn new(me: ValidatorId, local_shard: ShardId) -> Self {
        Self::with_shared_stores(
            me,
            local_shard,
            &RecoveredState::default(),
            Arc::new(ExecCertStore::new()),
            Arc::new(FinalizationStore::new()),
            Arc::new(ProvenAnchors::new()),
            Arc::new(CounterpartMirror::new()),
        )
    }

    /// Create a new execution state machine sharing both externally-owned
    /// `ExecCertStore` and `FinalizationStore`. Same-shard vnodes share
    /// one set of stores so the `IoLoop`'s inbound fetch handler and
    /// sync-inventory bloom read from a single canonical view per shard
    /// rather than vnode-0's incidentally-convergent copy.
    ///
    /// The commit-frontier scalars seed from where execution's own
    /// account resumes, which is not where consensus's does: a restart
    /// replays from the block committing the oldest transaction still
    /// owed an outcome, so the frontier seeds at the block below that one
    /// and the replay carries it up to the tip. With nothing owed there
    /// is nothing to replay and the recovered tip is the frontier.
    ///
    /// Either way the seed keeps the next commit on the exact carry path —
    /// height-contiguous, its committee anchor the previous block's — where
    /// a zero frontier would take the gap fallback and classify that
    /// block's ticks under the window it opens: a replica restarted just
    /// below an epoch cut that a reshape moves would group ticks
    /// differently from its peers and split execution votes. A seeded
    /// frontier also gives pre-first-commit bookkeeping (expected-cert
    /// ages, conflict detection) a real clock instead of a zero one.
    #[must_use]
    pub fn with_shared_stores(
        me: ValidatorId,
        local_shard: ShardId,
        recovered: &RecoveredState,
        exec_certs: Arc<ExecCertStore>,
        finalized: Arc<FinalizationStore>,
        proven_anchors: Arc<ProvenAnchors>,
        evidence: Arc<CounterpartMirror>,
    ) -> Self {
        // Execution resumes below the first block it replays, so the
        // replay carries the frontier up to the tip rather than starting
        // from it. Without the block under the floor there is no clock to
        // carry, and the first block replayed classifies under its own
        // anchor — the same fallback a chain with no history takes, which
        // a genesis frontier is what selects.
        let resume = recovered
            .replay
            .blocks
            .first()
            .map(|certified| (certified.block().height(), recovered.replay.anchor_wt));
        let (committed_height, committed_block_anchor_wt) = match resume {
            Some((first, Some(anchor))) => (first.saturating_sub(1), anchor),
            Some((_, None)) => (BlockHeight::GENESIS, WeightedTimestamp::ZERO),
            None => (recovered.committed_height, recovered.block_anchor_wt()),
        };
        // Whatever seeds the height seeds this too: the block above is
        // contiguous, so it carries `committed_ts` into the slot before
        // anything classifies against it.
        let committed_committee_anchor_wt = match resume {
            Some(_) => committed_block_anchor_wt,
            None => recovered.committee_anchor_wt(),
        };
        Self {
            proven_anchors,
            evidence,
            finalized,
            committed_height,
            committed_ts: committed_block_anchor_wt,
            committed_committee_anchor_wt,
            pending_ticks: VecDeque::new(),
            missing_packages: BTreeSet::new(),
            tick_in_flight: false,
            last_completed_tick: BlockHeight::GENESIS,
            ticked: BTreeMap::new(),
            unresolved: UnresolvedTxs::default(),
            inherited: recovered
                .inherited_records
                .iter()
                .filter_map(|(key, value)| Some((*key, CrossingCell::from_bytes(value)?)))
                .collect(),
            replay_blocks: recovered.replay.blocks.clone(),
            pending_tick_resolutions: Vec::new(),
            candidates: TickCandidates::new(local_shard),
            ticks: TickRegistry::new(),
            early: EarlyArrivalBuffer::new(),
            provisioning: ProvisioningTracker::new(),
            expected_certs: ExpectedCertTracker::new(),
            outbound_certs: OutboundExecutionCertificateTracker::new(),
            exec_certs,
            pending_ec_verifications: HashSet::new(),
            pending_finalization_verifications: HashSet::new(),
            awaiting_certs: AwaitingTopologyBuffer::new(),
            awaiting_finalizations: AwaitingTopologyBuffer::new(),

            unproven_ecs: AwaitingTopologyBuffer::new(),
            fetched: BTreeMap::new(),
            gated_finalized: BTreeMap::new(),
            me,
            local_shard,
        }
    }

    /// Reference to the shared finalization store. The `io_loop`
    /// clones this `Arc` into its `SharedCaches` so sync-inventory
    /// blooms and elided-block rehydration read from a single canonical
    /// per-shard store rather than vnode-0's incidentally-convergent
    /// copy.
    #[must_use]
    pub const fn finalization_store(&self) -> &Arc<FinalizationStore> {
        &self.finalized
    }

    /// Reference to the shared execution-certificate store. The `io_loop`
    /// clones this `Arc` into its `SharedCaches` so the inbound EC fetch
    /// handler can read aggregated local-shard certificates without
    /// acquiring a coordinator lock.
    #[must_use]
    pub const fn exec_cert_store(&self) -> &Arc<ExecCertStore> {
        &self.exec_certs
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tick Assignment
    // ═══════════════════════════════════════════════════════════════════════════

    /// The committed block's **anchored** committee — `at_for_shard(local_shard,
    /// anchor_wt)`, the same snapshot the proposer classified `ticks` against
    /// and the verifier validated against. Both of those resolve a block's
    /// committee from its parent, so `anchor_wt` is
    /// [`Self::committed_committee_anchor_wt`], not the block's own: the two
    /// straddle an epoch cut once per window, and a reshape cut there changes
    /// the shard set `compute_ticks` routes over.
    ///
    /// Tick and provision classification at commit keys on it, not the
    /// `ArcSwap` head, so every replica groups a block's transactions
    /// identically across a reshape boundary (a head-flipped replica would
    /// otherwise split execution votes). Falls back to the head only if the
    /// window was evicted — unreachable for a just-committed block, whose
    /// committee resolved at verification.
    fn classification_committee<'t>(
        &self,
        topology_schedule: &'t TopologySchedule,
        anchor_wt: WeightedTimestamp,
    ) -> &'t TopologySnapshot {
        topology_schedule
            .at_for_shard(self.local_shard, anchor_wt)
            .map_or_else(
                || topology_schedule.head().as_ref(),
                |(snapshot, _)| snapshot.as_ref(),
            )
    }

    /// The trie that says who was party to a transaction.
    ///
    /// One accessor rather than an anchor chosen per call site, because
    /// the two sites that ask it have to agree: composition derives an
    /// abandonment's participants from it, and the finalize gate
    /// re-derives the same set to put the fence's question to. A window
    /// later resolves a departed counterpart's *successor*, so a gate
    /// reading a different anchor than composition would ask about a
    /// shard that was never party — and pass, because a live successor is
    /// what [`settled_set_verdict`] steps over.
    ///
    /// The anchor is the block's committee anchor, not its own timestamp:
    /// the two straddle an epoch cut once per window, and it is the
    /// former that classified the block's content.
    fn counterpart_trie<'t>(&self, topology_schedule: &'t TopologySchedule) -> &'t ShardTrie {
        self.classification_committee(topology_schedule, self.committed_committee_anchor_wt)
            .shard_trie()
    }

    /// Set up per-tick execution state for a newly committed block.
    ///
    /// For each distinct tick, creates a [`TickState`], records tx → tick
    /// assignments, and pre-populates provisions that arrived before the
    /// block.
    ///
    /// Emits `ExecuteTransactions` actions
    /// for ticks that are fully provisioned at creation time: single-shard
    /// ticks always qualify; cross-shard ticks do when all required provisions
    /// arrived before block commit.
    ///
    /// Returns the emitted dispatch actions plus any early execution votes
    /// that need to be replayed through `dispatch_execution_vote()`.
    /// Register a new cross-shard tick's transactions: the dependency set
    /// execution waits on and the engagement echoes the payer's vote
    /// waits on.
    fn register_cross_shard_txs(
        &mut self,
        classification: &TopologySnapshot,
        txs: &[CommittedMember],
    ) -> Vec<EngagementWait> {
        let local_shard = self.local_shard;
        let mut engagement_waits: Vec<EngagementWait> = Vec::new();

        for CommittedMember {
            tx,
            participating,
            classified,
        } in txs
        {
            let tx_hash = tx.hash();
            // A divided member waits on its execution scope minus itself
            // and on the crossings its legs consume, and on nothing else.
            // Its inbound escrow is its engagement — reclaimable if
            // nothing follows — and the crossing bundle it consumes is
            // its counterpart's commitment, so neither half of the
            // engagement exchange is filed, and it runs under its own
            // committing block's clock.
            if classified.decomposed().holds() {
                let side = classified.first_side_at(local_shard);
                let validity_end = tx.validity_range().end_timestamp_exclusive;
                self.provisioning.record_required(
                    tx_hash,
                    divided_requirements(tx.legs(), tx.crossings(), classified, local_shard, side),
                    delivery_floor(side, validity_end),
                );
                continue;
            }
            let remote_participants = || -> BTreeSet<ShardId> {
                participating
                    .iter()
                    .filter(|&&s| s != local_shard)
                    .copied()
                    .collect()
            };

            // The payer shard votes only once every counterpart has echoed
            // its engagement — its own commitment of the transaction — or
            // the window closed without one. Keyed on the same participant
            // set the tick was grouped by, so every replica waits on the
            // same shards.
            if classification
                .shard_trie()
                .shard_for_prefix(tx.body().fee_payer)
                == local_shard
            {
                engagement_waits.push((
                    tx_hash,
                    remote_participants(),
                    tx.validity_range().end_timestamp_exclusive,
                ));
            }

            // The dependency set is what execution waits for: the shards
            // owning the transaction's read set, plus — on a non-payer
            // shard — the payer shard, whose bundle is the engagement
            // evidence and flows even with empty entries. Only the payer's
            // own commutative leg records an empty requirement and
            // dispatches without waiting.
            let trie = classification.shard_trie();
            let mut remote_shards: BTreeSet<ShardId> = tx
                .routing()
                .provision_prefixes
                .iter()
                .map(|prefix| trie.shard_for_prefix(*prefix))
                .filter(|&s| s != local_shard)
                .collect();
            let payer_shard = trie.shard_for_prefix(tx.body().fee_payer);
            if payer_shard != local_shard {
                remote_shards.insert(payer_shard);
                // The payer's bundle carries the transaction clock;
                // remember whose entry to read it from at dispatch.
                self.provisioning.record_payer_shard(tx_hash, payer_shard);
            }
            self.provisioning.record_required(
                tx_hash,
                remote_shards
                    .into_iter()
                    .map(Requirement::CommittedState)
                    .collect(),
                None,
            );
        }

        engagement_waits
    }

    /// Record what a committed block puts in flight: the ledger entry it
    /// owes an outcome for, the provisions and engagement echoes its
    /// cross-shard members wait on, and the candidate itself.
    ///
    /// `ts` is the committing block's, which is what a
    /// member executes under however many ticks later it runs — so a
    /// replay of the chain passes the anchors each transaction's own
    /// block carried rather than the tip's.
    ///
    /// Nothing executes here. Whether a transaction can reach its outcome
    /// at this commit is composition's question, asked again at every one.
    fn register_committed_txs(
        &mut self,
        classification: &TopologySnapshot,
        ts: WeightedTimestamp,
        transactions: &[Arc<Verifiable<Transaction>>],
    ) {
        let local_shard = self.local_shard;
        let members = assign_participants(classification, transactions);
        // One placement at one anchor: the trie this block committed
        // under, and the answer frozen onto each transaction from here.
        let trie = classification.shard_trie();
        // The ledger takes the transactions themselves. What it needs of
        // them — when they expire, what they reserved, what they reach
        // outside this shard — is theirs and this shard's, so a rebuild
        // reads the same account off the same blocks however long after.
        self.unresolved
            .register_committed(local_shard, ts, transactions.iter());
        let members: Vec<CommittedMember> = members
            .into_iter()
            .map(|(tx, participating)| CommittedMember {
                classified: Classified::freeze(tx.legs(), tx.owners(), trie),
                tx,
                participating,
            })
            .collect();
        let reaches_beyond: Vec<CommittedMember> = members
            .iter()
            .filter(|member| member.reaches_beyond(local_shard))
            .cloned()
            .collect();
        let engagement_waits = if reaches_beyond.is_empty() {
            Vec::new()
        } else {
            self.register_cross_shard_txs(classification, &reaches_beyond)
        };

        for CommittedMember {
            tx,
            participating,
            classified,
        } in members
        {
            // Block-container entries decoded from the wire land as
            // `Unverified`; lift via `from_persisted` under the same
            // BFT-transitive trust that gates the containing block. Honest
            // live-consensus blocks already carry `Verified` entries (the
            // `.into_verified()` arm short-circuits without re-validating).
            let verified: Arc<Verified<Transaction>> = match (*tx).clone().into_verified() {
                Ok(v) => Arc::new(v),
                Err(raw) => Arc::new(Verified::<Transaction>::from_persisted(raw)),
            };
            // A leg is what this shard runs of a member frozen divided
            // with it outside the core set. Marked here, beside the
            // freeze, so a replay marks the same entries. A shard that
            // only delivers holds no leg entry — nothing to reclaim, no
            // core whose refusal is its own — and runs on the delivery
            // window's clock instead.
            if classified.decomposed().holds() && !classified.core().contains(&local_shard) {
                if classified.delivers_at(local_shard) {
                    self.unresolved
                        .mark_delivery(tx.hash(), tx.validity_range().end_timestamp_exclusive);
                } else {
                    let shape = classified.shape(tx.legs(), tx.crossings());
                    self.unresolved.mark_leg(
                        tx.hash(),
                        Arc::clone(&verified),
                        classified.clone(),
                        shape.delivered_claims(local_shard),
                        shape.core_claims(local_shard),
                    );
                }
            } else if classified.decomposed().holds() {
                // A core member's verdict is its own, but what it issues
                // to deliveries elsewhere outlives it: kept for the
                // reclaim of whatever those deliveries never claim.
                self.unresolved.mark_issuer(
                    tx.hash(),
                    Arc::clone(&verified),
                    classified.clone(),
                    classified
                        .shape(tx.legs(), tx.crossings())
                        .delivered_claims(local_shard),
                );
            }
            self.candidates
                .register(verified, participating, ts, classified);
        }

        for (tx_hash, counterparts, validity_end) in engagement_waits {
            self.candidates.record_engagement_wait(
                tx_hash,
                counterparts,
                reclaim_probe_anchor(validity_end),
            );
        }
    }

    /// Replay the committed chain from where this coordinator's account
    /// of what is in flight has to resume.
    ///
    /// Tick state does not survive a restart and the chain does. Both
    /// halves of what was lost — which tick holds which transaction, and
    /// what each tick's baseline was — are functions of committed content
    /// alone, so re-driving the ordinary commit path over the stored
    /// blocks reproduces them exactly. A replica that skipped this would
    /// compose those transactions into a tick of its own, and its peers'
    /// certificate for that height would come back under a root it never
    /// computed.
    ///
    /// The blocks arrive with the provision bundles they carried already
    /// reattached, so a leg composes here on the evidence it composed on
    /// the first time rather than waiting for a fetch nobody will answer.
    ///
    /// Deferred to here rather than done at construction because
    /// composition needs a topology, and there is none until the schedule
    /// is up. Idempotent: the payload is taken, and a live commit that
    /// beat this call has already advanced the frontier past it.
    pub fn on_committed_state_restored(
        &mut self,
        topology_schedule: &TopologySchedule,
        derivation: &dyn Derivation,
    ) -> Vec<Action> {
        let blocks = std::mem::take(&mut self.replay_blocks);
        if blocks.is_empty() {
            return Vec::new();
        }
        // Provision deadlines stamp against this clock, which a commit
        // would have advanced before any of this ran.
        self.provisioning.advance_clock(self.committed_ts);

        tracing::info!(
            shard = %self.local_shard,
            blocks = blocks.len(),
            from = %self.committed_height.next(),
            "Replaying the chain the restart lost execution state for"
        );

        let mut actions = Vec::new();
        for certified in &blocks {
            // The replay window came off the store, so nothing derived
            // these on the way in.
            derive_block_transactions(certified.block(), derivation);
            // The whole of what a commit does to execution, in the order
            // a commit does it. A finalization the replay recomposes but
            // never releases leaves its members assigned to a tick that
            // has already settled — and a leg's reclaim, which is admitted
            // only where no tick speaks for the transaction, is then held
            // out for as long as the entry lives.
            self.cleanup_committed_finalizations(certified.block().certificates());
            actions.extend(self.on_block_committed(topology_schedule, certified));
        }
        actions
    }

    /// Compose this commit's tick and set it up to be attested.
    ///
    /// The tick takes the candidates that can reach their outcome in it
    /// and the transactions past their deadline that nothing else will,
    /// and nothing else. That is what makes it votable the moment its
    /// batch returns: every member of a tick has an outcome there, so no
    /// member of one waits on another.
    ///
    /// Returns the tick, if the commit composed one, and any early votes
    /// its leader can now replay.
    /// Admit into the tick being composed everything this commit
    /// abandons: past its deadline, with no shard left that could settle
    /// it.
    ///
    /// Read after composition's own assignments, so a member that just
    /// joined this tick is not taken from it — the tick that holds a
    /// transaction is the one that speaks for it.
    ///
    /// Each joins undispatched, reaching the shards its committing block
    /// named. Those are what routes this tick's certificate to the
    /// counterparts still waiting on a verdict for it — the abort is
    /// dominant, so their coverage closes on it — and what the fence
    /// asks its question about. Nothing is awaited: an abort needs no
    /// counterpart's verdict, so the whole shape's one set serves.
    fn admit_abandoned(
        &mut self,
        topology_schedule: &TopologySchedule,
        tick_id: TickId,
        state: &mut TickState,
    ) {
        let local_shard = self.local_shard;
        let trie = self.counterpart_trie(topology_schedule);
        for entry in self.abandonable(tick_id) {
            let Abandonable {
                tx_hash,
                declared_work,
                charge,
            } = entry;
            // A tick that held the member can never speak for it — its
            // coverage will not close — and its other legs would wait on
            // that coverage forever. It goes with the member; the rest of
            // its members reach their own deadlines instead.
            if let Some(held_by) = self.ticks.tick_assignment(tx_hash) {
                self.discard_tick(held_by);
            }
            let mut participating = self.unresolved.counterparts(tx_hash, trie);
            participating.insert(local_shard);
            state.admit(
                tx_hash,
                Membership::whole(participating).settling(),
                declared_work,
                Admission::Aborted,
            );
            // An abandonment reaches no engine, so the charge its verdict
            // settles is built here rather than read off a result. The
            // floor is owed whether or not the transaction ever ran: the
            // reservation engaged when its block committed it, and an
            // abort that released it without burning would price an
            // attempt nobody could execute below the success it was
            // competing with.
            //
            // Fees never move cross-shard, so only the shard holding the
            // vault settles it — the same question the engine asks of the
            // payers it prices.
            if trie.shard_for_prefix(charge.vault.owner) == local_shard {
                let fee =
                    build_fee_receipt(local_shard, trie, tx_hash, charge.vault, charge.amount);
                state.record_fee_receipt(StoredReceipt::synced(tx_hash, Arc::new(fee)));
            }
            self.candidates.remove(tx_hash);
            self.provisioning.remove_tx(tx_hash);
            self.ticks.assign_tx(tx_hash, tick_id);
            self.unresolved.certify(tx_hash);
        }
    }

    /// Drop every delivering candidate the delivery window has closed on,
    /// and the provisioning entry that was held for it.
    ///
    /// Past the close no tick can take the member, and a mixed shard's
    /// delivering candidate is removed by nothing else — its ledger entry
    /// is the leg's, and a leg is never abandoned.
    fn retire_closed_deliveries(&mut self) {
        for tx_hash in self.candidates.drop_closed_deliveries(self.committed_ts) {
            self.provisioning.remove_tx(tx_hash);
        }
    }

    /// Admit into the tick being composed every reclaim a committed
    /// record has licensed: the leg entries whose core, the record says,
    /// can never claim what they issued.
    ///
    /// Each joins dispatched and awaiting nobody but this shard, since
    /// its own certificate is the whole of its settlement; reserving
    /// nothing, since no block took a reservation for it; and running no
    /// node, since the engine takes the crossings back on the cell's own
    /// evidence. A tick still speaking for the transaction is left to:
    /// the reclaim waits for the leg's own finalization to commit.
    fn admit_reclaims(
        &mut self,
        tick_id: TickId,
        tick_ts: WeightedTimestamp,
        state: &mut TickState,
        requests: &mut Vec<CrossShardExecutionRequest>,
    ) {
        let local_shard = self.local_shard;
        for Reclaimable {
            tx_hash,
            body: transaction,
            classified,
            charged,
        } in self.unresolved.reclaimable()
        {
            if self.ticks.tick_assignment(tx_hash).is_some() {
                continue;
            }
            state.admit(
                tx_hash,
                Membership::whole(BTreeSet::from([local_shard])).settling(),
                0,
                Admission::Executes,
            );
            self.ticks.assign_tx(tx_hash, tick_id);
            self.unresolved.admit_reclaim(tx_hash);
            record_reclaim_admitted();
            // A mixed shard's delivering member waits on what the core
            // returns, and the evidence this reclaim is composed from is
            // that the core never claimed. Nothing is coming, so the
            // candidate goes with the leg it was registered beside.
            self.candidates.remove(tx_hash);
            self.provisioning.remove_tx(tx_hash);
            requests.push(CrossShardExecutionRequest {
                // The plan reads no body — every cell is the record's —
                // but the price still follows the vault, and this is the
                // shard that holds it.
                transaction: Some(Arc::clone(&transaction)),
                tx_hash,
                provisions: Vec::new(),
                clock: tick_ts,
                reaches_beyond: false,
                abortable: false,
                runs: Runs::Reclaim {
                    records: classified
                        .shape(transaction.legs(), transaction.crossings())
                        .records_issued(local_shard),
                    charged,
                },
                arrivals: Vec::new(),
            });
        }
    }

    /// Admit into the tick being composed the records this shard
    /// inherited with a prefix whose claim it can now read.
    ///
    /// One member per issuing transaction, under a name of this chain's
    /// own ([`inherited_member_name`]) rather than the transaction's.
    /// The transaction was decided on a chain that has ended, and this
    /// one never committed it: naming it here would put a second verdict
    /// on a transaction nothing local can speak for, and would offer the
    /// chain a resolution its own pre-cut rule exists to refuse. What
    /// this shard does decide is the housekeeping itself, which is
    /// nobody else's.
    ///
    /// The member carries the records and no body; whether each is
    /// credited back or deleted is the engine's to decide against the
    /// claim cell, which is the only reader holding a snapshot.
    ///
    /// Two things bound what is admitted, and both are read off
    /// committed content so every replica at one frontier admits the
    /// same set. The claim must route here, or this shard cannot read
    /// the answer at all and the record waits. And the block's clock
    /// must be inside the window an absent claim means something in.
    fn admit_inherited(
        &mut self,
        topology_schedule: &TopologySchedule,
        tick_id: TickId,
        tick_ts: WeightedTimestamp,
        state: &mut TickState,
        requests: &mut Vec<CrossShardExecutionRequest>,
    ) {
        if self.inherited.is_empty() {
            return;
        }
        let Some(committee) = topology_schedule.at(tick_ts) else {
            return;
        };
        let trie = committee.shard_trie();
        let local_shard = self.local_shard;
        let mut due: BTreeMap<TxHash, Vec<SubstateKey>> = BTreeMap::new();
        for (key, record) in &self.inherited {
            if trie.shard_for_prefix(record.consumer_claim.owner) != local_shard
                || !claim_readable_at(record.expiry_ms, tick_ts)
            {
                continue;
            }
            due.entry(record.tx).or_default().push(*key);
        }
        for (issued_by, records) in due {
            let tx_hash = inherited_member_name(issued_by, &records);
            state.admit(
                tx_hash,
                Membership::whole(BTreeSet::from([local_shard])).settling(),
                0,
                Admission::Executes,
            );
            self.ticks.assign_tx(tx_hash, tick_id);
            // Taken once: the credit deletes the cell, so a second
            // member over the same record would read nothing and the
            // records would be stranded behind a refusal.
            for key in &records {
                self.inherited.remove(key);
            }
            record_reclaim_admitted();
            requests.push(CrossShardExecutionRequest {
                tx_hash,
                // No body reached this shard: the chain that issued the
                // crossing ended at the cut, and the price it owed was
                // settled there.
                transaction: None,
                provisions: Vec::new(),
                clock: tick_ts,
                reaches_beyond: false,
                abortable: false,
                runs: Runs::Inherited { records },
                arrivals: Vec::new(),
            });
        }
    }

    /// Admit into the tick being composed every retirement a committed
    /// record has licensed: a member running no node, awaiting nobody,
    /// reserving nothing, charged nothing, that deletes the records of
    /// crossings every consumer has claimed.
    fn admit_retirements(
        &mut self,
        tick_id: TickId,
        tick_ts: WeightedTimestamp,
        state: &mut TickState,
        requests: &mut Vec<CrossShardExecutionRequest>,
    ) {
        let local_shard = self.local_shard;
        for Retirable {
            tx_hash,
            body: transaction,
            classified,
        } in self.unresolved.retirable()
        {
            // Only once nothing this shard runs of the transaction is
            // left: a mixed shard's delivering member is still a
            // candidate while the core's output is on its way, and the
            // retirement is the last word here, not a word beside it.
            if self.ticks.tick_assignment(tx_hash).is_some() || self.candidates.contains(tx_hash) {
                continue;
            }
            state.admit(
                tx_hash,
                Membership::housekeeping(local_shard),
                0,
                Admission::Executes,
            );
            self.ticks.assign_tx(tx_hash, tick_id);
            self.unresolved.admit_retire(tx_hash);
            requests.push(CrossShardExecutionRequest {
                transaction: Some(Arc::clone(&transaction)),
                tx_hash,
                provisions: Vec::new(),
                clock: tick_ts,
                reaches_beyond: false,
                abortable: false,
                runs: Runs::Retire {
                    records: classified
                        .shape(transaction.legs(), transaction.crossings())
                        .records_issued(local_shard),
                },
                arrivals: Vec::new(),
            });
        }
    }

    /// Seat one composed member in the tick: its reservation, its
    /// crossing targets, its tick assignment, and — for a mixed shard's
    /// issuing member — the delivering member it makes composable.
    fn admit_member(
        &mut self,
        tick_id: TickId,
        member: Admitted,
        state: &mut TickState,
        ticked: &mut TickedBatch,
        requests: &mut Vec<CrossShardExecutionRequest>,
    ) {
        let local_shard = self.local_shard;
        // The shape and the body travel together or not at all: a
        // housekeeping member has neither, and is admitted by its own
        // pass rather than through here.
        let shape = member
            .request
            .shape()
            .map(|(shape, body)| (shape.clone(), Arc::clone(body)));
        // A mixed shard's delivering member is the second this shard
        // runs of the transaction: the issuing one returned the
        // reservation its block took and settled the price, so this
        // one reserves nothing and issues nothing.
        let second_member = shape.as_ref().is_some_and(|(shape, _)| shape.is_second());
        let reach = member.membership_reach();
        state.admit(
            member.request.tx_hash,
            member.membership,
            match &shape {
                Some((_, body)) if !second_member => body.work(),
                _ => 0,
            },
            member.admission,
        );
        // Where this member's crossings land, off the frozen
        // classification: the shards its outcome promises a bundle
        // to, if it issues anything. Only an issuing member issues;
        // a delivery and a reclaim promise nobody a bundle.
        let targets: BTreeSet<ShardId> = match &shape {
            Some((shape, body)) if shape.side() == Side::Issuing => shape
                .classified()
                .shape(body.legs(), body.crossings())
                .edges()
                .into_iter()
                .filter(|edge| edge.from == local_shard)
                .flat_map(|edge| edge.to)
                .collect(),
            _ => BTreeSet::new(),
        };
        state.record_crossing_targets(member.request.tx_hash, targets);
        self.ticks.assign_tx(member.request.tx_hash, tick_id);
        self.unresolved.certify(member.request.tx_hash);
        // A shard with legs on both sides of the core runs them as two
        // members: the issuing one just admitted, and a delivering one
        // that waits on what the core returns. Registered here, at the
        // issuing admission, so every replica composes it from the
        // same commit; it joins a later tick once its arrival lands.
        if let Some((shape, body)) = &shape
            && shape.side() == Side::Issuing
            && shape.runs_both_sides()
        {
            self.provisioning.record_required(
                member.request.tx_hash,
                divided_requirements(
                    body.legs(),
                    body.crossings(),
                    shape.classified(),
                    local_shard,
                    Side::Delivering,
                ),
                delivery_floor(
                    Side::Delivering,
                    body.validity_range().end_timestamp_exclusive,
                ),
            );
            self.candidates.register_delivery(
                Arc::clone(body),
                reach,
                member.request.clock,
                shape.classified().clone(),
            );
        }
        if let Some((_, body)) = &shape
            && member.request.abortable
        {
            ticked.legs.insert(member.request.tx_hash);
            ticked
                .provisional_claims
                .extend(body.routing().declared_modes.clone());
        }
        requests.push(member.request);
    }

    fn compose_tick(
        &mut self,
        topology_schedule: &TopologySchedule,
        block: CommittingBlock,
        held: &mut ProvisionalCells,
    ) -> (
        Option<PendingTick>,
        Vec<Verifiable<ExecutionVote>>,
        Vec<TxHash>,
    ) {
        let local_shard = self.local_shard;
        let tick_id = TickId::new(local_shard, block.height);
        let admitted = self
            .candidates
            .compose(&self.provisioning, held, self.committed_ts);

        let mut state = TickState::new(tick_id, block.hash, block.ts);
        let mut requests: Vec<CrossShardExecutionRequest> = Vec::with_capacity(admitted.len());
        let mut ticked = TickedBatch {
            provisional_claims: Vec::new(),
            legs: BTreeSet::new(),
        };
        for member in admitted {
            self.admit_member(tick_id, member, &mut state, &mut ticked, &mut requests);
        }

        self.admit_abandoned(topology_schedule, tick_id, &mut state);
        self.admit_reclaims(tick_id, block.ts, &mut state, &mut requests);
        self.admit_retirements(tick_id, block.ts, &mut state, &mut requests);
        self.admit_inherited(
            topology_schedule,
            tick_id,
            block.ts,
            &mut state,
            &mut requests,
        );

        if state.is_empty() {
            return (None, Vec::new(), Vec::new());
        }

        // What the tick waits on is what a counterpart owes us, so the
        // wait-set is the expectation set: an entry arms the fallback
        // fetch that recovers a certificate the broadcast lost, and
        // retires when the tick lets the member go.
        for (tx_hash, shard) in state.awaited_counterparts() {
            self.expected_certs
                .register(shard, tx_hash, self.committed_ts);
        }

        let members: Vec<TxHash> = state.tx_hashes().to_vec();

        self.ticks.insert_tick(tick_id, state);
        // Only a tick that runs a batch appends an output, and only an
        // output has a fate to record. A tick that abandons and nothing
        // else claims no cell and settles nothing, so the chain never
        // hears of it.
        if !requests.is_empty() {
            self.ticked.insert(tick_id, ticked);
        }

        // Only the tick leader creates a `VoteTracker` for aggregation.
        // Resolved under the committee seated at the tick's own block,
        // which is the one that will verify the certificate. A window this
        // shard has already left seats nobody, and there is no leader to
        // be: the tick composes, but no vote it could carry would reach a
        // quorum.
        let mut votes_to_replay: Vec<Verifiable<ExecutionVote>> = Vec::new();
        if let Some(committee) = topology_schedule.at(block.ts)
            && let seated = committee.consensus_committee_for_shard(local_shard)
            && !seated.is_empty()
            && self.me == tick_leader(&tick_id, seated)
        {
            let quorum = committee.quorum_threshold_for_shard(local_shard);
            self.ticks
                .insert_tracker(tick_id, VoteTracker::new(tick_id, block.hash, quorum));
            let early_votes = self.early.drain_votes_for_tick(&tick_id);
            if !early_votes.is_empty() {
                tracing::debug!(
                    block_hash = ?block.hash,
                    tick = %tick_id,
                    count = early_votes.len(),
                    "Replaying early execution votes"
                );
                votes_to_replay.extend(early_votes);
            }
        }

        let pending = (!requests.is_empty()).then(|| PendingTick {
            tick: block.height,
            tick_ts: block.ts,
            // Off the block's anchored committee, on the same terms the
            // classification above reads it: what a seal opens onto is
            // execution output, and a window taken from this node's head
            // would make the answer depend on how far this node has
            // folded the beacon rather than on what the block committed.
            env: TickEnvironment::governing(
                self.classification_committee(
                    topology_schedule,
                    self.committed_committee_anchor_wt,
                ),
                topology_schedule.windows(),
            ),
            requests,
        });
        (pending, votes_to_replay, members)
    }

    /// Return completion data for every tick that can emit its vote.
    ///
    /// A tick becomes votable when its batch comes back — composition
    /// admitted only members that could reach their outcome in it, so
    /// there is nothing else to wait for.
    ///
    /// Ticks whose certificate is already dispatched or received are
    /// skipped, as are ticks whose committee the schedule cannot resolve.
    /// That last condition is part of votability rather than a check on
    /// the way out because building the vote consumes it: `build_vote_data`
    /// is one-shot and nothing clears the mark, so a tick scanned here and
    /// dropped afterwards has spent its vote without emitting one. A tick
    /// the schedule cannot route stays votable and is picked up by a later
    /// commit instead.
    ///
    /// # Panics
    ///
    /// Panics if `ticks_iter()` and `get_tick_mut()` disagree about tick
    /// presence — unreachable, no concurrent mutation between them.
    pub fn scan_votable_ticks(
        &mut self,
        topology_schedule: &TopologySchedule,
    ) -> Vec<CompletionData> {
        let local_shard = self.local_shard;
        let routable = |tick: &TickState| {
            topology_schedule
                .at(tick.vote_anchor_ts())
                .is_some_and(|snapshot| {
                    !snapshot
                        .consensus_committee_for_shard(local_shard)
                        .is_empty()
                })
        };

        let votable: Vec<TickId> = self
            .ticks
            .ticks_iter()
            .filter(|(tick_id, tick)| {
                !self.ticks.is_ec_dispatched(tick_id)
                    && !tick.local_ec_emitted()
                    && tick.can_emit_vote()
                    && routable(tick)
            })
            .map(|(tick_id, _)| *tick_id)
            .collect();

        let mut completions = Vec::new();
        for tick_id in votable {
            let tick = self
                .ticks
                .get_tick_mut(&tick_id)
                .expect("tick_id was just produced by ticks_iter() in this method");
            let block_hash = tick.block_hash();
            let block_height = tick.block_height();
            let Some((vote_anchor_ts, global_receipt_root, tx_outcomes)) = tick.build_vote_data()
            else {
                continue;
            };

            completions.push(CompletionData {
                block_hash,
                block_height,
                vote_anchor_ts,
                tick_id,
                global_receipt_root,
                tx_outcomes,
            });
        }

        // Building a vote is one of the two places a tick learns its
        // receipts disagree with its committee's.
        self.escalate_divergence();

        completions.sort_by_key(|a| a.tick_id);
        completions
    }

    /// Replace the set of registered packages this node lacks with the
    /// latest beacon reconciliation, releasing the dispatch head if what
    /// held it is no longer missing.
    ///
    /// Replaced wholesale rather than merged, so a package that arrived
    /// by any route — a fetch, this shard's own commit, a boot reseed —
    /// leaves the set at the next commit without anyone reporting it.
    pub fn on_missing_packages_updated(&mut self, packages: Vec<Hash>) -> Vec<Action> {
        let missing: BTreeSet<Hash> = packages.into_iter().collect();
        if missing == self.missing_packages {
            return Vec::new();
        }
        self.missing_packages = missing;
        self.dispatch_next_tick()
    }

    /// Drop installed packages from the missing set and release the
    /// dispatch head if they were what held it.
    ///
    /// Reported once the engine holds the code, not once the bytes
    /// arrive: what the head waits on is the ability to run, and
    /// installation is where that is acquired.
    pub fn on_packages_acquired(&mut self, packages: &[Hash]) -> Vec<Action> {
        let held = self.missing_packages.len();
        for package in packages {
            self.missing_packages.remove(package);
        }
        if self.missing_packages.len() == held {
            return Vec::new();
        }
        self.dispatch_next_tick()
    }

    /// Absorb a completed batch: route receipts and per-member outcomes
    /// onto the tick that ran them, then vote and, where the tick is
    /// already covered, finalize.
    ///
    /// Finalization can fall due here rather than at a commit: a tick
    /// whose local certificate arrived before this validator's engine
    /// finished defers under the `has_local_receipts_for_non_aborted`
    /// gate, and the batch landing is what releases it.
    pub fn on_execution_batch_completed(
        &mut self,
        topology_schedule: &TopologySchedule,
        tick: BlockHeight,
        outcome: TickBatchOutcome,
    ) -> Vec<Action> {
        self.tick_in_flight = false;
        self.last_completed_tick = self.last_completed_tick.max(tick);
        let mut actions = Vec::new();

        let TickBatchOutcome {
            tick_id,
            results,
            tx_outcomes,
            fee_receipts,
            attested_work,
        } = outcome;
        if let Some(state) = self.ticks.get_tick_mut(&tick_id) {
            for result in results {
                state.record_receipt(result);
            }
            for fee in fee_receipts {
                state.record_fee_receipt(fee);
            }
            for (tx_hash, work) in attested_work {
                state.record_attested_work(tx_hash, work);
            }
            for wr in tx_outcomes {
                state.record_escrowed(wr.tx_hash(), wr.escrowed().to_vec());
                let (tx_hash, outcome) = wr.into_parts();
                state.record_execution_result(tx_hash, outcome);
            }
            actions.extend(self.finalize(topology_schedule, &tick_id));
        } else {
            // The coordinator stopped tracking the tick between its
            // dispatch and its batch returning. That says where this
            // coordinator is, not what the tick's fate was, so it
            // resolves nothing: the first resolution recorded for a tick
            // is the one the chain applies, and claiming an abandonment
            // here would consume the entry the tick's real verdict needs.
            //
            // Every fate that reaches the chain reaches it elsewhere. The
            // tick that abandons a transaction discards the one holding
            // it and records that, a committed certificate records its
            // settlement ahead of the block work that untracks it, and a
            // reshape terminal tears the chain down outright.
            tracing::warn!(
                tick = tick.inner(),
                %tick_id,
                "ExecutionBatchCompleted for an untracked tick — dropping"
            );
        }

        // The completed tick's output is on the chain (the handler
        // appends before notifying), so fates waiting on it can resolve
        // and the next queued tick can go.
        actions.extend(self.drain_ready_tick_resolutions());
        actions.extend(self.dispatch_next_tick());
        actions
    }

    /// Scan complete ticks and emit `SignAndSendExecutionVote` actions.
    ///
    /// This is the SINGLE path to execution voting. Call after conflicts
    /// have been processed so tick state is deterministic at this height.
    /// Each vote is sent to the tick leader (unicast). The `vote_anchor_ts`
    /// is the tick's own block's shard consensus-authenticated weighted
    /// timestamp, which is what resolves the committee that attests.
    pub fn emit_vote_actions(&mut self, topology_schedule: &TopologySchedule) -> Vec<Action> {
        let local_vid = self.me;
        let completions = self.scan_votable_ticks(topology_schedule);
        let mut actions = Vec::with_capacity(completions.len());
        for completion in completions {
            // The tick's committee is the one seated at its vote anchor — the
            // same committee that will verify the EC. The scan admits only
            // ticks this resolves for, so a miss here is not reachable; a
            // tick dropped after the scan would have spent its one-shot vote
            // without casting it.
            let Some(committee) = topology_schedule
                .at(completion.vote_anchor_ts)
                .map(|s| s.consensus_committee_for_shard(self.local_shard).to_vec())
                .filter(|committee| !committee.is_empty())
            else {
                debug_assert!(false, "scan_votable_ticks admitted an unroutable tick");
                continue;
            };
            let leader = tick_leader(&completion.tick_id, &committee);
            // Track retry state for non-leaders so we can re-send to a
            // rotated leader if this one doesn't produce an EC.
            let tx_outcomes = Arc::new(completion.tx_outcomes);
            if local_vid != leader {
                self.ticks.record_vote_retry(
                    completion.tick_id,
                    PendingVoteRetry {
                        sent_at: self.committed_ts,
                        attempt: Attempt::INITIAL,
                        block_hash: completion.block_hash,
                        block_height: completion.block_height,
                        vote_anchor_ts: completion.vote_anchor_ts,
                        global_receipt_root: completion.global_receipt_root,
                        tx_outcomes: Arc::clone(&tx_outcomes),
                    },
                );
            }
            actions.push(Action::SignAndSendExecutionVote {
                block_hash: completion.block_hash,
                block_height: completion.block_height,
                vote_anchor_ts: completion.vote_anchor_ts,
                tick_id: completion.tick_id,
                global_receipt_root: completion.global_receipt_root,
                tx_outcomes: (*tx_outcomes).clone(),
                leader,
            });
        }
        actions
    }

    /// Clean up execution-local per-tick state for finalizations included in the
    /// committed block.
    ///
    /// Per-tx terminal state for the mempool is driven by
    /// `mempool::on_block_committed` reading `block.certificates` directly.
    /// This function only handles execution's own bookkeeping.
    pub fn cleanup_committed_finalizations(
        &mut self,
        certificates: &[Arc<Verifiable<Finalization>>],
    ) {
        for fw in certificates {
            // No-op for synced ticks we never aggregated locally; for ticks we
            // tracked, releases accumulator/cache state for the tick's txs.
            self.remove_finalization(fw.as_unverified());
        }
    }

    /// Apply provisions committed in a block.
    ///
    /// Absorbs every batch before reading any of it back: interleaving
    /// would let a candidate's readiness turn on provisions iteration
    /// order.
    ///
    /// Each batch is peeked for its [`Verifiable::verified`] marker before
    /// re-wrapping. Same-process upstream paths leave the marker live, so
    /// we borrow the existing [`Verified<Provisions>`] without a body
    /// clone. Wire-decoded blocks land at `Unverified`; the
    /// [`Verified::<Provisions>::from_committed_block`] gate then carries
    /// the BFT-transitive trust source via a re-wrap (one body clone).
    fn apply_committed_provisions(&mut self, batches: &[Arc<Verifiable<Provisions>>]) {
        // Sort for deterministic iteration (logs, action vector order).
        let mut ordered: Vec<&Arc<Verifiable<Provisions>>> = batches.iter().collect();
        ordered.sort_by_key(|b| b.hash());

        for provisions in &ordered {
            if let Some(v) = provisions.verified() {
                self.provisioning.absorb_provisions(v);
            } else {
                let verified = Verified::<Provisions>::from_committed_block(
                    provisions.as_unverified().clone(),
                );
                self.provisioning.absorb_provisions(&verified);
            }
        }

        self.candidates
            .absorb_engagement_evidence(&self.provisioning);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Vote handling
    // ═══════════════════════════════════════════════════════════════════════════

    /// Handle a locally-produced, pre-verified execution vote.
    /// Bypasses the batch-verify path and lands directly in the verified
    /// tally. See [`Self::dispatch_execution_vote`] for the leader,
    /// fallback, and early-buffer routing rules.
    pub fn on_verified_execution_vote(
        &mut self,
        topology_schedule: &TopologySchedule,
        vote: Verified<ExecutionVote>,
    ) -> Vec<Action> {
        self.dispatch_execution_vote(topology_schedule, vote.into())
    }

    /// Handle a wire-arrived execution vote. Buffered for batch
    /// verification once combined power could reach quorum. See
    /// [`Self::dispatch_execution_vote`] for the full routing rules.
    pub fn on_unverified_execution_vote(
        &mut self,
        topology_schedule: &TopologySchedule,
        vote: ExecutionVote,
    ) -> Vec<Action> {
        self.dispatch_execution_vote(topology_schedule, vote.into())
    }

    /// Routing hub for both ingestion paths.
    ///
    /// Only the tick leader (or a fallback leader via rotation)
    /// aggregates votes. If a vote arrives at a non-leader that has
    /// the accumulator but no tracker, a fallback `VoteTracker` is
    /// created on-demand (the sender determined this validator is the
    /// rotated leader for their retry attempt).
    ///
    /// The `Verifiable<ExecutionVote>` signature lets the
    /// early-arrivals buffer hold either taxonomy under one shape and
    /// replay them through the same path when a fallback tracker
    /// spins up.
    ///
    /// # Panics
    ///
    /// Panics if a vote tracker is created or recovered for a tick but
    /// is missing on the immediate `take_unverified_votes` lookup — the
    /// tracker is locked across `&mut self`, so this is unreachable.
    fn dispatch_execution_vote(
        &mut self,
        topology_schedule: &TopologySchedule,
        vote: Verifiable<ExecutionVote>,
    ) -> Vec<Action> {
        let tick_id = *vote.tick_id();
        let validator_id = vote.validator();

        // The committee seated at the vote's anchor — the same one whose
        // positional bitfield the EC will carry. `None` means our beacon
        // hasn't reached that epoch; drop and let the sender's retry re-deliver
        // once we catch up.
        let Some(committee) = topology_schedule.at(vote.vote_anchor_ts()) else {
            return vec![];
        };

        // Only votes from local-committee members count. A globally-known
        // validator outside this shard's committee whose vote pooled into
        // `unverified_power` would puff up the tracker into early
        // aggregation, producing an EC whose signature aggregate carries
        // signatures the verifier's bitfield-derived pubkey pool excludes
        // — guaranteed to fail verification and waste a leader rotation.
        // Mirrors `vote_keeper::record_received_vote`.
        if committee
            .committee_index_for_shard(self.local_shard, validator_id)
            .is_none()
        {
            tracing::warn!(
                validator = validator_id.inner(),
                "Execution vote from validator not in local committee"
            );
            return vec![];
        }

        if !self.ticks.contains_tracker(&tick_id) {
            if !self.ticks.contains_tick(&tick_id) {
                // Block hasn't committed yet — buffer as early vote.
                self.early.buffer_vote(tick_id, vote);
                return vec![];
            }
            if self.ticks.is_ec_dispatched(&tick_id) {
                // Already have EC for this tick — discard late vote.
                return vec![];
            }
            // Tick exists but no VoteTracker and no EC yet. This validator
            // was targeted as a fallback leader (rotated attempt). Create tracker.
            let quorum = committee.quorum_threshold_for_shard(self.local_shard);
            let block_hash = self
                .ticks
                .get_tick(&tick_id)
                .expect("contains_tick returned true two lines above")
                .block_hash();
            tracing::info!(
                tick = %tick_id,
                "Creating fallback VoteTracker — receiving votes as rotated leader"
            );
            let tracker = VoteTracker::new(tick_id, block_hash, quorum);
            self.ticks.insert_tracker(tick_id, tracker);

            // Replay any early votes that were buffered before block commit.
            // These may include retried votes from other validators who
            // committed faster and rotated to us before our block committed.
            let early = self.early.drain_votes_for_tick(&tick_id);
            if !early.is_empty() {
                tracing::debug!(
                    tick = %tick_id,
                    count = early.len(),
                    "Replaying early votes into fallback VoteTracker"
                );
                let mut actions = Vec::new();
                for ev in early {
                    actions.extend(self.dispatch_execution_vote(topology_schedule, ev));
                }
                // Process the current vote that triggered fallback creation.
                actions.extend(self.dispatch_execution_vote(topology_schedule, vote));
                return actions;
            }
        }

        // Already-verified votes (own votes from the sign-and-send gate, or
        // future cached-verified inputs) skip the buffer + batch-verify
        // round trip and land directly in the verified tally.
        let vote = match vote.into_verified() {
            Ok(verified) => return self.handle_verified_vote(topology_schedule, verified),
            Err(raw) => raw,
        };

        // Committee membership was confirmed above; the topology snapshot
        // invariant guarantees the public key resolves.
        let public_key = committee
            .public_key(validator_id)
            .expect("committee member has public key (TopologySnapshot invariant)");

        let tracker = self
            .ticks
            .get_tracker_mut(&tick_id)
            .expect("tracker was inserted above when contains_tracker returned false");

        // buffer_unverified_vote handles dedup per (validator, vote_anchor_ts).
        // Same validator can vote at multiple heights (round voting).
        if !tracker.buffer_unverified_vote(vote, public_key) {
            return vec![];
        }

        self.maybe_trigger_vote_verification(tick_id)
    }

    /// Check if we should trigger provisions verification for a tick's votes.
    fn maybe_trigger_vote_verification(&mut self, tick_id: TickId) -> Vec<Action> {
        let Some(tracker) = self.ticks.get_tracker_mut(&tick_id) else {
            return vec![];
        };

        if !tracker.should_trigger_verification() {
            return vec![];
        }

        let votes = tracker.take_unverified_votes();
        if votes.is_empty() {
            return vec![];
        }

        let block_hash = tracker.block_hash();

        tracing::debug!(
            block_hash = ?block_hash,
            tick = %tick_id,
            vote_count = votes.len(),
            "Dispatching execution vote provisions verification"
        );
        vec![Action::VerifyAndAggregateExecutionVotes {
            tick_id,
            block_hash,
            votes,
        }]
    }

    /// Handle a verified execution vote (own vote or already-verified).
    fn handle_verified_vote(
        &mut self,
        topology_schedule: &TopologySchedule,
        vote: Verified<ExecutionVote>,
    ) -> Vec<Action> {
        let tick_id = *vote.tick_id();
        // The vote anchors to a committee the beacon has reached: `at` returning
        // `None` means the beacon hasn't committed that epoch yet (drop and let
        // the sender retry). Membership was confirmed before delegating here.
        if topology_schedule.at(vote.vote_anchor_ts()).is_none() {
            return vec![];
        }

        let Some(tracker) = self.ticks.get_tracker_mut(&tick_id) else {
            return vec![];
        };

        tracker.add_verified_vote(vote);

        let mut actions = self.check_vote_quorum(topology_schedule, tick_id);
        actions.extend(self.maybe_trigger_vote_verification(tick_id));
        actions
    }

    /// Handle provisions execution vote verification completed.
    pub fn on_votes_verified(
        &mut self,
        topology_schedule: &TopologySchedule,
        tick_id: TickId,
        block_hash: BlockHash,
        verified_votes: Vec<Verified<ExecutionVote>>,
    ) -> Vec<Action> {
        // Diagnostic quorum threshold for the split-root warning below, keyed
        // on the votes' anchor before they're consumed into the tracker.
        let warn_quorum = verified_votes
            .first()
            .and_then(|v| topology_schedule.at(v.vote_anchor_ts()))
            .map(|s| s.quorum_threshold_for_shard(self.local_shard));

        let Some(tracker) = self.ticks.get_tracker_mut(&tick_id) else {
            return vec![];
        };

        tracker.on_verification_complete();

        for vote in verified_votes {
            tracker.add_verified_vote(vote);
        }

        // Warn if we have enough total power for quorum but it's split
        // across multiple global receipt roots — this means validators disagree
        // on execution results.
        if let Some(quorum) = warn_quorum
            && tracker.check_quorum().is_none()
            && tracker.total_verified_power() >= quorum
            && tracker.distinct_global_receipt_root_count() > 1
        {
            let summary = tracker.global_receipt_root_power_summary();
            tracing::warn!(
                block_hash = ?block_hash,
                tick = %tick_id,
                global_receipt_root_split = ?summary,
                quorum = quorum.inner(),
                "Execution vote quorum blocked: global receipt roots are split across validators"
            );
        }

        let mut actions = self.check_vote_quorum(topology_schedule, tick_id);
        actions.extend(self.maybe_trigger_vote_verification(tick_id));
        actions
    }

    /// Check if quorum is reached for a tick's votes.
    fn check_vote_quorum(
        &mut self,
        topology_schedule: &TopologySchedule,
        tick_id: TickId,
    ) -> Vec<Action> {
        let local_shard = self.local_shard;
        let Some(tracker) = self.ticks.get_tracker_mut(&tick_id) else {
            return vec![];
        };

        let Some((global_receipt_root, vote_anchor_ts, _total_power)) = tracker.check_quorum()
        else {
            return vec![];
        };

        // The EC's signer bitfield is positional against the committee seated
        // at `vote_anchor_ts` — the committee every verifier resolves from the
        // EC's own anchor. Resolve it before consuming the votes; `None`
        // (beacon behind this epoch) leaves the tracker intact to re-check on a
        // later commit.
        let Some(committee) = topology_schedule
            .at(vote_anchor_ts)
            .map(|s| s.committee_for_shard(local_shard).to_vec())
        else {
            return vec![];
        };

        let block_hash = tracker.block_hash();

        tracing::info!(
            block_hash = ?block_hash,
            tick = %tick_id,
            vote_anchor_ts = vote_anchor_ts.as_millis(),
            "Execution vote quorum reached — aggregating certificate"
        );

        let votes = tracker.take_votes(global_receipt_root, vote_anchor_ts);

        // Remove the vote tracker — this EC is the shard's final answer.
        // Mark tick as having an EC to skip it in scan_votable_ticks.
        self.ticks.remove_tracker(&tick_id);
        self.ticks.mark_ec_dispatched(tick_id);

        tracing::debug!(
            block_hash = ?block_hash,
            tick = %tick_id,
            votes = votes.len(),
            "Delegating signature aggregation to crypto pool"
        );

        // Stamp phase times for txs covered by the new local EC. Pure
        // telemetry — IoLoop's slow-tx finalization log reads it.
        let ec_tx_hashes = self
            .ticks
            .get_tick(&tick_id)
            .map(|w| w.tx_hashes().to_vec())
            .unwrap_or_default();

        // tx_outcomes are extracted from votes by the aggregation handler
        // (all quorum votes carry identical outcomes).
        let mut actions = vec![Action::AggregateExecutionCertificate {
            tick_id,
            global_receipt_root,
            votes,
            committee,
        }];
        if !ec_tx_hashes.is_empty() {
            actions.push(Action::RecordTxEcCreated {
                tx_hashes: ec_tx_hashes,
            });
        }
        actions
    }

    /// Handle execution certificate aggregation completed.
    ///
    /// Called when the crypto pool finishes signature aggregation for a tick's votes.
    /// Only the tick leader (primary or fallback) reaches this path.
    /// Broadcasts the EC to all local peers and remote participating shards,
    /// then feeds it to the tick-level certificate tracker for finalization.
    pub fn on_certificate_aggregated(
        &mut self,
        topology_schedule: &TopologySchedule,
        tick_id: &TickId,
        certificate: &Arc<Verified<ExecutionCertificate>>,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        // EC broadcast is routing — who should receive it now — so recipients
        // key on the active head, not the EC's anchor committee.
        let head = topology_schedule.head();

        // Who should receive this certificate is a question about the
        // batch's transactions, not about its identity: the shards they
        // reach, less our own — reach rather than awaited, since a shard
        // this tick waits on nothing from may still claim what a member
        // escrowed. Read off the batch's own record rather than the
        // provision accumulator, which may have been pruned by the time
        // the certificate is aggregated.
        //
        // The same record answers what each of them receives. A shard is
        // party to the transactions naming it and to no others, so that
        // is what its copy carries — the certificate a remote shard gets
        // is sized by its own stake in the batch rather than by the
        // batch.
        let per_target: Vec<(ShardId, HashSet<TxHash>)> = self
            .ticks
            .get_tick(tick_id)
            .map(|tick| {
                tick.counterpart_shards()
                    .into_iter()
                    .map(|shard| (shard, tick.txs_reaching(shard).collect()))
                    .collect()
            })
            .unwrap_or_default();

        // Make the cert available to the io_loop's inbound EC fetch handler
        // for fallback serving until the containing block commits.
        self.exec_certs.insert(Arc::clone(certificate));

        // Broadcast EC to all local peers (they don't aggregate — they need it).
        let local_peers = peers_excluding_self(head, self.me, self.local_shard);
        if !local_peers.is_empty() {
            actions.push(Action::BroadcastExecutionCertificate {
                shard: self.local_shard,
                certificate: Arc::clone(certificate),
                recipients: local_peers,
            });
        }

        // Broadcast each target shard its own projection. Track the
        // per-target send so a dropped notify is re-emitted before the
        // source's 24s fallback timer trips — symmetric to ef4eb45a on
        // provisions; the tracker holds what was sent, so a re-broadcast
        // repeats it byte for byte.
        for (target_shard, txs) in &per_target {
            let recipients: Vec<ValidatorId> = head.committee_for_shard(*target_shard).to_vec();
            let Some(projected) = certificate.project_to(txs) else {
                continue;
            };
            let projected = Arc::new(projected);
            self.outbound_certs.on_broadcast(
                Arc::clone(&projected),
                *target_shard,
                recipients.clone(),
            );
            actions.push(Action::BroadcastExecutionCertificate {
                shard: *target_shard,
                certificate: projected,
                recipients,
            });
        }

        tracing::debug!(
            tick = %tick_id,
            tx_count = certificate.tx_outcomes().len(),
            remote_shards = per_target.len(),
            "Tick leader broadcasting EC to local peers and remote shards"
        );

        // Feed the EC to the tick-level certificate tracker for finalization.
        actions.extend(self.handle_attestation(topology_schedule, certificate));

        actions
    }

    /// Handle an execution certificate received from another validator.
    ///
    /// Always dispatches signature verification before the cert can
    /// influence any tick state. Routing (and any buffering for txs whose
    /// blocks haven't committed yet) happens in `on_certificate_verified`
    /// once the crypto pool confirms the signature — buffering here without
    /// verifying would let a Byzantine remote inject forged `tx_outcomes`
    /// that the replay path later trusts.
    pub fn on_execution_certificate(
        &mut self,
        topology_schedule: &TopologySchedule,
        cert: Verifiable<ExecutionCertificate>,
    ) -> Vec<Action> {
        let shard = cert.shard_id();
        let wire_hash = cert.wire_hash();

        // Cached-verified short-circuit. `exec_certs` is shared across
        // same-shard vnodes (one `Arc<ExecCertStore>` per shard), so a
        // peer vnode's aggregation makes this EC available to ours
        // before the gossip arrives. A wire-hash match against the
        // cached entry means this is the same aggregation we already
        // verified and routed; a mismatch is a different aggregation
        // of the same logical EC and still needs its own signature check.
        if let Some(cached) = self.exec_certs.get(cert.tick_id())
            && cached.wire_hash() == wire_hash
        {
            tracing::debug!(
                shard = shard.inner(),
                tick = %cert.tick_id(),
                "Cached verified EC matches incoming wire hash — skipping verify dispatch"
            );
            return vec![];
        }

        // Skip verify dispatch for byte-identical retransmits while a
        // verification is already in flight. Different aggregations of the
        // same logical EC produce distinct wire bytes, so the legitimate
        // case of "first aggregation invalid, second valid" is preserved.
        if !self.pending_ec_verifications.insert(wire_hash) {
            tracing::debug!(
                shard = shard.inner(),
                tick = %cert.tick_id(),
                "Duplicate EC verification dispatch suppressed"
            );
            return vec![];
        }

        // The halt-recovery freeze: an EC from a recovering shard above the
        // beacon-attested frontier is one the retained beyond-f committee
        // could only have produced after the halt. It resolves the old
        // committee at its stale anchor and its signatures verify, so
        // without this fence a forged finalization would export
        // cross-shard. Drop it — the fence is the same authenticated cutoff
        // every consumer folds.
        if topology_schedule.recovery_fences(shard, cert.block_height()) {
            tracing::warn!(
                shard = shard.inner(),
                tick = %cert.tick_id(),
                height = cert.block_height().inner(),
                "Dropping EC from a recovering shard past the freeze frontier"
            );
            self.pending_ec_verifications.remove(&wire_hash);
            return vec![Action::AbandonFetch(FetchAbandon::ExecutionCerts {
                ids: fetch_keys_covered(&cert),
            })];
        }

        // Commit-proof gate: a cross-shard EC is consumable only against a
        // commit-proven source block. Certification alone is not enough —
        // an f+1..2f corrupt committee can certify a sibling block that
        // never commits and export ECs computed from it. Defer until the
        // remote-header coordinator holds the committing structure
        // (`RemoteHeaderCommitted` replays the buffer); the proof trails
        // the source header by one child header at worst. A departed
        // shard's settled set answers instead where it can — see
        // `settled_set_admits` — because a departed chain supplies no
        // further commit proofs.
        if shard != self.local_shard
            && self.proven_anchors.at(shard, cert.block_height()).is_none()
            && !self.settled_set_admits(shard, &cert)
        {
            let height = cert.block_height();
            tracing::debug!(
                shard = shard.inner(),
                tick = %cert.tick_id(),
                height = height.inner(),
                "Deferring EC until its source block is commit-proven"
            );
            self.pending_ec_verifications.remove(&wire_hash);
            self.unproven_ecs.push(shard, cert);
            // At or below the shard's attested boundary the height sits
            // under the remote-header sync anchor — a joiner or a fresh
            // recovery committee anchors there and syncs only forward, so
            // no range fetch ever delivers this block's committing
            // structure. Ask the remote-header coordinator for the commit
            // proof explicitly; above the boundary the forward sync (or
            // gossip) delivers it in the ordinary course.
            if topology_schedule
                .head()
                .boundary(shard)
                .is_some_and(|anchor| height <= anchor.height)
            {
                return vec![Action::Continuation(ProtocolEvent::CommitProofNeeded {
                    source_shard: shard,
                    block_height: height,
                })];
            }
            return vec![];
        }

        let committee = match topology_schedule.lookup(cert.vote_anchor_ts()) {
            ScheduleLookup::Committee(committee) => committee,
            ScheduleLookup::NotYetCommitted => {
                // Beacon hasn't reached this EC's epoch — buffer for replay on
                // catch-up rather than abandoning and re-fetching. Release the
                // in-flight slot so the replay re-dispatches.
                self.pending_ec_verifications.remove(&wire_hash);
                self.awaiting_certs.push(cert.shard_id(), cert);
                return vec![];
            }
            ScheduleLookup::Evicted => {
                // Below the schedule floor the EC is past its retention
                // horizon — provably terminal everywhere, never resolvable
                // again. Drop instead of buffering, releasing the in-flight
                // slot and the fetch binding.
                tracing::warn!(
                    shard = shard.inner(),
                    tick = %cert.tick_id(),
                    "EC's committee epoch is below the schedule floor — dropping"
                );
                self.pending_ec_verifications.remove(&wire_hash);
                return vec![Action::AbandonFetch(FetchAbandon::ExecutionCerts {
                    ids: fetch_keys_covered(&cert),
                })];
            }
        };
        let Some(public_keys) = committee_public_keys_for_shard(committee, shard) else {
            tracing::warn!(
                shard = shard.inner(),
                "Could not resolve EC committee keys — snapshot incomplete"
            );
            // Verification will never complete; release the in-flight slot
            // so a subsequent arrival isn't permanently shadowed.
            self.pending_ec_verifications.remove(&wire_hash);
            return vec![Action::AbandonFetch(FetchAbandon::ExecutionCerts {
                ids: fetch_keys_covered(&cert),
            })];
        };

        vec![Action::VerifyExecutionCertificateSignature {
            certificate: cert,
            public_keys,
        }]
    }

    /// Handle execution certificate signature verification result.
    ///
    /// If valid, hand the cert to `handle_attestation` which routes
    /// per-tx outcomes into any local tick trackers and buffers txs whose
    /// blocks haven't committed yet for replay.
    pub fn on_certificate_verified(
        &mut self,
        topology_schedule: &TopologySchedule,
        result: Result<
            Arc<Verified<ExecutionCertificate>>,
            (Arc<ExecutionCertificate>, ExecutionCertificateVerifyError),
        >,
    ) -> Vec<Action> {
        // Release the in-flight slot regardless of outcome — a failed
        // signature still lets the next byte-identical retransmit
        // dispatch again (in case the failure was transient pool error
        // rather than a real signature mismatch). Subsequent arrivals
        // with a different aggregation hash to a different `wire_hash`
        // and aren't gated by this slot.
        let ec_arc = match result {
            Ok(verified) => {
                self.pending_ec_verifications.remove(&verified.wire_hash());
                verified
            }
            Err((raw, err)) => {
                self.pending_ec_verifications.remove(&raw.wire_hash());
                tracing::warn!(
                    shard = raw.shard_id().inner(),
                    tick = %raw.tick_id(),
                    error = ?err,
                    "Invalid execution certificate signature"
                );
                return vec![Action::AbandonFetch(FetchAbandon::ExecutionCerts {
                    ids: fetch_keys_covered(&raw),
                })];
            }
        };

        // A single Byzantine signer can produce a cryptographically valid
        // EC; require 2f+1 voting power on the EC's own shard before any
        // state mutation downstream. The committee is the one seated at the
        // EC's anchor. `on_execution_certificate` already resolved it to dispatch
        // this verification, so `None` here means that epoch aged out of the
        // schedule in the interim (the beacon advanced past retention) — the
        // EC is stale, so abandon it.
        let Some(committee) = topology_schedule.at(ec_arc.vote_anchor_ts()) else {
            tracing::warn!(
                shard = ec_arc.shard_id().inner(),
                tick = %ec_arc.tick_id(),
                "Discarding execution certificate — epoch evicted from schedule before verification completed"
            );
            return vec![Action::AbandonFetch(FetchAbandon::ExecutionCerts {
                ids: fetch_keys_covered(&ec_arc),
            })];
        };
        if !ec_has_shard_quorum_power(committee, &ec_arc) {
            tracing::warn!(
                shard = ec_arc.shard_id().inner(),
                tick = %ec_arc.tick_id(),
                "Discarding sub-quorum execution certificate"
            );
            return vec![Action::AbandonFetch(FetchAbandon::ExecutionCerts {
                ids: fetch_keys_covered(&ec_arc),
            })];
        }
        // The recovery freeze, re-checked here: an EC dispatched before the
        // beacon folded a source-shard halt recovery reaches this point with
        // a valid signature and quorum, but if the freeze has since landed
        // and the EC sits past the attested frontier it is a forged orphan
        // the fence must still drop before it mutates any tick state.
        if topology_schedule.recovery_fences(ec_arc.shard_id(), ec_arc.block_height()) {
            tracing::warn!(
                shard = ec_arc.shard_id().inner(),
                tick = %ec_arc.tick_id(),
                height = ec_arc.block_height().inner(),
                "Discarding verified EC from a recovering shard past the freeze frontier"
            );
            return vec![Action::AbandonFetch(FetchAbandon::ExecutionCerts {
                ids: fetch_keys_covered(&ec_arc),
            })];
        }

        let shard = ec_arc.shard_id();

        // Clearing the tombstone before verification would let a Byzantine
        // peer ship an EC with a far-future `vote_anchor_ts`, populating
        // the fulfilled tombstone (deadline = vote_anchor_ts +
        // RETENTION_HORIZON) and suppressing legitimate fallback fetches
        // indefinitely while the verify pool silently rejects the forgery.
        let cleared = self.expected_certs.mark_fulfilled(
            shard,
            ec_arc.tx_outcomes().iter().map(TxOutcome::tx_hash),
            ec_arc.deadline(),
        );
        if cleared {
            tracing::debug!(
                source_shard = shard.inner(),
                block_height = ec_arc.block_height().inner(),
                txs = ec_arc.tx_outcomes().len(),
                at_local_ts_ms = self.committed_ts.as_millis(),
                "Fulfilled expected exec cert"
            );
        }

        let mut actions = vec![Action::Continuation(
            ProtocolEvent::ExecutionCertificateAdmitted {
                certificate: Arc::clone(&ec_arc),
            },
        )];

        // If this is a local shard EC, mark the tick as having an EC to skip
        // it in scan_votable_ticks, and persist it for fallback serving to
        // remote shards.
        if shard == self.local_shard {
            self.ticks.mark_ec_dispatched(*ec_arc.tick_id());
            // EC received from tick leader — cancel any pending vote retry.
            self.ticks.clear_vote_retry(ec_arc.tick_id());
            // Make the verified cert available to the io_loop's inbound EC
            // fetch handler for fallback serving until block commit.
            self.exec_certs.insert(Arc::clone(&ec_arc));
        }

        actions.extend(self.handle_attestation(topology_schedule, &ec_arc));
        actions
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Expected Execution Certificate Tracking
    // ═══════════════════════════════════════════════════════════════════════════

    /// The counterpart mirror this coordinator writes, for the vote
    /// fence to read.
    #[must_use]
    pub const fn evidence(&self) -> &Arc<CounterpartMirror> {
        &self.evidence
    }

    /// The shared mirror of commit-proven remote anchors, which the shard
    /// coordinator owns and fills.
    #[must_use]
    pub const fn proven_anchors(&self) -> &Arc<ProvenAnchors> {
        &self.proven_anchors
    }

    /// Handle a commit-proven remote header from the `RemoteHeaderCoordinator`.
    ///
    /// The anchor is already in the shared mirror — the shard coordinator
    /// records it before this runs, which is what opens the commit-proof
    /// gate. What is left here is replaying the shard's deferred ECs
    /// through [`Self::on_execution_certificate`], with entries whose
    /// source block is still unproven re-buffering, and taking the probes
    /// the new header now anchors.
    pub fn on_committed_remote_header(
        &mut self,
        topology_schedule: &TopologySchedule,
        source_shard: ShardId,
    ) -> Vec<Action> {
        if source_shard == self.local_shard {
            return vec![];
        }
        let deferred = self.unproven_ecs.drain_shard(source_shard);
        let mut actions = Vec::new();
        for cert in deferred {
            actions.extend(self.on_execution_certificate(topology_schedule, cert));
        }
        // A counterpart's header past a leg's deadline is what a probe
        // of its committed set, or of its claim cell, waits on.
        actions.extend(self.probe_silent_counterparts(topology_schedule));
        actions
    }

    /// Ask each silent counterpart whether it took the transaction a
    /// leg here issued for, once the transaction's deadline has passed.
    ///
    /// The deadline gates the probe and never the reclaim: absence at a
    /// block past the floor is the evidence, and before it the
    /// counterpart may still legitimately act. A core is asked about the
    /// transaction's committed cell past the deadline, and the probe
    /// goes to the core's lowest shard — any one core shard's absence
    /// suffices, and the choice has to be the same on every validator or
    /// a voter's mirror would name a shard the record does not. A
    /// delivering shard is asked about the crossing's claim cell past
    /// the lapse, the delivery window's close plus the finalization
    /// delay, since a delivery admitted under the close has claimed by
    /// then or never will. Each is asked against the newest commit-proven
    /// header of that shard inside its window — at or past its floor and
    /// short of the probed cell's own sweep, since a proof against a
    /// swept cell is a true proof of nothing — which is the header the
    /// shard is likeliest to still serve, a proof being taken from a
    /// bounded history behind its tip. A shard whose header has not
    /// reached here yet is asked when it does, and one whose every held
    /// header is past the window is not asked at all: the entry then
    /// waits out its horizon.
    ///
    /// A delivering shard that departs at a reshape may leave no header
    /// past the lapse at all, so the claim cell is asked about wherever
    /// its prefix sits: on the shard that was to deliver it and on the
    /// shard the trie names for its owner now, which is the successor
    /// holding the departed chain's cells. Both are asked rather than
    /// the trie's answer alone, because the vote fence checks a record
    /// against the voter's own proof of the shard it names, and two
    /// validators straddling the cut would otherwise prove different
    /// shards and never both vote one record.
    ///
    /// The cell is named from signed content and the counterpart shard
    /// alone, so nothing but the header and the proof is fetched.
    fn probe_silent_counterparts(&mut self, topology_schedule: &TopologySchedule) -> Vec<Action> {
        let trie = self.counterpart_trie(topology_schedule);
        let mut wanted: BTreeMap<StateAnchor, Vec<SubstateKey>> = BTreeMap::new();
        for entry in self.unresolved.probeable(self.committed_ts) {
            for (shard, key, floor, probed) in counterpart_cells(&entry, trie) {
                // The chain has answered: nothing is asked again.
                if self.unresolved.answered(entry.tx_hash, shard, probed) {
                    continue;
                }
                // The newest licensed header held: the one the shard is
                // likeliest to still serve, since a proof is taken from
                // a bounded history behind its tip.
                let Some((height, source)) = self
                    .proven_anchors
                    .newest_licensed(shard, |ts| probed.licenses(ts, entry.validity_end))
                else {
                    continue;
                };
                // A question in flight is left alone: a core's header
                // lands every block, and moving the probe to each new
                // one abandons the fetch before its answer returns. A
                // probe whose fetch has answered is moved on, which is
                // how a claim the chain read absent is asked again —
                // at a newer header, not of the same one every block.
                if self
                    .unresolved
                    .probe_stands(entry.tx_hash, shard, probed, height)
                {
                    continue;
                }
                let anchor = StateAnchor {
                    shard,
                    height,
                    state_root: source.state_root,
                };
                self.unresolved.record_probe(
                    entry.tx_hash,
                    shard,
                    Probe {
                        anchor,
                        key,
                        probed_wt: source.ts,
                        floor,
                        probed,
                        answered: false,
                    },
                );
                wanted.entry(anchor).or_default().push(key);
            }
        }
        wanted
            .into_iter()
            .map(|(anchor, keys)| {
                Action::Fetch(FetchRequest::StateProof {
                    anchor,
                    keys,
                    preferred: None,
                    class: None,
                })
            })
            .collect()
    }

    /// Keep a fetched proof for a block this validator proposes.
    ///
    /// The fetch is only how the proposer comes by the bytes: nothing is
    /// read off the answer here, since the answer is the chain's once a
    /// block carries the proof and every replica folds it there. The
    /// probes the proof spoke to are marked answered, so the question is
    /// not put to the same header again, and the bundle is kept beside
    /// the transactions it answered for, dated to the clock the probe
    /// read off the header.
    pub fn on_state_proof_verified(
        &mut self,
        anchor: StateAnchor,
        keys: Vec<SubstateKey>,
        proof: MerkleInclusionProof,
    ) {
        let (answered, anchor_ts) = self.unresolved.mark_probes_answered(anchor, &keys);
        // An answer nothing here asked about is nobody's to commit.
        if let Some(anchor_ts) = anchor_ts {
            self.fetched
                .entry(StateProofBundle::new(anchor, anchor_ts, keys, proof))
                .or_default()
                .extend(answered);
        }
    }

    /// Fold the proofs a committed block carries into the answers every
    /// replica holds, and hand each to the vote fence.
    ///
    /// A bundle answers every cell of the ledger's on the anchor's shard
    /// whose window the anchor's clock sits inside — whether or not this
    /// replica had a probe out, and wherever its own probe sat — so a
    /// replica that never fetched reads the same answer as the one that
    /// did. A key found present means the counterpart took the
    /// transaction: a core's own certificate speaks for it next — a
    /// refusal there is mirrored on arrival, and a success leaves
    /// nothing to reclaim — and a claim present is the consumer's
    /// settlement. A core consumer's claim absent says only that the
    /// core has not claimed yet, and is asked again at the next header.
    /// The first proof to answer a cell is the answer; a later one adds
    /// nothing. The hand-off is a continuation emitted here rather than
    /// a map the fence reads later, so an answer is never collected
    /// before it is drained.
    fn fold_state_proofs(
        &mut self,
        topology_schedule: &TopologySchedule,
        block: &Block,
    ) -> Vec<Action> {
        if block.state_proofs().is_empty() {
            return Vec::new();
        }
        let trie = self.counterpart_trie(topology_schedule);
        let cells: Vec<(Probeable, Vec<CounterpartCell>)> = self
            .unresolved
            .cells()
            .into_iter()
            .map(|entry| {
                let cells = counterpart_cells(&entry, trie);
                (entry, cells)
            })
            .collect();
        let mut actions = Vec::new();
        for claim in block.state_proofs() {
            // A verdict is the counterpart's own word, folded from the
            // chain rather than from whatever this replica happened to
            // hear broadcast — which is the whole point of committing it.
            match claim {
                CounterpartClaim::Verdict(verdict) => actions.extend(self.fold_verdict(verdict)),
                CounterpartClaim::Cells(bundle) => {
                    actions.extend(self.fold_cells(bundle, &cells));
                }
            }
        }
        actions
    }

    /// Fold one bundle's answers into the questions the ledger is
    /// waiting on.
    fn fold_cells(
        &mut self,
        bundle: &StateProofBundle,
        cells: &[(Probeable, Vec<CounterpartCell>)],
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        let inclusions = match bundle.inclusions() {
            Ok(inclusions) => inclusions,
            Err(error) => {
                tracing::error!(
                    shard = ?bundle.anchor.shard,
                    height = bundle.anchor.height.inner(),
                    %error,
                    "A committed state proof does not answer for its keys"
                );
                return actions;
            }
        };
        {
            for (entry, cells) in cells {
                for &(shard, key, floor, probed) in cells {
                    if shard != bundle.anchor.shard
                        || !probed.licenses(bundle.anchor_ts, entry.validity_end)
                    {
                        continue;
                    }
                    let Some(&(_, inclusion)) = inclusions.iter().find(|(asked, _)| *asked == key)
                    else {
                        continue;
                    };
                    let answer = match (inclusion, probed) {
                        (Inclusion::Present(_), Probed::Core) => Answer::Committed,
                        (Inclusion::Present(_), Probed::Claim | Probed::Delivery) => {
                            Answer::Present(ClaimProof {
                                probed_wt: bundle.anchor_ts,
                            })
                        }
                        // A claim absent on a core of one shard is the
                        // core never taking the crossing: its one
                        // execution wrote the claim by the deadline or
                        // never will, and the window opens there. On a
                        // core of more it says only that a sibling is
                        // pending, and the committed cell answers.
                        (Inclusion::Absent, Probed::Claim) if entry.core.len() != 1 => continue,
                        (Inclusion::Absent, Probed::Core | Probed::Delivery | Probed::Claim) => {
                            Answer::Absent(Absence {
                                probed_wt: bundle.anchor_ts,
                                floor,
                            })
                        }
                    };
                    // The question is answered, and a fetch still out
                    // for it is released with it.
                    if !self.unresolved.close_question(entry.tx_hash, shard, probed) {
                        continue;
                    }
                    match answer {
                        Answer::Committed => {}
                        Answer::Absent(absence) => {
                            self.evidence
                                .record_absence(entry.tx_hash, shard, probed, absence);
                        }
                        Answer::Present(presence) => {
                            self.evidence
                                .record_presence(entry.tx_hash, shard, presence);
                        }
                    }
                    record_reclaim_probe_answered(inclusion.is_present());
                    let tx_hash = entry.tx_hash;
                    actions.push(match answer {
                        // The core took it, and its certificate says how.
                        // Its broadcast may have missed this shard, so it
                        // is fetched rather than waited for.
                        Answer::Committed => Action::Fetch(FetchRequest::ExecutionCerts {
                            source_shard: shard,
                            tx_hash,
                            preferred: None,
                            class: None,
                        }),
                        Answer::Present(_) | Answer::Absent(_) => {
                            Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
                        }
                    });
                }
            }
        }
        actions
    }

    /// Fold a verdict the chain committed into the refusal the record
    /// arms are offered from.
    ///
    /// The mirror stays, as a cache in front of the fold rather than the
    /// source of truth it was: what a replica heard broadcast is what
    /// lets it check a claim before the claim commits, and what the chain
    /// committed is what every replica holds afterwards, restart or no.
    /// First write wins, as the fold itself is — a second claim for one
    /// `(transaction, shard)` restates a decision that is already the
    /// chain's.
    fn fold_verdict(&self, verdict: &VerdictClaim) -> Vec<Action> {
        if verdict.shard == self.local_shard || !verdict.refuses() {
            return Vec::new();
        }
        let Some(figures) = self.unresolved.unsettled_leg_figures(verdict.tx_hash) else {
            return Vec::new();
        };
        let refusal = Refusal {
            refused_wt: verdict.anchor_ts,
            deadline: figures.deadline,
            decision: verdict.decision,
            digest: verdict.digest,
        };
        if !self.unresolved.core_holds(verdict.tx_hash, verdict.shard)
            || !self
                .evidence
                .record_refusal(verdict.tx_hash, verdict.shard, refusal)
        {
            return Vec::new();
        }
        let decision = if verdict.decision == TransactionDecision::Aborted {
            TransactionDecision::Aborted
        } else {
            TransactionDecision::Reject
        };
        vec![
            Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved),
            Action::Continuation(ProtocolEvent::TransactionsResolved {
                resolutions: vec![(verdict.tx_hash, TxResolution::CoreDecided(decision))],
            }),
        ]
    }

    /// What this validator can claim about counterparts' chains that no
    /// block has carried yet, in the one order a block carries them,
    /// under the block's cap.
    ///
    /// The proofs its own fetches answered, and the verdicts its own
    /// broadcasts delivered. A verdict is offered only where a leg here
    /// still owes an outcome and the shard that made it is one of the
    /// transaction's core — which is where a verdict licenses a record
    /// at all — and the ledger keeps none that does not, so nothing is
    /// filtered here. That bound is what keeps the vote fence's deferral
    /// rare: it withholds the vote on the whole block, so an unbounded
    /// offer would couple every transaction in a block to the slowest
    /// broadcast on the abort path.
    #[must_use]
    pub fn pending_state_proofs(&self) -> Vec<CounterpartClaim> {
        let proofs = self.fetched.keys().cloned().map(CounterpartClaim::Cells);
        let verdicts = self
            .evidence
            .refusals()
            .into_iter()
            .map(|(tx_hash, shard, refusal)| {
                CounterpartClaim::Verdict(VerdictClaim {
                    shard,
                    tx_hash,
                    anchor_ts: refusal.refused_wt,
                    decision: refusal.decision,
                    digest: refusal.digest,
                })
            });
        proofs
            .chain(verdicts)
            .take(MAX_PROVISIONS_PER_BLOCK)
            .collect()
    }

    /// Drop the bundles no transaction they answered for still needs,
    /// and release every fetch the ledger let go — a question the chain
    /// answered first, or one whose entry is gone — so a counterpart
    /// that never serves the height does not pin the slot.
    fn release_answered_fetches(&mut self) -> Vec<Action> {
        let unresolved = &self.unresolved;
        self.fetched
            .retain(|_, answered| answered.iter().any(|tx_hash| unresolved.contains(*tx_hash)));
        // The one retention rule for what counterparts said: an entry
        // there speaks for a transaction this ledger still owes an
        // outcome for, and the ledger is here.
        self.evidence
            .retain(&|tx_hash| unresolved.contains(tx_hash));
        let ids = self.unresolved.take_released_fetches();
        if ids.is_empty() {
            Vec::new()
        } else {
            vec![Action::AbandonFetch(FetchAbandon::StateProofs { ids })]
        }
    }

    /// Eager-fetch every expected execution cert whose fallback hasn't fired,
    /// independent of block commit. The commit-driven [`Self::check_exec_cert_timeouts`]
    /// stops running when the shard stalls on the missing certs, so a
    /// commit-independent driver (the cleanup timer) flushes through here to
    /// break the deadlock.
    pub fn flush_expected_certs(&mut self) -> Vec<Action> {
        let now_ts = self.committed_ts;
        let awaited = self.awaited_txs();
        self.expected_certs
            .flush_all(&awaited, now_ts)
            .into_iter()
            .map(|(source_shard, tx_hash)| {
                Action::Fetch(FetchRequest::ExecutionCerts {
                    source_shard,
                    tx_hash,
                    preferred: None,
                    class: None,
                })
            })
            .collect()
    }

    /// Check for timed-out expected execution certs and emit fallback requests.
    ///
    /// Called during block commit processing. Returns actions for any certs
    /// that have exceeded the timeout.
    fn check_exec_cert_timeouts(&mut self) -> Vec<Action> {
        let now_ts = self.committed_ts;

        let awaited = self.awaited_txs();
        let fetches = self.expected_certs.check_timeouts(&awaited, now_ts);

        let mut actions = Vec::with_capacity(fetches.len());
        for (source_shard, tx_hash, is_retry) in fetches {
            tracing::info!(
                source_shard = source_shard.inner(),
                tx = %tx_hash,
                retry = is_retry,
                "Execution cert timeout — requesting fallback"
            );
            actions.push(Action::Fetch(FetchRequest::ExecutionCerts {
                source_shard,
                tx_hash,
                preferred: None,
                class: None,
            }));
        }

        self.expected_certs.retain_if_tx_needed(&awaited);
        self.expected_certs.prune_fulfilled(now_ts);

        actions
    }

    /// Transactions an outstanding local tick still holds — the authority on
    /// what this shard is waiting for coverage on. Tick entries are removed
    /// by `finalize` once a tick completes, so a transaction leaves this
    /// set exactly when it stops needing any counterpart's outcome.
    fn awaited_txs(&self) -> HashSet<TxHash> {
        self.ticks
            .ticks_iter()
            .flat_map(|(_, state)| state.tx_hashes().iter().copied())
            .collect()
    }

    /// The subset of [`Self::awaited_txs`] whose settlement waits on
    /// `shard` — what this shard owes us specifically.
    fn awaited_txs_from(&self, shard: ShardId) -> HashSet<TxHash> {
        self.ticks
            .ticks_iter()
            .flat_map(|(_, state)| state.txs_awaiting(shard))
            .collect()
    }

    /// Re-send votes to rotated leaders for ticks that haven't produced an EC.
    ///
    /// Called during block commit processing. When a retry's deadline has
    /// elapsed against the committed QC's weighted timestamp, the registry
    /// returns a [`RetryEffect`] for each fired retry with the new attempt
    /// number; the coordinator resolves the rotated leader via topology
    /// and lifts each effect to `Action::SignAndSendExecutionVote`.
    fn check_vote_retry_timeouts(&mut self, topology_schedule: &TopologySchedule) -> Vec<Action> {
        let effects = self.ticks.check_vote_retry_timeouts(self.committed_ts);
        if effects.is_empty() {
            return Vec::new();
        }

        let mut actions = Vec::with_capacity(effects.len());
        for RetryEffect {
            tick_id,
            attempt,
            block_hash,
            block_height,
            vote_anchor_ts,
            global_receipt_root,
            tx_outcomes,
        } in effects
        {
            // The rotated leader is drawn from the committee seated at the
            // tick's anchor — the one that will verify the EC. Two ways
            // there is nobody to rotate to, and both defer the retry to a
            // later commit rather than resolving a leader: the beacon is
            // behind the anchor, or the anchor resolves a window this
            // shard has already left, where its committee is empty and no
            // vote can reach a quorum anyway.
            let Some(committee) = topology_schedule
                .at(vote_anchor_ts)
                .map(|s| s.consensus_committee_for_shard(self.local_shard).to_vec())
                .filter(|committee| !committee.is_empty())
            else {
                continue;
            };
            let new_leader = tick_leader_at(&tick_id, attempt, &committee);
            tracing::info!(
                tick = %tick_id,
                attempt = attempt.inner(),
                new_leader = new_leader.inner(),
                "Vote retry timeout — re-sending to rotated leader"
            );
            actions.push(Action::SignAndSendExecutionVote {
                block_hash,
                block_height,
                vote_anchor_ts,
                tick_id,
                global_receipt_root,
                tx_outcomes: (*tx_outcomes).clone(),
                leader: new_leader,
            });
        }
        actions
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Block Commit Handling
    // ═══════════════════════════════════════════════════════════════════════════

    /// Handle block committed.
    ///
    /// Runs the variant-agnostic bookkeeping (height bump, timeout checks,
    /// pruning), then dispatches to either `on_live_block_committed` —
    /// which drives fresh execution — or `on_sealed_block_committed` —
    /// which only records tx → tick mappings so late-arriving certs can
    /// route back to the mempool.
    ///
    /// Orchestration order matters. The phases below run in sequence and
    /// depend on earlier phases completing first:
    ///
    /// 1. **Anchor time** — bump `committed_height` and `committed_ts`
    ///    from the QC. Every downstream phase reads these.
    /// 2. **First-commit retro-stamp** — entries buffered pre-first-commit
    ///    carry `WeightedTimestamp::ZERO`; stamp them with the new
    ///    `committed_ts` before timeout checks, otherwise
    ///    `elapsed_since(ZERO)` dwarfs every deadline and triggers a
    ///    fallback-fetch storm.
    /// 3. **Timeout checks** — expected-cert fallbacks and vote retries.
    ///    Read the freshly-bumped `committed_ts`.
    /// 4. **Pruning** — resolved ticks, stale buffered ECs, aged
    ///    conflict-detector provisions. Must follow timeouts so a retry
    ///    fires before the tick it references is pruned away.
    /// 5. **Dispatch** — route to the live or sealed path for block-specific
    ///    work (tick setup + dispatch, or late-cert routing).
    #[instrument(skip(self, certified, topology_schedule), fields(
        height = certified.block().height().inner(),
        block_hash = ?certified.block().hash(),
        tx_count = certified.block().transactions().len(),
        is_live = certified.block().is_live(),
    ))]
    pub fn on_block_committed(
        &mut self,
        topology_schedule: &TopologySchedule,
        certified: &CertifiedBlock,
    ) -> Vec<Action> {
        let block = certified.block();
        let height = block.height();

        // Update committed height + timestamp before anything else — needed
        // for timeout calculations and pruning even when there are no new
        // transactions.
        //
        // "First commit" means the chain has never committed, not that the
        // clock reads zero: a root chain's first blocks genuinely anchor at
        // zero, and their zero is a carriable parent anchor, not a gap.
        let first_commit = self.committed_height == BlockHeight::GENESIS
            && self.committed_ts == WeightedTimestamp::ZERO;
        if height > self.committed_height {
            let own_anchor = certified.block().header().parent_qc().weighted_timestamp();
            // This block's committee anchors on its parent, whose anchor is
            // the one the previous commit carried — or the recovered frontier
            // seeded at construction, which is the same value across a
            // restart. Contiguity is what makes the carry exact — across a
            // gap (a fresh chain's first commit, or a commit landing above a
            // hole) there is no parent anchor to inherit, so the block's own
            // stands in: the same window except when the gap ends on an
            // epoch's first block, and exact again at the next commit. A
            // fresh chain is exact too: its genesis QC carries the chain
            // origin's anchor, which *is* the parent anchor of block one.
            self.committed_committee_anchor_wt =
                if !first_commit && height == self.committed_height.next() {
                    self.committed_ts
                } else {
                    own_anchor
                };
            self.committed_height = height;
            self.committed_ts = own_anchor;
        }
        self.provisioning.advance_clock(self.committed_ts);
        self.gc_settled_sets(topology_schedule);
        // A proof the chain now carries is everybody's: its answers are
        // folded here, and nothing offers it again.
        for claim in block.state_proofs() {
            if let Some(bundle) = claim.cells() {
                self.fetched.remove(bundle);
            }
        }
        let mut actions = self.fold_state_proofs(topology_schedule, block);
        // Every verdict this block carries resolves its transactions,
        // whichever way it went; what is left past every window that
        // could still carry one is nobody's to resolve.
        self.unresolved.release_resolved(block.certificates());
        // What the block writes down about departed shards, before the
        // prune below reads what is still answerable.
        let rebuilt = self
            .unresolved
            .record_abandonment_records(block.abandonment_records());
        for _ in 0..rebuilt {
            record_rebuilt_verdict_entry();
        }
        self.stamp_departures(topology_schedule);
        self.retire_closed_deliveries();
        let unanswerable = self.unresolved.prune(self.committed_ts);
        self.release_unanswerable(&unanswerable);
        actions.extend(self.release_answered_fetches());
        // The committed clock is what opens a leg's deadline, so the
        // cores gone silent past it are asked here.
        actions.extend(self.probe_silent_counterparts(topology_schedule));

        // Timeout checks + pruning run every block, not just commits that
        // carry txs.
        actions.extend(self.check_exec_cert_timeouts());
        actions.extend(self.check_vote_retry_timeouts(topology_schedule));
        self.prune_execution_state();
        self.early.gc_stale_ecs(self.committed_ts);
        self.provisioning.gc_stale_provisions(self.committed_ts);
        // Commit-proof marks age out with the ECs they gate, and the
        // shard coordinator retires them: one mirror, one retirement.
        // Deferred ECs drop at their own deadline.
        // Re-check gate-held finalizations against the advanced schedule:
        // emit any it now resolves, and drop any whose partner it has
        // evicted from every retained window. Runs every block so a settled
        // set that never reconstructs can't pin the buffer; a rejected
        // straddler's transactions stay owed until their deadline.
        actions.extend(self.redrive_gated_finalizations(topology_schedule));

        // Re-broadcast outbound ECs that haven't been ACKed via tick
        // finalization. Driven from the commit cadence so the schedule is
        // deterministic across validators.
        for directive in self.outbound_certs.on_block_committed(self.committed_ts) {
            actions.push(Action::BroadcastExecutionCertificate {
                shard: directive.target_shard,
                certificate: directive.certificate,
                recipients: directive.recipients,
            });
        }

        for (_, tick) in self.ticks.ticks_iter_mut() {
            tick.log_if_overdue(self.committed_ts);
        }

        // Tick fates the block's committed certificates decide. Emitted
        // ahead of the block-specific work, so a tick dispatched below
        // reads the resolved chain.
        for fw in block.certificates().iter() {
            let fw = fw.as_unverified();
            let aborted: BTreeSet<TxHash> = fw
                .tx_decisions()
                .into_iter()
                .filter(|(_, decision)| !matches!(decision, TransactionDecision::Accept))
                .map(|(tx_hash, _)| tx_hash)
                .collect();
            // The members this finalization speaks for, which is one
            // half of its tick rather than the whole of it.
            let members: BTreeSet<TxHash> = fw.tx_hashes().collect();
            self.record_tick_resolution(
                fw.tick_id(),
                TickResolution::Settled {
                    height,
                    members,
                    aborted,
                },
            );
        }
        actions.extend(self.drain_ready_tick_resolutions());

        match block {
            Block::Live {
                header,
                transactions,
                certificates,
                provisions,
                ..
            } => actions.extend(self.on_live_block_committed(
                topology_schedule,
                block.hash(),
                header,
                transactions,
                certificates,
                provisions,
            )),
            Block::Sealed {
                header,
                transactions,
                ..
            } => actions.extend(self.on_sealed_block_committed(
                topology_schedule,
                header,
                transactions,
            )),
        }

        actions
    }

    /// Live path: still within the cross-shard execution window. Proposer
    /// broadcasts provisions, setup+dispatch runs for the block's txs, and
    /// inline provisions are applied so newly-created ticks can transition
    /// to `Provisioned` immediately.
    fn on_live_block_committed(
        &mut self,
        topology_schedule: &TopologySchedule,
        block_hash: BlockHash,
        header: &BlockHeader,
        transactions: &[Arc<Verifiable<Transaction>>],
        certificates: &[Arc<Verifiable<Finalization>>],
        provisions: &[Arc<Verifiable<Provisions>>],
    ) -> Vec<Action> {
        let height = header.height();
        let mut actions = Vec::new();

        // Classification anchors on the block's committee, not the head, so
        // every replica groups its ticks and provisions identically across a
        // reshape boundary.
        let anchored =
            self.classification_committee(topology_schedule, self.committed_committee_anchor_wt);

        // ── Provision broadcasting (proposer only) ─────────────────────
        if self.me == header.proposer() {
            let local_shard = self.local_shard;
            if let Some((requests, shard_recipients)) =
                build_provision_requests(anchored, transactions, certificates, self.me, local_shard)
            {
                actions.push(Action::FetchAndBroadcastProvisions {
                    block_hash,
                    requests,
                    source_shard: local_shard,
                    block_height: height,
                    source_block_ts: header.parent_qc().weighted_timestamp(),
                    shard_recipients,
                });
            }
        }

        let block = CommittingBlock {
            hash: block_hash,
            height,
            ts: self.committed_ts,
        };

        // Everything this commit puts in flight or unblocks, before
        // anything is composed from it: the block's own transactions, and
        // the provisions and engagement echoes its batches carry.
        if !transactions.is_empty() {
            self.register_committed_txs(anchored, block.ts, transactions);
        }
        if !provisions.is_empty() {
            self.apply_committed_provisions(provisions);
        }
        // Every commit, not only one carrying provisions: a bundle that
        // committed before its transaction did is evidence already in
        // hand, and a payer whose wait only ever cleared on a later
        // bundle would never execute.
        self.candidates
            .absorb_engagement_evidence(&self.provisioning);

        // What earlier ticks hold provisionally, and the set this commit's
        // own composition adds to.
        let mut held = self.provisional_cells();
        let (pending, early_votes, members) =
            self.compose_tick(topology_schedule, block, &mut held);
        for vote in early_votes {
            actions.extend(self.dispatch_execution_vote(topology_schedule, vote));
        }
        // The members that just gained an assignment: a counterpart's
        // certificate that arrived while they waited has a tick to route
        // to now, and nothing else will offer it one.
        if !members.is_empty() {
            actions.extend(self.replay_early_attestations(topology_schedule, &members));
        }
        if let Some(pending) = pending {
            tracing::debug!(
                height = height.inner(),
                members = pending.requests.len(),
                "Dispatching this commit's tick"
            );
            self.pending_ticks.push_back(pending);
        }
        actions.extend(self.dispatch_next_tick());

        actions
    }

    /// Escalate any divergence a tick has just latched.
    ///
    /// A wrong tick output is not one tick's problem to sit out. Under
    /// chaining it is the baseline every later tick reads, so a tick that
    /// declined to finalize would leave the node quietly producing
    /// receipts nobody else agrees with, for as long as its drain lasts.
    ///
    /// There is nothing to repair: re-execution is deterministic and
    /// reproduces the same disagreement. The tick named here is the first
    /// one whose output diverged, which chained baselines make exact — a
    /// later tick reading a wrong baseline would have diverged too, so
    /// the earliest report is the origin.
    fn escalate_divergence(&mut self) {
        let mut earliest: Option<(BlockHeight, TickId, Divergence)> = None;
        for (tick_id, tick) in self.ticks.ticks_iter_mut() {
            let Some(divergence) = tick.take_divergence() else {
                continue;
            };
            let height = tick_id.block_height();
            if earliest.as_ref().is_none_or(|(seen, ..)| height < *seen) {
                earliest = Some((height, *tick_id, divergence));
            }
        }
        let Some((tick, tick_id, divergence)) = earliest else {
            return;
        };
        tracing::error!(
            shard = %self.local_shard,
            tick = tick.inner(),
            tick = %tick_id,
            block_hash = ?divergence.block_hash,
            local_root = ?divergence.local_root,
            ec_root = ?divergence.ec_root,
            "Local execution diverged from the quorum. Every tick from this \
             one on reads a baseline nobody else agrees with. Rebuild \
             required: restore from a state snapshot or resync."
        );
        panic!(
            "BFT CRITICAL: local execution diverged at tick {} (tick {tick_id}): \
             voted receipt root {:?}, committee certified {:?}. Deterministic \
             re-execution reproduces it — operator intervention required.",
            tick.inner(),
            divergence.local_root,
            divergence.ec_root,
        );
    }

    /// The cells unresolved cross-shard legs hold provisionally.
    ///
    /// Rebuilt per commit from `ticked`, which is small: one entry per
    /// tick whose fate has not committed, and the drain budget bounds how
    /// many that can be.
    fn provisional_cells(&self) -> ProvisionalCells {
        let mut cells = ProvisionalCells::default();
        for ticked in self.ticked.values() {
            cells.claim(&ticked.provisional_claims);
        }
        cells
    }

    /// The transactions this commit's tick attests `Aborted`.
    ///
    /// The trigger is the transaction's own deadline: the last block that
    /// could have included it anywhere, plus the longest a cross-shard
    /// transaction can take to finalize. Both figures are its own and the
    /// clock is the committed weighted timestamp, so no replica can reach
    /// the deadline at a frontier where another has not — which is what
    /// lets a committee sign a verdict about it.
    ///
    /// Past it, only the transactions no shard can still settle, which is
    /// what makes abandoning one this shard's decision alone to take.
    ///
    /// Two narrowings that used to sit here are answered elsewhere. An
    /// assembled settlement needs none: its tick is lower, so the store
    /// offers it first and the duplicate-resolution rule refuses the
    /// abort, after which the ledger releases the transaction. And a
    /// transaction a terminating counterpart may have settled is the
    /// fence's question, which [`Self::fence_pairs`] is what lets the
    /// fence ask about an abandonment at all.
    fn abandonable(&self, composing: TickId) -> Vec<Abandonable> {
        self.unresolved
            .past_deadline(self.committed_ts)
            .into_iter()
            .filter(|entry| self.beyond_every_shard(composing, entry.tx_hash))
            .collect()
    }

    /// Drop a tick that can no longer speak for a member being abandoned,
    /// releasing the transactions it holds to their own deadlines.
    fn discard_tick(&mut self, tick_id: TickId) {
        let counts = self.ticks.discard_tick(&tick_id);
        self.ticked.remove(&tick_id);
        // Its finalization goes with it: a proposer offering one for a
        // member abandoned here would be refused by every voter, and
        // would keep offering it.
        self.finalized.remove_tick(&tick_id);
        tracing::info!(
            tick = %tick_id,
            released = counts.assignments,
            "Discarded a tick holding an abandoned member"
        );
    }

    /// Whether no shard can still settle `tx_hash`.
    ///
    /// A settlement needs a certificate from every shard party to the
    /// transaction, so it takes two things to put one out of reach: no
    /// certificate of ours for a counterpart to combine with its own, or
    /// no counterpart in a position to combine one.
    ///
    /// A tick holding the transaction is speaking for it and is left to.
    /// The one composing now is about to attest it, and its verdict can
    /// carry a charge an abandonment cannot — a payer's leg admitted at
    /// its engagement deadline being exactly that. An earlier one can
    /// still close its coverage, unless a record says the counterpart it
    /// waits on left the transaction unsettled, in which case it never
    /// will.
    ///
    /// With no tick speaking for it, what decides is whether a certificate
    /// of ours is out where a counterpart could settle against it. The
    /// account answers that, not the tick registry: the certificate
    /// outlives the tick that produced it — a discard drops the tick, a
    /// restart loses it — and a shard reading the registry would take the
    /// tick's absence for the certificate's and abandon a transaction its
    /// counterpart can still settle.
    ///
    /// That a counterpart has left is not that answer. A shard can settle
    /// its half and then depart, so its departure and its silence are
    /// different facts, and only the second puts a settlement out of
    /// reach. The committed record is what establishes the second, so it
    /// is what licenses spending a tick here — composing on the departure
    /// alone would discard a tick whose settlement had already closed, and
    /// the fence would then refuse the abort that replaced it, tearing the
    /// transaction across the two shards.
    fn beyond_every_shard(&self, composing: TickId, tx_hash: TxHash) -> bool {
        match self.ticks.tick_assignment(tx_hash) {
            Some(tick_id) if tick_id == composing => false,
            // A delivery's tick is not left to past the close: it awaits
            // nobody, so nothing could still close its coverage, and the
            // claim it would write past the close is one the crossing's
            // issuer may already have proved absent and taken back.
            Some(_) => {
                self.unresolved.is_unsettled_by_departed(tx_hash)
                    || self.unresolved.is_delivery(tx_hash)
            }
            None => {
                !self.unresolved.is_certified(tx_hash) || self.no_counterpart_can_settle(tx_hash)
            }
        }
    }

    /// Whether no counterpart is in a position to settle `tx_hash` against
    /// a certificate of ours: none is party to it, so there is no
    /// certificate but ours to combine with, or a committed record says
    /// the one that was left it unsettled.
    fn no_counterpart_can_settle(&self, tx_hash: TxHash) -> bool {
        !self.unresolved.reaches_beyond(tx_hash)
            || self.unresolved.is_unsettled_by_departed(tx_hash)
    }

    /// The records this shard has evidence for and has not yet written
    /// down — what each departed counterpart left of its business here.
    ///
    /// Composed from the settled sets, which is what bounds when this can
    /// speak at all: a set is acquired once the departed shard's terminal
    /// roots are attested and dropped at its evidence expiry, so a record
    /// is only ever offered while the evidence for it is readable, which
    /// is the same window every voter can check it in. Absence from a set
    /// is proof rather than ignorance — the set is complete and
    /// beacon-attested — so a transaction of ours it does not name is one
    /// that shard never settled and now never will.
    ///
    /// Bounded by [`MAX_UNSETTLED_PER_BLOCK`], one budget across every
    /// departure, with the remainder left for the next block.
    ///
    /// Ascending by shard, which is the one order a block may carry them
    /// in.
    #[must_use]
    pub fn pending_abandonment_records(&self) -> Vec<AbandonmentRecord> {
        let mut budget = MAX_UNSETTLED_PER_BLOCK;
        // One record per shard and arm, ascending: departures first,
        // since a departure covers everything the shard was party to,
        // refusals then for the shards still running, and the arms in
        // the order the block carries them.
        let mut records: BTreeMap<(ShardId, u8), AbandonmentRecord> = BTreeMap::new();
        // The sets are a hash map, so the shards are walked in sorted
        // order rather than its own: which departures the budget reaches
        // must not turn on a per-process iteration order.
        self.evidence.with_settled(|sets| {
            let mut shards: Vec<ShardId> = sets.keys().copied().collect();
            shards.sort_unstable();
            for shard in shards {
                if budget == 0 || records.len() == MAX_ABANDONMENT_RECORDS_PER_BLOCK {
                    break;
                }
                let settled = &sets[&shard];
                let mut unsettled = self.unresolved.outstanding_with(shard, settled.terminal_wt);
                unsettled.retain(|entry| !settled.txs.contains(&entry.tx_hash));
                unsettled.truncate(budget);
                if unsettled.is_empty() {
                    continue;
                }
                budget -= unsettled.len();
                let record = AbandonmentRecord::departed(shard, settled.terminal_wt, unsettled);
                records.insert((shard, record.evidence().discriminant()), record);
            }
        });
        // Mirrored refusals, grouped by shard and then by the anchor the
        // core refused at: one anchor per shard per block, earliest
        // first, because a record spanning two anchors satisfies the
        // fence's equality check for neither.
        let refused = self
            .evidence
            .refusals()
            .into_iter()
            .map(|(tx_hash, shard, refusal)| (tx_hash, shard, refusal.refused_wt));
        self.offer_at_earliest_anchor(
            &mut records,
            &mut budget,
            refused,
            AbandonmentRecord::refused,
        );
        // Answers the chain committed, grouped the same way, one arm per
        // kind of counterpart: a record states the one anchor every name
        // in it was proved at, and a core's committed cell absent, a
        // delivery's claim absent and a one-shard core's claim absent are
        // different claims held to different floors.
        for (probed, record) in [
            (
                Probed::Core,
                AbandonmentRecord::unclaimed as fn(_, _, _) -> _,
            ),
            (Probed::Delivery, AbandonmentRecord::lapsed),
            (Probed::Claim, AbandonmentRecord::untaken),
        ] {
            let absent = self
                .evidence
                .absences(probed)
                .into_iter()
                .map(|(tx_hash, shard, absence)| (tx_hash, shard, absence.probed_wt));
            self.offer_at_earliest_anchor(&mut records, &mut budget, absent, record);
        }
        // Proved claims, under the one settling arm: a consumer's claim
        // of a crossing a leg here issued, whichever cell it was proved
        // at, licensing the record's retirement rather than a reclaim.
        if CLAIMED_RECORDS_OFFERED {
            let claimed = self
                .evidence
                .presences()
                .into_iter()
                .map(|(tx_hash, shard, presence)| (tx_hash, shard, presence.probed_wt));
            self.offer_at_earliest_anchor(
                &mut records,
                &mut budget,
                claimed,
                AbandonmentRecord::claimed,
            );
        }
        records.into_values().collect()
    }

    /// Offer one record per shard from `mirrored`, at the shard's
    /// earliest anchor, for the shards `records` does not yet name and
    /// the names no record covers yet, under the shared budget.
    fn offer_at_earliest_anchor(
        &self,
        records: &mut BTreeMap<(ShardId, u8), AbandonmentRecord>,
        budget: &mut usize,
        mirrored: impl IntoIterator<Item = (TxHash, ShardId, WeightedTimestamp)>,
        record: fn(ShardId, WeightedTimestamp, Vec<UnsettledTx>) -> AbandonmentRecord,
    ) {
        let mut anchored: BTreeMap<ShardId, BTreeMap<WeightedTimestamp, Vec<UnsettledTx>>> =
            BTreeMap::new();
        for (tx_hash, shard, anchor) in mirrored {
            if let Some(figures) = self.unresolved.unsettled_leg_figures(tx_hash) {
                anchored
                    .entry(shard)
                    .or_default()
                    .entry(anchor)
                    .or_default()
                    .push(figures);
            }
        }
        for (shard, anchors) in anchored {
            if *budget == 0 || records.len() == MAX_ABANDONMENT_RECORDS_PER_BLOCK {
                break;
            }
            let Some((anchor, mut unsettled)) = anchors.into_iter().next() else {
                continue;
            };
            // A departure answers for everything the shard was party
            // to, so nothing else is offered beside one; the other
            // arms are different answers about different names.
            let arm = record(shard, anchor, Vec::new()).evidence().discriminant();
            if records.contains_key(&(shard, 0)) || records.contains_key(&(shard, arm)) {
                continue;
            }
            unsettled.truncate(*budget);
            *budget -= unsettled.len();
            records.insert((shard, arm), record(shard, anchor, unsettled));
        }
    }

    /// Let go of what this shard holds against transactions no shard can
    /// settle any more.
    ///
    /// Every counterpart has left and every settled set that could have
    /// spoken for them has stopped reading, so the tick holding one will
    /// never close its coverage — no certificate is coming to close it
    /// with. Its provisional claims are held against writes that will
    /// never apply, and a later transaction reaching those cells is
    /// waiting on nothing.
    ///
    /// Discarding here is not the discard a verdict makes. A verdict is
    /// composed while counterparts are live and spends a tick that might
    /// still have settled; this runs only once none of them can answer,
    /// which is the same condition that makes the transaction's own fate
    /// unreachable. Nothing that could still settle is destroyed, because
    /// by then nothing can.
    ///
    /// Each of these is also a reservation the drain never gets back —
    /// only a committed certificate returns one, and by here none is
    /// coming. Counted by cause, because the drain's baseline rises with
    /// them and a shard that accumulates enough admits nothing at all.
    fn release_unanswerable(&mut self, unanswerable: &[Unanswerable]) {
        for entry in unanswerable {
            record_unresolvable_tx(if entry.covered_by_record {
                "record_covered"
            } else {
                "no_record"
            });
            if let Some(tick_id) = self.ticks.tick_assignment(entry.tx_hash) {
                tracing::info!(
                    tx = %entry.tx_hash,
                    tick = %tick_id,
                    covered_by_record = entry.covered_by_record,
                    "Releasing a strand whose counterparts have all fallen silent"
                );
                self.discard_tick(tick_id);
            }
        }
    }

    /// Record where each departed shard's chain ended, for the entries
    /// whose fate only that shard's settled set can decide.
    ///
    /// Read on every commit, while the schedule still carries the window
    /// that proves the terminal — the account outlives that window, and a
    /// departure it never recorded reads afterwards as a counterpart that
    /// never left. Re-run rather than gated on first sight, because the
    /// expiry is not knowable at the cut: the beacon stamps the handoff
    /// complete some epochs later, and the ledger's entry fills in on the
    /// first commit after the stamp lands.
    fn stamp_departures(&mut self, topology_schedule: &TopologySchedule) {
        for (shard, cut) in topology_schedule.departures_at(self.committed_ts) {
            if shard != self.local_shard {
                self.unresolved.record_terminal(
                    shard,
                    cut,
                    topology_schedule.handoff_evidence_expiry(shard),
                );
            }
        }
        // A departure held open is asked about by name on every commit,
        // since the schedule lists it only while a retained window
        // carries the shard and the stamp lands on the head's boundary
        // record, which outlives that window. One whose evidence the
        // schedule no longer reads at all closes now — the same reading
        // the settled sets are dropped on — so an entry a record covers
        // against it retires with the set that could have answered.
        let now = self.committed_ts;
        for shard in self.unresolved.unstamped_departures() {
            if let Some(expiry) = topology_schedule.handoff_evidence_expiry(shard) {
                self.unresolved.stamp_terminal(shard, expiry);
            } else if !topology_schedule.terminal_evidence_readable(shard, now) {
                self.unresolved.stamp_terminal(shard, now);
            }
        }
    }

    /// Record a tick's fate for the tick chain.
    ///
    /// A tick with no tick entry — never dispatched, or committed by a
    /// shard already past its execution window — resolves nothing. The
    /// rest are held until their tick has appended: a block carrying a
    /// certificate can commit while the tick that executed the tick is
    /// still queued, and resolving against a chain that has never seen
    /// the tick would drop the promotion.
    fn record_tick_resolution(&mut self, tick_id: &TickId, resolution: TickResolution) {
        let Some(ticked) = self.ticked.get(tick_id) else {
            return;
        };
        // The claims clear with the fate rather than with the promotion:
        // both the commit path and the tick-completion pump emit pending
        // resolutions before dispatching, so the chain is always at least
        // as resolved as the claim set says by the time a later tick runs.
        //
        // Only the half carrying the legs clears them. A determined
        // member holds no cell — its writes are readable from the append
        // — so its own half settling releases nothing.
        let releases_claims = match &resolution {
            TickResolution::Settled { members, .. } => {
                ticked.legs.iter().all(|leg| members.contains(leg))
            }
            TickResolution::Abandoned { .. } => true,
        };
        if releases_claims {
            self.ticked.remove(tick_id);
        }
        self.pending_tick_resolutions
            .push((*tick_id, tick_id.block_height(), resolution));
    }

    /// Emit every buffered resolution whose tick is now on the chain.
    fn drain_ready_tick_resolutions(&mut self) -> Vec<Action> {
        let last = self.last_completed_tick;
        let mut ready: Vec<(TickId, TickResolution)> = Vec::new();
        self.pending_tick_resolutions
            .retain(|(tick_id, tick, resolution)| {
                if *tick <= last {
                    ready.push((*tick_id, resolution.clone()));
                    false
                } else {
                    true
                }
            });
        if ready.is_empty() {
            return Vec::new();
        }
        vec![Action::ResolveTicks { resolutions: ready }]
    }

    /// Dispatch the queued tick at the head, unless one is already in
    /// flight. Ticks execute serially: each output is the next tick's
    /// baseline, so the next dispatch waits for the previous
    /// `ExecutionBatchCompleted` — by which point the handler has
    /// appended the output to the tick chain.
    fn dispatch_next_tick(&mut self) -> Vec<Action> {
        if self.tick_in_flight {
            return Vec::new();
        }
        let Some(head) = self.pending_ticks.front() else {
            return Vec::new();
        };
        // Running a member whose code this node lacks would reach the
        // engine's no-code refusal while every replica holding the bytes
        // settles it — one tick, two receipt roots. Waiting is the whole
        // of the fix: the fetch heals, and the tick that runs then is the
        // tick that was composed now. Ticks are serial, so this shard's
        // execution stops here until the bytes land — the trade a
        // withheld artifact is meant to draw, liveness rather than a
        // fork.
        if head.runs_any_of(&self.missing_packages) {
            tracing::debug!(
                shard = %self.local_shard,
                tick = %head.tick,
                "Holding a tick whose members run code this node has not fetched"
            );
            return Vec::new();
        }
        let Some(tick) = self.pending_ticks.pop_front() else {
            return Vec::new();
        };
        self.tick_in_flight = true;
        vec![Action::ExecuteTransactions {
            tick: tick.tick,
            tick_ts: tick.tick_ts,
            env: tick.env,
            requests: tick.requests,
        }]
    }

    /// Sealed path: past the cross-shard execution window. Ticks will
    /// finalize from the already-aggregated cert + receipts included
    /// downstream, so we skip `TickState` creation, dispatch, and vote
    /// tracking. Only the tx → tick mapping is recorded (plus any early
    /// ECs replayed) so a late-arriving cert still routes back to each
    /// tx for mempool terminal-state bookkeeping.
    fn on_sealed_block_committed(
        &mut self,
        topology_schedule: &TopologySchedule,
        header: &BlockHeader,
        transactions: &[Arc<Verifiable<Transaction>>],
    ) -> Vec<Action> {
        if transactions.is_empty() {
            return Vec::new();
        }
        let anchored =
            self.classification_committee(topology_schedule, self.committed_committee_anchor_wt);
        self.register_sealed_assignments(anchored, header.height(), transactions);
        let tx_hashes: Vec<TxHash> = transactions.iter().map(|tx| tx.hash()).collect();
        self.replay_early_attestations(topology_schedule, &tx_hashes)
    }

    /// Replay buffered certificates for transactions that have just
    /// gained a tick assignment.
    ///
    /// Driven by assignment rather than by commit, because those are no
    /// longer the same moment: a member waits in the candidate pool until
    /// a tick can take it, and a counterpart's certificate arriving in
    /// that window has nowhere to route. Composition is what gives it
    /// one, so composition is what replays.
    fn replay_early_attestations(
        &mut self,
        topology_schedule: &TopologySchedule,
        tx_hashes: &[TxHash],
    ) -> Vec<Action> {
        let ecs_to_replay = self.early.drain_ecs_for_txs(tx_hashes);
        if ecs_to_replay.is_empty() {
            return Vec::new();
        }
        tracing::debug!(
            count = ecs_to_replay.len(),
            "Replaying early tick attestations for newly committed txs"
        );
        let mut actions = Vec::new();
        for ec in &ecs_to_replay {
            actions.extend(self.handle_attestation(topology_schedule, ec));
        }
        actions
    }

    /// What a committed block's finalizations settle about the
    /// transactions they name, for the mempool's status of each: the
    /// ledger's reading, taken before the block releases the entries.
    #[must_use]
    pub fn resolutions_of(
        &self,
        finalizations: &[Arc<Verifiable<Finalization>>],
    ) -> Vec<(TxHash, TxResolution)> {
        self.unresolved.resolutions_of(finalizations)
    }

    /// Register tx → tick assignments for a `Sealed` block without any of
    /// the execution-side state setup (`TickState`, vote tracker, conflict
    /// detector, required-provision tracking). The block's ticks are
    /// already settled; we only need the mapping so a future cert can
    /// route back to the tx for mempool terminal-state bookkeeping.
    fn register_sealed_assignments(
        &mut self,
        topology_snapshot: &TopologySnapshot,
        block_height: BlockHeight,
        transactions: &[Arc<Verifiable<Transaction>>],
    ) {
        let _ = topology_snapshot;
        let tick_id = TickId::new(self.local_shard, block_height);
        for tx in transactions {
            self.ticks.assign_tx(tx.hash(), tick_id);
            self.unresolved.certify(tx.hash());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 5: Finalization
    // ═══════════════════════════════════════════════════════════════════════════

    /// Mirror what `ec` says a core decided of the transactions legs
    /// here issued for: each new refusal is handed to the vote fence,
    /// and each verdict — a refusal, or a success once every core shard
    /// has given one — to the mempool, whose terminal for a leg is the
    /// core's verdict.
    ///
    /// Only a core's word counts: a leg elsewhere failing is its own
    /// outcome, not the transaction's verdict, and the leg entry names
    /// whose word it takes. First write wins per shard: a certificate is
    /// one per shard per transaction, so a second copy is a re-broadcast
    /// of the same word.
    fn mirror_verdicts(&mut self, ec: &Arc<Verified<ExecutionCertificate>>) -> Vec<Action> {
        let shard = ec.shard_id();
        if shard == self.local_shard {
            return Vec::new();
        }
        let mut actions = Vec::new();
        let mut resolutions = Vec::new();
        for outcome in ec.tx_outcomes() {
            let tx_hash = outcome.tx_hash();
            if !self
                .unresolved
                .leg_core(tx_hash)
                .is_some_and(|core| core.contains(&shard))
            {
                continue;
            }
            if matches!(outcome.outcome(), ExecutionOutcome::Succeeded { .. }) {
                if self.unresolved.record_acceptance(tx_hash, shard) {
                    resolutions.push((
                        tx_hash,
                        TxResolution::CoreDecided(TransactionDecision::Accept),
                    ));
                }
                continue;
            }
            let Some(figures) = self.unresolved.unsettled_leg_figures(tx_hash) else {
                continue;
            };
            let decision = if outcome.is_aborted() {
                TransactionDecision::Aborted
            } else {
                TransactionDecision::Reject
            };
            let refusal = Refusal {
                refused_wt: ec.vote_anchor_ts(),
                deadline: figures.deadline,
                decision,
                digest: ec.attested_digest(),
            };
            if self.evidence.record_refusal(tx_hash, shard, refusal) {
                actions.push(Action::Continuation(
                    ProtocolEvent::CounterpartEvidenceObserved,
                ));
                resolutions.push((tx_hash, TxResolution::CoreDecided(decision)));
            }
        }
        if !resolutions.is_empty() {
            actions.push(Action::Continuation(ProtocolEvent::TransactionsResolved {
                resolutions,
            }));
        }
        actions
    }

    /// Handle a tick-level attestation (execution certificate) from any shard.
    ///
    /// A remote EC's `tick_id` reflects the remote shard's tick composition,
    /// which differs from the local shard's. A single remote EC may contain
    /// outcomes for transactions in MULTIPLE local ticks.
    ///
    /// Routing: iterate `tx_outcomes` → look up local tick via `tick_assignments` →
    /// feed the EC to each affected local tick tracker. `tx_hashes` without a
    /// local assignment are buffered (or kept buffered) via `pending_routing`
    /// until their blocks commit; routed `tx_hashes` are cleared from the
    /// pending set, dropping the EC entirely once fully routed.
    fn handle_attestation(
        &mut self,
        topology_schedule: &TopologySchedule,
        ec: &Arc<Verified<ExecutionCertificate>>,
    ) -> Vec<Action> {
        // What a core says of a transaction a leg here issued for is
        // read before routing: the leg's tick settled long ago, so the
        // certificate routes nowhere, and the refusal is the one thing
        // in it this shard still has a use for.
        let mut actions = self.mirror_verdicts(ec);

        let routing = self.ticks.classify_attestation(ec);

        self.early.clear_routed(ec, &routing.routed_tx_hashes);
        self.early.buffer_ec(ec, &routing.unrouted_tx_hashes);

        if routing.affected_ticks.is_empty() {
            return actions;
        }

        // Feed the EC to each affected local tick. Completion requires both
        // the local EC and all remote shards' coverage (aborted txs are
        // terminal-covered). Once `local_ec_emitted` is true, every tx
        // already has an outcome and a matching receipt in the cache.
        for tick_id in &routing.affected_ticks {
            let Some(tick) = self.ticks.get_tick_mut(tick_id) else {
                continue;
            };
            tick.add_execution_certificate(Arc::clone(ec));
            actions.extend(self.finalize(topology_schedule, tick_id));
        }
        // The other: an admitted local EC that contradicts the vote this
        // validator already cast.
        self.escalate_divergence();
        actions
    }

    /// Finalize a tick: build the [`Finalization`], then admit it or hold
    /// it at the split-boundary gate.
    ///
    /// Called when the tick's local EC is present and every non-aborted tx is
    /// covered by all participating shards.
    fn finalize(&mut self, topology_schedule: &TopologySchedule, tick_id: &TickId) -> Vec<Action> {
        let mut halves: Vec<Finalization> = Vec::new();
        let Some(tick) = self.ticks.get_tick_mut(tick_id) else {
            return vec![];
        };
        // Determined first, and the order matters: the two halves of one
        // tick came out of a single batch fold, so a receipt in the second
        // states an absolute computed on top of the first.
        halves.extend(tick.take_determined_finalization());
        halves.extend(tick.take_legs_finalization());
        if halves.is_empty() {
            return vec![];
        }
        let spoken = tick.has_spoken();

        if spoken {
            // Every leg's settlement needed every participating shard's
            // certificate, which means each of them executed this tick —
            // strong evidence they also received our outbound EC (or are
            // about to). Drop the re-broadcast tracker entry to stop
            // wasting bandwidth. The tick itself stays until a committed
            // block resolves its members.
            self.outbound_certs.on_tick_finalized(tick_id);
        }

        // Local-finalization gate produces `Verified<Finalization>`; lift
        // into the `Block::Live.certificates` transport shape once so the
        // store, the admission event, and any downstream `PendingBlock`
        // entry share the same `Arc` without further per-consumer cloning.
        let mut actions = Vec::new();
        for half in halves {
            let finalized_arc = Arc::new(Verified::<Finalization>::seal(half).into());
            actions.extend(self.emit_or_gate_finalized(topology_schedule, finalized_arc));
        }
        actions
    }

    /// The `(shard, tx_hash, claim)` triples the split-boundary fence asks
    /// about.
    ///
    /// A settlement carries its participants in the certificates it
    /// collected, so reading them off is enough. An abandonment carries
    /// only this shard's certificate, because an abort is dominant and
    /// needs no counterpart's verdict — and that certificate's outcome
    /// names the counterparts the member awaited and never heard from,
    /// which is what the fence asks their settled sets about. A member
    /// that awaited nobody — a leg, a core answering alone, a reclaim —
    /// makes no claim at all: its verdict is its own, settled on this
    /// certificate, and no counterpart's set can contradict it.
    ///
    /// The two ask opposite questions of the same set, which is what
    /// [`TxClaim`] carries: a settlement needs the terminating partner to
    /// have settled its own half, an abandonment needs it not to have.
    /// One tick can hold both, so the claim is per transaction rather
    /// than per finalization.
    fn fence_pairs(&self, fw: &Finalization) -> Vec<(ShardId, TxHash, TxClaim)> {
        let mut pairs: Vec<(ShardId, TxHash, TxClaim)> = Vec::new();
        let mut attested_remotely: HashSet<TxHash> = HashSet::new();
        for ec in fw.execution_certificates() {
            let shard = ec.shard_id();
            for outcome in ec.tx_outcomes() {
                if shard != self.local_shard {
                    attested_remotely.insert(outcome.tx_hash());
                }
                pairs.push((shard, outcome.tx_hash(), TxClaim::Settled));
            }
        }
        for outcome in fw.local_ec().tx_outcomes() {
            let tx_hash = outcome.tx_hash();
            if attested_remotely.contains(&tx_hash) {
                continue;
            }
            // A committed record already answered for this one, in a form
            // that does not expire with the set it was read from. Putting
            // the question again would let the set's own horizon refuse a
            // verdict the chain has already established is safe.
            if self.unresolved.is_unsettled_by_departed(tx_hash) {
                continue;
            }
            pairs.extend(
                outcome
                    .counterparts()
                    .iter()
                    .map(|&s| (s, tx_hash, TxClaim::Abandoned)),
            );
        }
        pairs
    }

    /// Admit a freshly built finalization downstream, or withhold it at
    /// the split-boundary gate so we never produce a tick the vote fence
    /// would reject.
    ///
    /// `Pass` records the tick and emits the admission event (one event
    /// covers both the shard consensus subscriber and the `io_loop`
    /// serving cache). `Defer` buffers it until the terminating shard's
    /// settled set resolves it or its scheduled termination clears
    /// ([`Self::redrive_gated_finalizations`]).
    /// `Reject` drops it — the tick names a past-terminal shard that
    /// didn't settle it, so it must never be produced. Nothing here
    /// resolves its transactions; they stay owed, and the tick at their
    /// deadline abandons them.
    fn emit_or_gate_finalized(
        &mut self,
        topology_schedule: &TopologySchedule,
        finalized_arc: Arc<Verifiable<Finalization>>,
    ) -> Vec<Action> {
        let tick_id = *finalized_arc.tick_id();
        let verdict = {
            // Whether a shard is past-terminal is asked at the committed
            // frontier, which is what a node-local caller reads it at.
            let outcomes = self.fence_pairs(finalized_arc.as_unverified());
            self.evidence.with_settled(|settled| {
                settled_set_verdict(
                    settled,
                    topology_schedule,
                    self.local_shard,
                    self.committed_ts,
                    outcomes,
                )
            })
        };
        match verdict {
            SettledSetVerdict::Pass => {
                self.finalized.insert(tick_id, Arc::clone(&finalized_arc));
                vec![Action::Continuation(ProtocolEvent::FinalizationsAdmitted {
                    finalizations: vec![finalized_arc],
                })]
            }
            SettledSetVerdict::Defer => {
                // Hold until evidence resolves the tick: the partner's
                // settled set reconstructs (pass or reject on membership),
                // its scheduled termination clears (pass), or the schedule
                // evicts it from every retained window (reject). Never
                // dropped on a clock — a deadline verdict here can
                // contradict a settlement the partner already committed.
                self.gated_finalized
                    .insert(finalized_arc.receipt_hash(), finalized_arc);
                vec![]
            }
            SettledSetVerdict::Reject => {
                // The partner never settled this half, so it must never be
                // produced. Taking it was one-shot, so the tick will not
                // offer it again — and the members it named are left with
                // a tick that has stopped speaking for them. Releasing
                // their assignments hands them back to the deadline path,
                // which is the only thing that can still resolve them.
                for tx_hash in finalized_arc.tx_hashes() {
                    self.ticks.remove_assignment(tx_hash);
                }
                vec![]
            }
        }
    }

    /// Record a past-terminal shard's settled-transaction set for the finalize
    /// gate (mirrors the shard coordinator's fence feed). Pair with
    /// [`Self::redrive_gated_finalizations`] to release ticks that the
    /// gate held while the set was unknown.
    ///
    /// Also arms the fallback fetch: what the partner says it settled and
    /// we are still waiting on is exactly the certificates it owes us, and
    /// the header that first named them may never have reached us.
    ///
    /// Returns the actions of replaying the shard's commit-proof-deferred
    /// certificates: the set stands in for the proof of everything it
    /// names ([`Self::settled_set_admits`]), so a certificate parked on a
    /// proof the departed chain can no longer supply goes back through
    /// the gate now.
    pub fn record_settled_txs(
        &mut self,
        topology_schedule: &TopologySchedule,
        shard: ShardId,
        settled: SettledTxSet,
    ) -> Vec<Action> {
        let now_ts = self.committed_ts;

        let owed: Vec<TxHash> = self
            .awaited_txs_from(shard)
            .into_iter()
            .filter(|tx_hash| settled.txs.contains(tx_hash))
            .filter(|tx_hash| !self.expected_certs.is_fulfilled(shard, *tx_hash))
            .collect();
        for tx_hash in owed {
            self.expected_certs.register(shard, tx_hash, now_ts);
        }

        // What this shard's ledger says the departed shard was party to,
        // taken beside the set: a departure record may name only these,
        // and the fence reads it from the same mirror.
        let parties = self.unresolved.party_to(shard, settled.terminal_wt);
        self.evidence.record_settled(shard, settled, parties);

        let deferred = self.unproven_ecs.drain_shard(shard);
        let mut actions = Vec::new();
        for cert in deferred {
            actions.extend(self.on_execution_certificate(topology_schedule, cert));
        }
        actions
    }

    /// Whether `shard`'s settled set stands in for a commit proof of this
    /// certificate's source block.
    ///
    /// Membership means the transaction's certificate committed in the
    /// departed chain at or before its terminal, and the set itself was
    /// verified against the beacon-attested terminal root — a stronger
    /// statement than a commit proof of one source block, which is
    /// exactly what a departed chain can no longer supply. Every outcome
    /// must be named: one outside the set is a verdict the departed
    /// shard never settled, and a certificate naming nothing gives the
    /// set nothing to vouch for.
    fn settled_set_admits(&self, shard: ShardId, cert: &Verifiable<ExecutionCertificate>) -> bool {
        self.evidence.with_settled(|sets| {
            sets.get(&shard).is_some_and(|settled| {
                let outcomes = cert.tx_outcomes();
                !outcomes.is_empty()
                    && outcomes
                        .iter()
                        .all(|outcome| settled.txs.contains(&outcome.tx_hash()))
            })
        })
    }

    /// Drop settled sets past their evidence window. Past it the gate
    /// rejects any outcome naming the shard regardless of the set, so
    /// retaining it only leaks memory.
    fn gc_settled_sets(&self, topology_schedule: &TopologySchedule) {
        let now = self.committed_ts;
        self.evidence
            .retain_departures(&|shard| topology_schedule.terminal_evidence_readable(shard, now));
    }

    /// Re-check every gate-held finalization against the current settled
    /// sets and schedule: emit the ones now resolvable, drop the ones now
    /// known unsettled or schedule-evicted, and re-hold the rest.
    pub fn redrive_gated_finalizations(
        &mut self,
        topology_schedule: &TopologySchedule,
    ) -> Vec<Action> {
        if self.gated_finalized.is_empty() {
            return Vec::new();
        }
        let gated: Vec<Arc<Verifiable<Finalization>>> = std::mem::take(&mut self.gated_finalized)
            .into_values()
            .collect();
        let mut actions = Vec::new();
        for finalized_arc in gated {
            actions.extend(self.emit_or_gate_finalized(topology_schedule, finalized_arc));
        }
        actions
    }

    /// Admission entry point for fetch-delivered (or otherwise externally
    /// sourced) finalizations.
    ///
    /// Runs the cheap synchronous gates inline (per-EC quorum power and
    /// committee-key resolution) and dispatches signature verification to the
    /// crypto pool via [`Action::VerifyFinalization`]. The matching
    /// [`ProtocolEvent::FinalizationVerified`] feeds
    /// [`Self::on_finalization_verified`], which emits
    /// `Continuation(FinalizationsAdmitted)` only when every EC's
    /// signature passed.
    ///
    /// Without this gate a peer answering a `finalization.request` could
    /// poison `caches.finalization` with a bogus tick we'd re-serve.
    /// Locally produced finalizations bypass this path: `finalize` emits the
    /// same event from a WC built out of already-verified ECs. Synced
    /// blocks are likewise trusted at admission — the QC chain plus the
    /// synced-block apply path's quorum gate established their integrity
    /// upstream.
    #[must_use]
    pub fn admit_finalization(
        &mut self,
        topology_schedule: &TopologySchedule,
        tick: Arc<Verifiable<Finalization>>,
    ) -> Vec<Action> {
        let tick_id = *tick.tick_id();
        // A tick can settle in more than one part, so identity is the
        // finalization's own content — both here and in the fetch this
        // may abandon.
        let id = tick.receipt_hash();

        // Already-finalized short-circuit — a second fetch arrival for a
        // finalization we've already admitted is wasted verification work.
        if self.finalized.get(&id).is_some() {
            tracing::debug!(
                tick = %tick_id,
                "Finalization already in canonical store — skipping verification"
            );
            return Vec::new();
        }

        // In-flight dedup — guards against a peer flooding the same
        // fetched finalization while the first dispatch is still running.
        if !self.pending_finalization_verifications.insert(id) {
            tracing::debug!(
                tick = %tick_id,
                "Duplicate Finalization verification dispatch suppressed"
            );
            return Vec::new();
        }

        let ecs = tick.execution_certificates();
        let mut ec_public_keys = Vec::with_capacity(ecs.len());
        let mut beacon_behind = false;
        for ec in ecs {
            let shard = ec.shard_id();
            // The recovery freeze fences a contained EC from a recovering
            // source shard past its attested frontier — a forged orphan the
            // beyond-f retained committee produced after the halt, which would
            // otherwise resolve the old committee and carry a false tick
            // finalization into this consumer's state.
            if topology_schedule.recovery_fences(shard, ec.block_height()) {
                tracing::warn!(
                    tick = %tick.tick_id(),
                    shard = shard.inner(),
                    height = ec.block_height().inner(),
                    "Rejecting fetched Finalization: contained EC from a recovering \
                     shard past the freeze frontier"
                );
                self.pending_finalization_verifications.remove(&id);
                return vec![Action::AbandonFetch(FetchAbandon::Finalizations {
                    ids: vec![id],
                })];
            }
            // Each contained EC is verified against the committee seated at its
            // own anchor on its own shard. A not-yet-committed epoch (our
            // beacon behind) defers the whole tick for replay once the beacon
            // catches up, rather than abandoning and re-fetching; a below-floor
            // epoch rejects it — the EC is past its retention horizon and can
            // never resolve again.
            let committee = match topology_schedule.lookup(ec.vote_anchor_ts()) {
                ScheduleLookup::Committee(committee) => committee,
                ScheduleLookup::NotYetCommitted => {
                    beacon_behind = true;
                    break;
                }
                ScheduleLookup::Evicted => {
                    tracing::warn!(
                        tick = %tick.tick_id(),
                        shard = shard.inner(),
                        "Rejecting fetched Finalization: contained EC's committee epoch is \
                         below the schedule floor"
                    );
                    self.pending_finalization_verifications.remove(&id);
                    return vec![Action::AbandonFetch(FetchAbandon::Finalizations {
                        ids: vec![id],
                    })];
                }
            };
            if !ec_has_shard_quorum_power(committee, ec.as_unverified()) {
                tracing::warn!(
                    tick = %tick.tick_id(),
                    shard = shard.inner(),
                    "Rejecting fetched Finalization: contained EC lacks quorum power"
                );
                self.pending_finalization_verifications.remove(&id);
                return vec![Action::AbandonFetch(FetchAbandon::Finalizations {
                    ids: vec![id],
                })];
            }
            let Some(public_keys) = committee_public_keys_for_shard(committee, shard) else {
                tracing::warn!(
                    tick = %tick.tick_id(),
                    shard = shard.inner(),
                    "Rejecting fetched Finalization: cannot resolve EC committee keys"
                );
                self.pending_finalization_verifications.remove(&id);
                return vec![Action::AbandonFetch(FetchAbandon::Finalizations {
                    ids: vec![id],
                })];
            };
            ec_public_keys.push(public_keys);
        }
        if beacon_behind {
            // Buffer the whole tick; replayed on `BeaconBlockPersisted` once the
            // beacon reaches the deferred EC's epoch.
            self.pending_finalization_verifications.remove(&id);
            self.awaiting_finalizations
                .push(tick.tick_id().shard_id(), tick);
            return Vec::new();
        }
        vec![Action::VerifyFinalization {
            finalization: tick,
            ec_public_keys,
        }]
    }

    /// Re-attempt every buffered cross-shard EC and finalization now that the
    /// beacon has advanced. Drains both buffers and replays each through its
    /// normal admission path, which re-resolves the committee and re-buffers
    /// any still beyond the schedule. Called on `BeaconBlockPersisted`.
    pub fn on_beacon_block_persisted(
        &mut self,
        topology_schedule: &TopologySchedule,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        for cert in self.awaiting_certs.drain() {
            actions.extend(self.on_execution_certificate(topology_schedule, cert));
        }
        for tick in self.awaiting_finalizations.drain() {
            actions.extend(self.admit_finalization(topology_schedule, tick));
        }
        actions
    }

    /// Handle the result of [`Action::VerifyFinalization`]. Emits the
    /// admission continuation only when every EC's signature passed.
    #[must_use]
    pub fn on_finalization_verified(
        &mut self,
        result: Result<Arc<Verified<Finalization>>, (Arc<Finalization>, FinalizationVerifyError)>,
    ) -> Vec<Action> {
        // Release the in-flight slot regardless of outcome — future
        // arrivals can dispatch again.
        let tick = match result {
            Ok(verified) => {
                self.pending_finalization_verifications
                    .remove(&verified.receipt_hash());
                verified
            }
            Err((raw, err)) => {
                self.pending_finalization_verifications
                    .remove(&raw.receipt_hash());
                tracing::warn!(
                    tick = %raw.tick_id(),
                    error = ?err,
                    "Dropping fetched Finalization: contained EC signature invalid"
                );
                return vec![Action::AbandonFetch(FetchAbandon::Finalizations {
                    ids: vec![raw.receipt_hash()],
                })];
            }
        };
        // Lift the verification result into the `Block::Live.certificates`
        // transport shape exactly once so the admission event and any
        // downstream pending-block storage share the same `Arc`.
        let tick = Arc::new((*tick).clone().into());
        vec![Action::Continuation(ProtocolEvent::FinalizationsAdmitted {
            finalizations: vec![tick],
        })]
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Query Methods
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get the local tick assignment for a transaction.
    #[must_use]
    pub fn tick_assignment_for(&self, tx_hash: TxHash) -> Option<TickId> {
        self.ticks.tick_assignment(tx_hash)
    }

    /// Every finalization ready for inclusion, in the order a block must
    /// settle them.
    ///
    /// That order is tick order, and it is the store's own: a tick's
    /// certificate settles the writes that tick produced, and a later
    /// tick executed against the earlier one's output. `TickId` sorts by
    /// `(shard, height)`, so iterating the store is already the order
    /// receipts have to be applied in — there is nothing to sort and
    /// nothing to hold back.
    #[must_use]
    pub fn get_finalizations(&self) -> Vec<Arc<Verifiable<Finalization>>> {
        self.finalized.all()
    }

    /// Whether provisions from `shard` have been absorbed for `tx_hash` —
    /// committed chain content, since absorption runs at block commit.
    /// The proposal seam's engagement check reads this for the payer
    /// shard of a cross-shard transaction.
    #[must_use]
    pub fn has_provisions_from(&self, tx_hash: TxHash, shard: ShardId) -> bool {
        self.provisioning.has_received_from(tx_hash, shard)
    }

    /// Get a finalization by its identity (returns `Arc` for sharing).
    #[must_use]
    pub fn get_finalization(
        &self,
        hash: &FinalizationHash,
    ) -> Option<Arc<Verifiable<Finalization>>> {
        self.finalized.get(hash)
    }

    /// Bloom filter over every transaction in a tracked finalization.
    /// Attached to outgoing `GetBlockRequest`s so the responder can elide
    /// finalizations the requester already has. Returns `None` when the
    /// cached set is too large to size a filter within the configured cap.
    #[must_use]
    pub fn cert_bloom_snapshot(&self) -> Option<BloomFilter<TxHash>> {
        self.finalized.cert_bloom_snapshot()
    }

    /// Get the finalization containing a specific transaction.
    ///
    /// Returns the tick if the tx is part of one that finalized. Once
    /// committed, ticks are persisted to storage and should be fetched
    /// from there.
    #[must_use]
    pub fn get_finalization_for_tx(
        &self,
        tx_hash: TxHash,
    ) -> Option<Arc<Verifiable<Finalization>>> {
        self.finalized.get_for_tx(tx_hash)
    }

    /// Remove a finalization (after its finalization has been committed in a block).
    ///
    /// Cleans up all per-tx tracking state for transactions in this tick.
    /// Takes the `Finalization` directly (rather than just a `TickId`) so
    /// cleanup works even when the tick was never aggregated locally — e.g.
    /// for blocks received via sync. The committed `Finalization` is the
    /// authoritative tx-set source.
    pub fn remove_finalization(&mut self, fw: &Finalization) {
        let tick_id = fw.tick_id();
        self.finalized.remove(&fw.receipt_hash());

        let tx_hashes: Vec<TxHash> = fw.tx_hashes().collect();
        // A tick settles in two halves, so the first one committing says
        // nothing about the second. Drop the tick and its certificate
        // only once every member it holds has been resolved — otherwise
        // the legs half loses the state it is still owed from.
        let fully_resolved = self
            .ticks
            .get_tick_mut(tick_id)
            .is_none_or(|tick| tick.record_settled(tx_hashes.iter().copied()));
        if fully_resolved {
            // The local-shard EC is now durable in storage via the
            // committed finalization; drop the in-memory copy so peers
            // fetching after this point fall through to storage.
            self.exec_certs.evict(tick_id);
            // The tick may be absent entirely (sync path: the block was
            // received as committed without local tracking), which is fine.
            self.ticks.remove_tick(tick_id);
        }
        for &tx_hash in &tx_hashes {
            self.ticks.remove_assignment(tx_hash);
            // What the transaction was provisioned with belongs to every
            // member this shard runs of it: a mixed shard's delivering
            // member is still waiting on its arrival when the issuing
            // one's finalization commits, and drops it with its own.
            if !self.candidates.contains(tx_hash) {
                self.provisioning.remove_tx(tx_hash);
            }
        }
        // Drain pending-tx sets on fulfilled-cert tombstones referencing
        // any of these txs. When the EC's last referenced tx terminates,
        // the tombstone evicts — independent of any wall-clock window.
        self.expected_certs
            .on_txs_terminated(tx_hashes.iter().copied());
    }

    /// Drop every pending tick and EC expectation. Called once when the
    /// local chain terminates at a reshape boundary: finalization is a
    /// finalization in a later block, and a terminated chain commits
    /// no later block, so every pending tick here is permanently
    /// undecidable. Serving state (aggregated ECs, finalizations)
    /// stays — peers still fetch what this chain produced.
    pub fn abort_pending_ticks(&mut self) -> Vec<Action> {
        let counts = self.ticks.drain_all();
        let mut expected = self.expected_certs.drain_expected();
        expected.sort();
        tracing::info!(
            ticks = counts.ticks,
            trackers = counts.trackers,
            assignments = counts.assignments,
            expected_certs = expected.len(),
            unresolved = self.unresolved.len(),
            "Chain terminated — dropped pending execution state"
        );
        // What the chain owes an outcome for goes with the rest. The
        // ledger's entries are abandonable at their deadlines, and a
        // deadline falling after the terminal would have this chain
        // compose a tick to abandon them in — on a coast block, under a
        // committee it no longer has. Nothing here can reach a verdict
        // either way, which is the same reason the ticks above go.
        self.unresolved = UnresolvedTxs::default();
        // The terminated chain's tick outputs die with it: successors seed
        // from settled state, and pending resolutions have nothing left to
        // resolve against. A tick still in flight lands on a cleared
        // chain, so its completion must be able to release the queue.
        self.pending_tick_resolutions.clear();
        self.pending_ticks.clear();
        self.ticked.clear();
        self.tick_in_flight = false;
        let mut actions = vec![Action::ClearTickChain];
        if !expected.is_empty() {
            actions.push(Action::AbandonFetch(FetchAbandon::ExecutionCerts {
                ids: expected,
            }));
        }
        actions
    }

    /// Prune stale tick state (ticks, vote trackers, early votes).
    ///
    /// Ticks stay alive while their `tick_assignment`s list them — an
    /// active assignment means the transaction hasn't reached terminal
    /// state (TC committed or abort completed) so late-arriving votes and
    /// conflicts can still resolve it. Early execution votes follow a
    /// separate policy tied to the registry's state plus a timestamp
    /// retention floor.
    fn prune_execution_state(&mut self) {
        let counts = self.ticks.prune_resolved();

        // Early execution votes:
        // - Tick resolved (EC formed) → votes no longer needed
        // - Leader replayed them (VoteTracker exists) → already consumed
        // - No tick and older than `EARLY_VOTE_RETENTION` → block never
        //   committed, shard consensus broken
        //
        // Non-leaders with a tick but no VoteTracker KEEP early votes. They
        // may become fallback leaders via rotation and need to replay them
        // into the on-demand VoteTracker created in `on_execution_vote`.
        let ev_cutoff = self.committed_ts.minus(EARLY_VOTE_RETENTION);
        let before_ev = self.early.vote_len();
        let registry = &self.ticks;
        self.early.retain_votes(|key, votes| {
            if registry.is_ec_dispatched(key) {
                return false;
            }
            if registry.contains_tracker(key) {
                return false;
            }
            if registry.contains_tick(key) {
                return true;
            }
            votes
                .first()
                .is_some_and(|v| v.vote_anchor_ts() > ev_cutoff)
        });
        let pruned_ev = before_ev - self.early.vote_len();

        if counts.ticks > 0 || counts.trackers > 0 || pruned_ev > 0 || counts.assignments > 0 {
            tracing::debug!(
                pruned_ticks = counts.ticks,
                pruned_vt = counts.trackers,
                pruned_ev,
                pruned_wa = counts.assignments,
                "Pruned resolved tick state"
            );
        }
    }

    /// Check if a transaction is finalized (part of a finalization).
    #[must_use]
    pub fn is_finalized(&self, tx_hash: TxHash) -> bool {
        self.finalized.is_finalized(tx_hash)
    }

    /// Returns the set of all finalized transaction hashes.
    ///
    /// Used by the node orchestrator to pass to shard consensus for conflict filtering.
    #[must_use]
    pub fn finalized_tx_hashes(&self) -> HashSet<TxHash> {
        self.finalized.all_tx_hashes()
    }

    /// Get debug info about tick state for a transaction.
    #[must_use]
    pub fn certificate_tracking_debug(&self, tx_hash: TxHash) -> String {
        let tick_info = self.ticks.tick_assignment(tx_hash).map_or_else(
            || "no tick assignment".to_string(),
            |tick_id| {
                self.ticks.get_tick(&tick_id).map_or_else(
                    || {
                        if self.finalized.contains(&tick_id) {
                            format!("tick={tick_id}, finalized")
                        } else {
                            format!("tick={tick_id}, no tracker")
                        }
                    },
                    |tick| {
                        let determined = tick.determined_ready();
                        let legs = tick.legs_ready();
                        format!("tick={tick_id}, determined_ready={determined}, legs_ready={legs}")
                    },
                )
            },
        );

        let early_count = self.early.attestation_count_for_tx(tx_hash);

        format!("{tick_info}, early_attestations={early_count}")
    }

    /// Get execution memory statistics for monitoring collection sizes.
    #[must_use]
    pub fn memory_stats(&self) -> ExecutionMemoryStats {
        ExecutionMemoryStats {
            tick_execution_receipts: self
                .ticks
                .ticks_iter()
                .map(|(_, w)| w.receipt_count())
                .sum(),
            finalizations: self.finalized.len(),
            ticks: self.ticks.ticks_len(),
            unresolved_txs: self.unresolved.len(),
            vote_trackers: self.ticks.trackers_len(),
            early_votes: self.early.vote_len(),
            expected_exec_certs: self.expected_certs.expected_len(),
            verified_provisions: self.provisioning.verified_len(),
            required_provision_shards: self.provisioning.required_len(),
            received_provision_shards: self.provisioning.received_len(),
            ticks_with_ec: self.ticks.ec_dispatched_len(),
            pending_vote_retries: self.ticks.retries_len(),
            tick_assignments: self.ticks.assignments_len(),
            early_attestations: self.early.tx_index_len(),
            pending_routing: self.early.pending_routing_len(),
            fulfilled_exec_certs: self.expected_certs.fulfilled_len(),
            outbound_certs: self.outbound_certs.memory_stats().tracked_certificates,
            proven_remote_blocks: self.proven_anchors.len(),
            unproven_ecs: self.unproven_ecs.len(),
        }
    }

    /// Get the number of cross-shard transactions currently in flight.
    ///
    /// Counts unique transaction hashes in cross-shard ticks that haven't yet
    /// finalized. Covers provisioning, voting, and certificate collection
    /// phases uniformly (one `TickState` tracks all of them).
    #[must_use]
    pub fn cross_shard_pending_count(&self) -> usize {
        self.ticks.cross_shard_pending_count()
    }
}

impl std::fmt::Debug for ExecutionCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionCoordinator")
            .field("finalizations", &self.finalized.len())
            .field("ticks", &self.ticks.ticks_len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::time::Duration;

    use hyperscale_crypto_bls::BlsSigner;
    use hyperscale_storage::ReplayWindow;
    use hyperscale_types::test_utils::{
        StubVmStatics, certify as test_certify, make_finalization as helpers_make_finalization,
        make_leg_finalization, make_live_block as helpers_make_live_block, state_and_proof,
        test_prefix, test_transaction, test_transaction_running, test_transaction_with_prefixes,
    };
    use hyperscale_types::{
        AbortCharge, Address, AddressClass, AggregateSignature, BeaconWitnessLeafCount,
        ConsensusPublicKey, ConsensusReceipt, ConsensusSignature, EPOCH_DURATION, Epoch, EpochSeed,
        EpochWindows, ExecutionOutcome, GlobalReceiptHash, Hash, LocalKey, MAX_FINALIZATION_DELAY,
        MAX_VALIDITY_RANGE, NetworkDefinition, QuorumCertificate, RETENTION_HORIZON, Randomness,
        RecoveryCause, SeedRing, SeedSource, ShardAnchor, ShardRecovery, Signer, SignerBitfield,
        StateRoot, StoredReceipt, SubstateKey, TickHalf, TransactionDecision, TxResolution,
        UnsettledTx, ValidatorInfo, ValidatorSet, delivery_window_close,
    };
    use hyperscale_vm_types::Seeded;

    use super::*;

    fn make_test_topology() -> TopologySchedule {
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();

        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        let validator_set = ValidatorSet::new(validators);

        TopologySchedule::single(Arc::new(TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            validator_set,
        )))
    }

    /// A topology whose `shard` is under a halt recovery frozen at
    /// `frontier`: an old-committee EC from that shard above the frontier is
    /// the orphan the cross-shard freeze must fence.
    fn make_test_topology_recovering(shard: ShardId, frontier: BlockHeight) -> TopologySchedule {
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        let mut recoveries = BTreeMap::new();
        recoveries.insert(
            shard,
            ShardRecovery {
                cause: RecoveryCause::Halt,
                rotated_at: Epoch::GENESIS,
                retained: vec![ValidatorId::new(0)],
                attested_frontier: frontier,
            },
        );
        TopologySchedule::single(Arc::new(
            TopologySnapshot::new(
                NetworkDefinition::simulator(),
                1,
                ValidatorSet::new(validators),
            )
            .with_pending_recoveries(recoveries),
        ))
    }

    /// A tick holding `txs`, each admitted with its participating shards.
    fn tick_holding(
        tick_id: TickId,
        tick_ts: WeightedTimestamp,
        txs: Vec<(Arc<Verified<Transaction>>, BTreeSet<ShardId>)>,
    ) -> TickState {
        let mut state = TickState::new(
            tick_id,
            BlockHash::from_raw(Hash::from_bytes(b"block")),
            tick_ts,
        );
        for (tx, participating) in txs {
            state.admit(
                tx.hash(),
                Membership::whole(participating),
                tx.work(),
                Admission::Executes,
            );
        }
        state
    }

    /// Lift a test transaction into the verified form a tick holds.
    fn verified_arc(tx: &Arc<Transaction>) -> Arc<Verified<Transaction>> {
        Arc::new(Verified::new_unchecked_for_test((**tx).clone()))
    }

    fn make_test_state() -> ExecutionCoordinator {
        make_test_state_for(ValidatorId::new(0))
    }

    fn make_test_state_for(me: ValidatorId) -> ExecutionCoordinator {
        ExecutionCoordinator::new(me, ShardId::ROOT)
    }

    fn make_test_state_for_shard(me: ValidatorId, local_shard: ShardId) -> ExecutionCoordinator {
        ExecutionCoordinator::new(me, local_shard)
    }

    fn make_live_block(
        height: BlockHeight,
        timestamp_ms: u64,
        proposer: ValidatorId,
        transactions: Vec<Arc<Transaction>>,
    ) -> Block {
        helpers_make_live_block(
            ShardId::ROOT,
            height,
            timestamp_ms,
            proposer,
            transactions,
            vec![],
        )
    }

    fn make_live_block_on_shard(
        shard: ShardId,
        height: BlockHeight,
        timestamp_ms: u64,
        proposer: ValidatorId,
        transactions: Vec<Arc<Transaction>>,
    ) -> Block {
        helpers_make_live_block(shard, height, timestamp_ms, proposer, transactions, vec![])
    }

    fn certify(block: Block) -> CertifiedBlock {
        test_certify(block, 0)
    }

    #[test]
    fn test_single_shard_execution_flow() {
        let mut state = make_test_state();
        let topology_schedule = make_test_topology();

        let tx = test_transaction(1);
        let tx_hash = tx.hash();
        let block = make_live_block(
            BlockHeight::new(1),
            1000,
            ValidatorId::new(0),
            vec![Arc::new(tx)],
        );

        // Block committed with transaction
        let actions = state.on_block_committed(&topology_schedule, &certify(block));

        // Should request execution (single-shard path) and set up tick tracking
        assert!(!actions.is_empty());
        // First action should be ExecuteTransactions
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::ExecuteTransactions { .. }))
        );

        // TickState should be set up for this tick.
        let tick_id = state.ticks.tick_assignment(tx_hash);
        assert!(tick_id.is_some());
        assert!(state.ticks.contains_tick(&tick_id.unwrap()));
    }

    /// A tick composes whatever the block committed, and waits at the
    /// dispatch head for code this node has not fetched.
    ///
    /// Membership is what the committee votes over, so it cannot turn on
    /// local holdings; running a member whose code is missing would reach
    /// the engine's no-code refusal while every replica holding the bytes
    /// settled it. So the tick is composed either way and held here, and
    /// what dispatches when the fetch lands is the tick composed now.
    #[test]
    fn a_tick_waits_at_the_dispatch_head_for_code_this_node_lacks() {
        let mut state = make_test_state();
        let topology_schedule = make_test_topology();

        let package = Hash::from_bytes(b"a package this node has not fetched");
        state.on_missing_packages_updated(vec![package]);

        let tx = test_transaction_running(1, &[package]);
        let tx_hash = tx.hash();
        let block = make_live_block(
            BlockHeight::new(1),
            1000,
            ValidatorId::new(0),
            vec![Arc::new(tx)],
        );
        let actions = state.on_block_committed(&topology_schedule, &certify(block));

        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::ExecuteTransactions { .. })),
            "a tick running unfetched code must not dispatch"
        );
        assert!(
            state.ticks.tick_assignment(tx_hash).is_some(),
            "the tick is composed regardless — only its dispatch waits"
        );

        // The fetch lands and the held tick goes, unchanged.
        let released = state.on_packages_acquired(&[package]);
        let dispatched: Vec<&TxHash> = released
            .iter()
            .filter_map(|action| match action {
                Action::ExecuteTransactions { requests, .. } => Some(requests),
                _ => None,
            })
            .flatten()
            .map(|request| &request.tx_hash)
            .collect();
        assert_eq!(
            dispatched,
            vec![&tx_hash],
            "acquiring the package releases exactly the tick that waited on it"
        );
    }

    /// A package nothing in the queued tick runs never holds it.
    #[test]
    fn an_unrelated_missing_package_holds_no_tick() {
        let mut state = make_test_state();
        let topology_schedule = make_test_topology();

        state.on_missing_packages_updated(vec![Hash::from_bytes(b"someone else's code")]);

        let tx = test_transaction_running(1, &[Hash::from_bytes(b"code this node holds")]);
        let block = make_live_block(
            BlockHeight::new(1),
            1000,
            ValidatorId::new(0),
            vec![Arc::new(tx)],
        );
        let actions = state.on_block_committed(&topology_schedule, &certify(block));

        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::ExecuteTransactions { .. })),
            "only the code a member runs can hold its tick"
        );
    }

    fn make_topology() -> TopologySchedule {
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        let validator_set = ValidatorSet::new(validators);
        TopologySchedule::single(Arc::new(TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            validator_set,
        )))
    }

    /// A tick that completes while its committee is unresolvable must keep
    /// its vote.
    ///
    /// The vote is one-shot: `build_vote_data` marks the tick voted during
    /// the scan, and nothing ever clears that mark. The committee lookup
    /// happens afterwards, in `emit_vote_actions`, so a scan that finds no
    /// committee spends the vote and emits nothing — no action, no retry
    /// registration, and `can_emit_vote` false forever after. The tick then
    /// A batch returning for a tick this coordinator has stopped tracking
    /// says where the coordinator is, not what the tick's fate was.
    ///
    /// The first resolution recorded for a tick is the one the chain
    /// applies, so claiming an abandonment here would consume the entry
    /// the tick's real verdict needs — and the verdict that follows a
    /// local finalization is a settlement, which promotes writes an abort
    /// would have dropped.
    #[test]
    fn a_batch_for_an_untracked_tick_resolves_nothing() {
        let schedule = make_test_topology();
        let mut state = make_test_state();
        let tx = test_transaction(1);
        let tx_hash = tx.hash();
        let block = make_live_block(
            BlockHeight::new(1),
            1_000,
            ValidatorId::new(0),
            vec![Arc::new(tx)],
        );
        state.on_block_committed(&schedule, &test_certify(block, 1_000));

        let tick_id = state
            .ticks
            .tick_assignment(tx_hash)
            .expect("the committed tx is assigned to a tick");
        assert!(
            state.ticked.contains_key(&tick_id),
            "the tick joined a tick at commit",
        );

        // The tick finalizes locally, which untracks it, and its batch
        // returns afterwards.
        state.ticks.remove_tick(&tick_id);
        state.on_execution_batch_completed(
            &schedule,
            BlockHeight::new(1),
            TickBatchOutcome {
                tick_id,
                results: vec![],
                tx_outcomes: vec![TxOutcome::new(tx_hash, ExecutionOutcome::Failed)],
                fee_receipts: vec![],
                attested_work: vec![],
            },
        );

        assert!(
            state.ticked.contains_key(&tick_id),
            "the tick's tick entry must survive for its real verdict to claim",
        );
        assert!(
            state.pending_tick_resolutions.is_empty(),
            "no fate was decided, so none may be recorded",
        );
    }

    /// holds its locks and never certifies, which is indistinguishable at
    /// the mempool from a tick that was never ready.
    #[test]
    fn a_tick_keeps_its_vote_when_the_committee_cannot_be_resolved() {
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        let snapshot = Arc::new(TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            ValidatorSet::new(validators),
        ));

        // Second-long windows, with only epoch 0 recorded: a tick anchored
        // at 5000ms falls in epoch 5, which the schedule cannot answer for.
        let unresolvable = TopologySchedule::new(1_000, Epoch::new(0), Arc::clone(&snapshot));
        let mut resolved = unresolvable.clone();
        resolved.insert(Epoch::new(5), Arc::clone(&snapshot));

        let mut state = make_test_state();
        let tx = test_transaction(1);
        let tx_hash = tx.hash();
        let block = make_live_block(
            BlockHeight::new(1),
            5_000,
            ValidatorId::new(0),
            vec![Arc::new(tx)],
        );
        state.on_block_committed(&unresolvable, &test_certify(block, 5_000));

        // Execution lands, so the tick is complete and ready to vote.
        let tick_id = state
            .ticks
            .tick_assignment(tx_hash)
            .expect("the committed tx is assigned to a tick");
        state.on_execution_batch_completed(
            &unresolvable,
            BlockHeight::new(1),
            TickBatchOutcome {
                tick_id,
                results: vec![],
                tx_outcomes: vec![TxOutcome::new(tx_hash, ExecutionOutcome::Failed)],
                fee_receipts: vec![],
                attested_work: vec![],
            },
        );

        let blocked = state.emit_vote_actions(&unresolvable);
        assert!(
            blocked.is_empty(),
            "an unresolvable committee routes nowhere, got {blocked:?}",
        );

        // The tick was ready the whole time; only the routing was missing.
        // Once the schedule carries its epoch the vote must still be there.
        let recovered = state.emit_vote_actions(&resolved);
        assert!(
            recovered
                .iter()
                .any(|a| matches!(a, Action::SignAndSendExecutionVote { .. })),
            "the tick must still vote once its committee resolves, got {recovered:?}",
        );
    }

    #[test]
    fn test_only_leader_gets_vote_tracker() {
        let tx = test_transaction(1);

        // Determine who the tick leader will be for this block's tick.
        let topo0 = make_topology();
        let committee = topo0.head().committee_for_shard(ShardId::ROOT).to_vec();
        let block = make_live_block(
            BlockHeight::new(1),
            1000,
            ValidatorId::new(0),
            vec![Arc::new(tx.clone())],
        );

        // Commit the block as validator 0 to discover the tick_id.
        let mut state0 = make_test_state();
        state0.on_block_committed(&topo0, &certify(block));
        let tick_id = state0
            .ticks
            .ticks_iter()
            .next()
            .map(|(wid, _)| *wid)
            .unwrap();

        let leader = tick_leader(&tick_id, &committee);

        // Leader should have a VoteTracker.
        let topo_leader = make_topology();
        let block_leader = make_live_block(
            BlockHeight::new(1),
            1000,
            ValidatorId::new(0),
            vec![Arc::new(tx.clone())],
        );
        let mut state_leader = make_test_state_for(leader);
        state_leader.on_block_committed(&topo_leader, &certify(block_leader));
        assert!(
            state_leader.ticks.contains_tracker(&tick_id),
            "Leader should have VoteTracker"
        );

        // A non-leader should NOT have a VoteTracker.
        let non_leader_id = *committee.iter().find(|&&v| v != leader).unwrap();
        let topo_non = make_topology();
        let block_non = make_live_block(
            BlockHeight::new(1),
            1000,
            ValidatorId::new(0),
            vec![Arc::new(tx)],
        );
        let mut state_non = make_test_state_for(non_leader_id);
        state_non.on_block_committed(&topo_non, &certify(block_non));
        assert!(
            !state_non.ticks.contains_tracker(&tick_id),
            "Non-leader should NOT have VoteTracker"
        );
    }

    #[test]
    fn test_fallback_tracker_created_on_vote() {
        let tx = test_transaction(1);
        let topo = make_topology();
        let committee = topo.head().committee_for_shard(ShardId::ROOT).to_vec();
        let block = make_live_block(
            BlockHeight::new(1),
            1000,
            ValidatorId::new(0),
            vec![Arc::new(tx.clone())],
        );
        let block_hash = block.hash();

        let mut state = make_test_state();
        state.on_block_committed(&topo, &certify(block));

        let tick_id = state
            .ticks
            .ticks_iter()
            .next()
            .map(|(wid, _)| *wid)
            .unwrap();
        let leader = tick_leader(&tick_id, &committee);

        // If we're the leader, this test doesn't apply — find a non-leader topology.
        let non_leader_id = committee.iter().find(|&&v| v != leader).unwrap();
        let topo_non = make_topology();
        let block_non = make_live_block(
            BlockHeight::new(1),
            1000,
            ValidatorId::new(0),
            vec![Arc::new(tx)],
        );
        let mut state_non = make_test_state_for(*non_leader_id);
        state_non.on_block_committed(&topo_non, &certify(block_non));

        assert!(!state_non.ticks.contains_tracker(&tick_id));
        assert!(state_non.ticks.contains_tick(&tick_id));

        // Simulate receiving a vote (as if we're a fallback leader).
        let fake_vote = ExecutionVote::new(
            block_hash,
            BlockHeight::new(1),
            WeightedTimestamp::ZERO,
            tick_id,
            ShardId::ROOT,
            GlobalReceiptRoot::ZERO,
            1,
            vec![],
            leader,
            ConsensusSignature::ZERO,
        );

        state_non.on_unverified_execution_vote(&topo_non, fake_vote);

        // Should have created a fallback VoteTracker.
        assert!(
            state_non.ticks.contains_tracker(&tick_id),
            "Fallback VoteTracker should be created"
        );
    }

    #[test]
    fn on_execution_vote_drops_non_committee_voter() {
        // Vote claiming to be from a validator outside the local shard
        // committee must be rejected at the top of on_execution_vote, with
        // no early-buffer or tracker side effect. Otherwise the vote could
        // pool its cross-shard power into the tracker and trigger premature
        // aggregation that produces an EC the verifier will reject.
        let topo = make_two_shard_topology();
        let local = topo.head().committee_for_shard(ShardId::leaf(1, 0));
        let outsider = (0u64..4)
            .map(ValidatorId::new)
            .find(|v| !local.contains(v))
            .expect("two-shard topology has at least one non-local validator");

        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));
        let tick_id = TickId::new(ShardId::leaf(1, 0), BlockHeight::new(1));
        let vote = ExecutionVote::new(
            BlockHash::ZERO,
            BlockHeight::new(1),
            WeightedTimestamp::ZERO,
            tick_id,
            ShardId::leaf(1, 0),
            GlobalReceiptRoot::ZERO,
            0,
            vec![],
            outsider,
            ConsensusSignature::ZERO,
        );

        let actions = state.on_unverified_execution_vote(&topo, vote);
        assert!(actions.is_empty(), "non-committee vote must be dropped");
        assert!(
            !state.ticks.contains_tracker(&tick_id),
            "rejected vote must not seed a fallback VoteTracker"
        );
        assert_eq!(
            state.memory_stats().pending_routing,
            0,
            "rejected vote must not be early-buffered"
        );
    }

    #[test]
    fn test_vote_retry_timeout_emits_rotated_action() {
        use crate::ticks::VOTE_RETRY_TIMEOUT;
        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let topo = make_test_topology();
        let committee = topo.head().committee_for_shard(ShardId::ROOT).to_vec();

        let mut state = make_test_state();
        state.committed_height = BlockHeight::new(20);
        // "Now" timestamp exactly VOTE_RETRY_TIMEOUT past the original send.
        state.committed_ts = WeightedTimestamp::from_millis(10_000).plus(VOTE_RETRY_TIMEOUT);

        // Manually insert a pending retry as if we'd sent a vote at t=10_000ms.
        state.ticks.record_vote_retry(
            tick_id,
            PendingVoteRetry {
                sent_at: WeightedTimestamp::from_millis(10_000),
                attempt: Attempt::INITIAL,
                block_hash: BlockHash::from_raw(Hash::from_bytes(b"block1")),
                block_height: BlockHeight::new(1),
                vote_anchor_ts: WeightedTimestamp::ZERO,
                global_receipt_root: GlobalReceiptRoot::ZERO,
                tx_outcomes: Arc::new(vec![]),
            },
        );

        let actions = state.check_vote_retry_timeouts(&topo);

        // Elapsed == VOTE_RETRY_TIMEOUT, so should emit retry.
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            Action::SignAndSendExecutionVote {
                leader,
                tick_id: wid,
                ..
            } => {
                assert_eq!(wid, &tick_id);
                let expected_leader = tick_leader_at(&tick_id, Attempt::new(1), &committee);
                assert_eq!(*leader, expected_leader, "Should rotate to attempt 1");
            }
            other => panic!(
                "Expected SignAndSendExecutionVote, got {:?}",
                other.type_name()
            ),
        }

        // The retry is still tracked with its cooldown re-anchored at the
        // current committed timestamp — advance exactly one more
        // VOTE_RETRY_TIMEOUT and check that a retry at attempt 2 fires.
        state.committed_ts = state.committed_ts.plus(VOTE_RETRY_TIMEOUT);
        let next = state.check_vote_retry_timeouts(&topo);
        assert_eq!(next.len(), 1);
        if let Action::SignAndSendExecutionVote { leader, .. } = &next[0] {
            let expected = tick_leader_at(&tick_id, Attempt::new(2), &committee);
            assert_eq!(*leader, expected, "second fire rotates to attempt 2");
        } else {
            panic!("expected SignAndSendExecutionVote");
        }
    }

    #[test]
    fn test_vote_retry_cancelled_on_ec_receipt() {
        use crate::ticks::VOTE_RETRY_TIMEOUT;
        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let topo = make_test_topology();

        let mut state = make_test_state();
        state.committed_height = BlockHeight::new(10);
        state.ticks.record_vote_retry(
            tick_id,
            PendingVoteRetry {
                sent_at: WeightedTimestamp::from_millis(5_000),
                attempt: Attempt::INITIAL,
                block_hash: BlockHash::from_raw(Hash::from_bytes(b"block1")),
                block_height: BlockHeight::new(1),
                vote_anchor_ts: WeightedTimestamp::ZERO,
                global_receipt_root: GlobalReceiptRoot::ZERO,
                tx_outcomes: Arc::new(vec![]),
            },
        );

        // Simulate receiving a verified local shard EC with quorum signers.
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        );
        state.on_certificate_verified(&topo, Ok(Arc::new(Verified::new_unchecked_for_test(cert))));

        // Advance time past the retry deadline; if the retry had survived,
        // this would fire a SignAndSendExecutionVote action.
        state.committed_ts = WeightedTimestamp::from_millis(5_000).plus(VOTE_RETRY_TIMEOUT);
        let actions = state.check_vote_retry_timeouts(&topo);
        assert!(
            actions.is_empty(),
            "EC receipt must cancel the retry so no action fires"
        );
    }

    #[test]
    fn on_certificate_verified_rejects_subquorum_ec() {
        // A single Byzantine signer can produce a signature-valid EC. Without a
        // quorum-power gate, that sub-quorum EC would clear the expected-
        // cert record, populate the local-shard fallback-serving cache,
        // and feed tick attestation. The rejection also emits an
        // `AbandonFetch::ExecutionCerts` naming every transaction the cert
        // claimed, so each pinned fetch releases its FSM slot.
        let topo = make_test_topology();
        let mut state = make_test_state();
        state.committed_height = BlockHeight::new(10);

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));

        let mut signers = SignerBitfield::new(4);
        signers.set(0); // single signer — well below 2f+1 = 3
        let covered_tx = TxHash::from(Hash::from_bytes(b"covered by the refused cert"));
        let cert = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![TxOutcome::new(
                covered_tx,
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                },
            )],
            AggregateSignature::ZERO,
            signers,
        ));

        let verified = Arc::new(Verified::new_unchecked_for_test((*cert).clone()));
        let actions = state.on_certificate_verified(&topo, Ok(verified));
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::Continuation(ProtocolEvent::ExecutionCertificateAdmitted { .. })
            )),
            "sub-quorum EC must produce no admission continuation"
        );
        assert!(
            actions.iter().any(|a| matches!(
                a,
                Action::AbandonFetch(FetchAbandon::ExecutionCerts { ids }) if ids == &vec![(tick_id.shard_id(), covered_tx)]
            )),
            "sub-quorum drop must emit AbandonFetch::ExecutionCerts, got: {actions:?}"
        );
        assert!(
            state.exec_certs.get(&tick_id).is_none(),
            "sub-quorum EC must not enter the local-shard serving cache"
        );
    }

    #[test]
    fn on_certificate_verified_invalid_sig_abandons_fetch() {
        // signature verification returns `valid=false`. The cert is
        // dropped without admission, and the FSM is told to release the
        // in-flight slot.
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let covered_tx = TxHash::from(Hash::from_bytes(b"covered by the refused cert"));
        let cert = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![TxOutcome::new(
                covered_tx,
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                },
            )],
            AggregateSignature::ZERO,
            signers,
        ));

        let actions = state.on_certificate_verified(
            &topo,
            Err((
                cert,
                ExecutionCertificateVerifyError::BadAggregatedSignature,
            )),
        );
        assert!(
            actions.iter().any(|a| matches!(
                a,
                Action::AbandonFetch(FetchAbandon::ExecutionCerts { ids }) if ids == &vec![(tick_id.shard_id(), covered_tx)]
            )),
            "invalid-sig drop must emit AbandonFetch::ExecutionCerts, got: {actions:?}"
        );
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::Continuation(ProtocolEvent::ExecutionCertificateAdmitted { .. })
            )),
            "invalid-sig must not emit admission continuation"
        );
    }

    // Note: the committee-keys-fail branch of `on_execution_certificate` is
    // structurally covered (emits the abandon when
    // `committee_public_keys_for_shard` returns `None`) but is not
    // exercised by a unit test here — `None` only fires when a known
    // committee member is missing a public key in the topology, a
    // corruption condition the public test fixtures can't easily
    // construct. Realistic failures (unknown shard with empty committee)
    // dispatch with an empty key set and fall through to the invalid-sig
    // branch, which is covered above.

    /// Who a certificate is broadcast to is a question about the batch's
    /// transactions, not its identity: the shards their participants name.
    /// A batch holding a transaction shard 1 is party to owes shard 1 the
    /// certificate — and what shard 1 receives is the outcome for that
    /// transaction, not the batch.
    #[test]
    fn test_leader_broadcasts_ec_locally() {
        use hyperscale_types::compute_global_receipt_root;
        use hyperscale_types::test_utils::test_transaction;

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let topo = make_test_topology();

        let mut state = make_test_state();
        let tx = Arc::new(test_transaction(3));
        let participating: BTreeSet<ShardId> =
            [ShardId::ROOT, ShardId::leaf(1, 1)].into_iter().collect();
        state.ticks.insert_tick(
            tick_id,
            tick_holding(
                tick_id,
                WeightedTimestamp::ZERO,
                vec![(verified_arc(&tx), participating)],
            ),
        );

        let outcomes = vec![TxOutcome::new(tx.hash(), ExecutionOutcome::Aborted)];
        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            compute_global_receipt_root(&outcomes),
            outcomes,
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        let actions = state.on_certificate_aggregated(
            &topo,
            &tick_id,
            &Arc::new(Verified::new_unchecked_for_test(cert)),
        );

        // Should have: BroadcastEC(local) + BroadcastEC(remote shard 1)
        let broadcast_actions: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, Action::BroadcastExecutionCertificate { .. }))
            .collect();

        assert!(
            broadcast_actions.len() >= 2,
            "Should broadcast to local peers AND remote shards, got {}",
            broadcast_actions.len()
        );

        // One should be for the local shard (shard 0).
        let has_local = broadcast_actions.iter().any(|a| match a {
            Action::BroadcastExecutionCertificate { shard, .. } => *shard == ShardId::ROOT,
            _ => false,
        });
        assert!(has_local, "Should include local shard broadcast");

        // One should be for the remote shard (shard 1).
        let has_remote = broadcast_actions.iter().any(|a| match a {
            Action::BroadcastExecutionCertificate { shard, .. } => *shard == ShardId::leaf(1, 1),
            _ => false,
        });
        assert!(has_remote, "Should include remote shard broadcast");
    }

    /// A shard receives the outcomes for the transactions it is party to
    /// and nothing else, while this shard's own peers receive the whole
    /// batch — they are building the same finalization we are.
    #[test]
    fn a_remote_shard_receives_only_its_own_transactions() {
        use hyperscale_types::compute_global_receipt_root;
        use hyperscale_types::test_utils::test_transaction;

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let topo = make_test_topology();
        let mut state = make_test_state();

        let shared = Arc::new(test_transaction(3));
        let ours = Arc::new(test_transaction(4));
        state.ticks.insert_tick(
            tick_id,
            tick_holding(
                tick_id,
                WeightedTimestamp::ZERO,
                vec![
                    (
                        verified_arc(&shared),
                        [ShardId::ROOT, ShardId::leaf(1, 1)].into_iter().collect(),
                    ),
                    (
                        verified_arc(&ours),
                        std::iter::once(ShardId::ROOT).collect(),
                    ),
                ],
            ),
        );

        let outcomes = vec![
            TxOutcome::new(shared.hash(), ExecutionOutcome::Aborted),
            TxOutcome::new(ours.hash(), ExecutionOutcome::Aborted),
        ];
        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            compute_global_receipt_root(&outcomes),
            outcomes,
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        let actions = state.on_certificate_aggregated(
            &topo,
            &tick_id,
            &Arc::new(Verified::new_unchecked_for_test(cert)),
        );

        let sent = |target: ShardId| -> Arc<Verified<ExecutionCertificate>> {
            actions
                .iter()
                .find_map(|a| match a {
                    Action::BroadcastExecutionCertificate {
                        shard, certificate, ..
                    } if *shard == target => Some(Arc::clone(certificate)),
                    _ => None,
                })
                .expect("a broadcast for the target shard")
        };

        let remote = sent(ShardId::leaf(1, 1));
        assert_eq!(
            remote.tx_outcomes().len(),
            1,
            "the remote shard is party to one of the two"
        );
        assert!(remote.covers(&shared.hash()));
        assert!(!remote.covers(&ours.hash()));
        assert_eq!(
            remote.tx_count(),
            2,
            "the projection still names the whole batch it proves against"
        );
        assert_eq!(
            remote.global_receipt_root(),
            sent(ShardId::ROOT).global_receipt_root()
        );

        let local = sent(ShardId::ROOT);
        assert!(local.is_complete(), "our own peers get the whole batch");
    }

    /// `admit_finalization` must NOT emit `FinalizationsAdmitted`
    /// inline — that would mean signature verification ran on the state-machine
    /// thread, bringing back the pre-async stall on the consensus path.
    /// The expected output is a single `VerifyFinalization` action; the
    /// admission continuation only fires once the verify event lands.
    #[test]
    fn admit_finalization_dispatches_async_verify() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let ec = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        ));
        let tick: Arc<Verifiable<Finalization>> =
            Arc::new(Finalization::new(tick_id, TickHalf::Determined, vec![ec], vec![]).into());
        let _fw_hash = tick.receipt_hash();

        let actions = state.admit_finalization(&topo, tick);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::VerifyFinalization { .. }));
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::Continuation(ProtocolEvent::FinalizationsAdmitted { .. })
            )),
            "admission continuation must only fire after async verify"
        );
    }

    /// `on_finalization_verified` with `valid = false` must drop the tick
    /// rather than emit the admission continuation — that's exactly the
    /// poisoning vector this gate exists to close. The dropped tick also
    /// surfaces a `FetchAbandon::Finalizations` so any pinned fetch
    /// FSM entry releases its slot.
    #[test]
    fn on_finalization_verified_drops_invalid() {
        let mut state = make_test_state();
        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let ec = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        ));
        let tick = Arc::new(Finalization::new(
            tick_id,
            TickHalf::Determined,
            vec![ec],
            vec![],
        ));
        let fw_hash = tick.receipt_hash();
        let actions = state.on_finalization_verified(Err((
            tick,
            FinalizationVerifyError::ExecutionCertificate {
                index: 0,
                source: ExecutionCertificateVerifyError::BadAggregatedSignature,
            },
        )));
        assert!(
            actions.iter().any(|a| matches!(
                a,
                Action::AbandonFetch(FetchAbandon::Finalizations { ids }) if ids == &vec![fw_hash]
            )),
            "Signature-invalid drop must emit AbandonFetch::Finalizations, got: {actions:?}"
        );
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::Continuation(ProtocolEvent::FinalizationsAdmitted { .. })
            )),
            "must not emit admission continuation on invalid"
        );
    }

    /// `admit_finalization` with an EC lacking quorum power must emit
    /// the abandon (so the FSM doesn't pin) AND must clear the in-flight
    /// dedup set so future arrivals can retry — without that the same
    /// `TickId` would silently fail every subsequent admission.
    #[test]
    fn admit_finalization_quorum_power_fail_abandons_and_clears_dedup() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        // Only one signer in a 4-validator committee — sub-quorum
        // (2f+1=3 needed).
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        let ec = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        ));
        let tick: Arc<Verifiable<Finalization>> =
            Arc::new(Finalization::new(tick_id, TickHalf::Determined, vec![ec], vec![]).into());
        let fw_hash = tick.receipt_hash();

        let actions = state.admit_finalization(&topo, Arc::clone(&tick));
        assert!(
            actions.iter().any(|a| matches!(
                a,
                Action::AbandonFetch(FetchAbandon::Finalizations { ids }) if ids == &vec![fw_hash]
            )),
            "quorum-power drop must emit AbandonFetch::Finalizations, got: {actions:?}"
        );

        // Regression: dedup set must NOT retain this tick so a fresh
        // arrival of the same id (e.g., a peer retransmitting after
        // gossiping a corrected tick) is allowed to dispatch.
        let retry_actions = state.admit_finalization(&topo, tick);
        assert!(
            retry_actions
                .iter()
                .any(|a| matches!(a, Action::AbandonFetch(FetchAbandon::Finalizations { .. }))),
            "retry must still reach the quorum gate, got: {retry_actions:?}"
        );
    }

    /// `admit_finalization` with an unresolvable committee shard must
    /// emit the abandon AND clear the dedup set, same shape as the
    /// quorum-power path.
    #[test]
    fn admit_finalization_unknown_committee_abandons_and_clears_dedup() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        // EC for a shard the test topology doesn't know about — the
        // committee-keys lookup returns `None` and triggers the gate.
        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let ec = Arc::new(ExecutionCertificate::new(
            TickId::new(ShardId::leaf(8, 99), BlockHeight::new(1)),
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        ));
        let tick: Arc<Verifiable<Finalization>> =
            Arc::new(Finalization::new(tick_id, TickHalf::Determined, vec![ec], vec![]).into());
        let fw_hash = tick.receipt_hash();

        let actions = state.admit_finalization(&topo, Arc::clone(&tick));
        assert!(
            actions.iter().any(|a| matches!(
                a,
                Action::AbandonFetch(FetchAbandon::Finalizations { ids }) if ids == &vec![fw_hash]
            )),
            "unknown-committee drop must emit AbandonFetch::Finalizations, got: {actions:?}"
        );

        // Regression: dedup set clear lets retries through.
        let retry_actions = state.admit_finalization(&topo, tick);
        assert!(
            retry_actions
                .iter()
                .any(|a| matches!(a, Action::AbandonFetch(FetchAbandon::Finalizations { .. }))),
            "retry must still reach the committee-keys gate, got: {retry_actions:?}"
        );
    }

    /// `on_finalization_verified` with `valid = true` emits exactly the
    /// admission continuation — same shape as the prior synchronous path.
    #[test]
    fn on_finalization_verified_admits_valid() {
        let mut state = make_test_state();
        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let ec = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        ));
        let tick = Arc::new(Verified::new_unchecked_for_test(Finalization::new(
            tick_id,
            TickHalf::Determined,
            vec![ec],
            vec![],
        )));
        let actions = state.on_finalization_verified(Ok(tick));
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::Continuation(ProtocolEvent::FinalizationsAdmitted { .. })
        ));
    }

    /// Two byte-identical EC arrivals while the first is still in flight
    /// must produce only one `VerifyExecutionCertificateSignature`
    /// dispatch. This shields the crypto pool from a flooding peer.
    #[test]
    fn on_execution_certificate_dedups_byte_identical_retransmit() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        );

        let first = state.on_execution_certificate(&topo, cert.clone().into());
        assert_eq!(first.len(), 1);
        assert!(matches!(
            first[0],
            Action::VerifyExecutionCertificateSignature { .. }
        ));

        // Same bytes mid-flight — must drop without dispatching another
        // verify.
        let second = state.on_execution_certificate(&topo, cert.into());
        assert!(second.is_empty());
    }

    /// The cross-shard freeze: an EC from a recovering shard above its
    /// attested frontier is dropped without dispatching verification — the
    /// forged orphan a beyond-f retained committee would otherwise export.
    /// One at or below the frontier is legitimate pre-halt history and still
    /// dispatches.
    #[test]
    fn on_execution_certificate_fences_ec_past_recovery_frontier() {
        let recovering = ShardId::ROOT;
        let topo = make_test_topology_recovering(recovering, BlockHeight::new(5));

        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let ec_at = |height: u64| {
            ExecutionCertificate::new(
                TickId::new(recovering, BlockHeight::new(height)),
                WeightedTimestamp::ZERO,
                GlobalReceiptRoot::ZERO,
                vec![],
                AggregateSignature::ZERO,
                signers.clone(),
            )
        };

        // Above the frontier — the orphan. Fenced.
        let mut state = make_test_state();
        let orphan = state.on_execution_certificate(&topo, ec_at(6).into());
        assert!(
            matches!(
                orphan.as_slice(),
                [Action::AbandonFetch(FetchAbandon::ExecutionCerts { .. })]
            ),
            "an EC past the freeze frontier is dropped, got {orphan:?}"
        );

        // At the frontier — legitimate suffix, still dispatches to verify.
        let mut state = make_test_state();
        let suffix = state.on_execution_certificate(&topo, ec_at(5).into());
        assert!(
            matches!(
                suffix.as_slice(),
                [Action::VerifyExecutionCertificateSignature { .. }]
            ),
            "an EC within the frontier dispatches, got {suffix:?}"
        );
    }

    /// A cross-shard EC defers on its source block's commit proof: a bare
    /// QC-certified header is not consumability, and the committed event
    /// replays the deferred EC into verify dispatch.
    #[test]
    fn on_execution_certificate_defers_until_source_block_commit_proven() {
        let topo = make_two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let remote_shard = ShardId::leaf(1, 1);
        let tick_id = TickId::new(remote_shard, BlockHeight::new(5));
        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![TxOutcome::new(
                TxHash::from(Hash::from_bytes(b"deferred_tx")),
                ExecutionOutcome::Aborted,
            )],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        let actions = state.on_execution_certificate(&topo, cert.into());
        assert!(
            actions.is_empty(),
            "an EC from an unproven source block must defer, got {actions:?}"
        );

        // The commit proof lands: the deferred EC replays into dispatch.
        state.proven_anchors().record(
            remote_shard,
            BlockHeight::new(5),
            StateRoot::ZERO,
            WeightedTimestamp::ZERO,
        );
        let actions = state.on_committed_remote_header(&topo, remote_shard);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::VerifyExecutionCertificateSignature { .. })),
            "the committed event must replay the deferred EC, got {actions:?}"
        );
    }

    /// A departed shard's settled set is a commit proof for everything it
    /// names: an EC covered by the set dispatches without waiting on a
    /// remote-header proof no departed chain will supply. An outcome the
    /// set does not name is a verdict that shard never settled, and the
    /// gate still defers it.
    #[test]
    fn settled_set_membership_stands_in_for_the_commit_proof() {
        let topo = make_two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let remote_shard = ShardId::leaf(1, 1);
        let covered = TxHash::from(Hash::from_bytes(b"settled_tx"));
        let make_cert = |tx_hash| {
            ExecutionCertificate::new(
                TickId::new(remote_shard, BlockHeight::new(5)),
                WeightedTimestamp::ZERO,
                GlobalReceiptRoot::ZERO,
                vec![TxOutcome::new(tx_hash, ExecutionOutcome::Aborted)],
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            )
        };

        state.record_settled_txs(
            &topo,
            remote_shard,
            SettledTxSet {
                txs: BTreeSet::from([covered]),
                terminal_wt: WeightedTimestamp::from_millis(1_000),
            },
        );

        let actions = state.on_execution_certificate(&topo, make_cert(covered).into());
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::VerifyExecutionCertificateSignature { .. })),
            "a set-covered EC must dispatch without a commit proof, got {actions:?}"
        );

        let uncovered = TxHash::from(Hash::from_bytes(b"unsettled_tx"));
        let actions = state.on_execution_certificate(&topo, make_cert(uncovered).into());
        assert!(
            actions.is_empty(),
            "an uncovered EC must still defer, got {actions:?}"
        );
    }

    /// The order the recovery actually runs in: the certificate arrives
    /// first and parks on the missing proof, the settled set lands
    /// afterwards and replays it through the gate.
    #[test]
    fn a_settled_set_replays_the_certificates_parked_on_its_proof() {
        let topo = make_two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let remote_shard = ShardId::leaf(1, 1);
        let tx_hash = TxHash::from(Hash::from_bytes(b"parked_tx"));
        let cert = ExecutionCertificate::new(
            TickId::new(remote_shard, BlockHeight::new(5)),
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![TxOutcome::new(tx_hash, ExecutionOutcome::Aborted)],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        let actions = state.on_execution_certificate(&topo, cert.into());
        assert!(
            actions.is_empty(),
            "no proof and no set: the EC parks, got {actions:?}"
        );

        let actions = state.record_settled_txs(
            &topo,
            remote_shard,
            SettledTxSet {
                txs: BTreeSet::from([tx_hash]),
                terminal_wt: WeightedTimestamp::from_millis(1_000),
            },
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::VerifyExecutionCertificateSignature { .. })),
            "the set must replay the parked EC into dispatch, got {actions:?}"
        );
    }

    /// An EC parked at or below the source shard's attested boundary sits
    /// under a joiner's remote-header sync anchor, where forward sync
    /// never delivers the committing structure — parking must request the
    /// commit proof explicitly. Above the boundary it stays a silent
    /// defer: gossip or forward sync proves it in the ordinary course.
    #[test]
    fn deferred_ec_below_the_attested_boundary_requests_its_commit_proof() {
        let remote_shard = ShardId::leaf(1, 1);
        let boundary = BlockHeight::new(10);
        let topo = make_two_shard_topology_with_boundary(remote_shard, boundary);
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let cert_at = |height: u64| {
            ExecutionCertificate::new(
                TickId::new(remote_shard, BlockHeight::new(height)),
                WeightedTimestamp::ZERO,
                GlobalReceiptRoot::ZERO,
                vec![TxOutcome::new(
                    TxHash::from(Hash::from_bytes(&height.to_le_bytes())),
                    ExecutionOutcome::Aborted,
                )],
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            )
        };

        // At the boundary: below the sync anchor — the defer asks for the
        // commit proof.
        let actions = state.on_execution_certificate(&topo, cert_at(10).into());
        assert!(
            matches!(
                actions.as_slice(),
                [Action::Continuation(ProtocolEvent::CommitProofNeeded {
                    source_shard,
                    block_height,
                })] if *source_shard == remote_shard && *block_height == boundary
            ),
            "a below-anchor defer must request its commit proof, got {actions:?}"
        );

        // Above the boundary: an ordinary silent defer.
        let actions = state.on_execution_certificate(&topo, cert_at(11).into());
        assert!(
            actions.is_empty(),
            "an above-anchor defer needs no explicit request, got {actions:?}"
        );
    }

    /// No pending recovery for the shard: the fence is inert, an EC at any
    /// height dispatches as usual.
    #[test]
    fn on_execution_certificate_fence_inert_without_recovery() {
        let topo = make_test_topology();
        let mut state = make_test_state();
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let cert = ExecutionCertificate::new(
            TickId::new(ShardId::ROOT, BlockHeight::new(99)),
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        );
        let actions = state.on_execution_certificate(&topo, cert.into());
        assert!(matches!(
            actions.as_slice(),
            [Action::VerifyExecutionCertificateSignature { .. }]
        ));
    }

    /// Once verification completes (success or failure), the in-flight
    /// slot is released and a subsequent retransmit is allowed to
    /// re-dispatch.
    #[test]
    fn on_execution_certificate_releases_slot_after_verification() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        );

        let _ = state.on_execution_certificate(&topo, cert.clone().into());
        // Simulate the crypto pool returning an invalid result. The slot is
        // released so a follow-up arrival can re-dispatch.
        let _ = state.on_certificate_verified(
            &topo,
            Err((
                Arc::new(cert.clone()),
                ExecutionCertificateVerifyError::BadAggregatedSignature,
            )),
        );
        let again = state.on_execution_certificate(&topo, cert.into());
        assert_eq!(again.len(), 1);
        assert!(matches!(
            again[0],
            Action::VerifyExecutionCertificateSignature { .. }
        ));
    }

    /// An EC already in `exec_certs` (placed there by a co-hosted vnode's
    /// aggregation, or by an earlier verification of the same wire bytes)
    /// short-circuits the verify dispatch on a wire-hash match.
    #[test]
    fn on_execution_certificate_skips_dispatch_on_cached_wire_hash_match() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        );
        state
            .exec_certs
            .insert(Arc::new(Verified::new_unchecked_for_test(cert.clone())));

        let actions = state.on_execution_certificate(&topo, cert.into());
        assert!(
            actions.is_empty(),
            "cached wire-hash match must short-circuit"
        );
    }

    /// A different aggregation of the same logical EC (same `TickId` but
    /// distinct signers / signature, hence distinct wire bytes) is not
    /// short-circuited by an earlier cache entry — it still needs its own
    /// signature check.
    #[test]
    fn on_execution_certificate_falls_through_on_cached_tick_id_with_wire_hash_mismatch() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers_a = SignerBitfield::new(4);
        signers_a.set(0);
        signers_a.set(1);
        signers_a.set(2);
        let mut signers_b = SignerBitfield::new(4);
        signers_b.set(1);
        signers_b.set(2);
        signers_b.set(3);

        let cached = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers_a,
        );
        let incoming = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers_b,
        );
        assert_ne!(cached.wire_hash(), incoming.wire_hash());
        state
            .exec_certs
            .insert(Arc::new(Verified::new_unchecked_for_test(cached)));

        let actions = state.on_execution_certificate(&topo, incoming.into());
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::VerifyExecutionCertificateSignature { .. }
        ));
    }

    /// `admit_finalization` dedups a second arrival for the same
    /// `TickId` while verification is still in flight.
    #[test]
    fn admit_finalization_dedups_in_flight_arrival() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let ec = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        ));
        let tick: Arc<Verifiable<Finalization>> =
            Arc::new(Finalization::new(tick_id, TickHalf::Determined, vec![ec], vec![]).into());
        let _fw_hash = tick.receipt_hash();

        let first = state.admit_finalization(&topo, Arc::clone(&tick));
        assert_eq!(first.len(), 1);
        assert!(matches!(first[0], Action::VerifyFinalization { .. }));

        let second = state.admit_finalization(&topo, tick);
        assert!(second.is_empty());
    }

    /// A `Finalization` already in the canonical store short-circuits
    /// before any verify dispatch.
    #[test]
    fn admit_finalization_skips_when_already_finalized() {
        let topo = make_test_topology();
        let mut state = make_test_state();

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let ec = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        ));
        let raw_finalization = Finalization::new(tick_id, TickHalf::Determined, vec![ec], vec![]);
        let verifiable_finalization =
            Arc::new(Verified::new_unchecked_for_test(raw_finalization.clone()).into());
        // Seed the canonical store directly (mirrors what `finalize`
        // does on the local-aggregation path).
        state.finalized.insert(tick_id, verifiable_finalization);

        let actions = state.admit_finalization(&topo, Arc::new(Verifiable::from(raw_finalization)));
        assert!(actions.is_empty());
    }

    /// A `Finalization` delivered by `admit_finalization` (the fetch
    /// entry point) must reject any tick whose contained ECs lack quorum
    /// power or signature validity. Otherwise a peer answering
    /// `finalization.request` can poison the `io_loop` serving cache
    /// (via the `Continuation(FinalizationsAdmitted)` interception) and
    /// we re-serve the bogus tick to other peers.
    #[test]
    fn test_admit_finalization_rejects_subquorum_ec() {
        let topo = make_two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let tick_id = TickId::new(ShardId::leaf(1, 0), BlockHeight::new(1));
        let bogus_ec = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::from_millis(1_000_000),
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            SignerBitfield::empty(), // no signers — far below 2f+1
        ));
        let tick: Arc<Verifiable<Finalization>> = Arc::new(
            Finalization::new(tick_id, TickHalf::Determined, vec![bogus_ec], vec![]).into(),
        );
        let fw_hash = tick.receipt_hash();

        let actions = state.admit_finalization(&topo, tick);
        // No admission continuation — the poisoning vector this gate
        // exists to close. The rejection now emits a `FetchAbandon` so
        // any pinned fetch FSM entry releases its slot.
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::Continuation(ProtocolEvent::FinalizationsAdmitted { .. })
            )),
            "sub-quorum Finalization must produce no admission Continuation"
        );
        assert!(
            actions.iter().any(|a| matches!(
                a,
                Action::AbandonFetch(FetchAbandon::Finalizations { ids }) if ids == &vec![fw_hash]
            )),
            "sub-quorum drop must emit AbandonFetch::Finalizations, got: {actions:?}"
        );
    }

    /// Receipt of a cross-shard EC must NOT mark its expectation
    /// fulfilled until the signature has been verified. Otherwise a
    /// Byzantine peer can ship a forged EC, the tombstone is set with
    /// `vote_anchor_ts + RETENTION_HORIZON` (peer-controlled), legitimate
    /// fallback fetches are suppressed, and the verify pool's silent
    /// rejection leaves us stranded.
    #[test]
    fn test_on_execution_certificate_does_not_mark_fulfilled_before_verification() {
        let topo = make_two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let remote_shard = ShardId::leaf(1, 1);
        let tick_id = TickId::new(remote_shard, BlockHeight::new(5));
        let cross_shard_tx = TxHash::from(Hash::from_bytes(b"cross-shard tx"));
        state
            .expected_certs
            .register(remote_shard, cross_shard_tx, state.committed_ts);
        assert_eq!(state.expected_certs.expected_len(), 1);
        assert_eq!(state.expected_certs.fulfilled_len(), 0);

        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::from_millis(1_000_000),
            GlobalReceiptRoot::ZERO,
            vec![TxOutcome::new(
                cross_shard_tx,
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                },
            )],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );
        let _ = state.on_execution_certificate(&topo, cert.clone().into());
        assert_eq!(
            state.expected_certs.expected_len(),
            1,
            "expectation must remain pending until verification completes"
        );
        assert_eq!(
            state.expected_certs.fulfilled_len(),
            0,
            "no tombstone must be created from an unverified EC"
        );

        // Verification fails — the EC was a forgery. State is unchanged;
        // the legitimate cert can still arrive and clear the expectation.
        let _ = state.on_certificate_verified(
            &topo,
            Err((
                Arc::new(cert),
                ExecutionCertificateVerifyError::BadAggregatedSignature,
            )),
        );
        assert_eq!(state.expected_certs.expected_len(), 1);
        assert_eq!(state.expected_certs.fulfilled_len(), 0);
    }

    /// A received cross-shard EC must always dispatch signature verification
    /// before any tick state sees it — including when no local tick tracks
    /// any tx in the cert. Without that, a Byzantine remote could buffer
    /// forged `tx_outcomes` that the replay path later trusts at commit
    /// time.
    #[test]
    fn test_on_execution_certificate_always_dispatches_verification_even_without_tracker() {
        let topo = make_two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let remote_shard = ShardId::leaf(1, 1);
        let tick_id = TickId::new(remote_shard, BlockHeight::new(5));
        // No local ticks / trackers have been created for this tx.
        let cert = ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![TxOutcome::new(
                TxHash::from(Hash::from_bytes(b"untracked_tx")),
                ExecutionOutcome::Aborted,
            )],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        // The source block is commit-proven; the gate under test is verify
        // dispatch without a local tracker, not the commit-proof gate.
        state.proven_anchors().record(
            remote_shard,
            BlockHeight::new(5),
            StateRoot::ZERO,
            WeightedTimestamp::ZERO,
        );
        state.on_committed_remote_header(&topo, remote_shard);
        let actions = state.on_execution_certificate(&topo, cert.into());
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::VerifyExecutionCertificateSignature { .. })),
            "must dispatch signature verification even when no local tracker matches"
        );
        // Nothing lands in the early-arrival buffer until verification passes.
        assert_eq!(state.memory_stats().pending_routing, 0);
        assert_eq!(state.memory_stats().early_attestations, 0);
    }

    // ========================================================================
    // Expected Execution Cert Retention
    // ========================================================================

    /// Multi-shard topology for expected-cert tests: 4 validators, 2 shards.
    /// Local is validator 0 (shard 0); shard 1 = {1, 3}.
    fn make_two_shard_topology() -> TopologySchedule {
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        TopologySchedule::single(Arc::new(TopologySnapshot::new(
            NetworkDefinition::simulator(),
            2,
            ValidatorSet::new(validators),
        )))
    }

    /// A two-shard schedule whose head attests `shard`'s boundary at
    /// `height` — the anchor a joiner's remote-header sync starts from.
    fn make_two_shard_topology_with_boundary(
        shard: ShardId,
        height: BlockHeight,
    ) -> TopologySchedule {
        use hyperscale_types::{BeaconWitnessLeafCount, BlockHash, ShardAnchor, StateRoot};

        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        let validator_set = ValidatorSet::new(validators);
        let mut committees = HashMap::new();
        committees.insert(
            ShardId::leaf(1, 0),
            vec![ValidatorId::new(0), ValidatorId::new(1)],
        );
        committees.insert(
            ShardId::leaf(1, 1),
            vec![ValidatorId::new(2), ValidatorId::new(3)],
        );
        let mut boundaries = HashMap::new();
        boundaries.insert(
            shard,
            ShardAnchor {
                state_root: StateRoot::ZERO,
                block_hash: BlockHash::from_raw(Hash::from_bytes(b"boundary")),
                height,
                weighted_timestamp: WeightedTimestamp::from_millis(1),
                witness_base: BeaconWitnessLeafCount::ZERO,
                terminal_roots: None,
                handoff_complete: None,
            },
        );
        TopologySchedule::single(Arc::new(TopologySnapshot::from_explicit_committees(
            NetworkDefinition::simulator(),
            &validator_set,
            committees.clone(),
            committees,
            boundaries,
            HashMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeSet::new(),
        )))
    }

    /// A uniform-power committee over `ids`, one shard, plus its public keys in
    /// committee order (what `committee_public_keys_for_shard` returns).
    fn committee_snapshot(ids: &[u64]) -> (TopologySnapshot, Vec<ConsensusPublicKey>) {
        let validators: Vec<ValidatorInfo> = ids
            .iter()
            .map(|&id| {
                let k = BlsSigner::generate();
                ValidatorInfo {
                    validator_id: ValidatorId::new(id),
                    public_key: k.public_key(),
                }
            })
            .collect();
        let pubkeys = validators.iter().map(|v| v.public_key).collect();
        let snapshot = TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            ValidatorSet::new(validators),
        );
        (snapshot, pubkeys)
    }

    /// Committee A governs epoch 0 (the routing head); committee B, with
    /// disjoint signing keys, governs epoch 1. A remote EC whose
    /// `vote_anchor_ts` lands in epoch 1 must dispatch signature verification against
    /// B's keys — the committee seated at the EC's anchor — not the head.
    #[test]
    fn remote_ec_verification_resolves_committee_at_its_vote_anchor() {
        const ED: u64 = 1_000;
        let shard = ShardId::ROOT;

        let (snap_a, keys_a) = committee_snapshot(&[0, 1, 2, 3]);
        let (snap_b, keys_b) = committee_snapshot(&[4, 5, 6, 7]);
        assert_ne!(keys_a, keys_b, "committees must have distinct keys");

        let mut schedule = TopologySchedule::new(ED, Epoch::new(0), Arc::new(snap_a));
        schedule.insert(Epoch::new(1), Arc::new(snap_b));

        let mut coord = make_test_state();
        let cert = ExecutionCertificate::new(
            TickId::new(shard, BlockHeight::new(1)),
            WeightedTimestamp::from_millis(ED), // epoch 1
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        let actions = coord.on_execution_certificate(&schedule, Verifiable::from(cert));
        let public_keys = actions
            .iter()
            .find_map(|a| match a {
                Action::VerifyExecutionCertificateSignature { public_keys, .. } => {
                    Some(public_keys)
                }
                _ => None,
            })
            .expect("on_execution_certificate dispatches a signature verification");
        assert_eq!(
            *public_keys, keys_b,
            "remote EC must verify against the committee at its vote_anchor_ts, not the head",
        );
    }

    /// The leader-side counterpart to the test above: when a local vote quorum
    /// forms, `check_vote_quorum` packs the committee at the tick's
    /// `vote_anchor_ts` into the EC. That committee positions the signer
    /// bitfield every verifier later resolves, so it must be the epoch-1
    /// committee (the tick is anchored there), not the head.
    #[test]
    fn local_aggregation_packs_committee_at_vote_anchor() {
        const ED: u64 = 1_000;
        let shard = ShardId::ROOT;

        let (snap_a, _keys_a) = committee_snapshot(&[0, 1, 2, 3]);
        let (snap_b, _keys_b) = committee_snapshot(&[4, 5, 6, 7]);
        let committee_b: Vec<ValidatorId> = snap_b.committee_for_shard(shard).to_vec();
        let mut schedule = TopologySchedule::new(ED, Epoch::new(0), Arc::new(snap_a));
        schedule.insert(Epoch::new(1), Arc::new(snap_b));

        // Local node is a committee-B member; commit a local-only tick anchored
        // in epoch 1 (block weighted timestamp = ED, so vote_anchor_ts = ED).
        let mut coord = make_test_state_for(ValidatorId::new(4));
        let block = make_live_block(
            BlockHeight::new(1),
            ED,
            ValidatorId::new(4),
            vec![Arc::new(test_transaction(1))],
        );
        let block_hash = block.hash();
        coord.on_block_committed(&schedule, &test_certify(block, ED));
        let tick_id = coord
            .ticks
            .ticks_iter()
            .next()
            .map(|(w, _)| *w)
            .expect("local-only tick created on commit");

        // Feed a 2f+1 quorum of verified votes from committee B, all sharing the
        // tick's anchor and receipt root so they land in one quorum bucket.
        let mut actions = Vec::new();
        for v in [4u64, 5, 6] {
            let vote = ExecutionVote::new(
                block_hash,
                BlockHeight::new(1),
                WeightedTimestamp::from_millis(ED),
                tick_id,
                shard,
                GlobalReceiptRoot::ZERO,
                1,
                vec![],
                ValidatorId::new(v),
                ConsensusSignature::ZERO,
            );
            actions.extend(
                coord.on_verified_execution_vote(&schedule, Verified::new_unchecked_for_test(vote)),
            );
        }

        let committee = actions
            .iter()
            .find_map(|a| match a {
                Action::AggregateExecutionCertificate { committee, .. } => Some(committee),
                _ => None,
            })
            .expect("vote quorum dispatches certificate aggregation");
        assert_eq!(
            *committee, committee_b,
            "the EC's bitfield committee must be the one at vote_anchor_ts (epoch 1), not the head",
        );
    }

    #[test]
    fn cross_shard_ec_buffers_when_beacon_behind_then_drains_on_catch_up() {
        const ED: u64 = 1_000;
        let shard = ShardId::ROOT;

        // Schedule head is epoch 0; an EC anchored in epoch 5 is ahead of this
        // node's beacon and can't resolve yet.
        let behind = TopologySchedule::new(
            ED,
            Epoch::new(0),
            Arc::new(committee_snapshot(&[0, 1, 2, 3]).0),
        );
        let mut coord = make_test_state();
        let cert = ExecutionCertificate::new(
            TickId::new(shard, BlockHeight::new(1)),
            WeightedTimestamp::from_millis(5 * ED), // epoch 5, past the head
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        let actions = coord.on_execution_certificate(&behind, Verifiable::from(cert));
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::VerifyExecutionCertificateSignature { .. })),
            "an EC whose epoch the beacon hasn't reached must buffer, not dispatch",
        );

        let caught_up = TopologySchedule::single(Arc::new(committee_snapshot(&[0, 1, 2, 3]).0));
        let drained = coord.on_beacon_block_persisted(&caught_up);
        assert!(
            drained
                .iter()
                .any(|a| matches!(a, Action::VerifyExecutionCertificateSignature { .. })),
            "draining on catch-up must dispatch the buffered EC's verification",
        );
    }

    #[test]
    fn finalization_buffers_when_beacon_behind_then_drains_on_catch_up() {
        const ED: u64 = 1_000;
        let shard = ShardId::ROOT;

        let behind = TopologySchedule::new(
            ED,
            Epoch::new(0),
            Arc::new(committee_snapshot(&[0, 1, 2, 3]).0),
        );
        let mut coord = make_test_state();

        let tick_id = TickId::new(shard, BlockHeight::new(1));
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        let ec = Arc::new(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::from_millis(5 * ED), // epoch 5, past the head
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            signers,
        ));
        let tick: Arc<Verifiable<Finalization>> =
            Arc::new(Finalization::new(tick_id, TickHalf::Determined, vec![ec], vec![]).into());
        let _fw_hash = tick.receipt_hash();

        let actions = coord.admit_finalization(&behind, Arc::clone(&tick));
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::VerifyFinalization { .. })),
            "a finalization whose EC epoch the beacon hasn't reached must buffer, not dispatch",
        );

        let caught_up = TopologySchedule::single(Arc::new(committee_snapshot(&[0, 1, 2, 3]).0));
        let drained = coord.on_beacon_block_persisted(&caught_up);
        assert!(
            drained
                .iter()
                .any(|a| matches!(a, Action::VerifyFinalization { .. })),
            "draining on catch-up must dispatch the buffered tick's verification",
        );
    }

    /// Expected-cert entries must be retained while a local tick still holds
    /// their transaction — otherwise a cross-shard transaction whose remote
    /// EC missed the broadcast window would be stranded once the expectation
    /// aged out, with no fallback fetch continuing to fire.
    #[test]
    fn test_expected_exec_cert_retained_while_tracker_pending() {
        use std::collections::BTreeSet;

        use hyperscale_types::test_utils::test_transaction;

        let _topo = make_two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let remote_shard = ShardId::leaf(1, 1);
        let tx = Arc::new(test_transaction(7));
        let tx_hash = tx.hash();
        state
            .expected_certs
            .register(remote_shard, tx_hash, state.committed_ts);
        assert_eq!(
            state.expected_certs.expected_len(),
            1,
            "expectation should register for a transaction the source names"
        );

        // Simulate an outstanding local cross-shard tick needing shard 1's EC.
        let local_tick = TickId::new(ShardId::leaf(1, 0), BlockHeight::new(10));
        let mut participating = BTreeSet::new();
        participating.insert(ShardId::leaf(1, 0));
        participating.insert(remote_shard);
        state.ticks.insert_tick(
            local_tick,
            tick_holding(
                local_tick,
                WeightedTimestamp::from_millis(5_000),
                vec![(verified_arc(&tx), participating)],
            ),
        );
        state.ticks.assign_tx(tx_hash, local_tick);

        // Advance committed time past fallback + retry thresholds so the
        // age-based gate would fire. The expectation must survive regardless
        // because a local tick still needs shard 1's EC.
        state.committed_height = BlockHeight::new(500);
        state.committed_ts = WeightedTimestamp::from_millis(60_000);
        let actions = state.check_exec_cert_timeouts();

        assert_eq!(
            state.expected_certs.expected_len(),
            1,
            "expectation must survive age pruning while a local tick still needs shard 1"
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::Fetch(FetchRequest::ExecutionCerts { .. }))),
            "fallback fetch must keep firing while the expectation is retained"
        );

        // Once the local tick resolves (simulating finalize), the
        // expectation is no longer needed and gets pruned.
        state.ticks.remove_tick(&local_tick);
        state.ticks.remove_assignment(tx_hash);
        state.committed_height = BlockHeight::new(600);
        state.committed_ts = WeightedTimestamp::from_millis(120_000);
        let _ = state.check_exec_cert_timeouts();
        assert_eq!(
            state.expected_certs.expected_len(),
            0,
            "expectation must be pruned once no tick needs the source shard"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // finalize — verifies the critical cross-sub-machine fanout that
    // happens when a tick reaches terminal state.
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build a local-only tick in the "ready to finalize" state: every tx
    /// has an execution result and a local receipt, and the local EC has
    /// been added. `TickState::is_complete` returns true.
    fn make_ready_local_tick(tx_seeds: &[u8]) -> (TickId, TickState) {
        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        let txs: Vec<(Arc<Verified<Transaction>>, BTreeSet<ShardId>)> = tx_seeds
            .iter()
            .map(|s| {
                let mut participating = BTreeSet::new();
                participating.insert(ShardId::ROOT);
                (
                    Arc::new(Verified::new_unchecked_for_test(test_transaction(*s))),
                    participating,
                )
            })
            .collect();
        let mut tick = tick_holding(tick_id, WeightedTimestamp::from_millis(1_000), txs);

        // Record per-tx execution results + receipts.
        let tx_hashes: Vec<TxHash> = tick.tx_hashes().to_vec();
        let tx_outcomes: Vec<TxOutcome> = tx_hashes
            .iter()
            .map(|h| {
                TxOutcome::new(
                    *h,
                    ExecutionOutcome::Succeeded {
                        receipt_hash: GlobalReceiptHash::ZERO,
                    },
                )
            })
            .collect();
        for h in &tx_hashes {
            tick.record_execution_result(
                *h,
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                },
            );
            tick.record_receipt(StoredReceipt {
                tx_hash: *h,
                consensus: Arc::new(ConsensusReceipt::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                    #[allow(clippy::default_trait_access)]
                    writes: Default::default(),
                    beacon_witness_events: Vec::new(),
                    events: Vec::new(),
                }),
                metadata: None,
            });
        }

        // Add the local EC; same tick_id flips `local_ec_emitted` to true.
        let local_ec = Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::from_millis(1_000),
            GlobalReceiptRoot::from_raw(Hash::from_bytes(b"global_receipt_root")),
            tx_outcomes,
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        )));
        tick.add_execution_certificate(local_ec);
        assert!(
            tick.determined_ready(),
            "fixture precondition: a purely local tick settles on its own certificate"
        );

        (tick_id, tick)
    }

    #[test]
    fn finalize_populates_the_finalization_store() {
        let mut state = make_test_state();
        let (tick_id, tick) = make_ready_local_tick(&[1, 2]);
        state.ticks.insert_tick(tick_id, tick);

        let _actions = state.finalize(&make_test_topology(), &tick_id);

        assert!(
            state.finalized.contains(&tick_id),
            "finalized store populated"
        );
        assert_eq!(state.finalized.len(), 1);
        // The tick outlives its handoff: it is what tells a committing
        // block whether the members it resolved were all of them.
        assert!(state.ticks.contains_tick(&tick_id));
        assert!(
            state
                .ticks
                .get_tick(&tick_id)
                .is_some_and(TickState::has_spoken),
            "a purely local tick has nothing left to say",
        );
    }

    #[test]
    fn finalize_emits_the_admission_event() {
        let mut state = make_test_state();
        let (tick_id, tick) = make_ready_local_tick(&[1, 2]);
        state.ticks.insert_tick(tick_id, tick);

        let actions = state.finalize(&make_test_topology(), &tick_id);

        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::Continuation(ProtocolEvent::FinalizationsAdmitted { .. })
        ));
    }

    #[test]
    fn finalize_is_a_noop_for_an_absent_tick() {
        let mut state = make_test_state();
        let unknown = TickId::new(ShardId::ROOT, BlockHeight::new(99));
        let actions = state.finalize(&make_test_topology(), &unknown);
        assert!(actions.is_empty());
        assert!(state.finalized.is_empty());
    }

    /// A schedule whose final window (epoch 0) is single-shard `ROOT` and
    /// whose next window (epoch 1) splits it into two children — so any
    /// weighted timestamp in epoch 1 is past `ROOT`'s terminal window.
    fn terminating_schedule() -> TopologySchedule {
        terminating_schedule_over(1000)
    }

    /// A schedule in which `ShardId::ROOT` splits at the end of epoch 0, so
    /// it is past-terminal anywhere in epoch 1 and its two children are
    /// live there. `epoch_duration_ms` places the cut on the weighted-time
    /// grid: a caller reaching past a transaction deadline needs windows
    /// wide enough to still resolve there.
    fn terminating_schedule_over(epoch_duration_ms: u64) -> TopologySchedule {
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        // The cut rides the final window's own entry — the boundary
        // predicates read the scheduled terminal rather than comparing
        // this window's trie against the next.
        let final_window = Arc::new(
            TopologySnapshot::new(
                NetworkDefinition::simulator(),
                1,
                ValidatorSet::new(validators.clone()),
            )
            .with_scheduled_terminals(BTreeMap::from([(ShardId::ROOT, Epoch::new(0))])),
        );
        let post_split = Arc::new(
            TopologySnapshot::new(
                NetworkDefinition::simulator(),
                2,
                ValidatorSet::new(validators),
            )
            .with_boundaries(HashMap::from([(
                ShardId::ROOT,
                ShardAnchor {
                    state_root: StateRoot::ZERO,
                    block_hash: BlockHash::from_raw(Hash::from_bytes(b"terminal")),
                    height: BlockHeight::new(9),
                    weighted_timestamp: WeightedTimestamp::from_millis(1_000),
                    witness_base: BeaconWitnessLeafCount::ZERO,
                    terminal_roots: None,
                    handoff_complete: None,
                },
            )])),
        );
        let mut sched = TopologySchedule::new(epoch_duration_ms, Epoch::new(0), final_window);
        sched.insert(Epoch::new(1), post_split);
        sched
    }

    /// An empty block at `height` whose parent QC carries `anchor_ms` — the
    /// block's own position on the weighted-time grid.
    /// Commit-time classification resolves against the window the block's
    /// *parent* anchored in, because that is where a block's committee comes
    /// from — so it matches the snapshot the proposer built under and the
    /// verifier validated against. Reading the block's own anchor instead
    /// straddles an epoch cut once per window, and a reshape cut there
    /// changes the shard set `compute_ticks` routes over.
    #[test]
    fn classification_anchors_on_the_parents_window_across_commits() {
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::ROOT);
        let topo = make_test_topology();

        // `certify` stamps the block's parent-QC weighted timestamp, which is
        // the block's own anchor.
        let commit = |state: &mut ExecutionCoordinator, height: u64, anchor_ms: u64| {
            let block = make_live_block_on_shard(
                ShardId::ROOT,
                BlockHeight::new(height),
                anchor_ms,
                ValidatorId::new(0),
                vec![],
            );
            let _ = state.on_block_committed(&topo, &test_certify(block, anchor_ms));
        };

        commit(&mut state, 1, 500);
        commit(&mut state, 2, 1_500);

        assert_eq!(
            state.committed_committee_anchor_wt,
            WeightedTimestamp::from_millis(500),
            "the second block is classified in the window its parent anchored in",
        );
        assert_eq!(
            state.committed_ts,
            WeightedTimestamp::from_millis(1_500),
            "its own anchor still drives the deterministic clock",
        );
    }

    /// A root chain's first blocks genuinely anchor at zero, and that zero
    /// is a carriable parent anchor, not an uninitialized clock: the second
    /// commit classifies in the window its parent anchored (epoch 0) even
    /// when the chain stalled long enough at genesis for the block to date
    /// itself past a cut.
    #[test]
    fn a_zero_anchor_at_genesis_is_carried_not_treated_as_a_gap() {
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::ROOT);
        let topo = make_test_topology();

        let commit = |state: &mut ExecutionCoordinator, height: u64, anchor_ms: u64| {
            let block = make_live_block_on_shard(
                ShardId::ROOT,
                BlockHeight::new(height),
                anchor_ms,
                ValidatorId::new(0),
                vec![],
            );
            let _ = state.on_block_committed(&topo, &test_certify(block, anchor_ms));
        };

        // Block 1 anchors at genesis zero; block 2 dates itself far past it.
        commit(&mut state, 1, 0);
        commit(&mut state, 2, 1_500);

        assert_eq!(
            state.committed_committee_anchor_wt,
            WeightedTimestamp::ZERO,
            "block 2's committee anchors on block 1, at genesis zero",
        );
    }

    /// A restarted replica's first commit classifies exactly like a
    /// non-restarted peer's: the frontier seeded from the recovered tip is
    /// the parent anchor the carry needs, so the commit extending the tip
    /// resolves the window the tip anchored — not the one the new block
    /// opens, which a zero frontier's gap fallback would pick and which a
    /// reshape cut between the two turns into a vote-splitting divergence.
    #[test]
    fn a_seeded_frontier_carries_the_recovered_anchor_across_a_restart() {
        let topo = make_test_topology();
        // The recovered tip: height 5, its own anchor at 900 ms.
        let mut state = ExecutionCoordinator::with_shared_stores(
            ValidatorId::new(0),
            ShardId::ROOT,
            &RecoveredState {
                committed_height: BlockHeight::new(5),
                committed_block_anchor_wt: Some(WeightedTimestamp::from_millis(900)),
                committed_committee_anchor_wt: Some(WeightedTimestamp::from_millis(800)),
                ..RecoveredState::default()
            },
            Arc::new(ExecCertStore::new()),
            Arc::new(FinalizationStore::new()),
            Arc::new(ProvenAnchors::new()),
            Arc::new(CounterpartMirror::new()),
        );

        // The first post-restart commit extends the tip and dates itself
        // past it.
        let block = make_live_block_on_shard(
            ShardId::ROOT,
            BlockHeight::new(6),
            1_100,
            ValidatorId::new(0),
            vec![],
        );
        let _ = state.on_block_committed(&topo, &test_certify(block, 1_100));

        assert_eq!(
            state.committed_committee_anchor_wt,
            WeightedTimestamp::from_millis(900),
            "the commit extending the recovered tip is classified in the tip's window",
        );
        assert_eq!(state.committed_ts, WeightedTimestamp::from_millis(1_100));
    }

    /// Commit-time tick/provision classification anchors on the block's
    /// committee — not the `ArcSwap` head, so every replica groups a block's
    /// transactions identically across a reshape boundary (matching the
    /// proposer and the verifier).
    #[test]
    fn classification_committee_anchors_at_the_block_window_not_the_head() {
        let state = make_test_state_for_shard(ValidatorId::new(0), ShardId::ROOT);
        // Epoch 0 carries ROOT (one shard) — the block's anchor; epoch 1
        // splits it (two shards) and is installed as the flipped head.
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        let pre_split = Arc::new(TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            ValidatorSet::new(validators.clone()),
        ));
        let post_split = Arc::new(TopologySnapshot::new(
            NetworkDefinition::simulator(),
            2,
            ValidatorSet::new(validators),
        ));
        let mut sched = TopologySchedule::new(1000, Epoch::new(0), Arc::clone(&pre_split));
        sched.insert(Epoch::new(1), Arc::clone(&post_split));
        sched.set_head(post_split);

        let anchored = state.classification_committee(&sched, WeightedTimestamp::from_millis(500));
        assert_eq!(
            anchored.num_shards(),
            1,
            "classification anchors at the block's window (ROOT, one shard), not the two-shard head",
        );
    }

    /// The seed window a tick executes under is the one its block's
    /// committee carried, not the one this node's head has folded to.
    ///
    /// A head advances as a node folds the beacon and every node folds
    /// at its own pace, so a window read off the head would answer
    /// `Pending` on a laggard where it answers `Ready` on a leader —
    /// one tick, two receipt roots. The block fixes it, on the same
    /// terms as the clock beside it.
    #[test]
    fn a_tick_executes_under_its_blocks_seed_window_not_the_head() {
        const ED: u64 = 1_000;

        let mut state = make_test_state();
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        let seeded = |byte: u8| {
            let mut ring = SeedRing::default();
            ring.record(
                Epoch::GENESIS,
                EpochSeed {
                    randomness: Randomness::new([byte; 32]),
                    source: SeedSource::Reveals,
                },
            );
            Arc::new(
                TopologySnapshot::new(
                    NetworkDefinition::simulator(),
                    1,
                    ValidatorSet::new(validators.clone()),
                )
                .with_seeds(ring),
            )
        };
        // One window, two snapshots of it: what the block's committee
        // carried, and what a node further along the fold holds.
        let mut sched = TopologySchedule::new(ED, Epoch::GENESIS, seeded(0xA1));
        sched.set_head(seeded(0xB2));

        let block = make_live_block(
            BlockHeight::new(1),
            500,
            ValidatorId::new(0),
            vec![Arc::new(test_transaction(1))],
        );
        let actions = state.on_block_committed(&sched, &certify(block));
        let env = actions
            .iter()
            .find_map(|action| match action {
                Action::ExecuteTransactions { env, .. } => Some(env),
                _ => None,
            })
            .expect("the commit dispatches its tick");

        assert_eq!(
            env.seeds.at(Epoch::GENESIS.inner()),
            Seeded::Ready([0xA1; 32]),
            "the tick reads the seed its block's committee carried",
        );
        // And the grid is the schedule's own window length, which is
        // what the schedule is indexed by — not a default a snapshot
        // that nobody projected would have answered with.
        assert_eq!(
            env.windows,
            EpochWindows::new(ED),
            "the tick resolves its clock on the chain's epoch grid",
        );
    }

    /// A finalization whose certificate carries `local`'s EC plus a
    /// remote EC on `remote` — the cross-shard shape the gate inspects.
    fn cross_shard_finalization(
        local: ShardId,
        remote: ShardId,
        height: u64,
        tx_hash: TxHash,
    ) -> Arc<Verifiable<Finalization>> {
        let ec = |shard: ShardId| {
            let tick = TickId::new(shard, BlockHeight::new(height));
            ExecutionCertificate::new(
                tick,
                WeightedTimestamp::from_millis(height),
                GlobalReceiptRoot::ZERO,
                vec![TxOutcome::new(
                    tx_hash,
                    ExecutionOutcome::Succeeded {
                        receipt_hash: GlobalReceiptHash::ZERO,
                    },
                )],
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            )
        };
        let local_tick = TickId::new(local, BlockHeight::new(height));
        let tick = Finalization::new(
            local_tick,
            TickHalf::Determined,
            vec![Arc::new(ec(local)), Arc::new(ec(remote))],
            vec![],
        );
        Arc::new(Verified::new_unchecked_for_test(tick).into())
    }

    /// A tick naming a past-terminal shard whose settled set is unknown is
    /// withheld at the finalize gate, then released once the set records
    /// it — the produce-side mirror of the vote fence's defer-and-release.
    #[test]
    fn finalize_gate_defers_then_releases() {
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));
        // Past ROOT's terminal window — the gate's anchor is the committed ts.
        state.committed_ts = WeightedTimestamp::from_millis(1500);
        let sched = terminating_schedule();
        let tick = cross_shard_finalization(
            ShardId::leaf(1, 0),
            ShardId::ROOT,
            1,
            TxHash::from(Hash::from_bytes(b"tx")),
        );
        let tick_id = *tick.tick_id();

        let deferred = state.emit_or_gate_finalized(&sched, tick);
        assert!(
            deferred.is_empty(),
            "the gate withholds the tick while the settled set is unknown",
        );
        assert_eq!(state.gated_finalized.len(), 1, "held at the gate");
        assert!(!state.finalized.contains(&tick_id));

        state.record_settled_txs(
            &sched,
            ShardId::ROOT,
            SettledTxSet {
                txs: std::iter::once(TxHash::from(Hash::from_bytes(b"tx"))).collect(),
                terminal_wt: WeightedTimestamp::from_millis(1000),
            },
        );
        let released = state.redrive_gated_finalizations(&sched);
        assert!(
            matches!(
                released.as_slice(),
                [Action::Continuation(
                    ProtocolEvent::FinalizationsAdmitted { .. }
                )],
            ),
            "recording the settled set releases the held tick for admission",
        );
        assert!(state.gated_finalized.is_empty());
        assert!(state.finalized.contains(&tick_id));
    }

    /// A tick a past-terminal shard never settled is dropped, not produced
    /// and not buffered for retry. Nothing here resolves its transaction:
    /// it stays owed, and it goes back to the deadline path that can.
    #[test]
    fn finalize_gate_drops_an_unsettled_tick() {
        let local = ShardId::leaf(1, 0);
        let mut state = make_test_state_for_shard(ValidatorId::new(0), local);
        state.committed_ts = WeightedTimestamp::from_millis(1500);
        let sched = terminating_schedule();
        state.record_settled_txs(
            &sched,
            ShardId::ROOT,
            SettledTxSet {
                txs: BTreeSet::new(),
                terminal_wt: WeightedTimestamp::from_millis(1000),
            },
        );
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(test_transaction(9)),
        ));
        let tx_hash = transaction.hash();
        let tick = cross_shard_finalization(local, ShardId::ROOT, 1, tx_hash);
        let tick_id = *tick.tick_id();
        state.unresolved.register_committed(
            local,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        state.ticks.assign_tx(tx_hash, tick_id);

        let dropped = state.emit_or_gate_finalized(&sched, tick);
        assert!(
            dropped.is_empty(),
            "the gate drops a tick the terminated shard never settled",
        );
        assert!(
            state.gated_finalized.is_empty(),
            "a rejected tick is not buffered for retry",
        );
        assert!(!state.finalized.contains(&tick_id));
        assert_eq!(
            state.unresolved.len(),
            1,
            "a gate rejection is not a verdict — the transaction stays owed",
        );
        assert!(
            state.ticks.tick_assignment(tx_hash).is_none(),
            "and its tick has stopped speaking for it, so the deadline path can",
        );
    }

    /// A late execution certificate completes a tick naming a terminated
    /// partner *after* the cut. While the
    /// partner's settled set is unknown the gate **defers** (never emits —
    /// no one-sided application); once the set proves the partner never
    /// settled the tick, the gate **rejects** it. The fence/gate
    /// defer-release that `reshape_sibling`'s natural straddler can't
    /// reach (it finalizes pre-cut) is exercised here against a genuinely
    /// post-cut, unsettled transaction.
    #[test]
    fn late_unsettled_ec_defers_then_rejects_no_one_sided() {
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));
        // Post-cut: any epoch-1 timestamp is past ROOT's terminal window.
        state.committed_ts = WeightedTimestamp::from_millis(1500);
        let sched = terminating_schedule();
        let tick = cross_shard_finalization(
            ShardId::leaf(1, 0),
            ShardId::ROOT,
            1,
            TxHash::from(Hash::from_bytes(b"tx")),
        );
        let tick_id = *tick.tick_id();

        // The late certificate completes the tick before the settled set is
        // reconstructed: the gate defers — held, never emitted.
        let deferred = state.emit_or_gate_finalized(&sched, tick);
        assert!(
            deferred.is_empty(),
            "no one-sided finalize while the partner's settled set is unknown",
        );
        assert_eq!(state.gated_finalized.len(), 1, "held at the gate");

        // ROOT terminated having settled nothing → the held tick rejects on
        // redrive, is never finalized, and its tx aborts (not wedged).
        state.record_settled_txs(
            &sched,
            ShardId::ROOT,
            SettledTxSet {
                txs: BTreeSet::new(),
                terminal_wt: WeightedTimestamp::from_millis(1000),
            },
        );
        let released = state.redrive_gated_finalizations(&sched);
        assert!(
            released.is_empty(),
            "an unsettled transaction is never finalized — no one-sided application",
        );
        assert!(state.gated_finalized.is_empty());
        assert!(!state.finalized.contains(&tick_id));
    }

    /// A gate-held tick is never dropped on a clock: held past its own
    /// execution anchor by `RETENTION_HORIZON` while its partner's
    /// scheduled termination still stands, it stays held — the partner may
    /// yet prove it settled the tick, and a deadline abort would contradict
    /// that settlement. Only the schedule evicting the partner from every
    /// retained window rejects it, so the buffer cannot pin forever.
    #[test]
    fn a_gate_held_tick_survives_the_horizon_until_schedule_eviction() {
        // Windows long enough that the commit clock can pass the tick's
        // anchor plus the horizon while the pre-terminal window still
        // governs — the shape a terminating shard's multi-epoch coast has
        // at production epoch length.
        let epoch_ms = 2 * RETENTION_HORIZON.as_secs() * 1000;
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        // The cut rides the final window's own entry — the boundary
        // predicates read the scheduled terminal rather than comparing
        // this window's trie against the next.
        let final_window = Arc::new(
            TopologySnapshot::new(
                NetworkDefinition::simulator(),
                1,
                ValidatorSet::new(validators.clone()),
            )
            .with_scheduled_terminals(BTreeMap::from([(ShardId::ROOT, Epoch::new(0))])),
        );
        let post_split = Arc::new(TopologySnapshot::new(
            NetworkDefinition::simulator(),
            2,
            ValidatorSet::new(validators),
        ));
        let mut sched = TopologySchedule::new(epoch_ms, Epoch::new(0), final_window);
        sched.insert(Epoch::new(1), Arc::clone(&post_split));

        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));
        state.committed_ts = WeightedTimestamp::from_millis(1500);
        let tick = cross_shard_finalization(
            ShardId::leaf(1, 0),
            ShardId::ROOT,
            1,
            TxHash::from(Hash::from_bytes(b"tx")),
        );
        let tick_id = *tick.tick_id();

        // ROOT is live in epoch 0 but leaves the trie at its boundary: the
        // gate defers on the scheduled termination.
        let deferred = state.emit_or_gate_finalized(&sched, tick);
        assert!(
            deferred.is_empty(),
            "held while the partner is scheduled to terminate",
        );
        assert_eq!(state.gated_finalized.len(), 1, "held at the gate");

        // The settled set doesn't exist yet; the commit clock sails past
        // the tick's anchor (1ms) plus the horizon with epoch 0 still
        // governing. The hold must survive the clock.
        state.committed_ts = WeightedTimestamp::from_millis(2).plus(RETENTION_HORIZON);
        let released = state.redrive_gated_finalizations(&sched);
        assert!(released.is_empty(), "an unresolved tick is never finalized");
        assert!(
            !state.gated_finalized.is_empty(),
            "a gate-held tick is never dropped on a clock",
        );

        // ROOT falls out of every retained window: no honest artifact can
        // resolve the tick anymore, so the redrive rejects it.
        let evicted = TopologySchedule::new(epoch_ms, Epoch::new(0), post_split);
        let released = state.redrive_gated_finalizations(&evicted);
        assert!(released.is_empty(), "an unresolved tick is never finalized");
        assert!(state.gated_finalized.is_empty());
        assert!(!state.finalized.contains(&tick_id));
    }

    /// The transactions the ledger names past their deadline, and what
    /// this shard's next tick attests about them.
    fn abandonment_vote(
        state: &mut ExecutionCoordinator,
        schedule: &TopologySchedule,
        height: u64,
        now_ms: u64,
    ) -> Vec<TxOutcome> {
        let block = make_live_block(
            BlockHeight::new(height),
            now_ms,
            ValidatorId::new(0),
            vec![],
        );
        state.on_block_committed(schedule, &test_certify(block, now_ms));
        state
            .scan_votable_ticks(schedule)
            .into_iter()
            .flat_map(|completion| completion.tx_outcomes)
            .collect()
    }

    /// Committed and never resolved, a transaction is abandoned by the
    /// tick composed at the first commit past its deadline — attested
    /// `Aborted`, carrying the reservation its own block took, on a tick
    /// no tick stands behind.
    #[test]
    fn a_transaction_past_its_deadline_is_attested_aborted() {
        let schedule = make_test_topology();
        let mut state = make_test_state();
        let tx = test_transaction(1);
        let tx_hash = tx.hash();
        let reserved = tx.work();
        let deadline_ms = 60_000 + u64::try_from(MAX_FINALIZATION_DELAY.as_millis()).unwrap();

        state.on_block_committed(
            &schedule,
            &test_certify(
                make_live_block(
                    BlockHeight::new(1),
                    1_000,
                    ValidatorId::new(0),
                    vec![Arc::new(tx)],
                ),
                1_000,
            ),
        );
        assert_eq!(state.unresolved.len(), 1, "committed and owed an outcome");

        // Its own tick never finalizes; the shard drops it so nothing can
        // attest a second verdict for it.
        state
            .ticks
            .remove_tick(&TickId::new(ShardId::ROOT, BlockHeight::new(1)));

        let outcomes = abandonment_vote(&mut state, &schedule, 2, deadline_ms);
        assert_eq!(
            outcomes.len(),
            1,
            "the tick attests exactly what it abandons"
        );
        assert_eq!(outcomes[0].tx_hash(), tx_hash);
        assert!(outcomes[0].is_aborted(), "abandonment is an abort");
        assert_eq!(
            outcomes[0].declared_work(),
            reserved,
            "releasing exactly what the committing block reserved",
        );
    }

    /// The floor is burned by the abandonment too, which is what stops a
    /// transaction nobody could execute from aborting for free.
    ///
    /// An abandoned member never reaches an engine, so the charge its
    /// verdict settles is composed rather than executed — but the
    /// reservation engaged when its block committed it, and releasing
    /// that without a burn would price an unexecutable attempt below the
    /// success it was competing with.
    #[test]
    fn an_abandoned_transaction_still_settles_its_floor() {
        let schedule = make_test_topology();
        let mut state = make_test_state();
        let tx = test_transaction(1);
        let expected = build_fee_receipt(
            ShardId::ROOT,
            state.counterpart_trie(&schedule),
            tx.hash(),
            tx.fee_vault(),
            tx.price(),
        );
        let deadline_ms = 60_000 + u64::try_from(MAX_FINALIZATION_DELAY.as_millis()).unwrap();

        state.on_block_committed(
            &schedule,
            &test_certify(
                make_live_block(
                    BlockHeight::new(1),
                    1_000,
                    ValidatorId::new(0),
                    vec![Arc::new(tx)],
                ),
                1_000,
            ),
        );
        state
            .ticks
            .remove_tick(&TickId::new(ShardId::ROOT, BlockHeight::new(1)));

        let outcomes = abandonment_vote(&mut state, &schedule, 2, deadline_ms);
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].is_aborted());
        assert_eq!(
            outcomes[0].fee_receipt(),
            Some(expected.receipt_hash()),
            "the abandonment settles the same floor an attested abort would",
        );
    }

    /// And it is settled by the payer's own shard alone: fees never move
    /// cross-shard, so a participant abandoning a leg of somebody else's
    /// transaction charges nothing.
    #[test]
    fn an_abandonment_charges_nothing_where_the_vault_is_not() {
        let schedule = two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        // A payer whose prefix the other half of the split owns.
        let tx = test_transaction(0x81);
        assert_eq!(
            state
                .counterpart_trie(&schedule)
                .shard_for_prefix(tx.fee_vault().owner),
            PEER,
            "the fixture is only a test of this if the vault is elsewhere",
        );
        let deadline_ms = 60_000 + u64::try_from(MAX_FINALIZATION_DELAY.as_millis()).unwrap();

        state.on_block_committed(
            &schedule,
            &test_certify(
                make_live_block_on_shard(
                    HOME,
                    BlockHeight::new(1),
                    1_000,
                    ValidatorId::new(0),
                    vec![Arc::new(tx)],
                ),
                1_000,
            ),
        );
        state
            .ticks
            .remove_tick(&TickId::new(HOME, BlockHeight::new(1)));

        let block = make_live_block_on_shard(
            HOME,
            BlockHeight::new(2),
            deadline_ms,
            ValidatorId::new(0),
            vec![],
        );
        state.on_block_committed(&schedule, &test_certify(block, deadline_ms));
        let outcomes: Vec<TxOutcome> = state
            .scan_votable_ticks(&schedule)
            .into_iter()
            .flat_map(|completion| completion.tx_outcomes)
            .collect();

        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].is_aborted());
        assert_eq!(
            outcomes[0].fee_receipt(),
            None,
            "the vault is not here, so neither is the burn",
        );
    }

    /// A block a replay feeds back in, as storage hands it over.
    fn replayable(block: Block, now_ms: u64) -> Verified<CertifiedBlock> {
        Verified::<CertifiedBlock>::from_persisted(test_certify(block, now_ms))
    }

    /// The transactions a coordinator's tick at `height` holds.
    fn tick_members(state: &ExecutionCoordinator, height: u64) -> Vec<TxHash> {
        state
            .ticks
            .get_tick(&TickId::new(ShardId::ROOT, BlockHeight::new(height)))
            .map(|tick| tick.tx_hashes().to_vec())
            .unwrap_or_default()
    }

    /// A replay releases the ticks the blocks it re-drives finalized,
    /// exactly as a commit does.
    ///
    /// A replay recomposes the tick that held a transaction *and* commits
    /// the block whose finalization settled it, and the second is what
    /// hands the transaction back. Skipping it leaves the transaction
    /// assigned to a tick that has already settled, which nothing later
    /// clears — and a leg's reclaim, admitted only where no tick speaks
    /// for the transaction, is then held out for as long as its entry
    /// lives.
    #[test]
    fn a_replay_releases_what_the_blocks_it_replays_finalized() {
        let schedule = make_test_topology();
        let held = test_transaction(2);
        let held_hash = held.hash();
        let committing = make_live_block(
            BlockHeight::new(2),
            2_000,
            ValidatorId::new(0),
            vec![Arc::new(held)],
        );
        let finalization: Arc<Verifiable<Finalization>> = Arc::new(
            helpers_make_finalization(BlockHeight::new(2), held_hash, TransactionDecision::Accept)
                .into(),
        );
        let settling = helpers_make_live_block(
            ShardId::ROOT,
            BlockHeight::new(3),
            3_000,
            ValidatorId::new(0),
            vec![],
            vec![finalization],
        );

        let recovered = RecoveredState {
            committed_height: BlockHeight::new(3),
            replay: ReplayWindow {
                blocks: vec![replayable(committing, 2_000), replayable(settling, 3_000)],
                anchor_wt: Some(WeightedTimestamp::from_millis(1_000)),
            },
            ..RecoveredState::default()
        };
        let mut restarted = ExecutionCoordinator::with_shared_stores(
            ValidatorId::new(0),
            ShardId::ROOT,
            &recovered,
            Arc::new(ExecCertStore::new()),
            Arc::new(FinalizationStore::new()),
            Arc::new(ProvenAnchors::new()),
            Arc::new(CounterpartMirror::new()),
        );

        restarted.on_committed_state_restored(&schedule, &StubVmStatics);
        assert_eq!(
            restarted.ticks.tick_assignment(held_hash),
            None,
            "the replayed finalization hands the transaction back, so nothing \
             speaks for it",
        );
    }

    /// A restart replays the chain it lost execution state for, so it
    /// ends up holding what a replica that never went down holds.
    ///
    /// Which tick holds a transaction is a function of committed content,
    /// but not one anything on the chain records directly — it is
    /// composition's own output. Re-driving composition over the same
    /// blocks is what reproduces it.
    #[test]
    fn a_restart_replays_the_chain_it_lost_execution_state_for() {
        let schedule = make_test_topology();
        let held = test_transaction(1);
        let held_hash = held.hash();
        let seed = make_live_block(BlockHeight::new(1), 1_000, ValidatorId::new(0), vec![]);
        let committing = make_live_block(
            BlockHeight::new(2),
            2_000,
            ValidatorId::new(0),
            vec![Arc::new(held)],
        );

        // A replica that never went down: its tick at height 2 takes the
        // transaction and holds it until a finalization resolves it.
        let mut live = make_test_state();
        live.on_block_committed(&schedule, &test_certify(seed, 1_000));
        live.on_block_committed(&schedule, &test_certify(committing.clone(), 2_000));
        assert_eq!(
            live.ticks.tick_assignment(held_hash),
            Some(TickId::new(ShardId::ROOT, BlockHeight::new(2))),
            "fixture precondition: the live replica's tick holds it",
        );

        // A restarted one recovers the same chain: the block that
        // committed the transaction, under the clock of the block below
        // it so the replay stays on the carry path.
        let recovered = RecoveredState {
            committed_height: BlockHeight::new(2),
            replay: ReplayWindow {
                blocks: vec![replayable(committing, 2_000)],
                anchor_wt: Some(WeightedTimestamp::from_millis(1_000)),
            },
            ..RecoveredState::default()
        };
        let mut restarted = ExecutionCoordinator::with_shared_stores(
            ValidatorId::new(0),
            ShardId::ROOT,
            &recovered,
            Arc::new(ExecCertStore::new()),
            Arc::new(FinalizationStore::new()),
            Arc::new(ProvenAnchors::new()),
            Arc::new(CounterpartMirror::new()),
        );
        assert!(
            restarted.candidates.is_empty(),
            "construction has no topology to compose against, so it holds",
        );

        let actions = restarted.on_committed_state_restored(&schedule, &StubVmStatics);
        assert_eq!(
            restarted.ticks.tick_assignment(held_hash),
            live.ticks.tick_assignment(held_hash),
            "the replay puts it back on the tick that already held it",
        );
        assert!(
            actions.iter().any(
                |action| matches!(action, Action::ExecuteTransactions { requests, .. }
                    if requests.iter().any(|r| r.tx_hash == held_hash))
            ),
            "and runs it, because the tick that holds it is the tick that ran it",
        );
        assert_eq!(
            restarted.unresolved.len(),
            1,
            "still owed, as the chain says"
        );

        // The consequence: the next block composes one tick, not two
        // different ones. A membership disagreement here is a fail-stop —
        // the quorum's certificate comes back under this tick's own id,
        // carrying a root the odd replica never computed.
        let next = make_live_block(
            BlockHeight::new(3),
            3_000,
            ValidatorId::new(0),
            vec![Arc::new(test_transaction(2))],
        );
        live.on_block_committed(&schedule, &test_certify(next.clone(), 3_000));
        restarted.on_block_committed(&schedule, &test_certify(next, 3_000));
        assert_eq!(
            tick_members(&restarted, 3),
            tick_members(&live, 3),
            "a tick's membership decides what its certificate says",
        );
    }

    /// A tick whose receipts disagree with the quorum's still fail-stops.
    ///
    /// The replay closes the disagreement that a restart used to cause,
    /// upstream of this check rather than by loosening it — so the check
    /// has to still bite on the thing it exists for: a replica computing
    /// state nobody else can reproduce.
    #[test]
    #[should_panic(expected = "BFT CRITICAL")]
    fn a_divergent_fold_fail_stops() {
        let schedule = make_test_topology();
        let tx = test_transaction(1);
        let tx_hash = tx.hash();
        let mut state = make_test_state();
        state.on_block_committed(
            &schedule,
            &test_certify(
                make_live_block(
                    BlockHeight::new(1),
                    1_000,
                    ValidatorId::new(0),
                    vec![Arc::new(tx)],
                ),
                1_000,
            ),
        );

        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        state.on_execution_batch_completed(
            &schedule,
            BlockHeight::new(1),
            TickBatchOutcome {
                tick_id,
                results: Vec::new(),
                tx_outcomes: vec![TxOutcome::new(
                    tx_hash,
                    ExecutionOutcome::Succeeded {
                        receipt_hash: GlobalReceiptHash::ZERO,
                    },
                )],
                fee_receipts: Vec::new(),
                attested_work: Vec::new(),
            },
        );
        state.emit_vote_actions(&schedule);

        // The committee certified a different root for the tick this
        // replica voted on.
        state
            .ticks
            .get_tick_mut(&tick_id)
            .expect("the tick is still tracked")
            .add_execution_certificate(Arc::new(Verified::new_unchecked_for_test(
                ExecutionCertificate::new(
                    tick_id,
                    WeightedTimestamp::from_millis(1_000),
                    GlobalReceiptRoot::from_raw(Hash::from_bytes(b"quorum")),
                    vec![TxOutcome::new(
                        tx_hash,
                        ExecutionOutcome::Succeeded {
                            receipt_hash: GlobalReceiptHash::ZERO,
                        },
                    )],
                    AggregateSignature::ZERO,
                    SignerBitfield::new(4),
                ),
            )));
        state.scan_votable_ticks(&schedule);
    }

    /// A member that never ran joins on the shards holding the keyspace
    /// it reaches, not on this one alone.
    ///
    /// That is what routes the tick's certificate to the counterparts
    /// still owed a verdict for it: an abort is dominant, so their
    /// coverage closes on ours and neither side is left waiting on a
    /// transaction the other has already given up on.
    #[test]
    fn an_undispatched_member_joins_on_the_shards_holding_its_keyspace() {
        let schedule = two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let deadline_ms = 60_000 + u64::try_from(MAX_FINALIZATION_DELAY.as_millis()).unwrap();

        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );

        let block = make_live_block_on_shard(
            HOME,
            BlockHeight::new(1),
            deadline_ms,
            ValidatorId::new(0),
            vec![],
        );
        state.on_block_committed(&schedule, &test_certify(block, deadline_ms));
        let outcomes: Vec<TxOutcome> = state
            .scan_votable_ticks(&schedule)
            .into_iter()
            .flat_map(|completion| completion.tx_outcomes)
            .collect();
        assert_eq!(outcomes.len(), 1, "the tick speaks for it");
        assert!(outcomes[0].is_aborted());

        let tick = state
            .ticks
            .get_tick(&TickId::new(HOME, BlockHeight::new(1)))
            .expect("the commit past the deadline composed a tick for it");
        assert_eq!(
            tick.counterpart_shards(),
            vec![PEER],
            "so the certificate reaches the shard still waiting on it",
        );
        assert_eq!(
            tick.awaiting_tx_hashes().collect::<Vec<_>>(),
            vec![tx_hash],
            "and it settles as the leg it is, not as a determined member",
        );
    }

    /// A committed record naming a leg entry licenses its reclaim: the
    /// next commit composes the reclaim into its tick as a dispatched
    /// member running no node, awaiting nobody, reserving nothing — and
    /// never as an abandonment, whatever the clock reads.
    #[test]
    fn a_record_naming_a_leg_composes_its_reclaim() {
        let schedule = two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let past_deadline_ms = 60_000 + u64::try_from(MAX_FINALIZATION_DELAY.as_millis()).unwrap();

        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        state.unresolved.mark_leg(
            tx_hash,
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            leg_classified(),
            Vec::new(),
            Vec::new(),
        );
        state.unresolved.certify(tx_hash);
        state
            .unresolved
            .release_resolved(&[Arc::new(Verifiable::from(make_leg_finalization(
                BlockHeight::new(1),
                tx_hash,
            )))]);
        state
            .unresolved
            .record_abandonment_records(&[AbandonmentRecord::departed(
                PEER,
                WeightedTimestamp::from_millis(1_000),
                [UnsettledTx::for_transaction(&transaction)],
            )]);

        let block = make_live_block_on_shard(
            HOME,
            BlockHeight::new(1),
            past_deadline_ms,
            ValidatorId::new(0),
            vec![],
        );
        let actions = state.on_block_committed(&schedule, &test_certify(block, past_deadline_ms));

        let tick = state
            .ticks
            .get_tick(&TickId::new(HOME, BlockHeight::new(1)))
            .expect("the commit composed the reclaim into a tick");
        assert_eq!(
            tick.determined_members(),
            vec![tx_hash],
            "it settles on this shard's certificate alone"
        );
        assert_eq!(tick.awaited_counterparts().count(), 0);
        let request = actions
            .iter()
            .find_map(|action| match action {
                Action::ExecuteTransactions { requests, .. } => {
                    requests.iter().find(|request| request.tx_hash == tx_hash)
                }
                _ => None,
            })
            .expect("the reclaim is dispatched to the engine");
        assert!(
            matches!(request.runs, Runs::Reclaim { charged: true, .. }),
            "the leg's finalization committed here, so its certificate settled the price"
        );
        assert!(!request.abortable, "nothing retracts a reclaim");
        assert!(
            state.unresolved.reclaimable().is_empty(),
            "and the ledger has handed it to the tick"
        );
    }

    /// A verdict the chain committed reaches the refusal mirror on a
    /// replica that never heard the broadcast.
    ///
    /// This is the restart hole closing: the mirror is fed by
    /// certificate broadcast and nothing rebuilds it at startup, so
    /// before the chain carried the commitment a replica that came up
    /// between a core's refusal and the record's proposal could neither
    /// offer the record nor check one. Folding the claim gives it the
    /// same answer its peers hold, from the block alone.
    #[test]
    fn a_committed_verdict_reaches_a_replica_that_heard_no_broadcast() {
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        state.unresolved.mark_leg(
            tx_hash,
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            leg_classified(),
            Vec::new(),
            Vec::new(),
        );
        state.unresolved.certify(tx_hash);
        assert!(
            state.evidence.refusals().is_empty(),
            "nothing was broadcast to this replica"
        );

        let anchor = WeightedTimestamp::from_millis(7_000);
        let digest = Hash::from_bytes(b"digest");
        let verdict = VerdictClaim {
            shard: PEER,
            tx_hash,
            anchor_ts: anchor,
            decision: TransactionDecision::Reject,
            digest,
        };
        let actions = state.fold_verdict(&verdict);

        assert_eq!(
            state.evidence.refusal(tx_hash, PEER),
            Some(Refusal {
                refused_wt: anchor,
                deadline: UnsettledTx::for_transaction(&transaction).deadline,
                decision: TransactionDecision::Reject,
                digest,
            }),
            "the chain's own word reaches the mirror the record fence reads",
        );
        assert!(
            actions.iter().any(|action| matches!(
                action,
                Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
            )),
            "and the vote fence is told, as a broadcast would have told it: {actions:?}",
        );

        // The fold is first-write-wins, as the chain's answer is: a
        // second claim restates a decision already committed.
        let again = state.fold_verdict(&VerdictClaim {
            anchor_ts: WeightedTimestamp::from_millis(8_000),
            ..verdict
        });
        assert!(again.is_empty(), "{again:?}");
        assert_eq!(
            state
                .evidence
                .refusal(tx_hash, PEER)
                .map(|held| held.refused_wt),
            Some(anchor),
        );

        // An acceptance settles the transaction and licenses no record,
        // so it never reaches the mirror at all.
        let mut fresh = make_test_state_for_shard(ValidatorId::new(0), HOME);
        fresh.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        assert!(
            fresh
                .fold_verdict(&VerdictClaim {
                    decision: TransactionDecision::Accept,
                    ..verdict
                })
                .is_empty(),
        );
        assert!(fresh.evidence.refusals().is_empty());
    }

    /// A core's refusal of a transaction a leg here issued for is
    /// mirrored off its certificate and handed to the vote fence, and a
    /// `Refused` record is offered from it under the certificate's own
    /// anchor, and the mempool hears the verdict. A second copy adds
    /// nothing.
    #[test]
    fn a_cores_refusal_of_a_leg_is_mirrored_and_offered() {
        let schedule = two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        state.unresolved.mark_leg(
            tx_hash,
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            leg_classified(),
            Vec::new(),
            Vec::new(),
        );
        state.unresolved.certify(tx_hash);

        let certificate = |outcome: ExecutionOutcome| {
            Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
                TickId::new(PEER, BlockHeight::new(3)),
                WeightedTimestamp::from_millis(7_000),
                GlobalReceiptRoot::ZERO,
                vec![TxOutcome::new(tx_hash, outcome)],
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            )))
        };
        let actions = state.handle_attestation(&schedule, &certificate(ExecutionOutcome::Failed));
        let mirrored = state.evidence.refusal(tx_hash, PEER);
        assert_eq!(
            mirrored,
            Some(Refusal {
                refused_wt: WeightedTimestamp::from_millis(7_000),
                deadline: UnsettledTx::for_transaction(&transaction).deadline,
                decision: TransactionDecision::Reject,
                digest: mirrored.expect("mirrored above").digest,
            }),
            "the refusal reaches the mirror the vote fence reads"
        );
        assert!(
            actions.iter().any(|action| matches!(
                action,
                Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
            )),
            "and the fence is told to re-drive the votes that deferred without it"
        );
        assert_eq!(
            state.pending_abandonment_records(),
            vec![AbandonmentRecord::refused(
                PEER,
                WeightedTimestamp::from_millis(7_000),
                [UnsettledTx::for_transaction(&transaction)],
            )],
            "and a record is offered under the certificate's anchor"
        );
        assert_eq!(
            resolved(&actions),
            vec![(
                tx_hash,
                TxResolution::CoreDecided(TransactionDecision::Reject)
            )],
            "and the mempool hears the core's verdict"
        );
        let again = state.handle_attestation(&schedule, &certificate(ExecutionOutcome::Failed));
        assert!(
            !again
                .iter()
                .any(|action| matches!(action, Action::Continuation(_))),
            "a second copy adds nothing"
        );
    }

    /// A core's success is the transaction's verdict only once every
    /// core shard has given one, and it is reported to the mempool once:
    /// a second copy of the certificate adds nothing, and no refusal is
    /// mirrored or offered.
    #[test]
    fn a_cores_success_of_a_leg_is_the_verdict_once_the_whole_core_has_spoken() {
        let schedule = two_shard_topology();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let certificate = |outcome: ExecutionOutcome| {
            Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
                TickId::new(PEER, BlockHeight::new(3)),
                WeightedTimestamp::from_millis(7_000),
                GlobalReceiptRoot::ZERO,
                vec![TxOutcome::new(tx_hash, outcome)],
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            )))
        };
        let mut accepting = make_test_state_for_shard(ValidatorId::new(0), HOME);
        accepting.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        accepting.unresolved.mark_leg(
            tx_hash,
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            leg_classified(),
            Vec::new(),
            Vec::new(),
        );
        let actions = accepting.handle_attestation(
            &schedule,
            &certificate(ExecutionOutcome::Succeeded {
                receipt_hash: GlobalReceiptHash::ZERO,
            }),
        );
        assert!(
            !actions.iter().any(|action| matches!(
                action,
                Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
            )),
            "a success is not a refusal"
        );
        assert!(accepting.pending_abandonment_records().is_empty());
        assert_eq!(
            resolved(&actions),
            vec![(
                tx_hash,
                TxResolution::CoreDecided(TransactionDecision::Accept)
            )],
            "the whole core accepted, which is the transaction's verdict"
        );
        let again = accepting.handle_attestation(
            &schedule,
            &certificate(ExecutionOutcome::Succeeded {
                receipt_hash: GlobalReceiptHash::ZERO,
            }),
        );
        assert!(
            resolved(&again).is_empty(),
            "the core's verdict is reported once"
        );
    }

    /// The resolutions an attestation handed to the mempool.
    fn resolved(actions: &[Action]) -> Vec<(TxHash, TxResolution)> {
        actions
            .iter()
            .flat_map(|action| match action {
                Action::Continuation(ProtocolEvent::TransactionsResolved { resolutions }) => {
                    resolutions.clone()
                }
                _ => Vec::new(),
            })
            .collect()
    }

    /// A state on [`HOME`] holding `transaction` as a leg whose core is
    /// [`PEER`], certified and never resolved.
    fn leg_state(transaction: &Arc<Verifiable<Transaction>>) -> ExecutionCoordinator {
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(transaction),
        );
        state.unresolved.mark_leg(
            transaction.hash(),
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            leg_classified(),
            Vec::new(),
            Vec::new(),
        );
        state.unresolved.certify(transaction.hash());
        state
    }

    /// The state-proof fetches among `actions`, by anchor.
    fn state_proof_fetches(actions: &[Action]) -> Vec<(StateAnchor, Vec<SubstateKey>)> {
        actions
            .iter()
            .filter_map(|action| match action {
                Action::Fetch(FetchRequest::StateProof { anchor, keys, .. }) => {
                    Some((*anchor, keys.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// A proof over `asked` against a tree holding `present`, and the
    /// commit-proven header at `shard`'s `height` stamped `ts` naming
    /// its root: what a counterpart's chain answers, in the shape a
    /// block carries it.
    fn proven_at(
        state: &mut ExecutionCoordinator,
        schedule: &TopologySchedule,
        shard: ShardId,
        height: u64,
        ts: WeightedTimestamp,
        present: &[SubstateKey],
        asked: &[SubstateKey],
    ) -> (StateProofBundle, Vec<Action>) {
        let (state_root, proof) = state_and_proof(shard, present, asked);
        let height = BlockHeight::new(height);
        state.proven_anchors().record(shard, height, state_root, ts);
        let opened = state.on_committed_remote_header(schedule, shard);
        let anchor = StateAnchor {
            shard,
            height,
            state_root,
        };
        (
            StateProofBundle::new(anchor, ts, asked.iter().copied(), proof),
            opened,
        )
    }

    /// Commit a block on [`HOME`] carrying `bundles` — the seam every
    /// replica folds a proof at — and return what the fold emitted.
    fn commit_carrying(
        state: &mut ExecutionCoordinator,
        schedule: &TopologySchedule,
        height: u64,
        ts_ms: u64,
        bundles: Vec<StateProofBundle>,
    ) -> Vec<Action> {
        let Block::Live {
            header,
            transactions,
            certificates,
            provisions,
            abandonment_records,
            witness_sources,
            ..
        } = make_live_block_on_shard(
            HOME,
            BlockHeight::new(height),
            ts_ms,
            ValidatorId::new(0),
            vec![],
        )
        else {
            unreachable!("a live block")
        };
        let block = Block::Live {
            header,
            transactions,
            certificates,
            provisions,
            abandonment_records,
            state_proofs: Arc::new(bundles.into_iter().map(CounterpartClaim::Cells).collect()),
            witness_sources,
        };
        state.on_block_committed(schedule, &test_certify(block, ts_ms))
    }

    /// The absences of `tx_hash` at [`PEER`] handed to the fence among
    /// mirror.
    fn absences_observed(state: &ExecutionCoordinator, tx_hash: TxHash) -> Vec<Absence> {
        absences_observed_at(state, PEER, tx_hash)
    }

    /// The absences of `tx_hash` at `at` the mirror holds, whichever
    /// question proved them.
    fn absences_observed_at(
        state: &ExecutionCoordinator,
        at: ShardId,
        tx_hash: TxHash,
    ) -> Vec<Absence> {
        [Probed::Core, Probed::Delivery, Probed::Claim]
            .into_iter()
            .filter_map(|probed| state.evidence.absence(tx_hash, at, probed))
            .collect()
    }

    /// A delivery that never claimed is probed at its lapse, the
    /// deadline plus a validity range, and never at a header short of
    /// it — the deadline itself included, where a core would already be
    /// asked. The proof the chain carries reaches the vote fence with
    /// the lapse as its floor and is offered as a lapse record.
    #[test]
    fn a_silent_delivery_is_probed_past_the_lapse_and_its_lapse_offered() {
        let schedule = two_shard_topology();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let figures = UnsettledTx::for_transaction(&transaction);
        let deadline = figures.deadline;
        let lapse = deadline.plus(MAX_VALIDITY_RANGE);
        let claim = SubstateKey {
            owner: test_prefix(0x81),
            local: LocalKey([0xC1; 16]),
        };
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        state.unresolved.mark_leg(
            tx_hash,
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            delivery_classified(),
            vec![(PEER, claim)],
            Vec::new(),
        );
        state.unresolved.certify(tx_hash);

        let held: [(u64, WeightedTimestamp, &[u8]); 2] =
            [(3, deadline, b"deadline"), (4, lapse, b"at")];
        for (height, ts, tag) in held {
            state.proven_anchors().record(
                PEER,
                BlockHeight::new(height),
                StateRoot::from_raw(Hash::from_bytes(tag)),
                ts,
            );
            state.on_committed_remote_header(&schedule, PEER);
        }
        let later = lapse.plus(Duration::from_secs(1));
        let (bundle, _) = proven_at(&mut state, &schedule, PEER, 5, later, &[], &[claim]);
        state.committed_ts = deadline;
        assert_eq!(
            state_proof_fetches(&state.probe_silent_counterparts(&schedule)),
            vec![(bundle.anchor, vec![claim])],
            "the newest header inside the lapse window is the anchor, and the claim cell the key"
        );

        let _ = commit_carrying(&mut state, &schedule, 1, deadline.as_millis(), vec![bundle]);
        assert_eq!(
            absences_observed(&state, tx_hash),
            vec![Absence {
                probed_wt: later,
                floor: lapse
            }],
        );
        assert_eq!(
            state.pending_abandonment_records(),
            vec![AbandonmentRecord::lapsed(PEER, later, [figures])],
            "offered as a lapse, under the anchor it was proved at"
        );
    }

    /// A proof this validator's fetch answered is committed content
    /// waiting for a block: it is offered, dated to the clock the probe
    /// read off the header, until a block carries it, and not after.
    #[test]
    fn a_fetched_proof_is_offered_until_a_block_carries_it() {
        let schedule = two_shard_topology();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let deadline = UnsettledTx::for_transaction(&transaction).deadline;
        let later = deadline
            .plus(MAX_VALIDITY_RANGE)
            .plus(Duration::from_secs(1));
        let claim = SubstateKey {
            owner: test_prefix(0x81),
            local: LocalKey([0xC1; 16]),
        };
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        state.unresolved.mark_leg(
            tx_hash,
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            delivery_classified(),
            vec![(PEER, claim)],
            Vec::new(),
        );
        state.unresolved.certify(tx_hash);
        let (bundle, _) = proven_at(&mut state, &schedule, PEER, 5, later, &[], &[claim]);
        state.committed_ts = deadline;
        assert_eq!(
            state_proof_fetches(&state.probe_silent_counterparts(&schedule)),
            vec![(bundle.anchor, vec![claim])],
        );
        state.on_state_proof_verified(bundle.anchor, bundle.keys.clone(), bundle.proof.clone());
        assert_eq!(
            state.pending_state_proofs(),
            vec![CounterpartClaim::Cells(bundle.clone())],
            "dated to the clock the probe read off the header"
        );

        let deadline_ms = deadline.as_millis();
        commit_carrying(&mut state, &schedule, 1, deadline_ms, Vec::new());
        assert_eq!(
            state.pending_state_proofs(),
            vec![CounterpartClaim::Cells(bundle.clone())],
            "a block carrying no proofs leaves the offer standing"
        );
        commit_carrying(&mut state, &schedule, 2, deadline_ms, vec![bundle]);
        assert!(
            state.pending_state_proofs().is_empty(),
            "a proof the chain carries is everybody's"
        );
    }

    /// A delivering shard that departed at a reshape leaves no header
    /// past the lapse, so its claim cell is asked about on the successor
    /// the trie names for the cell's owner — the child holding the
    /// departed chain's cells — and the absence proved there is offered
    /// as a lapse under the successor's name. A header of the departed
    /// shard past the lapse, should one exist, is asked as well, so
    /// every validator proves whichever shard a record names.
    #[test]
    fn a_delivery_whose_deliverer_departed_is_probed_on_its_successor() {
        let schedule = peer_terminating_schedule(60_000);
        let (successor, _) = PEER.children();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let figures = UnsettledTx::for_transaction(&transaction);
        let deadline = figures.deadline;
        let lapse = deadline.plus(MAX_VALIDITY_RANGE);
        // An owner under the peer's left child, as the trie cuts it.
        let claim = SubstateKey {
            owner: test_prefix(0x81),
            local: LocalKey([0xC1; 16]),
        };
        assert_eq!(
            schedule.head().shard_trie().shard_for_prefix(claim.owner),
            successor,
            "the fixture's claim sits under the departed peer's left child"
        );
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        state.unresolved.mark_leg(
            tx_hash,
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            delivery_classified(),
            vec![(PEER, claim)],
            Vec::new(),
        );
        state.unresolved.certify(tx_hash);
        // The local chain has crossed the peer's cut: its committee is
        // anchored in a window whose trie names the children.
        state.committed_committee_anchor_wt = lapse;

        let held: [(u64, WeightedTimestamp, &[u8]); 2] =
            [(3, deadline, b"short"), (4, lapse, b"at")];
        for (height, ts, tag) in held {
            state.proven_anchors().record(
                successor,
                BlockHeight::new(height),
                StateRoot::from_raw(Hash::from_bytes(tag)),
                ts,
            );
            state.on_committed_remote_header(&schedule, successor);
        }
        let later = lapse.plus(Duration::from_secs(1));
        let (bundle, _) = proven_at(&mut state, &schedule, successor, 5, later, &[], &[claim]);
        state.committed_ts = deadline;
        assert_eq!(
            state_proof_fetches(&state.probe_silent_counterparts(&schedule)),
            vec![(bundle.anchor, vec![claim])],
            "the successor's newest header inside the lapse window is the anchor, and the \
             claim cell the key; the departed peer, with no header, is not asked"
        );

        // A header of the departed peer past the lapse is asked as well.
        let (peer_bundle, opened) = proven_at(&mut state, &schedule, PEER, 6, lapse, &[], &[claim]);
        assert_eq!(
            state_proof_fetches(&opened),
            vec![(peer_bundle.anchor, vec![claim])],
            "the shard that was to deliver is asked wherever it has a header past the lapse"
        );

        let _ = commit_carrying(&mut state, &schedule, 1, deadline.as_millis(), vec![bundle]);
        assert_eq!(
            absences_observed_at(&state, successor, tx_hash),
            vec![Absence {
                probed_wt: later,
                floor: lapse
            }],
        );
        assert_eq!(
            state.pending_abandonment_records(),
            vec![AbandonmentRecord::lapsed(successor, later, [figures])],
            "offered as a lapse under the successor's name"
        );
    }

    /// A leg whose core has fallen silent: the core's three headers held,
    /// none asked about while the committed clock was short of the
    /// deadline, and the clock now at it.
    struct SilentCore {
        schedule: TopologySchedule,
        state: ExecutionCoordinator,
        tx_hash: TxHash,
        key: SubstateKey,
        figures: UnsettledTx,
        deadline: WeightedTimestamp,
    }

    fn silent_core() -> SilentCore {
        let schedule = two_shard_topology();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let figures = UnsettledTx::for_transaction(&transaction);
        let deadline = figures.deadline;
        let key = committed_tx_cell_key(
            PEER,
            tx_hash,
            transaction.validity_range().end_timestamp_exclusive,
        );
        let root = |tag: &[u8]| StateRoot::from_raw(Hash::from_bytes(tag));
        let mut state = leg_state(&transaction);
        let held: [(u64, WeightedTimestamp, &[u8]); 3] = [
            (3, deadline.minus(Duration::from_millis(1)), b"short"),
            (5, deadline.plus(Duration::from_secs(1)), b"later"),
            (4, deadline, b"at"),
        ];
        for (height, ts, tag) in held {
            state
                .proven_anchors()
                .record(PEER, BlockHeight::new(height), root(tag), ts);
            let actions = state.on_committed_remote_header(&schedule, PEER);
            assert!(
                state_proof_fetches(&actions).is_empty(),
                "before the deadline nothing is asked"
            );
        }
        state.committed_ts = deadline;
        SilentCore {
            schedule,
            state,
            tx_hash,
            key,
            figures,
            deadline,
        }
    }

    /// The core's committed cell proved absent past the deadline
    /// reaches the vote fence and is offered as an `Unclaimed` record.
    /// The window is what licenses the answer, not the anchor the
    /// proposer happened to probe at: a proof taken short of the
    /// deadline says nothing, and a second copy adds nothing.
    #[test]
    fn a_silent_core_is_probed_past_the_deadline_and_its_absence_offered() {
        let SilentCore {
            schedule,
            mut state,
            tx_hash,
            key,
            figures,
            deadline,
        } = silent_core();
        let later = deadline.plus(Duration::from_secs(1));
        let (bundle, opened) = proven_at(&mut state, &schedule, PEER, 5, later, &[], &[key]);
        assert_eq!(
            state_proof_fetches(&opened),
            vec![(bundle.anchor, vec![key])],
            "the newest header inside the window is the anchor"
        );
        assert!(
            state_proof_fetches(&state.probe_silent_counterparts(&schedule)).is_empty(),
            "a probe in flight is not re-issued while nothing newer is held"
        );

        let (early, _) = proven_at(
            &mut state,
            &schedule,
            PEER,
            2,
            deadline.minus(Duration::from_millis(1)),
            &[],
            &[key],
        );
        let _ = commit_carrying(&mut state, &schedule, 1, deadline.as_millis(), vec![early]);
        assert!(
            absences_observed(&state, tx_hash).is_empty(),
            "a proof taken before the deadline says nothing: the core may still commit"
        );

        let folded = commit_carrying(
            &mut state,
            &schedule,
            2,
            deadline.as_millis(),
            vec![bundle.clone()],
        );
        assert!(
            folded.iter().any(|action| matches!(
                action,
                Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
            )),
            "the fence is told the absence landed"
        );
        assert_eq!(
            absences_observed(&state, tx_hash),
            vec![Absence {
                probed_wt: later,
                floor: deadline
            }],
            "the absence reaches the mirror the vote fence reads"
        );
        assert_eq!(
            state.pending_abandonment_records(),
            vec![AbandonmentRecord::unclaimed(PEER, later, [figures])],
            "and a record is offered under the anchor it was proved at"
        );

        let again = commit_carrying(&mut state, &schedule, 3, deadline.as_millis(), vec![bundle]);
        assert!(
            !again.iter().any(|action| matches!(
                action,
                Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
            )),
            "a second copy adds nothing"
        );
    }

    /// A core that turns out to have committed the transaction is not
    /// absent: the presence answers the question, offers nothing, and
    /// the core is not asked again — its own certificate speaks next.
    #[test]
    fn a_core_that_committed_the_transaction_is_not_probed_again() {
        let schedule = two_shard_topology();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let deadline = UnsettledTx::for_transaction(&transaction).deadline;
        let key = committed_tx_cell_key(
            PEER,
            tx_hash,
            transaction.validity_range().end_timestamp_exclusive,
        );
        let mut state = leg_state(&transaction);
        state.committed_ts = deadline;
        let (bundle, opened) = proven_at(&mut state, &schedule, PEER, 4, deadline, &[key], &[key]);
        assert_eq!(
            state_proof_fetches(&opened),
            vec![(bundle.anchor, vec![key])]
        );

        let folded = commit_carrying(&mut state, &schedule, 1, deadline.as_millis(), vec![bundle]);
        assert!(
            absences_observed(&state, tx_hash).is_empty(),
            "a core that committed it is not absent"
        );
        assert!(
            folded.iter().any(|action| matches!(
                action,
                Action::Fetch(FetchRequest::ExecutionCerts { source_shard, tx_hash: fetched, .. })
                    if *source_shard == PEER && *fetched == tx_hash
            )),
            "and its certificate is fetched, since a refusal there licenses the reclaim"
        );
        assert!(state.pending_abandonment_records().is_empty());
        assert!(
            state_proof_fetches(&state.probe_silent_counterparts(&schedule)).is_empty(),
            "and is not asked again"
        );
    }

    /// A leg entry on `HOME` whose core consumer's claim sits at `claim`
    /// on `PEER`, with the committed clock at the deadline: what a claim
    /// probe is issued for.
    fn claimed_leg_state(
        transaction: &Arc<Verifiable<Transaction>>,
        claim: SubstateKey,
    ) -> ExecutionCoordinator {
        claimed_leg_state_under(transaction, claim, leg_classified())
    }

    /// [`claimed_leg_state`] with the shape frozen as `classified` says:
    /// what fixes how many shards the core spans.
    fn claimed_leg_state_under(
        transaction: &Arc<Verifiable<Transaction>>,
        claim: SubstateKey,
        classified: Classified,
    ) -> ExecutionCoordinator {
        let mut state = make_test_state_for_shard(ValidatorId::new(0), HOME);
        state.unresolved.register_committed(
            HOME,
            WeightedTimestamp::ZERO,
            std::iter::once(transaction),
        );
        state.unresolved.mark_leg(
            transaction.hash(),
            Arc::new(Verified::new_unchecked_for_test(straddling_transaction(1))),
            classified,
            Vec::new(),
            vec![(PEER, claim)],
        );
        state.unresolved.certify(transaction.hash());
        state.committed_ts = UnsettledTx::for_transaction(transaction).deadline;
        state
    }

    /// A core consumer's claim is asked about beside the core's
    /// committed cell. On a core of one shard a claim proved absent
    /// past the deadline is the core never taking the crossing: it
    /// reaches the fence, is offered as an `Untaken` record, and neither
    /// question is asked again.
    #[test]
    fn a_single_shard_cores_claim_proved_absent_is_its_answer() {
        let schedule = two_shard_topology();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let figures = UnsettledTx::for_transaction(&transaction);
        let deadline = figures.deadline;
        let core_key = committed_tx_cell_key(
            PEER,
            tx_hash,
            transaction.validity_range().end_timestamp_exclusive,
        );
        let claim = SubstateKey {
            owner: core_key.owner,
            local: LocalKey([0x7C; 16]),
        };
        let mut state = claimed_leg_state(&transaction, claim);
        let (bundle, opened) = proven_at(
            &mut state,
            &schedule,
            PEER,
            4,
            deadline,
            &[core_key],
            &[core_key, claim],
        );
        assert_eq!(
            state_proof_fetches(&opened),
            vec![(bundle.anchor, vec![core_key, claim])],
            "the core's committed cell and the consumer's claim are asked about together"
        );

        state.on_state_proof_verified(bundle.anchor, bundle.keys.clone(), bundle.proof.clone());
        let folded = commit_carrying(&mut state, &schedule, 1, deadline.as_millis(), vec![bundle]);
        assert!(
            folded.iter().any(|action| matches!(
                action,
                Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
            )),
            "an absent claim on a core of one shard is evidence"
        );
        assert_eq!(
            state.pending_abandonment_records(),
            vec![AbandonmentRecord::untaken(PEER, deadline, [figures])],
            "offered as untaken, under the anchor it was proved at"
        );

        let (_, opened) = proven_at(
            &mut state,
            &schedule,
            PEER,
            5,
            deadline.plus(Duration::from_secs(2)),
            &[core_key],
            &[claim],
        );
        assert!(
            state_proof_fetches(&opened).is_empty(),
            "and neither question is asked again: both answered"
        );
    }

    /// On a core of more than one shard the same absence says only that
    /// a sibling is pending: the core settles on its siblings' clock, so
    /// nothing reaches the fence, nothing is offered, and the claim is
    /// asked again at the next header — alone, since the committed cell
    /// answered. That cell is what answers for such a core.
    #[test]
    fn a_multi_shard_cores_claim_proved_absent_is_asked_again() {
        let schedule = two_shard_topology();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let deadline = UnsettledTx::for_transaction(&transaction).deadline;
        let core_key = committed_tx_cell_key(
            PEER,
            tx_hash,
            transaction.validity_range().end_timestamp_exclusive,
        );
        let claim = SubstateKey {
            owner: core_key.owner,
            local: LocalKey([0x7C; 16]),
        };
        let mut state = claimed_leg_state_under(&transaction, claim, two_shard_core_classified());
        let (bundle, _) = proven_at(
            &mut state,
            &schedule,
            PEER,
            4,
            deadline,
            &[core_key],
            &[core_key, claim],
        );

        state.on_state_proof_verified(bundle.anchor, bundle.keys.clone(), bundle.proof.clone());
        let folded = commit_carrying(&mut state, &schedule, 1, deadline.as_millis(), vec![bundle]);
        assert!(
            !folded.iter().any(|action| matches!(
                action,
                Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
            )),
            "an absent claim on a core of two shards proves nothing"
        );
        assert!(state.pending_abandonment_records().is_empty());

        let (later, opened) = proven_at(
            &mut state,
            &schedule,
            PEER,
            5,
            deadline.plus(Duration::from_secs(2)),
            &[core_key],
            &[claim],
        );
        assert_eq!(
            state_proof_fetches(&opened),
            vec![(later.anchor, vec![claim])],
            "only the claim is asked again: the committed cell answered"
        );
    }

    /// A core consumer's claim proved present is the consumer's
    /// settlement: it reaches the vote fence, and once a `Claimed`
    /// record naming it commits the next commit composes the retirement
    /// into its tick — a dispatched member running no node, awaiting
    /// nobody, charged nothing.
    #[test]
    fn a_claim_proved_present_composes_the_retirement() {
        let schedule = two_shard_topology();
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(1)),
        ));
        let tx_hash = transaction.hash();
        let figures = UnsettledTx::for_transaction(&transaction);
        let claim = SubstateKey {
            owner: committed_tx_cell_key(
                PEER,
                tx_hash,
                transaction.validity_range().end_timestamp_exclusive,
            )
            .owner,
            local: LocalKey([0x7C; 16]),
        };
        let mut state = claimed_leg_state(&transaction, claim);
        let probed_wt = figures.deadline.plus(Duration::from_secs(2));
        let (bundle, opened) = proven_at(
            &mut state,
            &schedule,
            PEER,
            5,
            probed_wt,
            &[claim],
            &[claim],
        );
        assert!(
            state_proof_fetches(&opened)
                .iter()
                .any(|(at, keys)| *at == bundle.anchor && keys.contains(&claim)),
            "the claim is asked about"
        );

        let folded = commit_carrying(
            &mut state,
            &schedule,
            1,
            probed_wt.as_millis(),
            vec![bundle],
        );
        assert!(
            folded.iter().any(|action| matches!(
                action,
                Action::Continuation(ProtocolEvent::CounterpartEvidenceObserved)
            )) && state
                .evidence
                .presence(tx_hash, PEER)
                .is_some_and(|presence| presence.probed_wt == probed_wt),
            "a present claim reaches the mirror the vote fence reads"
        );
        assert!(
            state.pending_abandonment_records().is_empty(),
            "the offer is gated; what the fold and the retirement do with a record is not"
        );

        state
            .unresolved
            .record_abandonment_records(&[AbandonmentRecord::claimed(PEER, probed_wt, [figures])]);
        let actions = commit_carrying(&mut state, &schedule, 2, probed_wt.as_millis(), Vec::new());
        let request = actions
            .iter()
            .find_map(|action| match action {
                Action::ExecuteTransactions { requests, .. } => {
                    requests.iter().find(|request| request.tx_hash == tx_hash)
                }
                _ => None,
            })
            .expect("the retirement is dispatched to the engine");
        assert!(matches!(request.runs, Runs::Retire { .. }));
        assert!(!request.abortable, "nothing retracts a retirement");
        assert!(
            state.unresolved.retirable().is_empty(),
            "and the ledger has handed it to the tick"
        );
    }

    /// A delivery is abandoned at its window's close out of any tick
    /// still holding it: past the close its issuer may prove the claim
    /// absent and take the crossing back, so the tick that would write
    /// the claim is discarded, its finalization with it.
    #[test]
    fn a_delivery_held_by_a_tick_is_abandoned_at_the_close() {
        let schedule = make_test_topology();
        let mut state = make_test_state();
        let tx = test_transaction(1);
        let tx_hash = tx.hash();
        let validity_end = tx.validity_range().end_timestamp_exclusive;
        let close_ms = delivery_window_close(validity_end).as_millis();

        state.on_block_committed(
            &schedule,
            &test_certify(
                make_live_block(
                    BlockHeight::new(1),
                    1_000,
                    ValidatorId::new(0),
                    vec![Arc::new(tx)],
                ),
                1_000,
            ),
        );
        state.unresolved.mark_delivery(tx_hash, validity_end);
        state.unresolved.certify(tx_hash);
        let held_by = TickId::new(ShardId::ROOT, BlockHeight::new(1));
        assert_eq!(state.ticks.tick_assignment(tx_hash), Some(held_by));

        let outcomes = abandonment_vote(&mut state, &schedule, 2, close_ms - 1);
        assert!(
            outcomes.is_empty(),
            "inside the window the tick is left to it"
        );
        assert!(state.ticks.contains_tick(&held_by));

        let outcomes = abandonment_vote(&mut state, &schedule, 3, close_ms);
        assert!(
            outcomes
                .iter()
                .any(|outcome| outcome.tx_hash() == tx_hash && outcome.decides()),
            "at the close the delivery is abandoned: {outcomes:?}"
        );
        assert!(
            !state.ticks.contains_tick(&held_by),
            "and the tick that held it is discarded"
        );
    }

    /// Before its deadline a transaction is merely slow, and nothing
    /// abandons it — that is what stops a proposer discarding work.
    #[test]
    fn a_transaction_before_its_deadline_is_not_abandoned() {
        let schedule = make_test_topology();
        let mut state = make_test_state();
        let tx = test_transaction(1);
        let deadline_ms = 60_000 + u64::try_from(MAX_FINALIZATION_DELAY.as_millis()).unwrap();

        state.on_block_committed(
            &schedule,
            &test_certify(
                make_live_block(
                    BlockHeight::new(1),
                    1_000,
                    ValidatorId::new(0),
                    vec![Arc::new(tx)],
                ),
                1_000,
            ),
        );
        state
            .ticks
            .remove_tick(&TickId::new(ShardId::ROOT, BlockHeight::new(1)));

        let outcomes = abandonment_vote(&mut state, &schedule, 2, deadline_ms - 1);
        assert!(
            outcomes.is_empty(),
            "a transaction short of its deadline is not abandoned"
        );
        assert_eq!(state.unresolved.len(), 1, "and stays owed");
    }

    /// A tick that has not yet spoken withholds the abort — it is about
    /// to attest the transaction itself, and that verdict can carry a
    /// charge an abandonment cannot.
    ///
    /// The window is ordinary rather than pathological: a payer's leg
    /// joins a tick at its engagement deadline, which *is* its
    /// abandonment deadline, and the commits between that tick's
    /// composition and its certificate would otherwise abandon the member
    /// it is about to speak for — discarding the tick that carries the
    /// charge. `abort_charges_the_price_on_deadline` is the scenario.
    #[test]
    fn a_tick_that_has_not_attested_withholds_the_abort() {
        let schedule = make_test_topology();
        let mut state = make_test_state();
        let tx = test_transaction(1);
        let tx_hash = tx.hash();
        let deadline_ms = 60_000 + u64::try_from(MAX_FINALIZATION_DELAY.as_millis()).unwrap();

        state.on_block_committed(
            &schedule,
            &test_certify(
                make_live_block(
                    BlockHeight::new(1),
                    1_000,
                    ValidatorId::new(0),
                    vec![Arc::new(tx)],
                ),
                1_000,
            ),
        );
        state.committed_ts = WeightedTimestamp::from_millis(deadline_ms);

        assert!(
            state
                .abandonable(TickId::new(ShardId::ROOT, BlockHeight::new(1)))
                .is_empty(),
            "a tick still to vote is an outcome on its way",
        );
        let _ = tx_hash;
    }

    /// A snapshot over an explicit leaf set, with `cut` naming the shards
    /// scheduled to terminate at the epochs given.
    fn leaves_snap(leaves: &[ShardId], cut: &[(ShardId, u64)]) -> Arc<TopologySnapshot> {
        leaves_snap_departed(leaves, cut, &[])
    }

    /// [`leaves_snap`] carrying a terminal boundary record per departed
    /// shard, as every projected snapshot does while the beacon retains
    /// the record — which is exactly as long as the fence may still read
    /// the evidence. `handoff_complete: None` is an open window.
    fn leaves_snap_departed(
        leaves: &[ShardId],
        cut: &[(ShardId, u64)],
        departed: &[(ShardId, Option<Epoch>)],
    ) -> Arc<TopologySnapshot> {
        let boundaries: HashMap<ShardId, ShardAnchor> = departed
            .iter()
            .map(|(shard, handoff_complete)| {
                (
                    *shard,
                    ShardAnchor {
                        state_root: StateRoot::ZERO,
                        block_hash: BlockHash::from_raw(Hash::from_bytes(b"terminal")),
                        height: BlockHeight::new(9),
                        weighted_timestamp: WeightedTimestamp::from_millis(1_000),
                        witness_base: BeaconWitnessLeafCount::ZERO,
                        terminal_roots: None,
                        handoff_complete: *handoff_complete,
                    },
                )
            })
            .collect();
        Arc::new(
            TopologySnapshot::from_explicit_committees(
                NetworkDefinition::simulator(),
                &ValidatorSet::new(Vec::new()),
                leaves.iter().map(|s| (*s, Vec::new())).collect(),
                HashMap::new(),
                boundaries,
                HashMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeSet::new(),
            )
            .with_scheduled_terminals(cut.iter().map(|(s, e)| (*s, Epoch::new(*e))).collect()),
        )
    }

    /// This shard, and the peer holding the other half of a straddler.
    const HOME: ShardId = ShardId::leaf(1, 0);
    const PEER: ShardId = ShardId::leaf(1, 1);

    /// A shape frozen divided with an inbound leg on `HOME` feeding a
    /// core on `PEER`.
    fn leg_classified() -> Classified {
        use hyperscale_vm_types::LegRole;

        use crate::fixtures::leg;
        let legs = [
            leg(0, LegRole::Inbound, &[]),
            leg(2, LegRole::Core, &[(0, 0)]),
        ];
        let classified = Classified::freeze(&legs, &[], &ShardTrie::uniform(1));
        assert_eq!(classified.core(), &BTreeSet::from([PEER]));
        classified
    }

    /// A shape frozen divided with a core spanning two leaves, so a claim
    /// absent on either says only that the other is pending.
    fn two_shard_core_classified() -> Classified {
        use hyperscale_vm_types::LegRole;

        use crate::fixtures::leg;
        let legs = [
            leg(0, LegRole::Inbound, &[]),
            leg(2, LegRole::Core, &[(0, 0)]),
            leg(3, LegRole::Core, &[(1, 0)]),
        ];
        let classified = Classified::freeze(&legs, &[], &ShardTrie::uniform(2));
        assert_eq!(
            classified.core(),
            &BTreeSet::from([ShardId::leaf(2, 2), ShardId::leaf(2, 3)])
        );
        assert!(classified.decomposed().holds());
        classified
    }

    /// A shape frozen divided with its core on a leaf no held header
    /// names, so only the leg's deliveries are ever probed.
    fn delivery_classified() -> Classified {
        use hyperscale_vm_types::LegRole;

        use crate::fixtures::leg;
        let legs = [
            leg(0, LegRole::Inbound, &[]),
            leg(3, LegRole::Core, &[(0, 0)]),
        ];
        let classified = Classified::freeze(&legs, &[], &ShardTrie::uniform(2));
        assert_eq!(classified.core(), &BTreeSet::from([ShardId::leaf(2, 3)]));
        classified
    }

    /// [`HOME`] and [`PEER`] both live, both crewed — the topology a
    /// straddler between them composes and votes under.
    fn two_shard_topology() -> TopologySchedule {
        let keys: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let validators: Vec<ValidatorInfo> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| ValidatorInfo {
                validator_id: ValidatorId::new(i as u64),
                public_key: k.public_key(),
            })
            .collect();
        TopologySchedule::single(Arc::new(TopologySnapshot::new(
            NetworkDefinition::simulator(),
            2,
            ValidatorSet::new(validators),
        )))
    }

    /// A schedule in which [`PEER`] splits at the end of epoch 0 while
    /// [`HOME`] runs on, so the peer is past-terminal anywhere in epoch 1
    /// and its keyspace passes to its two children.
    ///
    /// A counterpart is a peer rather than an ancestor: a shard's own
    /// predecessor never held a transaction the shard committed, so a
    /// fixture that used one would be asking about a settlement that
    /// could not have happened.
    fn peer_terminating_schedule(epoch_duration_ms: u64) -> TopologySchedule {
        peer_terminating_schedule_stamped(epoch_duration_ms, None)
    }

    /// [`peer_terminating_schedule`] with the peer's handoff-complete
    /// stamp under the caller's control: `None` holds the evidence window
    /// open, `Some(epoch)` closes it `TERMINAL_EVIDENCE_EPOCHS` past that
    /// epoch's window.
    fn peer_terminating_schedule_stamped(
        epoch_duration_ms: u64,
        handoff_complete: Option<Epoch>,
    ) -> TopologySchedule {
        let (left, right) = PEER.children();
        let mut sched = TopologySchedule::new(
            epoch_duration_ms,
            Epoch::new(0),
            leaves_snap(&[HOME, PEER], &[(PEER, 0)]),
        );
        let post = leaves_snap_departed(&[HOME, left, right], &[], &[(PEER, handoff_complete)]);
        for epoch in 1..=12u64 {
            sched.insert(Epoch::new(epoch), Arc::clone(&post));
        }
        sched.set_head(post);
        sched
    }

    /// The weighted timestamp at which a transaction admitted under
    /// [`state_stranded_on`] can no longer finalize anywhere.
    const STRANDED_DEADLINE_MS: u64 = 60_000 + MAX_FINALIZATION_DELAY.as_secs() * 1000;

    /// A transaction paid for on the `0b0…` side of the keyspace and
    /// writing to the `0b1…` side, so it reaches beyond a shard holding
    /// the first however the trie is cut.
    fn straddling_transaction(seed: u8) -> Transaction {
        test_transaction_with_prefixes(
            &[seed & 0x7F],
            &[test_prefix(seed & 0x7F)],
            &[test_prefix(seed | 0x80)],
        )
    }

    /// A state past `tx`'s deadline whose tick at height 1 holds it with
    /// this shard's own certificate in hand and no counterpart coverage —
    /// the shape a counterpart's silence produces. The transaction
    /// straddles, so `partner` owns the half of it this shard does not.
    /// Commit the record naming `tx_hash` as what the departed peer left
    /// unsettled — the evidence composition requires before it will spend
    /// a tick on an abort.
    fn record_peer_left_unsettled(state: &mut ExecutionCoordinator, tx_hash: TxHash) {
        state
            .unresolved
            .record_abandonment_records(&[AbandonmentRecord::departed(
                PEER,
                WeightedTimestamp::from_millis(60_000),
                vec![UnsettledTx {
                    tx_hash,
                    deadline: WeightedTimestamp::from_millis(30_000),
                    declared_work: 1,
                    charge: AbortCharge {
                        vault: SubstateKey {
                            owner: Address::new([9; 31], AddressClass::Component),
                            local: LocalKey([9; 16]),
                        },
                        amount: 5,
                    },
                }],
            )]);
    }

    fn state_stranded_on(
        topology_schedule: &TopologySchedule,
        seed: u8,
    ) -> (ExecutionCoordinator, TickId, TxHash) {
        let (local, partner) = (HOME, PEER);
        let mut state = make_test_state_for_shard(ValidatorId::new(0), local);
        let tick_id = TickId::new(local, BlockHeight::new(1));
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(seed)),
        ));
        let tx_hash = transaction.hash();
        let mut tick = tick_holding(
            tick_id,
            WeightedTimestamp::from_millis(1_000),
            vec![(
                Arc::new(Verified::new_unchecked_for_test(straddling_transaction(
                    seed,
                ))),
                [local, partner].into_iter().collect(),
            )],
        );
        tick.add_execution_certificate(Arc::new(Verified::new_unchecked_for_test(
            ExecutionCertificate::new(
                tick_id,
                WeightedTimestamp::from_millis(1_000),
                GlobalReceiptRoot::from_raw(Hash::from_bytes(b"root")),
                vec![TxOutcome::new(tx_hash, ExecutionOutcome::Aborted)],
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            ),
        )));
        state.ticks.insert_tick(tick_id, tick);
        state.ticks.assign_tx(tx_hash, tick_id);
        state.unresolved.register_committed(
            local,
            WeightedTimestamp::ZERO,
            std::iter::once(&transaction),
        );
        state.unresolved.certify(tx_hash);
        state.committed_ts = WeightedTimestamp::from_millis(STRANDED_DEADLINE_MS);
        state.stamp_departures(topology_schedule);
        (state, tick_id, tx_hash)
    }

    /// While the counterpart is live the member stays with the tick that
    /// holds it. The counterpart's certificate is still to come, so the
    /// tick can yet speak for the member and a second verdict would
    /// contradict the one it is about to carry.
    #[test]
    fn a_stranded_tick_keeps_its_member_while_its_counterpart_lives() {
        // Windows wide enough that the frontier the fixture sits at is
        // still inside the peer's own epoch.
        let sched = peer_terminating_schedule(600_000);
        let (state, tick_id, _) = state_stranded_on(&sched, 1);

        assert!(
            state
                .abandonable(TickId::new(HOME, BlockHeight::new(9)))
                .is_empty(),
            "a live counterpart can still deliver the verdict this tick waits on",
        );
        assert!(
            state.ticks.contains_tick(&tick_id),
            "so the tick survives its own deadline",
        );
    }

    /// The whole round trip in one place: a counterpart leaves without
    /// settling, its set says so, this shard writes that down, and the
    /// record is what lets the abort be composed afterwards.
    #[test]
    fn a_departed_counterpart_s_silence_is_written_down_and_licenses_the_abort() {
        let sched = peer_terminating_schedule(60_000);
        let (mut state, _, tx_hash) = state_stranded_on(&sched, 1);

        // The peer's settled set arrives, naming nothing: it settled none
        // of what it was party to before it went.
        state.record_settled_txs(
            &sched,
            PEER,
            SettledTxSet {
                txs: BTreeSet::new(),
                terminal_wt: WeightedTimestamp::from_millis(60_000),
            },
        );

        let records = state.pending_abandonment_records();
        assert_eq!(records.len(), 1, "the peer's departure is answerable");
        assert_eq!(records[0].shard(), PEER);
        assert_eq!(
            records[0].tx_hashes().collect::<Vec<_>>(),
            vec![tx_hash],
            "the straddler is what it left unresolved of our business",
        );

        // Committed, the record is what the account reads afterwards.
        state.unresolved.record_abandonment_records(&records);
        assert!(state.unresolved.is_unsettled_by_departed(tx_hash));

        // And what it does not offer twice.
        assert!(
            state.pending_abandonment_records().is_empty(),
            "a departure is answered once",
        );

        assert_eq!(
            state
                .abandonable(TickId::new(HOME, BlockHeight::new(9)))
                .iter()
                .map(|entry| entry.tx_hash)
                .collect::<Vec<_>>(),
            vec![tx_hash],
            "and the record is what makes the abort this shard's to compose",
        );
    }

    /// One transaction can be named by two records, which is why the
    /// budget the composer spends is the block's and not each record's: a
    /// straddler reaching a departed shard reaches its departed successor
    /// too, so the records name more between them than the drain holds,
    /// and a block over that bound is one every voter refuses.
    ///
    /// Also pins the order: ascending by shard is the one form a block may
    /// carry them in, and `settled_sets` is a hash map, so the walk cannot
    /// take its iteration order.
    #[test]
    fn two_departures_over_one_transaction_share_the_block_s_budget() {
        let sched = peer_terminating_schedule(60_000);
        let (mut state, _, tx_hash) = state_stranded_on(&sched, 1);
        let (peer_left, _) = PEER.children();

        // Both cover the straddler's remote prefix — the bit test a shard
        // and its descendant both pass — so both are party to it.
        let set = |cut_ms: u64| SettledTxSet {
            txs: BTreeSet::new(),
            terminal_wt: WeightedTimestamp::from_millis(cut_ms),
        };
        state.record_settled_txs(&sched, peer_left, set(120_000));
        state.record_settled_txs(&sched, PEER, set(60_000));

        let records = state.pending_abandonment_records();
        assert_eq!(
            records
                .iter()
                .map(AbandonmentRecord::shard)
                .collect::<Vec<_>>(),
            vec![PEER, peer_left],
            "ascending by shard, whatever order the sets are held in",
        );
        let named: usize = records.iter().map(|r| r.unsettled().len()).sum();
        assert_eq!(
            named, 2,
            "one outstanding transaction, named twice — the sum is not the drain's count",
        );
        assert!(
            named <= MAX_UNSETTLED_PER_BLOCK,
            "and inside the block's own bound"
        );
        for record in &records {
            assert_eq!(record.tx_hashes().collect::<Vec<_>>(), vec![tx_hash]);
        }
    }

    /// The certificate outlives the tick that produced it, so losing the
    /// tick does not make the member this shard's to abandon: the
    /// counterpart holds a certificate of ours it can still settle
    /// against, and the account is what remembers that.
    #[test]
    fn a_lost_tick_does_not_release_a_member_a_counterpart_can_settle() {
        let sched = peer_terminating_schedule(600_000);
        let (mut state, tick_id, _) = state_stranded_on(&sched, 1);

        state.ticks.remove_tick(&tick_id);
        state.ticks.discard_tick(&tick_id);

        assert!(
            state
                .abandonable(TickId::new(HOME, BlockHeight::new(9)))
                .is_empty(),
            "the certificate is out there whether or not the tick still is",
        );
    }

    /// A counterpart's departure is not by itself what releases the
    /// member. A shard can settle its half and then leave, so its going
    /// says nothing about whether the transaction is still reachable —
    /// and spending the tick on the departure alone would discard the one
    /// settlement that had already closed. Only a committed record
    /// licenses the abort; until one lands the tick keeps speaking.
    #[test]
    fn a_departure_alone_does_not_release_the_member_its_tick_strands() {
        // Windows placed so the frontier the fixture sits at is past the
        // peer's cut.
        let sched = peer_terminating_schedule(60_000);
        let (state, _, _) = state_stranded_on(&sched, 1);

        assert!(
            state
                .abandonable(TickId::new(HOME, BlockHeight::new(9)))
                .is_empty(),
            "the peer may have settled before it went, and nothing committed says otherwise",
        );
    }

    /// The tick composing now keeps the member it just took, whatever the
    /// counterpart's fate. It is about to attest the transaction itself,
    /// and that verdict can carry a charge an abandonment cannot — a
    /// payer's leg joins a tick at its engagement deadline, which *is* its
    /// abandonment deadline. `abort_charges_the_price_on_deadline` is the
    /// scenario.
    #[test]
    fn the_tick_composing_now_keeps_the_member_it_just_took() {
        let sched = peer_terminating_schedule(60_000);
        let (state, tick_id, _) = state_stranded_on(&sched, 1);

        assert!(
            state.abandonable(tick_id).is_empty(),
            "the tick composing now is the one that speaks for its own member",
        );
    }

    /// A tick that stranded a member is discarded when that member is
    /// abandoned, and the transactions it held alongside are released to
    /// their own deadlines rather than waiting on coverage that will never
    /// close.
    #[test]
    fn the_tick_that_stranded_a_member_is_discarded_with_it() {
        let local = HOME;
        let sched = peer_terminating_schedule(60_000);
        let (mut state, tick_id, tx_hash) = state_stranded_on(&sched, 1);
        record_peer_left_unsettled(&mut state, tx_hash);

        // A second member of the same tick, reaching only into keyspace
        // nobody has left — so its own counterpart is still live.
        let sibling: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(test_transaction(2)),
        ));
        let sibling_hash = sibling.hash();
        state.ticks.assign_tx(sibling_hash, tick_id);
        state.unresolved.register_committed(
            local,
            WeightedTimestamp::ZERO,
            std::iter::once(&sibling),
        );
        state.unresolved.certify(sibling_hash);

        // The commit that composes the abandonment, on the shard that
        // stranded the member.
        let block = make_live_block_on_shard(
            local,
            BlockHeight::new(9),
            STRANDED_DEADLINE_MS,
            ValidatorId::new(0),
            vec![],
        );
        state.on_block_committed(&sched, &test_certify(block, STRANDED_DEADLINE_MS));

        let composed = TickId::new(local, BlockHeight::new(9));
        assert_eq!(
            state.ticks.tick_assignment(tx_hash),
            Some(composed),
            "the abandonment is composed into this commit's tick",
        );
        assert!(
            !state.ticks.contains_tick(&tick_id),
            "the tick that can never speak for the abandoned member goes with it",
        );
        assert_eq!(
            state.ticks.tick_assignment(sibling_hash),
            None,
            "and its other members are released to their own deadlines",
        );
    }

    /// A finalization whose only certificate is this shard's, attesting
    /// `tx_hash` aborted after awaiting `partner` — the shape composition
    /// produces past a deadline.
    fn abandonment_of(local: ShardId, partner: ShardId, tx_hash: TxHash) -> Finalization {
        lone_finalization(
            local,
            TxOutcome::new(tx_hash, ExecutionOutcome::Aborted).awaiting([partner]),
        )
    }

    /// A finalization whose only certificate is this shard's, attesting
    /// `tx_hash` refused by a member that awaited nobody — a leg's own
    /// verdict.
    fn lone_verdict_of(local: ShardId, tx_hash: TxHash) -> Finalization {
        lone_finalization(local, TxOutcome::new(tx_hash, ExecutionOutcome::Failed))
    }

    fn lone_finalization(local: ShardId, outcome: TxOutcome) -> Finalization {
        let tick_id = TickId::new(local, BlockHeight::new(1));
        Finalization::new(
            tick_id,
            TickHalf::Determined,
            vec![Arc::new(ExecutionCertificate::new(
                tick_id,
                WeightedTimestamp::from_millis(1),
                GlobalReceiptRoot::ZERO,
                vec![outcome],
                AggregateSignature::ZERO,
                SignerBitfield::new(4),
            ))],
            vec![],
        )
    }

    /// A state whose ledger names `partner` party to `tx_hash`, on the
    /// schedule that has `partner` past-terminal at 1500ms.
    fn state_abandoning(
        topology_schedule: &TopologySchedule,
        local: ShardId,
        transaction: &Arc<Verifiable<Transaction>>,
    ) -> ExecutionCoordinator {
        let mut state = make_test_state_for_shard(ValidatorId::new(0), local);
        state.committed_ts = WeightedTimestamp::from_millis(1500);
        state.unresolved.register_committed(
            local,
            WeightedTimestamp::ZERO,
            std::iter::once(transaction),
        );
        state.stamp_departures(topology_schedule);
        state
    }

    /// The abort a terminating counterpart might have settled is held at
    /// the fence, not at composition.
    ///
    /// An abandonment carries only this shard's certificate, because an
    /// abort needs no counterpart's verdict — so `settled_set_verdict`,
    /// which skips the local shard, would wave it through. The counterparts
    /// its outcome names as awaited are what let the fence see it, and
    /// while the partner's settled set is unknown it holds: the partner
    /// may already have committed a settlement, and aborting under that is
    /// the one-sided settlement the fence exists to prevent.
    #[test]
    fn the_fence_holds_an_abort_a_terminating_partner_might_have_settled() {
        let (local, partner) = (HOME, PEER);
        let sched = peer_terminating_schedule(1_000);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(7)),
        ));
        let tx_hash = transaction.hash();
        let abort = abandonment_of(local, partner, tx_hash);
        assert!(
            abort
                .execution_certificates()
                .iter()
                .all(|ec| ec.shard_id() == local),
            "an abandonment carries no counterpart certificate",
        );

        // The abandonment's own outcome is where its counterparts come
        // from: the terminating partner is named, and the gate holds.
        let mut state = state_abandoning(&sched, local, &transaction);
        assert!(
            state
                .fence_pairs(&abort)
                .iter()
                .any(|(shard, _, claim)| *shard == partner && *claim == TxClaim::Abandoned),
            "an abandonment names the counterparts it awaited",
        );

        let held: Arc<Verifiable<Finalization>> =
            Arc::new(Verified::<Finalization>::seal(abort).into());
        assert!(
            state.emit_or_gate_finalized(&sched, held).is_empty(),
            "held while the partner's settled set is unknown",
        );
        assert_eq!(state.gated_finalized.len(), 1, "held at the gate");
    }

    /// The abort of a transaction the terminated partner never settled is
    /// admitted — the stranded case the deadline path exists for.
    ///
    /// The partner's settled set answers the opposite question from the
    /// one a settlement asks of it: a settlement needs the partner to have
    /// settled its half, an abandonment needs it not to have. Reading the
    /// set the settlement way here would reject the only outcome the
    /// transaction can still reach, and the work it reserved would never
    /// return to the drain.
    #[test]
    fn a_partner_that_never_settled_it_is_what_makes_the_abort_admissible() {
        let (local, partner) = (HOME, PEER);
        let sched = peer_terminating_schedule(1_000);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(7)),
        ));
        let tx_hash = transaction.hash();
        let mut state = state_abandoning(&sched, local, &transaction);

        state.record_settled_txs(
            &sched,
            partner,
            SettledTxSet {
                txs: BTreeSet::new(),
                terminal_wt: WeightedTimestamp::from_millis(1000),
            },
        );

        let abort: Arc<Verifiable<Finalization>> = Arc::new(
            Verified::<Finalization>::seal(abandonment_of(local, partner, tx_hash)).into(),
        );
        assert!(
            !state.emit_or_gate_finalized(&sched, abort).is_empty(),
            "the partner terminated without settling it, so the abort is the outcome",
        );
        assert!(state.gated_finalized.is_empty(), "and nothing is held back");
    }

    /// The abort of a transaction the terminated partner *did* settle is
    /// refused: its half applied, and aborting here would tear the
    /// transaction in two. The settlement path is what resolves it, on the
    /// certificate `record_settled_txs` arms the fetch for.
    #[test]
    fn a_partner_that_settled_it_refuses_the_abort() {
        let (local, partner) = (HOME, PEER);
        let sched = peer_terminating_schedule(1_000);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(7)),
        ));
        let tx_hash = transaction.hash();
        let mut state = state_abandoning(&sched, local, &transaction);

        state.record_settled_txs(
            &sched,
            partner,
            SettledTxSet {
                txs: BTreeSet::from([tx_hash]),
                terminal_wt: WeightedTimestamp::from_millis(1000),
            },
        );

        let abort: Arc<Verifiable<Finalization>> = Arc::new(
            Verified::<Finalization>::seal(abandonment_of(local, partner, tx_hash)).into(),
        );
        assert!(
            state.emit_or_gate_finalized(&sched, abort).is_empty(),
            "the partner settled its half, so this shard may not abort",
        );
        assert!(
            state.gated_finalized.is_empty(),
            "and it is refused rather than held: the set already answered",
        );
    }

    /// A verdict that awaited nobody is not fenced on a counterpart's set.
    ///
    /// A leg's finalization carries only this shard's certificate, as an
    /// abandonment does, but its member awaited nobody: the verdict is
    /// this shard's own, and the core it issued to settles its half on
    /// the record cell rather than on this certificate. Reading it as an
    /// abandonment claim would refuse the leg once the core's settled set
    /// named the transaction — its debit released to the deadline path
    /// after the core had already claimed the crossing.
    #[test]
    fn a_verdict_that_awaited_nobody_is_not_fenced_on_its_counterpart() {
        let (local, partner) = (HOME, PEER);
        let sched = peer_terminating_schedule(1_000);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(7)),
        ));
        let tx_hash = transaction.hash();
        let mut state = state_abandoning(&sched, local, &transaction);
        state.record_settled_txs(
            &sched,
            partner,
            SettledTxSet {
                txs: BTreeSet::from([tx_hash]),
                terminal_wt: WeightedTimestamp::from_millis(1000),
            },
        );

        let verdict = lone_verdict_of(local, tx_hash);
        assert!(
            state
                .fence_pairs(&verdict)
                .iter()
                .all(|(shard, _, _)| *shard == local),
            "a member that awaited nobody names no counterpart",
        );
        let verdict: Arc<Verifiable<Finalization>> =
            Arc::new(Verified::<Finalization>::seal(verdict).into());
        assert!(
            !state.emit_or_gate_finalized(&sched, verdict).is_empty(),
            "the partner settled its half on the record; this shard's verdict is its own",
        );
        assert!(state.gated_finalized.is_empty(), "and nothing is held back");
    }

    /// An abort naming a partner whose settled set can never be read is
    /// refused. The set is what says whether the partner settled, and
    /// opposite questions of it are still questions of it — so a set
    /// nobody can acquire leaves the abort unproven, exactly as it leaves
    /// a settlement unreachable.
    ///
    /// Refusing costs nothing a readable set would have given: a
    /// transaction's deadline falls well inside its partner's evidence
    /// window, so an abort composed at the deadline reads a set that is
    /// still there. Only a late one arrives here.
    #[test]
    fn a_partner_past_its_evidence_window_refuses_the_abort() {
        let (local, partner) = (HOME, PEER);
        let sched = peer_terminating_schedule(1_000);
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(7)),
        ));
        let tx_hash = transaction.hash();
        let mut state = state_abandoning(&sched, local, &transaction);

        // The handoff completed long enough ago that the set has stopped
        // answering: expiry = the stamp's window end plus the evidence
        // window, and the committed frontier sits past it.
        let sched = peer_terminating_schedule_stamped(1_000, Some(Epoch::new(0)));
        state.record_settled_txs(
            &sched,
            partner,
            SettledTxSet {
                txs: BTreeSet::new(),
                terminal_wt: WeightedTimestamp::ZERO,
            },
        );
        state.committed_ts = WeightedTimestamp::from_millis(6_001);

        let abort: Arc<Verifiable<Finalization>> = Arc::new(
            Verified::<Finalization>::seal(abandonment_of(local, partner, tx_hash)).into(),
        );
        assert!(
            state.emit_or_gate_finalized(&sched, abort).is_empty(),
            "past the window the set cannot establish that the partner did not settle",
        );
        assert!(
            state.gated_finalized.is_empty(),
            "and it is refused rather than held: no later set will answer",
        );
    }

    /// A departure the ledger recorded before its window went is
    /// stamped off the head's boundary record when the handoff completes,
    /// though no retained window lists it any more — and the entry a
    /// record covers against it then retires on that clock rather than
    /// holding the departure open for good.
    #[test]
    fn a_departure_no_window_carries_is_stamped_off_the_head() {
        let (left, right) = PEER.children();
        let stamped = TopologySchedule::single(leaves_snap_departed(
            &[HOME, left, right],
            &[],
            &[(PEER, Some(Epoch::new(0)))],
        ));
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(7)),
        ));
        let tx_hash = transaction.hash();
        let mut state = state_abandoning(&stamped, HOME, &transaction);
        state
            .unresolved
            .record_terminal(PEER, WeightedTimestamp::from_millis(1000), None);
        record_peer_left_unsettled(&mut state, tx_hash);
        assert_eq!(state.unresolved.unstamped_departures(), vec![PEER]);

        state.stamp_departures(&stamped);
        assert!(
            state.unresolved.unstamped_departures().is_empty(),
            "the head's stamp reaches a departure no window lists"
        );
        let expiry = stamped
            .handoff_evidence_expiry(PEER)
            .expect("the head carries the stamp");
        assert!(
            state
                .unresolved
                .prune(expiry.plus(Duration::from_millis(1)))
                .iter()
                .any(|entry| entry.tx_hash == tx_hash && entry.covered_by_record),
            "and the covered entry retires past it"
        );

        // With no stamp and no evidence readable at all, the departure
        // closes at the commit that finds it so, as the settled sets do.
        let gone = TopologySchedule::single(leaves_snap(&[HOME, left, right], &[]));
        let mut state = state_abandoning(&gone, HOME, &transaction);
        state
            .unresolved
            .record_terminal(PEER, WeightedTimestamp::from_millis(1000), None);
        record_peer_left_unsettled(&mut state, tx_hash);
        state.stamp_departures(&gone);
        assert!(state.unresolved.unstamped_departures().is_empty());
        assert!(
            state
                .unresolved
                .prune(state.committed_ts.plus(Duration::from_millis(1)))
                .iter()
                .any(|entry| entry.tx_hash == tx_hash && entry.covered_by_record),
            "an unreadable departure closes at once"
        );
    }

    /// An abort naming a partner evicted from every retained window is
    /// refused for the same reason: there is no set to read, and the
    /// certificate this shard already produced is enough for the partner
    /// to have settled against.
    #[test]
    fn a_partner_evicted_from_every_window_refuses_the_abort() {
        // The peer's own window is gone; only the successors of its split
        // are carried. The account outlives the window, so the ledger
        // still names the peer as what held the keyspace then.
        let (left, right) = PEER.children();
        let sched = TopologySchedule::single(leaves_snap(&[HOME, left, right], &[]));
        let transaction: Arc<Verifiable<Transaction>> = Arc::new(Verifiable::from(
            Verified::new_unchecked_for_test(straddling_transaction(7)),
        ));
        let tx_hash = transaction.hash();
        let mut state = state_abandoning(&sched, HOME, &transaction);
        state.unresolved.record_terminal(
            PEER,
            WeightedTimestamp::from_millis(1000),
            Some(WeightedTimestamp::from_millis(1000).plus(EPOCH_DURATION * 5)),
        );

        let abort: Arc<Verifiable<Finalization>> =
            Arc::new(Verified::<Finalization>::seal(abandonment_of(HOME, PEER, tx_hash)).into());
        assert!(
            state.emit_or_gate_finalized(&sched, abort).is_empty(),
            "a shard no retained window carries answers neither question",
        );
        assert!(state.gated_finalized.is_empty(), "and is refused, not held");
    }

    /// A tick already attesting the abandonment withholds the next one:
    /// the ledger releases when that certificate commits, so re-composing
    /// it every commit in between would discard the tick carrying it.
    #[test]
    fn a_tick_already_abandoning_it_withholds_the_next() {
        let schedule = make_test_topology();
        let mut state = make_test_state();
        let tx = test_transaction(1);
        let deadline_ms = 60_000 + u64::try_from(MAX_FINALIZATION_DELAY.as_millis()).unwrap();

        state.on_block_committed(
            &schedule,
            &test_certify(
                make_live_block(
                    BlockHeight::new(1),
                    1_000,
                    ValidatorId::new(0),
                    vec![Arc::new(tx)],
                ),
                1_000,
            ),
        );
        state
            .ticks
            .remove_tick(&TickId::new(ShardId::ROOT, BlockHeight::new(1)));

        let outcomes = abandonment_vote(&mut state, &schedule, 2, deadline_ms);
        assert_eq!(outcomes.len(), 1, "the tick at the deadline abandons it");
        assert!(
            state
                .abandonable(TickId::new(ShardId::ROOT, BlockHeight::new(3)))
                .is_empty(),
            "and the next commit leaves that tick alone",
        );
        assert!(
            state
                .ticks
                .contains_tick(&TickId::new(ShardId::ROOT, BlockHeight::new(2))),
            "so the tick attesting the abort survives to be certified",
        );
        assert_eq!(state.unresolved.len(), 1, "released only when it commits");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // remove_finalization — cascade correctness across all sub-machines.
    // Refactor plan called this out as a key risk: any new sub-machine added
    // to the coordinator must be updated here or its per-tx state leaks.
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_remove_finalization_cascades_across_every_sub_machine() {
        let mut state = make_test_state();
        let (tick_id, tick) = make_ready_local_tick(&[7]);
        let tx_hash = tick.tx_hashes()[0];

        // Seed every sub-machine with state for this tick's tx.
        state.ticks.insert_tick(tick_id, tick);
        state.ticks.assign_tx(tx_hash, tick_id);
        state.provisioning.record_required(
            tx_hash,
            std::iter::once(Requirement::CommittedState(ShardId::leaf(1, 1))).collect(),
            None,
        );
        // Drive finalize to populate the FinalizationStore naturally.
        let _ = state.finalize(&make_test_topology(), &tick_id);
        let finalized = state
            .finalized
            .get_for_tx(tx_hash)
            .expect("tick must be in the finalized store after finalize");

        // Sanity: state is populated across sub-machines.
        let before = state.memory_stats();
        assert_eq!(before.finalizations, 1);
        assert_eq!(before.tick_assignments, 1);
        assert_eq!(before.required_provision_shards, 1);

        state.remove_finalization(&finalized);

        let after = state.memory_stats();
        assert_eq!(after.finalizations, 0);
        assert_eq!(after.ticks, 0);
        assert_eq!(after.tick_assignments, 0);
        assert_eq!(after.verified_provisions, 0);
        assert_eq!(after.required_provision_shards, 0);
        assert_eq!(after.received_provision_shards, 0);
    }

    /// An expectation is stamped with the weighted timestamp of the
    /// commit that composed the tick holding it, so the fallback window
    /// is measured from a real clock reading and the first commit after a
    /// restart cannot read as decades overdue.
    #[test]
    fn a_composed_tick_stamps_its_expectations_with_the_commit_clock() {
        let topo = make_two_shard_topology();
        let mut state = make_test_state_for_shard(ValidatorId::new(0), ShardId::leaf(1, 0));

        let remote_shard = ShardId::leaf(1, 1);
        let tx = Arc::new(test_transaction(1));
        let local_tick = TickId::new(ShardId::leaf(1, 0), BlockHeight::new(10));
        let participating = BTreeSet::from([ShardId::leaf(1, 0), remote_shard]);
        state.ticks.insert_tick(
            local_tick,
            tick_holding(
                local_tick,
                WeightedTimestamp::from_millis(0),
                vec![(verified_arc(&tx), participating)],
            ),
        );
        state.expected_certs.register(
            remote_shard,
            tx.hash(),
            WeightedTimestamp::from_millis(30_000),
        );

        let block = make_live_block_on_shard(
            ShardId::leaf(1, 0),
            BlockHeight::new(1),
            30_000,
            ValidatorId::new(0),
            vec![],
        );
        let (block, qc) = certify(block).into_parts();
        let qc = QuorumCertificate::new(
            qc.block_hash(),
            qc.shard_id(),
            qc.height(),
            qc.parent_block_hash(),
            qc.round(),
            qc.signers().clone(),
            qc.aggregated_signature(),
            WeightedTimestamp::from_millis(30_000),
        );
        let certified = CertifiedBlock::new_unchecked(block, qc);

        let actions = state.on_block_committed(&topo, &certified);

        let fallback_fired = actions
            .iter()
            .any(|a| matches!(a, Action::Fetch(FetchRequest::ExecutionCerts { .. })));
        assert!(
            !fallback_fired,
            "an expectation stamped at the commit clock is not already overdue at that commit"
        );
    }
}
