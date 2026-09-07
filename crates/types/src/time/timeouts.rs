//! Duration constants that are part of the consensus protocol.
//!
//! Every constant here must be enforced identically on every validator.
//! Two flavors live side by side:
//!
//! - **Retention / abort windows** (`MAX_FINALIZATION_DELAY`, `REMOTE_HEADER_RETENTION`,
//!   `RETENTION_HORIZON`) — durations after which a tick aborts or a piece
//!   of derived state becomes safe to drop on every node simultaneously.
//!   Most downstream invariants derive from `MAX_FINALIZATION_DELAY`.
//! - **shard consensus liveness timers** (`VIEW_CHANGE_TIMEOUT*`, `MAX_PROGRESS_WAIT`) —
//!   round-timer cadences and the absolute ceiling on view-change
//!   suppression while a proposal is in flight. Validators that disagree on
//!   these values either time out asymmetrically (degraded liveness) or
//!   weaken the stall-attack bound that `MAX_PROGRESS_WAIT` enforces.
//!
//! Sub-state-machine-local timeouts (fallback fetch, IO retry backoff, etc.)
//! stay in their owning crate.

use std::time::Duration;

use hyperscale_vm_types::{ESCROW_GRACE_MS, NULLIFIER_GRACE_MS};

use crate::{CLAIM_WINDOW, MAX_VALIDITY_RANGE};

/// The longest a cross-shard transaction may take to finalize, past the
/// last block that could have included it.
///
/// This is the cross-shard settlement window — every retention window
/// that must outlive a transaction in flight is sized in terms of it, and
/// past it the transaction is abandoned rather than waited for. So the
/// last moment a transaction can be included plus this is
/// [`RETENTION_HORIZON`].
///
/// Sized at 3× `VOTE_RETRY_TIMEOUT` (8s) so at least two vote retries can
/// fire against rotated tick leaders inside it.
///
/// Deterministic — measured against the shard consensus-authenticated
/// `weighted_timestamp_ms` of the committing QC, so every validator
/// derives the same deadline.
pub const MAX_FINALIZATION_DELAY: Duration = Duration::from_secs(24);

/// How long to retain remote block headers below each shard's tip.
///
/// Shared by `hyperscale-shard` (deferral-proof verification) and
/// `hyperscale-remote-headers` (provision/exec-cert verification). Measured
/// against the shard consensus-authenticated `weighted_timestamp_ms` on the tip vs the
/// stored header. Sized generously above `MAX_FINALIZATION_DELAY` so late-arriving
/// proofs still find a header to verify against.
pub const REMOTE_HEADER_RETENTION: Duration = Duration::from_secs(30);

/// Single principled retention bound for every artefact derived from a tx
/// — provisions, ECs, mempool tombstones, conflict-detector entries.
///
/// A tx included at the latest possible moment
/// (`weighted_ts ≈ end_timestamp_exclusive - 1ms`) gets `MAX_FINALIZATION_DELAY`
/// after that to terminate (success or abort, both via WC). After both
/// elapse, the tx is provably terminal everywhere — no shard can still
/// need its provision data, EC, or any other artefact. Safe to drop on
/// every node simultaneously.
///
/// Sized from the transaction window and never from
/// [`MAX_SUBINTENT_VALIDITY_RANGE`](crate::MAX_SUBINTENT_VALIDITY_RANGE),
/// which is far wider: what is retained here is derived from a
/// transaction, and a transaction binding a long-standing offer still
/// runs inside its own window.
pub const RETENTION_HORIZON: Duration =
    Duration::from_secs(MAX_VALIDITY_RANGE.as_secs() + MAX_FINALIZATION_DELAY.as_secs());

/// How far back a chain is folded to rebuild the committed-artifact
/// dedup window.
///
/// The widest of the index's tiers. A transaction is held to the close of
/// its delivery window — one [`MAX_VALIDITY_RANGE`] past a validity end
/// that may itself sit a whole range past the block that committed it —
/// so an entry still live can come from a block two ranges back. The
/// resolution and provision tiers are keyed at most
/// [`RETENTION_HORIZON`] past their own block, which this covers.
pub const DEDUP_WINDOW: Duration = Duration::from_secs(MAX_VALIDITY_RANGE.as_secs() * 2);

const _: () = assert!(
    DEDUP_WINDOW.as_secs() >= RETENTION_HORIZON.as_secs(),
    "the dedup walk covers every tier of the index it rebuilds",
);

/// A nullifier's life and every other tx-derived artifact's are the same
/// bound, asserted rather than assumed: the VM keys and values a
/// nullifier by an expiry it computes from its own constant, and this is
/// where the two spellings are held together. If the horizon moves, that
/// constant moves with it — a nullifier swept while some chain can still
/// be deciding a spend of it is a replay, and one retained past it is
/// state nobody can retire.
const _: () = assert!(
    RETENTION_HORIZON.as_secs() * 1_000 == NULLIFIER_GRACE_MS,
    "a nullifier's grace is the protocol's retention horizon",
);

/// An escrow record outlives the nullifier by the claim window: the
/// lapse is proved no earlier than one validity range past the
/// deadline, and the reclaim then has one more to commit before the
/// record it takes back is swept.
const _: () = assert!(
    (MAX_FINALIZATION_DELAY.as_secs() + CLAIM_WINDOW.as_secs()) * 1_000 == ESCROW_GRACE_MS,
    "an escrow cell's grace is the deadline plus the claim window",
);

/// The horizon must not outlive the epoch that produced what it retains.
///
/// A reshape's cut is scheduled one window ahead, so a successor
/// inheriting a predecessor's tx-derived state has one epoch of chain to
/// read it from. A horizon past that reaches back further than the
/// reshape spans, and the successor has to fetch below what it already
/// walks. Half an epoch is the working margin, not the hard bound.
const _: () = assert!(RETENTION_HORIZON.as_secs() < EPOCH_DURATION.as_secs());

/// A skipped epoch and its recovery must not expire the transactions a
/// shard is holding. `SKIP_TIMEOUT` bounds the wait before the pool
/// prevotes a skip, and ratification rounds follow it; a validity window
/// under a small multiple of that turns one stalled epoch into a
/// cleared mempool.
const _: () = assert!(MAX_VALIDITY_RANGE.as_secs() >= 2 * SKIP_TIMEOUT.as_secs());

/// Base view-change timeout for the first round at any height.
///
/// Combined with `VIEW_CHANGE_TIMEOUT_INCREMENT` and capped by
/// `VIEW_CHANGE_TIMEOUT_MAX` to produce the per-round timeout:
/// `min(base + increment * rounds_at_height, max)`. Round numbers are
/// QC- and header-attested, so every validator computes the same
/// effective timeout for any `(height, round)`.
pub const VIEW_CHANGE_TIMEOUT: Duration = Duration::from_secs(3);

/// Linear backoff increment per failed round at the same height.
///
/// Prevents thundering-herd view changes when the network is briefly
/// stressed: each successive round at the same height extends the
/// timeout by this much before the cap kicks in.
pub const VIEW_CHANGE_TIMEOUT_INCREMENT: Duration = Duration::from_secs(1);

/// Cap on the effective view-change timeout after linear backoff.
///
/// Bounds round latency in extreme network conditions so a stuck height
/// can't ratchet timeouts upward indefinitely.
pub const VIEW_CHANGE_TIMEOUT_MAX: Duration = Duration::from_secs(30);

/// Absolute ceiling on view-change suppression while a block is in
/// progress at the proposal tip.
///
/// View changes are normally suppressed while we're fetching block
/// content, awaiting our own QC, or processing the leader's pending
/// block. This cap bounds how long a Byzantine proposer can stall the
/// round timer purely by keeping a header alive without ever advancing
/// the chain. Once this elapses since the last leader-activity reset,
/// the timer fires regardless of pending work.
pub const MAX_PROGRESS_WAIT: Duration = Duration::from_secs(9);

/// Beacon-chain epoch length, measured against committed beacon-slot
/// `weighted_timestamp`.
///
/// Epoch boundaries are time-based, not slot-count-based: a slot's epoch
/// is `(slot.weighted_timestamp - genesis_wt) / EPOCH_DURATION`, derivable
/// independently by every validator without consensus on which slot
/// counts as the boundary. Recovery slots can wedge in mid-epoch without
/// rolling the epoch counter, decoupling committee-replacement from
/// natural epoch rotation.
///
/// Also bounds the witness-inclusion window: a witness leaf is includable
/// in a beacon proposal during epoch `E` if its source block's
/// `weighted_timestamp ≤ t_end_E`.
pub const EPOCH_DURATION: Duration = Duration::from_mins(5);

/// Wall-clock interval an active validator waits past an epoch's
/// expected block time before prevoting the skip block in
/// ratification round 1.
///
/// Loosely synchronized clocks suffice: a validator that prevoted the
/// candidate before its deadline and one that prevoted skip after it
/// split round 1 below both quorums at worst, and round 2 converges.
/// Sized so a normal SPC commit (well under 10 s on a healthy network)
/// never trips the timer, while a genuine stall doesn't burn an entire
/// epoch waiting. Starting value picked mid-range against the 30–60 s
/// envelope; tune from operational data.
pub const SKIP_TIMEOUT: Duration = Duration::from_secs(45);

/// Wall-clock length of one ratification round past the first: a
/// round with neither a commit certificate nor a polka worth waiting
/// on re-prevotes in the next round after this long.
///
/// Long enough for a pool-wide vote broadcast plus aggregation
/// (seconds), short enough that a split round-1 vote converges well
/// within the epoch. A spuriously short value costs an extra benign
/// round; a long one only delays skipping a genuinely dead committee.
pub const RATIFY_ROUND_TIMEOUT: Duration = Duration::from_secs(15);
