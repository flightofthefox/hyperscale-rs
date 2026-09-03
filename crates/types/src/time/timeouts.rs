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

use crate::{MAX_VALIDITY_RANGE, WeightedTimestamp};

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

/// The earliest core-shard anchor at which a transaction's absence from
/// the committed set means anything: its validity end plus
/// [`MAX_FINALIZATION_DELAY`].
///
/// Before it the core may still legitimately commit — admission fences
/// the core at `block_wt < validity_end`, and the extra term is the
/// propagation budget for that block to be committed and served, not
/// admission slack. Both sides derive this from signed content, so it is
/// the same figure everywhere without coordinating.
#[must_use]
pub fn reclaim_probe_anchor(validity_end: WeightedTimestamp) -> WeightedTimestamp {
    validity_end.plus(MAX_FINALIZATION_DELAY)
}

/// The validity end a transaction's deadline was derived from: the
/// inverse of [`reclaim_probe_anchor`], for a voter holding a name's
/// deadline and no body.
#[must_use]
pub fn validity_end_of(deadline: WeightedTimestamp) -> WeightedTimestamp {
    deadline.minus(MAX_FINALIZATION_DELAY)
}

/// The core-shard anchor from which a transaction's committed cell may
/// have been swept: its validity end plus [`RETENTION_HORIZON`].
///
/// That is the cell's own grace. A proof against a block at or past it
/// comes to a cell that may be gone, and its absence says nothing.
#[must_use]
pub fn absence_probe_ceiling(validity_end: WeightedTimestamp) -> WeightedTimestamp {
    validity_end.plus(RETENTION_HORIZON)
}

/// Whether an absence proof taken against a core-shard block at
/// `probe_wt` licenses reclaiming a transaction's escrow.
///
/// A half-open window. Absence at or past the anchor says the core did
/// not commit it and, by its own admission rule, never can; absence
/// before the anchor says nothing, since a core block admitted at
/// `validity_end - 1ms` may still be on its way. And absence at or past
/// the cell's own sweep says nothing either: the committed cell is
/// retired on time, so a proof there is a true proof of a cell that was
/// present. Misreading either end by one term is a double spend, so
/// both inequalities are stated once, here, rather than at each
/// consumer.
#[must_use]
pub fn absence_licenses_reclaim(
    probe_wt: WeightedTimestamp,
    validity_end: WeightedTimestamp,
) -> bool {
    probe_wt >= reclaim_probe_anchor(validity_end) && probe_wt < absence_probe_ceiling(validity_end)
}

/// The moment past which a delivery of a transaction's outbound value
/// is no longer admissible: its validity end plus [`MAX_VALIDITY_RANGE`].
///
/// A delivery bears no verdict, so the transaction's deadline does not
/// bound it: the record cell it claims is durable and consumed once.
/// What bounds the delivery is that a delivery admitted at the last
/// moment has claimed by [`MAX_FINALIZATION_DELAY`] past this close or
/// never will — the same budget the core's admission leaves before the
/// probe anchor — which is where the crossing lapses and its issuer may
/// prove it unclaimed; the record is swept a further
/// [`MAX_VALIDITY_RANGE`] on, the room that reclaim needs. Derived from
/// signed content, so every shard names the same instant.
#[must_use]
pub fn delivery_window_close(validity_end: WeightedTimestamp) -> WeightedTimestamp {
    validity_end.plus(MAX_VALIDITY_RANGE)
}

/// Whether a delivery may still be admitted at `anchor`: past the
/// validity end, since inside it the transaction is admissible as
/// itself, and short of the window's close.
#[must_use]
pub fn delivery_admissible(anchor: WeightedTimestamp, validity_end: WeightedTimestamp) -> bool {
    anchor >= validity_end && anchor < delivery_window_close(validity_end)
}

/// The earliest delivering-shard anchor at which a crossing's claim
/// cell being absent means the crossing lapsed: the delivery window's
/// close plus [`MAX_FINALIZATION_DELAY`].
///
/// A delivery admitted inside the window has committed its claim by
/// then, or never will — the same propagation budget the core's
/// admission leaves before [`reclaim_probe_anchor`], and for the same
/// reason. It is one [`MAX_VALIDITY_RANGE`] past that anchor, which is
/// how a voter holding a name's deadline derives it without the body.
#[must_use]
pub fn lapse_probe_anchor(validity_end: WeightedTimestamp) -> WeightedTimestamp {
    delivery_window_close(validity_end).plus(MAX_FINALIZATION_DELAY)
}

/// The delivering-shard anchor from which a crossing's claim cell may
/// have been swept: the transaction's validity end plus the escrow
/// families' grace.
///
/// The cell's own expiry is keyed to the producing intent's window,
/// which is never earlier than the transaction's, so this is at or
/// before the earliest sweep.
#[must_use]
pub fn lapse_probe_ceiling(validity_end: WeightedTimestamp) -> WeightedTimestamp {
    validity_end
        .plus(RETENTION_HORIZON)
        .plus(MAX_VALIDITY_RANGE)
}

/// Whether an absence proof of a crossing's claim cell, taken against a
/// delivering-shard block at `probe_wt`, licenses reclaiming the
/// crossing.
///
/// A half-open window, as the core's is. Absence at or past the anchor
/// says the delivery never claimed it and, the window having closed,
/// never can; absence before it says nothing, since a delivery admitted
/// at the last moment may still be committing; absence at or past the
/// claim cell's sweep says nothing, since the cell is retired on time.
#[must_use]
pub fn lapse_licenses_reclaim(
    probe_wt: WeightedTimestamp,
    validity_end: WeightedTimestamp,
) -> bool {
    probe_wt >= lapse_probe_anchor(validity_end) && probe_wt < lapse_probe_ceiling(validity_end)
}

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

/// An escrow record outlives the nullifier by the room a lapsed
/// crossing's reclaim needs: the lapse is proved no earlier than the
/// retention horizon past the validity end, and the reclaim then has a
/// validity range — the room every abandonment gets — to commit before
/// the record it takes back is swept.
const _: () = assert!(
    (RETENTION_HORIZON.as_secs() + MAX_VALIDITY_RANGE.as_secs()) * 1_000 == ESCROW_GRACE_MS,
    "an escrow cell's grace is the retention horizon plus a reclaim's room",
);

/// Each absence window is open: the cell a probe asks about outlives
/// the earliest anchor a probe may take against it, by the whole of a
/// validity range, so there is always a header to prove an absence at.
/// A window that closed before it opened would license nothing and
/// strand every escrow whose core fell silent.
const _: () = assert!(
    MAX_FINALIZATION_DELAY.as_secs() + MAX_VALIDITY_RANGE.as_secs() == RETENTION_HORIZON.as_secs(),
    "the committed cell outlives the absence anchor by one validity range",
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use hyperscale_vm_types::ESCROW_GRACE_MS;

    use super::{
        MAX_FINALIZATION_DELAY, RETENTION_HORIZON, absence_licenses_reclaim, absence_probe_ceiling,
        delivery_admissible, delivery_window_close, lapse_licenses_reclaim, lapse_probe_anchor,
        lapse_probe_ceiling, reclaim_probe_anchor, validity_end_of,
    };
    use crate::{MAX_VALIDITY_RANGE, WeightedTimestamp};

    /// A delivery is admissible from the validity end to the window's
    /// close, half-open at both ends the way the window itself is, and
    /// the close sits one finalization delay short of the record's sweep.
    #[test]
    fn the_delivery_window_opens_at_the_validity_end_and_closes_short_of_the_sweep() {
        let validity_end = WeightedTimestamp::from_millis(60_000);
        let close = delivery_window_close(validity_end);
        assert_eq!(close, validity_end.plus(MAX_VALIDITY_RANGE));
        assert_eq!(
            validity_end.plus(RETENTION_HORIZON).elapsed_since(close),
            MAX_FINALIZATION_DELAY,
            "the sweep is a full delay past the close"
        );

        assert!(!delivery_admissible(
            validity_end.minus(Duration::from_millis(1)),
            validity_end
        ));
        assert!(delivery_admissible(validity_end, validity_end));
        assert!(delivery_admissible(
            close.minus(Duration::from_millis(1)),
            validity_end
        ));
        assert!(!delivery_admissible(close, validity_end));
    }

    /// The anchor is a boundary, and a reclaim is licensed on one side
    /// of it and not the other — one millisecond either way.
    #[test]
    fn the_absence_anchor_is_inclusive_at_the_deadline_and_not_before() {
        let validity_end = WeightedTimestamp::from_millis(60_000);
        let anchor = reclaim_probe_anchor(validity_end);
        assert_eq!(anchor, validity_end.plus(MAX_FINALIZATION_DELAY));

        assert!(!absence_licenses_reclaim(
            anchor.minus(Duration::from_millis(1)),
            validity_end
        ));
        assert!(absence_licenses_reclaim(anchor, validity_end));
        assert!(absence_licenses_reclaim(
            anchor.plus(Duration::from_millis(1)),
            validity_end
        ));
        assert_eq!(validity_end_of(anchor), validity_end);
    }

    /// The window closes where the committed cell may be swept: a proof
    /// there is a true proof of a cell that was present, so it licenses
    /// nothing. The lapse window closes at the escrow families' grace,
    /// the claim cell's own sweep, and both close a validity range past
    /// where they open.
    #[test]
    fn an_absence_licenses_nothing_once_the_cell_it_asks_about_may_be_swept() {
        let validity_end = WeightedTimestamp::from_millis(60_000);
        let ceiling = absence_probe_ceiling(validity_end);
        assert_eq!(ceiling, validity_end.plus(RETENTION_HORIZON));
        assert_eq!(
            ceiling.elapsed_since(reclaim_probe_anchor(validity_end)),
            MAX_VALIDITY_RANGE,
        );
        assert!(absence_licenses_reclaim(
            ceiling.minus(Duration::from_millis(1)),
            validity_end
        ));
        assert!(!absence_licenses_reclaim(ceiling, validity_end));
        assert!(!absence_licenses_reclaim(
            ceiling.plus(Duration::from_secs(60)),
            validity_end
        ));

        let lapse_ceiling = lapse_probe_ceiling(validity_end);
        assert_eq!(
            lapse_ceiling,
            validity_end.plus(Duration::from_millis(ESCROW_GRACE_MS)),
            "the claim cell's grace, keyed to a window never earlier than this one",
        );
        assert_eq!(
            lapse_ceiling.elapsed_since(lapse_probe_anchor(validity_end)),
            MAX_VALIDITY_RANGE,
        );
        assert!(lapse_licenses_reclaim(
            lapse_ceiling.minus(Duration::from_millis(1)),
            validity_end
        ));
        assert!(!lapse_licenses_reclaim(lapse_ceiling, validity_end));
    }

    /// A probe at the validity end itself licenses nothing: a core block
    /// admitted a millisecond before it is inside the propagation budget
    /// and may still commit. The gap is the whole of the delay, so the
    /// latest legitimately admitted core block has that long to land.
    #[test]
    fn a_probe_at_the_validity_end_is_inside_the_propagation_budget() {
        let validity_end = WeightedTimestamp::from_millis(60_000);
        let latest_core_admission = validity_end.minus(Duration::from_millis(1));
        assert!(!absence_licenses_reclaim(validity_end, validity_end));
        assert!(
            reclaim_probe_anchor(validity_end).elapsed_since(latest_core_admission)
                > MAX_FINALIZATION_DELAY,
            "the anchor sits a full delay past the last block the core could admit",
        );
    }

    /// The lapse anchor sits one validity range past the reclaim probe's:
    /// the window's close plus the same propagation budget the core's
    /// admission leaves, so a delivery admitted at the last moment has
    /// committed its claim by it or never will. Absence licenses a
    /// reclaim at the anchor and past it, and never a millisecond short.
    #[test]
    fn a_lapse_is_proved_no_earlier_than_the_close_plus_the_finalization_delay() {
        let validity_end = WeightedTimestamp::from_millis(300_000);
        let anchor = lapse_probe_anchor(validity_end);
        assert_eq!(
            anchor,
            reclaim_probe_anchor(validity_end).plus(MAX_VALIDITY_RANGE),
            "one validity range past the core's probe anchor",
        );
        assert_eq!(
            anchor,
            delivery_window_close(validity_end).plus(MAX_FINALIZATION_DELAY)
        );
        assert!(!lapse_licenses_reclaim(
            anchor.minus(Duration::from_millis(1)),
            validity_end
        ));
        assert!(lapse_licenses_reclaim(anchor, validity_end));
        assert!(lapse_licenses_reclaim(
            anchor.plus(Duration::from_secs(1)),
            validity_end
        ));
        assert!(
            !lapse_licenses_reclaim(delivery_window_close(validity_end), validity_end),
            "the close itself is not the lapse: a claim admitted under it may still commit",
        );
    }
}
