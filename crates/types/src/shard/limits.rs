//! Block content limits.
//!
//! Hard caps on per-block payload sizes. Wire decoders enforce them at
//! decode time, admission paths enforce them at header ingress, and
//! proposers respect them when building blocks.
//!
//! These are protocol invariants, not operator-tunable config: every
//! validator must be able to handle a peak-sized block, so dialing
//! limits down on a single node only degrades that node's responsiveness
//! without reducing the protocol-wide load it has to keep up with.

use hyperscale_vm_types::TX_UNITS;

use crate::WorkInFlight;

/// Hard cap on the number of live transactions any single block can carry.
///
/// Bounds the `tx_hashes` array in [`BlockManifest`](crate::BlockManifest),
/// the `transactions` array inside [`Block`](crate::Block), the
/// `tx_outcomes` array inside any one
/// [`ExecutionCertificate`](crate::ExecutionCertificate) for a tick from
/// this block, and the `transactions` (per-tx state-entry sets) inside
/// any one [`Provisions`](crate::Provisions) batch sourced from this
/// block.
pub const MAX_TXS_PER_BLOCK: usize = 4_096;

/// Cap on the number of shards a block can name as provision targets, at
/// decode time.
///
/// A block can export provisions to at most `num_shards - 1` others. Real
/// deployments run far below this cap; it exists so a peer can't claim a
/// huge target map and force the decoder to build millions of entries
/// before the first frame check fires.
pub const MAX_PROVISION_TARGET_SHARDS: usize = 1_024;

/// Cap on the number of finalized transactions a proposer includes in a
/// single block, summed across all finalizations.
///
/// Truncation is a suffix of the order the proposer offers, which is the
/// order the ticks executed in — a tick settling ahead of one it shares a
/// cell with reverts a committed write, so nothing here may reorder to
/// fit. Also serves as the outer-`Vec<Finalization>` decode bound: every
/// tick's local EC carries at least one outcome in practice, so the count
/// of finalizations a block can carry is implicitly bounded by this
/// same cap.
pub const MAX_FINALIZED_TX_PER_BLOCK: usize = 8_192;

/// Hard cap on the number of provision batches any single block can carry.
///
/// A [`Provisions`](crate::Provisions) batch is keyed on `(source_shard,
/// target_shard, source_block_height)`. The count per local block scales
/// with the number of remote shards we depend on for cross-shard work
/// and the recent source-block-heights we still need state from. Sized
/// for small-to-mid-shard topologies; widening the topology may require
/// revisiting.
pub const MAX_PROVISIONS_PER_BLOCK: usize = 256;

/// How far the drain's transaction *count* may run past the depth its
/// work budget is sized for, when every transaction is as cheap as one
/// can be.
///
/// One scalar has to bound two things: how much work the drain owes, and
/// how many transactions owe it. Those stay within a factor of each other
/// only while the engine's fixed per-transaction charge is comparable to
/// [`MAX_GAS_LIMIT`] — otherwise a flood of zero-gas transactions fits
/// the same budget as a handful of heavy ones and the count runs free.
/// Two is chosen rather than derived; what the assert below does is hold
/// the gas ceiling to it.
const DRAIN_COUNT_SLACK: u64 = 2;

/// How much unsettled work this shard's chain may owe at once.
///
/// The packing bound: a proposer adds transactions only while the
/// drain's summed work stays under this, so a shard that is not settling
/// admits less until it does. Replaces the transaction *count* as the
/// packing rule — counting priced a publish and a transfer the same, and
/// bounded the drain by how many transactions it held rather than by
/// what they would cost to execute and settle.
///
/// A block carrying transactions is valid only if the total it leaves is
/// under this, so a chain of valid blocks never owes more than the
/// budget. A block carrying none is exempt whatever the total reads:
/// those are the blocks that carry the certificates the drain retreats
/// on, and refusing them would leave a chain that somehow sat above the
/// budget no way back down.
///
/// The total advances on commit and retreats when a certificate resolves
/// the transaction, whichever verdict it carries: one still unresolved at
/// its own deadline is certified aborted at the reservation its block
/// took, so a shard that stops receiving traffic returns to zero.
///
/// Sized like the count it replaces: a full pipeline of blocks
/// (commit → execute → certify) at a representative gas limit, so a
/// shard settling normally never feels it. Every number here is a
/// placeholder, and they calibrate together against measured throughput
/// rather than one at a time.
pub const MAX_DRAIN_WORK: u64 = 3 * MAX_TXS_PER_BLOCK as u64 * (TX_UNITS + MAX_GAS_LIMIT);

/// The count bound the engine's fixed charge exists to provide, asserted
/// rather than assumed: the cheapest a transaction can be is that charge
/// alone, so this is how many the drain could ever hold at once.
///
/// It is [`MAX_GAS_LIMIT`] that has to give if this fails. The charge is
/// the engine's, set against its own schedule, and the ceiling is the
/// one quantity here free to be chosen — so raising the ceiling past the
/// charge is what unbounds the count, and this is where that is caught.
const _: () = assert!(
    MAX_DRAIN_WORK / TX_UNITS <= DRAIN_COUNT_SLACK * 3 * MAX_TXS_PER_BLOCK as u64,
    "the work budget must bound the drain's transaction count, not only its weight",
);

/// Whether a block carrying `tx_count` transactions, and leaving the
/// drain owing `work_in_flight`, is one a validator may vote for.
///
/// `work_in_flight` is what the block leaves owing, not what it
/// inherited, so the bound is on the level a block produces: one that
/// would carry the drain past the budget is refused, and a chain whose
/// blocks all pass this never exceeds it.
///
/// A block that adds nothing is exempt from the level entirely. Those are
/// the blocks that carry the certificates the drain retreats on, so
/// refusing them would be refusing the only way back under.
#[must_use]
pub const fn drain_admits_block(work_in_flight: WorkInFlight, tx_count: usize) -> bool {
    tx_count == 0 || work_in_flight.inner() <= MAX_DRAIN_WORK
}

/// The largest execution ceiling a transaction may sign for.
///
/// A sender's `gas_limit` is theirs to choose, and it enters the drain
/// budget at face value — so without a bound one envelope could reserve
/// the whole of it and stall the shard for the price of a single
/// signature. The engine's per-invocation fuel backstop is a different
/// thing: it stops a runaway guest, not a runaway declaration.
///
/// Bounded above by the engine's fixed per-transaction charge and
/// [`DRAIN_COUNT_SLACK`], which the assert above enforces: a ceiling far
/// past that charge lets a handful of heavy envelopes fill the budget a
/// flood of trivial ones also fills, and the drain stops bounding the
/// count.
///
/// Placeholder like the rest of the work model, and set against what
/// the heaviest legitimate transaction actually needs.
pub const MAX_GAS_LIMIT: u64 = 1_000_000;

/// Hard cap on `header.round() - header.parent_qc().round()` — how many
/// skipped consensus rounds a single block may span.
///
/// Via the shard pacemaker's ceiling (`high_qc.round + MAX_ROUND_GAP`), it
/// also caps how far the view can ever run past certified progress.
///
/// Every validator re-derives one `MissedProposal` beacon-witness leaf per
/// skipped round when verifying and committing a block (see
/// [`missed_proposals_since_prev_commit`](crate::missed_proposals_since_prev_commit)),
/// so an unbounded round gap is an unbounded per-block allocation. The
/// proposer for `(height, round)` rotates with `round`, so a Byzantine
/// validator is the deterministic proposer for arbitrarily large rounds:
/// without this cap, one self-named header at `round ≈ u64::MAX` forces
/// every honest validator to materialize a `Vec` of that length.
///
/// The value is the shard's stall runway. Round gaps accrue only through
/// 2f+1 timeout quorums (Byzantine nodes alone can't advance the pacemaker),
/// each costing one view-change timeout — 30s at the backoff cap — so the
/// cap is reached after roughly `100_000` × 30s ≈ 35 days of continuous
/// certification stall, at which point the view parks at the ceiling and the
/// shard needs operator recovery. The wire cap and the pacemaker ceiling
/// must be the same constant: the view must never enter a round where no
/// proposal extending an adoptable QC would be wire-valid.
pub const MAX_ROUND_GAP: u64 = 100_000;

#[cfg(test)]
mod tests {
    use super::{MAX_DRAIN_WORK, WorkInFlight, drain_admits_block};

    /// The bound bites on the level a block leaves: one carrying
    /// transactions is refused the moment its own total clears the
    /// budget, and admitted right up to it.
    #[test]
    fn a_block_is_refused_for_the_total_it_leaves() {
        let over = WorkInFlight::new(MAX_DRAIN_WORK + 1);
        assert!(!drain_admits_block(over, 1));
        assert!(drain_admits_block(WorkInFlight::new(MAX_DRAIN_WORK), 1));
    }

    /// And it never bites on a block that adds nothing. Those carry the
    /// certificates that release the drain, so refusing them would leave a
    /// chain that touched the ceiling unable to come back under it.
    #[test]
    fn a_block_adding_nothing_is_admitted_at_any_total() {
        assert!(drain_admits_block(WorkInFlight::new(u64::MAX), 0));
    }
}
