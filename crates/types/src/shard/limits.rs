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

/// Hard cap on the sweepable cells one block's transactions may create
/// between them.
///
/// A sweepable cell is state the chain must carry until a sweep retires
/// it, so what bounds the resident population is the pair of rates: how
/// fast cells are created and how fast they are removed. Removal is
/// capped by [`MAX_SWEEP_PER_BLOCK`], and a creation rate above that cap
/// is a backlog that grows for as long as the load lasts. Capping only
/// the removal side would leave a sweep that bounds ordinary operation
/// and not the peak, which is not a bound.
///
/// Three families are sweepable and share that capacity — the nullifier,
/// the escrow claim (a reclaim's included) and the committed-transaction
/// cell the chain writes for every transaction it carries — because the
/// removal capacity they draw on is one capacity. A budget per family
/// would cost throughput as well as counters: a block could be invalid
/// on one family's peak while the shared walk had room.
///
/// The count has a fourth term that is not one of them. An escrow record
/// is a balance, retired by whoever consumes it and reachable by no
/// clock, so no sweep removes it and it draws on none of the capacity
/// this cap rations. It is counted anyway, and counting it only tightens
/// the ceiling on the three: the alternative is a figure whose slack
/// depends on how many of a block's edges cross.
///
/// Counted per shard, since that is where a cell lands and where the
/// sweep that retires it runs: a transaction's kernel cells whose owner
/// the shard holds, plus the committed cell. Sized at four times
/// [`MAX_TXS_PER_BLOCK`] so a full block of any shape the corpus
/// produces stays admissible on its busiest shard — a transfer's payer
/// writes the record and the committed cell, two; a swap's caller adds
/// the claim of what the venue issued, three; a liquidity provider
/// paying two resources writes two records, one claim and the committed
/// cell, four. A bound subintent adds its nullifier on its signer's
/// shard, so a subintent-heavy block packs fewer transactions, which is
/// the trade the figure already made when nullifiers were the only
/// family. Nothing else in the block budget reaches this: a nullifier
/// costs its transaction a footprint unit and a signature, so the work
/// budget admits thirty-two of them per transaction and the creation
/// ceiling would otherwise sit two orders of magnitude above any
/// removal count a block can carry.
pub const MAX_SWEEPABLE_CREATED_PER_BLOCK: usize = 4 * MAX_TXS_PER_BLOCK;

/// Hard cap on the cells one block's sweep may remove.
///
/// A count and not a work term: removals earn no fee because they are
/// not optional, and nothing is in flight for them — they resolve inside
/// the block that states them, so they never enter
/// [`MAX_DRAIN_WORK`]. That is the same trade the frontier's mandatory
/// advance makes, and it is why a proposer cannot decline to sweep in
/// order to keep block space for what does pay.
///
/// One walk over the three sweepable families, in key order: the bucket
/// leads every sweepable cell's local key, so a block's removals are one
/// contiguous range whatever families they mix, and the cap is on the
/// range rather than on any family's share of it.
///
/// Twice the creation cap, so a backlog drains rather than holding
/// station: a shard that fell behind under peak load catches up in
/// bounded time once the load stops, and one running at the creation cap
/// still removes what it creates with room to spare. The margin is also
/// what carries the one family member no block budgets at creation — a
/// reclaim's claim is written where the reclaim runs, one per record and
/// never beside the record's own claim, so it adds at most the record's
/// own rate to what the sweep must retire.
pub const MAX_SWEEP_PER_BLOCK: usize = 2 * MAX_SWEEPABLE_CREATED_PER_BLOCK;

/// The removal cap must outrun the creation cap, or the resident
/// population is bounded by nothing.
const _: () = assert!(
    MAX_SWEEP_PER_BLOCK >= 2 * MAX_SWEEPABLE_CREATED_PER_BLOCK,
    "a sweep must drain a backlog faster than a block can form one",
);

/// Whether a block whose transactions create `sweepable` cells between
/// them is one a validator may vote for.
///
/// On the block's own content, like every other content limit here, so a
/// validator reaches the verdict with no history behind it.
#[must_use]
pub const fn sweep_admits_block(sweepable: usize) -> bool {
    sweepable <= MAX_SWEEPABLE_CREATED_PER_BLOCK
}

/// Hard cap on the transactions a block's
/// [`AbandonmentRecord`](crate::AbandonmentRecord) records may name between
/// them.
///
/// A record answers for the transactions this chain still owes an outcome
/// for that a counterpart can never settle, and how many that can be is
/// what [`MAX_DRAIN_WORK`] already bounds — a transaction is only owed
/// while its reservation stands. So the ceiling is the drain's own count
/// bound rather than a figure of its own, and a counterpart with more
/// outstanding than one block will carry is answered over several, each
/// record standing alone.
///
/// The bound is on the block rather than on any one record, because the
/// drain is one budget shared across every counterpart the block answers
/// for. It doubles as each record's own decode cap, since a single
/// counterpart can hold the whole of it — but that cap alone would let a
/// block carry the budget once per record, which is why the sum is
/// checked as well.
pub const MAX_UNSETTLED_PER_BLOCK: usize = (MAX_DRAIN_WORK / TX_UNITS) as usize;

/// Hard cap on the abandonment records one block may carry.
///
/// A block's records are in strictly ascending `(shard, arm)` order, so
/// it carries at most one per counterpart shard per arm of the evidence
/// vocabulary, and a record may name a live shard — a refusing core, a
/// consumer that claimed — as readily as one that left. So the bound is
/// the shard count times the arms: more records than that names a shard
/// that does not exist or an arm twice. A shard with evidence at several
/// anchors under one arm drains one anchor per block, which costs
/// settlement rate rather than this bound.
pub const MAX_ABANDONMENT_RECORDS_PER_BLOCK: usize =
    MAX_PROVISION_TARGET_SHARDS * UNSETTLEABLE_ARMS;

/// The arms of the record vocabulary: departed, refused, unclaimed,
/// lapsed and claimed. Stated beside the cap that multiplies by it.
pub const UNSETTLEABLE_ARMS: usize = 5;

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

/// Cap on the transactions a successor asks about in one
/// committed-transaction query.
///
/// Bounds the response, whose per-entry cost is an absence proof rather
/// than a hash. The population it draws from is small by construction:
/// only transactions submitted to the successor whose validity window
/// opened before its origin, and only until the chain outlives that
/// origin by `MAX_VALIDITY_RANGE`. A successor with more than this
/// outstanding asks across several requests.
pub const MAX_COMMITTED_TX_QUERY: usize = 256;

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
/// the transaction, whichever verdict it carries. One still unresolved at
/// its own deadline is certified aborted at the reservation its block
/// took — unless a certificate of the shard's own already covers it, in
/// which case a counterpart could have settled against that certificate,
/// and only that counterpart's departure makes the abort admissible. A
/// straddler waiting on a counterpart that never leaves holds its
/// reservation for as long as it waits.
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
    use super::{
        MAX_DRAIN_WORK, MAX_SWEEPABLE_CREATED_PER_BLOCK, WorkInFlight, drain_admits_block,
        sweep_admits_block,
    };

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

    /// The creation cap bites at the cell. That it is outrun by the
    /// removal cap — the pair being what bounds the resident population,
    /// where either alone bounds one side of it — is the const assert's
    /// business rather than this one's.
    #[test]
    fn a_block_may_create_up_to_the_cap_and_no_more() {
        assert!(sweep_admits_block(MAX_SWEEPABLE_CREATED_PER_BLOCK));
        assert!(!sweep_admits_block(MAX_SWEEPABLE_CREATED_PER_BLOCK + 1));
        assert!(sweep_admits_block(0));
    }
}
