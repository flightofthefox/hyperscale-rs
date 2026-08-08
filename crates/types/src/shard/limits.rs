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

/// Hard cap on the number of live transactions any single block can carry.
///
/// Bounds the `tx_hashes` array in [`BlockManifest`](crate::BlockManifest),
/// the `transactions` array inside [`Block`](crate::Block), the
/// `tx_outcomes` array inside any one
/// [`ExecutionCertificate`](crate::ExecutionCertificate) for a wave from
/// this block, and the `transactions` (per-tx state-entry sets) inside
/// any one [`Provisions`](crate::Provisions) batch sourced from this
/// block.
pub const MAX_TXS_PER_BLOCK: usize = 4_096;

/// Cap on the number of finalized transactions a proposer includes in a
/// single block, summed across all wave certificates.
///
/// Older waves (by kickoff `block_height`) are prioritized over newer
/// ones. Also serves as the outer-`Vec<FinalizedWave>` decode bound:
/// every wave's local EC carries at least one outcome in practice, so
/// the count of wave certificates a block can carry is implicitly
/// bounded by this same cap.
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
/// only while the fixed per-transaction charge is comparable to the
/// largest gas limit a sender may declare — otherwise a flood of
/// zero-gas transactions fits the same budget as a handful of heavy ones
/// and the count runs free. Two is chosen rather than derived; what
/// matters is that it is finite and that the charge below follows from
/// it.
const DRAIN_COUNT_SLACK: u64 = 2;

/// What admitting and tracking any transaction costs a block, before
/// anything it declares.
///
/// Every committed transaction occupies a wave entry, a tick-chain
/// entry, a receipt and mempool tracking whatever its manifest touches,
/// and that cost is per transaction rather than per unit of declared
/// work. Without this term a budget over declared work alone would bound
/// weight and not number, and a flood of minimal zero-gas transactions
/// would be close to free.
///
/// Derived from [`MAX_GAS_LIMIT`] and [`DRAIN_COUNT_SLACK`] rather than
/// chosen beside them, because the three only do their job in
/// combination: picking this independently is what lets a later change
/// to the gas ceiling quietly unbound the count.
///
/// A placeholder like every other quantity in the fee model: phase 6
/// sets the pair against measured baselines.
pub const TX_ADMISSION_WORK: u64 = MAX_GAS_LIMIT / (DRAIN_COUNT_SLACK - 1);

/// How much unsettled work this shard's chain may owe at once.
///
/// The packing bound: a proposer adds transactions only while the
/// drain's summed work stays under this, so a shard that is not settling
/// admits less until it does. Replaces the transaction *count* as the
/// packing rule — counting priced a publish and a transfer the same, and
/// bounded the drain by how many transactions it held rather than by
/// what they would cost to execute and settle.
///
/// Sized like the count it replaces: a full pipeline of blocks
/// (commit → execute → certify) at a representative gas limit, so a
/// shard settling normally never feels it. Every number here is a
/// placeholder; phase 6 calibrates them together against measured
/// throughput.
pub const MAX_DRAIN_WORK: u64 = 3 * MAX_TXS_PER_BLOCK as u64 * (TX_ADMISSION_WORK + MAX_GAS_LIMIT);

/// The count bound the fixed charge exists to provide, asserted rather
/// than assumed: the cheapest a transaction can be is the admission
/// charge, so this is how many the drain could ever hold at once.
const _: () = assert!(
    MAX_DRAIN_WORK / TX_ADMISSION_WORK <= DRAIN_COUNT_SLACK * 3 * MAX_TXS_PER_BLOCK as u64,
    "the work budget must bound the drain's transaction count, not only its weight",
);

/// The largest execution ceiling a transaction may sign for.
///
/// A sender's `gas_limit` is theirs to choose, and it enters the drain
/// budget at face value — so without a bound one envelope could reserve
/// the whole of it and stall the shard for the price of a single
/// signature. The engine's per-invocation fuel backstop is a different
/// thing: it stops a runaway guest, not a runaway declaration.
///
/// Placeholder like the rest of the work model; phase 6 sets it against
/// what the heaviest legitimate transaction actually needs.
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
