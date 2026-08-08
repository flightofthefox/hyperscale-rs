//! What a tick's batch is executed against.
//!
//! The snapshot borrow, the per-tick context, and the cross-shard input
//! an [`Executor`](crate::Executor) reads besides the transactions
//! themselves.
//!
//! Storage is NOT owned by the executor — the runner provides it as a
//! method argument so the same executor can serve multiple snapshots
//! and so the runner can hoist a single snapshot across an entire
//! action batch.
//!
//! Execution is READ-ONLY: results are returned as `ExecutedTx` values
//! whose writes the state machine caches and applies later, when the
//! tick's certificate is included in a committed block.

use std::sync::Arc;

use hyperscale_dispatch::Parallelism;
use hyperscale_types::{
    BlockHash, ProvisionalHolds, RevealChain, ShardId, ShardTrie, SubstateEntry, Transaction,
    Verified, WeightedTimestamp,
};

/// Per-tick inputs an engine's batch execution reads besides the
/// transactions themselves.
pub struct TickBatchContext<'a> {
    /// Batch fan-out strategy, sourced from the dispatch backend.
    pub par: Parallelism,
    /// The executing vnode's shard — the projection target.
    pub local_shard: ShardId,
    /// The active shard partition.
    pub shard_trie: &'a ShardTrie,
    /// The block whose tick this batch executes.
    pub block_hash: BlockHash,
    /// The tick-starting block's parent-QC weighted timestamp. For a
    /// single-shard batch this is the transaction clock of every member;
    /// cross-shard batches carry per-transaction clocks on their inputs.
    pub tick_ts: WeightedTimestamp,
    /// The tick-starting block's reveal chain. For a single-shard batch
    /// this is the randomness anchor of every member; cross-shard
    /// batches carry per-transaction anchors on their inputs.
    pub tick_reveal: RevealChain,
    /// Reservations still held by legs of ticks this batch's baseline
    /// cannot see, because nothing an unresolved tick wrote is readable.
    /// The kernel judges a reservation and a debit against committed
    /// balance less what is held, so these are what keep one vault from
    /// funding two withdrawals in successive ticks.
    pub holds: &'a ProvisionalHolds,
}

/// One cross-shard transaction as an engine consumes it: the
/// transaction plus what its remote counterparts shipped.
pub struct CrossShardTxInput<'a> {
    /// The transaction to execute.
    pub transaction: &'a Arc<Verified<Transaction>>,
    /// Verified provision entry lists, one per source shard contribution.
    pub provisions: &'a [Arc<Vec<SubstateEntry>>],
    /// The transaction clock: the payer-shard committing block's
    /// parent-QC weighted timestamp, identical on every participant.
    pub clock: WeightedTimestamp,
    /// The randomness anchor: the same block's reveal chain, likewise
    /// identical on every participant.
    pub randomness: RevealChain,
}

/// One member of a tick's batch.
///
/// A single-shard transaction or a cross-shard leg, each carrying the
/// environment its committing block fixed. The whole tick executes as
/// one batch, so the executor's canonical order and conflict groups
/// sequence members across ticks.
pub struct TickTxInput<'a> {
    /// The transaction to execute.
    pub transaction: &'a Arc<Verified<Transaction>>,
    /// Verified provision entry lists, one per source shard contribution.
    /// Empty for a single-shard member.
    pub provisions: &'a [Arc<Vec<SubstateEntry>>],
    /// The transaction clock, identical on every participant: the
    /// tick anchor for a single-shard member, the payer-shard
    /// committing block's parent-QC weighted timestamp for a cross-shard
    /// leg.
    pub clock: WeightedTimestamp,
    /// The randomness anchor, resolved the same way as `clock`.
    pub randomness: RevealChain,
    /// Whether a tick verdict can still discard this member's effects
    /// after execution — true for a cross-shard leg. Decides both the
    /// reserve fee receipt and the batch's write locality.
    pub abortable: bool,
}
