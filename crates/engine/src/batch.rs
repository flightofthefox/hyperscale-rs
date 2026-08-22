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

use hyperscale_types::{
    Epoch, EpochWindows, ProvisionalHolds, ShardId, ShardTrie, SubstateEntry, TopologySnapshot,
    Transaction, Verified, WeightedTimestamp,
};
pub use hyperscale_vm_types::SeedWindow;

/// What a block fixes about the environment its tick executes under.
///
/// Resolved from the snapshot governing the block rather than from a
/// node's own head: a node's head advances as it folds the beacon, and
/// two members fold at their own pace, so a window taken from one would
/// answer `Pending` on a laggard where it answers `Ready` on a leader —
/// two receipts for one tick. Carried from the commit that composed the
/// tick, on the same terms as the clock beside it.
#[derive(Clone, Debug)]
pub struct TickEnvironment {
    /// The epochs a sealed draw in this batch may settle on.
    ///
    /// Global rather than per transaction: the seeds are the beacon's,
    /// so every shard resolves the same value for the same epoch and
    /// nothing about which block executes a leg reaches the answer.
    pub seeds: SeedWindow,
    /// The grid that turns a member's clock into the epoch a seal it
    /// writes records. Derived rather than carried, because the clock
    /// already travels with the transaction and the epoch is a function
    /// of it.
    pub windows: EpochWindows,
}

impl TickEnvironment {
    /// The environment `snapshot` governs.
    ///
    /// Only the reveal-folded epochs cross into the window. A ceremony
    /// roll is a seed a beacon member could have withheld from, and a
    /// draw settled on one is settled on a value somebody had a lever
    /// over — so a seal maturing into such an epoch answers `Expired`
    /// and the round closes again.
    #[must_use]
    pub fn governing(snapshot: &TopologySnapshot) -> Self {
        let ring = snapshot.seeds();
        Self {
            seeds: SeedWindow::new(
                ring.folded()
                    .map(|(epoch, seed)| (epoch.inner(), *seed.as_bytes()))
                    .collect(),
                ring.newest().map(Epoch::inner),
            ),
            windows: snapshot.epoch_windows(),
        }
    }

    /// An environment no seal can open, over the single-window grid.
    /// For callers with no committed block to take one from.
    #[must_use]
    pub const fn unfolded() -> Self {
        Self {
            seeds: SeedWindow::unfolded(),
            windows: EpochWindows::new(0),
        }
    }
}

/// Per-tick inputs an engine's batch execution reads besides the
/// transactions themselves.
pub struct TickBatchContext<'a> {
    /// The executing vnode's shard — the projection target.
    pub local_shard: ShardId,
    /// The active shard partition.
    pub shard_trie: &'a ShardTrie,
    /// The tick-starting block's parent-QC weighted timestamp. For a
    /// single-shard batch this is the transaction clock of every member;
    /// cross-shard batches carry per-transaction clocks on their inputs.
    pub tick_ts: WeightedTimestamp,
    /// What the tick-starting block fixed about the environment: the
    /// seeds a seal may settle on and the epoch grid a clock resolves
    /// against.
    pub env: TickEnvironment,
    /// Reservations still held by legs of ticks this batch's baseline
    /// cannot see, because nothing an unresolved tick wrote is readable.
    /// The kernel judges a reservation and a debit against committed
    /// balance less what is held, so these are what keep one vault from
    /// funding two withdrawals in successive ticks.
    pub holds: &'a ProvisionalHolds,
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
    /// Whether a tick verdict can still discard this member's effects
    /// after execution — true for a cross-shard leg. Decides both the
    /// reserve fee receipt and the batch's write locality.
    pub abortable: bool,
}
