//! Chain writer trait.
//!
//! Abstracts the prepare-then-commit pattern used by both runners.
//! `prepare_block_commit` returns a [`PreparedCommit`] closure that
//! carries precomputed work; invoking the closure with a
//! [`SyncHint`] applies it efficiently.

use std::sync::Arc;

use hyperscale_types::{
    BeaconWitnessCommit, BlockHeight, CertifiedBlock, FinalizedWave, PreparedCommit, StateRoot,
    Verifiable, Verified,
};

use crate::{BaseReadCache, JmtSnapshot, SubstateDatabase};

/// The block a new one builds on: what it committed, and the state it
/// left behind.
///
/// One value rather than three parameters because they answer one
/// question and have to agree. A settling receipt says what it *moved*,
/// and a movement is nothing without the value it moves from — so a
/// `state` that did not come from the block at `height` would resolve
/// movements against the wrong baseline and fork the state root against
/// every replica that anchored correctly.
pub struct ParentAnchor<'a> {
    /// The parent's committed state root.
    pub state_root: StateRoot,
    /// The parent's height.
    pub height: BlockHeight,
    /// The state as the parent left it — what this block's writes land
    /// on. Anchored at the parent, which is not always the committed
    /// tip: a proposer builds on blocks that have not persisted yet.
    pub state: &'a dyn SubstateDatabase,
}

/// Abstracts state commitment for both simulation and production storage.
///
/// The prepare/commit flow:
/// 1. `prepare_block_commit` computes the speculative state root and returns
///    `(state_root, jmt_snapshot, prepared)`. The closure captures
///    everything needed to perform the commit; the snapshot rides into
///    `PendingChain` so child verifications can chain on top of speculative
///    state.
/// 2. The runner stores the closure (keyed by block hash or however it likes).
/// 3. At commit time, the runner invokes each closure with a `SyncHint`,
///    batching fsyncs across the flush.
/// 4. If no closure is available (e.g. sync blocks without verification),
///    `commit_block` recomputes from scratch.
///
/// Execution certificates are extracted from `block.certificates` (wave certs
/// contain the ECs directly) — no separate parameter needed.
///
/// All methods take `&self` — implementations use interior mutability.
pub trait ShardChainWriter: Send + Sync + 'static {
    /// Compute speculative state root and return precomputed commit work
    /// as a closure.
    ///
    /// Extracts and merges the writes from each finalized wave's receipts
    /// internally, then computes the speculative JMT root.
    ///
    /// `parent_block_height` is the height of the parent block whose state we
    /// build on. Used as the JMT parent version for `put_at_version`. This
    /// must be a committed height or have its tree nodes provided via
    /// `pending_snapshots`.
    ///
    /// `block_height` is the height of the block being prepared (used as JMT
    /// new version).
    ///
    /// `pending_snapshots` contains JMT snapshots from prior verifications
    /// that haven't been committed yet. Their tree nodes are overlaid on the
    /// base store so chained verifications can find parent nodes.
    ///
    /// `base_reads` is an optional cache of reads observed through the
    /// originating `SubstateView` during execution. When provided, it
    /// lets the commit path skip a `multi_get_cf` on `StateCf` for keys
    /// already read — a large fraction at high TPS. Callers without a
    /// view (e.g. sync / tests) pass `None`; implementations fall back
    /// to reading `StateCf` for any key missing from the cache.
    ///
    /// Returns `(computed_state_root, jmt_snapshot, prepared)`.
    fn prepare_block_commit(
        self: &Arc<Self>,
        parent: ParentAnchor<'_>,
        finalized_waves: &[Arc<Verifiable<FinalizedWave>>],
        block_height: BlockHeight,
        pending_snapshots: &[Arc<JmtSnapshot>],
        base_reads: Option<&BaseReadCache>,
    ) -> (StateRoot, Arc<JmtSnapshot>, PreparedCommit);

    /// Commit a block's state writes from scratch (no prepared closure).
    ///
    /// Extracts receipts and execution certificates from `block.certificates`,
    /// merges the writes internally. The `witness` carries the
    /// beacon-witness leaves to fold into the same atomic batch. Used when
    /// no `PreparedCommit` is available (e.g. sync blocks, cache eviction).
    fn commit_block(
        &self,
        certified: &Arc<Verified<CertifiedBlock>>,
        witness: &BeaconWitnessCommit,
    ) -> StateRoot;

    /// Memory usage of storage caches in bytes: `(block_cache, memtable)`.
    ///
    /// Returns `(0, 0)` by default. Overridden by `RocksDB` to report actual usage.
    fn memory_usage_bytes(&self) -> (u64, u64) {
        (0, 0)
    }
}
