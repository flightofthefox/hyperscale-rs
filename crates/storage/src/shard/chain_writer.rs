//! Chain writer trait.
//!
//! Abstracts the prepare-then-commit pattern used by both runners.
//! `prepare_block_commit` returns a [`PreparedCommit`] closure that
//! carries precomputed work; invoking the closure with a
//! [`SyncHint`] applies it efficiently.

use std::sync::Arc;

use hyperscale_types::{
    BeaconWitnessCommit, BlockHeight, CertifiedBlock, Finalization, PreparedCommit, StateRoot,
    Verifiable, Verified,
};

use crate::{Anchored, BaseReadCache, JmtSnapshot};

/// The block a new one builds on: what it committed, and the state it
/// left behind.
///
/// One value rather than five parameters because they answer one
/// question — what state did the parent leave? — and have to agree. A
/// settling receipt says what it *moved*, and a movement is nothing
/// without the value it moves from — so a `state` that did not come
/// from the block at `height` would resolve movements against the wrong
/// baseline and fork the state root against every replica that anchored
/// correctly. The pending chain and the read cache are that same answer
/// in two more forms, and carrying them here is what keeps the tree
/// overlay, the prior capture, and the movement baseline structurally
/// one chain: a mismatched pairing is not expressible.
pub struct ParentAnchor<'a> {
    /// The parent's committed state root.
    pub state_root: StateRoot,
    /// The parent's height.
    pub height: BlockHeight,
    /// The state as the parent left it — what this block's writes land
    /// on. Anchored at the parent, which is not always the committed
    /// tip: a proposer builds on blocks that have not persisted yet, so
    /// only a snapshot can answer for that height.
    pub state: &'a dyn Anchored,
    /// The certified-but-unpersisted ancestors between the committed tip
    /// and the parent, as their verification snapshots. Their tree nodes
    /// overlay the base store for the JMT computation, and their settled
    /// writes are the priors a prepared batch judges its no-op skip and
    /// history capture against — a batch built here applies only after
    /// they have. Empty when the parent is the committed tip.
    pub pending: &'a [Arc<JmtSnapshot>],
    /// Reads observed through the originating `SubstateView` during
    /// execution, keyed at the persisted base. Lets the prior capture
    /// skip a `StateCf` multi-get for keys already read — a large
    /// fraction at high TPS. `None` for callers without a view (sync,
    /// genesis, tests); implementations fall back to reading the store.
    pub base_reads: Option<&'a BaseReadCache>,
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
/// Execution certificates are extracted from `block.certificates` (finalizations
/// contain the ECs directly) — no separate parameter needed.
///
/// All methods take `&self` — implementations use interior mutability.
pub trait ShardChainWriter: Send + Sync + 'static {
    /// Compute speculative state root and return precomputed commit work
    /// as a closure.
    ///
    /// Extracts and merges the writes from each finalization's receipts
    /// internally, then computes the speculative JMT root.
    ///
    /// `parent` carries everything "the state the parent left" means —
    /// the anchor root and height, the readable state, the unpersisted
    /// ancestor chain, and the execution read cache; see [`ParentAnchor`].
    /// The parent's height must be a committed height or have its tree
    /// nodes provided via `parent.pending`.
    ///
    /// `block_height` is the height of the block being prepared (used as
    /// the JMT new version).
    ///
    /// Returns `(computed_state_root, jmt_snapshot, prepared)`.
    fn prepare_block_commit(
        self: &Arc<Self>,
        parent: ParentAnchor<'_>,
        finalizations: &[Arc<Verifiable<Finalization>>],
        block_height: BlockHeight,
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
}
