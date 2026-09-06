//! Serving-side access to shard state pinned at epoch boundaries.
//!
//! A serving shard member pins its committed state at recent epoch
//! boundary blocks so a joining vnode can snap-sync against the
//! beacon-attested boundary `state_root` while the live store keeps
//! committing and garbage-collecting. The pinning mechanics are the
//! backend's business — `RocksDB` uses hard-link checkpoints, the
//! in-memory store serves its versioned tree directly — so the node
//! layer's boundary trigger and range-serving code stay generic and run
//! identically in simulation and production.

use hyperscale_jmt::{Key, TreeReader};
use hyperscale_types::{
    BeaconWitnessLeafCount, Block, BlockHeight, ChainOrigin, ShardWitnessPayload, StateRoot,
    SubstateKey, SubstateLeaf,
};

use crate::Substates;

/// The default number of boundary pins a backend retains before
/// evicting the oldest.
///
/// The memory backend's fixed retention and the `RocksDB` config
/// default. Production serving retention must cover the join budget
/// plus the attestation lag of the anchor a joiner selects, so the
/// validator overrides it with the chain-derived
/// `boundary_retention_epochs`.
pub const BOUNDARY_RETAIN: usize = 3;

/// The beacon-witness window a snap-synced import seeds alongside the
/// state.
///
/// Payloads for `[base, base + payloads.len())`, already verified
/// against the anchor header's witness commitment by the assembler.
/// Restores what a store that committed through the boundary would hold
/// in its witness column: the accumulator rebuilds from it on restart,
/// and the beacon fold's witness fetches answer from it. Empty for an
/// import with no witness history — a reshape successor's fresh domain.
#[derive(Debug, Clone, Default)]
pub struct WitnessSeed {
    /// Absolute leaf index of `payloads[0]` — the anchor window's base.
    pub base: BeaconWitnessLeafCount,
    /// The window's payloads in leaf-index order.
    pub payloads: Vec<ShardWitnessPayload>,
}

/// One sub-range's fetch cursor within a staged import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportCursor {
    /// Next un-fetched key (inclusive). Meaningless when `done`.
    pub next: Key,
    /// Last key of the sub-range (inclusive).
    pub end: Key,
    /// Whether the sub-range is exhausted through `end`.
    pub done: bool,
}

/// The durable progress record of a streamed boundary import, written
/// atomically with every staged chunk.
///
/// Binds the staged data to the exact anchor its chunks were proven
/// against: staged leaves are meaningless against any other
/// `state_root`, so a resume whose attested anchor (or fetch geometry)
/// differs must wipe the staging area and start fresh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportProgress {
    /// The anchor height the chunks verify at — the height finalize
    /// installs.
    pub anchor_height: BlockHeight,
    /// The attested state root every staged chunk was proven into.
    pub anchor_state_root: StateRoot,
    /// The sub-range fan-out the cursors were partitioned under.
    pub split_bits: u8,
    /// The per-request leaf limit the cursors advanced by.
    pub chunk_limit: u32,
    /// Accumulated leaf value bytes across every staged chunk — the
    /// imported substate byte total once the assembly completes. Restored
    /// on resume so the recovered state's byte frontier covers chunks
    /// staged before the restart.
    pub staged_bytes: u64,
    /// Per-sub-range fetch cursors.
    pub cursors: Vec<ImportCursor>,
}

/// How a reshape successor's store reaches the version its genesis sits at
/// — the only thing that differs between the three adoptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdoptSource {
    /// A split child seeded by checkpoint-cloning its parent: the child
    /// subtree is extracted from the parent's root node and re-pointed at
    /// the genesis version.
    ParentSubtree,
    /// A split child assembled by an observer: the store's own tip *is* the
    /// adopted subtree — snap-synced at the parent's anchor, then carried
    /// forward by following the parent's child-half writes. The version
    /// line is sparse on the parent's heights, so the tip's root node is
    /// re-pointed at the genesis version.
    FollowedTip,
    /// A merged parent's union, already assembled at the genesis version by
    /// the boundary import. Nothing to re-point; unlike a split child the
    /// prefix may be the trie root.
    InPlace,
}

/// Whether a JMT at `height` carrying `root` already holds authenticated
/// state — anything a snap-sync import would overwrite.
///
/// The gate every import call owes the [`BoundaryStore`] contract, stated
/// once rather than spelled at each of them. Neither term implies the
/// other: a store past `BlockHeight::GENESIS` always holds state, and one
/// at `GENESIS` holds it as soon as a genesis build or a reshape clone has
/// filled its trie. Callers pass the pair they read under whatever lock
/// makes the two atomic for their backend.
///
/// Not "has a chain to resume from". A reshape seat boots with its
/// coordinator at `GENESIS` over a trie its adoption already filled, so a
/// caller choosing between resuming a chain and snap-syncing one asks the
/// chain, not this.
#[must_use]
pub fn holds_state(height: BlockHeight, root: StateRoot) -> bool {
    height != BlockHeight::GENESIS || root != StateRoot::ZERO
}

/// Pin and serve committed state at epoch boundary heights.
pub trait BoundaryStore {
    /// A pinned boundary opened for serving: the JMT at the pinned
    /// version plus substate reads at that same state — a leaf
    /// enumerated out of the tree is read back by its own key.
    type Boundary: TreeReader + Substates + Send;

    /// Pin the committed state at `height` — the shard's epoch boundary
    /// block — keeping a backend-configured number of recent pins
    /// (default [`BOUNDARY_RETAIN`]). A pinned boundary must outlive
    /// the join budget of a peer syncing against it. Idempotent per
    /// height.
    ///
    /// # Errors
    ///
    /// Returns a description of the failure (e.g. checkpoint I/O). A
    /// failed pin degrades serving, never correctness — callers log and
    /// continue.
    fn pin_boundary(&self, height: BlockHeight) -> Result<(), String>;

    /// Open the pin at exactly `height`, or `None` if it was never
    /// pinned or has been evicted from the ring.
    fn open_boundary(&self, height: BlockHeight) -> Option<Self::Boundary>;

    /// Durably stage one verified snap-sync chunk together with the
    /// import's progress record, in one atomic write. The store proper is
    /// untouched — staged bytes are not an import until
    /// [`Self::finalize_boundary_import`] builds the state from them —
    /// so the empty-store gate keeps its meaning throughout the
    /// assembly. Chunks may stage under different progress snapshots
    /// (a merge keeper stages both terminating children's disjoint
    /// spans into one store); the record reflects the latest write.
    ///
    /// # Errors
    ///
    /// Returns a description of the failure. Staging into a non-empty
    /// store is an error — the import is a bootstrap, not a merge.
    fn stage_import_chunk(
        &self,
        progress: &ImportProgress,
        leaves: &[SubstateLeaf],
    ) -> Result<(), String>;

    /// The staged import's progress record, or `None` when nothing is
    /// staged.
    fn read_import_progress(&self) -> Option<ImportProgress>;

    /// Discard every staged chunk, the progress record, and any partial
    /// state a [`finalize_boundary_import`](Self::finalize_boundary_import)
    /// interrupted before its completion marker left behind — the caller
    /// may build a fresh assembly directly on top of the wiped store. A
    /// no-op when nothing is staged.
    ///
    /// # Errors
    ///
    /// Returns a description of the failure (backend write error).
    fn wipe_import_staging(&self) -> Result<(), String>;

    /// Install the staged boundary state at `height` into this (empty)
    /// store: raw substates, the JMT rebuilt from the staged leaf keys,
    /// and the anchor window's witness payloads — the state-level image
    /// of a store that committed through the boundary. The staging area
    /// is cleared on success. Chain metadata is not touched; tail block-sync
    /// from `height + 1` layers on top.
    ///
    /// Returns the resulting state root, which the caller must compare
    /// against the beacon-attested anchor before trusting the store.
    /// Idempotent under re-runs after a crash mid-finalize: the JMT
    /// metadata write is the completion marker, and re-applying the same
    /// staged leaves overwrites deterministically.
    ///
    /// # Errors
    ///
    /// Returns a description of the failure. Finalizing a non-empty
    /// store is an error — the import is a bootstrap, not a merge.
    fn finalize_boundary_import(
        &self,
        height: BlockHeight,
        witnesses: WitnessSeed,
    ) -> Result<StateRoot, String>;

    /// Apply the subset of a followed chain's block that falls under
    /// this store's prefix, at the block's height — substate values, the
    /// JMT, and the count, advancing the store's version.
    ///
    /// This is how a reshape observer's child-rooted store stays current
    /// with the splitting parent between its snap-synced anchor and the
    /// parent's terminal crossing: the followed blocks are the parent
    /// chain's (QC-trusted by the driver — the observer cannot verify
    /// the parent's full roots from a half store), and partition
    /// independence keeps the resulting root exactly the parent tree's
    /// subtree node at the prefix. The block is applied as the chain
    /// applied it — the receipts its ticks settled, the committed cell of
    /// every transaction it carries, and the sweep its header names — so
    /// the follower's half reads exactly as the parent's. A block that
    /// touches nothing under the prefix is a no-op: the version does not
    /// advance, so the store's version line stays sparse on the parent's
    /// heights.
    ///
    /// Returns the store's state root after the application.
    ///
    /// # Errors
    ///
    /// Returns a description of the failure — a height at or below the
    /// store's current version, or a backend write failure.
    fn follow_block_writes(&self, block: &Block) -> Result<StateRoot, String>;

    /// Install a reshape successor's derived `genesis` as this store's
    /// chain origin and committed tip, returning the adopted state root.
    ///
    /// `source` names only how the tree reaches the genesis version; the
    /// adopted root is then checked against the root the `genesis` names,
    /// which is what gates the seat. A successor's genesis derives from
    /// frozen chain content its duty commit-proved, so it cannot name a
    /// subtree no terminal committed — and a store that does not hold what
    /// the genesis names must not seat.
    ///
    /// Idempotent: a re-run over an already-adopted store returns the
    /// recorded adoption.
    ///
    /// # Errors
    ///
    /// Returns a description of the failure — a genesis block off the
    /// origin's height, a store vintage `source` does not admit, an
    /// unresolvable subtree, or an adopted root the genesis does not name.
    fn adopt_genesis(
        &self,
        origin: ChainOrigin,
        genesis: &Block,
        source: AdoptSource,
    ) -> Result<StateRoot, String>;

    /// The committed substate byte total at `version`, or `None` when the
    /// store's version line doesn't carry it.
    fn substate_bytes_at_version(&self, version: u64) -> Option<u64>;

    /// Every escrow record the committed state holds, with its bytes.
    ///
    /// Derived on demand rather than indexed, because the state is the
    /// authority and the one caller asks once: a reshape successor whose
    /// adoption just filled its trie, and whose ledger begins empty
    /// while the value its predecessors escrowed rides the prefix in.
    /// Nothing else names those records — the entry that would is the
    /// predecessor's ledger's, a fold over a chain the successor never
    /// replays, and the cell is outside every sweep's reach.
    ///
    /// A scan, and affordable for being one: it follows an import that
    /// wrote every leaf it reads.
    fn escrow_records(&self) -> Vec<(SubstateKey, Vec<u8>)>;
}
