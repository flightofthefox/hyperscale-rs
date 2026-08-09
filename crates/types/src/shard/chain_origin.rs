//! Where a shard chain starts: genesis height plus start-time anchor.

use crate::{BlockHash, BlockHeight, CommittedTxsRoot, ShardId, WeightedTimestamp};

/// Where a shard chain starts: the height of its genesis block and the
/// weighted-time anchor its genesis QC carries.
///
/// Chains born at network genesis start at height 0 with a `ZERO` anchor
/// ([`Self::ROOT`]). A child chain created by a shard split continues its
/// parent's lines instead of restarting them: its genesis sits at the
/// parent's terminal height + 1 (so JMT versions stay equal to block
/// heights over the hard-linked parent data) and anchors at the parent's
/// final committed canonical weighted timestamp (so the child's BFT clock
/// is continuous with the parent it inherits).
///
/// The origin is a per-chain constant. Consensus components reconstruct
/// the chain's canonical genesis QC from it, so a value that doesn't
/// byte-match the chain's real genesis QC breaks verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainOrigin {
    /// Height of the chain's genesis block.
    pub genesis_height: BlockHeight,
    /// Start-time anchor, carried as the genesis QC's weighted timestamp.
    pub anchor_wt: WeightedTimestamp,
}

impl ChainOrigin {
    /// Origin of a chain born at network genesis: height 0, `ZERO` anchor.
    pub const ROOT: Self = Self {
        genesis_height: BlockHeight::GENESIS,
        anchor_wt: WeightedTimestamp::ZERO,
    };
}

impl Default for ChainOrigin {
    fn default() -> Self {
        Self::ROOT
    }
}

/// A chain this one succeeds, and the commitment it left behind.
///
/// A successor refuses every transaction whose validity window opened
/// before its own origin, because nothing it holds says what ran before
/// the cut. This is what lets it ask: the terminal identifies the block
/// to query, and `committed_txs_root` is what the answer is checked
/// against — read off a header the successor commit-proved, so no server
/// is trusted for it.
///
/// A split child has one. A merged parent has two, and a transaction is
/// only safe to admit when it is absent from both: a proof from one child
/// says nothing about what the other committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PredecessorTerminal {
    /// The shard that terminated — whose committee answers the query.
    pub shard: ShardId,
    /// Height of its terminal block.
    pub height: BlockHeight,
    /// Hash of its terminal block.
    pub block_hash: BlockHash,
    /// The terminal header's committed-transaction commitment.
    pub committed_txs_root: CommittedTxsRoot,
}
