//! A commit-proven state root of one shard at one height, and the clock
//! its block carries.

use hyperscale_hbor::Hbor;

use crate::{BlockHeight, CertifiedBlockHeader, ShardId, StateRoot, Verified, WeightedTimestamp};

/// One shard's state at one committed height, named by the root a
/// commit-proven header carries and dated by the header's clock.
///
/// What a state proof is fetched against and checked against: the
/// shard names the committee that serves the proof, the height names
/// the JMT version it walks, the root is the one the proof has to
/// reconstruct, and the clock is what every window an answer is held
/// to is read against. The root rides the fetch key rather than being
/// looked up at verification, so a response is checked where it lands
/// and a peer serving a proof against some other root is rotated off
/// like any other unusable answer; the clock rides it so a window read
/// off it at commit is read off chain content and never the proposer's
/// word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Hbor)]
pub struct Anchor {
    /// The shard whose state this is.
    pub shard: ShardId,
    /// The committed height the root was taken at.
    pub height: BlockHeight,
    /// The header's state root at that height.
    pub state_root: StateRoot,
    /// The header's parent-QC weighted timestamp.
    pub ts: WeightedTimestamp,
}

impl Anchor {
    /// The anchor a commit-proven header fixes.
    #[must_use]
    pub fn of(header: &Verified<CertifiedBlockHeader>) -> Self {
        Self {
            shard: header.shard_id(),
            height: header.height(),
            state_root: header.state_root(),
            ts: header.header().parent_qc().weighted_timestamp(),
        }
    }
}
