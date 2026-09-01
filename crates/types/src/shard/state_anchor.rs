//! A commit-proven state root of one shard at one height.

use crate::{BlockHeight, CertifiedBlockHeader, ShardId, StateRoot, Verified};

/// One shard's state at one committed height, named by the root a
/// commit-proven header carries.
///
/// What a state proof is fetched against and checked against: the
/// shard names the committee that serves the proof, the height names
/// the JMT version it walks, and the root is the one the proof has to
/// reconstruct. The root rides the fetch key rather than being looked
/// up at verification, so a response is checked where it lands and a
/// peer serving a proof against some other root is rotated off like
/// any other unusable answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateAnchor {
    /// The shard whose state this is.
    pub shard: ShardId,
    /// The committed height the root was taken at.
    pub height: BlockHeight,
    /// The header's state root at that height.
    pub state_root: StateRoot,
}

impl StateAnchor {
    /// The anchor a commit-proven header fixes.
    #[must_use]
    pub fn of(header: &Verified<CertifiedBlockHeader>) -> Self {
        Self {
            shard: header.shard_id(),
            height: header.height(),
            state_root: header.state_root(),
        }
    }
}
