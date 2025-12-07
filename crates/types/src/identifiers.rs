//! Domain-specific identifier types.

use sbor::prelude::*;
use std::fmt;

/// Validator identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, BasicSbor)]
#[sbor(transparent)]
pub struct ValidatorId(pub u64);

impl fmt::Display for ValidatorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Validator({})", self.0)
    }
}

/// Shard group identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, BasicSbor)]
#[sbor(transparent)]
pub struct ShardGroupId(pub u64);

impl fmt::Display for ShardGroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shard({})", self.0)
    }
}

/// Block height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, BasicSbor)]
#[sbor(transparent)]
pub struct BlockHeight(pub u64);

impl BlockHeight {
    /// Genesis block height.
    pub const GENESIS: Self = BlockHeight(0);

    /// Get the next block height.
    pub fn next(self) -> Self {
        BlockHeight(self.0 + 1)
    }

    /// Get the previous block height (returns None if at genesis).
    pub fn prev(self) -> Option<Self> {
        if self.0 > 0 {
            Some(BlockHeight(self.0 - 1))
        } else {
            None
        }
    }
}

impl fmt::Display for BlockHeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Block({})", self.0)
    }
}

/// Vote power (stake weight).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, BasicSbor)]
#[sbor(transparent)]
pub struct VotePower(pub u64);

impl VotePower {
    /// Minimum vote power.
    pub const MIN: Self = VotePower(1);

    /// Create from u64, ensuring it's at least 1.
    pub fn new(power: u64) -> Self {
        VotePower(power.max(1))
    }

    /// Get the raw value.
    pub fn get(&self) -> u64 {
        self.0
    }

    /// Calculate total vote power from a list.
    pub fn sum(powers: &[VotePower]) -> u64 {
        powers.iter().map(|p| p.0).sum()
    }

    /// Calculate if we have 2f+1 quorum (>2/3 of total).
    pub fn has_quorum(voted: u64, total: u64) -> bool {
        voted * 3 > total * 2
    }
}

impl fmt::Display for VotePower {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Node identifier (30-byte address).
///
/// This is a simplified version that doesn't depend on Radix types.
/// It represents an address in the state tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, BasicSbor)]
pub struct NodeId(pub [u8; 30]);

impl NodeId {
    /// Create a NodeId from bytes.
    ///
    /// # Panics
    ///
    /// Panics if bytes length is not exactly 30.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 30, "NodeId must be exactly 30 bytes");
        let mut arr = [0u8; 30];
        arr.copy_from_slice(bytes);
        Self(arr)
    }

    /// Get the bytes as a slice.
    pub fn as_bytes(&self) -> &[u8; 30] {
        &self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({}..)", &hex::encode(&self.0[..4]))
    }
}

/// Partition number within a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, BasicSbor)]
#[sbor(transparent)]
pub struct PartitionNumber(pub u8);

impl PartitionNumber {
    /// Create a new partition number.
    pub fn new(n: u8) -> Self {
        Self(n)
    }
}

impl fmt::Display for PartitionNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Partition({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_height_next_prev() {
        let height = BlockHeight(10);
        assert_eq!(height.next(), BlockHeight(11));
        assert_eq!(height.prev(), Some(BlockHeight(9)));

        assert_eq!(BlockHeight::GENESIS.prev(), None);
        assert_eq!(BlockHeight::GENESIS.next(), BlockHeight(1));
    }

    #[test]
    fn test_vote_power_quorum() {
        let total = 4;

        assert!(!VotePower::has_quorum(2, total)); // 2/4 = 50% (not enough)
        assert!(VotePower::has_quorum(3, total)); // 3/4 = 75% (quorum!)
        assert!(VotePower::has_quorum(4, total)); // 4/4 = 100% (quorum!)
    }

    #[test]
    fn test_node_id() {
        let bytes = [42u8; 30];
        let node_id = NodeId(bytes);
        assert_eq!(node_id.as_bytes(), &bytes);
    }
}
