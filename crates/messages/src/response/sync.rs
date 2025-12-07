//! Sync-related response messages.

use hyperscale_types::{BlockHeight, Hash, NetworkMessage};
use sbor::prelude::BasicSbor;

/// Response listing available block hashes starting from `starting_height`.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct GetBlockInventoryResponse {
    /// Hashes of blocks this node has available, starting from `starting_height`
    pub available_blocks: Vec<Hash>,

    /// The height of the first block in `available_blocks`
    pub starting_height: BlockHeight,

    /// Highest block height this node has reached
    pub highest_height: BlockHeight,
}

impl GetBlockInventoryResponse {
    /// Create a new block inventory response.
    ///
    /// # Arguments
    /// * `available_blocks` - Block hashes starting from `starting_height`
    /// * `starting_height` - The height of the first block in the list
    /// * `highest_height` - The highest block height this node has reached
    pub fn new(
        available_blocks: Vec<Hash>,
        starting_height: BlockHeight,
        highest_height: BlockHeight,
    ) -> Self {
        Self {
            available_blocks,
            starting_height,
            highest_height,
        }
    }

    /// Get the block hash at a specific height, if available.
    pub fn hash_at_height(&self, height: BlockHeight) -> Option<Hash> {
        if height.0 < self.starting_height.0 {
            return None;
        }
        let index = (height.0 - self.starting_height.0) as usize;
        self.available_blocks.get(index).copied()
    }
}

// Network message implementation
impl NetworkMessage for GetBlockInventoryResponse {
    fn message_type_id() -> &'static str {
        "block.inventory.response"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_block_inventory_response() {
        let blocks = vec![Hash::from_bytes(b"block1"), Hash::from_bytes(b"block2")];
        let response =
            GetBlockInventoryResponse::new(blocks.clone(), BlockHeight(49), BlockHeight(50));

        assert_eq!(response.available_blocks, blocks);
        assert_eq!(response.starting_height, BlockHeight(49));
        assert_eq!(response.highest_height, BlockHeight(50));
    }

    #[test]
    fn test_hash_at_height() {
        let block1 = Hash::from_bytes(b"block1");
        let block2 = Hash::from_bytes(b"block2");
        let block3 = Hash::from_bytes(b"block3");
        let blocks = vec![block1, block2, block3];
        let response = GetBlockInventoryResponse::new(blocks, BlockHeight(10), BlockHeight(12));

        // Heights before starting_height should return None
        assert_eq!(response.hash_at_height(BlockHeight(9)), None);

        // Valid heights should return the correct hash
        assert_eq!(response.hash_at_height(BlockHeight(10)), Some(block1));
        assert_eq!(response.hash_at_height(BlockHeight(11)), Some(block2));
        assert_eq!(response.hash_at_height(BlockHeight(12)), Some(block3));

        // Heights beyond available blocks should return None
        assert_eq!(response.hash_at_height(BlockHeight(13)), None);
    }
}
