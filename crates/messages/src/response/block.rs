//! Block fetch response.

use hyperscale_types::{Block, NetworkMessage};
use sbor::prelude::BasicSbor;

/// Response to a block fetch request containing the full Block (or None if not found).
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct GetBlockResponse {
    /// The requested block (None if not found)
    pub block: Option<Block>,
}

impl GetBlockResponse {
    /// Create a response with a found block.
    pub fn found(block: Block) -> Self {
        Self { block: Some(block) }
    }

    /// Create a response for a block not found.
    pub fn not_found() -> Self {
        Self { block: None }
    }

    /// Check if the block was found.
    pub fn has_block(&self) -> bool {
        self.block.is_some()
    }

    /// Get the block if present.
    pub fn block(&self) -> Option<&Block> {
        self.block.as_ref()
    }

    /// Consume and return the block if present.
    pub fn into_block(self) -> Option<Block> {
        self.block
    }
}

// Network message implementation
impl NetworkMessage for GetBlockResponse {
    fn message_type_id() -> &'static str {
        "block.response"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_types::{
        test_utils::test_transaction, BlockHeader, BlockHeight, Hash, QuorumCertificate,
        ValidatorId,
    };

    fn create_test_block() -> Block {
        let tx = test_transaction(1);

        Block {
            header: BlockHeader {
                height: BlockHeight(1),
                parent_hash: Hash::from_bytes(b"parent"),
                parent_qc: QuorumCertificate::genesis(),
                proposer: ValidatorId(0),
                timestamp: 1234567890,
                round: 0,
                is_fallback: false,
            },
            transactions: vec![tx],
            committed_certificates: vec![],
            deferred: vec![],
            aborted: vec![],
        }
    }

    #[test]
    fn test_block_response_found() {
        let block = create_test_block();
        let response = GetBlockResponse::found(block.clone());

        assert!(response.has_block());
        assert_eq!(response.block(), Some(&block));
    }

    #[test]
    fn test_block_response_not_found() {
        let response = GetBlockResponse::not_found();

        assert!(!response.has_block());
        assert_eq!(response.block(), None);
    }

    #[test]
    fn test_block_response_into_block() {
        let block = create_test_block();
        let response = GetBlockResponse::found(block.clone());

        let extracted = response.into_block();
        assert_eq!(extracted, Some(block));
    }
}
