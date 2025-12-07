//! Block fetch request.

use crate::response::GetBlockResponse;
use hyperscale_types::{Hash, NetworkMessage, Request};
use sbor::prelude::BasicSbor;

/// Request to fetch a full Block by hash during sync or catch-up.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct GetBlockRequest {
    /// Hash of the block being requested
    pub block_hash: Hash,
}

impl GetBlockRequest {
    /// Create a new block fetch request.
    pub fn new(block_hash: Hash) -> Self {
        Self { block_hash }
    }
}

// Network message implementation
impl NetworkMessage for GetBlockRequest {
    fn message_type_id() -> &'static str {
        "block.request"
    }
}

/// Type-safe request/response pairing.
/// GetBlockRequest expects GetBlockResponse.
impl Request for GetBlockRequest {
    type Response = GetBlockResponse;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_block_request() {
        let hash = Hash::from_bytes(b"test_block");
        let request = GetBlockRequest::new(hash);
        assert_eq!(request.block_hash, hash);
    }
}
