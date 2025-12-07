//! Sync-related request messages.

use crate::response::GetBlockInventoryResponse;
use hyperscale_types::{
    BlockHeight, NetworkMessage, Request, ShardMessage, Signature, ValidatorId,
};
use sbor::prelude::BasicSbor;

/// Request for block inventory from a peer starting at a given height.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct GetBlockInventoryRequest {
    /// Validator requesting the inventory
    pub requester: ValidatorId,

    /// Starting block height for the inventory request.
    ///
    /// The peer should return blocks from this height onwards.
    /// This allows efficient pagination during sync.
    pub from_height: BlockHeight,
}

impl GetBlockInventoryRequest {
    /// Create a new block inventory request.
    pub fn new(requester: ValidatorId, from_height: BlockHeight) -> Self {
        Self {
            requester,
            from_height,
        }
    }
}

// Network message implementation
impl NetworkMessage for GetBlockInventoryRequest {
    fn message_type_id() -> &'static str {
        "block.inventory.request"
    }
}

/// Type-safe request/response pairing.
/// GetBlockInventoryRequest expects GetBlockInventoryResponse.
impl Request for GetBlockInventoryRequest {
    type Response = GetBlockInventoryResponse;
}

/// Broadcast that validator has caught up to network head and is ready to participate.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct SyncCompleteAnnouncement {
    /// Height synced to
    pub synced_height: BlockHeight,

    /// Validator announcing sync completion
    pub validator: ValidatorId,

    /// Signature proving this is authentic
    pub signature: Signature,
}

impl SyncCompleteAnnouncement {
    /// Create a new sync complete announcement.
    pub fn new(synced_height: BlockHeight, validator: ValidatorId, signature: Signature) -> Self {
        Self {
            synced_height,
            validator,
            signature,
        }
    }
}

// Network message implementation
impl NetworkMessage for SyncCompleteAnnouncement {
    fn message_type_id() -> &'static str {
        "sync.complete"
    }
}

impl ShardMessage for SyncCompleteAnnouncement {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_block_inventory_request() {
        let request = GetBlockInventoryRequest::new(ValidatorId(0), BlockHeight(100));
        assert_eq!(request.requester, ValidatorId(0));
        assert_eq!(request.from_height, BlockHeight(100));
    }

    #[test]
    fn test_sync_complete_announcement() {
        use hyperscale_types::Signature;
        let announcement =
            SyncCompleteAnnouncement::new(BlockHeight(100), ValidatorId(1), Signature::zero());
        assert_eq!(announcement.synced_height, BlockHeight(100));
        assert_eq!(announcement.validator, ValidatorId(1));
    }
}
