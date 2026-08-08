//! Provision fetch request for fallback recovery.

use hyperscale_hbor::Hbor;

use crate::network::response::GetProvisionResponse;
use crate::{BlockHeight, MessageClass, NetworkMessage, Request, ShardId};

/// Request to fetch missing provisions from a source shard.
///
/// Sent by target shards when a remote block's `ticks` field indicates
/// the target shard but no provisions arrived within the timeout window.
/// This is the fallback recovery mechanism for byzantine proposers that
/// silently drop provisions.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct GetProvisionsRequest {
    /// Height of the source block whose provisions are needed.
    pub block_height: BlockHeight,
    /// The shard requesting provisions (so the source knows which
    /// state entries to include in the response).
    pub target_shard: ShardId,
}

impl NetworkMessage for GetProvisionsRequest {
    fn message_type_id() -> &'static str {
        "provision.request"
    }

    fn class() -> MessageClass {
        MessageClass::CrossShardProgress
    }
}

impl Request for GetProvisionsRequest {
    type Response = GetProvisionResponse;

    fn is_empty_response(response: &Self::Response) -> bool {
        response.provisions.is_none()
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};

    use super::*;

    #[test]
    fn test_hbor_roundtrip() {
        let request = GetProvisionsRequest {
            block_height: BlockHeight::new(42),
            target_shard: ShardId::ROOT,
        };

        let encoded = hbor_to_vec(&request).unwrap();
        let decoded: GetProvisionsRequest = hbor_from_slice(&encoded).unwrap();
        assert_eq!(request, decoded);
    }
}
