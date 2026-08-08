//! Finalization fetch request (intra-shard DA).

use hyperscale_hbor::Hbor;

use crate::network::response::GetFinalizationsResponse;
use crate::{MessageClass, NetworkMessage, Request, TickId};

/// Request to fetch finalizations by id.
///
/// Used when a validator is missing finalizations referenced by a pending
/// block. The responder resolves each id from the local finalization cache
/// (and falls through to storage where supported) — no scope information is
/// needed since `TickId` self-contains shard, height, and dependency set.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct GetFinalizationsRequest {
    /// Wave IDs being requested.
    pub tick_ids: Vec<TickId>,
}

impl GetFinalizationsRequest {
    /// Build a request for the listed `tick_ids`.
    #[must_use]
    pub const fn new(tick_ids: Vec<TickId>) -> Self {
        Self { tick_ids }
    }
}

impl NetworkMessage for GetFinalizationsRequest {
    fn message_type_id() -> &'static str {
        "finalization.request"
    }

    fn class() -> MessageClass {
        MessageClass::BlockCompletion
    }
}

impl Request for GetFinalizationsRequest {
    type Response = GetFinalizationsResponse;

    fn is_empty_response(response: &Self::Response) -> bool {
        response.finalizations.is_empty()
    }
}

#[cfg(test)]
mod tests {

    use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};

    use super::*;
    use crate::{BlockHeight, ShardId};

    #[test]
    fn test_hbor_roundtrip() {
        let request = GetFinalizationsRequest {
            tick_ids: vec![
                TickId::new(ShardId::ROOT, BlockHeight::new(1)),
                TickId::new(ShardId::ROOT, BlockHeight::new(2)),
            ],
        };
        let encoded = hbor_to_vec(&request).unwrap();
        let decoded: GetFinalizationsRequest = hbor_from_slice(&encoded).unwrap();
        assert_eq!(request, decoded);
    }
}
