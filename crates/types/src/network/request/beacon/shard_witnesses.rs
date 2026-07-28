//! Shard-witness fetch request — beacon validator pulls witnesses
//! lifted by a shard whose committee they're not a member of.

use sbor::prelude::BasicSbor;

use crate::network::response::beacon::GetShardWitnessesResponse;
use crate::{BlockHash, BlockHeight, LeafIndex, MessageClass, NetworkMessage, Request, ShardId};

/// Fetch a contiguous run of shard-witness leaves against a specific
/// committed block's accumulator root.
///
/// Served by any validator in `shard_id`'s committee at
/// `(block_height, committed_block_hash)` (the shard's
/// [`CertifiedBlockHeader`](crate::CertifiedBlockHeader) at that height
/// carries the [`BeaconWitnessRoot`](crate::BeaconWitnessRoot) the
/// responder's proof verifies against). The receiver recomputes that root
/// from the returned payloads plus the range proof.
///
/// `block_height` is the height-keyed lookup primary (matching
/// [`GetBlockRequest`](crate::network::request::GetBlockRequest) and
/// [`GetRemoteHeadersRequest`](crate::network::request::GetRemoteHeadersRequest));
/// `committed_block_hash` is the fork-divergence guard so a responder
/// on a different fork returns empty rather than a proof against a
/// silently mismatched root.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct GetShardWitnessesRequest {
    /// Shard whose witnesses are being fetched.
    pub shard_id: ShardId,
    /// Height of the anchor block in `shard_id`'s chain.
    pub block_height: BlockHeight,
    /// Hash of the anchor block. The responder cross-checks this
    /// against the block at `block_height` and returns empty on
    /// mismatch (fork divergence) rather than serving a proof against
    /// the wrong root.
    pub committed_block_hash: BlockHash,
    /// First leaf position requested, in the shard's monotonic
    /// beacon-witness accumulator.
    pub lo: LeafIndex,
    /// End of the requested run, exclusive. A responder may answer with
    /// a shorter contiguous prefix starting at `lo`; the requester
    /// re-requests the tail.
    pub hi: LeafIndex,
}

impl GetShardWitnessesRequest {
    /// Build a request from its parts.
    #[must_use]
    pub const fn new(
        shard_id: ShardId,
        block_height: BlockHeight,
        committed_block_hash: BlockHash,
        lo: LeafIndex,
        hi: LeafIndex,
    ) -> Self {
        Self {
            shard_id,
            block_height,
            committed_block_hash,
            lo,
            hi,
        }
    }
}

impl NetworkMessage for GetShardWitnessesRequest {
    fn message_type_id() -> &'static str {
        "beacon.shard_witnesses.request"
    }

    fn class() -> MessageClass {
        MessageClass::CrossShardProgress
    }
}

impl Request for GetShardWitnessesRequest {
    type Response = GetShardWitnessesResponse;

    fn is_empty_response(response: &Self::Response) -> bool {
        response.payloads.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use sbor::{basic_decode, basic_encode};

    use super::*;

    #[test]
    fn sbor_round_trip() {
        let req = GetShardWitnessesRequest::new(
            ShardId::ROOT,
            BlockHeight::new(42),
            BlockHash::ZERO,
            LeafIndex::new(1),
            LeafIndex::new(42),
        );
        let bytes = basic_encode(&req).unwrap();
        let decoded: GetShardWitnessesRequest = basic_decode(&bytes).unwrap();
        assert_eq!(req, decoded);
    }

    #[test]
    fn class_is_cross_shard_progress() {
        assert_eq!(
            GetShardWitnessesRequest::class(),
            MessageClass::CrossShardProgress
        );
    }
}
