//! Shard-witness fetch response.

use sbor::prelude::BasicSbor;

use crate::{
    BoundedVec, Hash, MAX_RANGE_PROOF_NODES, MAX_WITNESSES_PER_SHARD, MessageClass, NetworkMessage,
    ShardWitnessPayload,
};

/// Response to a
/// [`GetShardWitnessesRequest`](crate::network::request::beacon::GetShardWitnessesRequest).
///
/// Carries a contiguous run of witness payloads starting at the request's
/// `lo`, with one range proof lifting them to the requested block's
/// [`BeaconWitnessRoot`](crate::BeaconWitnessRoot). Leaf positions are
/// implied by `lo` and the payload order, so the requester verifies
/// against the window it already resolved from the anchor header.
///
/// The run is served whole: the requester admits a chunk only when it
/// covers exactly the range the fold will apply, so a prefix would be
/// dropped on arrival. A responder that can't cover the request clamps to
/// what its window holds, and the requester re-requests against a later
/// anchor. Empty when the responder can serve nothing at the named
/// committed block, in which case the requester falls through to another
/// peer in the shard's committee.
///
/// The run's width is bounded by the fold's per-epoch budget
/// ([`MAX_WITNESSES_PER_SHARD`]) — the same bound the beacon block itself
/// carries, so a servable chunk always fits in the block that commits it.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct GetShardWitnessesResponse {
    /// Witness payloads in leaf-index order, starting at the request's
    /// `lo`.
    pub payloads: BoundedVec<ShardWitnessPayload, MAX_WITNESSES_PER_SHARD>,
    /// Flanking merkle nodes lifting `payloads` to the anchor block's
    /// beacon-witness root.
    pub range_proof: BoundedVec<Hash, MAX_RANGE_PROOF_NODES>,
}

impl GetShardWitnessesResponse {
    /// Build a response from a payload run and its range proof.
    ///
    /// # Panics
    ///
    /// Panics if `payloads.len() > MAX_WITNESSES_PER_SHARD` or
    /// `range_proof.len() > MAX_RANGE_PROOF_NODES`.
    #[must_use]
    pub fn new(payloads: Vec<ShardWitnessPayload>, range_proof: Vec<Hash>) -> Self {
        Self {
            payloads: payloads.into(),
            range_proof: range_proof.into(),
        }
    }

    /// Empty response — responder can serve nothing at the named block.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            payloads: Vec::new().into(),
            range_proof: Vec::new().into(),
        }
    }
}

impl NetworkMessage for GetShardWitnessesResponse {
    fn message_type_id() -> &'static str {
        "beacon.shard_witnesses.response"
    }

    fn class() -> MessageClass {
        MessageClass::CrossShardProgress
    }
}

#[cfg(test)]
mod tests {
    use sbor::{basic_decode, basic_encode};

    use super::*;
    use crate::{Stake, StakePoolId};

    fn sample_payload(pool: u32) -> ShardWitnessPayload {
        ShardWitnessPayload::StakeDeposit {
            pool_id: StakePoolId::new(pool),
            amount: Stake::from_whole_tokens(1_000),
        }
    }

    #[test]
    fn sbor_round_trip_populated() {
        let resp = GetShardWitnessesResponse::new(
            vec![sample_payload(1), sample_payload(2), sample_payload(42)],
            vec![Hash::from_bytes(b"flank0"), Hash::from_bytes(b"flank1")],
        );
        let bytes = basic_encode(&resp).unwrap();
        let decoded: GetShardWitnessesResponse = basic_decode(&bytes).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn sbor_round_trip_empty() {
        let resp = GetShardWitnessesResponse::empty();
        let bytes = basic_encode(&resp).unwrap();
        let decoded: GetShardWitnessesResponse = basic_decode(&bytes).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn class_is_cross_shard_progress() {
        assert_eq!(
            GetShardWitnessesResponse::class(),
            MessageClass::CrossShardProgress
        );
    }
}
