//! Request for a state proof of named keys at a committed height.
//!
//! The requester already holds the commit-proven header for the height
//! and checks the answer against its state root, so the request names
//! nothing but the height and the keys: a peer on another branch at
//! that height serves a proof that fails to reconstruct the root and is
//! rotated off. The answer is a multiproof covering every key asked,
//! inclusion and non-inclusion alike — see
//! [`GetStateProofResponse`](crate::network::response::GetStateProofResponse).

use hyperscale_hbor::Hbor;

use crate::network::response::GetStateProofResponse;
use crate::{
    BlockHeight, MAX_COMMITTED_TX_QUERY, MessageClass, NetworkMessage, Request, SubstateKey,
};

/// The keys to prove and the committed height to prove them at.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct GetStateProofRequest {
    /// The committed height whose state root the proof reconstructs.
    pub height: BlockHeight,
    /// The keys to prove, present or absent. Bounded by the
    /// committed-transaction query cap: one probe asks about one
    /// transaction's committed cell, so the two questions share a size.
    #[hbor(max = MAX_COMMITTED_TX_QUERY)]
    pub keys: Vec<SubstateKey>,
}

impl GetStateProofRequest {
    /// A request for `keys` at `height`.
    #[must_use]
    pub const fn new(height: BlockHeight, keys: Vec<SubstateKey>) -> Self {
        Self { height, keys }
    }
}

impl NetworkMessage for GetStateProofRequest {
    fn message_type_id() -> &'static str {
        "state_proof.request"
    }

    fn class() -> MessageClass {
        MessageClass::Bulk
    }
}

impl Request for GetStateProofRequest {
    type Response = GetStateProofResponse;

    fn is_empty_response(response: &Self::Response) -> bool {
        response.proof.is_none()
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};

    use super::*;
    use crate::test_utils::test_key;

    #[test]
    fn test_hbor_roundtrip() {
        let request = GetStateProofRequest::new(BlockHeight::new(42), vec![test_key(7)]);
        let encoded = hbor_to_vec(&request).unwrap();
        let decoded: GetStateProofRequest = hbor_from_slice(&encoded).unwrap();
        assert_eq!(request, decoded);
    }
}
