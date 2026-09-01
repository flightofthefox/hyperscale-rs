//! Response to a state-proof request: one multiproof over every key
//! asked, or nothing when the height is not served here.

use hyperscale_hbor::Hbor;

use crate::{MerkleInclusionProof, MessageClass, NetworkMessage};

/// The proof, or `None` when this peer does not hold the JMT version
/// the height names — never committed here, or pruned past its history.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct GetStateProofResponse {
    /// A multiproof over every requested key against the height's root.
    pub proof: Option<MerkleInclusionProof>,
}

impl GetStateProofResponse {
    /// A served proof.
    #[must_use]
    pub const fn found(proof: MerkleInclusionProof) -> Self {
        Self { proof: Some(proof) }
    }

    /// The height is not served here.
    #[must_use]
    pub const fn not_found() -> Self {
        Self { proof: None }
    }
}

impl NetworkMessage for GetStateProofResponse {
    fn message_type_id() -> &'static str {
        "state_proof.response"
    }

    fn class() -> MessageClass {
        MessageClass::Bulk
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};

    use super::*;

    #[test]
    fn test_hbor_roundtrip() {
        for response in [
            GetStateProofResponse::not_found(),
            GetStateProofResponse::found(MerkleInclusionProof::new(vec![1, 2, 3])),
        ] {
            let encoded = hbor_to_vec(&response).unwrap();
            let decoded: GetStateProofResponse = hbor_from_slice(&encoded).unwrap();
            assert_eq!(response, decoded);
        }
    }
}
