//! [`Verifier`] over the keyed-hash mock scheme.

use blake3::Hasher;
use hyperscale_crypto::{
    AggregateError, AggregateSignature, ConsensusPublicKey, ConsensusSignature, Verifier,
    VrfOutput, VrfProof,
};

use crate::derive;

/// Domain tag for hashing a [`VrfProof`] into its [`VrfOutput`].
/// Byte-identical to the BLS impl's binding (and to the canonical
/// `vrf_output_from_proof` in `types`), so mixed call sites derive the
/// same output regardless of which path computes it.
const DOMAIN_VRF_OUTPUT: &[u8] = b"HYPERSCALE_VRF_OUTPUT_v1";

/// Mock verification: every check recomputes the expected signature
/// from the public key and compares bytes. Aggregate checks refold the
/// recomputed per-signer signatures in the caller's key order.
#[derive(Debug, Clone, Copy, Default)]
pub struct MockVerifier;

impl Verifier for MockVerifier {
    fn verify(&self, key: &ConsensusPublicKey, message: &[u8], sig: &ConsensusSignature) -> bool {
        derive::signature(key, message) == *sig
    }

    fn aggregate(&self, sigs: &[ConsensusSignature]) -> Result<AggregateSignature, AggregateError> {
        if sigs.is_empty() {
            return Err(AggregateError::Empty);
        }
        // No scheme-level validity to check without the (key, message)
        // pair — a bad input signature surfaces at aggregate verify.
        Ok(derive::aggregate(sigs))
    }

    fn verify_aggregate_same_message(
        &self,
        message: &[u8],
        agg: &AggregateSignature,
        keys: &[ConsensusPublicKey],
    ) -> bool {
        if keys.is_empty() {
            return false;
        }
        let sigs: Vec<ConsensusSignature> =
            keys.iter().map(|k| derive::signature(k, message)).collect();
        derive::aggregate(&sigs) == *agg
    }

    fn verify_aggregate_different_messages(
        &self,
        messages: &[&[u8]],
        agg: &AggregateSignature,
        keys: &[ConsensusPublicKey],
    ) -> bool {
        if messages.len() != keys.len() || messages.is_empty() {
            return false;
        }
        let sigs: Vec<ConsensusSignature> = keys
            .iter()
            .zip(messages)
            .map(|(k, m)| derive::signature(k, m))
            .collect();
        derive::aggregate(&sigs) == *agg
    }

    fn batch_verify(
        &self,
        messages: &[&[u8]],
        sigs: &[ConsensusSignature],
        keys: &[ConsensusPublicKey],
    ) -> Vec<bool> {
        if messages.len() != sigs.len() || sigs.len() != keys.len() {
            return vec![false; messages.len().max(sigs.len()).max(keys.len())];
        }
        messages
            .iter()
            .zip(sigs)
            .zip(keys)
            .map(|((m, s), k)| self.verify(k, m, s))
            .collect()
    }

    fn verify_vrf(&self, key: &ConsensusPublicKey, message: &[u8], proof: &VrfProof) -> bool {
        *derive::signature(key, message).as_bytes() == *proof.as_bytes()
    }

    fn vrf_output(&self, proof: &VrfProof) -> VrfOutput {
        let mut h = Hasher::new();
        h.update(DOMAIN_VRF_OUTPUT);
        h.update(proof.as_bytes());
        VrfOutput::new(*h.finalize().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_crypto::{Signer, run_conformance_suite};

    use super::*;
    use crate::MockSigner;

    fn signer(seed_index: u8) -> MockSigner {
        let mut seed = [0u8; 32];
        seed[0] = seed_index;
        MockSigner::from_seed(&seed)
    }

    #[test]
    fn conformance() {
        run_conformance_suite(signer, &MockVerifier);
    }

    /// Ordered fold: permuting the aggregate's inputs (or the key order
    /// at verify) breaks it. Mock-stricter-than-BLS by design — this
    /// enforces the committee-index canonicalization convention, so it
    /// is pinned here rather than in the shared battery (BLS cannot
    /// have it).
    #[test]
    fn aggregate_is_order_sensitive() {
        let signers: Vec<MockSigner> = (0..3).map(signer).collect();
        let keys: Vec<_> = signers.iter().map(Signer::public_key).collect();
        let message = b"order test";
        let sigs: Vec<_> = signers
            .iter()
            .map(|s| s.sign(message).expect("mock sign cannot fail"))
            .collect();

        let agg = MockVerifier.aggregate(&sigs).expect("non-empty");
        assert!(MockVerifier.verify_aggregate_same_message(message, &agg, &keys));

        let mut swapped = keys;
        swapped.swap(0, 1);
        assert!(
            !MockVerifier.verify_aggregate_same_message(message, &agg, &swapped),
            "permuted key order must refold to a different aggregate"
        );

        let mut permuted = sigs;
        permuted.swap(1, 2);
        let permuted_agg = MockVerifier.aggregate(&permuted).expect("non-empty");
        assert_ne!(
            agg, permuted_agg,
            "permuted input signatures must fold to a different aggregate"
        );
    }

    /// The proof-to-output binding matches the canonical digest used by
    /// the BLS impl and `types::vrf_output_from_proof`, so mixed call
    /// sites agree on the output bytes.
    #[test]
    fn vrf_output_matches_canonical_binding() {
        let proof = signer(1).vrf_sign(b"vrf").expect("mock sign cannot fail");
        let mut h = Hasher::new();
        h.update(b"HYPERSCALE_VRF_OUTPUT_v1");
        h.update(proof.as_bytes());
        assert_eq!(
            MockVerifier.vrf_output(&proof),
            VrfOutput::new(*h.finalize().as_bytes())
        );
    }
}
