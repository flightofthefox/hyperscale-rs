//! [`Verifier`] over BLS12-381 (min-pk) signatures.

use blst::min_pk::{PublicKey as BlstPublicKey, Signature as BlstSignature};
use blst::{BLST_ERROR, blst_scalar, blst_scalar_from_bendian};
use hyperscale_crypto::{
    AggregateError, AggregateSignature, ConsensusPublicKey, ConsensusSignature, Verifier, VrfProof,
};
use radix_common::crypto::{
    BLS12381_CIPHERSITE_V1, Bls12381G1PublicKey, Bls12381G2Signature, aggregate_verify_bls12381_v1,
    verify_bls12381_v1,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, rng};

/// BLS verification.
///
/// Aggregates are G2 sums, same-message aggregate checks run one
/// pairing against the aggregated pubkey, and batches use blst's
/// random-linear-combination fast path with individual fallback. The
/// random scalars affect performance only, never outcomes.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlsVerifier;

const fn pk(key: &ConsensusPublicKey) -> Bls12381G1PublicKey {
    Bls12381G1PublicKey(*key.as_bytes())
}

const fn sig(s: &ConsensusSignature) -> Bls12381G2Signature {
    Bls12381G2Signature(*s.as_bytes())
}

const fn agg(a: &AggregateSignature) -> Bls12381G2Signature {
    Bls12381G2Signature(*a.as_bytes())
}

/// All-or-nothing batch verification over distinct messages via blst's
/// random linear combination (~2 pairings for the whole batch).
fn batch_all_or_nothing(
    messages: &[&[u8]],
    signatures: &[Bls12381G2Signature],
    pubkeys: &[Bls12381G1PublicKey],
) -> bool {
    let mut bls_sigs = Vec::with_capacity(signatures.len());
    let mut bls_pks = Vec::with_capacity(pubkeys.len());
    for (s, p) in signatures.iter().zip(pubkeys.iter()) {
        let Ok(s) = BlstSignature::from_bytes(&s.0) else {
            return false;
        };
        let Ok(p) = BlstPublicKey::from_bytes(&p.0) else {
            return false;
        };
        bls_sigs.push(s);
        bls_pks.push(p);
    }

    let mut seed = [0u8; 32];
    rng().fill_bytes(&mut seed);
    let mut rng = StdRng::from_seed(seed);
    let mut rands = Vec::with_capacity(signatures.len());
    for _ in 0..signatures.len() {
        let mut rand_bytes = [0u8; 32];
        rng.fill_bytes(&mut rand_bytes);
        let mut scalar = blst_scalar::default();
        // SAFETY: `scalar` is a valid `blst_scalar` (zero-initialised above) and
        // `rand_bytes` is a 32-byte array whose pointer is valid for 32 bytes.
        // `blst_scalar_from_bendian` reads exactly 32 bytes from the pointer.
        unsafe {
            blst_scalar_from_bendian(&raw mut scalar, rand_bytes.as_ptr());
        }
        rands.push(scalar);
    }

    let sig_refs: Vec<&BlstSignature> = bls_sigs.iter().collect();
    let pk_refs: Vec<&BlstPublicKey> = bls_pks.iter().collect();

    let result = BlstSignature::verify_multiple_aggregate_signatures(
        messages,
        BLS12381_CIPHERSITE_V1, // DST must match sign_v1/verify_bls12381_v1
        &pk_refs,
        false, // pks_validate - possession-proven or genesis-trusted
        &sig_refs,
        true, // sigs_groupcheck - verify signatures are in the group
        &rands,
        64, // rand_bits - 64 bits of randomness
    );

    result == BLST_ERROR::BLST_SUCCESS
}

/// Same-message batch fast path: aggregate signatures and pubkeys, then
/// run a single pairing check.
fn batch_same_message(
    message: &[u8],
    signatures: &[Bls12381G2Signature],
    pubkeys: &[Bls12381G1PublicKey],
) -> bool {
    let Ok(agg_sig) = Bls12381G2Signature::aggregate(signatures, true) else {
        return false;
    };
    // Pubkey aggregation skips G1 subgroup validation: every topology key
    // is possession-proven at registration (or genesis-trusted), which
    // both guarantees real G1 points and forecloses rogue-key
    // constructions.
    let Ok(agg_pk) = Bls12381G1PublicKey::aggregate(pubkeys, false) else {
        return false;
    };
    verify_bls12381_v1(message, &agg_pk, &agg_sig)
}

impl Verifier for BlsVerifier {
    fn verify(&self, key: &ConsensusPublicKey, message: &[u8], s: &ConsensusSignature) -> bool {
        verify_bls12381_v1(message, &pk(key), &sig(s))
    }

    fn aggregate(&self, sigs: &[ConsensusSignature]) -> Result<AggregateSignature, AggregateError> {
        if sigs.is_empty() {
            return Err(AggregateError::Empty);
        }
        let bls: Vec<Bls12381G2Signature> = sigs.iter().map(sig).collect();
        Bls12381G2Signature::aggregate(&bls, true)
            .map(|a| AggregateSignature::new(a.0))
            .map_err(|_| AggregateError::InvalidSignature)
    }

    fn verify_aggregate_same_message(
        &self,
        message: &[u8],
        aggregate: &AggregateSignature,
        keys: &[ConsensusPublicKey],
    ) -> bool {
        if keys.is_empty() {
            return false;
        }
        let pks: Vec<Bls12381G1PublicKey> = keys.iter().map(pk).collect();
        // Unvalidated aggregation: see `batch_same_message`.
        let Ok(agg_pk) = Bls12381G1PublicKey::aggregate(&pks, false) else {
            return false;
        };
        verify_bls12381_v1(message, &agg_pk, &agg(aggregate))
    }

    fn verify_aggregate_different_messages(
        &self,
        messages: &[&[u8]],
        aggregate: &AggregateSignature,
        keys: &[ConsensusPublicKey],
    ) -> bool {
        if messages.len() != keys.len() || messages.is_empty() {
            return false;
        }
        let pairs: Vec<(Bls12381G1PublicKey, Vec<u8>)> = keys
            .iter()
            .zip(messages.iter())
            .map(|(k, m)| (pk(k), m.to_vec()))
            .collect();
        aggregate_verify_bls12381_v1(&pairs, &agg(aggregate))
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
        if messages.is_empty() {
            return vec![];
        }
        let bls_sigs: Vec<Bls12381G2Signature> = sigs.iter().map(sig).collect();
        let bls_pks: Vec<Bls12381G1PublicKey> = keys.iter().map(pk).collect();

        // Fast path; blst's different-messages combination requires
        // distinct messages, so uniform batches take the aggregate route.
        let all_same = messages.windows(2).all(|w| w[0] == w[1]);
        let batch_ok = if all_same {
            batch_same_message(messages[0], &bls_sigs, &bls_pks)
        } else {
            batch_all_or_nothing(messages, &bls_sigs, &bls_pks)
        };
        if batch_ok {
            return vec![true; sigs.len()];
        }

        // Slow path: batch failed, verify individually to find failures
        messages
            .iter()
            .zip(bls_sigs.iter())
            .zip(bls_pks.iter())
            .map(|((m, s), p)| verify_bls12381_v1(m, p, s))
            .collect()
    }

    fn verify_vrf(&self, key: &ConsensusPublicKey, message: &[u8], proof: &VrfProof) -> bool {
        verify_bls12381_v1(message, &pk(key), &Bls12381G2Signature(*proof.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_crypto::run_conformance_suite;

    use super::*;
    use crate::BlsSigner;

    #[test]
    fn conformance() {
        run_conformance_suite(
            |seed_index| {
                let mut seed = [0u8; 32];
                seed[0] = seed_index;
                BlsSigner::from_seed(&seed)
            },
            &BlsVerifier,
        );
    }

    #[test]
    fn batch_verify_same_message_batches_take_the_aggregate_path() {
        use hyperscale_crypto::Signer;
        let signers: Vec<BlsSigner> = (0..3u8)
            .map(|i| BlsSigner::from_seed(&[i + 1; 32]))
            .collect();
        let keys: Vec<_> = signers.iter().map(Signer::public_key).collect();
        let message = b"same message for everyone".as_slice();
        let sigs: Vec<_> = signers
            .iter()
            .map(|s| s.sign(message).expect("bls sign cannot fail"))
            .collect();
        let messages = vec![message; 3];
        assert_eq!(
            BlsVerifier.batch_verify(&messages, &sigs, &keys),
            vec![true; 3]
        );

        // One forged entry must be singled out by the fallback.
        let mut bad = sigs;
        bad[1] = signers[1]
            .sign(b"other message")
            .expect("bls sign cannot fail");
        assert_eq!(
            BlsVerifier.batch_verify(&messages, &bad, &keys),
            vec![true, false, true]
        );
    }
}
