//! Scheme conformance battery.
//!
//! Every scheme impl crate runs [`run_conformance_suite`] against its
//! `Signer`/`Verifier` pair. The battery is the guard against an
//! accept-everything scheme silently defanging verification in the test
//! suites that run on it: it pins the acceptance *and* rejection
//! behavior every caller of the traits relies on.
//!
//! Deliberately absent: any assertion about aggregate input *order*.
//! Some schemes' aggregates are order-insensitive (a commutative
//! combine), others' are not; callers canonicalize to committee-index
//! order and the battery only exercises that canonical path.

use crate::{AggregateError, ConsensusSignature, Signer, Verifier};

/// Run the full battery against one scheme.
///
/// `mk_signer` builds a deterministic signer from a seed index; distinct
/// indices must yield distinct keys.
///
/// # Panics
///
/// Panics (via `assert!`) on the first conformance violation.
#[allow(clippy::too_many_lines)] // one linear battery; splitting would scatter the contract
pub fn run_conformance_suite<S, F>(mk_signer: F, verifier: &dyn Verifier)
where
    S: Signer,
    F: Fn(u8) -> S,
{
    let signers: Vec<S> = (0..4).map(&mk_signer).collect();
    let keys: Vec<_> = signers.iter().map(Signer::public_key).collect();

    assert_ne!(keys[0], keys[1], "distinct seeds must yield distinct keys");
    assert_eq!(
        mk_signer(0).public_key(),
        keys[0],
        "signer construction must be deterministic in the seed"
    );

    let message = b"conformance: canonical message".as_slice();
    let other_message = b"conformance: a different message".as_slice();

    // Single-signature accept and reject axes.
    let sig = signers[0].sign(message).expect("sign must succeed");
    assert!(
        verifier.verify(&keys[0], message, &sig),
        "valid signature must verify"
    );
    assert!(
        !verifier.verify(&keys[0], other_message, &sig),
        "wrong message must reject"
    );
    assert!(
        !verifier.verify(&keys[1], message, &sig),
        "wrong key must reject"
    );
    let mut tampered = *sig.as_bytes();
    tampered[17] ^= 0x01;
    assert!(
        !verifier.verify(&keys[0], message, &ConsensusSignature::new(tampered)),
        "tampered signature must reject"
    );

    // Same-message aggregation: accept, then shrink/swap the signer set.
    let sigs: Vec<_> = signers
        .iter()
        .map(|s| s.sign(message).expect("sign must succeed"))
        .collect();
    let agg = verifier
        .aggregate(&sigs)
        .expect("aggregation of valid signatures must succeed");
    assert!(
        verifier.verify_aggregate_same_message(message, &agg, &keys),
        "aggregate over the full signer set must verify"
    );
    assert!(
        !verifier.verify_aggregate_same_message(other_message, &agg, &keys),
        "aggregate against the wrong message must reject"
    );
    assert!(
        !verifier.verify_aggregate_same_message(message, &agg, &keys[..3]),
        "aggregate against a subset of the signer set must reject"
    );
    let mut swapped_set = keys.clone();
    swapped_set[3] = mk_signer(9).public_key();
    assert!(
        !verifier.verify_aggregate_same_message(message, &agg, &swapped_set),
        "aggregate against a swapped-in non-signer must reject"
    );
    let sub_agg = verifier
        .aggregate(&sigs[..3])
        .expect("aggregation of valid signatures must succeed");
    assert!(
        !verifier.verify_aggregate_same_message(message, &sub_agg, &keys),
        "aggregate missing a signature must reject against the full set"
    );
    assert!(
        matches!(verifier.aggregate(&[]), Err(AggregateError::Empty)),
        "empty aggregation input must be rejected"
    );
    assert!(
        !verifier.verify_aggregate_same_message(message, &agg, &[]),
        "aggregate against an empty key set must reject"
    );

    // Different-messages aggregation: each signer binds to its own
    // message; permuting the key/message pairing must break the binding.
    let messages: Vec<Vec<u8>> = (0..signers.len())
        .map(|i| format!("conformance: per-signer message {i}").into_bytes())
        .collect();
    let message_refs: Vec<&[u8]> = messages.iter().map(Vec::as_slice).collect();
    let dm_sigs: Vec<_> = signers
        .iter()
        .zip(&message_refs)
        .map(|(s, m)| s.sign(m).expect("sign must succeed"))
        .collect();
    let dm_agg = verifier
        .aggregate(&dm_sigs)
        .expect("aggregation of valid signatures must succeed");
    assert!(
        verifier.verify_aggregate_different_messages(&message_refs, &dm_agg, &keys),
        "different-messages aggregate must verify with the matched pairing"
    );
    let mut swapped_keys = keys.clone();
    swapped_keys.swap(0, 1);
    assert!(
        !verifier.verify_aggregate_different_messages(&message_refs, &dm_agg, &swapped_keys),
        "swapped pubkeys must break the per-signer message binding"
    );
    assert!(
        !verifier.verify_aggregate_different_messages(&message_refs[..3], &dm_agg, &keys),
        "message/key length mismatch must reject"
    );

    // Batch verification: per-item verdicts.
    let all = verifier.batch_verify(&message_refs, &dm_sigs, &keys);
    assert_eq!(
        all,
        vec![true; signers.len()],
        "batch of valid triples must be all-true"
    );
    let mut one_bad = dm_sigs.clone();
    one_bad[2] = signers[2].sign(other_message).expect("sign must succeed");
    let verdicts = verifier.batch_verify(&message_refs, &one_bad, &keys);
    assert!(
        verdicts[0] && verdicts[1] && !verdicts[2] && verdicts[3],
        "batch with one bad triple must single it out, got {verdicts:?}"
    );
    let mismatched = verifier.batch_verify(&message_refs[..2], &dm_sigs, &keys);
    assert!(
        mismatched.iter().all(|ok| !ok),
        "batch length mismatch must be all-false"
    );
    assert!(
        verifier.batch_verify(&[], &[], &[]).is_empty(),
        "empty batch must be empty"
    );

    // VRF: determinism, verification, and output binding.
    let vrf_message = b"conformance: vrf message".as_slice();
    let proof = signers[0]
        .vrf_sign(vrf_message)
        .expect("vrf_sign must succeed");
    assert_eq!(
        signers[0]
            .vrf_sign(vrf_message)
            .expect("vrf_sign must succeed"),
        proof,
        "vrf_sign must be deterministic in (key, message)"
    );
    assert!(
        verifier.verify_vrf(&keys[0], vrf_message, &proof),
        "valid VRF proof must verify"
    );
    assert!(
        !verifier.verify_vrf(&keys[1], vrf_message, &proof),
        "VRF proof under the wrong key must reject"
    );
    assert!(
        !verifier.verify_vrf(&keys[0], message, &proof),
        "VRF proof against the wrong message must reject"
    );
    let other_proof = signers[1]
        .vrf_sign(vrf_message)
        .expect("vrf_sign must succeed");
    assert_ne!(
        proof, other_proof,
        "distinct keys must yield distinct proofs"
    );
}

#[cfg(test)]
mod tests {
    use blake3::{Hasher, keyed_hash};

    use super::*;
    use crate::{AggregateSignature, ConsensusPublicKey, SignError, VrfProof};

    /// Throwaway keyed-hash scheme, exercised only to prove the battery
    /// itself runs and discriminates. Not exported; the real mock lives
    /// in its own impl crate.
    #[derive(Debug)]
    struct TestSigner {
        pk: ConsensusPublicKey,
    }

    fn derive_pk(seed: &[u8; 32]) -> ConsensusPublicKey {
        let digest = keyed_hash(b"conformance-test-scheme-pubkey--", seed);
        let mut pk = [0u8; 48];
        pk[..32].copy_from_slice(digest.as_bytes());
        ConsensusPublicKey::new(pk)
    }

    fn derive_sig(pk: &ConsensusPublicKey, domain: u8, message: &[u8]) -> [u8; 96] {
        let mut hasher = Hasher::new();
        hasher.update(&[domain]);
        hasher.update(pk.as_bytes());
        hasher.update(message);
        let digest = hasher.finalize();
        let mut sig = [0u8; 96];
        sig[..32].copy_from_slice(digest.as_bytes());
        sig
    }

    impl Signer for TestSigner {
        fn public_key(&self) -> ConsensusPublicKey {
            self.pk
        }
        fn sign(&self, message: &[u8]) -> Result<ConsensusSignature, SignError> {
            Ok(ConsensusSignature::new(derive_sig(&self.pk, 0, message)))
        }
        fn vrf_sign(&self, message: &[u8]) -> Result<VrfProof, SignError> {
            Ok(VrfProof::new(derive_sig(&self.pk, 1, message)))
        }
    }

    #[derive(Debug)]
    struct TestVerifier;

    fn fold(sigs: &[ConsensusSignature]) -> AggregateSignature {
        let mut hasher = Hasher::new();
        for sig in sigs {
            hasher.update(sig.as_bytes());
        }
        let mut agg = [0u8; 96];
        agg[..32].copy_from_slice(hasher.finalize().as_bytes());
        AggregateSignature::new(agg)
    }

    impl Verifier for TestVerifier {
        fn verify(
            &self,
            key: &ConsensusPublicKey,
            message: &[u8],
            sig: &ConsensusSignature,
        ) -> bool {
            sig.as_bytes() == &derive_sig(key, 0, message)
        }
        fn aggregate(
            &self,
            sigs: &[ConsensusSignature],
        ) -> Result<AggregateSignature, AggregateError> {
            if sigs.is_empty() {
                return Err(AggregateError::Empty);
            }
            Ok(fold(sigs))
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
            let sigs: Vec<_> = keys
                .iter()
                .map(|k| ConsensusSignature::new(derive_sig(k, 0, message)))
                .collect();
            fold(&sigs) == *agg
        }
        fn verify_aggregate_different_messages(
            &self,
            messages: &[&[u8]],
            agg: &AggregateSignature,
            keys: &[ConsensusPublicKey],
        ) -> bool {
            if keys.is_empty() || messages.len() != keys.len() {
                return false;
            }
            let sigs: Vec<_> = keys
                .iter()
                .zip(messages)
                .map(|(k, m)| ConsensusSignature::new(derive_sig(k, 0, m)))
                .collect();
            fold(&sigs) == *agg
        }
        fn batch_verify(
            &self,
            messages: &[&[u8]],
            sigs: &[ConsensusSignature],
            keys: &[ConsensusPublicKey],
        ) -> Vec<bool> {
            let len = messages.len().max(sigs.len()).max(keys.len());
            if messages.len() != sigs.len() || sigs.len() != keys.len() {
                return vec![false; len];
            }
            messages
                .iter()
                .zip(sigs)
                .zip(keys)
                .map(|((m, s), k)| self.verify(k, m, s))
                .collect()
        }
        fn verify_vrf(&self, key: &ConsensusPublicKey, message: &[u8], proof: &VrfProof) -> bool {
            proof.as_bytes() == &derive_sig(key, 1, message)
        }
    }

    #[test]
    fn battery_passes_on_a_discriminating_scheme() {
        run_conformance_suite(
            |seed_index| {
                let mut seed = [0u8; 32];
                seed[0] = seed_index;
                TestSigner {
                    pk: derive_pk(&seed),
                }
            },
            &TestVerifier,
        );
    }
}
