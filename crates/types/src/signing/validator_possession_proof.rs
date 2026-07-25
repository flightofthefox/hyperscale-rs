//! Domain-separated signing for validator BLS proof-of-possession.
//!
//! Every validator key proves possession of its secret at registration:
//! the registrant signs `(network, validator_id, pubkey)` under
//! [`DOMAIN_VALIDATOR_POSSESSION_PROOF`] with the key being registered. The beacon
//! fold verifies the proof before inserting the validator record, so no
//! key enters the registry that its registrant cannot sign for. This is
//! what makes rogue-key constructions (`pk_rogue = g^r · pk_H^{-1}`)
//! unregisterable — producing a valid proof for `pk_rogue` requires its
//! secret `r − x_H`, which the adversary does not know — and it is the
//! precondition the aggregate-signature verifiers rely on when they
//! aggregate topology pubkeys without further validation.
//!
//! Binding `validator_id` and `network` means a captured proof cannot be
//! replayed to register the same key under a different identity or on a
//! different network.

use hyperscale_crypto::{SignError, Signer, Verifier};

use crate::{ConsensusPublicKey, ConsensusSignature, NetworkDefinition, ValidatorId};

/// Domain tag for validator BLS proof-of-possession.
pub const DOMAIN_VALIDATOR_POSSESSION_PROOF: &[u8] = b"HYPERSCALE_VALIDATOR_POSSESSION_PROOF_v1";

/// Build the canonical signing bytes for a proof-of-possession of
/// `pubkey` claimed under `validator_id` on `network`.
///
/// Layout: `domain || network.id || validator_id (8) || pubkey (48)`.
/// All fields are fixed-width, so no length prefixes are needed.
#[must_use]
pub fn validator_possession_proof_message(
    network: &NetworkDefinition,
    validator_id: ValidatorId,
    pubkey: &ConsensusPublicKey,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(DOMAIN_VALIDATOR_POSSESSION_PROOF.len() + 1 + 8 + 48);
    out.extend_from_slice(DOMAIN_VALIDATOR_POSSESSION_PROOF);
    out.push(network.id);
    out.extend_from_slice(&validator_id.to_le_bytes());
    out.extend_from_slice(pubkey.as_bytes());
    out
}

/// Sign the proof-of-possession for `signer`'s public key claimed under
/// `validator_id` on `network`.
///
/// The message covers `signer.public_key()` itself, so the proof is
/// bound to exactly the key that signs it.
///
/// # Errors
///
/// Propagates [`SignError`] when the signer cannot sign.
pub fn validator_possession_proof_sign(
    signer: &dyn Signer,
    network: &NetworkDefinition,
    validator_id: ValidatorId,
) -> Result<ConsensusSignature, SignError> {
    let msg = validator_possession_proof_message(network, validator_id, &signer.public_key());
    signer.sign(&msg)
}

/// Verify that `possession_proof` proves possession of `pubkey` claimed under
/// `validator_id` on `network`.
#[must_use]
pub fn validator_possession_proof_verify(
    verifier: &dyn Verifier,
    network: &NetworkDefinition,
    validator_id: ValidatorId,
    pubkey: &ConsensusPublicKey,
    possession_proof: &ConsensusSignature,
) -> bool {
    let msg = validator_possession_proof_message(network, validator_id, pubkey);
    verifier.verify(pubkey, &msg, possession_proof)
}

#[cfg(test)]
mod tests {
    use hyperscale_crypto_bls::{BlsVerifier, signer_from_u64_seed as signer};

    use super::*;
    use crate::signing::ready_signal::DOMAIN_READY_SIGNAL;
    use crate::signing::shard_reveal::DOMAIN_SHARD_REVEAL;

    fn net() -> NetworkDefinition {
        NetworkDefinition::simulator()
    }

    /// Pins the byte layout of `validator_possession_proof_message`. Any change to the
    /// encoder — field order, width, domain tag — shifts these bytes and
    /// fails this test.
    #[test]
    fn validator_possession_proof_message_byte_layout_is_pinned() {
        let pk = signer(1).public_key();
        let id = ValidatorId::new(0x0123_4567_89AB_CDEF);
        let bytes = validator_possession_proof_message(&net(), id, &pk);

        let mut expected = Vec::new();
        expected.extend_from_slice(DOMAIN_VALIDATOR_POSSESSION_PROOF);
        expected.push(net().id);
        expected.extend_from_slice(&id.to_le_bytes());
        expected.extend_from_slice(pk.as_bytes());

        assert_eq!(bytes, expected);
        assert_eq!(
            bytes.len(),
            DOMAIN_VALIDATOR_POSSESSION_PROOF.len() + 1 + 8 + 48
        );
    }

    /// A proof is bound to the claimed identity: the same key's proof under
    /// a different `ValidatorId` must not verify.
    #[test]
    fn validator_possession_proof_message_differs_across_ids() {
        let pk = signer(1).public_key();
        let a = validator_possession_proof_message(&net(), ValidatorId::new(1), &pk);
        let b = validator_possession_proof_message(&net(), ValidatorId::new(2), &pk);
        assert_ne!(a, b);
    }

    /// Cross-network replay protection: identical `(id, pubkey)` under
    /// different networks must produce different signing bytes.
    #[test]
    fn validator_possession_proof_message_differs_across_networks() {
        let pk = signer(1).public_key();
        let id = ValidatorId::new(7);
        let mainnet = validator_possession_proof_message(&NetworkDefinition::mainnet(), id, &pk);
        let stokenet = validator_possession_proof_message(&NetworkDefinition::stokenet(), id, &pk);
        assert_ne!(mainnet, stokenet);
    }

    /// Cross-domain replay protection: the domain tag diverges from the
    /// sibling tags at the prefix.
    #[test]
    fn validator_possession_proof_domain_differs_from_other_domains() {
        assert_ne!(DOMAIN_VALIDATOR_POSSESSION_PROOF, DOMAIN_READY_SIGNAL);
        assert_ne!(DOMAIN_VALIDATOR_POSSESSION_PROOF, DOMAIN_SHARD_REVEAL);
    }

    #[test]
    fn validator_possession_proof_sign_verify_round_trip() {
        let signer = signer(3);
        let id = ValidatorId::new(42);
        let proof = validator_possession_proof_sign(&signer, &net(), id).expect("sign");
        assert!(validator_possession_proof_verify(
            &BlsVerifier,
            &net(),
            id,
            &signer.public_key(),
            &proof
        ));
    }

    /// A proof signed by one key does not prove possession of another.
    #[test]
    fn validator_possession_proof_verify_rejects_cross_key() {
        let signer_a = signer(3);
        let signer_b = signer(4);
        let id = ValidatorId::new(42);
        let proof = validator_possession_proof_sign(&signer_a, &net(), id).expect("sign");
        assert!(!validator_possession_proof_verify(
            &BlsVerifier,
            &net(),
            id,
            &signer_b.public_key(),
            &proof
        ));
    }

    /// A proof for one identity does not verify under another — replay of a
    /// captured proof against a different `ValidatorId` fails.
    #[test]
    fn validator_possession_proof_verify_rejects_wrong_id() {
        let signer = signer(3);
        let proof =
            validator_possession_proof_sign(&signer, &net(), ValidatorId::new(42)).expect("sign");
        assert!(!validator_possession_proof_verify(
            &BlsVerifier,
            &net(),
            ValidatorId::new(43),
            &signer.public_key(),
            &proof
        ));
    }

    #[test]
    fn validator_possession_proof_verify_rejects_zero_signature() {
        let signer = signer(3);
        let id = ValidatorId::new(42);
        assert!(!validator_possession_proof_verify(
            &BlsVerifier,
            &net(),
            id,
            &signer.public_key(),
            &ConsensusSignature::ZERO
        ));
    }
}
