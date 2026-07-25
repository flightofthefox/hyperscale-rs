//! Batch verification for Ed25519 transaction signatures.

use ed25519_dalek::{Signature as DalekSignature, VerifyingKey as DalekVerifyingKey, verify_batch};
use radix_common::crypto::{Ed25519PublicKey, Ed25519Signature};

/// Batch verify multiple Ed25519 signatures.
///
/// This uses the ed25519-dalek batch verification which is significantly faster
/// than verifying signatures one at a time (roughly 2x speedup for batches of 64+).
///
/// Returns `true` only if ALL signatures are valid. If any signature is invalid,
/// returns `false` without indicating which one failed.
#[must_use]
pub fn batch_verify_ed25519(
    messages: &[&[u8]],
    signatures: &[Ed25519Signature],
    pubkeys: &[Ed25519PublicKey],
) -> bool {
    if messages.len() != signatures.len() || signatures.len() != pubkeys.len() {
        return false;
    }
    if messages.is_empty() {
        return true;
    }

    // Convert to ed25519-dalek types
    let mut dalek_sigs = Vec::with_capacity(signatures.len());
    let mut dalek_pks = Vec::with_capacity(pubkeys.len());

    for (sig, pk) in signatures.iter().zip(pubkeys.iter()) {
        dalek_sigs.push(DalekSignature::from_bytes(&sig.0));

        match DalekVerifyingKey::from_bytes(&pk.0) {
            Ok(vk) => dalek_pks.push(vk),
            Err(_) => return false,
        }
    }

    verify_batch(messages, &dalek_sigs, &dalek_pks).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generate_ed25519_keypair, verify_ed25519};

    #[test]
    fn test_ed25519_sign_verify() {
        let keypair = generate_ed25519_keypair();
        let message = b"test message";

        let signature = keypair.sign(message);
        let pubkey = keypair.public_key();

        assert!(verify_ed25519(message, &pubkey, &signature));
    }

    #[test]
    fn test_ed25519_verify_fails_wrong_message() {
        let keypair = generate_ed25519_keypair();
        let message = b"test message";
        let wrong = b"wrong message";

        let signature = keypair.sign(message);
        let pubkey = keypair.public_key();

        assert!(!verify_ed25519(wrong, &pubkey, &signature));
    }

    #[test]
    fn test_batch_verify_ed25519() {
        let kp1 = generate_ed25519_keypair();
        let kp2 = generate_ed25519_keypair();
        let kp3 = generate_ed25519_keypair();

        let msg1 = b"message 1";
        let msg2 = b"message 2";
        let msg3 = b"message 3";

        let sig1 = kp1.sign(msg1);
        let sig2 = kp2.sign(msg2);
        let sig3 = kp3.sign(msg3);

        let messages: Vec<&[u8]> = vec![msg1, msg2, msg3];
        let signatures = vec![sig1, sig2, sig3];
        let pubkeys = vec![kp1.public_key(), kp2.public_key(), kp3.public_key()];

        assert!(batch_verify_ed25519(&messages, &signatures, &pubkeys));
    }

    #[test]
    fn test_batch_verify_ed25519_fails_with_bad_signature() {
        let kp1 = generate_ed25519_keypair();
        let kp2 = generate_ed25519_keypair();

        let msg1 = b"message 1";
        let msg2 = b"message 2";

        let sig1 = kp1.sign(msg1);
        let sig2 = kp2.sign(b"wrong message"); // Sign wrong message

        let messages: Vec<&[u8]> = vec![msg1, msg2];
        let signatures = vec![sig1, sig2];
        let pubkeys = vec![kp1.public_key(), kp2.public_key()];

        assert!(!batch_verify_ed25519(&messages, &signatures, &pubkeys));
    }

    #[test]
    fn test_batch_verify_ed25519_empty() {
        let messages: Vec<&[u8]> = vec![];
        let signatures: Vec<Ed25519Signature> = vec![];
        let pubkeys: Vec<Ed25519PublicKey> = vec![];

        assert!(batch_verify_ed25519(&messages, &signatures, &pubkeys));
    }
}
