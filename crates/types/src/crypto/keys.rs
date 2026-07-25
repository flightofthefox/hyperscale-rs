//! Ed25519 key generation for the transaction path.

use radix_common::crypto::Ed25519PrivateKey;
use rand::{Rng, rng};

/// Generate a new random Ed25519 keypair.
///
/// # Panics
///
/// Cannot panic: any 32 bytes form a valid Ed25519 private key.
#[must_use]
pub fn generate_ed25519_keypair() -> Ed25519PrivateKey {
    let mut secret = [0u8; 32];
    rng().fill_bytes(&mut secret);
    Ed25519PrivateKey::from_bytes(&secret).expect("valid key bytes")
}

/// Generate an Ed25519 keypair from a seed (deterministic, for testing/simulation).
///
/// # Panics
///
/// Cannot panic: any 32 bytes form a valid Ed25519 private key.
#[must_use]
pub fn ed25519_keypair_from_seed(seed: &[u8; 32]) -> Ed25519PrivateKey {
    Ed25519PrivateKey::from_bytes(seed).expect("valid seed bytes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ed25519_keypair_from_seed() {
        let seed = [42u8; 32];

        let kp1 = ed25519_keypair_from_seed(&seed);
        let kp2 = ed25519_keypair_from_seed(&seed);

        let msg = b"test";
        assert_eq!(kp1.sign(msg).0, kp2.sign(msg).0);
        assert_eq!(kp1.public_key(), kp2.public_key());
    }
}
