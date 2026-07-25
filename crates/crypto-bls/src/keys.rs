//! BLS keypair generation.

use blst::min_pk::SecretKey;
use radix_common::crypto::Bls12381G1PrivateKey;
use rand::{Rng, rng};

/// Generate a new random BLS12-381 keypair.
///
/// Uses a random 32-byte seed with blst's `key_gen` for proper key derivation.
#[must_use]
pub fn generate_bls_keypair() -> Bls12381G1PrivateKey {
    let mut ikm = [0u8; 32];
    rng().fill_bytes(&mut ikm);
    bls_keypair_from_seed(&ikm)
}

/// Generate a BLS12-381 keypair from a seed (deterministic, for testing/simulation).
///
/// Uses blst's `key_gen` which hashes the full seed to derive a valid BLS scalar.
/// This is the proper way to deterministically generate BLS keys from arbitrary seeds.
///
/// # Panics
///
/// Cannot panic: `blst::min_pk::SecretKey::key_gen` succeeds for any 32-byte seed.
#[must_use]
pub fn bls_keypair_from_seed(seed: &[u8; 32]) -> Bls12381G1PrivateKey {
    // Use blst's key_gen which properly hashes the seed to derive a valid scalar
    let blst_sk = SecretKey::key_gen(seed, &[]).expect("key_gen should not fail");

    // Convert to radix-common type
    // blst secret key is a 32-byte scalar in big-endian format
    let sk_bytes = blst_sk.to_bytes();
    Bls12381G1PrivateKey::from_bytes(&sk_bytes).expect("valid BLS scalar bytes")
}

/// Deterministic seeded-key fixtures shared across the workspace's test
/// suites, so the same integer names the same key in every crate.
#[cfg(any(test, feature = "test-utils"))]
mod fixtures {
    use hyperscale_crypto::{ConsensusPublicKey, Signer};

    use crate::BlsSigner;

    /// Derive a signer from a small integer, widened into the seed space
    /// as little-endian bytes in the low 8 positions with the rest zero.
    #[must_use]
    pub fn signer_from_u64_seed(seed: u64) -> BlsSigner {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&seed.to_le_bytes());
        BlsSigner::from_seed(&bytes)
    }

    /// Public key of [`signer_from_u64_seed`] for the same integer.
    #[must_use]
    pub fn public_key_from_u64_seed(seed: u64) -> ConsensusPublicKey {
        signer_from_u64_seed(seed).public_key()
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub use fixtures::{public_key_from_u64_seed, signer_from_u64_seed};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bls_keypair_from_seed_is_deterministic_and_seed_sensitive() {
        let seed = [42u8; 32];
        assert_eq!(
            bls_keypair_from_seed(&seed).public_key(),
            bls_keypair_from_seed(&seed).public_key()
        );

        // Seeds differing only past the first 8 bytes must still produce
        // distinct keys (the full seed feeds key derivation).
        let mut seed_a = [0u8; 32];
        seed_a[30] = 0x30;
        seed_a[31] = 0x39;
        let mut seed_b = [0u8; 32];
        seed_b[30] = 0x30;
        seed_b[31] = 0x3a;
        assert_ne!(
            bls_keypair_from_seed(&seed_a).public_key(),
            bls_keypair_from_seed(&seed_b).public_key()
        );
    }
}
