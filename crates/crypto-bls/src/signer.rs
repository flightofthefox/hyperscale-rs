//! [`Signer`] over a BLS12-381 (min-pk) private key.

use hyperscale_crypto::{ConsensusPublicKey, ConsensusSignature, SignError, Signer, VrfProof};
use radix_common::crypto::Bls12381G1PrivateKey;

use crate::bls_keypair_from_seed;

/// A validator's BLS signing identity.
///
/// Stateless: signing never consumes key material, so [`Signer::sign`]
/// and [`Signer::vrf_sign`] never fail. BLS min-pk signatures are
/// deterministic in `(key, message)`, which is what lets `vrf_sign`
/// reuse the plain signing core.
pub struct BlsSigner {
    key: Bls12381G1PrivateKey,
}

impl std::fmt::Debug for BlsSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlsSigner").finish_non_exhaustive()
    }
}

impl BlsSigner {
    /// Derive a signer from a 32-byte seed (deterministic; tests and
    /// simulation).
    #[must_use]
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        Self {
            key: bls_keypair_from_seed(seed),
        }
    }

    /// Wrap an already-constructed private key.
    #[must_use]
    pub const fn new(key: Bls12381G1PrivateKey) -> Self {
        Self { key }
    }
}

impl Signer for BlsSigner {
    fn public_key(&self) -> ConsensusPublicKey {
        ConsensusPublicKey::new(self.key.public_key().0)
    }

    fn sign(&self, message: &[u8]) -> Result<ConsensusSignature, SignError> {
        Ok(ConsensusSignature::new(self.key.sign_v1(message).0))
    }

    fn vrf_sign(&self, message: &[u8]) -> Result<VrfProof, SignError> {
        Ok(VrfProof::new(self.key.sign_v1(message).0))
    }
}
