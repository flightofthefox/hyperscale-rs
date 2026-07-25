//! [`Signer`] over the keyed-hash mock scheme.

use hyperscale_crypto::{ConsensusPublicKey, ConsensusSignature, SignError, Signer, VrfProof};

use crate::derive;

/// A validator's mock signing identity: the 32-byte seed is the private
/// key.
///
/// Stateless and infallible, like BLS. Signatures are deterministic in
/// `(key, message)`, which is what lets [`Signer::vrf_sign`] reuse the
/// plain signing core.
#[derive(Clone)]
pub struct MockSigner {
    pk: ConsensusPublicKey,
}

impl std::fmt::Debug for MockSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockSigner").finish_non_exhaustive()
    }
}

impl MockSigner {
    /// Derive a signer from a 32-byte seed (deterministic).
    #[must_use]
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        Self {
            pk: derive::public_key(seed),
        }
    }
}

impl Signer for MockSigner {
    fn public_key(&self) -> ConsensusPublicKey {
        self.pk
    }

    fn sign(&self, message: &[u8]) -> Result<ConsensusSignature, SignError> {
        Ok(derive::signature(&self.pk, message))
    }

    fn vrf_sign(&self, message: &[u8]) -> Result<VrfProof, SignError> {
        Ok(VrfProof::new(
            *derive::signature(&self.pk, message).as_bytes(),
        ))
    }
}
