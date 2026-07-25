//! `Signed` trait + blanket `Verify` impl for sender/proposer-attested wire
//! messages.
//!
//! Every wire-form message that carries "this validator signed these bytes"
//! has the same shape: a signer identity, a signature, and a domain-
//! separated message that ties the signature to the payload. [`Signed`] is
//! the trait that names this pattern; each notification type implements it
//! by exposing its existing `signer` / `signature` / signing-message fields.
//!
//! The blanket [`impl<T: Signed + Clone> Verify<&SignedContext<'_>> for T`]
//! is the single source-of-truth signature check for the whole family: resolve the
//! signer's public key (caller's responsibility — see below), build a
//! [`SignedContext`], and call the predicate. On success the wire form is
//! promoted to [`Verified<T>`] in place via the standard typestate machinery.
//!
//! # Why the caller resolves the public key
//!
//! Committee-membership policy differs per message type: a block-header
//! verifier looks up the proposer's key directly, while sender-attested
//! messages first check that the sender is in the relevant shard committee.
//! That policy belongs at the handler, so [`SignedContext`] takes a
//! pre-resolved [`ConsensusPublicKey`] and the trait stays out of topology.

use hyperscale_crypto::Verifier;

use crate::{
    ConsensusPublicKey, ConsensusSignature, NetworkDefinition, ValidatorId, Verified, Verify,
};

/// A wire message that carries its own signer identity, signature, and
/// domain-separated signing-message bytes.
///
/// Implementors expose existing fields — no wrapper struct, no payload
/// repackaging. The blanket [`Verify<&SignedContext<'_>>`] impl is the
/// single signature check for every implementor.
pub trait Signed {
    /// Validator who claims to have signed this message.
    fn signer(&self) -> ValidatorId;

    /// signature carried on the message.
    fn signature(&self) -> &ConsensusSignature;

    /// Domain-separated bytes the signature is over. Reconstructed at
    /// verify time from the payload's own fields.
    fn signing_message(&self, network: &NetworkDefinition) -> Vec<u8>;

    /// Run the signature check without producing a [`Verified`] handle. Used by
    /// `IoLoop` handlers that peel the payload immediately after verifying
    /// and never need the wrapped form. Cheaper than [`Verify::verify`]
    /// because it skips the success-path clone.
    ///
    /// # Errors
    ///
    /// Returns [`SignedVerifyError::InvalidSignature`] when the
    /// signature does not validate.
    fn verify_signature(&self, ctx: &SignedContext<'_>) -> Result<(), SignedVerifyError> {
        let msg = self.signing_message(ctx.network);
        if ctx.verifier.verify(ctx.public_key, &msg, self.signature()) {
            Ok(())
        } else {
            Err(SignedVerifyError::InvalidSignature)
        }
    }
}

/// Context for [`Signed`] verification. The caller resolves the signer's
/// public key (typically via [`TopologySnapshot`](crate::TopologySnapshot)
/// committee lookup) and hands it in.
pub struct SignedContext<'a> {
    /// Active network definition; folded into the signing message for
    /// cross-network replay protection.
    pub network: &'a NetworkDefinition,
    /// Pre-resolved public key of the claimed signer.
    pub public_key: &'a ConsensusPublicKey,
    /// Scheme verifier the signature check runs through.
    pub verifier: &'a dyn Verifier,
}

/// Failure modes for [`Signed`] verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SignedVerifyError {
    /// signature did not validate against the signing message + public key.
    #[error("invalid signature on signed message")]
    InvalidSignature,
}

impl<T: Signed + Clone> Verify<&SignedContext<'_>> for T {
    type Error = SignedVerifyError;

    fn verify(&self, ctx: &SignedContext<'_>) -> Result<Verified<Self>, Self::Error> {
        self.verify_signature(ctx)?;
        Ok(Verified::new_unchecked(self.clone()))
    }
}
