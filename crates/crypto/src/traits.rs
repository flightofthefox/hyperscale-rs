//! The [`Signer`] and [`Verifier`] scheme traits.

use thiserror::Error;

use crate::{AggregateSignature, ConsensusPublicKey, ConsensusSignature, VrfOutput, VrfProof};

/// Signing failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SignError {
    /// The signer's one-time key material is spent. Stateful schemes
    /// only; call sites treat any `Err` as "cannot sign" — emit
    /// nothing, log at error level.
    #[error("signing key material exhausted")]
    Exhausted,
}

/// Aggregation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AggregateError {
    /// No signatures were provided.
    #[error("cannot aggregate zero signatures")]
    Empty,
    /// A signature failed scheme-level validation during aggregation.
    #[error("signature rejected during aggregation")]
    InvalidSignature,
}

/// A validator's signing identity under one scheme.
///
/// A stateful, fallible object with no exposed private-key type:
/// construction is per-impl (from a seed or stored key bytes), and
/// stateful schemes may consume safety-bearing state on every call.
pub trait Signer: Send + Sync {
    /// The public key other validators verify this signer's output
    /// against.
    fn public_key(&self) -> ConsensusPublicKey;

    /// Sign a consensus message. The message arrives fully
    /// domain-separated; the signer adds no framing.
    ///
    /// # Errors
    ///
    /// [`SignError::Exhausted`] when the scheme's one-time key material
    /// is spent. Stateless schemes never fail.
    fn sign(&self, message: &[u8]) -> Result<ConsensusSignature, SignError>;

    /// Sign a VRF message with the scheme's deterministic signing core.
    ///
    /// Distinct from [`sign`](Self::sign) because VRF soundness
    /// requires determinism — the same `(key, message)` must always
    /// produce the same proof — which a general signing path need not
    /// guarantee.
    ///
    /// # Errors
    ///
    /// [`SignError::Exhausted`] when the scheme's one-time key material
    /// is spent. Stateless schemes never fail.
    fn vrf_sign(&self, message: &[u8]) -> Result<VrfProof, SignError>;
}

/// Verification and aggregation under one scheme.
///
/// Ops sit at certificate altitude: each is a semantic statement about
/// signers and messages, answered however the scheme answers it.
/// Threshold and voting-power checks are committee policy and stay at
/// call sites; callers select the pubkeys (via signer bitfields) before
/// calling in.
pub trait Verifier: Send + Sync {
    /// Did the holder of `key` sign `message`?
    fn verify(&self, key: &ConsensusPublicKey, message: &[u8], sig: &ConsensusSignature) -> bool;

    /// Combine per-signer signatures into one aggregate.
    ///
    /// Callers canonicalize input order to committee-index order before
    /// aggregating; schemes whose aggregates are order-sensitive rely
    /// on verification recomputing in that same canonical order.
    ///
    /// # Errors
    ///
    /// [`AggregateError::Empty`] on empty input;
    /// [`AggregateError::InvalidSignature`] when a signature fails
    /// scheme-level validation.
    fn aggregate(&self, sigs: &[ConsensusSignature]) -> Result<AggregateSignature, AggregateError>;

    /// Did every holder of `keys` sign `message`, and is `agg` the
    /// aggregate of exactly those signatures?
    fn verify_aggregate_same_message(
        &self,
        message: &[u8],
        agg: &AggregateSignature,
        keys: &[ConsensusPublicKey],
    ) -> bool;

    /// Did the holder of `keys[i]` sign `messages[i]` for every `i`,
    /// and is `agg` the aggregate of exactly those signatures?
    fn verify_aggregate_different_messages(
        &self,
        messages: &[&[u8]],
        agg: &AggregateSignature,
        keys: &[ConsensusPublicKey],
    ) -> bool;

    /// Per-item verdicts for `(messages[i], sigs[i], keys[i])` triples.
    /// Schemes may verify the batch as a whole and only fall back to
    /// per-item checks on failure. Length mismatch yields all-false.
    fn batch_verify(
        &self,
        messages: &[&[u8]],
        sigs: &[ConsensusSignature],
        keys: &[ConsensusPublicKey],
    ) -> Vec<bool>;

    /// Is `proof` the holder of `key`'s deterministic signature over
    /// `message`?
    fn verify_vrf(&self, key: &ConsensusPublicKey, message: &[u8], proof: &VrfProof) -> bool;

    /// The 32-byte output a proof commits to. Pure digest — callers
    /// must have verified the proof first.
    fn vrf_output(&self, proof: &VrfProof) -> VrfOutput;
}
