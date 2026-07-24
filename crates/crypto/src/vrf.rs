//! Verifiable Random Function (VRF) outputs and proofs.
//!
//! Beacon-chain proposals carry per-slot VRF reveals: every committee
//! member's `(signer, slot, network.id)` triple deterministically
//! produces a `(VrfOutput, VrfProof)` pair, and the proof is verifiable
//! by anyone holding the signer's pubkey. The outputs are mixed into
//! the beacon's randomness for committee resampling.
//!
//! The proof is a signature container under the scheme's deterministic
//! signing core; the output is the 32-byte digest of that proof. Both
//! are distinct newtypes (rather than aliases for
//! [`ConsensusSignature`](crate::ConsensusSignature) or a bare hash) so
//! the type system catches accidental conflation with block-vote
//! signatures or general-purpose digests at every call site.

use sbor::prelude::*;

/// Wire length of a `VrfOutput` in bytes.
pub const VRF_OUTPUT_BYTES: usize = 32;

/// Wire length of a `VrfProof` in bytes (a signature container).
pub const VRF_PROOF_BYTES: usize = 96;

/// 32-byte VRF output. Digest of the proof; mixed into beacon randomness.
///
/// A VRF output is the digest of a specific VRF proof under a specific
/// `(signer, slot, network.id)` binding, not a free-floating 32-byte
/// hash. Type confusion between the two would silently break randomness
/// derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BasicSbor)]
#[sbor(transparent)]
pub struct VrfOutput([u8; VRF_OUTPUT_BYTES]);

impl VrfOutput {
    /// All-zero VRF output — used as a placeholder where no VRF reveal
    /// is present (e.g. genesis randomness seed before any committee
    /// has signed).
    pub const ZERO: Self = Self([0u8; VRF_OUTPUT_BYTES]);

    /// Build a `VrfOutput` from a raw 32-byte digest. The
    /// proof-to-output binding lives in
    /// [`Verifier::vrf_output`](crate::Verifier::vrf_output); this
    /// constructor takes the digest directly for wire deserialisation
    /// and adversarial test setup.
    #[must_use]
    pub const fn new(bytes: [u8; VRF_OUTPUT_BYTES]) -> Self {
        Self(bytes)
    }

    /// Get the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; VRF_OUTPUT_BYTES] {
        &self.0
    }
}

/// 96-byte VRF proof — a signature over the `(network.id, slot)` VRF
/// message under the signer's key.
///
/// Verifiable by anyone holding the signer's pubkey. The proof's digest
/// is the corresponding [`VrfOutput`]; the pair-binding lives in
/// [`Verifier::vrf_output`](crate::Verifier::vrf_output), not at the
/// type level.
///
/// Distinct from [`ConsensusSignature`](crate::ConsensusSignature) at
/// the type level — VRF proofs and block-vote signatures share the same
/// container width but sign under different domain tags, verify against
/// different message constructions, and the VRF path requires the
/// scheme's deterministic signing core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BasicSbor)]
#[sbor(transparent)]
pub struct VrfProof([u8; VRF_PROOF_BYTES]);

impl VrfProof {
    /// All-zero VRF proof — used as a placeholder (genesis block has
    /// no signing committee yet, so it carries a sentinel).
    pub const ZERO: Self = Self([0u8; VRF_PROOF_BYTES]);

    /// Build a `VrfProof` from raw bytes. Honest construction goes
    /// through [`Signer::vrf_sign`](crate::Signer::vrf_sign); this
    /// constructor takes the bytes directly for wire deserialisation
    /// and adversarial test setup.
    #[must_use]
    pub const fn new(bytes: [u8; VRF_PROOF_BYTES]) -> Self {
        Self(bytes)
    }

    /// Get the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; VRF_PROOF_BYTES] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vrf_output_sbor_round_trip() {
        let original = VrfOutput::new([0xAB; VRF_OUTPUT_BYTES]);
        let bytes = basic_encode(&original).unwrap();
        let decoded: VrfOutput = basic_decode(&bytes).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn vrf_proof_sbor_round_trip() {
        let original = VrfProof::new([0xCD; VRF_PROOF_BYTES]);
        let bytes = basic_encode(&original).unwrap();
        let decoded: VrfProof = basic_decode(&bytes).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn zero_sentinels() {
        assert_eq!(VrfOutput::ZERO.as_bytes(), &[0u8; VRF_OUTPUT_BYTES]);
        assert_eq!(VrfProof::ZERO.as_bytes(), &[0u8; VRF_PROOF_BYTES]);
    }

    #[test]
    fn sbor_encoding_is_transparent_to_inner_bytes() {
        let raw: [u8; VRF_OUTPUT_BYTES] = [0xAB; VRF_OUTPUT_BYTES];
        let wrapped = VrfOutput::new(raw);
        let raw_bytes = basic_encode(&raw).unwrap();
        let wrapped_bytes = basic_encode(&wrapped).unwrap();
        assert_eq!(
            raw_bytes, wrapped_bytes,
            "#[sbor(transparent)] must make newtype encoding byte-identical to inner array"
        );
    }
}
