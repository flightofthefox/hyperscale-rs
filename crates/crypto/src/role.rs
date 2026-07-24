//! Role newtypes for consensus key and signature material.
//!
//! Each type names the *role* the bytes play in the protocol, not the
//! scheme that produced them. They are opaque byte containers: nothing
//! outside a scheme impl crate may assume curve structure (a mock
//! signature need not be a valid G2 point). Widths match the current
//! scheme's compressed encodings so wire encodings stay byte-identical.

use sbor::prelude::*;

/// Wire length of a [`ConsensusPublicKey`] in bytes.
pub const CONSENSUS_PUBLIC_KEY_BYTES: usize = 48;

/// Wire length of a [`ConsensusSignature`] in bytes.
pub const CONSENSUS_SIGNATURE_BYTES: usize = 96;

/// Wire length of an [`AggregateSignature`] in bytes.
pub const AGGREGATE_SIGNATURE_BYTES: usize = 96;

/// A validator's consensus public key.
///
/// Identifies a validator for vote, timeout, proposal, and possession
/// proof verification. Only scheme impl crates interpret the bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, BasicSbor)]
#[sbor(transparent)]
pub struct ConsensusPublicKey([u8; CONSENSUS_PUBLIC_KEY_BYTES]);

impl ConsensusPublicKey {
    /// All-zero placeholder key — never a real validator's key.
    pub const ZERO: Self = Self([0u8; CONSENSUS_PUBLIC_KEY_BYTES]);

    /// Build from raw bytes. Honest key material comes from a scheme
    /// impl crate; this constructor exists for wire deserialisation and
    /// adversarial test setup.
    #[must_use]
    pub const fn new(bytes: [u8; CONSENSUS_PUBLIC_KEY_BYTES]) -> Self {
        Self(bytes)
    }

    /// Get the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; CONSENSUS_PUBLIC_KEY_BYTES] {
        &self.0
    }
}

/// A single validator's signature over a consensus message.
///
/// Carried by block votes, timeouts, proposer signatures, ready signals,
/// possession proofs, and signed network envelopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, BasicSbor)]
#[sbor(transparent)]
pub struct ConsensusSignature([u8; CONSENSUS_SIGNATURE_BYTES]);

impl ConsensusSignature {
    /// All-zero placeholder signature — used where a sentinel is
    /// structural (genesis artifacts) and never verified.
    pub const ZERO: Self = Self([0u8; CONSENSUS_SIGNATURE_BYTES]);

    /// Build from raw bytes. Honest construction goes through
    /// [`Signer::sign`](crate::Signer::sign); this constructor exists
    /// for wire deserialisation and adversarial test setup.
    #[must_use]
    pub const fn new(bytes: [u8; CONSENSUS_SIGNATURE_BYTES]) -> Self {
        Self(bytes)
    }

    /// Get the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; CONSENSUS_SIGNATURE_BYTES] {
        &self.0
    }
}

/// A multi-signer aggregate over one or more consensus messages.
///
/// Carried by quorum certificates, beacon PC/SPC certificates, ratify
/// certificates, and execution certificates. The signer set travels
/// beside it (as a bitfield or positional bundle), never inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, BasicSbor)]
#[sbor(transparent)]
pub struct AggregateSignature([u8; AGGREGATE_SIGNATURE_BYTES]);

impl AggregateSignature {
    /// All-zero placeholder aggregate — genesis QCs carry it and it is
    /// never verified.
    pub const ZERO: Self = Self([0u8; AGGREGATE_SIGNATURE_BYTES]);

    /// Build from raw bytes. Honest construction goes through
    /// [`Verifier::aggregate`](crate::Verifier::aggregate); this
    /// constructor exists for wire deserialisation and adversarial test
    /// setup.
    #[must_use]
    pub const fn new(bytes: [u8; AGGREGATE_SIGNATURE_BYTES]) -> Self {
        Self(bytes)
    }

    /// Get the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; AGGREGATE_SIGNATURE_BYTES] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbor_encoding_is_transparent_to_inner_bytes() {
        let raw = [0xABu8; CONSENSUS_PUBLIC_KEY_BYTES];
        let raw_bytes = basic_encode(&raw).unwrap();
        let wrapped_bytes = basic_encode(&ConsensusPublicKey::new(raw)).unwrap();
        assert_eq!(
            raw_bytes, wrapped_bytes,
            "#[sbor(transparent)] must make newtype encoding byte-identical to inner array"
        );
    }

    #[test]
    fn sbor_round_trips() {
        let key = ConsensusPublicKey::new([0x11; CONSENSUS_PUBLIC_KEY_BYTES]);
        let sig = ConsensusSignature::new([0x22; CONSENSUS_SIGNATURE_BYTES]);
        let agg = AggregateSignature::new([0x33; AGGREGATE_SIGNATURE_BYTES]);
        assert_eq!(
            basic_decode::<ConsensusPublicKey>(&basic_encode(&key).unwrap()).unwrap(),
            key
        );
        assert_eq!(
            basic_decode::<ConsensusSignature>(&basic_encode(&sig).unwrap()).unwrap(),
            sig
        );
        assert_eq!(
            basic_decode::<AggregateSignature>(&basic_encode(&agg).unwrap()).unwrap(),
            agg
        );
    }

    #[test]
    fn zero_sentinels() {
        assert_eq!(
            ConsensusPublicKey::ZERO.as_bytes(),
            &[0u8; CONSENSUS_PUBLIC_KEY_BYTES]
        );
        assert_eq!(
            ConsensusSignature::ZERO.as_bytes(),
            &[0u8; CONSENSUS_SIGNATURE_BYTES]
        );
        assert_eq!(
            AggregateSignature::ZERO.as_bytes(),
            &[0u8; AGGREGATE_SIGNATURE_BYTES]
        );
    }
}
