//! The beacon's running randomness seed.

use sbor::prelude::*;

/// Wire length of a [`Randomness`] in bytes.
pub const RANDOMNESS_BYTES: usize = 32;

/// 32-byte beacon randomness.
///
/// BLAKE3 digest of the prior randomness concatenated with each slot's
/// accepted [`VrfOutput`](hyperscale_crypto::VrfOutput)s. Seeds
/// committee sampling and pool draws on the beacon, and feeds per-shard
/// randomness derivations downstream.
///
/// Distinct from [`Hash`](crate::Hash) and
/// [`VrfOutput`](hyperscale_crypto::VrfOutput) at the type level: this
/// is the running beacon seed, not a free-floating digest or a per-slot
/// VRF output. The type forces call sites to be explicit about which
/// 32-byte input the PRNG seed actually is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BasicSbor)]
#[sbor(transparent)]
pub struct Randomness([u8; RANDOMNESS_BYTES]);

impl Randomness {
    /// All-zero randomness — bootstrap value used as the genesis seed
    /// before any VRF reveal has been folded in.
    pub const ZERO: Self = Self([0u8; RANDOMNESS_BYTES]);

    /// Build a `Randomness` from a raw 32-byte digest.
    #[must_use]
    pub const fn new(bytes: [u8; RANDOMNESS_BYTES]) -> Self {
        Self(bytes)
    }

    /// Get the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; RANDOMNESS_BYTES] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomness_sbor_round_trip() {
        let original = Randomness::new([0x42; RANDOMNESS_BYTES]);
        let bytes = basic_encode(&original).unwrap();
        let decoded: Randomness = basic_decode(&bytes).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn zero_sentinel() {
        assert_eq!(Randomness::ZERO.as_bytes(), &[0u8; RANDOMNESS_BYTES]);
    }
}
