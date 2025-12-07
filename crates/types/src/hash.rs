//! Cryptographic hash type using Blake3.

use sbor::prelude::*;
use std::fmt;

/// A 32-byte cryptographic hash using Blake3.
///
/// Provides constant-time comparison and is safe to use as a HashMap key.
/// All hashing operations are deterministic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, BasicSbor)]
#[sbor(transparent)]
pub struct Hash([u8; 32]);

impl Hash {
    /// Size of hash in bytes.
    pub const BYTES: usize = 32;

    /// Zero hash (all bytes are 0x00).
    pub const ZERO: Self = Self([0u8; 32]);

    /// Max hash (all bytes are 0xFF).
    pub const MAX: Self = Self([0xFFu8; 32]);

    /// Create hash from bytes using Blake3.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hash = blake3::hash(bytes);
        Self(*hash.as_bytes())
    }

    /// Create a Hash from raw hash bytes (without hashing).
    ///
    /// # Panics
    ///
    /// Panics if bytes length is not exactly 32.
    pub fn from_hash_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 32, "Hash must be exactly 32 bytes");
        let mut arr = [0u8; 32];
        arr.copy_from_slice(bytes);
        Self(arr)
    }

    /// Create hash from multiple byte slices.
    pub fn from_parts(parts: &[&[u8]]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for part in parts {
            hasher.update(part);
        }
        Self(*hasher.finalize().as_bytes())
    }

    /// Parse hash from hex string.
    pub fn from_hex(hex: &str) -> Result<Self, HexError> {
        if hex.len() != 64 {
            return Err(HexError::InvalidLength {
                expected: 64,
                actual: hex.len(),
            });
        }

        let mut bytes = [0u8; 32];
        hex::decode_to_slice(hex, &mut bytes).map_err(|_| HexError::InvalidHex)?;

        Ok(Self(bytes))
    }

    /// Convert hash to hex string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Get bytes as slice reference.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to bytes array.
    pub fn to_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Count leading zero bits.
    pub fn leading_zero_bits(&self) -> u32 {
        let mut count = 0u32;
        for &byte in &self.0 {
            if byte == 0 {
                count += 8;
            } else {
                count += byte.leading_zeros();
                break;
            }
        }
        count
    }

    /// Interpret first 8 bytes as u64 (little-endian).
    pub fn as_u64(&self) -> u64 {
        u64::from_le_bytes(self.0[0..8].try_into().unwrap())
    }

    /// Check if this is the zero hash.
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }

    /// Compute a 64-bit value from all 32 bytes using polynomial hash.
    pub fn as_long(&self) -> i64 {
        let mut hash: i64 = 17;
        for &byte in &self.0 {
            hash = hash.wrapping_mul(31).wrapping_add(byte as i64);
        }
        hash
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = self.to_hex();
        write!(f, "Hash({}..{})", &hex[..8], &hex[56..])
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Errors that can occur when parsing hex strings.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HexError {
    /// Invalid hex string length.
    #[error("Invalid hex length: expected {expected}, got {actual}")]
    InvalidLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Invalid hex characters.
    #[error("Invalid hex string")]
    InvalidHex,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let data = b"hello world";
        let hash1 = Hash::from_bytes(data);
        let hash2 = Hash::from_bytes(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_collision_resistance() {
        let hash1 = Hash::from_bytes(b"hello");
        let hash2 = Hash::from_bytes(b"world");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hex_roundtrip() {
        let original = Hash::from_bytes(b"test data");
        let hex = original.to_hex();
        assert_eq!(hex.len(), 64);

        let parsed = Hash::from_hex(&hex).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_is_zero() {
        assert!(Hash::ZERO.is_zero());
        assert!(!Hash::MAX.is_zero());
        assert!(!Hash::from_bytes(b"test").is_zero());
    }
}
