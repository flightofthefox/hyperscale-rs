//! Composite key type for the `state_history` column family.
//!
//! Layout: `[substate_key_bytes (32B)][write_version_BE_8B]`
//!
//! The big-endian version suffix ensures that for a given substate key
//! prefix, entries sort in ascending lexicographic order on version —
//! enabling the forward seek used by historical reads to find the
//! smallest `write_version > V` for a key.

use hyperscale_types::SubstateKey;

use super::substate_key::SubstateKeyCodec;
use crate::typed_cf::{DbCodec, DbEncode};

const VERSION_LEN: usize = 8;

/// Key type for the versioned substates CF.
type VersionedKey = (SubstateKey, u64);

/// Codec for versioned substate keys: `substate_key_bytes ++ version_BE_8B`.
///
/// Composes [`SubstateKeyCodec`] (for the substate key portion) with a
/// big-endian u64 suffix (for the version). The version suffix preserves
/// lexicographic ordering so that for a given substate key prefix, versions
/// sort ascending — enabling efficient "find latest version <= N" scans.
#[derive(Default)]
pub struct VersionedSubstateKeyCodec;

impl DbEncode<VersionedKey> for VersionedSubstateKeyCodec {
    fn encode_to(&self, value: &VersionedKey, buf: &mut Vec<u8>) {
        let (key, version) = value;
        SubstateKeyCodec.encode_to(key, buf);
        buf.extend_from_slice(&version.to_be_bytes());
    }
}

impl DbCodec<VersionedKey> for VersionedSubstateKeyCodec {
    fn decode(&self, bytes: &[u8]) -> VersionedKey {
        assert!(
            bytes.len() >= VERSION_LEN,
            "versioned key must be at least {VERSION_LEN} bytes, got {}",
            bytes.len()
        );
        let key_len = bytes.len() - VERSION_LEN;
        let key = SubstateKeyCodec.decode(&bytes[..key_len]);
        let version = u64::from_be_bytes(bytes[key_len..].try_into().unwrap());
        (key, version)
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::{Address, AddressClass, LocalKey};

    use super::*;

    fn make_test_key(local: [u8; 16]) -> SubstateKey {
        SubstateKey {
            owner: Address::new([0u8; 31], AddressClass::Component),
            local: LocalKey(local),
        }
    }

    #[test]
    fn round_trip() {
        let key = make_test_key([7u8; 16]);
        let version = 42u64;

        let encoded = VersionedSubstateKeyCodec.encode(&(key, version));
        let (decoded_key, decoded_version) = VersionedSubstateKeyCodec.decode(&encoded);

        assert_eq!(decoded_key, key);
        assert_eq!(decoded_version, version);
    }

    #[test]
    fn lexicographic_version_ordering() {
        let key = make_test_key([9u8; 16]);

        let buf1 = VersionedSubstateKeyCodec.encode(&(key, 1));
        let buf2 = VersionedSubstateKeyCodec.encode(&(key, 2));

        // Version 1 sorts before version 2 for the same storage key.
        assert!(buf1 < buf2);
    }

    #[test]
    #[should_panic(expected = "versioned key must be at least 8 bytes")]
    fn decode_too_short() {
        VersionedSubstateKeyCodec.decode(&[0; 7]);
    }
}
