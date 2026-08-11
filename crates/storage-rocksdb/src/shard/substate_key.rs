//! Substate key encoding for `RocksDB`.
//!
//! The `state` column family keys on a [`SubstateKey`]'s own leaf bytes —
//! owner address, then local half. Both halves are fixed-width, so the
//! concatenation preserves lexicographic ordering for prefix scans and
//! decodes back without a length prefix.

use hyperscale_types::{LEAF_KEY_BYTES, SubstateKey};

use crate::typed_cf::{DbCodec, DbEncode};

/// Codec for substate keys: the key's leaf bytes, by identity.
#[derive(Default)]
pub struct SubstateKeyCodec;

impl DbEncode<SubstateKey> for SubstateKeyCodec {
    fn encode_to(&self, value: &SubstateKey, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&value.to_bytes());
    }
}

impl DbCodec<SubstateKey> for SubstateKeyCodec {
    fn decode(&self, bytes: &[u8]) -> SubstateKey {
        let key: [u8; LEAF_KEY_BYTES] = bytes.try_into().expect("a substate key is its leaf bytes");
        SubstateKey::from_bytes(key).expect("a stored leaf key names an address")
    }
}
