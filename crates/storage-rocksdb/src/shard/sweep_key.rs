//! Key encoding for the sweep index.
//!
//! The index answers one question: which owners hold sweepable cells in
//! a given expiry bucket. So it keys bucket-major — `bucket_BE ++ owner`
//! — and a sweep seeks straight to its frontier's bucket and walks
//! forward, over a state keyspace that is owner-major and cannot be
//! walked that way.
//!
//! Which *cells* an owner holds in a bucket is a question the leaves
//! answer for themselves: the bucket leads a sweepable cell's local
//! half, so one owner's bucket is a contiguous leaf-key range.

use hyperscale_types::{Address, SWEEP_BUCKET_BYTES, SweepBucket};

use crate::typed_cf::{DbCodec, DbEncode};

/// The encoded width of a sweep-index row key: bucket, then owner.
pub const SWEEP_ROW_LEN: usize = SWEEP_BUCKET_BYTES + 32;

/// Codec for sweep-index keys: `bucket_BE_4B ++ owner_32B`.
#[derive(Default)]
pub struct SweepRowCodec;

impl DbEncode<(SweepBucket, Address)> for SweepRowCodec {
    fn encode_to(&self, value: &(SweepBucket, Address), buf: &mut Vec<u8>) {
        let (bucket, owner) = value;
        buf.extend_from_slice(&bucket.to_bytes());
        buf.extend_from_slice(&owner.to_bytes());
    }
}

impl DbCodec<(SweepBucket, Address)> for SweepRowCodec {
    fn decode(&self, bytes: &[u8]) -> (SweepBucket, Address) {
        assert_eq!(bytes.len(), SWEEP_ROW_LEN, "a sweep-index key is 36 bytes");
        let bucket: [u8; SWEEP_BUCKET_BYTES] =
            bytes[..SWEEP_BUCKET_BYTES].try_into().expect("bucket half");
        let owner: [u8; 32] = bytes[SWEEP_BUCKET_BYTES..].try_into().expect("owner half");
        (
            SweepBucket(u32::from_be_bytes(bucket)),
            Address::from_bytes(owner).expect("a stored sweep row names an address"),
        )
    }
}

/// The leaf-key range covering one owner's cells in one bucket: the
/// half-open interval every sweepable leaf of that pair falls in, and
/// nothing else does.
///
/// The bucket leads a sweepable cell's local half, so the pair is a
/// 36-byte prefix of the leaf key and the end is that prefix with the
/// remaining local bytes maxed out.
#[must_use]
pub fn leaf_bucket_bounds(owner: Address, bucket: SweepBucket) -> (Vec<u8>, Vec<u8>) {
    const BODY_LEN: usize = 16 - SWEEP_BUCKET_BYTES;
    let mut start = Vec::with_capacity(SWEEP_ROW_LEN + BODY_LEN);
    start.extend_from_slice(&owner.to_bytes());
    start.extend_from_slice(&bucket.to_bytes());
    let mut end = start.clone();
    start.extend_from_slice(&[0x00; BODY_LEN]);
    end.extend_from_slice(&[0xFF; BODY_LEN]);
    end.push(0x00);
    (start, end)
}

/// The raw sweep-index key a walk seeks to when it resumes at `bucket`.
#[must_use]
pub fn row_seek(bucket: SweepBucket) -> Vec<u8> {
    let mut seek = Vec::with_capacity(SWEEP_ROW_LEN);
    seek.extend_from_slice(&bucket.to_bytes());
    seek.extend_from_slice(&[0u8; 32]);
    seek
}

#[cfg(test)]
mod tests {
    use hyperscale_types::{AddressClass, LocalKey, SubstateKey};

    use super::*;

    fn owner(tag: u8) -> Address {
        Address::new([tag; 31], AddressClass::Principal)
    }

    #[test]
    fn rows_round_trip_and_order_by_bucket_then_owner() {
        let row = (SweepBucket(9), owner(3));
        assert_eq!(SweepRowCodec.decode(&SweepRowCodec.encode(&row)), row);
        assert!(
            SweepRowCodec.encode(&(SweepBucket(8), owner(0xFF)))
                < SweepRowCodec.encode(&(SweepBucket(9), owner(0)))
        );
    }

    #[test]
    fn bounds_cover_a_buckets_leaves_and_no_others() {
        let (start, end) = leaf_bucket_bounds(owner(3), SweepBucket(9));
        let leaf = |bucket: u32, body: u8| {
            let mut local = [body; 16];
            local[..SWEEP_BUCKET_BYTES].copy_from_slice(&bucket.to_be_bytes());
            SubstateKey {
                owner: owner(3),
                local: LocalKey(local),
            }
            .to_bytes()
            .to_vec()
        };
        for body in [0x00, 0x7F, 0xFF] {
            let inside = leaf(9, body);
            assert!(start <= inside && inside < end, "body {body:02x}");
        }
        assert!(leaf(8, 0xFF) < start);
        assert!(leaf(10, 0x00) >= end);
        // Another owner's cells in the same bucket sit outside entirely,
        // which is what makes the pair the unit the index rows count.
        let elsewhere = SubstateKey {
            owner: owner(4),
            local: LocalKey([0; 16]),
        }
        .to_bytes()
        .to_vec();
        assert!(elsewhere >= end);
    }

    #[test]
    fn a_row_seek_lands_at_or_below_every_row_of_its_bucket() {
        let seek = row_seek(SweepBucket(9));
        assert!(seek <= SweepRowCodec.encode(&(SweepBucket(9), owner(0))));
        assert!(seek > SweepRowCodec.encode(&(SweepBucket(8), owner(0xFF))));
    }
}
