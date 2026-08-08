//! [`WaveId`] — self-contained globally unique wave identifier.

use std::collections::BTreeSet;
use std::fmt::{self, Display};

use hyperscale_hbor::Hbor;

use crate::{BlockHeight, ShardId};

/// Cap on `WaveId.remote_shards` length at decode time.
///
/// A wave's remote shard set is at most `num_shards - 1` (a wave can
/// depend on every other shard). Real deployments run far below this
/// cap; it exists so a peer can't claim a huge dependency set and force
/// the decoder to insert millions of `ShardId`s into a `BTreeSet`
/// before the first frame check fires.
pub const MAX_REMOTE_SHARDS_PER_WAVE: usize = 1024;

/// Self-contained wave identifier.
///
/// Globally unique: includes the local shard, block height, and the provision
/// dependency set (remote shards). This eliminates composite `(block_hash, wave_id)`
/// keys throughout the codebase.
///
/// The provision dependency set for a transaction is the set of remote shards
/// it needs state provisions from before execution. Transactions with identical
/// dependency sets belong to the same wave and can be voted on together.
///
/// A wave with empty `remote_shards` represents single-shard transactions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Hbor)]
pub struct WaveId {
    shard_id: ShardId,
    block_height: BlockHeight,
    #[hbor(max = MAX_REMOTE_SHARDS_PER_WAVE)]
    remote_shards: BTreeSet<ShardId>,
}

impl WaveId {
    /// Create a new `WaveId`. The remote-shard cap is enforced at encode
    /// and decode, not here.
    #[must_use]
    pub const fn new(
        shard_id: ShardId,
        block_height: BlockHeight,
        remote_shards: BTreeSet<ShardId>,
    ) -> Self {
        Self {
            shard_id,
            block_height,
            remote_shards,
        }
    }

    /// The shard that committed the block containing this wave's transactions.
    #[must_use]
    pub const fn shard_id(&self) -> ShardId {
        self.shard_id
    }

    /// Block height at which the wave's transactions were committed.
    #[must_use]
    pub const fn block_height(&self) -> BlockHeight {
        self.block_height
    }

    /// Set of remote shards the transactions depend on (empty for single-shard waves).
    #[must_use]
    pub const fn remote_shards(&self) -> &BTreeSet<ShardId> {
        &self.remote_shards
    }

    /// Whether this is a single-shard wave (no remote dependencies).
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.remote_shards.is_empty()
    }

    /// Number of provision source shards.
    #[must_use]
    pub fn dependency_count(&self) -> usize {
        self.remote_shards.len()
    }
}

impl Display for WaveId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(
                f,
                "Wave(shard={}, h={}, ∅)",
                self.shard_id.inner(),
                self.block_height.inner()
            )
        } else {
            write!(
                f,
                "Wave(shard={}, h={}, {{",
                self.shard_id.inner(),
                self.block_height.inner()
            )?;
            for (i, shard) in self.remote_shards.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", shard.inner())?;
            }
            write!(f, "}})")
        }
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_hbor::{
        DecodeError, from_slice as hbor_from_slice, to_vec as hbor_to_vec, varint,
    };

    use super::*;

    fn sample_wave_id() -> WaveId {
        WaveId::new(
            ShardId::leaf(3, 3),
            BlockHeight::new(42),
            [ShardId::leaf(3, 1), ShardId::leaf(3, 7)]
                .into_iter()
                .collect(),
        )
    }

    #[test]
    fn hbor_roundtrip() {
        let wave = sample_wave_id();
        let bytes = hbor_to_vec(&wave).unwrap();
        let decoded: WaveId = hbor_from_slice(&bytes).unwrap();
        assert_eq!(decoded, wave);
    }

    #[test]
    fn hbor_roundtrip_empty_remote_shards() {
        let wave = WaveId::new(ShardId::leaf(3, 0), BlockHeight::new(1), BTreeSet::new());
        let bytes = hbor_to_vec(&wave).unwrap();
        let decoded: WaveId = hbor_from_slice(&bytes).unwrap();
        assert_eq!(decoded, wave);
    }

    /// Hand-roll a `WaveId` whose `remote_shards` length exceeds the cap and
    /// verify decode rejects it before iterating.
    #[test]
    fn decode_rejects_oversized_remote_shards() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&hbor_to_vec(&ShardId::leaf(3, 0)).unwrap());
        buf.extend_from_slice(&hbor_to_vec(&BlockHeight::new(0)).unwrap());
        varint::write(&mut buf, MAX_REMOTE_SHARDS_PER_WAVE + 1).unwrap();
        buf.extend(std::iter::repeat_n(
            0u8,
            (MAX_REMOTE_SHARDS_PER_WAVE + 1) * 12,
        ));
        let err = hbor_from_slice::<WaveId>(&buf).unwrap_err();
        assert!(matches!(
            err,
            DecodeError::BoundExceeded { max, actual }
                if max == MAX_REMOTE_SHARDS_PER_WAVE
                    && actual == MAX_REMOTE_SHARDS_PER_WAVE + 1
        ));
    }

    /// Duplicate `BTreeSet` elements reject at decode time.
    #[test]
    fn decode_rejects_duplicate_remote_shards() {
        #[derive(Hbor)]
        struct ForgedWaveId {
            shard_id: ShardId,
            block_height: BlockHeight,
            remote_shards: Vec<ShardId>,
        }
        let forged = ForgedWaveId {
            shard_id: ShardId::leaf(3, 0),
            block_height: BlockHeight::new(0),
            remote_shards: vec![ShardId::leaf(3, 5), ShardId::leaf(3, 5)],
        };
        let buf = hbor_to_vec(&forged).unwrap();
        let err = hbor_from_slice::<WaveId>(&buf).unwrap_err();
        assert!(matches!(err, DecodeError::UnsortedKeys));
    }
}
