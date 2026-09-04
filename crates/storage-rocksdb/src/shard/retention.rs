//! What history a shard keeps, and until when.
//!
//! Retention is a span of weighted time. Every window that licenses a
//! read of history — a reclaim probe's, a provision's, a state proof's —
//! is stated in milliseconds against the consensus-authenticated
//! weighted timestamp, so a count of versions answers those windows only
//! through whatever block rate the chain happens to be running at.
//! Nothing enforces a rate, which makes a count an answer to a different
//! question: whether a licensed anchor is servable would depend on how
//! fast blocks had happened to arrive.
//!
//! So the floor is [`RETENTION_HORIZON`] behind the tip's own timestamp,
//! and every version a consumer may name is servable by construction.
//! What that costs is history proportional to the block rate, and what
//! bounds it is the per-block write caps the protocol already sets: what
//! is kept is the writes of the last horizon, which is what a consumer is
//! licensed to ask about.
//!
//! # One floor, four readers
//!
//! The floor is stored rather than recomputed. A historical cell read, a
//! historical range read, `snapshot_at` and both collectors ask for it,
//! and what a reader may ask for has to be exactly what the collector has
//! not deleted. Admitting `height >= floor` and deleting below it is that
//! relationship, and it holds because there is one value rather than four
//! arithmetic expressions that have to agree.
//!
//! # Versions this store never committed
//!
//! The floor is the first *dated* version at or above where it stands, so
//! a store whose history begins above zero needs nothing said about the
//! versions below. A snap-synced store dates its first committed block
//! and the floor arrives there; a split child's dates start at its
//! adoption. Neither needs a seed, because a version with no date is one
//! with no history to serve.

use hyperscale_types::{RETENTION_HORIZON, WeightedTimestamp};
use rocksdb::WriteBatch;

use super::column_families::VersionTimeCf;
use super::core::RocksDbShardStorage;
use super::metadata::{read_retention_floor, write_retention_floor};
use crate::typed_cf::{self, ReadableStore, TypedCf};

/// The oldest version `store` answers historical reads at.
///
/// Read through whatever view the caller already holds: a reader that
/// took a snapshot and then consulted the live floor could be told a
/// version is retained by one and gone by the other.
pub fn retention_floor(store: &impl ReadableStore) -> u64 {
    read_retention_floor(store)
}

impl RocksDbShardStorage {
    /// The oldest version this store answers historical reads at.
    #[must_use]
    pub fn retention_floor(&self) -> u64 {
        read_retention_floor(&*self.db)
    }

    /// Date `version` with the weighted timestamp of the QC that
    /// certified it, and move the floor to the oldest version still
    /// inside [`RETENTION_HORIZON`] of it.
    ///
    /// Written into the batch that advances the tree, so a version is
    /// dated atomically with the state it commits: a date with no tree at
    /// that version, or the reverse, would leave the floor ruling on a
    /// version that does not exist.
    ///
    /// The scan runs forward from the stored floor, so each dated version
    /// is passed once over the life of the store whatever the block rate.
    ///
    /// Returns the floor this commit establishes, which is what anything
    /// else pruning in the same batch cuts at: reading it back off the
    /// store would give the floor before this commit and cut a version
    /// short.
    pub(crate) fn advance_retention_floor(
        &self,
        batch: &mut WriteBatch,
        version: u64,
        tip_ts: WeightedTimestamp,
    ) -> u64 {
        let cf = self.cf();
        let handle = VersionTimeCf::handle(&cf);
        typed_cf::batch_put::<VersionTimeCf>(batch, handle, &version, &tip_ts.as_millis());

        // The floor moves only past what this commit retires. A version
        // with no date of its own — the empty tree at zero, or anything
        // below where a snap-synced store's history begins — is left
        // where it was rather than skipped over, since nothing about it
        // has aged out.
        let mut next = read_retention_floor(&*self.db);
        let mut retire: Vec<u64> = Vec::new();
        let cutoff = tip_ts.minus(RETENTION_HORIZON).as_millis();
        for (dated, ts) in typed_cf::iter_from::<VersionTimeCf>(&self.db, handle, &next) {
            if dated >= version || ts >= cutoff {
                break;
            }
            retire.push(dated);
            next = dated + 1;
        }
        for dated in retire {
            typed_cf::batch_delete::<VersionTimeCf>(batch, handle, &dated);
        }
        write_retention_floor(batch, next);
        next
    }
}
