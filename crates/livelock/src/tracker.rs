//! Tracker types for cross-shard cycle detection.
//!
//! These trackers maintain the state needed for detecting bidirectional
//! dependencies between shards that could cause livelock.

use hyperscale_types::{Hash, ShardGroupId};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Tracks committed cross-shard transactions for cycle detection.
///
/// When a cross-shard transaction is committed, we need to know which shards
/// it requires provisions from. This tracker maintains a bidirectional index
/// for efficient lookups in both directions:
///
/// 1. Given a TX, which shards does it need provisions from?
/// 2. Given a shard, which TXs need provisions from it?
///
/// The second lookup is critical for cycle detection: when we receive a
/// provision from shard S, we can quickly find all local TXs that need S's
/// state, and check if S has any TXs that need our state (bidirectional cycle).
#[derive(Debug, Default)]
pub struct CommittedCrossShardTracker {
    /// tx_hash -> shards we need provisions from
    txs_needing_shards: HashMap<Hash, BTreeSet<ShardGroupId>>,
    /// Reverse index: shard -> tx_hashes that need provisions from it
    shards_needed_by: HashMap<ShardGroupId, HashSet<Hash>>,
}

impl CommittedCrossShardTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a transaction that needs provisions from the given shards.
    ///
    /// # Arguments
    ///
    /// * `tx_hash` - The transaction hash
    /// * `shards` - Set of shards this TX needs provisions from
    pub fn add(&mut self, tx_hash: Hash, shards: BTreeSet<ShardGroupId>) {
        // Build reverse index
        for &shard in &shards {
            self.shards_needed_by
                .entry(shard)
                .or_default()
                .insert(tx_hash);
        }

        // Store forward mapping
        self.txs_needing_shards.insert(tx_hash, shards);
    }

    /// Remove a transaction (completed, deferred, or aborted).
    ///
    /// Cleans up both the forward and reverse indexes.
    pub fn remove(&mut self, tx_hash: &Hash) {
        if let Some(shards) = self.txs_needing_shards.remove(tx_hash) {
            // Clean up reverse index
            for shard in shards {
                if let Some(txs) = self.shards_needed_by.get_mut(&shard) {
                    txs.remove(tx_hash);
                    if txs.is_empty() {
                        self.shards_needed_by.remove(&shard);
                    }
                }
            }
        }
    }

    /// Get all TXs that need provisions from a specific shard.
    ///
    /// Used during cycle detection: when we receive a provision from shard S,
    /// we check if any of our committed TXs need provisions from S.
    pub fn txs_needing_shard(&self, shard: ShardGroupId) -> Option<&HashSet<Hash>> {
        self.shards_needed_by.get(&shard)
    }

    /// Check if a transaction is being tracked.
    pub fn contains(&self, tx_hash: &Hash) -> bool {
        self.txs_needing_shards.contains_key(tx_hash)
    }

    /// Get the shards a transaction needs provisions from.
    pub fn shards_for_tx(&self, tx_hash: &Hash) -> Option<&BTreeSet<ShardGroupId>> {
        self.txs_needing_shards.get(tx_hash)
    }

    /// Get the number of transactions being tracked.
    pub fn len(&self) -> usize {
        self.txs_needing_shards.len()
    }

    /// Check if the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.txs_needing_shards.is_empty()
    }
}

/// Tracks provisions for cycle detection and deduplication.
///
/// Records which (tx_hash, source_shard) pairs we've seen provisions for.
/// This serves two purposes:
///
/// 1. **Cycle detection**: When we receive a provision from shard S for TX_R,
///    we check if we have any local TXs that need provisions from S's shard.
///    If so, and S's TX needs provisions from us, we have a bidirectional cycle.
///
/// 2. **Deduplication**: We only process the first provision from each (tx, shard)
///    pair for cycle detection purposes. Subsequent provisions are for quorum
///    counting but don't trigger additional cycle checks.
#[derive(Debug, Default)]
pub struct ProvisionTracker {
    /// (tx_hash, source_shard) pairs we've seen.
    /// Only stores first provision per (tx, shard) pair.
    seen: HashSet<(Hash, ShardGroupId)>,
}

impl ProvisionTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a provision. Returns true if this is the first from this (tx, shard).
    ///
    /// The caller should only perform cycle detection if this returns true.
    pub fn add(&mut self, tx_hash: Hash, source_shard: ShardGroupId) -> bool {
        self.seen.insert((tx_hash, source_shard))
    }

    /// Get all TX hashes that have provisions from a specific shard.
    ///
    /// Returns the set of transactions that have received provisions from
    /// the given source shard. Used for cycle detection.
    pub fn txs_with_provision_from(&self, source_shard: ShardGroupId) -> Vec<Hash> {
        self.seen
            .iter()
            .filter(|(_, s)| *s == source_shard)
            .map(|(h, _)| *h)
            .collect()
    }

    /// Remove all provisions for a transaction.
    ///
    /// Called when a transaction is completed, deferred, or aborted.
    pub fn remove_tx(&mut self, tx_hash: &Hash) {
        self.seen.retain(|(h, _)| h != tx_hash);
    }

    /// Check if we've seen any provision for a transaction from a specific shard.
    pub fn has_provision(&self, tx_hash: Hash, source_shard: ShardGroupId) -> bool {
        self.seen.contains(&(tx_hash, source_shard))
    }

    /// Get the number of (tx, shard) pairs being tracked.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Check if the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_committed_tracker_basic() {
        let mut tracker = CommittedCrossShardTracker::new();

        let tx1 = Hash::from_bytes(b"tx1");
        let tx2 = Hash::from_bytes(b"tx2");
        let shard0 = ShardGroupId(0);
        let shard1 = ShardGroupId(1);
        let shard2 = ShardGroupId(2);

        // tx1 needs provisions from shard0 and shard1
        tracker.add(tx1, [shard0, shard1].into_iter().collect());

        // tx2 needs provisions from shard1 and shard2
        tracker.add(tx2, [shard1, shard2].into_iter().collect());

        // Check forward lookups
        assert!(tracker.contains(&tx1));
        assert!(tracker.contains(&tx2));
        assert_eq!(
            tracker.shards_for_tx(&tx1),
            Some(&[shard0, shard1].into_iter().collect())
        );

        // Check reverse lookups
        assert_eq!(
            tracker.txs_needing_shard(shard0),
            Some(&[tx1].into_iter().collect())
        );
        assert_eq!(
            tracker.txs_needing_shard(shard1),
            Some(&[tx1, tx2].into_iter().collect())
        );
        assert_eq!(
            tracker.txs_needing_shard(shard2),
            Some(&[tx2].into_iter().collect())
        );

        // Remove tx1
        tracker.remove(&tx1);
        assert!(!tracker.contains(&tx1));
        assert!(tracker.txs_needing_shard(shard0).is_none());
        assert_eq!(
            tracker.txs_needing_shard(shard1),
            Some(&[tx2].into_iter().collect())
        );
    }

    #[test]
    fn test_provision_tracker_basic() {
        let mut tracker = ProvisionTracker::new();

        let tx1 = Hash::from_bytes(b"tx1");
        let tx2 = Hash::from_bytes(b"tx2");
        let shard0 = ShardGroupId(0);
        let shard1 = ShardGroupId(1);

        // First provision from shard0 for tx1 returns true
        assert!(tracker.add(tx1, shard0));

        // Second provision from shard0 for tx1 returns false (already seen)
        assert!(!tracker.add(tx1, shard0));

        // First provision from shard1 for tx1 returns true (different shard)
        assert!(tracker.add(tx1, shard1));

        // First provision from shard0 for tx2 returns true (different tx)
        assert!(tracker.add(tx2, shard0));

        // Check lookups
        let from_shard0 = tracker.txs_with_provision_from(shard0);
        assert!(from_shard0.contains(&tx1));
        assert!(from_shard0.contains(&tx2));

        // Remove tx1
        tracker.remove_tx(&tx1);
        assert!(!tracker.has_provision(tx1, shard0));
        assert!(tracker.has_provision(tx2, shard0));
    }
}
