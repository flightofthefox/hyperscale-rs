//! `SubstateStore` implementation for `SimShardStorage`.

use std::sync::Arc;

use hyperscale_jmt::{NibblePath, Node as JmtNode, NodeKey as JmtNodeKey, TreeReader};
use hyperscale_storage::lock_recover::read_or_recover;
use hyperscale_storage::{
    PackageArtifactStore, SubstateStore, Substates, SweepIndex, VersionedStore, sweepable_expiry,
};
use hyperscale_types::{
    Address, BlockHeight, DeclaredRange, Hash, LocalKey, SWEEP_BUCKET_BYTES, StateRoot,
    SubstateKey, SweepBucket, SweepFrontier,
};

use super::core::SimShardStorage;
use super::snapshot::SimSnapshot;

impl SubstateStore for SimShardStorage {
    type Snapshot<'a> = SimSnapshot;

    fn snapshot(&self) -> Self::Snapshot<'_> {
        // Default height = current committed tip. Equivalent to reading
        // latest state but uniform snapshot type across all call sites.
        self.snapshot_at(self.jmt_height())
    }

    fn jmt_height(&self) -> BlockHeight {
        read_or_recover(&self.state).current_block_height
    }

    fn state_root(&self) -> StateRoot {
        read_or_recover(&self.state).current_root_hash
    }

    fn get_substate_at_height(
        &self,
        key: SubstateKey,
        block_height: BlockHeight,
    ) -> Option<Option<Vec<u8>>> {
        use hyperscale_storage::Substates;
        let current_version = read_or_recover(&self.state).current_block_height.inner();
        if block_height.inner() > current_version {
            return None;
        }
        let floor = current_version.saturating_sub(self.jmt_history_length);
        if block_height.inner() < floor {
            return None;
        }
        Some(self.snapshot_at(block_height).cell(key))
    }

    fn get_entries_at_height(
        &self,
        range: DeclaredRange,
        block_height: BlockHeight,
    ) -> Option<Vec<(u128, Vec<u8>)>> {
        let current_version = read_or_recover(&self.state).current_block_height.inner();
        if block_height.inner() > current_version {
            return None;
        }
        let floor = current_version.saturating_sub(self.jmt_history_length);
        if block_height.inner() < floor {
            return None;
        }
        Some(self.snapshot_at(block_height).entries_in_range(
            range.owner,
            range.collection,
            range.lo,
            range.hi,
            range.cap as usize,
        ))
    }
}

impl VersionedStore for SimShardStorage {
    fn snapshot_at(&self, height: BlockHeight) -> Self::Snapshot<'_> {
        // Retention invariant: see `RocksDbShardStorage::snapshot_at` for the
        // full reasoning. Below the floor we can't serve historical
        // reads; hitting this is a DA-assumption bug in the caller.
        let guard = read_or_recover(&self.state);
        let current_version = guard.current_block_height.inner();
        let floor = current_version.saturating_sub(self.jmt_history_length);
        assert!(
            height.inner() >= floor,
            "snapshot_at({height}) below retention floor {floor} \
             (current_version={current_version}, jmt_history_length={}) — \
             Shard consensus + DA invariant broken; caller must anchor within retention",
            self.jmt_history_length,
        );
        // Clone state + state-history for snapshot isolation. Memory
        // snapshots are point-in-time copies — they don't observe later
        // mutations of the backing store.
        SimSnapshot {
            current_state: guard.current_state.clone(),
            state_history: guard.state_history.clone(),
            current_entries: guard.current_entries.clone(),
            entries_history: guard.entries_history.clone(),
            version: height.inner(),
            current_version,
        }
    }

    fn substate_bytes_at(&self, height: BlockHeight) -> Option<u64> {
        read_or_recover(&self.state)
            .substate_bytes
            .get(&height.inner())
            .copied()
    }
}

impl TreeReader for SimShardStorage {
    fn get_node(&self, key: &JmtNodeKey) -> Option<Arc<JmtNode>> {
        read_or_recover(&self.state).tree_store.get_node(key)
    }

    fn get_root_key(&self, version: u64) -> Option<JmtNodeKey> {
        read_or_recover(&self.state)
            .tree_store
            .get_root_key(version)
    }

    fn root_path(&self) -> NibblePath {
        read_or_recover(&self.state).tree_store.root_path()
    }
}

impl SweepIndex for SimShardStorage {
    fn sweep_candidates(
        &self,
        frontier: SweepFrontier,
        ceiling: SweepFrontier,
        limit: usize,
    ) -> Vec<(SubstateKey, u64)> {
        if limit == 0 || frontier >= ceiling {
            return Vec::new();
        }
        let state = read_or_recover(&self.state);
        let mut found = Vec::new();
        // The index rows in (bucket, owner) order, then that pair's
        // leaves in local order — the two walks composed are already
        // sweep order, exactly as the RocksDB backend walks them.
        for (&(bucket, owner), _) in state
            .sweep_index
            .range((SweepBucket(frontier.bucket().0), Address::MIN)..)
        {
            if bucket >= ceiling.bucket() || found.len() >= limit {
                break;
            }
            let (lo, hi) = leaf_bucket_span(owner, bucket);
            for (&key, value) in state.current_state.range(lo..=hi) {
                if found.len() >= limit {
                    break;
                }
                if SweepFrontier::of_leaf(key) > frontier
                    && let Some(expiry) = sweepable_expiry(key, value)
                {
                    found.push((key, expiry));
                }
            }
        }
        found
    }
}

/// The lowest and highest keys one owner's cells in one bucket can take,
/// the `BTreeMap` spelling of the `RocksDB` backend's raw-key bounds.
fn leaf_bucket_span(owner: Address, bucket: SweepBucket) -> (SubstateKey, SubstateKey) {
    let bounded = |fill: u8| {
        let mut local = [fill; 16];
        local[..SWEEP_BUCKET_BYTES].copy_from_slice(&bucket.to_bytes());
        SubstateKey {
            owner,
            local: LocalKey(local),
        }
    };
    (bounded(0x00), bounded(0xFF))
}

impl PackageArtifactStore for SimShardStorage {
    fn package_artifacts(&self) -> Vec<Vec<u8>> {
        read_or_recover(&self.state)
            .package_artifacts
            .values()
            .cloned()
            .collect()
    }

    fn package_artifact(&self, package: Hash) -> Option<Vec<u8>> {
        read_or_recover(&self.state)
            .package_artifacts
            .get(&package)
            .cloned()
    }
}
