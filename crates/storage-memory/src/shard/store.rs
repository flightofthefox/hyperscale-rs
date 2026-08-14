//! `SubstateStore` implementation for `SimShardStorage`.

use std::sync::Arc;

use hyperscale_jmt::{NibblePath, Node as JmtNode, NodeKey as JmtNodeKey, TreeReader};
use hyperscale_storage::lock_recover::read_or_recover;
use hyperscale_storage::{PackageArtifactStore, SubstateStore, Substates, VersionedStore};
use hyperscale_types::{BlockHeight, DeclaredRange, StateRoot, SubstateKey};

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

impl PackageArtifactStore for SimShardStorage {
    fn package_artifacts(&self) -> Vec<Vec<u8>> {
        read_or_recover(&self.state)
            .package_artifacts
            .values()
            .cloned()
            .collect()
    }
}
