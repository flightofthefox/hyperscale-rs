//! `ShardChainWriter` implementation for `RocksDbShardStorage`.

use std::sync::Arc;

use hyperscale_storage::tree::{
    OverlayTreeReader, jmt_parent_height, noop_jmt_snapshot, put_at_version,
};
use hyperscale_storage::{
    JmtSnapshot, ParentAnchor, ShardChainWriter, SubstateStore, block_settled_writes,
    merge_writes_from_receipts, with_sweep,
};
use hyperscale_types::{
    BeaconWitnessCommit, Block, BlockHeight, CertifiedBlock, Finalization, PreparedCommit,
    QuorumCertificate, SettledWrites, StateRoot, StoredReceipt, SubstateKey, SyncHint, Verifiable,
    Verified,
};
use rocksdb::{WriteBatch, WriteOptions};

use super::column_families::{ConsensusReceiptsCf, ExecutionMetadataCf};
use super::core::RocksDbShardStorage;
use super::execution_certs::append_block_certs_to_batch;
use super::jmt_snapshot_store::SnapshotTreeStore;
use super::receipts::add_receipt_to_batch;
use crate::typed_cf::TypedCf;

impl ShardChainWriter for RocksDbShardStorage {
    fn prepare_block_commit(
        self: &Arc<Self>,
        parent: ParentAnchor<'_>,
        finalizations: &[Arc<Verifiable<Finalization>>],
        creations: &[(SubstateKey, Vec<u8>)],
        removals: &[SubstateKey],
        block_height: BlockHeight,
    ) -> (StateRoot, Arc<JmtSnapshot>, PreparedCommit) {
        // Everything the ticks carried, for storage; only what they
        // decided reaches state.
        let receipts: Vec<&StoredReceipt> = finalizations
            .iter()
            .flat_map(|fw| fw.receipts().iter())
            .collect();
        let settling: Vec<StoredReceipt> = finalizations
            .iter()
            .flat_map(|fw| fw.settling_receipts())
            .collect();

        // Nothing to write → state root is unchanged. Build a no-op
        // JmtSnapshot directly, avoiding put_at_version which would fail
        // if the parent's tree nodes aren't in the store yet (e.g.,
        // proposer just exited sync and BlockPersisted hasn't fired).
        // A block's sweep and its committed cells are writes like any
        // other, so a block that removes or creates something is not one
        // of these however few receipts it carries.
        if receipts.is_empty() && creations.is_empty() && removals.is_empty() {
            let jmt_snapshot = Arc::new(noop_jmt_snapshot(
                &SnapshotTreeStore::new(&self.db, self.root_path.clone()),
                parent.pending,
                parent.state_root,
                parent.height,
                block_height,
            ));
            let prepared = build_prepared_commit(
                Arc::clone(self),
                WriteBatch::default(),
                Arc::clone(&jmt_snapshot),
            );
            return (parent.state_root, jmt_snapshot, prepared);
        }

        let snapshot_store = SnapshotTreeStore::new(&self.db, self.root_path.clone());
        let parent_version =
            jmt_parent_height(parent.height, parent.state_root).map(BlockHeight::inner);

        // Collect per-receipt writes references — no merge needed.
        // State locking guarantees no key conflicts between receipts, so
        // put_at_version can flatten them directly into JMT work items.
        // One resolution, feeding both the tree and the substate batch —
        // they commit the same values or they disagree about state. A
        // receipt says what it moved, and two receipts moving one cell
        // compose only once something has said what they moved from.
        // The type says the baseline was fixed when it was made; which
        // block it was fixed at is this caller's to check, and a movement
        // resolved against any other is as wrong as one resolved live.
        assert_eq!(
            parent.state.anchor(),
            parent.height,
            "a movement's baseline is anchored at the wrong height",
        );
        let settled = with_sweep(
            merge_writes_from_receipts(&settling, parent.state),
            creations,
            removals,
        );

        let (computed_root, collected) = if parent.pending.is_empty() {
            put_at_version(
                &snapshot_store,
                parent_version,
                block_height.inner(),
                &settled,
            )
        } else {
            let overlay = OverlayTreeReader::new(&snapshot_store, parent.pending);
            put_at_version(&overlay, parent_version, block_height.inner(), &settled)
        };

        let jmt_snapshot = Arc::new(JmtSnapshot::from_collected_writes(
            collected,
            settled.clone(),
            parent.state_root,
            parent.height,
            computed_root,
            block_height,
        ));

        // Merge writes for the substate WriteBatch (off the state_root critical path).
        // Pre-build substate + receipt writes into a WriteBatch for efficient commit.
        let mut write_batch = self.build_substate_write_batch(
            &settled,
            block_height.inner(),
            /* write_history */ true,
            parent.base_reads,
            parent.pending,
        );

        let cf = self.cf();
        let consensus_cf = ConsensusReceiptsCf::handle(&cf);
        let metadata_cf = ExecutionMetadataCf::handle(&cf);
        for receipt in &receipts {
            add_receipt_to_batch(&mut write_batch, consensus_cf, metadata_cf, receipt);
        }

        let prepared =
            build_prepared_commit(Arc::clone(self), write_batch, Arc::clone(&jmt_snapshot));

        (computed_root, jmt_snapshot, prepared)
    }

    fn commit_block(
        &self,
        certified: &Arc<Verified<CertifiedBlock>>,
        creations: &[(SubstateKey, Vec<u8>)],
        removals: &[SubstateKey],
        witness: &BeaconWitnessCommit,
    ) -> StateRoot {
        let block = certified.block();
        let qc = certified.qc_verified();
        let receipts: Vec<StoredReceipt> = block
            .certificates()
            .iter()
            .flat_map(|fw| fw.receipts().iter().cloned())
            .collect();
        // Under the lock, and off a snapshot rather than the live store:
        // the baseline a movement resolves against has to be the state
        // this block is about to land on, and both a concurrent commit
        // and this node's own persistence depth move a live read out
        // from under it.
        let _commit_guard = self.commit_lock.lock().unwrap();
        let merged_writes = block_settled_writes(block, &self.snapshot(), creations, removals);
        self.commit_block_inner_locked(&merged_writes, block, qc, &receipts, witness)
    }
}

/// Build the closure that performs the atomic block commit.
///
/// Captures the storage handle, the pre-built `WriteBatch`, and the JMT
/// snapshot. At invocation time the closure receives the
/// `Verified<CertifiedBlock>` and beacon-witness commit, folds them into
/// the batch, and writes — with a fallback through
/// [`RocksDbShardStorage::commit_block_inner_locked`] if a concurrent sync
/// commit advanced past us.
fn build_prepared_commit(
    storage: Arc<RocksDbShardStorage>,
    write_batch: WriteBatch,
    jmt_snapshot: Arc<JmtSnapshot>,
) -> PreparedCommit {
    Box::new(
        move |sync_hint: SyncHint,
              certified: &Arc<Verified<CertifiedBlock>>,
              witness: &BeaconWitnessCommit|
              -> StateRoot {
            let result_root = jmt_snapshot.result_root;
            let mut write_batch = write_batch;

            let block = certified.block();
            let qc = certified.qc_verified();

            let floor = storage.advance_retention_floor(
                &mut write_batch,
                block.height().inner(),
                qc.weighted_timestamp(),
            );
            storage.append_block_to_batch(
                &mut write_batch,
                block,
                qc,
                witness.leaf_count_at_block_end,
                floor,
            );
            storage.append_beacon_witnesses_to_batch(&mut write_batch, witness);

            // The block's execution certificates append inside
            // `try_apply_prepared_commit`, which holds `commit_lock`
            // across the read their write depends on.
            let applied = storage.try_apply_prepared_commit(
                write_batch,
                &jmt_snapshot,
                block,
                qc,
                sync_hint.is_flush_now(),
            );
            if applied {
                return result_root;
            }

            // The fast path refused: the store advanced since the commit
            // was prepared. A sync commit that landed this very block is
            // the one benign cause, and the block is then already in.
            // Anything else — the store at a height below this block
            // with a base the prepared batch was not built on — is a
            // divergence between what the chain committed and what this
            // node prepared, and a fresh fold here would commit a root
            // no verifier compared with the header's.
            let _guard = storage.commit_lock.lock().unwrap();
            let (current_version, _) =
                SnapshotTreeStore::new(&storage.db, storage.root_path.clone()).read_jmt_metadata();
            assert!(
                block.height().inner() <= current_version,
                "BFT CRITICAL: prepared commit for height {} refused with the store at {}",
                block.height().inner(),
                current_version,
            );
            tracing::debug!(
                height = block.height().inner(),
                current_version,
                "PreparedCommit stale — block already committed, skipping"
            );
            result_root
        },
    )
}

impl RocksDbShardStorage {
    /// Internal commit path used by `commit_block` (sync blocks without a `PreparedCommit`).
    ///
    /// The caller MUST hold `self.commit_lock`. The callers that do are
    /// [`Self::commit_block`] and the fallback branch inside the closure
    /// returned by `build_prepared_commit`; the latter holds the lock
    /// across its own `read_jmt_metadata` so the contiguity check and
    /// the commit see the same `base_version`.
    pub(crate) fn commit_block_inner_locked(
        &self,
        merged_writes: &SettledWrites,
        block: &Block,
        qc: &Verified<QuorumCertificate>,
        receipts: &[StoredReceipt],
        witness: &BeaconWitnessCommit,
    ) -> StateRoot {
        let block_height = block.height().inner();

        let snapshot_store = SnapshotTreeStore::new(&self.db, self.root_path.clone());
        let (base_version, base_root) = snapshot_store.read_jmt_metadata();

        // A genesis commit re-records the height the install already wrote
        // (the chain's genesis height — 0 only for chains born at network
        // genesis); every other block advances the version by exactly one.
        assert!(
            block_height == base_version + 1
                || (block.is_genesis() && block_height == base_version),
            "commit_block: block_height ({block_height}) must be exactly current_version + 1 ({base_version})"
        );

        // Sync path commits at the persisted tip under `commit_lock`: no
        // view, no base-read cache, no pending ancestors — every prior
        // comes from one multi_get_cf.
        let mut batch = self.build_substate_write_batch(
            merged_writes,
            block_height,
            /* write_history */ true,
            /* base_reads */ None,
            /* pending */ &[],
        );

        let floor = self.advance_retention_floor(&mut batch, block_height, qc.weighted_timestamp());
        self.append_block_to_batch(
            &mut batch,
            block,
            qc,
            witness.leaf_count_at_block_end,
            floor,
        );
        self.append_beacon_witnesses_to_batch(&mut batch, witness);

        append_block_certs_to_batch(self, &mut batch, block);

        let cf = self.cf();
        let consensus_cf = ConsensusReceiptsCf::handle(&cf);
        let metadata_cf = ExecutionMetadataCf::handle(&cf);
        for receipt in receipts {
            add_receipt_to_batch(&mut batch, consensus_cf, metadata_cf, receipt);
        }

        // Compute JMT update.
        let parent_version =
            jmt_parent_height(BlockHeight::new(base_version), base_root).map(BlockHeight::inner);
        let (new_root, collected) =
            put_at_version(&snapshot_store, parent_version, block_height, merged_writes);
        let jmt_snapshot = JmtSnapshot::from_collected_writes(
            collected,
            merged_writes.clone(),
            base_root,
            BlockHeight::new(base_version),
            new_root,
            BlockHeight::new(block_height),
        );
        self.append_jmt_to_batch(&mut batch, &jmt_snapshot, block_height);

        // Fold consensus metadata into the same batch for crash-safe atomicity.
        Self::append_consensus_to_batch(&mut batch, block, qc);

        // Single atomic write with sync — one fsync instead of N.
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        self.db.write_opt(batch, &write_opts).expect(
            "BFT SAFETY CRITICAL: block commit failed - node state would diverge from network",
        );

        new_root
    }
}
