//! `ShardChainWriter` implementation for `SimShardStorage`.

use std::collections::hash_map::Entry;
use std::sync::Arc;

use hyperscale_storage::lock_recover::{read_or_recover, write_or_recover};
use hyperscale_storage::tree::{
    OverlayTreeReader, jmt_parent_height, noop_jmt_snapshot, put_at_version,
};
use hyperscale_storage::{
    JmtSnapshot, ParentAnchor, ShardChainWriter, SubstateStore, committed_tx_cells,
    covers_strictly_more, merge_writes_from_receipts, widest_tick_copies, with_sweep,
};
use hyperscale_types::{
    BeaconWitnessCommit, Block, BlockHeight, CertifiedBlock, Finalization, PreparedCommit,
    QuorumCertificate, SettledWrites, StateRoot, StoredReceipt, SubstateKey, SyncHint, Verifiable,
    Verified,
};

use super::core::SimShardStorage;
use super::state::{ConsensusState, apply_state_writes, apply_writes};

impl ShardChainWriter for SimShardStorage {
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
        let receipts: Vec<StoredReceipt> = finalizations
            .iter()
            .flat_map(|fw| fw.receipts().iter().cloned())
            .collect();
        let settling: Vec<StoredReceipt> = finalizations
            .iter()
            .flat_map(|fw| fw.settling_receipts())
            .collect();

        // Nothing to write → state root is unchanged. Build a no-op
        // JmtSnapshot directly, avoiding put_at_version which would fail
        // if the parent's tree nodes aren't in the store yet. A block's
        // sweep and its committed cells are writes like any other, so a
        // block that removes or creates something is not one of these
        // however few receipts it carries.
        if receipts.is_empty() && creations.is_empty() && removals.is_empty() {
            let s = read_or_recover(&self.state);
            let snapshot = Arc::new(noop_jmt_snapshot(
                &s.tree_store,
                parent.pending,
                parent.state_root,
                parent.height,
                block_height,
            ));
            drop(s);
            let prepared = build_prepared_commit(
                Arc::clone(self),
                Arc::clone(&snapshot),
                SettledWrites::default(),
                Vec::new(),
            );
            return (parent.state_root, snapshot, prepared);
        }

        // Read lock: compute speculative JMT root.
        let s = read_or_recover(&self.state);

        let parent_version =
            jmt_parent_height(parent.height, parent.state_root).map(BlockHeight::inner);

        // One resolution, feeding both the tree and the substate store —
        // they commit the same values or they disagree about state. It
        // happens here rather than per receipt because a receipt says
        // what it moved, and two receipts moving one cell compose only
        // once something has said what they moved from.
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

        let (result_root, collected) = if parent.pending.is_empty() {
            put_at_version(
                &s.tree_store,
                parent_version,
                block_height.inner(),
                &settled,
            )
        } else {
            let overlay = OverlayTreeReader::new(&s.tree_store, parent.pending);
            put_at_version(&overlay, parent_version, block_height.inner(), &settled)
        };

        let snapshot = Arc::new(JmtSnapshot::from_collected_writes(
            collected,
            settled.clone(),
            parent.state_root,
            parent.height,
            result_root,
            block_height,
        ));

        drop(s); // Release read lock

        let prepared =
            build_prepared_commit(Arc::clone(self), Arc::clone(&snapshot), settled, receipts);

        (result_root, snapshot, prepared)
    }

    fn commit_block(
        &self,
        certified: &Arc<Verified<CertifiedBlock>>,
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
        let creations = committed_tx_cells(
            block.header().shard_id(),
            block.transactions().iter().map(|tx| tx.as_unverified()),
        );
        let merged_writes = with_sweep(
            merge_writes_from_receipts(
                &block
                    .certificates()
                    .iter()
                    .flat_map(|fw| fw.settling_receipts())
                    .collect::<Vec<_>>(),
                &self.snapshot(),
            ),
            &creations,
            removals,
        );
        self.append_beacon_witnesses(witness);
        self.commit_block_inner(&merged_writes, block, qc, &receipts)
    }
}

/// Build the closure that performs the in-memory atomic block commit.
///
/// Captures the storage handle, the JMT snapshot, the merged updates,
/// and the receipts. At invocation time the closure receives the
/// `Verified<CertifiedBlock>` and witness, applies the snapshot/state/
/// consensus changes, and returns the resulting state root.
#[allow(clippy::significant_drop_tightening)] // state write held across snapshot + substate apply by design
fn build_prepared_commit(
    storage: Arc<SimShardStorage>,
    snapshot: Arc<JmtSnapshot>,
    merged_writes: SettledWrites,
    receipts: Vec<StoredReceipt>,
) -> PreparedCommit {
    Box::new(
        move |_sync_hint: SyncHint,
              certified: &Arc<Verified<CertifiedBlock>>,
              witness: &BeaconWitnessCommit|
              -> StateRoot {
            storage.append_beacon_witnesses(witness);

            let block_height_u64 = snapshot.new_height.inner();
            let result_root = snapshot.result_root;

            {
                let mut s = write_or_recover(&storage.state);
                s.apply_jmt_snapshot(&snapshot);
                apply_writes(
                    &mut s,
                    &merged_writes,
                    block_height_u64,
                    /* write_history */ true,
                );
            }

            let block = certified.block();
            let qc = certified.qc_verified();

            // SAFETY: synthetic in-memory commit wrapper; the certified
            // value is already verified upstream and we're just copying
            // its inner shape into the consensus map.
            let unwrapped = CertifiedBlock::new_unchecked(block.clone().into_sealed(), qc.clone());

            let mut c = write_or_recover(&storage.consensus);
            for tx in block.transactions().iter() {
                c.transactions.insert(tx.hash(), (***tx).clone());
            }
            c.blocks.insert(block.height(), unwrapped);
            for fw in block.certificates().iter() {
                let tick_id = *fw.tick_id();
                c.certificates.insert(fw.receipt_hash(), fw.attestation());
                c.finalizations_by_height
                    .entry(tick_id.block_height())
                    .or_default()
                    .push(tick_id);
            }
            c.record_provisions(block, storage.jmt_history_length);
            c.insert_receipts(&receipts);
            record_execution_certs(&mut c, block);
            c.committed_height = block.height();
            c.committed_hash = Some(block.hash());
            c.committed_qc = Some(qc.as_ref().clone());
            c.prune_receipts(block.height());

            result_root
        },
    )
}

impl SimShardStorage {
    /// Fold a block's beacon-witness commit into the in-memory map:
    /// append `witness.leaves` and drop entries below a carried
    /// retention floor. Lives next to the commit paths so both
    /// prepared-commit and from-scratch commits share one entry point.
    fn append_beacon_witnesses(&self, witness: &BeaconWitnessCommit) {
        if witness.leaves.is_empty() && witness.prune_persisted_below.is_none() {
            return;
        }
        let mut c = write_or_recover(&self.consensus);
        if let Some(floor) = witness.prune_persisted_below {
            c.beacon_witnesses = c.beacon_witnesses.split_off(&floor.inner());
        }
        let start = witness.starting_leaf_index.inner();
        for (offset, payload) in witness.leaves.iter().enumerate() {
            c.beacon_witnesses
                .insert(start + offset as u64, payload.clone());
        }
    }
}

impl SimShardStorage {
    /// Internal commit path used by `commit_block` (sync blocks without a `PreparedCommit`).
    fn commit_block_inner(
        &self,
        merged_writes: &SettledWrites,
        block: &Block,
        qc: &Verified<QuorumCertificate>,
        receipts: &[StoredReceipt],
    ) -> StateRoot {
        let block_height = block.height();
        let mut s = write_or_recover(&self.state);

        // A genesis commit re-records the height the install already wrote
        // (the chain's genesis height — 0 only for chains born at network
        // genesis); every other block advances the version by exactly one.
        assert!(
            block_height == s.current_block_height + 1
                || (block.is_genesis() && block_height == s.current_block_height),
            "commit_block: block_height ({block_height}) must be exactly current_version + 1 ({})",
            s.current_block_height
        );

        let new_root = apply_state_writes(&mut s, merged_writes, block_height);

        drop(s);

        // Store block + certificate + consensus state atomically.
        {
            let mut c = write_or_recover(&self.consensus);
            for tx in block.transactions().iter() {
                c.transactions.insert(tx.hash(), (***tx).clone());
            }
            // SAFETY: sync-path commit; certified value is already
            // verified upstream.
            c.blocks.insert(
                block.height(),
                CertifiedBlock::new_unchecked(block.clone().into_sealed(), qc.clone()),
            );
            for fw in block.certificates().iter() {
                let tick_id = *fw.tick_id();
                c.certificates.insert(fw.receipt_hash(), fw.attestation());
                c.finalizations_by_height
                    .entry(tick_id.block_height())
                    .or_default()
                    .push(tick_id);
            }
            c.record_provisions(block, self.jmt_history_length);
            // Store receipts atomically with block commit.
            c.insert_receipts(receipts);
            // Store execution certificates (extracted from finalizations) atomically.
            record_execution_certs(&mut c, block);
            c.committed_height = block.height();
            c.committed_hash = Some(block.hash());
            c.committed_qc = Some(qc.as_ref().clone());
            c.prune_receipts(block.height());
        }

        new_root
    }
}

/// Fold a block's execution certificates into the consensus map, keeping
/// the widest copy of each tick and indexing the transactions that copy
/// attests.
///
/// Only an accepted copy of this shard's own certificate is indexed. A
/// settled cross-shard transaction lands here under both sides'
/// certificates, and the index answers "what did THIS shard attest for
/// the transaction" — the question a counterpart's fallback fetch asks
/// this shard. Letting a remote copy win the single slot serves the
/// requester its own certificate back, which it rightly refuses as
/// unsolicited, and the fetch loops forever.
fn record_execution_certs(consensus: &mut ConsensusState, block: &Block) {
    let local_shard = block
        .certificates()
        .first()
        .map(|finalization| finalization.tick_id().shard_id());
    for cert in widest_tick_copies(block).into_values() {
        match consensus.execution_certs.entry(*cert.tick_id()) {
            Entry::Occupied(mut held) => {
                if !covers_strictly_more(cert, held.get()) {
                    continue;
                }
                held.insert(cert.clone());
            }
            Entry::Vacant(slot) => {
                slot.insert(cert.clone());
            }
        }
        if Some(cert.tick_id().shard_id()) != local_shard {
            continue;
        }
        for outcome in cert.tx_outcomes() {
            consensus
                .tx_cert_index
                .insert(outcome.tx_hash(), *cert.tick_id());
        }
    }
}
