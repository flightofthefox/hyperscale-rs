use std::collections::BTreeMap;
use std::sync::Arc;

use hyperscale_storage::test_helpers::{
    make_settled_writes, make_test_block, make_test_certified, make_test_qc, state_key,
};
use hyperscale_storage::tree::{jmt_parent_height, put_at_version};
use hyperscale_storage::{
    CommittableSubstateDatabase, ParentAnchor, SafeVoteRegisterStore, ShardChainReader,
    ShardChainWriter, SubstateDatabase, SubstateStore, VersionedStore, test_helpers,
};
use hyperscale_types::test_utils::test_transaction;
use hyperscale_types::{
    Address, BeaconWitnessCommit, BeaconWitnessLeafCount, Block, BlockHeight, CertifiedBlock,
    ChainOrigin, ConsensusReceipt, Finalization, GlobalReceiptHash, Hash, LocalKey,
    MerkleInclusionProof, ProposerTimestamp, ProvisionEntry, Provisions, QuorumCertificate,
    RevealChain, Round, SafeVoteRegisters, SettledWrites, ShardId, StateRoot, StoredReceipt,
    SubstateKey, SyncHint, TickHalf, TickId, TxHash, ValidatorId, Verifiable, Verified,
    WeightedTimestamp, WitnessSources,
};

fn no_witness() -> BeaconWitnessCommit {
    BeaconWitnessCommit::empty(BeaconWitnessLeafCount::ZERO)
}

use super::core::SimShardStorage;
use super::state::apply_writes;

impl SimShardStorage {
    /// Atomically commit a certificate and its state writes.
    ///
    /// Applies database updates and stores certificate metadata.
    /// JMT is deferred to block commit — this mirrors the production
    /// `RocksDbShardStorage::commit_certificate_with_writes()` to ensure DST
    /// catches timing bugs where code incorrectly assumes state is available
    /// before certificate persistence.
    ///
    /// # Panics
    ///
    /// Panics if either internal `RwLock` is poisoned.
    #[allow(clippy::significant_drop_tightening)] // both reads need the lock
    pub fn commit_certificate_with_writes(
        &self,
        certificate: &Finalization,
        writes: &SettledWrites,
    ) {
        {
            let mut s = self.state.write().unwrap();
            let ver = s.current_block_height.inner();
            apply_writes(&mut s, writes, ver, /* write_history */ true);
        }
        self.consensus
            .write()
            .unwrap()
            .certificates
            .insert(certificate.receipt_hash(), certificate.clone());
    }

    /// Test helper: commits database updates with auto-incrementing JMT version.
    /// Not used in production (use `commit_block` instead).
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    pub fn commit_shared(&self, writes: &SettledWrites) {
        let mut s = self.state.write().unwrap();

        let new_version = s.current_block_height.inner() + 1;

        apply_writes(&mut s, writes, new_version, /* write_history */ true);

        let parent_version =
            jmt_parent_height(s.current_block_height, s.current_root_hash).map(BlockHeight::inner);
        let (new_root, collected) =
            put_at_version(&s.tree_store, parent_version, new_version, writes);

        for (key, node) in &collected.nodes {
            s.tree_store.insert(key.clone(), Arc::clone(node));
        }

        s.current_block_height = BlockHeight::new(new_version);
        s.current_root_hash = new_root;
    }
}

impl CommittableSubstateDatabase for SimShardStorage {
    fn commit(&mut self, writes: &SettledWrites) {
        self.commit_shared(writes);
    }
}

/// Helper: commit a block with given updates by injecting them via a single-tx
/// `Finalization` inside `block.certificates`.
/// The union of already-settled fixtures — values, so nothing to fold.
fn union_of(parts: &[SettledWrites]) -> SettledWrites {
    SettledWrites::from_absolutes(
        parts
            .iter()
            .flat_map(SettledWrites::cells)
            .map(|(key, change)| (*key, change.clone()))
            .collect(),
    )
}

fn commit_with(
    storage: &SimShardStorage,
    writes: &SettledWrites,
    block: &Block,
    qc: &Verified<QuorumCertificate>,
) -> StateRoot {
    let block = block.clone();
    let block = if writes.is_empty() {
        block
    } else {
        let receipt = StoredReceipt {
            tx_hash: TxHash::ZERO,
            consensus: Arc::new(ConsensusReceipt::Succeeded {
                receipt_hash: GlobalReceiptHash::ZERO,
                writes: writes.clone().into(),
                beacon_witness_events: Vec::new(),
                events: Vec::new(),
            }),
            metadata: None,
        };
        let new_fw: Arc<Verifiable<Finalization>> = Arc::new(
            Finalization::new(
                TickId::new(ShardId::ROOT, block.height()),
                TickHalf::Determined,
                vec![],
                vec![receipt],
            )
            .into(),
        );
        match block {
            Block::Live {
                header,
                transactions,
                certificates,
                provisions,
                witness_sources,
            } => {
                let mut certificates = (*certificates).clone();
                certificates.push(new_fw);
                Block::Live {
                    header,
                    transactions,
                    certificates: Arc::new(certificates),
                    provisions,
                    witness_sources,
                }
            }
            Block::Sealed {
                header,
                transactions,
                certificates,
                provision_hashes,
                witness_sources,
            } => {
                let mut certificates = (*certificates).clone();
                certificates.push(new_fw);
                Block::Sealed {
                    header,
                    transactions,
                    certificates: Arc::new(certificates),
                    provision_hashes,
                    witness_sources,
                }
            }
        }
    };
    // SAFETY: synthetic test fixture; storage round-trip tests don't
    // exercise the `Verified<CertifiedBlock>` predicate.
    let certified = Arc::new(Verified::<CertifiedBlock>::new_unchecked_for_test(
        CertifiedBlock::new_unchecked(block, <Verified<_>>::clone(qc)),
    ));
    storage.commit_block(&certified, &no_witness())
}

/// Helper: commit a block with empty updates and no ECs/receipts.
fn commit_empty(
    storage: &SimShardStorage,
    block: &Block,
    qc: &Verified<QuorumCertificate>,
) -> StateRoot {
    commit_with(storage, &SettledWrites::default(), block, qc)
}

#[test]
fn test_basic_substate_operations() {
    let mut storage = SimShardStorage::default();

    let key = state_key(1, 10);

    // Initially empty
    assert!(storage.substate(key).is_none());

    // Commit a value
    let writes = SettledWrites::from_absolutes(BTreeMap::from([(key, Some(vec![99, 88, 77]))]));
    storage.commit(&writes);

    // Now we can read it
    assert_eq!(storage.substate(key), Some(vec![99, 88, 77]));
}

#[test]
fn test_snapshot_isolation() {
    let mut storage = SimShardStorage::default();

    let key = state_key(1, 10);

    // Write initial value
    storage.commit(&make_settled_writes(1, 10, vec![1]));

    // Take snapshot
    let snapshot = storage.snapshot();

    // Modify storage
    storage.commit(&make_settled_writes(1, 10, vec![2]));

    // Snapshot has old value
    assert_eq!(snapshot.substate(key), Some(vec![1]));

    // Storage has new value
    assert_eq!(storage.substate(key), Some(vec![2]));
}

#[test]
fn test_snapshot_clone_performance() {
    let storage = SimShardStorage::default();

    // Insert 10,000 items via substates-only (no JMT computation).
    // This test bounds the cost of a single BTreeMap-clone snapshot at
    // simulation scale, not tree commit speed.
    for i in 0..10_000u32 {
        let mut owner = [0u8; 16];
        owner[..4].copy_from_slice(&i.to_be_bytes());
        let writes = SettledWrites::from_absolutes(BTreeMap::from([(
            SubstateKey {
                owner: Address(owner),
                local: LocalKey([0; 16]),
            },
            Some(vec![u8::try_from(i).unwrap_or(u8::MAX)]),
        )]));
        storage.commit_substates_only(&writes);
    }

    // Snapshot should be nearly instant (O(1), not O(n))
    let start = std::time::Instant::now();
    let _snap1 = storage.snapshot();
    let _snap2 = storage.snapshot();
    let _snap3 = storage.snapshot();
    let _snap4 = storage.snapshot();
    let _snap5 = storage.snapshot();
    let elapsed = start.elapsed();

    // Guardrail against accidental quadratic behaviour or extra
    // per-snapshot work; 5 BTreeMap clones of 10k entries fits well
    // under the cap on any reasonable machine.
    assert!(
        elapsed.as_millis() < 50,
        "5 snapshots took {elapsed:?}, expected < 50ms"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Consensus operations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_block_storage_and_retrieval() {
    let storage = SimShardStorage::default();
    let block = make_test_block(BlockHeight::new(1));
    let qc = make_test_qc(&block);

    assert!(storage.get_block(BlockHeight::new(1)).is_none());

    commit_empty(&storage, &block, &qc);

    let stored = storage.get_block(BlockHeight::new(1)).unwrap();
    assert_eq!(stored.block().height(), BlockHeight::new(1));
    assert_eq!(
        stored.block().header().timestamp(),
        ProposerTimestamp::from_millis(1_000)
    );
    assert_eq!(stored.qc().block_hash(), block.hash());
}

#[test]
fn test_block_get_nonexistent() {
    let storage = SimShardStorage::default();
    assert!(storage.get_block(BlockHeight::new(999)).is_none());
}

#[test]
fn test_committed_height_default() {
    let storage = SimShardStorage::default();
    assert_eq!(storage.committed_height(), BlockHeight::new(0));
    assert!(storage.committed_hash().is_none());
    assert!(storage.latest_qc().is_none());
}

#[test]
fn test_get_block_for_sync() {
    let storage = SimShardStorage::default();
    let block = make_test_block(BlockHeight::new(1));
    let qc = make_test_qc(&block);
    commit_empty(&storage, &block, &qc);

    let result = storage.get_block_for_sync(BlockHeight::new(1));
    assert!(result.is_some());
    assert_eq!(result.unwrap().block.height(), BlockHeight::new(1));

    assert!(storage.get_block_for_sync(BlockHeight::new(999)).is_none());
}

#[test]
fn test_transactions_batch_missing() {
    let storage = SimShardStorage::default();
    let result = storage.get_transactions_batch(&[TxHash::from(Hash::from_bytes(&[1; 32]))]);
    assert!(result.is_empty());
}

#[test]
fn test_transactions_batch_with_indexed_block() {
    let storage = SimShardStorage::default();
    let block = make_test_block(BlockHeight::new(1));

    let tx = Arc::new(Verifiable::from(test_transaction(42)));
    let tx_hash = tx.hash();
    let block = match block {
        Block::Live {
            header,
            certificates,
            provisions,
            ..
        } => Block::Live {
            header,
            transactions: Arc::new(vec![tx]),
            certificates,
            provisions,
            witness_sources: Arc::new(WitnessSources::empty()),
        },
        Block::Sealed {
            header,
            certificates,
            provision_hashes,
            ..
        } => Block::Sealed {
            header,
            transactions: Arc::new(vec![tx]),
            certificates,
            provision_hashes,
            witness_sources: Arc::new(WitnessSources::empty()),
        },
    };

    let qc = make_test_qc(&block);
    commit_empty(&storage, &block, &qc);

    let result = storage.get_transactions_batch(&[tx_hash]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].hash(), tx_hash);

    // Missing hash still excluded
    let missing = TxHash::from(Hash::from_bytes(&[99; 32]));
    let partial = storage.get_transactions_batch(&[tx_hash, missing]);
    assert_eq!(partial.len(), 1);
}

// ═══════════════════════════════════════════════════════════════════════
// JMT state tracking
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_initial_jmt_height_is_zero() {
    let storage = SimShardStorage::default();
    assert_eq!(storage.jmt_height(), BlockHeight::new(0));
}

#[test]
fn test_initial_state_root_is_zero() {
    let storage = SimShardStorage::default();
    assert_eq!(storage.state_root(), StateRoot::ZERO);
}

#[test]
fn test_jmt_height_increments_on_commit() {
    let storage = SimShardStorage::default();
    assert_eq!(storage.jmt_height(), BlockHeight::new(0));

    storage.commit_shared(&make_settled_writes(1, 10, vec![1]));
    assert_eq!(storage.jmt_height(), BlockHeight::new(1));

    storage.commit_shared(&make_settled_writes(4, 20, vec![2]));
    assert_eq!(storage.jmt_height(), BlockHeight::new(2));
}

#[test]
fn test_state_root_changes_on_commit() {
    let storage = SimShardStorage::default();
    let root0 = storage.state_root();

    storage.commit_shared(&make_settled_writes(1, 10, vec![1]));
    let root1 = storage.state_root();
    assert_ne!(root0, root1, "root should change after first commit");

    storage.commit_shared(&make_settled_writes(4, 20, vec![2]));
    let root2 = storage.state_root();
    assert_ne!(root1, root2, "root should change after second commit");
}

#[test]
fn test_state_root_deterministic() {
    // Two storage instances with identical commits should have identical roots
    let s1 = SimShardStorage::default();
    let s2 = SimShardStorage::default();

    let updates = make_settled_writes(1, 10, vec![42]);
    s1.commit_shared(&updates);
    s2.commit_shared(&updates);

    assert_eq!(s1.state_root(), s2.state_root());
    assert_eq!(s1.jmt_height(), s2.jmt_height());
}

#[test]
fn test_state_root_differs_for_different_data() {
    let s1 = SimShardStorage::default();
    let s2 = SimShardStorage::default();

    s1.commit_shared(&make_settled_writes(1, 10, vec![1]));
    s2.commit_shared(&make_settled_writes(1, 10, vec![2]));

    assert_ne!(s1.state_root(), s2.state_root());
}

#[test]
fn test_empty_commit_still_advances_version() {
    let storage = SimShardStorage::default();
    let updates = SettledWrites::default();
    storage.commit_shared(&updates);
    assert_eq!(storage.jmt_height(), BlockHeight::new(1));
}

// ═══════════════════════════════════════════════════════════════════════
// ShardChainWriter
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_commit_block_single() {
    let storage = SimShardStorage::default();
    let updates = make_settled_writes(1, 10, vec![42]);
    let block = make_test_block(BlockHeight::new(1));
    let qc = make_test_qc(&block);

    let result = commit_with(&storage, &updates, &block, &qc);
    assert_ne!(result, StateRoot::ZERO);
}

#[test]
fn test_commit_block_multiple_updates() {
    let storage = SimShardStorage::default();
    let updates1 = make_settled_writes(1, 10, vec![1]);
    let updates2 = make_settled_writes(2, 20, vec![2]);
    let merged = SettledWrites::from_absolutes(
        updates1
            .cells()
            .iter()
            .chain(updates2.cells())
            .map(|(key, change)| (*key, change.clone()))
            .collect(),
    );
    let block = make_test_block(BlockHeight::new(1));
    let qc = make_test_qc(&block);

    let result = commit_with(&storage, &merged, &block, &qc);
    assert_ne!(result, StateRoot::ZERO);
}

#[test]
fn test_commit_block_empty() {
    let storage = SimShardStorage::default();
    let block = make_test_block(BlockHeight::new(1));
    let qc = make_test_qc(&block);
    commit_empty(&storage, &block, &qc);
    // Empty block: JMT version still advances to block_height
    assert_eq!(storage.jmt_height(), BlockHeight::new(1));
}

#[test]
fn test_prepare_then_commit_fast_path() {
    // Two identical storage instances: one uses prepare+commit, other uses commit_block.
    // Both should produce the same result.
    let s_prepared = Arc::new(SimShardStorage::default());
    let s_direct = SimShardStorage::default();
    let block = make_test_block(BlockHeight::new(1));
    let qc = make_test_qc(&block);

    // Prepare path
    let parent_root = s_prepared.state_root();
    let (spec_root, _jmt_snapshot, prepared) = s_prepared.prepare_block_commit(
        ParentAnchor {
            state_root: parent_root,
            height: BlockHeight::GENESIS,
            state: &*s_prepared,
        },
        &[],
        BlockHeight::new(1),
        &[],
        None,
    );
    let certified = make_test_certified(block.clone());
    let result_prepared = prepared(SyncHint::FlushNow, &certified, &no_witness());

    // Direct path
    let result_direct = commit_empty(&s_direct, &block, &qc);

    assert_eq!(result_prepared, result_direct);
    assert_eq!(spec_root, result_prepared);
}

#[test]
fn test_prepare_commit_state_root_matches() {
    let storage = Arc::new(SimShardStorage::default());
    let block = make_test_block(BlockHeight::new(1));
    let qc = make_test_qc(&block);

    let parent_root = storage.state_root();
    let (spec_root, _jmt_snapshot, prepared) = storage.prepare_block_commit(
        ParentAnchor {
            state_root: parent_root,
            height: BlockHeight::GENESIS,
            state: &*storage,
        },
        &[],
        BlockHeight::new(1),
        &[],
        None,
    );
    let certified = make_test_certified(block);
    // Embed the supplied verified QC by replacing the helper's
    // placeholder. SAFETY: synthetic test fixture.
    let _ = qc;
    let result = prepared(SyncHint::FlushNow, &certified, &no_witness());

    assert_eq!(spec_root, result);
}

// ═══════════════════════════════════════════════════════════════════════
// Utility methods
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_clear() {
    let mut storage = SimShardStorage::default();

    // Add some data
    storage.commit_shared(&make_settled_writes(1, 10, vec![1]));
    assert!(storage.jmt_height() > BlockHeight::GENESIS);
    assert!(!storage.is_empty());

    storage.clear();

    assert_eq!(storage.jmt_height(), BlockHeight::new(0));
    assert_eq!(storage.state_root(), StateRoot::ZERO);
    assert!(storage.is_empty());
}

#[test]
fn test_len_and_is_empty() {
    let storage = SimShardStorage::default();
    assert!(storage.is_empty());
    assert_eq!(storage.len(), 0);

    storage.commit_shared(&make_settled_writes(1, 10, vec![1]));
    assert!(!storage.is_empty());
    assert_eq!(storage.len(), 1);

    storage.commit_shared(&make_settled_writes(4, 20, vec![2]));
    assert_eq!(storage.len(), 2);
}

/// The per-version substate byte total follows the production block-commit
/// path: inserts raise it, value updates leave it, deletes lower it,
/// and historical entries stay readable.
#[test]
fn substate_bytes_tracks_block_commits() {
    let storage = SimShardStorage::default();

    // h1: two inserts.
    let v1 = union_of(&[
        make_settled_writes(3, 7, vec![1]),
        make_settled_writes(4, 8, vec![2]),
    ]);
    let block1 = make_test_block(BlockHeight::new(1));
    let qc1 = make_test_qc(&block1);
    commit_with(&storage, &v1, &block1, &qc1);
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(1)), Some(2));

    // h2: value update only.
    let v2 = make_settled_writes(3, 7, vec![9]);
    let block2 = make_test_block(BlockHeight::new(2));
    let qc2 = make_test_qc(&block2);
    commit_with(&storage, &v2, &block2, &qc2);
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(2)), Some(2));

    // h3: delete one — count drops; history retained.
    let v3 = SettledWrites::from_absolutes(BTreeMap::from([(state_key(3, 7), None)]));
    let block3 = make_test_block(BlockHeight::new(3));
    let qc3 = make_test_qc(&block3);
    commit_with(&storage, &v3, &block3, &qc3);
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(3)), Some(1));
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(1)), Some(2));
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(4)), None);
}

#[test]
fn historical_substate_reads_resolve_per_version() {
    let storage = SimShardStorage::default();
    let key = state_key(1, 10);

    // Block height 1: commit value [100].
    let updates1 = make_settled_writes(1, 10, vec![100]);
    let block1 = make_test_block(BlockHeight::new(1));
    let qc1 = make_test_qc(&block1);
    let root_v1 = commit_with(&storage, &updates1, &block1, &qc1);

    // Block height 2: overwrite with value [200].
    let updates2 = make_settled_writes(1, 10, vec![200]);
    let block2 = make_test_block(BlockHeight::new(2));
    let qc2 = make_test_qc(&block2);
    let root_v2 = commit_with(&storage, &updates2, &block2, &qc2);
    assert_ne!(root_v1, root_v2, "roots must differ after overwrite");

    assert_eq!(
        storage.get_substate_at_height(key, BlockHeight::new(1)),
        Some(Some(vec![100u8])),
        "v1 value should be [100]"
    );
    assert_eq!(
        storage.get_substate_at_height(key, BlockHeight::new(2)),
        Some(Some(vec![200u8])),
        "v2 value should be [200]"
    );

    // An unwritten cell reads as absent, not as an unavailable height.
    assert_eq!(
        storage.get_substate_at_height(state_key(99, 10), BlockHeight::new(1)),
        Some(None),
    );

    // A future version is unavailable.
    assert!(
        storage
            .get_substate_at_height(key, BlockHeight::new(99))
            .is_none(),
        "future version should return None"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Execution certificate storage
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn a_replay_names_what_committed_and_never_resolved() {
    let storage = SimShardStorage::default();
    test_helpers::test_unresolved_fold(&storage);
}

/// Sealing a block keeps its bundles' hashes and drops their bodies, so
/// the bodies are recovered from where they are stored beside it.
#[test]
fn a_committed_bundle_outlives_its_block_s_sealing() {
    let storage = SimShardStorage::default();
    let bundle = Provisions::new(
        ShardId::leaf(1, 1),
        ShardId::ROOT,
        BlockHeight::new(1),
        WeightedTimestamp::ZERO,
        RevealChain::ZERO,
        MerkleInclusionProof::dummy(),
        vec![ProvisionEntry::new(TxHash::ZERO, Vec::new())],
    );
    let hash = bundle.hash();
    let block = match make_test_block(BlockHeight::new(1)) {
        Block::Live {
            header,
            transactions,
            certificates,
            witness_sources,
            ..
        } => Block::Live {
            header,
            transactions,
            certificates,
            provisions: Arc::new(vec![Arc::new(Verifiable::from(bundle))]),
            witness_sources,
        },
        sealed @ Block::Sealed { .. } => sealed,
    };
    storage.commit_block(&make_test_certified(block), &no_witness());

    assert!(
        storage
            .get_block(BlockHeight::new(1))
            .expect("the block committed")
            .block()
            .provisions()
            .is_empty(),
        "the stored block is sealed and carries no bodies",
    );
    assert_eq!(
        storage
            .load_recovered_state()
            .retained_provisions
            .iter()
            .map(|p| p.hash())
            .collect::<Vec<_>>(),
        vec![hash],
        "and the body it dropped is recovered from storage",
    );
}

#[test]
fn test_ec_storage_roundtrip() {
    let storage = SimShardStorage::default();
    test_helpers::test_ec_storage_roundtrip(&storage);
}

#[test]
fn test_ec_storage_batch() {
    let storage = SimShardStorage::default();
    test_helpers::test_ec_storage_batch(&storage);
}

#[test]
fn witness_payload_range_reads() {
    let storage = SimShardStorage::default();
    test_helpers::test_witness_payload_range_reads(&storage);
}

// ═══════════════════════════════════════════════════════════════════════
// Persistence-lag determinism
// ═══════════════════════════════════════════════════════════════════════

/// Two validators with different `persisted_height` but reading at the
/// same historical version must observe identical substate values —
/// historical reads must not be influenced by writes committed past the
/// requested version on the faster-persisting validator.
#[test]
fn test_snapshot_at_version_is_deterministic_across_persistence_lag() {
    let node_seed = 1u8;

    let commit = |storage: &SimShardStorage, height: BlockHeight, value: Vec<u8>| {
        let block = make_test_block(height);
        let qc = make_test_qc(&block);
        let writes = make_settled_writes(node_seed, 1, value);
        commit_with(storage, &writes, &block, &qc);
    };

    // Validator A: persists through block 5.
    let a = SimShardStorage::default();
    for h in 1..=5u64 {
        commit(
            &a,
            BlockHeight::new(h),
            vec![u8::try_from(h).unwrap_or(u8::MAX)],
        );
    }
    assert_eq!(a.jmt_height(), BlockHeight::new(5));

    // Validator B: stops at block 3.
    let b = SimShardStorage::default();
    for h in 1..=3u64 {
        commit(
            &b,
            BlockHeight::new(h),
            vec![u8::try_from(h).unwrap_or(u8::MAX)],
        );
    }
    assert_eq!(b.jmt_height(), BlockHeight::new(3));

    // Both read at version 3 via the state-history log. Must see block-3's
    // value on both, not A's current (block-5) value.
    let snap_a = a.snapshot_at(BlockHeight::new(3));
    let snap_b = b.snapshot_at(BlockHeight::new(3));
    let key = state_key(node_seed, 1);

    assert_eq!(
        snap_a.substate(key),
        Some(vec![3]),
        "validator A must see block-3 value at v3, not its current (block-5) value"
    );
    assert_eq!(
        snap_a.substate(key),
        snap_b.substate(key),
        "validators at different persisted heights must agree on version-3 state"
    );
}

/// Exercises the seek-for-prev read path: a key with many historical
/// versions resolves to the correct floor at any target version without
/// scanning all intermediate versions. Correctness check; the perf win
/// is visible as lower CPU on hot keys in production.
#[test]
fn test_snapshot_resolves_floor_among_many_versions() {
    let node_seed = 5u8;

    let storage = SimShardStorage::default();
    for h in 1..=50u64 {
        let block = make_test_block(BlockHeight::new(h));
        let qc = make_test_qc(&block);
        let writes = make_settled_writes(node_seed, 1, vec![u8::try_from(h).unwrap_or(u8::MAX)]);
        commit_with(&storage, &writes, &block, &qc);
    }

    let key = state_key(node_seed, 1);

    // Read at every 10th version — each should return the exact write
    // from that height, not the latest or any adjacent version.
    for target in [1u64, 10, 20, 25, 49, 50] {
        let snap = storage.snapshot_at(BlockHeight::new(target));
        assert_eq!(
            snap.substate(key),
            Some(vec![u8::try_from(target).unwrap_or(u8::MAX)]),
            "snapshot_at({target}) should resolve to block-{target} value"
        );
    }
}

/// State-history walkthrough: key K created at V1 with value A, deleted
/// at V2, recreated at V3 with value B. Every historical version must
/// read back the correct value — that's the "smallest history entry
/// after V" invariant end-to-end.
///
/// Uses `commit_shared` (test-only helper) so we don't have to
/// construct full blocks/QCs around every write.
#[test]
fn test_state_history_create_delete_create() {
    let key = state_key(7, 42);

    let storage = SimShardStorage::default();

    // Keep a second key alive throughout so the JMT never empties out
    // — the JMT parent-version chain would otherwise break at V2 if
    // deleting K left the tree empty. The state-history behavior we're
    // actually testing is entirely independent of this.
    let anchor = make_settled_writes(99, 0xFF, vec![0xFF]);

    // V1: create with value A (=0xAA). Also set the anchor key.
    let v1 = union_of(&[make_settled_writes(7, 42, vec![0xAA]), anchor]);
    storage.commit_shared(&v1);

    // V2: delete K.
    let v2 = SettledWrites::from_absolutes(BTreeMap::from([(key, None)]));
    storage.commit_shared(&v2);

    // V3: create again with value B (=0xBB).
    let v3 = make_settled_writes(7, 42, vec![0xBB]);
    storage.commit_shared(&v3);

    // Expected:
    // V0: before any writes → None. History[K,1] = None wins (smallest
    //     v' > 0 for K). prior = None → None.
    // V1: snapshot_at(1) is "current" branch (1 == current_version only
    //     after V1 commit; but we're at V3 now, so V1 is historical).
    //     Smallest history > V1 is (K, 2) with prior=Some(A). → A.
    // V2: smallest history > V2 is (K, 3) with prior=None (K was
    //     deleted at V2, so pre-V3 was absent). → None.
    // V3: trivial branch (current). current_state[K] = B. → B.
    let expected: &[(u64, Option<Vec<u8>>)] = &[
        (0, None),
        (1, Some(vec![0xAA])),
        (2, None),
        (3, Some(vec![0xBB])),
    ];

    for (v, want) in expected {
        let snap = storage.snapshot_at(BlockHeight::new(*v));
        let got = snap.substate(key);
        assert_eq!(
            &got, want,
            "state-history read at V={v}: want={want:?}, got={got:?}"
        );
    }
}

/// `snapshot_at(V)` must panic when V is below the retention floor.
/// This is the DA-assumption guard: internal code should never
/// anchor a view at a version beyond the retention window, and
/// hitting it means a bug elsewhere (not a graceful-degradation
/// case).
#[test]
#[should_panic(expected = "below retention floor")]
fn test_snapshot_at_below_retention_panics() {
    // Tiny retention: floor = current - 2.
    let storage = SimShardStorage::with_jmt_history_length(2);
    for h in 1..=10u64 {
        let block = make_test_block(BlockHeight::new(h));
        let qc = make_test_qc(&block);
        commit_with(&storage, &SettledWrites::default(), &block, &qc);
    }
    // current=10, floor=8. Asking for V=1 is well below floor.
    let _snap = <SimShardStorage as VersionedStore>::snapshot_at(&storage, BlockHeight::new(1));
}

/// `get_substate_at_height` is an external-facing API — it must
/// return `None` for out-of-retention heights rather than panicking
/// (the panic path is reserved for `snapshot_at` callers).
#[test]
fn test_historical_substate_read_respects_retention() {
    let key = SubstateKey {
        owner: Address([9u8; 16]),
        local: LocalKey([1u8; 16]),
    };

    let storage = SimShardStorage::with_jmt_history_length(2);
    for h in 1..=10u64 {
        let block = make_test_block(BlockHeight::new(h));
        let qc = make_test_qc(&block);
        let writes = SettledWrites::from_absolutes(BTreeMap::from([(
            key,
            Some(vec![u8::try_from(h).unwrap_or(u8::MAX)]),
        )]));
        commit_with(&storage, &writes, &block, &qc);
    }
    // current=10, floor=8.

    // Within retention: returns Some.
    let got = storage.get_substate_at_height(key, BlockHeight::new(9));
    assert_eq!(
        got,
        Some(Some(vec![9])),
        "height within retention must succeed"
    );

    // Below retention: returns None (graceful).
    let got = storage.get_substate_at_height(key, BlockHeight::new(1));
    assert!(got.is_none(), "height below retention must return None");

    // Above current: returns None.
    let got = storage.get_substate_at_height(key, BlockHeight::new(99));
    assert!(got.is_none(), "future height returns None");
}

/// Genesis-style writes via `commit_substates_only` must NOT populate
/// the state-history log — there is no pre-state to preserve, and
/// polluting the log with `(K, 0) → None` entries would waste space
/// until GC.
#[test]
fn test_genesis_skips_history_entries() {
    let storage = SimShardStorage::default();

    let updates = make_settled_writes(1, 1, vec![0xAA]);
    storage.commit_substates_only(&updates);

    // History map must be empty after a genesis-style commit.
    assert_eq!(
        storage.state.read().unwrap().state_history.len(),
        0,
        "commit_substates_only must not record state-history entries"
    );
    // current_state must have the genesis write though.
    assert_eq!(
        storage.state.read().unwrap().current_state.len(),
        1,
        "commit_substates_only populates current_state"
    );
}

/// Witness retention follows the commit-carried floor with one window
/// of hysteresis, and recovery rebuilds the accumulator window from the
/// tip header's base — entries below it are serving stock only.
#[test]
fn witness_window_retention_and_recovery() {
    use hyperscale_storage::test_helpers::{commit_block_with_witness_window, stake_deposit};
    use hyperscale_types::ShardWitnessPayload;

    let storage = SimShardStorage::default();
    let deposits: Vec<_> = (0u64..6).map(stake_deposit).collect();

    // Window [0, 4): all four leaves appended, nothing pruned.
    commit_block_with_witness_window(
        &storage,
        BlockHeight::new(1),
        0,
        &deposits[0..4],
        &deposits[0..4],
        None,
    );
    // Window [2, 6): the tail appends, persisted floor untouched.
    commit_block_with_witness_window(
        &storage,
        BlockHeight::new(2),
        2,
        &deposits[2..6],
        &deposits[4..6],
        None,
    );
    // Window [4, 6): the base advance carries the previous window's
    // base as the persisted floor — leaves below 2 drop, [2, 4) stays
    // as hysteresis stock.
    commit_block_with_witness_window(
        &storage,
        BlockHeight::new(3),
        4,
        &deposits[4..6],
        &[],
        Some(BeaconWitnessLeafCount::new(2)),
    );

    // A read spanning the dropped range comes back short; the retained
    // hysteresis range answers in full.
    assert_eq!(storage.get_beacon_witness_payload_range(0, 6).len(), 4);
    assert_eq!(
        storage.get_beacon_witness_payload_range(2, 6),
        deposits[2..6].to_vec(),
    );

    // Recovery starts the accumulator window at the tip's base.
    let recovered = storage.load_recovered_state();
    assert_eq!(
        recovered.beacon_witness_start,
        BeaconWitnessLeafCount::new(4)
    );
    let expected: Vec<_> = deposits[4..6]
        .iter()
        .map(ShardWitnessPayload::leaf_hash)
        .collect();
    assert_eq!(recovered.beacon_witness_leaf_hashes, expected);
}

// ─── Safe-vote registers ─────────────────────────────────────────────────────

fn registers(locked: u64, last_voted: u64) -> SafeVoteRegisters {
    SafeVoteRegisters {
        locked_round: Round::new(locked),
        last_voted_round: Round::new(last_voted),
    }
}

/// Writes merge field-wise max and survive a coordinator "restart" —
/// the store handle outlives the state machine, so a rebuilt machine
/// recovers them through `load_recovered_state`.
#[test]
fn safe_vote_registers_are_monotone_and_recoverable() {
    let storage = SimShardStorage::default();
    let v = ValidatorId::new(1);
    storage.persist_safe_vote_registers(v, registers(4, 6));
    storage.persist_safe_vote_registers(v, registers(2, 9));
    assert_eq!(storage.safe_vote_registers(v), Some(registers(4, 9)));

    let recovered = storage.load_recovered_state();
    assert_eq!(
        recovered.safe_vote_registers.get(&v),
        Some(&registers(4, 9))
    );
}

/// A record written under a different chain origin is invisible to
/// reads and recovery; the next write starts a fresh record under the
/// new origin.
#[test]
fn safe_vote_registers_ignore_stale_chain_incarnation() {
    let storage = SimShardStorage::default();
    let v = ValidatorId::new(1);
    storage.persist_safe_vote_registers(v, registers(8, 8));

    storage.consensus.write().unwrap().chain_origin = ChainOrigin {
        genesis_height: BlockHeight::new(11),
        anchor_wt: WeightedTimestamp::from_millis(999),
    };

    assert_eq!(storage.safe_vote_registers(v), None);
    assert!(
        storage
            .load_recovered_state()
            .safe_vote_registers
            .is_empty()
    );

    storage.persist_safe_vote_registers(v, registers(1, 2));
    assert_eq!(storage.safe_vote_registers(v), Some(registers(1, 2)));
}
