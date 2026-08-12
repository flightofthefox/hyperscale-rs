use std::collections::BTreeMap;
use std::sync::Arc;

use hyperscale_jmt::NibblePath;
use hyperscale_storage::test_helpers::{
    make_settled_writes, make_test_block, make_test_block_with_anchor_wt, make_test_certified,
    make_test_execution_certificate, make_test_finalization, make_test_qc, make_test_receipt,
    state_key, test_committed_bundle_outlives_sealing,
    test_ec_storage_batch as helpers_test_ec_storage_batch,
    test_ec_storage_roundtrip as helpers_test_ec_storage_roundtrip,
    test_recovery_carries_the_tip_drain_total, test_registers_recover_their_justification,
    test_retained_bundle_drops_below_the_history_floor, test_undischarged_record_holds_the_floor,
    test_unresolved_fold, test_widest_tick_copy_holds_the_slot,
    test_witness_payload_range_reads as helpers_test_witness_payload_range_reads, with_provisions,
};
use hyperscale_storage::{
    ParentAnchor, SafeVoteRegisterStore, ShardChainReader, ShardChainWriter, SubstateDatabase,
    SubstateStore, VersionedStore,
};
use hyperscale_types::{
    Address, AddressClass, AggregateSignature, BeaconWitnessCommit, BeaconWitnessLeafCount, Block,
    BlockHash, BlockHeight, CertifiedBlock, ChainOrigin, ConsensusReceipt, ExecutionCertificate,
    Finalization, FinalizationHash, GlobalReceiptHash, GlobalReceiptRoot, Hash, LocalKey,
    ProposerTimestamp, QuorumCertificate, Round, SafeVoteRegisters, SettledWrites, ShardId,
    SignerBitfield, StateRoot, StoredReceipt, SubstateKey, SyncHint, TickHalf, TickId, TxHash,
    ValidatorId, Verifiable, Verified, WeightedTimestamp, WitnessSources,
};

fn no_witness() -> BeaconWitnessCommit {
    BeaconWitnessCommit::empty(BeaconWitnessLeafCount::ZERO)
}

/// Build a placeholder EC whose `tick_id` matches the WC the caller is about
/// to construct, so the WC satisfies the local-EC invariant enforced at
/// HBOR decode time. The EC carries no signers / outcomes — these tests
/// exercise the storage codec, not consensus.
fn placeholder_local_ec(shard: ShardId, height: BlockHeight) -> Arc<ExecutionCertificate> {
    Arc::new(ExecutionCertificate::new(
        TickId::new(shard, height),
        WeightedTimestamp::from_millis(0),
        GlobalReceiptRoot::ZERO,
        Vec::new(),
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::empty(),
    ))
}
use rocksdb::WriteBatch;
use tempfile::TempDir;

use super::column_families::STATE_HISTORY_CF;
use super::core::RocksDbShardStorage;
use super::metadata::write_chain_origin;
use crate::config::RocksDbConfig;

/// Helper: wrap writes into a single `StoredReceipt` for test commit calls.
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

fn updates_to_receipts(writes: &SettledWrites) -> Vec<StoredReceipt> {
    if writes.is_empty() {
        return vec![];
    }
    vec![StoredReceipt {
        tx_hash: TxHash::ZERO,
        consensus: Arc::new(ConsensusReceipt::Succeeded {
            receipt_hash: GlobalReceiptHash::ZERO,
            writes: writes.clone().into(),
            beacon_witness_events: Vec::new(),
            events: Vec::new(),
        }),
        metadata: None,
    }]
}

/// Helper: commit a block with empty updates and no ECs/receipts.
fn commit_empty(storage: &RocksDbShardStorage, block: &Block, qc: &Verified<QuorumCertificate>) {
    // SAFETY: synthetic test fixture; round-trip tests don't exercise
    // the `Verified<CertifiedBlock>` predicate.
    let certified = Arc::new(Verified::<CertifiedBlock>::new_unchecked_for_test(
        CertifiedBlock::new_unchecked(block.clone(), <Verified<_>>::clone(qc)),
    ));
    storage.commit_block(&certified, &no_witness());
}

/// Writes holding a single removal.
fn make_state_delete(owner_seed: u8, local_seed: u8) -> SettledWrites {
    SettledWrites::from_absolutes(BTreeMap::from([(state_key(owner_seed, local_seed), None)]))
}

/// The per-version substate byte total tracks inserts, value updates and
/// deletes; historical entries stay readable and survive a reopen.
#[test]
fn substate_bytes_tracks_commits_and_survives_reopen() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    // v1: two inserts.
    let v1 = union_of(&[
        make_settled_writes(3, 7, vec![1]),
        make_settled_writes(4, 8, vec![2]),
    ]);
    storage.commit(&v1).unwrap();
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(1)), Some(2));

    // v2: value update only — count unchanged.
    storage.commit(&make_settled_writes(3, 7, vec![9])).unwrap();
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(2)), Some(2));

    // v3: delete one — count drops; the historical entry is untouched.
    storage.commit(&make_state_delete(3, 7)).unwrap();
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(3)), Some(1));
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(1)), Some(2));

    drop(storage);
    let reopened = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    assert_eq!(reopened.substate_bytes_at(BlockHeight::new(3)), Some(1));
    assert_eq!(reopened.substate_bytes_at(BlockHeight::new(2)), Some(2));
    assert_eq!(reopened.substate_bytes_at(BlockHeight::new(4)), None);
}

/// Recovery seeds the coordinator's count frontier with the substate
/// count behind the committed tip.
#[test]
fn recovered_state_carries_substate_bytes() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    for h in 1..=3u64 {
        let block = make_test_block(BlockHeight::new(h));
        let qc = make_test_qc(&block);
        let updates = make_settled_writes(u8::try_from(h).unwrap_or(u8::MAX), 1, vec![1]);
        rocks_commit_with(&storage, &updates, &block, &qc);
    }

    assert_eq!(storage.load_recovered_state().substate_bytes, 3);
}

#[test]
fn test_basic_substate_operations() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let key = state_key(3, 10);

    // Initially empty
    assert!(storage.substate(key).is_none());

    // Commit a value
    storage
        .commit(&make_settled_writes(3, 10, vec![99, 88, 77]))
        .unwrap();

    // Now we can read it
    assert_eq!(storage.substate(key), Some(vec![99, 88, 77]));
}

#[test]
fn test_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let key = state_key(7, 10);

    // Write initial value
    storage
        .commit(&make_settled_writes(7, 10, vec![1]))
        .unwrap();

    // Take snapshot
    let snapshot = storage.snapshot();

    // Snapshot can read data
    assert_eq!(snapshot.substate(key), Some(vec![1]));
}

#[test]
fn test_recovery_resumes_at_correct_height() {
    let temp_dir = TempDir::new().unwrap();

    let expected_hash = Hash::from_hash_bytes(&[50; 32]);

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        storage.set_chain_metadata(BlockHeight::new(50), Some(expected_hash), None);
    }

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        let recovered = storage.load_recovered_state();

        assert_eq!(recovered.committed_height, BlockHeight::new(50));
        assert_eq!(
            recovered.committed_hash,
            Some(BlockHash::from_raw(expected_hash))
        );
    }
}

#[test]
fn test_commit_certificate_with_writes_persists_both() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let writes = make_settled_writes(1, 10, vec![99, 88, 77]);
    let cert = make_test_finalization(BlockHeight::new(42), ShardId::ROOT);
    let tick_id = *cert.tick_id();

    storage.commit_certificate_with_writes(&cert, &writes);

    let stored_cert = storage.get_certificate(&cert.receipt_hash());
    assert!(stored_cert.is_some());
    assert_eq!(stored_cert.unwrap().tick_id(), &tick_id);

    // Verify the substate was written to the state CF via direct key lookup.
    assert_eq!(
        storage.substate(state_key(1, 10)),
        Some(vec![99, 88, 77]),
        "value should match what was written"
    );
}

#[test]
fn test_block_storage_and_retrieval() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

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
fn test_block_range_retrieval() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    for h in 1..=5u64 {
        let block = make_test_block(BlockHeight::new(h));
        let qc = make_test_qc(&block);
        commit_empty(&storage, &block, &qc);
    }

    let blocks = storage.get_blocks_range(BlockHeight::new(2), BlockHeight::new(5));
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].block().height(), BlockHeight::new(2));
    assert_eq!(blocks[1].block().height(), BlockHeight::new(3));
    assert_eq!(blocks[2].block().height(), BlockHeight::new(4));
}

#[test]
fn test_recovery_with_qc() {
    use hyperscale_types::SignerBitfield;

    let temp_dir = TempDir::new().unwrap();
    let expected_raw = Hash::from_hash_bytes(&[99; 32]);
    let expected_hash = BlockHash::from_raw(expected_raw);

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        let qc = QuorumCertificate::new(
            expected_hash,
            ShardId::ROOT,
            BlockHeight::new(100),
            BlockHash::from_raw(Hash::from_bytes(&[98; 32])),
            Round::new(5),
            SignerBitfield::new(4),
            AggregateSignature::ZERO,
            WeightedTimestamp::from_millis(100_000),
        );
        storage.set_chain_metadata(BlockHeight::new(100), Some(expected_raw), Some(&qc));
    }

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        let recovered = storage.load_recovered_state();

        assert_eq!(recovered.committed_height, BlockHeight::new(100));
        assert_eq!(recovered.committed_hash, Some(expected_hash));
        assert!(recovered.latest_qc.is_some());

        let qc = recovered.latest_qc.unwrap();
        assert_eq!(qc.height(), BlockHeight::new(100));
        assert_eq!(qc.round(), Round::new(5));
        assert_eq!(qc.block_hash(), expected_hash);
    }
}

#[test]
fn test_recovery_seeds_committed_anchor_from_parent_qc() {
    let temp_dir = TempDir::new().unwrap();
    let height = BlockHeight::new(1);
    let block = make_test_block(height); // parent QC is the genesis QC (WT zero)
    let tip_qc = make_test_qc(&block); // tip's own QC: WT = block timestamp (1000)

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        commit_empty(&storage, &block, &tip_qc);
        storage.set_chain_metadata(
            height,
            Some(Hash::from_hash_bytes(&[42; 32])),
            Some(&*tip_qc),
        );
    }

    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    let recovered = storage.load_recovered_state();

    // The anchor is the committed tip's *parent* QC weighted timestamp (the
    // genesis QC's zero here), read back from the tip's stored header — not the
    // tip's own QC timestamp (1000) that `latest_qc` carries. Seeding from the
    // tip's own WT would resolve the wrong committee for the first post-restart
    // child of a tip that is an epoch's first block.
    assert_eq!(
        recovered.committed_block_anchor_wt,
        Some(WeightedTimestamp::ZERO),
        "anchor must come from the tip's parent QC, not its own QC",
    );
    assert_eq!(
        recovered.latest_qc.map(|qc| qc.weighted_timestamp()),
        Some(WeightedTimestamp::from_millis(1000)),
        "sanity: the tip's own QC carries the distinct, non-zero timestamp",
    );
}

#[test]
fn test_recovery_seeds_committee_anchor_from_the_header_below_the_tip() {
    let temp_dir = TempDir::new().unwrap();
    // Two committed heights whose anchors sit in different windows: the tip's
    // own at 200, the one its parent carried at 100.
    let parent = make_test_block_with_anchor_wt(BlockHeight::new(1), 100);
    let tip = make_test_block_with_anchor_wt(BlockHeight::new(2), 200);
    let tip_qc = make_test_qc(&tip);

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        commit_empty(&storage, &parent, &make_test_qc(&parent));
        commit_empty(&storage, &tip, &tip_qc);
        storage.set_chain_metadata(
            BlockHeight::new(2),
            Some(*tip.hash().as_raw()),
            Some(&*tip_qc),
        );
    }

    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    let recovered = storage.load_recovered_state();

    // A block's committee keys on its parent, so the committee that signed the
    // tip anchors one height below it. Recovering only the tip's own anchor
    // would resolve the tip against the window it opens.
    assert_eq!(
        recovered.committed_block_anchor_wt,
        Some(WeightedTimestamp::from_millis(200)),
    );
    assert_eq!(
        recovered.committed_committee_anchor_wt,
        Some(WeightedTimestamp::from_millis(100)),
        "the tip's committee anchor comes from the header below it",
    );
}

#[test]
fn test_certificate_idempotency() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let updates = make_settled_writes(1, 10, vec![99, 88, 77]);
    let cert = make_test_finalization(BlockHeight::new(42), ShardId::ROOT);
    let tick_id = *cert.tick_id();

    storage.commit_certificate_with_writes(&cert, &updates);
    storage.commit_certificate_with_writes(&cert, &updates);

    let stored = storage.get_certificate(&cert.receipt_hash());
    assert!(stored.is_some());
    assert_eq!(stored.unwrap().tick_id(), &tick_id);
}

#[test]
fn test_empty_state_on_fresh_database() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let recovered = storage.load_recovered_state();

    assert_eq!(recovered.committed_height, BlockHeight::new(0));
    assert!(recovered.committed_hash.is_none());
    assert!(recovered.latest_qc.is_none());
}

// ═══════════════════════════════════════════════════════════════════════
// JMT state tracking
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_block_height_increments_on_commit() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    assert_eq!(storage.jmt_height(), BlockHeight::new(0));

    storage
        .commit(&make_settled_writes(1, 10, vec![1]))
        .unwrap();
    assert_eq!(storage.jmt_height(), BlockHeight::new(1));

    storage
        .commit(&make_settled_writes(4, 20, vec![2]))
        .unwrap();
    assert_eq!(storage.jmt_height(), BlockHeight::new(2));
}

#[test]
fn test_state_root_changes_on_commit() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let root0 = storage.state_root();

    storage
        .commit(&make_settled_writes(1, 10, vec![1]))
        .unwrap();
    let root1 = storage.state_root();
    assert_ne!(root0, root1, "root should change after first commit");

    storage
        .commit(&make_settled_writes(4, 20, vec![2]))
        .unwrap();
    let root2 = storage.state_root();
    assert_ne!(root1, root2, "root should change after second commit");
}

// ═══════════════════════════════════════════════════════════════════════
// ShardChainWriter
// ═══════════════════════════════════════════════════════════════════════

/// Append a `Finalization` to a block in place. Because `Block` is an enum,
/// this replaces the whole value via `std::mem::replace`.
fn push_finalization(block: &mut Block, fw: Arc<Verifiable<Finalization>>) {
    let taken = std::mem::replace(
        block,
        Block::Sealed {
            header: block.header().clone(),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provision_hashes: Arc::new(Vec::new()),
            terminal_verdicts: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        },
    );
    *block = match taken {
        Block::Live {
            header,
            transactions,
            certificates,
            provisions,
            terminal_verdicts,
            witness_sources,
        } => {
            let mut certificates = (*certificates).clone();
            certificates.push(fw);
            Block::Live {
                header,
                transactions,
                certificates: Arc::new(certificates),
                provisions,
                terminal_verdicts,
                witness_sources,
            }
        }
        Block::Sealed {
            header,
            transactions,
            certificates,
            provision_hashes,
            terminal_verdicts,
            witness_sources,
        } => {
            let mut certificates = (*certificates).clone();
            certificates.push(fw);
            Block::Sealed {
                header,
                transactions,
                certificates: Arc::new(certificates),
                provision_hashes,
                terminal_verdicts,
                witness_sources,
            }
        }
    };
}

/// Wrap receipts into a single `Finalization` attached to `block.certificates`,
/// so the new `commit_block` (which derives receipts from `block.certificates`)
/// can apply them.
fn attach_receipts(block: &mut Block, receipts: Vec<StoredReceipt>) {
    let new_fw: Arc<Verifiable<Finalization>> = Arc::new(
        Finalization::new(
            TickId::new(ShardId::ROOT, block.height()),
            TickHalf::Determined,
            vec![placeholder_local_ec(ShardId::ROOT, block.height())],
            receipts,
        )
        .into(),
    );
    // Take block out, mutate, and put back.
    let taken = std::mem::replace(
        block,
        Block::Sealed {
            header: block.header().clone(),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provision_hashes: Arc::new(Vec::new()),
            terminal_verdicts: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        },
    );
    *block = match taken {
        Block::Live {
            header,
            transactions,
            certificates,
            provisions,
            terminal_verdicts,
            witness_sources,
        } => {
            let mut certificates = (*certificates).clone();
            certificates.push(new_fw);
            Block::Live {
                header,
                transactions,
                certificates: Arc::new(certificates),
                provisions,
                terminal_verdicts,
                witness_sources,
            }
        }
        Block::Sealed {
            header,
            transactions,
            certificates,
            provision_hashes,
            terminal_verdicts,
            witness_sources,
        } => {
            let mut certificates = (*certificates).clone();
            certificates.push(new_fw);
            Block::Sealed {
                header,
                transactions,
                certificates: Arc::new(certificates),
                provision_hashes,
                terminal_verdicts,
                witness_sources,
            }
        }
    };
}

#[test]
fn test_commit_block_applies_writes() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let updates = make_settled_writes(1, 10, vec![42]);
    let mut block = make_test_block(BlockHeight::new(1));
    let receipts = updates_to_receipts(&updates);
    attach_receipts(&mut block, receipts);
    let result = storage.commit_block(&make_test_certified(block), &no_witness());
    assert_ne!(result, StateRoot::ZERO);
}

#[test]
fn test_commit_block_multiple_certs() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let updates1 = make_settled_writes(1, 10, vec![1]);
    let updates2 = make_settled_writes(2, 20, vec![2]);
    let merged = union_of(&[updates1, updates2]);
    let mut block = make_test_block(BlockHeight::new(1));
    let receipts = updates_to_receipts(&merged);
    attach_receipts(&mut block, receipts);
    let result = storage.commit_block(&make_test_certified(block), &no_witness());
    assert_ne!(result, StateRoot::ZERO);
}

#[test]
fn test_commit_block_empty_certs() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let block = make_test_block(BlockHeight::new(1));
    storage.commit_block(&make_test_certified(block), &no_witness());
    assert_eq!(storage.jmt_height(), BlockHeight::new(1));
}

#[test]
fn test_prepare_then_commit_matches_direct() {
    let temp_dir1 = TempDir::new().unwrap();
    let s_prepared =
        Arc::new(RocksDbShardStorage::open(temp_dir1.path(), NibblePath::empty()).unwrap());
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
    let block = make_test_block(BlockHeight::new(1));
    let result_prepared = prepared(
        SyncHint::FlushNow,
        &make_test_certified(block),
        &no_witness(),
    );

    let temp_dir2 = TempDir::new().unwrap();
    let s_direct = RocksDbShardStorage::open(temp_dir2.path(), NibblePath::empty()).unwrap();
    let block2 = make_test_block(BlockHeight::new(1));
    let result_direct = s_direct.commit_block(&make_test_certified(block2), &no_witness());

    assert_eq!(result_prepared, result_direct);
    assert_eq!(spec_root, result_prepared);
}

#[test]
fn test_commit_block_stores_certificates() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let shard = ShardId::ROOT;
    let cert = make_test_finalization(BlockHeight::new(1), shard);
    let cert_hash = cert.receipt_hash();

    // Create a block that includes this certificate
    let block = make_test_block(BlockHeight::new(1));
    let fw_certificates = Arc::new(vec![Arc::new(cert.into())]);
    let block = match block {
        Block::Live {
            header,
            transactions,
            provisions,
            ..
        } => Block::Live {
            header,
            transactions,
            certificates: fw_certificates,
            provisions,
            terminal_verdicts: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        },
        Block::Sealed {
            header,
            transactions,
            provision_hashes,
            ..
        } => Block::Sealed {
            header,
            transactions,
            certificates: fw_certificates,
            provision_hashes,
            terminal_verdicts: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        },
    };
    let _ = storage.commit_block(&make_test_certified(block), &no_witness());

    assert!(storage.get_certificate(&cert_hash).is_some());
}

// ═══════════════════════════════════════════════════════════════════════
// Batch operations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_transactions_batch_missing() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let result = storage.get_transactions_batch(&[TxHash::from(Hash::from_bytes(&[1; 32]))]);
    assert!(result.is_empty());
}

#[test]
fn test_certificates_batch() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let cert1 = make_test_finalization(BlockHeight::new(1), ShardId::ROOT);
    let cert2 = make_test_finalization(BlockHeight::new(2), ShardId::ROOT);
    let id1 = cert1.receipt_hash();
    let id2 = cert2.receipt_hash();

    storage.put_certificate(&cert1.receipt_hash(), &cert1);
    storage.put_certificate(&cert2.receipt_hash(), &cert2);

    let result = storage.get_certificates_batch(&[id1, id2]);
    assert_eq!(result.len(), 2);

    let missing = FinalizationHash::from_raw(Hash::from_bytes(b"absent"));
    let partial = storage.get_certificates_batch(&[id1, missing]);
    assert_eq!(partial.len(), 1);
    assert_eq!(partial[0].receipt_hash(), id1);
}

// ═══════════════════════════════════════════════════════════════════════
// Parity tests with SimShardStorage
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_initial_block_height_is_zero() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    assert_eq!(storage.jmt_height(), BlockHeight::new(0));
}

#[test]
fn test_initial_state_root_is_zero() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    assert_eq!(storage.state_root(), StateRoot::ZERO);
}

#[test]
fn test_state_root_deterministic() {
    let updates = make_settled_writes(1, 10, vec![42]);

    let td1 = TempDir::new().unwrap();
    let s1 = RocksDbShardStorage::open(td1.path(), NibblePath::empty()).unwrap();
    s1.commit(&updates).unwrap();

    let td2 = TempDir::new().unwrap();
    let s2 = RocksDbShardStorage::open(td2.path(), NibblePath::empty()).unwrap();
    s2.commit(&updates).unwrap();

    assert_eq!(s1.state_root(), s2.state_root());
    assert_eq!(s1.jmt_height(), s2.jmt_height());
}

#[test]
fn test_state_root_differs_for_different_data() {
    let td1 = TempDir::new().unwrap();
    let s1 = RocksDbShardStorage::open(td1.path(), NibblePath::empty()).unwrap();
    s1.commit(&make_settled_writes(1, 10, vec![1])).unwrap();

    let td2 = TempDir::new().unwrap();
    let s2 = RocksDbShardStorage::open(td2.path(), NibblePath::empty()).unwrap();
    s2.commit(&make_settled_writes(1, 10, vec![2])).unwrap();

    assert_ne!(s1.state_root(), s2.state_root());
}

#[test]
fn test_certificate_store_and_retrieve() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let cert = make_test_finalization(BlockHeight::new(1), ShardId::ROOT);
    let tick_id = *cert.tick_id();

    storage.put_certificate(&cert.receipt_hash(), &cert);

    let stored = storage.get_certificate(&cert.receipt_hash()).unwrap();
    assert_eq!(stored.tick_id(), &tick_id);
}

#[test]
fn test_certificate_get_missing() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    let missing = FinalizationHash::from_raw(Hash::from_bytes(b"absent"));
    assert!(storage.get_certificate(&missing).is_none());
}

#[test]
fn test_get_block_for_sync() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let block = make_test_block(BlockHeight::new(1));
    let qc = make_test_qc(&block);
    commit_empty(&storage, &block, &qc);

    let result = storage.get_block_for_sync(BlockHeight::new(1));
    assert!(result.is_some());
    assert_eq!(result.unwrap().0.height(), BlockHeight::new(1));

    assert!(storage.get_block_for_sync(BlockHeight::new(999)).is_none());
}

#[test]
fn test_commit_certificate_via_commit_store() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let updates = make_settled_writes(1, 10, vec![42]);
    let cert = make_test_finalization(BlockHeight::new(1), ShardId::ROOT);

    storage.commit_certificate_with_writes(&cert, &updates);

    assert_eq!(storage.jmt_height(), BlockHeight::new(0));
    assert_eq!(storage.state_root(), StateRoot::ZERO);
    assert!(storage.get_certificate(&cert.receipt_hash()).is_some());
}

#[test]
fn test_empty_commit_still_advances_version() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let updates = SettledWrites::default();
    storage.commit(&updates).unwrap();
    assert_eq!(storage.jmt_height(), BlockHeight::new(1));
}

// ═══════════════════════════════════════════════════════════════════════
// Persistence across reopen
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_substates_survive_reopen() {
    let temp_dir = TempDir::new().unwrap();

    let root_after_write;
    let version_after_write;
    let cert_id;
    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        let updates = make_settled_writes(1, 10, vec![42]);
        let cert = make_test_finalization(BlockHeight::new(1), ShardId::ROOT);
        cert_id = cert.receipt_hash();
        storage.commit_certificate_with_writes(&cert, &updates);
        root_after_write = storage.state_root();
        version_after_write = storage.jmt_height();
    }

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

        assert_eq!(storage.jmt_height(), version_after_write);
        assert_eq!(storage.state_root(), root_after_write);

        let cert = storage.get_certificate(&cert_id);
        assert!(cert.is_some(), "certificate should survive reopen");
        assert_eq!(cert.unwrap().receipt_hash(), cert_id);

        // Verify the substate was written via direct key lookup.
        let value = storage.substate(state_key(1, 10));
        assert_eq!(value, Some(vec![42]), "substate should survive reopen");
    }
}

#[test]
fn test_blocks_survive_reopen() {
    let temp_dir = TempDir::new().unwrap();

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        let block = make_test_block(BlockHeight::new(1));
        let qc = make_test_qc(&block);
        commit_empty(&storage, &block, &qc);
    }

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

        let stored = storage
            .get_block(BlockHeight::new(1))
            .expect("block should survive reopen");
        assert_eq!(stored.block().height(), BlockHeight::new(1));
        assert_eq!(stored.qc().height(), BlockHeight::new(1));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Receipt storage
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_receipt_survives_reopen() {
    let temp_dir = TempDir::new().unwrap();
    let receipt = make_test_receipt(55);
    let tx_hash = receipt.tx_hash;

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        storage.store_receipt(&receipt);
    }

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        assert!(storage.get_consensus_receipt(&tx_hash).is_some());
        let retrieved = storage.get_consensus_receipt(&tx_hash).unwrap();
        assert_eq!(retrieved, receipt.consensus);
        let local = storage.get_execution_metadata(&tx_hash).unwrap();
        assert_eq!(local, receipt.metadata.unwrap());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Execution certificate storage
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn a_replay_names_what_committed_and_never_resolved() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    test_unresolved_fold(&storage);
}

#[test]
fn a_replay_reaches_a_record_no_verdict_has_discharged() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    test_undischarged_record_holds_the_floor(&storage);
}

#[test]
fn test_ec_storage_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    helpers_test_ec_storage_roundtrip(&storage);
}

#[test]
fn test_ec_storage_batch() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    helpers_test_ec_storage_batch(&storage);
}

#[test]
fn witness_payload_range_reads() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    helpers_test_witness_payload_range_reads(&storage);
}

#[test]
fn test_ec_survives_reopen() {
    let temp_dir = TempDir::new().unwrap();
    let ec = make_test_execution_certificate(1, BlockHeight::new(1));
    let tick_id = *ec.tick_id();

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        let block = make_test_block(BlockHeight::new(0));
        storage.commit_block(&make_test_certified(block), &no_witness());
        let mut block = make_test_block(BlockHeight::new(1));
        push_finalization(
            &mut block,
            Arc::new(
                Finalization::new(tick_id, TickHalf::Determined, vec![Arc::new(ec)], vec![]).into(),
            ),
        );
        storage.commit_block(&make_test_certified(block), &no_witness());
    }

    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        let cert = storage
            .get_execution_certificate(&tick_id)
            .expect("EC must survive reopen");
        assert_eq!(cert.block_height(), BlockHeight::new(1));
    }
}

#[test]
fn test_ec_atomic_with_block_commit() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let ec = make_test_execution_certificate(1, BlockHeight::new(1));
    let tick_id = *ec.tick_id();
    let mut block = make_test_block(BlockHeight::new(1));
    push_finalization(
        &mut block,
        Arc::new(
            Finalization::new(tick_id, TickHalf::Determined, vec![Arc::new(ec)], vec![]).into(),
        ),
    );
    // Commit block with EC atomically
    storage.commit_block(&make_test_certified(block), &no_witness());

    let cert = storage
        .get_execution_certificate(&tick_id)
        .expect("EC must be retrievable after commit");
    assert_eq!(cert.block_height(), BlockHeight::new(1));
}

// ─── State-history semantics (parity with storage-memory tests) ─────────────
//
// These mirror `storage-memory/src/tests.rs`:
//   - test_state_history_create_delete_create
//   - test_snapshot_at_below_retention_panics
//   - test_historical_substate_read_respects_retention
//   - test_reset_partition_captures_history_for_all_removed_keys
//   - test_genesis_skips_history_entries
//
// RocksDB encodes the history log differently (wire codec, prefix extractor,
// snapshot isolation, column family) so backend parity is not free.

/// Helper: port of `commit_with` from the memory tests. Injects the updates
/// as a single-tx `Finalization` receipt inside a block and commits it.
fn rocks_commit_with(
    storage: &RocksDbShardStorage,
    writes: &SettledWrites,
    block: &Block,
    qc: &Verified<QuorumCertificate>,
) {
    let mut block = block.clone();
    if !writes.is_empty() {
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
        let tick = Arc::new(
            Finalization::new(
                TickId::new(ShardId::ROOT, block.height()),
                TickHalf::Determined,
                vec![placeholder_local_ec(ShardId::ROOT, block.height())],
                vec![receipt],
            )
            .into(),
        );
        push_finalization(&mut block, tick);
    }
    // SAFETY: synthetic test fixture; round-trip tests don't exercise
    // the `Verified<CertifiedBlock>` predicate.
    let certified = Arc::new(Verified::<CertifiedBlock>::new_unchecked_for_test(
        CertifiedBlock::new_unchecked(block, <Verified<_>>::clone(qc)),
    ));
    storage.commit_block(&certified, &no_witness());
}

/// State-history walkthrough: key K created at V1 with value A, deleted
/// at V2, recreated at V3 with value B. Every historical version must
/// read back the correct value — that's the "smallest history entry
/// after V" invariant end-to-end.
#[test]
fn test_state_history_create_delete_create() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let key = state_key(7, 42);

    // Keep an anchor key alive throughout so the JMT never empties out —
    // deleting K alone at V2 would otherwise break the parent-version
    // chain. The state-history behavior under test is independent of this.
    let anchor = make_settled_writes(99, 0xFF, vec![0xFF]);

    // V1: create K=A, plus anchor.
    let v1 = union_of(&[make_settled_writes(7, 42, vec![0xAA]), anchor]);
    storage.commit(&v1).unwrap();

    // V2: delete K.
    storage.commit(&make_state_delete(7, 42)).unwrap();

    // V3: recreate K=B.
    storage
        .commit(&make_settled_writes(7, 42, vec![0xBB]))
        .unwrap();

    // See memory test for derivation:
    let expected: &[(u64, Option<Vec<u8>>)] = &[
        (0, None),
        (1, Some(vec![0xAA])),
        (2, None),
        (3, Some(vec![0xBB])),
    ];
    for (v, want) in expected {
        let snap =
            <RocksDbShardStorage as VersionedStore>::snapshot_at(&storage, BlockHeight::new(*v));
        let got = snap.substate(key);
        assert_eq!(
            &got, want,
            "state-history read at V={v}: want={want:?}, got={got:?}"
        );
    }
}

/// JMT GC prunes per-version substate byte totals below the retention
/// cutoff and retains everything at or above it.
#[test]
fn substate_bytes_pruned_by_jmt_gc() {
    let temp_dir = TempDir::new().unwrap();
    let config = RocksDbConfig {
        jmt_history_length: 2,
        ..Default::default()
    };
    let storage =
        RocksDbShardStorage::open_with_config(temp_dir.path(), &config, NibblePath::empty())
            .unwrap();

    for h in 1..=10u64 {
        let writes = make_settled_writes(u8::try_from(h).unwrap_or(u8::MAX), 1, vec![1]);
        storage.commit(&writes).unwrap();
    }
    storage.run_jmt_gc();

    // current=10, cutoff=8: below-cutoff entries pruned, the rest intact.
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(7)), None);
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(8)), Some(8));
    assert_eq!(storage.substate_bytes_at(BlockHeight::new(10)), Some(10));
}

/// `snapshot_at(V)` must panic when V is below the retention floor.
#[test]
#[should_panic(expected = "below retention floor")]
fn test_snapshot_at_below_retention_panics() {
    let temp_dir = TempDir::new().unwrap();
    let config = RocksDbConfig {
        jmt_history_length: 2,
        ..Default::default()
    };
    let storage =
        RocksDbShardStorage::open_with_config(temp_dir.path(), &config, NibblePath::empty())
            .unwrap();

    for h in 1..=10u64 {
        let block = make_test_block(BlockHeight::new(h));
        let qc = make_test_qc(&block);
        commit_empty(&storage, &block, &qc);
    }
    // current=10, floor=8. V=1 is well below floor.
    let _snap = <RocksDbShardStorage as VersionedStore>::snapshot_at(&storage, BlockHeight::new(1));
}

/// `get_substate_at_height` is an external-facing API — it must
/// return `None` for out-of-retention heights rather than panicking.
#[test]
fn test_historical_substate_read_respects_retention() {
    let temp_dir = TempDir::new().unwrap();
    let config = RocksDbConfig {
        jmt_history_length: 2,
        ..Default::default()
    };
    let storage =
        RocksDbShardStorage::open_with_config(temp_dir.path(), &config, NibblePath::empty())
            .unwrap();

    let key = SubstateKey {
        owner: Address::new([9u8; 31], AddressClass::Component),
        local: LocalKey([1u8; 16]),
    };

    for h in 1..=10u64 {
        let block = make_test_block(BlockHeight::new(h));
        let qc = make_test_qc(&block);
        let writes = SettledWrites::from_absolutes(BTreeMap::from([(
            key,
            Some(vec![u8::try_from(h).unwrap_or(u8::MAX)]),
        )]));
        rocks_commit_with(&storage, &writes, &block, &qc);
    }
    // current=10, floor=8.

    // Within retention: returns Some.
    assert_eq!(
        storage.get_substate_at_height(key, BlockHeight::new(9)),
        Some(Some(vec![9])),
        "height within retention must succeed"
    );
    // Below retention: returns None.
    assert!(
        storage
            .get_substate_at_height(key, BlockHeight::new(1))
            .is_none(),
        "height below retention must return None"
    );
    // Above current: returns None.
    assert!(
        storage
            .get_substate_at_height(key, BlockHeight::new(99))
            .is_none(),
        "future height returns None"
    );
}

/// Genesis-style writes via `commit_substates_only` must NOT populate the
/// state-history CF — there is no pre-state to preserve.
#[test]
fn test_genesis_skips_history_entries() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();

    let updates = make_settled_writes(1, 1, vec![0xAA]);
    storage.commit_substates_only(&updates);

    // StateHistoryCf must be empty after a genesis-style commit.
    let history_count = {
        let cf = storage
            .db
            .cf_handle(STATE_HISTORY_CF)
            .expect("state_history CF exists");
        let mut iter = storage.db.raw_iterator_cf(cf);
        iter.seek_to_first();
        let mut n = 0usize;
        while iter.valid() {
            n += 1;
            iter.next();
        }
        n
    };
    assert_eq!(
        history_count, 0,
        "commit_substates_only must not record state-history entries"
    );

    // StateCf must hold the genesis write (readable via current-tip snapshot).
    assert_eq!(storage.substate(state_key(1, 1)), Some(vec![0xAA]));
}

/// Witness retention follows the commit-carried floor (a `WriteBatch`
/// range delete) with one window of hysteresis, and recovery rebuilds
/// the accumulator window from the tip header's base — entries below it
/// are serving stock only.
#[test]
fn witness_window_retention_and_recovery() {
    use hyperscale_storage::test_helpers::{commit_block_with_witness_window, stake_deposit};
    use hyperscale_types::ShardWitnessPayload;

    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
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
        high_qc: None,
    }
}

#[test]
fn safe_vote_registers_recover_their_justification() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    test_registers_recover_their_justification(&storage, || storage.load_recovered_state());
}

/// Persisted registers read back, survive a reopen, and land in
/// `load_recovered_state`.
#[test]
fn safe_vote_registers_survive_reopen() {
    let temp_dir = TempDir::new().unwrap();
    let v1 = ValidatorId::new(1);
    let v2 = ValidatorId::new(2);
    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        storage.persist_safe_vote_registers(v1, registers(3, 5));
        storage.persist_safe_vote_registers(v2, registers(2, 2));
        assert_eq!(storage.safe_vote_registers(v1), Some(registers(3, 5)));
    }

    let reopened = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    assert_eq!(reopened.safe_vote_registers(v1), Some(registers(3, 5)));
    let recovered = reopened.load_recovered_state();
    assert_eq!(
        recovered.safe_vote_registers.get(&v1),
        Some(&registers(3, 5))
    );
    assert_eq!(
        recovered.safe_vote_registers.get(&v2),
        Some(&registers(2, 2))
    );
}

/// Writes merge field-wise max, so a lower or mixed write can never
/// regress either register — including on a cold write-path cache after
/// a reopen, where the merge must consult the stored record.
#[test]
fn safe_vote_registers_writes_are_monotone() {
    let temp_dir = TempDir::new().unwrap();
    let v = ValidatorId::new(7);
    {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        storage.persist_safe_vote_registers(v, registers(4, 6));
        storage.persist_safe_vote_registers(v, registers(2, 9));
        assert_eq!(storage.safe_vote_registers(v), Some(registers(4, 9)));
    }

    let reopened = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    reopened.persist_safe_vote_registers(v, registers(3, 3));
    assert_eq!(reopened.safe_vote_registers(v), Some(registers(4, 9)));
}

/// A record written under a different chain origin is invisible to
/// reads and recovery — a checkpoint-seeded child store inherits the
/// parent's records but must not apply them to the child chain's
/// unrelated round numbering. The next write starts a fresh record
/// under the new origin.
#[test]
fn safe_vote_registers_ignore_stale_chain_incarnation() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    let v = ValidatorId::new(1);
    storage.persist_safe_vote_registers(v, registers(8, 8));

    let mut batch = WriteBatch::default();
    write_chain_origin(
        &mut batch,
        ChainOrigin {
            genesis_height: BlockHeight::new(11),
            anchor_wt: WeightedTimestamp::from_millis(999),
        },
    );
    storage.db.write(batch).unwrap();

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

#[test]
fn recovery_carries_the_tip_drain_total() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    test_recovery_carries_the_tip_drain_total(&storage, || storage.load_recovered_state());
}

#[test]
fn a_committed_bundle_outlives_its_block_s_sealing() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    test_committed_bundle_outlives_sealing(&storage, || storage.load_recovered_state());
}

/// Storing the bodies is only worth anything if they outlive the process
/// that committed them, which no in-memory backend can show.
#[test]
fn a_committed_bundle_survives_a_reopen() {
    let temp_dir = TempDir::new().unwrap();
    let hash = {
        let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
        let block = with_provisions(
            make_test_block(BlockHeight::new(1)),
            ShardId::leaf(1, 1),
            TxHash::ZERO,
        );
        let hash = block.provisions()[0].hash();
        storage.commit_block(&make_test_certified(block), &no_witness());
        hash
    };

    let reopened = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    assert_eq!(
        reopened
            .load_recovered_state()
            .retained_provisions
            .iter()
            .map(|bundle| bundle.hash())
            .collect::<Vec<_>>(),
        vec![hash],
    );
}

#[test]
fn a_retained_bundle_drops_below_the_history_floor() {
    let temp_dir = TempDir::new().unwrap();
    let config = RocksDbConfig {
        jmt_history_length: 3,
        ..RocksDbConfig::default()
    };
    let storage =
        RocksDbShardStorage::open_with_config(temp_dir.path(), &config, NibblePath::empty())
            .unwrap();
    test_retained_bundle_drops_below_the_history_floor(&storage, 3, || {
        storage.load_recovered_state()
    });
}

#[test]
fn the_widest_copy_of_a_tick_holds_the_slot() {
    let temp_dir = TempDir::new().unwrap();
    let storage = RocksDbShardStorage::open(temp_dir.path(), NibblePath::empty()).unwrap();
    test_widest_tick_copy_holds_the_slot(&storage);
}
