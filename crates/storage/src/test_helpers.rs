//! Shared test helpers for storage crate tests.
//!
//! Provides reusable builder functions for [`StateWrites`],
//! `Finalization`, `Block`, and `QuorumCertificate` so that
//! storage-memory and storage-rocksdb tests can share a single source of truth.

use std::collections::{BTreeMap, HashSet};
use std::slice::from_ref;
use std::sync::Arc;

use hyperscale_hbor::from_slice;
use hyperscale_jmt::{KEY_BYTES, TreeReader};
use hyperscale_types::test_utils::{
    install_stub_protocol_statics, make_finalization, make_leg_finalization, stub_sweepable_cell,
    test_transaction,
};
use hyperscale_types::{
    AbandonmentRecord, AbortCharge, Address, AddressClass, AggregateSignature, BeaconBlock,
    BeaconBlockHash, BeaconCert, BeaconChainConfig, BeaconState, BeaconWitnessCommit,
    BeaconWitnessLeafCount, BeaconWitnessRoot, Block, BlockHash, BlockHeader, BlockHeaderParts,
    BlockHeight, CertifiedBeaconBlock, CertifiedBlock, ChainOrigin, CollectionId, ConsensusReceipt,
    EntryKey, EntryLeaf, Epoch, Event, ExecutionCertificate, ExecutionMetadata, ExecutionOutcome,
    FeeSummary, Finalization, GlobalReceiptHash, GlobalReceiptRoot, Hash, LocalKey, LogLevel,
    MerkleInclusionProof, PcQc2, PcQc3, PcSignerLengths, PcVector, PcXpProof, ProposerTimestamp,
    ProtocolHasher, ProvisionEntry, ProvisionHash, Provisions, QuorumCertificate, Randomness,
    RatifyCert, RatifyRound, Round, SWEEP_BUCKET_MS, SafeVoteRegisters, SettledWrites, ShardAnchor,
    ShardId, ShardWitnessPayload, SignerBitfield, SpcCert, SpcView, Stake, StakePoolId, StateRoot,
    StateWrites, StoredReceipt, SubstateKey, SubstateLeaf, SweepBucket, SweepFrontier, SyncHint,
    TickHalf, TickId, Transaction, TransactionDecision, TxHash, TxOutcome, UnsettledTx,
    ValidatorId, Verifiable, Verified, WeightedTimestamp, WitnessSources, WorkInFlight,
    compute_global_receipt_root, compute_merkle_root, entry_leaf_key,
};

use crate::shard::unresolved::{replay_window, unresolved_replay_floor};
use crate::tree::Jmt;
use crate::{
    Anchored, BOUNDARY_RETAIN, BoundaryStore, ImportCursor, ImportProgress, JmtSnapshot,
    ParentAnchor, RecoveredState, SafeVoteRegisterStore, ShardChainReader, ShardChainWriter,
    SubstateStore, Substates, SweepIndex, VersionedStore, WitnessSeed, committed_tx_cell_key,
    committed_tx_cells, sweep_for_block,
};

/// The state a parent left, where the parent is certified but not yet
/// persisted: the store's own tip, overlaid with what the blocks between
/// have settled, answering for the height they reach.
///
/// This is what a [`SubstateView`](crate::SubstateView) is in production,
/// spelled small enough for a backend's own tests. It exists because
/// `ParentAnchor::state` must answer for the parent's height and a store
/// snapshot answers for the last block it wrote — the two differ exactly
/// while a proposer is building on blocks its own disk has not caught up
/// with, which is the case worth having a fixture for.
pub struct PendingBaseline<Snap> {
    base: Snap,
    settled: BTreeMap<SubstateKey, Option<Vec<u8>>>,
    anchor: BlockHeight,
}

impl<Snap: Substates> PendingBaseline<Snap> {
    /// `base` overlaid with `pending`'s settled cells, anchored at
    /// `height`.
    #[must_use]
    pub fn new(base: Snap, pending: &[Arc<JmtSnapshot>], height: BlockHeight) -> Self {
        let mut settled = BTreeMap::new();
        for snapshot in pending {
            for (key, change) in snapshot.settled.cells() {
                settled.insert(*key, change.clone());
            }
        }
        Self {
            base,
            settled,
            anchor: height,
        }
    }
}

impl<Snap: Substates> Anchored for PendingBaseline<Snap> {
    fn anchor(&self) -> BlockHeight {
        self.anchor
    }
}

impl<Snap: Substates> Substates for PendingBaseline<Snap> {
    fn cell(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.settled
            .get(&key)
            .map_or_else(|| self.base.cell(key), Clone::clone)
    }

    fn entries_in_range(
        &self,
        owner: Address,
        collection: CollectionId,
        lo: u128,
        hi: u128,
        limit: usize,
    ) -> Vec<(u128, Vec<u8>)> {
        self.base.entries_in_range(owner, collection, lo, hi, limit)
    }
}

/// A completed [`ImportProgress`] covering the whole key span as one
/// exhausted sub-range — the record a one-shot staged import carries.
#[must_use]
pub fn completed_import_progress(height: BlockHeight, staged_bytes: u64) -> ImportProgress {
    ImportProgress {
        anchor_height: height,
        anchor_state_root: StateRoot::ZERO,
        split_bits: 0,
        chunk_limit: 0,
        staged_bytes,
        cursors: vec![ImportCursor {
            next: [0u8; KEY_BYTES],
            end: [0xFF; KEY_BYTES],
            done: true,
        }],
    }
}

/// Stage `leaves` as one chunk and finalize — the one-shot import shape
/// for tests where streaming granularity is irrelevant.
///
/// # Errors
///
/// Propagates the staging or finalize failure.
pub fn import_boundary_state<S: BoundaryStore>(
    storage: &S,
    height: BlockHeight,
    leaves: &[SubstateLeaf],
    witnesses: WitnessSeed,
) -> Result<StateRoot, String> {
    let staged_bytes = leaves.iter().map(|l| l.value.len() as u64).sum();
    storage.stage_import_chunk(&completed_import_progress(height, staged_bytes), leaves)?;
    storage.finalize_boundary_import(height, witnesses)
}

/// A receipt's writes holding one cell: owner `[owner_seed; 16]`, local
/// zero-padded from `local_seed`.
#[must_use]
pub fn make_state_writes(owner_seed: u8, local_seed: u8, value: Vec<u8>) -> StateWrites {
    let mut writes = StateWrites::default();
    writes
        .cells
        .insert(state_key(owner_seed, local_seed), Some(value));
    writes
}

/// The same cell as a value a store can commit directly, for the tests
/// that skip the receipt path.
#[must_use]
pub fn make_settled_writes(owner_seed: u8, local_seed: u8, value: Vec<u8>) -> SettledWrites {
    SettledWrites::from_absolutes(BTreeMap::from([(
        state_key(owner_seed, local_seed),
        Some(value),
    )]))
}

/// A settled set carrying only ordered-collection entries of one
/// collection, for the tests that exercise the entry pipeline.
#[must_use]
pub fn make_settled_entries(owner_seed: u8, entries: &[(u128, Option<Vec<u8>>)]) -> SettledWrites {
    SettledWrites::from_parts(
        BTreeMap::new(),
        entries
            .iter()
            .map(|(order, change)| (entry_key(owner_seed, *order), change.clone()))
            .collect(),
    )
}

/// The entry key for `order` in the test collection under owner
/// `[owner_seed; 31]` — the identity [`make_settled_entries`] writes
/// under.
#[must_use]
pub const fn entry_key(owner_seed: u8, order: u128) -> EntryKey {
    EntryKey {
        owner: Address::new([owner_seed; 31], AddressClass::Component),
        collection: CollectionId([0xEE; 16]),
        order,
    }
}

/// The substate key for owner `[owner_seed; 16]`, local zero-padded from
/// `local_seed` — the key [`make_state_writes`] writes under.
#[must_use]
pub const fn state_key(owner_seed: u8, local_seed: u8) -> SubstateKey {
    let mut local = [0u8; 16];
    local[0] = local_seed;
    SubstateKey {
        owner: Address::new([owner_seed; 31], AddressClass::Component),
        local: LocalKey(local),
    }
}

/// Build a test attestation at the given height.
///
/// Includes a single placeholder local EC so it satisfies the invariant
/// enforced at decode time (one EC whose `tick_id` matches the tick's own).
#[must_use]
pub fn make_test_finalization(height: BlockHeight, shard: ShardId) -> Finalization {
    let tick_id = TickId::new(shard, height);
    let local_ec = Arc::new(ExecutionCertificate::new(
        tick_id,
        WeightedTimestamp::from_millis(0),
        GlobalReceiptRoot::ZERO,
        Vec::new(),
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::empty(),
    ));
    Finalization::new(tick_id, TickHalf::Determined, vec![local_ec], vec![])
}

/// Build a minimal `Block` at the given height.
#[must_use]
pub fn make_test_block(height: BlockHeight) -> Block {
    make_test_block_with_anchor_wt(height, 0)
}

/// Build a single-validator test block at `height` whose `parent_qc` carries
/// `anchor_wt_ms` as its weighted timestamp — the canonical clock anchor a
/// block exposes for window and floor tests.
#[must_use]
pub fn make_test_block_with_anchor_wt(height: BlockHeight, anchor_wt_ms: u64) -> Block {
    // Use the full u64 bytes for the parent hash so heights > 255 don't alias.
    let mut parent_bytes = [0u8; 32];
    parent_bytes[..8].copy_from_slice(&height.to_le_bytes());
    let parent_qc = QuorumCertificate::genesis(
        ShardId::ROOT,
        ChainOrigin {
            anchor_wt: WeightedTimestamp::from_millis(anchor_wt_ms),
            ..ChainOrigin::ROOT
        },
    );
    Block::Live {
        header: BlockHeader::new(BlockHeaderParts {
            height,
            parent_block_hash: BlockHash::from_raw(Hash::from_bytes(&parent_bytes)),
            parent_qc: parent_qc.into(),
            timestamp: ProposerTimestamp::from_millis(height.inner() * 1000),
            ..Default::default()
        }),
        transactions: Arc::new(Vec::new()),
        certificates: Arc::new(Vec::new()),
        provisions: Arc::new(Vec::new()),
        abandonment_records: Arc::new(Vec::new()),
        state_proofs: Arc::new(Vec::new()),
        witness_sources: Arc::new(WitnessSources::empty()),
    }
}

/// Build a verified `QuorumCertificate` that references the given block.
///
/// The signature is the zero placeholder — these fixtures don't drive real
/// verification, they exercise storage and pipeline shapes. The `Verified`
/// wrapper is `new_unchecked` because the test cluster predates a real
/// signing path; consumers downstream of storage and the commit pipeline
/// require the verified marker.
#[must_use]
pub fn make_test_qc(block: &Block) -> Verified<QuorumCertificate> {
    // SAFETY: synthetic test fixture, no real signature.
    Verified::<QuorumCertificate>::new_unchecked_for_test(QuorumCertificate::new(
        block.hash(),
        ShardId::ROOT,
        block.height(),
        block.header().parent_block_hash(),
        Round::INITIAL,
        SignerBitfield::new(4),
        AggregateSignature::ZERO,
        WeightedTimestamp::from_millis(block.header().timestamp().as_millis()),
    ))
}

/// Build a `Verified<CertifiedBlock>` for use with `commit_block` and the
/// commit-pipeline test fixtures.
///
/// # Panics
///
/// Panics if internal `CertifiedBlock` construction fails — only happens
/// when callers feed a `qc` whose `block_hash` doesn't match `block`, which
/// the helper precludes by construction.
#[must_use]
pub fn make_test_certified(block: Block) -> Arc<Verified<CertifiedBlock>> {
    let qc = make_test_qc(&block);
    let certified = CertifiedBlock::new_unchecked(block, qc);
    // SAFETY: synthetic test fixture; storage round-trip tests don't
    // exercise the `Verified<CertifiedBlock>` predicate.
    Arc::new(Verified::<CertifiedBlock>::new_unchecked_for_test(
        certified,
    ))
}

/// Build a placeholder [`SpcCert::Direct`] for test fixtures.
///
/// The embedded `PcQc3` is structurally well-formed but doesn't
/// verify; the cert is intended for storage round-trip tests, not
/// consensus verification.
#[must_use]
fn placeholder_cert() -> SpcCert {
    let qc2 = PcQc2::new(
        PcVector::empty(),
        SignerBitfield::new(4),
        AggregateSignature::new([0x11; 96]),
        PcXpProof::Full,
    );
    let proof = PcQc3::new(
        PcVector::empty(),
        qc2,
        None,
        None,
        SignerBitfield::new(4),
        PcSignerLengths::Uniform(0),
        AggregateSignature::new([0x33; 96]),
    );
    SpcCert::Direct {
        prev_view: SpcView::new(1),
        value: PcVector::empty(),
        proof: proof.into(),
    }
}

/// Build a certified beacon block at `epoch` with tag-derived
/// `prev_block_hash`.
///
/// The cert is a structurally-valid but cryptographically-unverified
/// placeholder. Suitable for storage round-trip tests, not for
/// consensus verification.
#[must_use]
pub fn make_test_beacon_block(epoch: u64, tag: &[u8]) -> Arc<Verified<CertifiedBeaconBlock>> {
    let block = BeaconBlock::new(
        Epoch::new(epoch),
        BeaconBlockHash::from_raw(Hash::from_bytes(tag)),
        Vec::new(),
    );
    let ratify = RatifyCert::new(
        block.prev_block_hash(),
        block.epoch(),
        RatifyRound::INITIAL,
        block.block_hash(),
        SignerBitfield::new(4),
        AggregateSignature::new([0x22; 96]),
    );
    Arc::new(Verified::new_unchecked_for_test(
        CertifiedBeaconBlock::new_unchecked(
            block,
            BeaconCert::Normal {
                spc: Box::new(placeholder_cert()),
                ratify,
            },
        ),
    ))
}

/// Build a minimal `BeaconState` at `epoch` whose `randomness` is
/// derived from `tag`. All collection fields are empty.
///
/// Sufficient to drive storage round-trip tests — every field is
/// stable across HBOR encoding and two calls with identical inputs
/// produce equal states. Not a valid state under beacon-state
/// verification.
#[must_use]
pub fn make_test_beacon_state(epoch: u64, tag: &[u8]) -> Arc<BeaconState> {
    let mut randomness = [0u8; 32];
    let copy_len = tag.len().min(32);
    randomness[..copy_len].copy_from_slice(&tag[..copy_len]);
    let mut state = BeaconState::empty(BeaconChainConfig::default());
    state.current_epoch = Epoch::new(epoch);
    state.randomness = Randomness::new(randomness);
    Arc::new(state)
}

/// Build a `(block, state)` pair for storage round-trip tests.
///
/// Under the cert-as-authenticator model the block's `state_root` is
/// no longer carried on-chain (it's derived by re-running `apply_epoch`),
/// so this helper just produces a structurally well-formed block paired
/// with an arbitrary state.
#[must_use]
pub fn make_test_block_and_state(
    epoch: u64,
    tag: &[u8],
) -> (Arc<Verified<CertifiedBeaconBlock>>, Arc<BeaconState>) {
    let state = make_test_beacon_state(epoch, tag);
    let block = make_test_beacon_block(epoch, tag);
    (block, state)
}

/// Build a deterministic locally-executed `StoredReceipt` from `seed`
/// — succeeded, with a single event and a non-empty fee summary so
/// equality checks across seeds distinguish entries.
#[must_use]
pub fn make_test_receipt(seed: u8) -> StoredReceipt {
    let tx_hash = TxHash::from(Hash::from_bytes(&[seed; 32]));
    let consensus = ConsensusReceipt::Succeeded {
        receipt_hash: GlobalReceiptHash::ZERO,
        writes: StateWrites::default(),
        beacon_witness_events: Vec::new(),
        events: vec![Event {
            emitter: Address::new([seed; 31], AddressClass::Component),
            event_type: u32::from(seed),
            payload: vec![seed, seed + 1],
        }],
    };
    let metadata = Some(ExecutionMetadata::new(
        FeeSummary {
            total_execution_cost: Some(u128::from(seed) * Stake::ATTOS_PER_WHOLE),
            total_royalty_cost: None,
            total_storage_cost: None,
            total_tipping_cost: None,
        },
        vec![(LogLevel::Info, format!("tx {seed}"))],
        None,
    ));
    StoredReceipt {
        tx_hash,
        consensus: Arc::new(consensus),
        metadata,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Execution Certificate helpers
// ═══════════════════════════════════════════════════════════════════════

/// Build a test `ExecutionCertificate` at the given block height with a
/// deterministic outcome derived from `seed`.
///
/// `seed` also disambiguates the `TickId` (via `remote_shards`), so two ECs
/// at the same `block_height` with different seeds have distinct identities
/// — matching the protocol invariant that one tick produces one EC.
#[must_use]
pub fn make_test_execution_certificate(
    seed: u8,
    block_height: BlockHeight,
) -> ExecutionCertificate {
    let outcomes = vec![TxOutcome::new(
        TxHash::from(Hash::from_bytes(&[seed + 100; 32])),
        ExecutionOutcome::Succeeded {
            receipt_hash: GlobalReceiptHash::from_raw(Hash::from_bytes(&[seed + 150; 32])),
        },
    )];
    let global_receipt_root = compute_global_receipt_root(&outcomes);
    ExecutionCertificate::new(
        TickId::new(ShardId::ROOT, block_height),
        WeightedTimestamp::from_millis(block_height.inner() + 1),
        global_receipt_root,
        outcomes,
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::new(4),
    )
}

/// Build a test block that carries ECs inside its finalizations.
///
/// The tick's `tick_id` is taken from the first EC's `tick_id` so
/// the local-EC decode invariant is satisfied without injecting a placeholder.
fn make_test_block_with_ecs(height: BlockHeight, ecs: Vec<Arc<ExecutionCertificate>>) -> Block {
    let block = make_test_block(height);
    if ecs.is_empty() {
        return block;
    }
    let certificate = Finalization::new(*ecs[0].tick_id(), TickHalf::Determined, ecs, vec![]);
    push_certificate(block, Arc::new(certificate.into()))
}

/// Append a finalization to `block`'s certificate list, preserving
/// the block variant.
fn push_certificate(block: Block, fw: Arc<Verifiable<Finalization>>) -> Block {
    match block {
        Block::Live {
            header,
            transactions,
            certificates,
            provisions,
            abandonment_records,
            state_proofs,
            witness_sources,
        } => {
            let mut certificates = (*certificates).clone();
            certificates.push(fw);
            Block::Live {
                header,
                transactions,
                certificates: Arc::new(certificates),
                provisions,
                abandonment_records,
                state_proofs,
                witness_sources,
            }
        }
        Block::Sealed {
            header,
            transactions,
            certificates,
            provision_hashes,
            abandonment_records,
            state_proofs,
            witness_sources,
        } => {
            let mut certificates = (*certificates).clone();
            certificates.push(fw);
            Block::Sealed {
                header,
                transactions,
                certificates: Arc::new(certificates),
                provision_hashes,
                abandonment_records,
                state_proofs,
                witness_sources,
            }
        }
    }
}

/// Put `record` on `block`, preserving the block variant.
fn with_abandonment(block: Block, record: AbandonmentRecord) -> Block {
    match block {
        Block::Live {
            header,
            transactions,
            certificates,
            provisions,
            state_proofs,
            witness_sources,
            ..
        } => Block::Live {
            header,
            transactions,
            certificates,
            provisions,
            abandonment_records: Arc::new(vec![record]),
            state_proofs,
            witness_sources,
        },
        Block::Sealed {
            header,
            transactions,
            certificates,
            provision_hashes,
            state_proofs,
            witness_sources,
            ..
        } => Block::Sealed {
            header,
            transactions,
            certificates,
            provision_hashes,
            abandonment_records: Arc::new(vec![record]),
            state_proofs,
            witness_sources,
        },
    }
}

/// Helper to commit empty blocks up to (but not including) the target height.
fn commit_empty_blocks_up_to(storage: &impl ShardChainWriter, target: BlockHeight) {
    let witness = empty_witness();
    for h in 0..target.inner() {
        let certified = make_test_certified(make_test_block(BlockHeight::new(h)));
        storage.commit_block(&certified, &[], &witness);
    }
}

/// Commit `writes` at `height` through the production block-commit path.
///
/// The writes ride a single-receipt finalization inside a test block,
/// so substates, state history, and the JMT all land exactly as a live
/// commit writes them. Returns the resulting state root.
pub fn commit_block_with_updates(
    storage: &impl ShardChainWriter,
    height: BlockHeight,
    writes: &StateWrites,
) -> StateRoot {
    let receipt = StoredReceipt {
        tx_hash: TxHash::ZERO,
        consensus: Arc::new(ConsensusReceipt::Succeeded {
            receipt_hash: GlobalReceiptHash::ZERO,
            writes: writes.clone(),
            beacon_witness_events: Vec::new(),
            events: Vec::new(),
        }),
        metadata: None,
    };
    let finalized = Arc::new(
        Finalization::new(
            TickId::new(ShardId::ROOT, height),
            TickHalf::Determined,
            vec![],
            vec![receipt],
        )
        .into(),
    );
    let block = push_certificate(make_test_block(height), finalized);
    storage.commit_block(&make_test_certified(block), &[], &empty_witness())
}

/// A block at `height` whose one tick settles `receipts` and which
/// carries no transactions — what a follow applies when only receipts
/// move state.
#[must_use]
pub fn block_settling(height: BlockHeight, receipts: Vec<StoredReceipt>) -> Block {
    let finalized = Arc::new(
        Finalization::new(
            TickId::new(ShardId::ROOT, height),
            TickHalf::Determined,
            vec![],
            receipts,
        )
        .into(),
    );
    push_certificate(make_test_block(height), finalized)
}

const fn empty_witness() -> BeaconWitnessCommit {
    BeaconWitnessCommit::empty(BeaconWitnessLeafCount::ZERO)
}

/// Commit a block at `height` whose header commits the beacon-witness
/// accumulator state after appending `leaves`.
///
/// The header carries the leaves' merkle root and cumulative count, and
/// the leaves fold into the same atomic write. Returns the committed
/// block hash.
pub fn commit_block_with_witnesses(
    storage: &impl ShardChainWriter,
    height: BlockHeight,
    leaves: &[ShardWitnessPayload],
) -> BlockHash {
    let leaf_hashes: Vec<Hash> = leaves.iter().map(ShardWitnessPayload::leaf_hash).collect();
    let root = BeaconWitnessRoot::from_raw(compute_merkle_root(&leaf_hashes));
    let count = BeaconWitnessLeafCount::new(leaves.len() as u64);
    let mut parent_bytes = [0u8; 32];
    parent_bytes[..8].copy_from_slice(&height.to_le_bytes());
    let block = Block::Live {
        header: BlockHeader::new(BlockHeaderParts {
            height,
            parent_block_hash: BlockHash::from_raw(Hash::from_bytes(&parent_bytes)),
            parent_qc: QuorumCertificate::genesis(ShardId::ROOT, ChainOrigin::ROOT).into(),
            timestamp: ProposerTimestamp::from_millis(height.inner() * 1000),
            beacon_witness_root: root,
            beacon_witness_leaf_count: count,
            ..Default::default()
        }),
        transactions: Arc::new(Vec::new()),
        certificates: Arc::new(Vec::new()),
        provisions: Arc::new(Vec::new()),
        abandonment_records: Arc::new(Vec::new()),
        state_proofs: Arc::new(Vec::new()),
        witness_sources: Arc::new(WitnessSources::empty()),
    };
    let block_hash = block.hash();
    let witness = BeaconWitnessCommit {
        starting_leaf_index: BeaconWitnessLeafCount::ZERO,
        leaves: leaves.to_vec(),
        leaf_count_at_block_end: count,
        prune_persisted_below: None,
    };
    storage.commit_block(&make_test_certified(block), &[], &witness);
    block_hash
}

/// Commit a block at `height` whose header commits the witness window
/// `[base, base + window.len())`.
///
/// The header carries the root over `window`, the cumulative count, and
/// `base` as its window base. The commit appends `appended` (the
/// window's tail at `base + window.len() - appended.len()`) and carries
/// `prune_persisted_below` so backend retention behaviour is
/// observable. Returns the committed block hash.
///
/// # Panics
///
/// Panics if `appended` is longer than `window` — the appended tail
/// must lie inside the committed window.
pub fn commit_block_with_witness_window(
    storage: &impl ShardChainWriter,
    height: BlockHeight,
    base: u64,
    window: &[ShardWitnessPayload],
    appended: &[ShardWitnessPayload],
    prune_persisted_below: Option<BeaconWitnessLeafCount>,
) -> BlockHash {
    assert!(appended.len() <= window.len());
    let leaf_hashes: Vec<Hash> = window.iter().map(ShardWitnessPayload::leaf_hash).collect();
    let root = BeaconWitnessRoot::from_raw(compute_merkle_root(&leaf_hashes));
    let count = BeaconWitnessLeafCount::new(base + window.len() as u64);
    let mut parent_bytes = [0u8; 32];
    parent_bytes[..8].copy_from_slice(&height.to_le_bytes());
    let block = Block::Live {
        header: BlockHeader::new(BlockHeaderParts {
            height,
            parent_block_hash: BlockHash::from_raw(Hash::from_bytes(&parent_bytes)),
            parent_qc: QuorumCertificate::genesis(ShardId::ROOT, ChainOrigin::ROOT).into(),
            timestamp: ProposerTimestamp::from_millis(height.inner() * 1000),
            beacon_witness_root: root,
            beacon_witness_leaf_count: count,
            beacon_witness_base: BeaconWitnessLeafCount::new(base),
            ..Default::default()
        }),
        transactions: Arc::new(Vec::new()),
        certificates: Arc::new(Vec::new()),
        provisions: Arc::new(Vec::new()),
        abandonment_records: Arc::new(Vec::new()),
        state_proofs: Arc::new(Vec::new()),
        witness_sources: Arc::new(WitnessSources::empty()),
    };
    let block_hash = block.hash();
    let witness = BeaconWitnessCommit {
        starting_leaf_index: BeaconWitnessLeafCount::new(count.inner() - appended.len() as u64),
        leaves: appended.to_vec(),
        leaf_count_at_block_end: count,
        prune_persisted_below,
    };
    storage.commit_block(&make_test_certified(block), &[], &witness);
    block_hash
}

/// A `ShardWitnessPayload::StakeDeposit` fixture.
#[must_use]
pub const fn stake_deposit(amount: u64) -> ShardWitnessPayload {
    ShardWitnessPayload::StakeDeposit {
        pool_id: StakePoolId::new(1),
        amount: Stake::from_whole_tokens(amount),
    }
}

/// The owner prefix seeded at `seed`, with the leading bit alternating.
///
/// A leaf key's leading bit is the first bit a depth-1 shard prefix
/// routes on, and a leaf key is its owner and local halves by identity —
/// so alternating the owner's top bit is what makes a seeded population
/// straddle the root split rather than piling into one child.
#[must_use]
pub const fn seeded_owner(seed: u8) -> u8 {
    if seed.is_multiple_of(2) {
        seed
    } else {
        seed | 0x80
    }
}

/// Seed `entries` single-substate block commits at heights
/// `1..=entries`, each writing one distinct owner keyed by
/// [`seeded_owner`] of its seed byte.
pub fn seed_substate_commits(storage: &impl ShardChainWriter, entries: u8) {
    for seed in 1..=entries {
        let writes = make_state_writes(seeded_owner(seed), seed, vec![seed, seed, seed]);
        commit_block_with_updates(storage, BlockHeight::new(u64::from(seed)), &writes);
    }
}

/// A snap-sync serving replica.
///
/// Seeds `entries` substate commits, then a boundary block at
/// `entries + 1` whose header carries the witness commitment over
/// `leaves`, pinned for serving. Returns the anchor a joiner verifies
/// against.
///
/// # Panics
///
/// Panics if pinning fails (this is a test helper).
pub fn pin_snap_sync_replica(
    storage: &(impl ShardChainWriter + BoundaryStore + SubstateStore),
    entries: u8,
    leaves: &[ShardWitnessPayload],
) -> ShardAnchor {
    seed_substate_commits(storage, entries);
    let anchor_height = BlockHeight::new(u64::from(entries) + 1);
    let block_hash = commit_block_with_witnesses(storage, anchor_height, leaves);
    storage.pin_boundary(anchor_height).unwrap();
    ShardAnchor {
        state_root: storage.state_root(),
        block_hash,
        height: anchor_height,
        weighted_timestamp: WeightedTimestamp::from_millis(anchor_height.inner()),
        witness_base: BeaconWitnessLeafCount::ZERO,
        terminal_roots: None,
        handoff_complete: None,
    }
}

/// Shared range-read test for `get_beacon_witness_payload_range`.
///
/// The range read must agree with the full prefix read on interior
/// pages, clamp nothing (callers bound `end`), and return empty for
/// degenerate or out-of-range spans.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_witness_payload_range_reads(storage: &(impl ShardChainReader + ShardChainWriter)) {
    let leaves: Vec<ShardWitnessPayload> = (1u64..=5).map(stake_deposit).collect();
    commit_block_with_witnesses(storage, BlockHeight::new(1), &leaves);

    let all = storage.get_beacon_witness_payloads(BeaconWitnessLeafCount::new(5));
    assert_eq!(all.len(), 5);
    assert_eq!(storage.get_beacon_witness_payload_range(0, 5), all);
    assert_eq!(storage.get_beacon_witness_payload_range(1, 3), all[1..3]);
    assert_eq!(storage.get_beacon_witness_payload_range(4, 9), all[4..]);
    assert!(storage.get_beacon_witness_payload_range(3, 3).is_empty());
    assert!(storage.get_beacon_witness_payload_range(7, 9).is_empty());
}

/// Shared EC roundtrip test: commit a block carrying an EC, then read it
/// back by `tick_id`.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_ec_storage_roundtrip(storage: &(impl ShardChainReader + ShardChainWriter)) {
    let ec = make_test_execution_certificate(1, BlockHeight::new(10));
    let tick_id = *ec.tick_id();

    // Initially absent.
    assert!(storage.get_execution_certificate(&tick_id).is_none());

    commit_empty_blocks_up_to(storage, BlockHeight::new(10));
    let block = make_test_block_with_ecs(BlockHeight::new(10), vec![Arc::new(ec)]);
    let certified = make_test_certified(block);
    storage.commit_block(&certified, &[], &empty_witness());

    let direct = storage
        .get_execution_certificate(&tick_id)
        .expect("EC must be retrievable by tick_id");
    assert_eq!(direct.tick_id(), &tick_id);
    assert_eq!(direct.block_height(), BlockHeight::new(10));
}

/// Shared EC batch test: commit two ECs at one height plus one at another,
/// confirm batch read returns hits and skips misses.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_ec_storage_batch(storage: &(impl ShardChainReader + ShardChainWriter)) {
    let ec1 = make_test_execution_certificate(1, BlockHeight::new(10));
    let ec2 = make_test_execution_certificate(2, BlockHeight::new(10));
    let ec3 = make_test_execution_certificate(3, BlockHeight::new(20));

    commit_empty_blocks_up_to(storage, BlockHeight::new(10));
    let block10 = make_test_block_with_ecs(
        BlockHeight::new(10),
        vec![Arc::new(ec1.clone()), Arc::new(ec2.clone())],
    );
    storage.commit_block(&make_test_certified(block10), &[], &empty_witness());

    for h in 11..20 {
        let certified = make_test_certified(make_test_block(BlockHeight::new(h)));
        storage.commit_block(&certified, &[], &empty_witness());
    }
    let block20 = make_test_block_with_ecs(BlockHeight::new(20), vec![Arc::new(ec3.clone())]);
    storage.commit_block(&make_test_certified(block20), &[], &empty_witness());

    let known = [*ec1.tick_id(), *ec2.tick_id(), *ec3.tick_id()];
    let batch = storage.get_execution_certificates_batch(&known);
    assert_eq!(batch.len(), 3);

    let missing_tick_id = TickId::new(known[0].shard_id(), BlockHeight::new(999));
    let partial = storage.get_execution_certificates_batch(&[*ec3.tick_id(), missing_tick_id]);
    assert_eq!(partial.len(), 1);
    assert_eq!(partial[0].tick_id(), ec3.tick_id());
}

/// Shared boundary retention test: pin one height past
/// [`BOUNDARY_RETAIN`] and check eviction stops serving only the
/// oldest pin.
///
/// `commit_one` performs one backend-native substate commit for the
/// given seed — backends differ in their raw commit entry points.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_boundary_retention_evicts_oldest<S: BoundaryStore>(
    storage: &S,
    commit_one: impl Fn(u8),
) {
    let last = u64::try_from(BOUNDARY_RETAIN).expect("small const") + 1;
    for height in 1..=last {
        commit_one(u8::try_from(height).expect("small loop bound"));
        storage.pin_boundary(BlockHeight::new(height)).unwrap();
    }
    assert!(storage.open_boundary(BlockHeight::new(1)).is_none());
    assert!(storage.open_boundary(BlockHeight::new(2)).is_some());
    assert!(storage.open_boundary(BlockHeight::new(last)).is_some());
}

/// Shared boundary gating test: a committed but never-pinned height is
/// not served.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_boundary_unpinned_height_not_served<S: BoundaryStore>(
    storage: &S,
    commit_one: impl Fn(u8),
) {
    commit_one(1);
    assert!(storage.open_boundary(BlockHeight::new(1)).is_none());
}

/// The entry pipeline both backends serve identically.
///
/// Two commits over one collection — create, overwrite, remove, add — with
/// range scans, the cap, the self-describing leaf, and the historical scan at
/// the first version asserted between them. Backend-specific tails (GC,
/// retention) stay with their backend.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_entries_commit_serve_and_history<S: VersionedStore>(
    storage: &S,
    commit: impl Fn(&SettledWrites),
) {
    commit(&make_settled_entries(
        7,
        &[
            (5, Some(vec![5])),
            (10, Some(vec![10])),
            (20, Some(vec![20])),
        ],
    ));
    let root_v1 = storage.state_root();
    assert_ne!(root_v1, StateRoot::ZERO, "entries move the root");

    let key = entry_key(7, 5);
    assert_eq!(
        storage.entries_in_range(key.owner, key.collection, 0, u128::MAX, 10),
        vec![(5, vec![5]), (10, vec![10]), (20, vec![20])],
    );
    // Bounds and the cap hold.
    assert_eq!(
        storage.entries_in_range(key.owner, key.collection, 6, 20, 1),
        vec![(10, vec![10])],
    );
    // The leaf form is readable by its derived key and self-describes.
    let leaf = storage
        .cell(entry_leaf_key(&ProtocolHasher, key))
        .expect("the entry commits a leaf");
    let decoded = from_slice::<EntryLeaf>(&leaf).unwrap();
    assert_eq!((decoded.order, decoded.value), (5, vec![5]));

    // Version 2: overwrite one, remove one, add one.
    commit(&make_settled_entries(
        7,
        &[(10, Some(vec![99])), (20, None), (30, Some(vec![30]))],
    ));
    assert_ne!(storage.state_root(), root_v1);
    assert_eq!(
        storage.entries_in_range(key.owner, key.collection, 0, u128::MAX, 10),
        vec![(5, vec![5]), (10, vec![99]), (30, vec![30])],
    );

    // The historical scan at version 1 answers the old interval.
    assert_eq!(
        storage.snapshot_at(BlockHeight::new(1)).entries_in_range(
            key.owner,
            key.collection,
            0,
            u128::MAX,
            10
        ),
        vec![(5, vec![5]), (10, vec![10]), (20, vec![20])],
    );
    // And another collection's interval stays empty.
    assert!(
        storage
            .entries_in_range(state_key(9, 0).owner, key.collection, 0, u128::MAX, 10)
            .is_empty()
    );
}

/// Shared sweep-index conformance: the index tracks what the leaves say,
/// and the walk that reads it visits cells in sweep order.
///
/// Runs against [`StubVmStatics`](hyperscale_types::test_utils::StubVmStatics)'s
/// sweepable family, so a backend's index is tested without a VM: what
/// the backend owes is that it indexes whatever the judgement calls
/// sweepable, which is a property of the backend and not of the family.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_sweep_index_tracks_the_leaves<S>(storage: &S, commit: impl Fn(&SettledWrites))
where
    S: SubstateStore + SweepIndex,
{
    install_stub_protocol_statics();
    let bucket_ms = SWEEP_BUCKET_MS;
    // Three cells across two buckets under two owners, so the walk has
    // to order by bucket before owner — the property a leaf-key walk
    // does not have.
    let cells: Vec<(SubstateKey, u64)> = [
        (0xF0u8, 3 * bucket_ms, 0x11u8),
        (0x10, 3 * bucket_ms + 5, 0x22),
        (0x10, 5 * bucket_ms, 0x33),
    ]
    .into_iter()
    .map(|(owner, expiry, body)| {
        let (local, value) = stub_sweepable_cell(expiry, body);
        let key = SubstateKey {
            owner: Address::new([owner; 31], AddressClass::Component),
            local,
        };
        commit(&SettledWrites::from_absolutes(BTreeMap::from([(
            key,
            Some(value),
        )])));
        (key, expiry)
    })
    .collect();
    // And one ordinary cell, which the index must not see.
    commit(&make_settled_writes(0x10, 7, vec![9, 9, 9]));

    // Sweep order is bucket, then owner, then local — so the late
    // bucket goes last even though its owner sorts first, which is the
    // property a walk over leaf keys would not have.
    let (low_owner, high_owner, late) = (cells[1], cells[0], cells[2]);
    let expected = vec![low_owner, high_owner, late];

    let all = SweepFrontier::start_of(SweepBucket(u32::MAX));
    assert_eq!(
        storage.sweep_candidates(SweepFrontier::ZERO, all, 10),
        expected,
        "bucket orders before owner, and the plain cell is not swept"
    );

    // The ceiling excludes the clock's own bucket and everything above.
    let clock = WeightedTimestamp::from_millis(4 * bucket_ms);
    assert_eq!(
        storage.sweep_candidates(SweepFrontier::ZERO, SweepFrontier::ceiling_at(clock), 10),
        vec![low_owner, high_owner],
    );

    // The cap stops the walk, and resuming from where it stopped picks
    // up exactly the rest — no gap, no repeat.
    let first = storage.sweep_candidates(SweepFrontier::ZERO, all, 2);
    assert_eq!(first, vec![low_owner, high_owner]);
    let resumed = SweepFrontier::of_leaf(first[1].0);
    assert_eq!(storage.sweep_candidates(resumed, all, 10), vec![late]);

    // Removing a cell takes it out of the index, and the row it emptied
    // with it — the walk then stops one short.
    commit(&SettledWrites::from_absolutes(BTreeMap::from([(
        late.0, None,
    )])));
    assert_eq!(
        storage.sweep_candidates(SweepFrontier::ZERO, all, 10),
        vec![low_owner, high_owner]
    );
}

/// Shared block-sweep conformance: where a block's frontier lands, and
/// that resuming from it loses nothing.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_sweep_stops_at_the_ceiling_or_the_cap<S>(storage: &S, commit: impl Fn(&SettledWrites))
where
    S: SubstateStore + SweepIndex,
{
    install_stub_protocol_statics();
    // Two cells a bucket apart, both long past a clock well above them.
    let cells: Vec<(SubstateKey, u64)> = [(3u64, 0x11u8), (5, 0x22)]
        .into_iter()
        .map(|(bucket, body)| {
            let expiry = bucket * SWEEP_BUCKET_MS;
            let (local, value) = stub_sweepable_cell(expiry, body);
            let key = SubstateKey {
                owner: Address::new([0x40; 31], AddressClass::Component),
                local,
            };
            commit(&SettledWrites::from_absolutes(BTreeMap::from([(
                key,
                Some(value),
            )])));
            (key, expiry)
        })
        .collect();
    let clock = WeightedTimestamp::from_millis(20 * SWEEP_BUCKET_MS);

    // Under the cap, the frontier takes the ceiling: nothing sweepable
    // is left below the clock's own bucket, so the next block starts
    // from there rather than from the last cell.
    let (removals, frontier) = sweep_for_block(storage, SweepFrontier::ZERO, clock);
    assert_eq!(removals, vec![cells[0].0, cells[1].0]);
    assert_eq!(frontier, SweepFrontier::ceiling_at(clock));

    // A block whose ceiling has not moved past its parent's frontier
    // removes nothing and repeats the frontier it inherited. That is
    // every block at sub-second times against a minute-wide bucket, so
    // the frontier's rule is monotone rather than strictly advancing.
    let (again, stood_still) = sweep_for_block(storage, frontier, clock);
    assert!(again.is_empty());
    assert_eq!(stood_still, frontier);

    // A clock inside the first cell's own bucket reaches neither, since
    // the ceiling excludes that bucket entirely.
    let early = WeightedTimestamp::from_millis(3 * SWEEP_BUCKET_MS + 1);
    let (none, early_frontier) = sweep_for_block(storage, SweepFrontier::ZERO, early);
    assert!(none.is_empty());
    assert_eq!(early_frontier, SweepFrontier::ceiling_at(early));

    // And resuming from that frontier still reaches both, so a block
    // that swept nothing has not skipped anything.
    let (resumed, _) = sweep_for_block(storage, early_frontier, clock);
    assert_eq!(resumed, vec![cells[0].0, cells[1].0]);
}

/// Shared serve → import round trip: leaves enumerated and resolved
/// from `serving`'s pinned boundary rebuild an identical store in
/// `fresh`, with the raw substates readable and a second import
/// rejected.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_boundary_import_roundtrip<S>(serving: &S, fresh: &S, commit: impl Fn(&SettledWrites))
where
    S: BoundaryStore + SubstateStore,
{
    for seed in 1..=5u8 {
        commit(&make_settled_writes(seed, seed, vec![seed, seed, seed]));
    }
    // The sixth commit is two ordered-collection entries, so the round
    // trip covers entry leaves and the index rebuild beside the cells.
    commit(&make_settled_entries(
        7,
        &[(5, Some(vec![5])), (10, Some(vec![10]))],
    ));
    let source_root = serving.state_root();
    serving.pin_boundary(BlockHeight::new(6)).unwrap();

    let boundary = serving.open_boundary(BlockHeight::new(6)).expect("pinned");
    let root_key = boundary.get_root_key(6).expect("root resolves");
    let chunk = Jmt::collect_range(
        &boundary,
        &root_key,
        &[0u8; KEY_BYTES],
        &[0xFF; KEY_BYTES],
        1_000,
    )
    .unwrap();
    let leaves: Vec<SubstateLeaf> = chunk
        .leaves
        .iter()
        .map(|(leaf_key, _)| {
            let value = boundary
                .cell(
                    SubstateKey::from_bytes(*leaf_key).expect("a stored leaf key names an address"),
                )
                .expect("resolves");
            SubstateLeaf {
                key: SubstateKey::from_bytes(*leaf_key)
                    .expect("a stored leaf key names an address"),
                value,
            }
        })
        .collect();
    assert_eq!(leaves.len(), 7);
    let probe = leaves
        .iter()
        .find(|l| l.value == [3, 3, 3])
        .map(|l| l.key)
        .expect("seed-3 leaf present");

    let imported_root =
        import_boundary_state(fresh, BlockHeight::new(6), &leaves, WitnessSeed::default()).unwrap();
    assert_eq!(imported_root, source_root);
    assert_eq!(fresh.state_root(), source_root);

    // Imported raw substates read back at the imported state.
    fresh.pin_boundary(BlockHeight::new(6)).unwrap();
    let fresh_boundary = fresh.open_boundary(BlockHeight::new(6)).expect("pinned");
    assert_eq!(fresh_boundary.cell(probe), Some(vec![3, 3, 3]),);

    // The ordered index re-derived from the imported leaves answers the
    // same interval the serving store does.
    let entry = entry_key(7, 5);
    assert_eq!(
        fresh.entries_in_range(entry.owner, entry.collection, 0, u128::MAX, 10),
        serving.entries_in_range(entry.owner, entry.collection, 0, u128::MAX, 10),
    );
    assert_eq!(
        fresh.entries_in_range(entry.owner, entry.collection, 0, u128::MAX, 10),
        vec![(5, vec![5]), (10, vec![10])],
    );

    // A second import is rejected — the store is no longer empty.
    assert!(
        import_boundary_state(fresh, BlockHeight::new(6), &[], WitnessSeed::default()).is_err()
    );
}

/// A complete EC over `tx_hashes`, every transaction succeeding — the
/// producing shard's copy, and the only one [`ExecutionCertificate::project_to`]
/// can narrow.
fn execution_certificate_over(
    block_height: BlockHeight,
    tx_hashes: &[TxHash],
) -> ExecutionCertificate {
    let outcomes: Vec<TxOutcome> = tx_hashes
        .iter()
        .enumerate()
        .map(|(position, tx_hash)| {
            let seed = u8::try_from(position).expect("small fixture") + 150;
            TxOutcome::new(
                *tx_hash,
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::from_raw(Hash::from_bytes(&[seed; 32])),
                },
            )
        })
        .collect();
    ExecutionCertificate::new(
        TickId::new(ShardId::ROOT, block_height),
        WeightedTimestamp::from_millis(block_height.inner() + 1),
        compute_global_receipt_root(&outcomes),
        outcomes,
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::new(4),
    )
}

/// Shared coverage test for the tick slot: a store keeps the widest copy
/// of a tick it has seen, and answers a by-transaction lookup only for
/// what that copy carries.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_widest_tick_copy_holds_the_slot(storage: &(impl ShardChainReader + ShardChainWriter)) {
    let txs: Vec<TxHash> = (1u8..=3)
        .map(|seed| TxHash::from(Hash::from_bytes(&[seed; 32])))
        .collect();
    let complete = execution_certificate_over(BlockHeight::new(1), &txs);
    let leg = |tx: TxHash| {
        complete
            .project_to(&HashSet::from([tx]))
            .expect("the copy carries it")
    };

    // One leg arrives, and answers for the transaction it carries.
    commit_empty_blocks_up_to(storage, BlockHeight::new(1));
    let first = make_test_block_with_ecs(BlockHeight::new(1), vec![Arc::new(leg(txs[0]))]);
    storage.commit_block(&make_test_certified(first), &[], &empty_witness());
    let served = storage.get_execution_certificates_for_txs(&[txs[0]]);
    assert_eq!(served.len(), 1, "the transaction its copy carries");
    assert!(served[0].covers(&txs[0]));

    // A disjoint leg does not take the slot from it — the transaction
    // only that leg covered is served from its own shard instead.
    let second = make_test_block_with_ecs(BlockHeight::new(2), vec![Arc::new(leg(txs[1]))]);
    storage.commit_block(&make_test_certified(second), &[], &empty_witness());
    assert!(
        storage.get_execution_certificates_for_txs(&[txs[0]])[0].covers(&txs[0]),
        "the copy already held keeps the slot",
    );
    assert!(
        storage
            .get_execution_certificates_for_txs(&[txs[1]])
            .is_empty(),
        "and nothing points at a copy that lost",
    );

    // The complete copy carries everything the slot held and more, so it
    // takes it, and the index reaches every transaction of the tick.
    let third = make_test_block_with_ecs(BlockHeight::new(3), vec![Arc::new(complete.clone())]);
    storage.commit_block(&make_test_certified(third), &[], &empty_witness());
    for tx in &txs {
        let served = storage.get_execution_certificates_for_txs(from_ref(tx));
        assert_eq!(served.len(), 1);
        assert!(served[0].covers(tx), "the complete copy answers for it");
    }
    assert_eq!(
        storage.get_execution_certificates_for_txs(&txs).len(),
        1,
        "transactions of one tick resolve to one certificate",
    );
}

/// Shared index test: the by-transaction certificate lookup answers with
/// this shard's own certificate.
///
/// A settled cross-shard transaction lands under both sides' certificates
/// in one finalization, and the lookup serves the question a
/// counterpart's fallback fetch asks this shard. The remote copy sorts
/// after the local one in the write walk, so an unfiltered single-slot
/// index would leave it the winner and serve the requester its own
/// certificate back.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_tx_index_answers_with_the_local_shards_certificate(
    storage: &(impl ShardChainReader + ShardChainWriter),
) {
    let tx = TxHash::from(Hash::from_bytes(&[7u8; 32]));
    let local = execution_certificate_over(BlockHeight::new(1), from_ref(&tx));
    let remote_outcomes = vec![TxOutcome::new(
        tx,
        ExecutionOutcome::Succeeded {
            receipt_hash: GlobalReceiptHash::from_raw(Hash::from_bytes(&[99u8; 32])),
        },
    )];
    let remote = ExecutionCertificate::new(
        TickId::new(ShardId::leaf(1, 1), BlockHeight::new(4)),
        WeightedTimestamp::from_millis(5),
        compute_global_receipt_root(&remote_outcomes),
        remote_outcomes,
        AggregateSignature::new([0u8; 96]),
        SignerBitfield::new(4),
    );

    commit_empty_blocks_up_to(storage, BlockHeight::new(1));
    let certificate = Finalization::new(
        *local.tick_id(),
        TickHalf::Legs,
        vec![Arc::new(local.clone()), Arc::new(remote)],
        vec![],
    );
    let block = push_certificate(
        make_test_block(BlockHeight::new(1)),
        Arc::new(certificate.into()),
    );
    storage.commit_block(&make_test_certified(block), &[], &empty_witness());

    let served = storage.get_execution_certificates_for_txs(from_ref(&tx));
    assert_eq!(served.len(), 1, "one certificate answers for the tx");
    assert_eq!(
        served[0].tick_id(),
        local.tick_id(),
        "and it is this shard's own, not the counterpart copy riding beside it",
    );
}

/// Attach a provisions bundle for `tx_hash` to a live block, preserving
/// everything else.
#[must_use]
pub fn with_provisions(block: Block, source: ShardId, tx_hash: TxHash) -> Block {
    let bundle = Provisions::new(
        source,
        ShardId::ROOT,
        BlockHeight::new(1),
        WeightedTimestamp::ZERO,
        MerkleInclusionProof::dummy(),
        vec![ProvisionEntry::new(tx_hash, vec![])],
    );
    match block {
        Block::Live {
            header,
            transactions,
            certificates,
            abandonment_records,
            state_proofs,
            witness_sources,
            ..
        } => Block::Live {
            header,
            transactions,
            certificates,
            provisions: Arc::new(vec![Arc::new(Verifiable::from(bundle))]),
            abandonment_records,
            state_proofs,
            witness_sources,
        },
        sealed @ Block::Sealed { .. } => sealed,
    }
}

/// Attach `txs` to a live block, preserving everything else.
fn with_transactions(block: Block, txs: Vec<Arc<Verifiable<Transaction>>>) -> Block {
    match block {
        Block::Live {
            header,
            certificates,
            provisions,
            abandonment_records,
            state_proofs,
            witness_sources,
            ..
        } => Block::Live {
            header,
            transactions: Arc::new(txs),
            certificates,
            provisions,
            abandonment_records,
            state_proofs,
            witness_sources,
        },
        sealed @ Block::Sealed { .. } => sealed,
    }
}

/// Shared prepare-path test: a block that carries a transaction and no
/// certificates still writes its committed cell.
///
/// The prepare path treats a block with nothing to write as a no-op
/// whose root is its parent's, and a block carrying only transactions
/// has no receipts. Its committed cells are writes all the same — the
/// prober asking whether this shard committed the transaction reads
/// them off the root — so the root moves, the cell is served under it,
/// and the direct path reaches the same root.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_prepared_commit_writes_committed_cells<S>(prepared: &Arc<S>, direct: &S)
where
    S: ShardChainReader + ShardChainWriter + SubstateStore,
{
    let tx = test_transaction(1);
    let cell = committed_tx_cell_key(
        ShardId::ROOT,
        tx.hash(),
        tx.validity_range().end_timestamp_exclusive,
    );
    let block = with_transactions(
        make_test_block(BlockHeight::new(1)),
        vec![Arc::new(Verifiable::from(tx))],
    );
    let creations = committed_tx_cells(
        block.header().shard_id(),
        block.transactions().iter().map(|tx| tx.as_unverified()),
    );

    let parent_root = prepared.state_root();
    let (spec_root, _snapshot, commit) = prepared.prepare_block_commit(
        ParentAnchor {
            state_root: parent_root,
            height: BlockHeight::GENESIS,
            state: &prepared.snapshot(),
            pending: &[],
            base_reads: None,
        },
        &[],
        &creations,
        &[],
        BlockHeight::new(1),
    );
    let certified = make_test_certified(block);
    let committed_root = commit(SyncHint::FlushNow, &certified, &empty_witness());
    assert_ne!(spec_root, parent_root, "a committed cell moves the root");
    assert_eq!(committed_root, spec_root);
    assert!(
        matches!(
            prepared.get_substate_at_height(cell, BlockHeight::new(1)),
            Some(Some(_))
        ),
        "the prepared commit serves the committed cell under its root",
    );

    let direct_root = direct.commit_block(&certified, &[], &empty_witness());
    assert_eq!(direct_root, spec_root, "both paths reach one root");
}

/// Commit a block at `height` carrying one provision bundle, and return
/// the bundle's hash. The bundle's transaction varies with the height, so
/// each block's bundle has its own identity.
fn commit_block_with_provisions(storage: &impl ShardChainWriter, height: u64) -> ProvisionHash {
    let seed = u8::try_from(height).expect("small fixture");
    let block = with_provisions(
        make_test_block(BlockHeight::new(height)),
        ShardId::leaf(1, 1),
        TxHash::from(Hash::from_bytes(&[seed; 32])),
    );
    let hash = block
        .provisions()
        .first()
        .expect("the block carries one")
        .hash();
    storage.commit_block(&make_test_certified(block), &[], &empty_witness());
    hash
}

/// Shared recovery test: a durable lock comes back with the certificate
/// that justifies it.
///
/// A validator refuses to vote for a block whose `parent_qc` sits below
/// `locked_round`. A record that keeps the round and drops the
/// certificate therefore describes a position nothing can satisfy: every
/// proposal extends a lower QC, and the QC that would raise the lock can
/// only form out of the votes being refused.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_registers_recover_their_justification(
    storage: &impl SafeVoteRegisterStore,
    recovered: impl Fn() -> RecoveredState,
) {
    let validator = ValidatorId::new(1);
    let justification = make_test_qc(&make_test_block(BlockHeight::new(4)));
    let locked = SafeVoteRegisters {
        locked_round: Round::new(6),
        last_voted_round: Round::new(7),
        high_qc: Some((*justification).clone()),
    };
    storage.persist_safe_vote_registers(validator, locked);

    assert_eq!(
        storage
            .safe_vote_registers(validator)
            .and_then(|r| r.high_qc),
        Some((*justification).clone()),
        "the lock's justification is part of the record",
    );

    // A later write that raises nothing keeps the justification: the
    // certificate travels with the higher lock, not with the newer write.
    storage.persist_safe_vote_registers(
        validator,
        SafeVoteRegisters {
            locked_round: Round::new(2),
            last_voted_round: Round::new(9),
            high_qc: None,
        },
    );
    let merged = recovered()
        .safe_vote_registers
        .get(&validator)
        .cloned()
        .expect("the record survives into recovery");
    assert_eq!(merged.locked_round, Round::new(6));
    assert_eq!(merged.last_voted_round, Round::new(9));
    assert_eq!(
        merged.high_qc,
        Some((*justification).clone()),
        "a write that lowers the lock cannot strip the justification off it",
    );
}

/// Shared recovery test: the committed tip's drain total comes back off
/// its stored header.
///
/// A block extending the tip claims a total the vote path checks against
/// the parent's, and skips the vote when it cannot resolve one. A replica
/// that recovers no total therefore cannot vote until a commit reseats it
/// — and a commit needs a quorum it is part of, so a shard where enough
/// replicas restart together never forms one.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_recovery_carries_the_tip_drain_total(
    storage: &impl ShardChainWriter,
    recovered: impl Fn() -> RecoveredState,
) {
    let in_flight = WorkInFlight::new(7);
    let mut block = make_test_block(BlockHeight::new(1));
    let Block::Live { header, .. } = &mut block else {
        panic!("the fixture builds a live block");
    };
    *header = BlockHeader::new(BlockHeaderParts {
        height: header.height(),
        parent_block_hash: header.parent_block_hash(),
        parent_qc: header.parent_qc().clone().into(),
        timestamp: header.timestamp(),
        work_in_flight: in_flight,
        ..Default::default()
    });
    storage.commit_block(&make_test_certified(block), &[], &empty_witness());

    assert_eq!(
        recovered().committed_tip.map(|tip| tip.work_in_flight),
        Some(in_flight),
        "the tip's drain total is on its stored header, so recovery reads it",
    );
}

/// Shared retention test: sealing a block keeps its bundles' hashes and
/// drops their bodies, so the bodies live beside it and a restart reads
/// them back.
///
/// `recovered` loads recovered state the way the backend does — the
/// `RocksDB` caller reopens the store first, which is the crossing the
/// bodies have to survive.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_committed_bundle_outlives_sealing(
    storage: &(impl ShardChainReader + ShardChainWriter),
    recovered: impl Fn() -> RecoveredState,
) {
    let hash = commit_block_with_provisions(storage, 1);

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
        recovered()
            .retained_provisions
            .iter()
            .map(|bundle| bundle.hash())
            .collect::<Vec<_>>(),
        vec![hash],
        "and the body it dropped is recovered from storage",
    );
}

/// Shared sweep test: a body outlives the depth a replay could start
/// from, and no longer.
///
/// The floor is the history retention floor — below it `snapshot_at`
/// cannot serve the baseline a replayed tick reads, so a body kept there
/// could never be replayed against. `storage` must be configured with
/// `history_length` as its JMT history length; the walk is derived from
/// it.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_retained_bundle_drops_below_the_history_floor(
    storage: &(impl ShardChainReader + ShardChainWriter),
    history_length: u64,
    recovered: impl Fn() -> RecoveredState,
) {
    let first = commit_block_with_provisions(storage, 1);
    for height in 2..=history_length + 1 {
        commit_block_with_provisions(storage, height);
    }
    assert!(
        recovered()
            .retained_provisions
            .iter()
            .any(|bundle| bundle.hash() == first),
        "at the floor the body is still readable",
    );

    commit_block_with_provisions(storage, history_length + 2);
    let retained = recovered().retained_provisions;
    assert!(
        !retained.iter().any(|bundle| bundle.hash() == first),
        "past it the sweep drops it",
    );
    assert_eq!(
        retained.len() as u64,
        history_length + 1,
        "and keeps the readable window, plus the block of slack at its floor",
    );
}

/// Shared rebuild test: a replica that lost its execution state replays
/// the chain and names exactly the transactions it committed and never
/// resolved.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_unresolved_fold(storage: &(impl ShardChainReader + ShardChainWriter)) {
    let resolved = test_transaction(1);
    let open = test_transaction(2);
    let source = ShardId::leaf(1, 1);

    commit_empty_blocks_up_to(storage, BlockHeight::new(1));
    let committing = with_transactions(
        make_test_block(BlockHeight::new(1)),
        vec![
            Arc::new(Verifiable::from(resolved.clone())),
            Arc::new(Verifiable::from(open.clone())),
        ],
    );
    storage.commit_block(&make_test_certified(committing), &[], &empty_witness());

    // A counterpart's bundle for the one that stays open, so the sealed
    // block below is read against a height that actually carried one.
    let provisioning = with_provisions(make_test_block(BlockHeight::new(2)), source, open.hash());
    storage.commit_block(&make_test_certified(provisioning), &[], &empty_witness());

    // Only one of them gets an outcome. An abort resolves a transaction
    // exactly as a settlement does, and owes no receipt — which is what
    // lets the rebuilt block carry it on every backend.
    let resolving = push_certificate(
        make_test_block(BlockHeight::new(3)),
        Arc::new(Verifiable::from(make_finalization(
            BlockHeight::new(3),
            resolved.hash(),
            TransactionDecision::Aborted,
        ))),
    );
    storage.commit_block(&make_test_certified(resolving), &[], &empty_witness());

    // The floor is the height that committed the one still open. The
    // one resolved at height 3 committed there too, and does not hold
    // the floor down: an outcome is what releases it.
    assert_eq!(
        unresolved_replay_floor(storage, BlockHeight::new(3), WeightedTimestamp::ZERO),
        Some(BlockHeight::new(1)),
        "the replay starts at the block committing what is still owed",
    );
    assert!(
        storage
            .get_block(BlockHeight::new(2))
            .expect("committed")
            .block()
            .provisions()
            .is_empty(),
        "a stored block keeps its bundles' hashes, not their contents",
    );

    // And the window puts them back, so replaying it composes the leg
    // rather than waiting on evidence nothing will send again.
    let window = replay_window(storage, BlockHeight::new(3), WeightedTimestamp::ZERO);
    let heights: Vec<BlockHeight> = window
        .blocks
        .iter()
        .map(|certified| certified.block().height())
        .collect();
    assert_eq!(
        heights,
        vec![
            BlockHeight::new(1),
            BlockHeight::new(2),
            BlockHeight::new(3)
        ],
        "opening at the floor and running to the tip",
    );
    assert_eq!(
        window.anchor_wt,
        Some(WeightedTimestamp::ZERO),
        "carrying the clock of the block below it, so the first block replayed keeps the carry",
    );
    assert_eq!(
        window.blocks[1]
            .block()
            .provisions()
            .iter()
            .map(|p| p.hash())
            .collect::<Vec<_>>(),
        storage
            .provisions_at(BlockHeight::new(2))
            .iter()
            .map(|p| p.hash())
            .collect::<Vec<_>>(),
        "with the bodies sealing dropped reattached",
    );

    // Every one of them live, including the floor — which carried no
    // bundles at all. A sealed block owes nothing and composes nothing,
    // so a window handing one back would skip the very transaction that
    // put the block in the window.
    assert!(
        window
            .blocks
            .iter()
            .all(|certified| certified.block().is_live()),
        "a replayed block arrives in the shape a commit runs on, bundles or not",
    );
}

/// Shared rebuild test: an undischarged record holds the replay floor.
///
/// It does so from a window of its own — the transaction it names
/// committed arbitrarily far below, so the record is what a rebuild has to
/// reach.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_undischarged_record_holds_the_floor(
    storage: &(impl ShardChainReader + ShardChainWriter),
) {
    let stranded = test_transaction(3);
    let record = AbandonmentRecord::departed(
        ShardId::leaf(1, 1),
        WeightedTimestamp::from_millis(1_000),
        [UnsettledTx {
            tx_hash: stranded.hash(),
            deadline: WeightedTimestamp::from_millis(500),
            declared_work: stranded.work(),
            charge: AbortCharge {
                vault: stranded.fee_vault(),
                amount: stranded.price(),
            },
        }],
    );

    // The record commits without its transaction ever appearing: the block
    // that carried it is below anything this chain holds, which is the
    // rebuild the record exists to repair.
    commit_empty_blocks_up_to(storage, BlockHeight::new(2));
    let naming = with_abandonment(make_test_block(BlockHeight::new(2)), record);
    storage.commit_block(&make_test_certified(naming), &[], &empty_witness());
    storage.commit_block(
        &make_test_certified(make_test_block(BlockHeight::new(3))),
        &[],
        &empty_witness(),
    );

    assert_eq!(
        unresolved_replay_floor(storage, BlockHeight::new(3), WeightedTimestamp::ZERO),
        Some(BlockHeight::new(2)),
        "the replay opens at the record, which is where the entry comes from",
    );

    // The abort discharges it, and the floor lifts with it.
    let aborting = push_certificate(
        make_test_block(BlockHeight::new(4)),
        Arc::new(Verifiable::from(make_finalization(
            BlockHeight::new(4),
            stranded.hash(),
            TransactionDecision::Aborted,
        ))),
    );
    storage.commit_block(&make_test_certified(aborting), &[], &empty_witness());

    assert_eq!(
        unresolved_replay_floor(storage, BlockHeight::new(4), WeightedTimestamp::ZERO),
        None,
        "a discharged record holds nothing down",
    );
}

/// Shared rebuild test: a leg's own finalization does not retire it from
/// the replay floor.
///
/// Its entry lives on for the reclaim its core's refusal or its
/// deliveries' lapse may license; the reclaim's deciding finalization is
/// what retires it.
///
/// # Panics
///
/// Panics if any assertion fails (this is a test helper).
pub fn test_a_legs_own_finalization_keeps_the_floor(
    storage: &(impl ShardChainReader + ShardChainWriter),
) {
    let leg = test_transaction(4);

    commit_empty_blocks_up_to(storage, BlockHeight::new(1));
    let committing = with_transactions(
        make_test_block(BlockHeight::new(1)),
        vec![Arc::new(Verifiable::from(leg.clone()))],
    );
    storage.commit_block(&make_test_certified(committing), &[], &empty_witness());

    let settling = push_certificate(
        make_test_block(BlockHeight::new(2)),
        Arc::new(Verifiable::from(make_leg_finalization(
            BlockHeight::new(2),
            leg.hash(),
        ))),
    );
    storage.commit_block(&make_test_certified(settling), &[], &empty_witness());
    assert_eq!(
        unresolved_replay_floor(storage, BlockHeight::new(2), WeightedTimestamp::ZERO),
        Some(BlockHeight::new(1)),
        "a leg that succeeded is still owed a reclaim, so its commit holds the floor",
    );

    let reclaiming = push_certificate(
        make_test_block(BlockHeight::new(3)),
        Arc::new(Verifiable::from(make_finalization(
            BlockHeight::new(3),
            leg.hash(),
            TransactionDecision::Aborted,
        ))),
    );
    storage.commit_block(&make_test_certified(reclaiming), &[], &empty_witness());
    assert_eq!(
        unresolved_replay_floor(storage, BlockHeight::new(3), WeightedTimestamp::ZERO),
        None,
        "the reclaim's finalization decides it and releases the floor",
    );
}
