//! `SharedStorage` newtype — Arc-wrapped `RocksDbShardStorage` with full trait impls.
//!
//! Production wraps `Arc<RocksDbShardStorage>` in this newtype so that the pinned
//! `IoLoop` thread and async tasks (e.g. `InboundRouter`) can both hold the
//! same underlying database via cheap Arc clones, each going through the same
//! storage-trait implementations.
//!
//! The orphan rule prevents implementing foreign traits for
//! `Arc<RocksDbShardStorage>` directly. This newtype sidesteps that
//! while providing zero-cost delegation.

use std::sync::Arc;

use hyperscale_jmt::{NibblePath, Node as JmtNode, NodeKey as JmtNodeKey, TreeReader};
use hyperscale_storage::{
    AdoptSource, BlockForSync, BoundaryStore, GenesisCommit, ImportProgress, JmtSnapshot,
    LegEntryStore, PackageArtifactStore, ParentAnchor, SafeVoteRegisterStore, ShardChainReader,
    ShardChainWriter, SubstateStore, Substates, SweepIndex, VersionedStore, WitnessSeed,
};
use hyperscale_types::{
    BeaconWitnessCommit, BeaconWitnessLeafCount, Block, BlockHash, BlockHeight, CertifiedBlock,
    CertifiedBlockHeader, ChainOrigin, ConsensusReceipt, DeclaredRange, ExecutionCertificate,
    Finalization, FinalizationHash, Hash, LegEntry, PreparedCommit, Provisions, QuorumCertificate,
    SafeVoteRegisters, SettledWrites, ShardWitnessPayload, StateRoot, SubstateKey, SubstateLeaf,
    SweepFrontier, TickId, Transaction, TxHash, ValidatorId, Verifiable, Verified, VotePosition,
};
use hyperscale_vm_types::{Address, CollectionId};

use super::core::RocksDbShardStorage;
use super::snapshot::RocksDbSnapshot;

/// Shared `RocksDB` storage handle with full storage trait implementations.
///
/// A cheap-to-clone wrapper around `Arc<RocksDbShardStorage>` that implements all
/// storage traits needed by `IoLoop`. The pinned thread and async tasks
/// share the same underlying database via Arc clones of this handle.
///
/// # Why a newtype?
///
/// Rust's orphan rule prevents implementing foreign traits for
/// `Arc<RocksDbShardStorage>`. This local newtype can implement all
/// traits while `Arc::clone` keeps sharing cheap.
#[derive(Clone)]
pub struct SharedStorage(pub Arc<RocksDbShardStorage>);

impl SharedStorage {
    /// Create a new shared storage handle.
    pub const fn new(storage: Arc<RocksDbShardStorage>) -> Self {
        Self(storage)
    }

    /// Get a reference to the underlying `Arc<RocksDbShardStorage>`.
    #[must_use]
    pub const fn arc(&self) -> &Arc<RocksDbShardStorage> {
        &self.0
    }
}

impl std::ops::Deref for SharedStorage {
    type Target = RocksDbShardStorage;
    fn deref(&self) -> &RocksDbShardStorage {
        &self.0
    }
}

impl SweepIndex for SharedStorage {
    fn sweep_candidates(
        &self,
        frontier: SweepFrontier,
        ceiling: SweepFrontier,
        limit: usize,
    ) -> Vec<(SubstateKey, u64)> {
        self.0.sweep_candidates(frontier, ceiling, limit)
    }
}

impl PackageArtifactStore for SharedStorage {
    fn package_artifacts(&self) -> Vec<Vec<u8>> {
        self.0.package_artifacts()
    }

    fn package_artifact(&self, package: Hash) -> Option<Vec<u8>> {
        self.0.package_artifact(package)
    }
}

impl Substates for SharedStorage {
    fn cell(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.0.cell(key)
    }

    fn entries_in_range(
        &self,
        owner: Address,
        collection: CollectionId,
        lo: u128,
        hi: u128,
        limit: usize,
    ) -> Vec<(u128, Vec<u8>)> {
        self.0.entries_in_range(owner, collection, lo, hi, limit)
    }
}

impl GenesisCommit for SharedStorage {
    fn install_genesis(&self, substates: &SettledWrites, jmt_writes: &SettledWrites) -> StateRoot {
        self.0.commit_substates_only(substates);
        self.0.finalize_genesis_jmt(jmt_writes)
    }

    fn replicate_genesis_substates(&self, substates: &SettledWrites) {
        self.0.commit_substates_only(substates);
    }
}

impl SubstateStore for SharedStorage {
    type Snapshot<'a>
        = RocksDbSnapshot<'a>
    where
        Self: 'a;

    fn snapshot(&self) -> Self::Snapshot<'_> {
        self.0.snapshot()
    }

    fn jmt_height(&self) -> BlockHeight {
        self.0.jmt_height()
    }

    fn state_root(&self) -> StateRoot {
        self.0.state_root()
    }

    fn get_substate_at_height(
        &self,
        key: SubstateKey,
        block_height: BlockHeight,
    ) -> Option<Option<Vec<u8>>> {
        self.0.get_substate_at_height(key, block_height)
    }

    fn get_entries_at_height(
        &self,
        range: DeclaredRange,
        block_height: BlockHeight,
    ) -> Option<Vec<(u128, Vec<u8>)>> {
        self.0.get_entries_at_height(range, block_height)
    }
}

impl VersionedStore for SharedStorage {
    fn snapshot_at(&self, height: BlockHeight) -> Self::Snapshot<'_> {
        self.0.snapshot_at(height)
    }

    fn substate_bytes_at(&self, height: BlockHeight) -> Option<u64> {
        self.0.substate_bytes_at(height)
    }

    fn retention_floor(&self) -> u64 {
        VersionedStore::retention_floor(&*self.0)
    }
}

impl TreeReader for SharedStorage {
    fn get_node(&self, key: &JmtNodeKey) -> Option<Arc<JmtNode>> {
        self.0.get_node(key)
    }

    fn get_root_key(&self, version: u64) -> Option<JmtNodeKey> {
        self.0.get_root_key(version)
    }

    fn root_path(&self) -> NibblePath {
        self.0.root_path()
    }
}

impl BoundaryStore for SharedStorage {
    type Boundary = super::checkpoints::CheckpointStore;

    fn pin_boundary(&self, height: BlockHeight) -> Result<(), String> {
        self.0.pin_boundary(height)
    }

    fn open_boundary(&self, height: BlockHeight) -> Option<Self::Boundary> {
        self.0.open_boundary(height)
    }

    fn stage_import_chunk(
        &self,
        progress: &ImportProgress,
        leaves: &[SubstateLeaf],
    ) -> Result<(), String> {
        self.0.stage_import_chunk(progress, leaves)
    }

    fn read_import_progress(&self) -> Option<ImportProgress> {
        self.0.read_import_progress()
    }

    fn wipe_import_staging(&self) -> Result<(), String> {
        self.0.wipe_import_staging()
    }

    fn finalize_boundary_import(
        &self,
        height: BlockHeight,
        witnesses: WitnessSeed,
    ) -> Result<StateRoot, String> {
        self.0.finalize_boundary_import(height, witnesses)
    }

    fn follow_block_writes(&self, block: &Block) -> Result<StateRoot, String> {
        self.0.follow_block_writes(block)
    }

    fn adopt_genesis(
        &self,
        origin: ChainOrigin,
        genesis: &Block,
        source: AdoptSource,
    ) -> Result<StateRoot, String> {
        BoundaryStore::adopt_genesis(&*self.0, origin, genesis, source)
    }

    fn substate_bytes_at_version(&self, version: u64) -> Option<u64> {
        self.0.substate_bytes_at_version(version)
    }
}

impl ShardChainWriter for SharedStorage {
    fn prepare_block_commit(
        self: &Arc<Self>,
        parent: ParentAnchor<'_>,
        finalizations: &[Arc<Verifiable<Finalization>>],
        creations: &[(SubstateKey, Vec<u8>)],
        removals: &[SubstateKey],
        block_height: BlockHeight,
    ) -> (StateRoot, Arc<JmtSnapshot>, PreparedCommit) {
        self.0
            .prepare_block_commit(parent, finalizations, creations, removals, block_height)
    }

    fn commit_block(
        &self,
        certified: &Arc<Verified<CertifiedBlock>>,
        removals: &[SubstateKey],
        witness: &BeaconWitnessCommit,
    ) -> StateRoot {
        self.0.commit_block(certified, removals, witness)
    }
}

impl LegEntryStore for SharedStorage {
    fn persist_leg_entries(&self, entries: &[LegEntry], released: &[TxHash]) {
        self.0.persist_leg_entries(entries, released);
    }

    fn leg_entries(&self) -> Vec<LegEntry> {
        self.0.leg_entries()
    }
}

impl SafeVoteRegisterStore for SharedStorage {
    fn persist_vote_position(&self, validator: ValidatorId, position: &VotePosition) {
        self.0.persist_vote_position(validator, position);
    }

    fn voted_blocks_above(&self, committed_height: BlockHeight) -> Vec<Arc<Block>> {
        self.0.voted_blocks_above(committed_height)
    }

    fn safe_vote_registers(&self, validator: ValidatorId) -> Option<SafeVoteRegisters> {
        self.0.safe_vote_registers(validator)
    }
}

impl ShardChainReader for SharedStorage {
    fn get_block(&self, height: BlockHeight) -> Option<Verified<CertifiedBlock>> {
        self.0.get_block(height)
    }

    fn provisions_at(&self, height: BlockHeight) -> Vec<Arc<Verifiable<Provisions>>> {
        ShardChainReader::provisions_at(&*self.0, height)
    }

    fn get_certified_header(&self, height: BlockHeight) -> Option<Verified<CertifiedBlockHeader>> {
        ShardChainReader::get_certified_header(&*self.0, height)
    }

    fn committed_height(&self) -> BlockHeight {
        self.0.committed_height()
    }

    fn committed_hash(&self) -> Option<BlockHash> {
        self.0.committed_hash()
    }

    fn latest_qc(&self) -> Option<Verified<QuorumCertificate>> {
        self.0.latest_qc()
    }

    fn get_block_for_sync(&self, height: BlockHeight) -> Option<BlockForSync> {
        ShardChainReader::get_block_for_sync(&*self.0, height)
    }

    fn get_transactions_batch(&self, hashes: &[TxHash]) -> Vec<Verified<Transaction>> {
        ShardChainReader::get_transactions_batch(&*self.0, hashes)
    }

    fn get_certificates_batch(&self, ids: &[FinalizationHash]) -> Vec<Finalization> {
        self.0.get_certificates_batch(ids)
    }

    fn get_consensus_receipt(&self, tx_hash: &TxHash) -> Option<Arc<ConsensusReceipt>> {
        self.0.get_consensus_receipt(tx_hash)
    }

    fn get_execution_certificate(
        &self,
        tick_id: &TickId,
    ) -> Option<Verified<ExecutionCertificate>> {
        self.0.get_execution_certificate(tick_id)
    }

    fn get_execution_certificates_batch(
        &self,
        tick_ids: &[TickId],
    ) -> Vec<Verified<ExecutionCertificate>> {
        self.0.get_execution_certificates_batch(tick_ids)
    }

    fn get_execution_certificates_for_txs(
        &self,
        tx_hashes: &[TxHash],
    ) -> Vec<Verified<ExecutionCertificate>> {
        self.0.get_execution_certificates_for_txs(tx_hashes)
    }

    fn get_beacon_witness_payloads(&self, end: BeaconWitnessLeafCount) -> Vec<ShardWitnessPayload> {
        self.0.get_beacon_witness_payloads(end)
    }

    fn get_beacon_witness_payload_range(&self, start: u64, end: u64) -> Vec<ShardWitnessPayload> {
        self.0.get_beacon_witness_payload_range(start, end)
    }
}
