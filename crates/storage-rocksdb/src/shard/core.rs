//! # `RocksDB` Storage
//!
//! Production storage implementation using `RocksDB`.
//!
//! All operations are synchronous blocking I/O. Callers in async contexts
//! should use `spawn_blocking` if needed to avoid blocking the runtime.
//!
//! # JMT Integration
//!
//! Uses a binary Jellyfish Merkle Tree (Blake3) for cryptographic state
//! commitment. JMT data is stored in dedicated column families
//! (`jmt_nodes`, `stale_jmt_nodes`) plus metadata under `jmt:metadata`.
//! On each commit, the JMT is updated and a new state root hash is
//! computed.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hyperscale_hbor::from_slice;
use hyperscale_jmt::{NibblePath, Node as JmtNode, NodeKey as JmtNodeKey, TreeReader};
use hyperscale_metrics::record_storage_read;
use hyperscale_storage::{
    BaseReadCache, GenesisCommit, JmtSnapshot, SubstateStore, Substates, entry_leaf_value,
    package_of_cell, pending_write, sweepable_expiry, tree,
};
use hyperscale_types::{
    Block, BlockHeight, ChainOrigin, EntryLeaf, ProtocolHasher, QuorumCertificate,
    SafeVoteRegisters, SettledWrites, StateRoot, SubstateKey, ValidatorId, Verified,
    entry_leaf_key,
};
use hyperscale_vm_types::{Address, CollectionId, SweepBucket};
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamilyDescriptor, DB, DBCompressionType, Options,
    SliceTransform, WriteBatch,
};
use tracing::field::Empty;
use tracing::{Level, Span, instrument};

use super::column_families::{
    ALL_COLUMN_FAMILIES, CfHandles, EntriesCf, EntriesHistoryCf, HOT_WRITE_COLUMN_FAMILIES,
    JmtNodesCf, PackageArtifactsCf, STATE_HISTORY_CF, StaleEntriesHistoryCf, StaleJmtNodesCf,
    StaleStateHistoryCf, StateCf, StateHistoryCf, SubstateBytesCf, SweepIndexCf,
};
use super::entry_key::VersionedEntryKeyCodec;
use super::jmt_snapshot_store::SnapshotTreeStore;
use super::jmt_stored::{StaleTreePart, StoredNode, StoredNodeKey, VersionedStoredNode};
use super::metadata::{
    read_jmt_metadata, write_committed_hash, write_committed_height, write_committed_qc,
    write_jmt_metadata,
};
use super::versioned_key::VersionedSubstateKeyCodec;
use crate::StorageError;
use crate::config::RocksDbConfig;
use crate::typed_cf::{DbEncode, TypedCf, batch_delete, batch_put, get, multi_get};

/// RocksDB-based storage for production use.
///
/// Features:
/// - Column families for logical separation
/// - LZ4 compression for disk efficiency
/// - Block cache for read performance
/// - Bloom filters for key existence checks
/// - Binary Blake3 JMT for cryptographic state commitment
///
/// Implements `Substates` directly, plus the `SubstateStore` extension
/// for snapshots, node listing, and JMT state roots.
///
/// JMT tree nodes are persisted in the `jmt_nodes` column family. JMT metadata
/// (version and root hash) is in the default CF under `jmt:metadata` and read
/// directly from `RocksDB` on demand — always hot in the memtable since they're
/// written on every commit.
pub struct RocksDbShardStorage {
    pub(crate) db: Arc<DB>,

    /// Serializes JMT-mutating commits to prevent interleaved read-modify-write
    /// sequences (e.g., `read_jmt_metadata` + `WriteBatch` write).
    pub(crate) commit_lock: Mutex<()>,

    /// Number of block heights of JMT history to retain before garbage collection.

    /// Path this store's JMT is rooted at — its shard's prefix, so the root is
    /// the global tree's subtree at that prefix. Empty for a single-shard /
    /// whole-keyspace store. Set once at open from the shard's `ShardId`.
    pub(crate) root_path: NibblePath,

    /// Checkpoint ring for snap-sync boundary pins, rooted at the
    /// `checkpoints` directory beside the database.
    pub(crate) checkpoints: super::checkpoints::CheckpointRing,

    /// Write-path cache of the last-persisted safe-vote register record
    /// per validator. One guard spans the read-merge-write in
    /// `persist_safe_vote_registers`, keeping concurrent writes monotone
    /// and letting a write that raises nothing (e.g. a timeout
    /// retransmit) skip the fsync entirely.
    pub(crate) vote_registers: Mutex<HashMap<ValidatorId, (ChainOrigin, SafeVoteRegisters)>>,
}

/// Move one cell's write across the sweep index's rows.
///
/// The index counts live sweepable cells per owner and bucket, so what
/// matters is whether the cell was sweepable before and whether it is
/// after — both judged from bytes alone. A value equal to its prior is
/// sweepable in exactly the same row, which is why the no-op writes the
/// caller skips need no entry here either.
fn record_sweep_delta(
    deltas: &mut BTreeMap<(SweepBucket, Address), i64>,
    key: SubstateKey,
    prior: Option<&[u8]>,
    change: Option<&[u8]>,
) {
    let was = prior.and_then(|bytes| sweepable_expiry(key, bytes));
    let now = change.and_then(|bytes| sweepable_expiry(key, bytes));
    if was == now {
        return;
    }
    if let Some(expiry) = was {
        *deltas
            .entry((SweepBucket::of(expiry), key.owner))
            .or_default() -= 1;
    }
    if let Some(expiry) = now {
        *deltas
            .entry((SweepBucket::of(expiry), key.owner))
            .or_default() += 1;
    }
}

impl RocksDbShardStorage {
    /// Open or create a shard store rooted at the given directory.
    ///
    /// See [`Self::open_with_config`] for the directory layout.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if `RocksDB` fails to open the database.
    pub fn open<P: AsRef<Path>>(path: P, root_path: NibblePath) -> Result<Self, StorageError> {
        Self::open_with_config(path, &RocksDbConfig::default(), root_path)
    }

    /// Open with custom configuration.
    ///
    /// `path` is the shard's storage directory: the database lives at
    /// `path/db`, and the snap-sync checkpoint ring at `path/checkpoints`
    /// (`RocksDB` checkpoints hard-link the database's SSTs, so the ring
    /// sits beside — never inside — the `RocksDB`-owned directory).
    ///
    /// `root_path` is the prefix of the shard this store serves (via
    /// [`hyperscale_types::shard_prefix_path`]), so its JMT roots there and its
    /// `state_root` is the global tree's subtree at that prefix. Pass
    /// [`NibblePath::empty`] for a single-shard / whole-keyspace store.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if `RocksDB` fails to open the database.
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: &RocksDbConfig,
        root_path: NibblePath,
    ) -> Result<Self, StorageError> {
        let dir = path.as_ref();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Performance tuning
        opts.set_max_background_jobs(config.max_background_jobs);
        if config.bytes_per_sync > 0 {
            opts.set_bytes_per_sync(config.bytes_per_sync as u64);
        }
        opts.set_keep_log_file_num(config.keep_log_file_num);
        opts.set_max_write_buffer_number(config.max_write_buffer_number);
        opts.set_write_buffer_size(config.write_buffer_size);

        // Allow WAL write and memtable insertion to overlap. Safe because
        // all block commits use a single WriteBatch (already atomic).
        opts.set_enable_pipelined_write(true);

        // Compression
        opts.set_compression_type(config.compression.to_rocksdb());

        // Block cache and bloom filter — shared across ALL column families.
        // SST index/filter blocks are pinned inside this cache to prevent
        // unbounded heap growth as the database accumulates SST files.
        let mut block_opts = BlockBasedOptions::default();
        if let Some(cache_size) = config.block_cache_size {
            let cache = Cache::new_lru_cache(cache_size);
            block_opts.set_block_cache(&cache);
        }
        if config.bloom_filter_bits > 0.0 {
            block_opts.set_bloom_filter(config.bloom_filter_bits, false);
        }
        // Whole-key bloom is enabled explicitly. StateHistoryCf has a
        // 51-byte prefix extractor, and rocksdb's default flips whole-key
        // filtering OFF once any CF uses a prefix extractor — but StateCf
        // (no prefix extractor) and the metadata / receipts / certs CFs
        // all rely on whole-key bloom for their point-lookup-dominated
        // access pattern, so we re-enable it here at the global
        // block-options level.
        block_opts.set_whole_key_filtering(true);
        // Pin SST index/filter blocks inside the bounded block cache instead
        // of letting them consume unbounded heap memory as the DB grows.
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        opts.set_block_based_table_factory(&block_opts);

        // Column families — all share the bounded block cache but get
        // per-CF tuning for write buffers and compression.
        //
        // Hot-write CFs get larger write buffers and tiered compression.
        // Cold/low-volume CFs use smaller write buffers (16MB) to free
        // memory for the hot CFs and block cache.
        let hot_write_cfs = HOT_WRITE_COLUMN_FAMILIES;

        // Tiered compression: L0-L1 uncompressed (fast flushes, data gets
        // compacted away quickly), L2-L4 LZ4, L5+ Zstd.
        let tiered_compression = &[
            DBCompressionType::None, // L0
            DBCompressionType::None, // L1
            DBCompressionType::Lz4,  // L2
            DBCompressionType::Lz4,  // L3
            DBCompressionType::Lz4,  // L4
            DBCompressionType::Zstd, // L5
            DBCompressionType::Zstd, // L6
        ];

        let cold_write_buffer_size: usize = 16 * 1024 * 1024; // 16MB

        let cf_descriptors: Vec<_> = ALL_COLUMN_FAMILIES
            .iter()
            .copied()
            .map(|name| {
                let mut cf_opts = Options::default();
                cf_opts.set_block_based_table_factory(&block_opts);
                cf_opts.set_max_write_buffer_number(config.max_write_buffer_number);

                let is_hot = hot_write_cfs.contains(&name);
                if is_hot {
                    cf_opts.set_write_buffer_size(config.write_buffer_size);
                    cf_opts.set_compression_per_level(tiered_compression);
                } else {
                    cf_opts.set_write_buffer_size(cold_write_buffer_size);
                    cf_opts.set_compression_type(config.compression.to_rocksdb());
                }

                // StateHistoryCf: fixed 32-byte prefix (the substate
                // key's owner ++ local halves) gates historical reads
                // and `list_at_prefix` scans. Keys carry an 8-byte
                // write_version suffix beyond the prefix, so historical
                // seeks at `storage_key ++ BE8(V+1)` and prefix scans
                // both benefit from per-substate SST pruning via prefix
                // bloom over the 32-byte key ahead of the version
                // suffix. StateCf is point-read dominated and uses
                // whole-key bloom only — see its type doc.
                if name == STATE_HISTORY_CF {
                    cf_opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(32));
                }

                ColumnFamilyDescriptor::new(name, cf_opts)
            })
            .collect();

        let db = DB::open_cf_descriptors(&opts, dir.join("db"), cf_descriptors)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        // Validate all expected column families exist at startup.
        // This fails fast instead of panicking on first access at runtime.
        CfHandles::resolve(&db);

        let db = Arc::new(db);
        let checkpoints = super::checkpoints::CheckpointRing::from_db(
            Arc::clone(&db),
            dir.join("checkpoints"),
            config.boundary_retain,
        );

        Ok(Self {
            db,
            commit_lock: Mutex::new(()),
            root_path,
            checkpoints,
            vote_registers: Mutex::new(HashMap::new()),
        })
    }

    /// Resolve all column family handles from the database.
    ///
    /// This is cheap (`HashMap` lookups only, ~10ns per CF) and provides typed
    /// access to all column families without repeating
    /// `.cf_handle(NAME).expect(...)` at each call site.
    pub(crate) fn cf(&self) -> CfHandles<'_> {
        CfHandles::resolve(&self.db)
    }

    // ─── Typed CF helpers ────────────────────────────────────────────────
    //
    // Thin wrappers over the free functions in typed_cf.rs.
    // These resolve CfHandles and pass &self.db as the ReadableStore.
    //
    // Constrained to CFs whose `Handles<'_>` is the shard tier's
    // `CfHandles<'_>` — the beacon RocksDB instance has its own
    // handles struct and its own helper layer.

    /// Get a typed value from a column family.
    pub(crate) fn cf_get<CF>(&self, key: &CF::Key) -> Option<CF::Value>
    where
        for<'a> CF: TypedCf<Handles<'a> = CfHandles<'a>>,
    {
        get::<CF>(&*self.db, CF::handle(&self.cf()), key)
    }

    /// Put a typed key/value into a `WriteBatch`. Production per-block
    /// loops pre-resolve column-family handles outside the loop and call
    /// [`batch_put`] directly; this method is the right shape for one-shot
    /// writes where re-resolving handles per call doesn't matter.
    #[allow(dead_code)]
    pub(crate) fn cf_put<CF>(&self, batch: &mut WriteBatch, key: &CF::Key, value: &CF::Value)
    where
        for<'a> CF: TypedCf<Handles<'a> = CfHandles<'a>>,
    {
        batch_put::<CF>(batch, CF::handle(&self.cf()), key, value);
    }

    /// Batch get typed values (`RocksDB` `multi_get_cf`).
    pub(crate) fn cf_multi_get<CF>(&self, keys: &[CF::Key]) -> Vec<Option<CF::Value>>
    where
        for<'a> CF: TypedCf<Handles<'a> = CfHandles<'a>>,
    {
        multi_get::<CF>(&*self.db, CF::handle(&self.cf()), keys)
    }

    /// Delete a typed key in a `WriteBatch`.
    #[allow(dead_code)]
    pub(crate) fn cf_delete<CF>(&self, batch: &mut WriteBatch, key: &CF::Key)
    where
        for<'a> CF: TypedCf<Handles<'a> = CfHandles<'a>>,
    {
        batch_delete::<CF>(batch, CF::handle(&self.cf()), key);
    }

    /// Typed single put (immediate write, no batch).
    pub(crate) fn cf_put_sync<CF>(&self, key: &CF::Key, value: &CF::Value)
    where
        for<'a> CF: TypedCf<Handles<'a> = CfHandles<'a>>,
    {
        let cf = CF::handle(&self.cf());
        let key_bytes = CF::KeyCodec::default().encode(key);
        let value_bytes = CF::ValueCodec::default().encode(value);
        self.db
            .put_cf(cf, &key_bytes, &value_bytes)
            .expect("BFT CRITICAL: write failed");
    }

    /// Read JMT version and root hash directly from `RocksDB`.
    ///
    /// These are stored as a single 40-byte value under `jmt:metadata`:
    /// `[version_BE_8B][root_hash_32B]`. Always hot in the memtable since
    /// they're written on every commit.
    pub(crate) fn read_jmt_metadata(&self) -> (u64, StateRoot) {
        read_jmt_metadata(&*self.db)
    }

    /// Append JMT data from a snapshot to a `WriteBatch`.
    ///
    /// Writes JMT nodes, stale tree parts (for deferred GC), and JMT
    /// metadata (version + root hash).
    ///
    /// This is the write-side complement to `read_jmt_metadata`.
    pub(crate) fn append_jmt_to_batch(
        &self,
        batch: &mut WriteBatch,
        snapshot: &JmtSnapshot,
        new_version: u64,
    ) {
        // JMT nodes — serialize hydrated nodes to stored form at write time.
        let cf = self.cf();
        for (jmt_key, jmt_node) in &snapshot.nodes {
            let stored_key = StoredNodeKey::from_jmt(jmt_key);
            let stored_node = StoredNode::from_jmt(jmt_node);
            batch_put::<JmtNodesCf>(
                batch,
                JmtNodesCf::handle(&cf),
                &stored_key,
                &VersionedStoredNode::from_latest(stored_node),
            );
        }

        // Stale nodes for deferred GC — keyed by the version at which they became stale.
        if !snapshot.stale_node_keys.is_empty() {
            // Wrap keys as StaleTreePart::Node for wire serialization.
            let stale_parts: Vec<StaleTreePart> = snapshot
                .stale_node_keys
                .iter()
                .map(|k| StaleTreePart::Node(StoredNodeKey::from_jmt(k)))
                .collect();
            batch_put::<StaleJmtNodesCf>(
                batch,
                StaleJmtNodesCf::handle(&cf),
                &new_version,
                &stale_parts,
            );
        }

        // JMT metadata — single key, atomic read.
        write_jmt_metadata(batch, new_version, snapshot.result_root);

        // Committed substate byte total — derived from the byte total behind the
        // currently committed version (the parent of this commit; equal
        // across any interleaved empty commits) plus this commit's leaf
        // delta. Written in the same batch so the count is
        // crash-consistent with the tree. Consensus-critical: witness
        // derivation reads it, so it must be identical on every replica.
        let (current_version, _) = self.read_jmt_metadata();
        let prior = self.substate_bytes_at_version(current_version).unwrap_or(0);
        let count = prior
            .checked_add_signed(snapshot.bytes_delta)
            .expect("substate byte total must not go negative");
        batch_put::<SubstateBytesCf>(batch, SubstateBytesCf::handle(&cf), &new_version, &count);
    }

    /// Committed substate byte total after the commit at `version`,
    /// or `None` if no commit at that version recorded one (never
    /// committed, or pruned past the retention horizon).
    pub fn substate_bytes_at_version(&self, version: u64) -> Option<u64> {
        let cf = self.cf();
        get::<SubstateBytesCf>(&*self.db, SubstateBytesCf::handle(&cf), &version)
    }

    /// Append consensus metadata (`committed_height`, `committed_hash`, `committed_qc`)
    /// to a `WriteBatch` so it is persisted atomically with JMT + substate data.
    pub(crate) fn append_consensus_to_batch(
        batch: &mut WriteBatch,
        block: &Block,
        qc: &Verified<QuorumCertificate>,
    ) {
        write_committed_height(batch, block.height());
        write_committed_hash(batch, block.hash().as_raw());
        write_committed_qc(batch, qc.as_ref());
    }

    /// Build a `WriteBatch` containing all substate puts/deletes from `writes`.
    ///
    /// For each write, captures the prior value (if `write_history`) into
    /// `StateHistoryCf` at `(key, version)` before mutating `StateCf`.
    /// The `write_history` flag lets the genesis / bootstrap path skip
    /// history writes (no pre-state to preserve).
    ///
    /// `pending` is the unpersisted ancestor chain the block was prepared
    /// over. Priors MUST come from it before the persisted store: a batch
    /// built at prepare time applies only after those ancestors have, and
    /// a prior read past them would judge the no-op skip — and record
    /// history — against state older than the parent's. A caller building
    /// at the committed tip passes `&[]`.
    pub(crate) fn build_substate_write_batch(
        &self,
        writes: &SettledWrites,
        version: u64,
        write_history: bool,
        base_reads: Option<&BaseReadCache>,
        pending: &[Arc<JmtSnapshot>],
    ) -> WriteBatch {
        let mut batch = WriteBatch::default();
        self.append_substate_writes_to_batch(
            &mut batch,
            writes,
            version,
            write_history,
            base_reads,
            pending,
        );
        batch
    }

    /// Same as `build_substate_write_batch` but appends to an existing
    /// `WriteBatch`. Used by callers that want to fold substate writes
    /// into a larger atomic batch (e.g. the test-only
    /// `commit_certificate_with_writes`).
    ///
    /// `base_reads`, when provided, is the read cache accumulated by the
    /// originating `SubstateView` during execution. Priors for keys
    /// already in the cache skip the fallback `multi_get_cf`; only keys
    /// NOT in the cache (typically blind writes that weren't preceded
    /// by a read) require a `StateCf` lookup.
    pub(crate) fn append_substate_writes_to_batch(
        &self,
        batch: &mut WriteBatch,
        writes: &SettledWrites,
        version: u64,
        write_history: bool,
        base_reads: Option<&BaseReadCache>,
        pending: &[Arc<JmtSnapshot>],
    ) {
        let cf = self.cf();
        let state_cf = StateCf::handle(&cf);
        let history_cf = StateHistoryCf::handle(&cf);
        let stale_history_cf = StaleStateHistoryCf::handle(&cf);

        // Each write needs its prior value for the state-history entry.
        // The pending overlay answers first — an unpersisted ancestor's
        // write IS the parent state, whatever the store still says. Then
        // the view-cache (`base_reads`), holding what execution read from
        // the persisted base. The rest batch-`multi_get_cf` in one FFI
        // call. Priors aligned 1:1 with `writes.cells` iteration order;
        // `None` entry = miss, needs the multi_get fallback.
        let mut priors: Vec<Option<Option<Vec<u8>>>> = Vec::with_capacity(writes.cells().len());
        let mut miss_keys: Vec<SubstateKey> = Vec::new();
        let mut miss_indices: Vec<usize> = Vec::new();
        for (index, key) in writes.cells().keys().enumerate() {
            if let Some(prior) = pending_write(pending, |settled| settled.cells(), key) {
                priors.push(Some(prior));
                continue;
            }
            if let Some(cache) = base_reads
                && let Some(cached) = cache.get(key)
            {
                priors.push(Some(cached.clone()));
                continue;
            }
            priors.push(None);
            miss_keys.push(*key);
            miss_indices.push(index);
        }

        // Fill cache misses with a single batched StateCf read. This is
        // the fallback for blind writes (keys execution didn't read) and
        // for callers without a view at all (sync path).
        if !miss_keys.is_empty() {
            let fetched: Vec<Option<Vec<u8>>> =
                multi_get::<StateCf>(&*self.db, state_cf, &miss_keys);
            debug_assert_eq!(fetched.len(), miss_indices.len(), "one fetched per miss");
            for (idx, value) in miss_indices.into_iter().zip(fetched) {
                priors[idx] = Some(value);
            }
        }

        // Emit history + state batch puts.
        // Accumulate the raw history keys written so we can record the
        // stale-set entry for this version in one shot.
        let history_key_codec = VersionedSubstateKeyCodec;
        let mut stale_history_keys: Vec<Vec<u8>> = Vec::new();
        let mut sweep_deltas: BTreeMap<(SweepBucket, Address), i64> = BTreeMap::new();
        for ((key, change), prior_slot) in writes.cells().iter().zip(priors) {
            let prior =
                prior_slot.expect("every write must have a resolved prior (cache hit or fetched)");
            record_sweep_delta(&mut sweep_deltas, *key, prior.as_deref(), change.as_deref());
            if let Some(new_value) = change {
                // No-op short-circuit: setting a key to the value it
                // already holds changes nothing. Skip both the history
                // entry (redundant — reads fall through to StateCf which
                // already holds it) and the StateCf put (rocksdb would
                // memtable/compact a useless same-value write).
                let is_noop = matches!(&prior, Some(p) if p == new_value);
                if is_noop {
                    continue;
                }
                if write_history {
                    let history_key = (*key, version);
                    stale_history_keys.push(history_key_codec.encode(&history_key));
                    batch_put::<StateHistoryCf>(batch, history_cf, &history_key, &prior);
                }
                batch_put::<StateCf>(batch, state_cf, key, new_value);
            } else {
                // No-op short-circuit: deleting an absent key is a no-op.
                // Skip both history and state writes.
                if prior.is_none() {
                    continue;
                }
                if write_history {
                    let history_key = (*key, version);
                    stale_history_keys.push(history_key_codec.encode(&history_key));
                    batch_put::<StateHistoryCf>(batch, history_cf, &history_key, &prior);
                }
                batch_delete::<StateCf>(batch, state_cf, key);
            }
        }

        self.append_entry_writes_to_batch(
            batch,
            writes,
            version,
            write_history,
            &mut stale_history_keys,
            pending,
        );

        // A committed cell that self-identifies as a package lands its
        // artifact in the content-addressed index, in the same atomic
        // batch as the state that carries it.
        let artifacts_cf = PackageArtifactsCf::handle(&cf);
        for (key, change) in writes.cells() {
            if let Some(value) = change
                && let Some(package) = package_of_cell(*key, value)
            {
                batch_put::<PackageArtifactsCf>(batch, artifacts_cf, &package, value);
            }
        }

        self.append_sweep_rows_to_batch(batch, &cf, &sweep_deltas);

        // Index the history keys by version so GC can delete them without
        // scanning StateHistoryCf. Skipped when write_history is false
        // (genesis) — nothing was written.
        if write_history && !stale_history_keys.is_empty() {
            batch_put::<StaleStateHistoryCf>(
                batch,
                stale_history_cf,
                &version,
                &stale_history_keys,
            );
        }
    }

    /// Fold this batch's sweep-index deltas into the rows they touch.
    ///
    /// One read-modify-write per touched `(bucket, owner)` pair, which
    /// is at most one per distinct owner a block writes a sweepable cell
    /// for — never one per cell. A row that reaches zero is deleted, so
    /// the index holds exactly the pairs that have something in them and
    /// a sweep's walk skips nothing and visits nothing empty.
    ///
    /// The read is against the persisted store rather than the pending
    /// overlay because the rows are a total over what has *committed*,
    /// and every batch that moved one has already been applied by the
    /// time this one is built.
    ///
    /// # Panics
    ///
    /// If a row would go negative, which means the index disagrees with
    /// the leaves it is derived from — the same class of fault as the
    /// byte total's checked add, and caught here rather than allowed to
    /// under-report a sweep's candidates.
    fn append_sweep_rows_to_batch(
        &self,
        batch: &mut WriteBatch,
        cf: &CfHandles<'_>,
        deltas: &BTreeMap<(SweepBucket, Address), i64>,
    ) {
        if deltas.is_empty() {
            return;
        }
        let sweep_cf = SweepIndexCf::handle(cf);
        let rows: Vec<(SweepBucket, Address)> = deltas.keys().copied().collect();
        let current: Vec<Option<u32>> = multi_get::<SweepIndexCf>(&*self.db, sweep_cf, &rows);
        for ((row, delta), held) in deltas.iter().zip(current) {
            let count = i64::from(held.unwrap_or(0))
                .checked_add(*delta)
                .expect("a sweep-index count stays inside i64");
            let count = u32::try_from(count).unwrap_or_else(|_| {
                panic!("sweep index for {row:?} went to {count}, so it disagrees with the leaves")
            });
            if count == 0 {
                batch_delete::<SweepIndexCf>(batch, sweep_cf, row);
            } else {
                batch_put::<SweepIndexCf>(batch, sweep_cf, row, &count);
            }
        }
    }

    /// The entries half of a substate write batch: each entry's leaf row
    /// rides the same state/history pipeline a cell does, and the
    /// order-keyed index row beside it keeps range scans native.
    ///
    /// Priors are entry values, resolved like the cells': the pending
    /// overlay first, then one batched leaf read for the rest. Each
    /// prior serves both logs — the leaf history re-encodes it, the
    /// index history takes it as it stands — so the two cannot disagree.
    /// Leaf history keys accumulate into the caller's
    /// `stale_history_keys`, which owns the version's stale-set row.
    fn append_entry_writes_to_batch(
        &self,
        batch: &mut WriteBatch,
        writes: &SettledWrites,
        version: u64,
        write_history: bool,
        stale_history_keys: &mut Vec<Vec<u8>>,
        pending: &[Arc<JmtSnapshot>],
    ) {
        if writes.entries().is_empty() {
            return;
        }
        let cf = self.cf();
        let state_cf = StateCf::handle(&cf);
        let history_cf = StateHistoryCf::handle(&cf);
        let entries_cf = EntriesCf::handle(&cf);
        let entries_history_cf = EntriesHistoryCf::handle(&cf);
        let stale_entries_history_cf = StaleEntriesHistoryCf::handle(&cf);

        let mut priors: Vec<Option<Option<Vec<u8>>>> = Vec::with_capacity(writes.entries().len());
        let mut miss_keys: Vec<SubstateKey> = Vec::new();
        let mut miss_indices: Vec<usize> = Vec::new();
        for (index, entry) in writes.entries().keys().enumerate() {
            if let Some(prior) = pending_write(pending, |settled| settled.entries(), entry) {
                priors.push(Some(prior));
                continue;
            }
            priors.push(None);
            miss_keys.push(entry_leaf_key(&ProtocolHasher, *entry));
            miss_indices.push(index);
        }
        if !miss_keys.is_empty() {
            let fetched: Vec<Option<Vec<u8>>> =
                multi_get::<StateCf>(&*self.db, state_cf, &miss_keys);
            debug_assert_eq!(fetched.len(), miss_indices.len(), "one fetched per miss");
            for (idx, leaf) in miss_indices.into_iter().zip(fetched) {
                priors[idx] = Some(leaf.as_deref().map(|bytes| {
                    from_slice::<EntryLeaf>(bytes)
                        .expect("a committed entry leaf decodes")
                        .value
                }));
            }
        }

        let history_key_codec = VersionedSubstateKeyCodec;
        let entries_history_key_codec = VersionedEntryKeyCodec;
        let mut stale_entries_history_keys: Vec<Vec<u8>> = Vec::new();
        for ((entry, change), prior_slot) in writes.entries().iter().zip(priors) {
            let prior =
                prior_slot.expect("every write must have a resolved prior (overlay or fetched)");
            // Setting an entry to the value it already holds changes
            // nothing anywhere; neither does removing an absent one.
            let is_noop = change
                .as_ref()
                .map_or_else(|| prior.is_none(), |new| prior.as_ref() == Some(new));
            if is_noop {
                continue;
            }
            let leaf_key = entry_leaf_key(&ProtocolHasher, *entry);
            if write_history {
                let leaf_prior = prior.as_deref().map(|value| entry_leaf_value(entry, value));
                let history_key = (leaf_key, version);
                stale_history_keys.push(history_key_codec.encode(&history_key));
                batch_put::<StateHistoryCf>(batch, history_cf, &history_key, &leaf_prior);
                let entries_history_key = (*entry, version);
                stale_entries_history_keys
                    .push(entries_history_key_codec.encode(&entries_history_key));
                batch_put::<EntriesHistoryCf>(
                    batch,
                    entries_history_cf,
                    &entries_history_key,
                    &prior,
                );
            }
            if let Some(value) = change {
                batch_put::<StateCf>(batch, state_cf, &leaf_key, &entry_leaf_value(entry, value));
                batch_put::<EntriesCf>(batch, entries_cf, entry, value);
            } else {
                batch_delete::<StateCf>(batch, state_cf, &leaf_key);
                batch_delete::<EntriesCf>(batch, entries_cf, entry);
            }
        }

        if write_history && !stale_entries_history_keys.is_empty() {
            batch_put::<StaleEntriesHistoryCf>(
                batch,
                stale_entries_history_cf,
                &version,
                &stale_entries_history_keys,
            );
        }
    }

    /// Write substate data at version 0 (no JMT computation).
    ///
    /// Genesis-install primitive: writes land in the unversioned `state` CF
    /// with **no state-history entries** — genesis has no pre-state to
    /// preserve. Pair with [`Self::finalize_genesis_jmt`] to compute the JMT
    /// root over the same updates;
    /// [`GenesisCommit::install_genesis`] composes both.
    ///
    /// # Panics
    ///
    /// Panics if the underlying `RocksDB` write fails.
    pub fn commit_substates_only(&self, writes: &SettledWrites) {
        // Genesis writes at version 0. Repeat Sets to the same key
        // overwrite — idempotent by RocksDB write semantics. No history
        // entries: genesis has no pre-state to preserve.
        let batch = self.build_substate_write_batch(
            writes,
            0,
            /* write_history */ false,
            /* base_reads */ None,
            /* pending */ &[],
        );

        // Substates only — no JMT, no sync (genesis isn't durability-critical).
        self.db
            .write(batch)
            .expect("genesis substate-only commit failed");
    }

    /// Compute the JMT once at version 0 from the merged genesis updates.
    ///
    /// Called after [`Self::commit_substates_only`] has placed the substates
    /// in the state CF; this adds the JMT tree for cryptographic commitment.
    ///
    /// # Returns
    /// The genesis state root hash (JMT root at version 0).
    ///
    /// # Panics
    ///
    /// Panics if called after the JMT has already been initialized, or
    /// if the underlying `RocksDB` write fails.
    pub fn finalize_genesis_jmt(&self, merged: &SettledWrites) -> StateRoot {
        let _commit_guard = self.commit_lock.lock().unwrap();

        // Guard: finalize_genesis_jmt must only be called once, on an uninitialized JMT.
        let (current_version, current_root) = self.read_jmt_metadata();
        assert!(
            current_version == 0 && current_root == StateRoot::ZERO,
            "finalize_genesis_jmt called but JMT already initialized (version={current_version})"
        );

        let snapshot_store = SnapshotTreeStore::new(&self.db, self.root_path.clone());

        // parent=None, version=0: genesis is the first JMT state.
        let (root, collected) = tree::put_at_version(&snapshot_store, None, 0, merged);
        let jmt_snapshot = JmtSnapshot::from_collected_writes(
            collected,
            merged.clone(),
            StateRoot::ZERO,
            BlockHeight::GENESIS,
            root,
            BlockHeight::GENESIS,
        );

        let mut batch = WriteBatch::default();
        self.append_jmt_to_batch(&mut batch, &jmt_snapshot, 0);

        self.db
            .write(batch)
            .expect("genesis JMT finalization failed");

        root
    }
}

impl GenesisCommit for RocksDbShardStorage {
    fn install_genesis(&self, substates: &SettledWrites, jmt_writes: &SettledWrites) -> StateRoot {
        Self::commit_substates_only(self, substates);
        Self::finalize_genesis_jmt(self, jmt_writes)
    }

    fn replicate_genesis_substates(&self, substates: &SettledWrites) {
        Self::commit_substates_only(self, substates);
    }
}

impl Substates for RocksDbShardStorage {
    #[instrument(level = Level::DEBUG, skip_all, fields(
        found = Empty,
        latency_us = Empty,
    ))]
    fn cell(&self, key: SubstateKey) -> Option<Vec<u8>> {
        // Default-version snapshot (= current committed tip) reads
        // the latest value for this key. Delegating to `snapshot()`
        // keeps a single read path.
        let start = Instant::now();
        let result = <Self as SubstateStore>::snapshot(self).cell(key);
        let elapsed = start.elapsed();
        record_storage_read(elapsed.as_secs_f64());

        let span = Span::current();
        span.record("found", result.is_some());
        span.record(
            "latency_us",
            u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
        );

        result
    }

    fn entries_in_range(
        &self,
        owner: Address,
        collection: CollectionId,
        lo: u128,
        hi: u128,
        limit: usize,
    ) -> Vec<(u128, Vec<u8>)> {
        // Default-version snapshot, like `cell` — one read path.
        <Self as SubstateStore>::snapshot(self).entries_in_range(owner, collection, lo, hi, limit)
    }
}

impl TreeReader for RocksDbShardStorage {
    fn get_node(&self, key: &JmtNodeKey) -> Option<Arc<JmtNode>> {
        let stored_key = StoredNodeKey::from_jmt(key);
        self.cf_get::<JmtNodesCf>(&stored_key)
            .map(|v| Arc::new(v.into_latest().to_jmt()))
    }

    fn get_root_key(&self, version: u64) -> Option<JmtNodeKey> {
        let root = JmtNodeKey::new(version, self.root_path.clone());
        let stored_key = StoredNodeKey::from_jmt(&root);
        if self.cf_get::<JmtNodesCf>(&stored_key).is_some() {
            Some(root)
        } else {
            None
        }
    }

    fn root_path(&self) -> NibblePath {
        self.root_path.clone()
    }
}

#[cfg(test)]
mod test_helpers {
    use hyperscale_metrics::record_storage_write;
    use hyperscale_types::WeightedTimestamp;

    use super::*;

    impl RocksDbShardStorage {
        /// Test helper: commits database updates with auto-incrementing JMT version.
        /// Production uses `commit_block` / `commit_prepared_block` instead.
        ///
        /// # Errors
        ///
        /// Returns [`StorageError`] if the underlying `RocksDB` write fails.
        ///
        /// # Panics
        ///
        /// Panics if the commit lock is poisoned.
        #[instrument(level = Level::DEBUG, skip_all, fields(
            cell_count = writes.cells().len(),
            latency_us = Empty,
        ))]
        pub fn commit(&self, writes: &SettledWrites) -> Result<(), StorageError> {
            self.commit_at(writes, WeightedTimestamp::from_millis(0))
        }

        /// [`Self::commit`] with the version dated, for a test that wants
        /// the retention floor to move: a commit a horizon's worth of
        /// weighted time after an earlier one retires it.
        ///
        /// # Errors
        ///
        /// Returns [`StorageError`] if the underlying `RocksDB` write fails.
        ///
        /// # Panics
        ///
        /// Panics if the commit lock is poisoned.
        pub fn commit_at(
            &self,
            writes: &SettledWrites,
            at: WeightedTimestamp,
        ) -> Result<(), StorageError> {
            let _commit_guard = self.commit_lock.lock().unwrap();

            let start = Instant::now();

            // Compute JMT updates using a snapshot-based store for isolation
            let snapshot_store = SnapshotTreeStore::new(&self.db, self.root_path.clone());
            let (base_version, base_root) = snapshot_store.read_jmt_metadata();

            // Version 0 with a non-zero root means genesis has been computed at version 0.
            // Only treat as "no parent" when the JMT is truly empty.
            let parent_version = tree::jmt_parent_height(BlockHeight::new(base_version), base_root)
                .map(BlockHeight::inner);
            let new_version = base_version + 1;

            let mut batch = self.build_substate_write_batch(
                writes,
                new_version,
                /* write_history */ true,
                /* base_reads */ None,
                /* pending */ &[],
            );

            let (new_root, collected) =
                tree::put_at_version(&snapshot_store, parent_version, new_version, writes);
            let jmt_snapshot = JmtSnapshot::from_collected_writes(
                collected,
                writes.clone(),
                base_root,
                BlockHeight::new(base_version),
                new_root,
                BlockHeight::new(new_version),
            );

            self.append_jmt_to_batch(&mut batch, &jmt_snapshot, new_version);
            self.advance_retention_floor(&mut batch, new_version, at);

            self.db
                .write(batch)
                .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

            let elapsed = start.elapsed();
            record_storage_write(elapsed.as_secs_f64());

            // Record span fields
            let span = Span::current();
            span.record(
                "latency_us",
                u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
            );
            tracing::debug!(new_version, "commit complete");

            Ok(())
        }
    }
}
