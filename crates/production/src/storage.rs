//! # RocksDB Storage
//!
//! Production storage implementation using RocksDB.
//!
//! All operations are synchronous blocking I/O. Callers in async contexts
//! should use `spawn_blocking` if needed to avoid blocking the runtime.

use hyperscale_engine::{
    keys, CommittableSubstateDatabase, DatabaseUpdate, DatabaseUpdates, DbPartitionKey, DbSortKey,
    DbSubstateValue, PartitionDatabaseUpdates, PartitionEntry, SubstateDatabase, SubstateStore,
};
use hyperscale_types::NodeId;
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, WriteBatch, DB};
use sbor::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// RocksDB-based storage for production use.
///
/// Features:
/// - Column families for logical separation
/// - LZ4 compression for disk efficiency
/// - Block cache for read performance
/// - Bloom filters for key existence checks
///
/// Implements Radix's `SubstateDatabase` and `CommittableSubstateDatabase` directly,
/// plus our `SubstateStore` extension for snapshots and node listing.
pub struct RocksDbStorage {
    db: Arc<DB>,
}

/// Error type for storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// Database error
    #[error("Database error: {0}")]
    DatabaseError(String),
}

impl RocksDbStorage {
    /// Open or create a RocksDB database at the given path.
    ///
    /// Creates default column families: default, blocks, transactions, state, certificates.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let config = RocksDbConfig::default();
        Self::open_with_config(path, config)
    }

    /// Open with custom configuration.
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: RocksDbConfig,
    ) -> Result<Self, StorageError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Performance tuning
        opts.set_max_background_jobs(config.max_background_jobs);
        opts.set_bytes_per_sync(1024 * 1024); // 1MB
        opts.set_keep_log_file_num(10);
        opts.set_max_write_buffer_number(config.max_write_buffer_number);
        opts.set_write_buffer_size(config.write_buffer_size);

        // Compression
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Block cache and bloom filter
        if let Some(cache_size) = config.block_cache_size {
            let cache = rocksdb::Cache::new_lru_cache(cache_size);
            let mut block_opts = rocksdb::BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            block_opts.set_bloom_filter(10.0, false);
            opts.set_block_based_table_factory(&block_opts);
        }

        // Column families
        let cf_descriptors: Vec<_> = config
            .column_families
            .into_iter()
            .map(|name| ColumnFamilyDescriptor::new(name, Options::default()))
            .collect();

        let db = DB::open_cf_descriptors(&opts, path, cf_descriptors)
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        Ok(Self { db: Arc::new(db) })
    }

    /// Get a column family handle by name.
    #[allow(dead_code)]
    fn cf(&self, name: &str) -> Result<&ColumnFamily, StorageError> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| StorageError::DatabaseError(format!("Column family {} not found", name)))
    }

    /// Internal: iterate over a key range.
    fn iter_range(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let iter = self.db.iterator(rocksdb::IteratorMode::From(
            start,
            rocksdb::Direction::Forward,
        ));

        iter.take_while(|item| match item {
            Ok((key, _)) => key.as_ref() < end,
            Err(_) => false,
        })
        .filter_map(|item| item.ok().map(|(k, v)| (k.to_vec(), v.to_vec())))
        .collect()
    }
}

impl SubstateDatabase for RocksDbStorage {
    fn get_raw_substate_by_db_key(
        &self,
        partition_key: &DbPartitionKey,
        sort_key: &DbSortKey,
    ) -> Option<DbSubstateValue> {
        let key = keys::to_storage_key(partition_key, sort_key);
        self.db.get(&key).ok().flatten()
    }

    fn list_raw_values_from_db_key(
        &self,
        partition_key: &DbPartitionKey,
        from_sort_key: Option<&DbSortKey>,
    ) -> Box<dyn Iterator<Item = PartitionEntry> + '_> {
        let prefix = keys::partition_prefix(partition_key);
        let prefix_len = prefix.len();

        let start = match from_sort_key {
            Some(sort_key) => {
                let mut s = prefix.clone();
                s.extend_from_slice(&sort_key.0);
                s
            }
            None => prefix.clone(),
        };
        let end = keys::next_prefix(&prefix);

        let items = self.iter_range(&start, &end);

        Box::new(items.into_iter().filter_map(move |(full_key, value)| {
            if full_key.len() > prefix_len {
                let sort_key_bytes = full_key[prefix_len..].to_vec();
                Some((DbSortKey(sort_key_bytes), value))
            } else {
                None
            }
        }))
    }
}

impl CommittableSubstateDatabase for RocksDbStorage {
    fn commit(&mut self, updates: &DatabaseUpdates) {
        let mut batch = WriteBatch::default();

        for (node_key, node_updates) in &updates.node_updates {
            for (partition_num, partition_updates) in &node_updates.partition_updates {
                let partition_key = DbPartitionKey {
                    node_key: node_key.clone(),
                    partition_num: *partition_num,
                };

                match partition_updates {
                    PartitionDatabaseUpdates::Delta { substate_updates } => {
                        for (sort_key, update) in substate_updates {
                            let key = keys::to_storage_key(&partition_key, sort_key);
                            match update {
                                DatabaseUpdate::Set(value) => {
                                    batch.put(&key, value);
                                }
                                DatabaseUpdate::Delete => {
                                    batch.delete(&key);
                                }
                            }
                        }
                    }
                    PartitionDatabaseUpdates::Reset {
                        new_substate_values,
                    } => {
                        // Delete all existing in partition
                        let prefix = keys::partition_prefix(&partition_key);
                        let end = keys::next_prefix(&prefix);

                        for (key, _) in self.iter_range(&prefix, &end) {
                            batch.delete(&key);
                        }

                        // Insert new values
                        for (sort_key, value) in new_substate_values {
                            let key = keys::to_storage_key(&partition_key, sort_key);
                            batch.put(&key, value);
                        }
                    }
                }
            }
        }

        // Write batch atomically
        if let Err(e) = self.db.write(batch) {
            tracing::error!("Failed to commit updates: {}", e);
        }
    }
}

impl SubstateStore for RocksDbStorage {
    type Snapshot = RocksDbSnapshot;

    fn snapshot(&self) -> Arc<Self::Snapshot> {
        // Note: Currently uses the DB directly for snapshot reads.
        // In the future, this could be optimized with RocksDB's native snapshot feature.
        Arc::new(RocksDbSnapshot {
            db: self.db.clone(),
        })
    }

    fn list_substates_for_node(
        &self,
        node_id: &NodeId,
    ) -> Box<dyn Iterator<Item = (u8, DbSortKey, Vec<u8>)> + '_> {
        let prefix = keys::node_prefix(node_id);
        let prefix_len = prefix.len();
        let end = keys::next_prefix(&prefix);

        let items = self.iter_range(&prefix, &end);

        Box::new(items.into_iter().filter_map(move |(full_key, value)| {
            if full_key.len() > prefix_len {
                let partition_num = full_key[prefix_len];
                let sort_key_bytes = full_key[prefix_len + 1..].to_vec();
                Some((partition_num, DbSortKey(sort_key_bytes), value))
            } else {
                None
            }
        }))
    }
}

/// RocksDB snapshot for consistent reads.
///
/// Note: Currently uses the DB directly. In the future, this could be
/// optimized to use RocksDB's native snapshot feature.
pub struct RocksDbSnapshot {
    db: Arc<DB>,
}

impl SubstateDatabase for RocksDbSnapshot {
    fn get_raw_substate_by_db_key(
        &self,
        partition_key: &DbPartitionKey,
        sort_key: &DbSortKey,
    ) -> Option<DbSubstateValue> {
        let key = keys::to_storage_key(partition_key, sort_key);
        self.db.get(&key).ok().flatten()
    }

    fn list_raw_values_from_db_key(
        &self,
        partition_key: &DbPartitionKey,
        from_sort_key: Option<&DbSortKey>,
    ) -> Box<dyn Iterator<Item = PartitionEntry> + '_> {
        let prefix = keys::partition_prefix(partition_key);
        let prefix_len = prefix.len();

        let start = match from_sort_key {
            Some(sort_key) => {
                let mut s = prefix.clone();
                s.extend_from_slice(&sort_key.0);
                s
            }
            None => prefix.clone(),
        };
        let end = keys::next_prefix(&prefix);

        // Collect into Vec to avoid lifetime issues with RocksDB iterators
        let iter = self.db.iterator(rocksdb::IteratorMode::From(
            &start,
            rocksdb::Direction::Forward,
        ));

        let items: Vec<_> = iter
            .take_while(|item| match item {
                Ok((key, _)) => key.as_ref() < end.as_slice(),
                Err(_) => false,
            })
            .filter_map(|item| item.ok().map(|(k, v)| (k.to_vec(), v.to_vec())))
            .collect();

        Box::new(items.into_iter().filter_map(move |(full_key, value)| {
            if full_key.len() > prefix_len {
                let sort_key_bytes = full_key[prefix_len..].to_vec();
                Some((DbSortKey(sort_key_bytes), value))
            } else {
                None
            }
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Block storage
// ═══════════════════════════════════════════════════════════════════════

use hyperscale_types::{Block, BlockHeight, Hash, QuorumCertificate, TransactionCertificate};

impl RocksDbStorage {
    /// Store a committed block with its quorum certificate.
    pub fn put_block(&self, height: BlockHeight, block: &Block, qc: &QuorumCertificate) {
        let cf = match self.db.cf_handle("blocks") {
            Some(cf) => cf,
            None => {
                tracing::error!("blocks column family not found");
                return;
            }
        };

        // Key: height as big-endian bytes (for natural ordering)
        let key = height.0.to_be_bytes();

        // Value: SBOR-encoded (block, qc) tuple
        let value = match sbor::basic_encode(&(block, qc)) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Failed to encode block: {:?}", e);
                return;
            }
        };

        if let Err(e) = self.db.put_cf(cf, key, value) {
            tracing::error!("Failed to store block at height {}: {}", height.0, e);
        }
    }

    /// Get a committed block by height.
    pub fn get_block(&self, height: BlockHeight) -> Option<(Block, QuorumCertificate)> {
        let cf = self.db.cf_handle("blocks")?;
        let key = height.0.to_be_bytes();

        match self.db.get_cf(cf, key) {
            Ok(Some(value)) => match sbor::basic_decode::<(Block, QuorumCertificate)>(&value) {
                Ok(result) => Some(result),
                Err(e) => {
                    tracing::error!("Failed to decode block at height {}: {:?}", height.0, e);
                    None
                }
            },
            Ok(None) => None,
            Err(e) => {
                tracing::error!("Failed to read block at height {}: {}", height.0, e);
                None
            }
        }
    }

    /// Get a range of committed blocks [from, to).
    ///
    /// Returns blocks in ascending height order.
    pub fn get_blocks_range(
        &self,
        from: BlockHeight,
        to: BlockHeight,
    ) -> Vec<(Block, QuorumCertificate)> {
        let cf = match self.db.cf_handle("blocks") {
            Some(cf) => cf,
            None => return vec![],
        };

        let start_key = from.0.to_be_bytes();
        let end_key = to.0.to_be_bytes();

        let iter = self.db.iterator_cf(
            cf,
            rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
        );

        iter.take_while(|item| match item {
            Ok((key, _)) => key.as_ref() < end_key.as_slice(),
            Err(_) => false,
        })
        .filter_map(|item| {
            item.ok().and_then(|(_, value)| {
                sbor::basic_decode::<(Block, QuorumCertificate)>(&value).ok()
            })
        })
        .collect()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Chain metadata
    // ═══════════════════════════════════════════════════════════════════════

    /// Set the highest committed block height and hash.
    pub fn set_chain_metadata(
        &self,
        height: BlockHeight,
        hash: Option<Hash>,
        qc: Option<&QuorumCertificate>,
    ) {
        // Store in default column family with well-known keys
        if let Err(e) = self
            .db
            .put(b"chain:committed_height", height.0.to_be_bytes())
        {
            tracing::error!("Failed to store committed height: {}", e);
        }

        if let Some(h) = hash {
            if let Err(e) = self.db.put(b"chain:committed_hash", h.as_bytes()) {
                tracing::error!("Failed to store committed hash: {}", e);
            }
        }

        if let Some(qc) = qc {
            if let Ok(encoded) = sbor::basic_encode(qc) {
                if let Err(e) = self.db.put(b"chain:committed_qc", encoded) {
                    tracing::error!("Failed to store committed QC: {}", e);
                }
            }
        }
    }

    /// Get the chain metadata (committed height, hash, and QC).
    pub fn get_chain_metadata(&self) -> (BlockHeight, Option<Hash>, Option<QuorumCertificate>) {
        let height = self
            .db
            .get(b"chain:committed_height")
            .ok()
            .flatten()
            .map(|v| {
                let bytes: [u8; 8] = v.as_slice().try_into().unwrap_or([0; 8]);
                BlockHeight(u64::from_be_bytes(bytes))
            })
            .unwrap_or(BlockHeight(0));

        let hash = self
            .db
            .get(b"chain:committed_hash")
            .ok()
            .flatten()
            .map(|v| Hash::from_bytes(&v));

        let qc = self
            .db
            .get(b"chain:committed_qc")
            .ok()
            .flatten()
            .and_then(|v| sbor::basic_decode(&v).ok());

        (height, hash, qc)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Certificate storage
    // ═══════════════════════════════════════════════════════════════════════

    /// Store a transaction certificate.
    pub fn put_certificate(&self, hash: &Hash, cert: &TransactionCertificate) {
        let cf = match self.db.cf_handle("certificates") {
            Some(cf) => cf,
            None => {
                tracing::error!("certificates column family not found");
                return;
            }
        };

        let value = match sbor::basic_encode(cert) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Failed to encode certificate: {:?}", e);
                return;
            }
        };

        if let Err(e) = self.db.put_cf(cf, hash.as_bytes(), value) {
            tracing::error!("Failed to store certificate: {}", e);
        }
    }

    /// Get a transaction certificate by transaction hash.
    pub fn get_certificate(&self, hash: &Hash) -> Option<TransactionCertificate> {
        let cf = self.db.cf_handle("certificates")?;

        match self.db.get_cf(cf, hash.as_bytes()) {
            Ok(Some(value)) => sbor::basic_decode(&value).ok(),
            _ => None,
        }
    }
}

/// Configuration for RocksDB storage.
#[derive(Debug, Clone)]
pub struct RocksDbConfig {
    /// Maximum number of background jobs
    pub max_background_jobs: i32,
    /// Write buffer size in bytes
    pub write_buffer_size: usize,
    /// Maximum number of write buffers
    pub max_write_buffer_number: i32,
    /// Block cache size in bytes (None to disable)
    pub block_cache_size: Option<usize>,
    /// Column families to create
    pub column_families: Vec<String>,
}

impl Default for RocksDbConfig {
    fn default() -> Self {
        Self {
            max_background_jobs: 4,
            write_buffer_size: 128 * 1024 * 1024, // 128MB
            max_write_buffer_number: 3,
            block_cache_size: Some(512 * 1024 * 1024), // 512MB
            column_families: vec![
                "default".to_string(),
                "blocks".to_string(),
                "transactions".to_string(),
                "state".to_string(),
                "certificates".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_engine::NodeDatabaseUpdates;
    use tempfile::TempDir;

    #[test]
    fn test_basic_substate_operations() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = RocksDbStorage::open(temp_dir.path()).unwrap();

        let partition_key = DbPartitionKey {
            node_key: vec![1, 2, 3],
            partition_num: 0,
        };
        let sort_key = DbSortKey(vec![10, 20]);

        // Initially empty
        assert!(storage
            .get_raw_substate_by_db_key(&partition_key, &sort_key)
            .is_none());

        // Commit a value
        let mut updates = DatabaseUpdates::default();
        updates.node_updates.insert(
            partition_key.node_key.clone(),
            NodeDatabaseUpdates {
                partition_updates: [(
                    partition_key.partition_num,
                    PartitionDatabaseUpdates::Delta {
                        substate_updates: [(
                            sort_key.clone(),
                            DatabaseUpdate::Set(vec![99, 88, 77]),
                        )]
                        .into_iter()
                        .collect(),
                    },
                )]
                .into_iter()
                .collect(),
            },
        );
        storage.commit(&updates);

        // Now we can read it
        let value = storage.get_raw_substate_by_db_key(&partition_key, &sort_key);
        assert_eq!(value, Some(vec![99, 88, 77]));
    }

    #[test]
    fn test_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = RocksDbStorage::open(temp_dir.path()).unwrap();

        let partition_key = DbPartitionKey {
            node_key: vec![1, 2, 3],
            partition_num: 0,
        };
        let sort_key = DbSortKey(vec![10]);

        // Write initial value
        let mut updates = DatabaseUpdates::default();
        updates.node_updates.insert(
            partition_key.node_key.clone(),
            NodeDatabaseUpdates {
                partition_updates: [(
                    partition_key.partition_num,
                    PartitionDatabaseUpdates::Delta {
                        substate_updates: [(sort_key.clone(), DatabaseUpdate::Set(vec![1]))]
                            .into_iter()
                            .collect(),
                    },
                )]
                .into_iter()
                .collect(),
            },
        );
        storage.commit(&updates);

        // Take snapshot
        let snapshot = storage.snapshot();

        // Snapshot can read data
        assert_eq!(
            snapshot.get_raw_substate_by_db_key(&partition_key, &sort_key),
            Some(vec![1])
        );

        // Note: Current implementation doesn't provide point-in-time isolation
        // This is acceptable for Phase 1 and can be optimized later with
        // RocksDB's native snapshot feature if needed
    }
}
