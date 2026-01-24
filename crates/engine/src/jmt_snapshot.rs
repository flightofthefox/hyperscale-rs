//! JMT snapshot for deferred application of speculative computation.
//!
//! This module provides [`JmtSnapshot`], which captures JMT nodes computed during
//! speculative state root computation. The snapshot can be cached and applied
//! during block commit, avoiding redundant recomputation.

use crate::{StaleTreePart, StateRootHash, StoredTreeNodeKey, TreeNode};
use std::collections::HashMap;

/// A snapshot of JMT nodes computed during speculative execution.
///
/// Created during speculative state root computation (e.g., block verification).
/// Can be applied to the real JMT during block commit, avoiding redundant computation.
///
/// # Usage
///
/// ```ignore
/// // During verification (in production storage)
/// let overlay = OverlayTreeStore::new(&storage);
/// let root = compute_root(&overlay, writes);
/// let snapshot = overlay.into_snapshot(base_root, base_version, root, num_certs);
/// cache.insert(block_hash, snapshot);
///
/// // During commit
/// let snapshot = cache.remove(&block_hash);
/// storage.apply_jmt_snapshot(snapshot);
/// ```
#[derive(Debug, Clone)]
pub struct JmtSnapshot {
    /// The JMT root this snapshot was computed from.
    /// Used to verify the JMT is in the expected state before applying.
    pub base_root: StateRootHash,

    /// The JMT version this snapshot was computed from.
    /// Used together with base_root to verify the JMT is in the expected state.
    pub base_version: u64,

    /// The resulting state root after applying all certificate writes.
    pub result_root: StateRootHash,

    /// Number of JMT versions this snapshot advances.
    /// Equal to the number of certificates processed.
    pub num_versions: u64,

    /// Nodes created during speculative computation.
    /// These are inserted directly into the real JMT on apply.
    pub nodes: HashMap<StoredTreeNodeKey, TreeNode>,

    /// Stale tree parts to prune when applying the snapshot.
    pub stale_tree_parts: Vec<StaleTreePart>,
}
