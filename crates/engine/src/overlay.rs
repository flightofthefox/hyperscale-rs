//! Overlay tree store for speculative JMT computation.
//!
//! This module provides `OverlayTreeStore`, which wraps a base tree store and
//! captures all writes without modifying the underlying storage. This enables
//! speculative state root computation for block validation.

use crate::{
    AssociatedSubstateValue, DbPartitionKey, DbSortKey, ReadableTreeStore, StaleTreePart,
    StoredTreeNodeKey, TreeNode, TypedInMemoryTreeStore, WriteableTreeStore,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// An overlay tree store that captures writes without modifying the underlying store.
///
/// Reads check the overlay first, then fall through to the base store.
/// Writes only go to the overlay and are discarded when the overlay is dropped.
///
/// This enables speculative JMT root computation where we need to compute what
/// the root WOULD be without actually persisting any nodes. Multiple concurrent
/// speculative computations can run without corrupting each other.
///
/// # Example
///
/// ```ignore
/// use hyperscale_engine::{OverlayTreeStore, put_at_next_version, TypedInMemoryTreeStore};
///
/// let base_store = TypedInMemoryTreeStore::new();
/// let overlay = OverlayTreeStore::new(&base_store);
///
/// // Compute speculative root - writes go to overlay only
/// let new_root = put_at_next_version(&overlay, Some(current_version), &updates);
///
/// // overlay is dropped here - base_store is unchanged
/// ```
pub struct OverlayTreeStore<'a> {
    /// The underlying tree store (read-only access).
    base: &'a TypedInMemoryTreeStore,

    /// Overlay of node insertions. Maps key -> node.
    /// Nodes inserted during speculative computation are stored here.
    inserted_nodes: RefCell<HashMap<StoredTreeNodeKey, TreeNode>>,

    /// Set of keys that have been marked as stale (deleted) in the overlay.
    /// Lookups for these keys return None even if they exist in the base store.
    stale_keys: RefCell<HashSet<StoredTreeNodeKey>>,
}

impl<'a> OverlayTreeStore<'a> {
    /// Create a new overlay wrapping the given base store.
    pub fn new(base: &'a TypedInMemoryTreeStore) -> Self {
        Self {
            base,
            inserted_nodes: RefCell::new(HashMap::new()),
            stale_keys: RefCell::new(HashSet::new()),
        }
    }
}

impl<'a> ReadableTreeStore for OverlayTreeStore<'a> {
    fn get_node(&self, key: &StoredTreeNodeKey) -> Option<TreeNode> {
        // Check if the key was marked as stale (deleted)
        if self.stale_keys.borrow().contains(key) {
            return None;
        }

        // Check overlay first
        if let Some(node) = self.inserted_nodes.borrow().get(key) {
            return Some(node.clone());
        }

        // Fall through to base store
        self.base.get_node(key)
    }
}

impl<'a> WriteableTreeStore for OverlayTreeStore<'a> {
    fn insert_node(&self, key: StoredTreeNodeKey, node: TreeNode) {
        // Remove from stale set if it was previously marked stale
        self.stale_keys.borrow_mut().remove(&key);
        // Insert into overlay
        self.inserted_nodes.borrow_mut().insert(key, node);
    }

    fn associate_substate(
        &self,
        _state_tree_leaf_key: &StoredTreeNodeKey,
        _partition_key: &DbPartitionKey,
        _sort_key: &DbSortKey,
        _substate_value: AssociatedSubstateValue,
    ) {
        // No-op for speculative computation.
        // Substate associations are only needed for historical queries.
    }

    fn record_stale_tree_part(&self, part: StaleTreePart) {
        // Mark nodes as stale in the overlay so subsequent reads return None.
        //
        // NOTE: For `StaleTreePart::Subtree`, we only mark the root as stale rather than
        // recursively walking the entire subtree. This is correct for speculative computation
        // because:
        //
        // 1. Stale nodes are from the PREVIOUS version of the tree (before our speculative update)
        // 2. The speculative computation creates NEW nodes at a NEW version number
        // 3. Reads during `put_at_next_version` use version-qualified keys: (version, nibble_path)
        // 4. New nodes are stored at version N+1, stale nodes were at version N
        // 5. Therefore, reads for child nodes will either:
        //    a) Find them in the overlay (newly created) - correct
        //    b) Find them in the base store at their old version - correct (we're reading
        //       parent chain nodes that weren't modified)
        //    c) Not find them because the whole subtree was replaced - also correct
        //
        // The real tree store DOES need full recursive deletion for garbage collection,
        // but the overlay is temporary and discarded after computing the speculative root.
        match part {
            StaleTreePart::Node(key) => {
                self.stale_keys.borrow_mut().insert(key);
            }
            StaleTreePart::Subtree(root_key) => {
                // Mark only the subtree root as stale. See explanation above for why
                // recursive deletion is not needed for speculative computation.
                self.stale_keys.borrow_mut().insert(root_key);
            }
        }
    }
}
