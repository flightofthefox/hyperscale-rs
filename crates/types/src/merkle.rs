//! Merkle tree utilities for batched vote signatures.
//!
//! This module provides a simple binary Merkle tree implementation optimized
//! for batching state vote signatures. Each validator signs the Merkle root
//! of all their votes, and individual votes carry inclusion proofs.
//!
//! # Performance
//!
//! - Tree construction: O(n) hashes for n leaves
//! - Proof generation: O(log n) per proof, O(n log n) total
//! - Proof verification: O(log n) hashes
//!
//! For 1000 votes, this is ~10,000 hash operations vs 1000 BLS signatures,
//! providing ~100x speedup since hashing is ~1000x faster than BLS signing.

use crate::Hash;
use sbor::prelude::*;

/// Merkle inclusion proof for a leaf in a binary Merkle tree.
///
/// The proof consists of sibling hashes from the leaf to the root.
/// Verification recomputes the path and checks against the expected root.
#[derive(Clone, Debug, PartialEq, Eq, BasicSbor)]
pub struct MerkleProof {
    /// Index of the leaf in the tree (0-based).
    pub leaf_index: u32,

    /// Sibling hashes from leaf to root.
    ///
    /// For a tree of depth d, this contains d-1 hashes.
    /// siblings[0] is the immediate sibling, siblings[d-2] is near the root.
    pub siblings: Vec<Hash>,
}

impl MerkleProof {
    /// Verify that `leaf_hash` is included in `root` at `leaf_index`.
    ///
    /// Returns true if the proof is valid.
    pub fn verify(&self, leaf_hash: &Hash, root: &Hash) -> bool {
        let mut current = *leaf_hash;
        let mut index = self.leaf_index;

        for sibling in &self.siblings {
            current = if index.is_multiple_of(2) {
                // Current is left child, sibling is right
                hash_pair(&current, sibling)
            } else {
                // Current is right child, sibling is left
                hash_pair(sibling, &current)
            };
            index /= 2;
        }

        current == *root
    }

    /// Get the depth of the tree this proof is for.
    pub fn depth(&self) -> usize {
        self.siblings.len()
    }
}

/// Hash two child nodes to produce parent hash.
///
/// Uses Blake3 with concatenated inputs for efficiency.
#[inline]
fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    let mut data = [0u8; 64];
    data[..32].copy_from_slice(left.as_bytes());
    data[32..].copy_from_slice(right.as_bytes());
    Hash::from_bytes(&data)
}

/// Build a Merkle tree from leaf hashes and generate proofs for all leaves.
///
/// Returns the Merkle root and a proof for each leaf (in the same order as input).
///
/// # Algorithm
///
/// 1. Pad leaves to next power of 2 with zero hashes
/// 2. Build tree bottom-up, storing all intermediate nodes
/// 3. Extract sibling path for each original leaf
///
/// # Panics
///
/// Panics if `leaves` is empty.
pub fn build_merkle_tree_with_proofs(leaves: &[Hash]) -> (Hash, Vec<MerkleProof>) {
    assert!(
        !leaves.is_empty(),
        "Cannot build Merkle tree with no leaves"
    );

    // Handle single leaf case
    if leaves.len() == 1 {
        return (
            leaves[0],
            vec![MerkleProof {
                leaf_index: 0,
                siblings: vec![],
            }],
        );
    }

    // Pad to next power of 2
    let n = leaves.len().next_power_of_two();
    let depth = n.trailing_zeros() as usize;

    // Allocate tree storage: 2n - 1 nodes for n leaves
    // Layout: [leaves (n), level 1 (n/2), level 2 (n/4), ..., root (1)]
    let mut tree = vec![Hash::ZERO; 2 * n - 1];

    // Copy leaves to start of tree
    tree[..leaves.len()].copy_from_slice(leaves);
    // Remaining leaf slots already zeroed (padding)

    // Build tree bottom-up
    let mut level_start = 0;
    let mut level_size = n;

    for _ in 0..depth {
        let next_level_start = level_start + level_size;
        let next_level_size = level_size / 2;

        for i in 0..next_level_size {
            let left = &tree[level_start + 2 * i];
            let right = &tree[level_start + 2 * i + 1];
            tree[next_level_start + i] = hash_pair(left, right);
        }

        level_start = next_level_start;
        level_size = next_level_size;
    }

    // Root is the last element
    let root = tree[tree.len() - 1];

    // Generate proofs for original leaves only
    let proofs: Vec<MerkleProof> = (0..leaves.len())
        .map(|leaf_idx| {
            let mut siblings = Vec::with_capacity(depth);
            let mut level_start = 0;
            let mut level_size = n;
            let mut idx = leaf_idx;

            for _ in 0..depth {
                // Sibling index at this level
                let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
                siblings.push(tree[level_start + sibling_idx]);

                // Move to parent level
                level_start += level_size;
                level_size /= 2;
                idx /= 2;
            }

            MerkleProof {
                leaf_index: leaf_idx as u32,
                siblings,
            }
        })
        .collect();

    (root, proofs)
}

/// Compute the leaf hash for a state vote.
///
/// This produces a deterministic hash of the vote contents for Merkle tree inclusion.
/// The format matches what validators sign, ensuring the proof covers the actual vote data.
pub fn vote_leaf_hash(
    tx_hash: &Hash,
    state_root: &Hash,
    shard_group_id: u64,
    success: bool,
) -> Hash {
    // Pre-allocate exact size: 32 + 32 + 8 + 1 = 73 bytes
    let mut data = Vec::with_capacity(73);
    data.extend_from_slice(tx_hash.as_bytes());
    data.extend_from_slice(state_root.as_bytes());
    data.extend_from_slice(&shard_group_id.to_le_bytes());
    data.push(if success { 1 } else { 0 });
    Hash::from_bytes(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_leaf() {
        let leaf = Hash::from_bytes(b"single leaf");
        let (root, proofs) = build_merkle_tree_with_proofs(&[leaf]);

        assert_eq!(root, leaf);
        assert_eq!(proofs.len(), 1);
        assert!(proofs[0].verify(&leaf, &root));
    }

    #[test]
    fn test_two_leaves() {
        let leaf0 = Hash::from_bytes(b"leaf 0");
        let leaf1 = Hash::from_bytes(b"leaf 1");
        let (root, proofs) = build_merkle_tree_with_proofs(&[leaf0, leaf1]);

        // Root should be hash of the two leaves
        let expected_root = hash_pair(&leaf0, &leaf1);
        assert_eq!(root, expected_root);

        // Both proofs should verify
        assert_eq!(proofs.len(), 2);
        assert!(proofs[0].verify(&leaf0, &root));
        assert!(proofs[1].verify(&leaf1, &root));

        // Proofs should have depth 1
        assert_eq!(proofs[0].depth(), 1);
        assert_eq!(proofs[1].depth(), 1);
    }

    #[test]
    fn test_four_leaves() {
        let leaves: Vec<Hash> = (0..4).map(|i| Hash::from_bytes(&[i])).collect();
        let (root, proofs) = build_merkle_tree_with_proofs(&leaves);

        // All proofs should verify
        for (i, (proof, leaf)) in proofs.iter().zip(leaves.iter()).enumerate() {
            assert!(proof.verify(leaf, &root), "Proof {} failed to verify", i);
            assert_eq!(proof.leaf_index, i as u32);
            assert_eq!(proof.depth(), 2); // log2(4) = 2
        }
    }

    #[test]
    fn test_non_power_of_two_leaves() {
        // 5 leaves -> padded to 8
        let leaves: Vec<Hash> = (0..5).map(|i| Hash::from_bytes(&[i])).collect();
        let (root, proofs) = build_merkle_tree_with_proofs(&leaves);

        // All 5 proofs should verify
        assert_eq!(proofs.len(), 5);
        for (i, (proof, leaf)) in proofs.iter().zip(leaves.iter()).enumerate() {
            assert!(proof.verify(leaf, &root), "Proof {} failed to verify", i);
            assert_eq!(proof.depth(), 3); // ceil(log2(5)) = 3
        }
    }

    #[test]
    fn test_large_tree() {
        // 1000 leaves (realistic batch size)
        let leaves: Vec<Hash> = (0u32..1000)
            .map(|i| Hash::from_bytes(&i.to_le_bytes()))
            .collect();
        let (root, proofs) = build_merkle_tree_with_proofs(&leaves);

        // All proofs should verify
        assert_eq!(proofs.len(), 1000);
        for (i, (proof, leaf)) in proofs.iter().zip(leaves.iter()).enumerate() {
            assert!(proof.verify(leaf, &root), "Proof {} failed to verify", i);
            assert_eq!(proof.depth(), 10); // ceil(log2(1000)) = 10
        }
    }

    #[test]
    fn test_proof_rejects_wrong_leaf() {
        let leaves: Vec<Hash> = (0..4).map(|i| Hash::from_bytes(&[i])).collect();
        let (root, proofs) = build_merkle_tree_with_proofs(&leaves);

        // Proof for leaf 0 should not verify with leaf 1's hash
        assert!(!proofs[0].verify(&leaves[1], &root));
    }

    #[test]
    fn test_proof_rejects_wrong_root() {
        let leaves: Vec<Hash> = (0..4).map(|i| Hash::from_bytes(&[i])).collect();
        let (root, proofs) = build_merkle_tree_with_proofs(&leaves);

        let wrong_root = Hash::from_bytes(b"wrong root");
        assert!(!proofs[0].verify(&leaves[0], &wrong_root));
        assert_eq!(root, root); // Suppress unused warning
    }

    #[test]
    fn test_vote_leaf_hash_deterministic() {
        let tx_hash = Hash::from_bytes(b"tx");
        let state_root = Hash::from_bytes(b"state");

        let hash1 = vote_leaf_hash(&tx_hash, &state_root, 1, true);
        let hash2 = vote_leaf_hash(&tx_hash, &state_root, 1, true);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_vote_leaf_hash_differs_on_fields() {
        let tx_hash = Hash::from_bytes(b"tx");
        let state_root = Hash::from_bytes(b"state");

        let base = vote_leaf_hash(&tx_hash, &state_root, 1, true);

        // Different shard
        let diff_shard = vote_leaf_hash(&tx_hash, &state_root, 2, true);
        assert_ne!(base, diff_shard);

        // Different success
        let diff_success = vote_leaf_hash(&tx_hash, &state_root, 1, false);
        assert_ne!(base, diff_success);

        // Different tx_hash
        let other_tx = Hash::from_bytes(b"other");
        let diff_tx = vote_leaf_hash(&other_tx, &state_root, 1, true);
        assert_ne!(base, diff_tx);
    }

    #[test]
    #[should_panic(expected = "Cannot build Merkle tree with no leaves")]
    fn test_empty_leaves_panics() {
        build_merkle_tree_with_proofs(&[]);
    }
}
