//! Binary merkle tree helpers over [`Hash`].
//!
//! Trees pad the leaf list to the next power of two with `Hash::ZERO`,
//! producing a perfect binary tree of depth `ceil(log2(N))`. This makes
//! inclusion proofs fixed-size and eliminates the odd-node-promotion
//! second-preimage attractor.

use crate::Hash;

/// Compute a binary merkle root from a list of hashes.
///
/// Pads the leaf list to the next power of two with `Hash::ZERO` so the
/// tree is always perfect. Returns `Hash::ZERO` for an empty list.
#[must_use]
pub fn compute_merkle_root(hashes: &[Hash]) -> Hash {
    if hashes.is_empty() {
        return Hash::ZERO;
    }
    let padded_len = hashes.len().next_power_of_two();
    let mut level: Vec<Hash> = Vec::with_capacity(padded_len);
    level.extend_from_slice(hashes);
    level.resize(padded_len, Hash::ZERO);

    while level.len() > 1 {
        let mut next_level = Vec::with_capacity(level.len() / 2);
        for i in (0..level.len()).step_by(2) {
            next_level.push(Hash::from_parts(&[
                level[i].as_bytes(),
                level[i + 1].as_bytes(),
            ]));
        }
        level = next_level;
    }
    level[0]
}

/// Compute a binary merkle root AND a proof (siblings + leaf index) for a specific leaf.
///
/// Produces the same root as [`compute_merkle_root`] for the same input.
/// Proofs are fixed-size at `ceil(log2(N))` siblings.
///
/// Returns `(root, siblings, leaf_index)`.
///
/// # Panics
///
/// Panics if `index >= hashes.len()` or `hashes` is empty.
#[must_use]
pub fn compute_merkle_root_with_proof(hashes: &[Hash], index: usize) -> (Hash, Vec<Hash>, u32) {
    assert!(!hashes.is_empty(), "cannot prove in empty tree");
    assert!(index < hashes.len(), "index out of bounds");

    // Pad to next power of 2
    let padded_len = hashes.len().next_power_of_two();
    let mut level: Vec<Hash> = Vec::with_capacity(padded_len);
    level.extend_from_slice(hashes);
    level.resize(padded_len, Hash::ZERO);

    let mut siblings = Vec::new();
    let mut target = index;

    while level.len() > 1 {
        let mut next_level = Vec::with_capacity(level.len() / 2);

        for i in (0..level.len()).step_by(2) {
            let combined = Hash::from_parts(&[level[i].as_bytes(), level[i + 1].as_bytes()]);
            if target == i {
                siblings.push(level[i + 1]);
            } else if target == i + 1 {
                siblings.push(level[i]);
            }
            next_level.push(combined);
        }

        target /= 2;
        level = next_level;
    }

    (level[0], siblings, u32::try_from(index).unwrap_or(u32::MAX))
}

/// Zero-subtree hashes indexed by height: `Z[0]` is `Hash::ZERO` and
/// `Z[h + 1]` is the parent of two `Z[h]`.
///
/// A node whose leaf span lies wholly in the tree's zero padding equals
/// `Z[height]`. Both sides of a range proof derive those nodes from this
/// table instead of transmitting them, which is what makes a proof over a
/// range ending at the leaf count empty.
fn zero_subtrees(depth: usize) -> Vec<Hash> {
    let mut zeros = Vec::with_capacity(depth + 1);
    zeros.push(Hash::ZERO);
    for height in 0..depth {
        let below = zeros[height];
        zeros.push(Hash::from_parts(&[below.as_bytes(), below.as_bytes()]));
    }
    zeros
}

/// Whether the node at `(height, index)` covers only padding — its leaf
/// span starts at or past `leaf_count`.
fn is_padding(height: usize, index: usize, leaf_count: usize) -> bool {
    index
        .checked_mul(1usize << height)
        .is_none_or(|span_start| span_start >= leaf_count)
}

/// Compute a multiproof for the contiguous leaf range `[lo, hi)` — the
/// flanking nodes needed to rebuild the root from those leaves alone.
///
/// At most one left and one right flank per level, so the proof is bounded
/// at `2 * ceil(log2(N))` nodes for the whole range however wide it is.
/// Flanks that lie in the tree's zero padding are omitted: the verifier
/// derives them from [`zero_subtrees`] under the same rule. A range ending
/// at `hashes.len()` therefore needs no right flank at any level, and a
/// full-width range `[0, len)` produces an empty proof.
///
/// Pairs with [`verify_range_inclusion`], which consumes the nodes in the
/// order produced here: left flank before right flank, leaf level upward.
///
/// # Panics
///
/// Panics if `lo > hi` or `hi > hashes.len()`.
#[must_use]
pub fn compute_range_proof(hashes: &[Hash], lo: usize, hi: usize) -> Vec<Hash> {
    assert!(lo <= hi, "range start after end");
    assert!(hi <= hashes.len(), "range end past leaf count");
    if hashes.is_empty() || lo == hi {
        return Vec::new();
    }

    let leaf_count = hashes.len();
    let padded_len = leaf_count.next_power_of_two();
    let depth = padded_len.trailing_zeros() as usize;

    let mut level: Vec<Hash> = Vec::with_capacity(padded_len);
    level.extend_from_slice(hashes);
    level.resize(padded_len, Hash::ZERO);

    let (mut a, mut b) = (lo, hi);
    let mut proof = Vec::new();
    for height in 0..depth {
        if a % 2 == 1 && !is_padding(height, a - 1, leaf_count) {
            proof.push(level[a - 1]);
        }
        if b % 2 == 1 && !is_padding(height, b, leaf_count) {
            proof.push(level[b]);
        }
        a /= 2;
        b = b.div_ceil(2);

        let mut next_level = Vec::with_capacity(level.len() / 2);
        for i in (0..level.len()).step_by(2) {
            next_level.push(Hash::from_parts(&[
                level[i].as_bytes(),
                level[i + 1].as_bytes(),
            ]));
        }
        level = next_level;
    }
    proof
}

/// Verify that `leaves` are exactly the contiguous run at `[lo, lo +
/// leaves.len())` of a `leaf_count`-leaf tree with the given `root`.
///
/// Lifts the covered interval level by level, taking each flank from
/// `proof` or — when the flank lies in the tree's zero padding — deriving
/// it locally. Requires `proof` to be consumed exactly, so a padded or
/// truncated proof is rejected rather than ignored.
///
/// Contiguity and leaf ordering are structural here: the run's position is
/// an input, not a per-element claim, so a reordered or gapped set cannot
/// reach the root. An empty `leaves` verifies against an empty `proof` for
/// any root — a zero-width range makes no claim about the tree.
#[must_use]
pub fn verify_range_inclusion(
    root: Hash,
    leaves: &[Hash],
    lo: usize,
    leaf_count: usize,
    proof: &[Hash],
) -> bool {
    if leaves.is_empty() {
        return proof.is_empty();
    }
    let Some(hi) = lo.checked_add(leaves.len()) else {
        return false;
    };
    if hi > leaf_count {
        return false;
    }

    let padded_len = leaf_count.next_power_of_two();
    let depth = padded_len.trailing_zeros() as usize;
    let zeros = zero_subtrees(depth);

    let mut level = leaves.to_vec();
    let (mut a, mut b) = (lo, hi);
    let mut cursor = 0usize;
    for height in 0..depth {
        let mut extended = Vec::with_capacity(level.len() + 2);
        if a % 2 == 1 {
            let Some(node) = take_flank(height, a - 1, leaf_count, &zeros, proof, &mut cursor)
            else {
                return false;
            };
            extended.push(node);
        }
        extended.extend_from_slice(&level);
        if b % 2 == 1 {
            let Some(node) = take_flank(height, b, leaf_count, &zeros, proof, &mut cursor) else {
                return false;
            };
            extended.push(node);
        }

        // The flanks above make the covered interval even-width at every
        // level, so the pairing consumes `extended` with no remainder.
        let mut next_level = Vec::with_capacity(extended.len() / 2);
        for [left, right] in extended.as_chunks::<2>().0 {
            next_level.push(Hash::from_parts(&[left.as_bytes(), right.as_bytes()]));
        }
        a /= 2;
        b = b.div_ceil(2);
        level = next_level;
    }

    cursor == proof.len() && level.len() == 1 && level[0] == root
}

/// Resolve one flank node: derived when it covers only padding, otherwise
/// taken from the proof. `None` when the proof is exhausted early.
fn take_flank(
    height: usize,
    index: usize,
    leaf_count: usize,
    zeros: &[Hash],
    proof: &[Hash],
    cursor: &mut usize,
) -> Option<Hash> {
    if is_padding(height, index, leaf_count) {
        return zeros.get(height).copied();
    }
    let node = proof.get(*cursor).copied()?;
    *cursor += 1;
    Some(node)
}

/// Verify a merkle inclusion proof against a known root.
///
/// Reconstructs the root from the leaf hash and sibling path, then compares
/// against the expected root.
#[must_use]
pub fn verify_merkle_inclusion(
    root: Hash,
    leaf_hash: Hash,
    siblings: &[Hash],
    leaf_index: u32,
) -> bool {
    let mut current = leaf_hash;
    let mut index = leaf_index as usize;

    for sibling in siblings {
        if index.is_multiple_of(2) {
            current = Hash::from_parts(&[current.as_bytes(), sibling.as_bytes()]);
        } else {
            current = Hash::from_parts(&[sibling.as_bytes(), current.as_bytes()]);
        }
        index /= 2;
    }

    current == root
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_merkle_root_empty() {
        assert_eq!(compute_merkle_root(&[]), Hash::ZERO);
    }

    #[test]
    fn test_merkle_root_single() {
        let h = Hash::from_bytes(b"single");
        assert_eq!(compute_merkle_root(&[h]), h);
    }

    #[test]
    fn test_merkle_root_two() {
        let h0 = Hash::from_bytes(b"left");
        let h1 = Hash::from_bytes(b"right");
        let expected = Hash::from_parts(&[h0.as_bytes(), h1.as_bytes()]);
        assert_eq!(compute_merkle_root(&[h0, h1]), expected);
    }

    #[test]
    fn test_merkle_root_deterministic() {
        let hashes: Vec<Hash> = (0..5).map(|i| Hash::from_bytes(&[i])).collect();
        let root1 = compute_merkle_root(&hashes);
        let root2 = compute_merkle_root(&hashes);
        assert_eq!(root1, root2);
    }

    #[test]
    fn test_merkle_root_order_matters() {
        let h0 = Hash::from_bytes(b"a");
        let h1 = Hash::from_bytes(b"b");
        let root_ab = compute_merkle_root(&[h0, h1]);
        let root_ba = compute_merkle_root(&[h1, h0]);
        assert_ne!(root_ab, root_ba);
    }

    #[test]
    fn test_merkle_root_odd_count_pads_with_zero() {
        // 3 leaves pad to 4: [h0, h1, h2, ZERO]
        let h0 = Hash::from_bytes(b"0");
        let h1 = Hash::from_bytes(b"1");
        let h2 = Hash::from_bytes(b"2");

        let level1_left = Hash::from_parts(&[h0.as_bytes(), h1.as_bytes()]);
        let level1_right = Hash::from_parts(&[h2.as_bytes(), Hash::ZERO.as_bytes()]);
        let expected = Hash::from_parts(&[level1_left.as_bytes(), level1_right.as_bytes()]);

        assert_eq!(compute_merkle_root(&[h0, h1, h2]), expected);
    }

    #[test]
    fn test_inclusion_proof_two_leaves() {
        let h0 = Hash::from_bytes(b"left");
        let h1 = Hash::from_bytes(b"right");
        let hashes = vec![h0, h1];
        let expected_root = compute_merkle_root(&hashes);

        for idx in 0..2 {
            let (root, siblings, leaf_index) = compute_merkle_root_with_proof(&hashes, idx);
            assert_eq!(root, expected_root);
            assert!(verify_merkle_inclusion(
                root,
                hashes[idx],
                &siblings,
                leaf_index
            ));
        }
    }

    #[test]
    fn test_inclusion_proof_single_leaf() {
        let h = Hash::from_bytes(b"only");
        let (root, siblings, leaf_index) = compute_merkle_root_with_proof(&[h], 0);
        assert_eq!(root, h);
        assert!(siblings.is_empty());
        assert!(verify_merkle_inclusion(root, h, &siblings, leaf_index));
    }

    #[test]
    fn test_inclusion_proof_odd_count() {
        let hashes: Vec<Hash> = (0..5u8).map(|i| Hash::from_bytes(&[i])).collect();
        let root = compute_merkle_root(&hashes);

        for idx in 0..5 {
            let (proof_root, siblings, leaf_index) = compute_merkle_root_with_proof(&hashes, idx);
            assert_eq!(proof_root, root);
            assert!(
                verify_merkle_inclusion(root, hashes[idx], &siblings, leaf_index),
                "proof failed for index {idx}"
            );
        }
    }

    #[test]
    fn test_inclusion_proof_large_tree() {
        let hashes: Vec<Hash> = (0..100u8).map(|i| Hash::from_bytes(&[i])).collect();
        let root = compute_merkle_root(&hashes);

        for idx in 0..100 {
            let (proof_root, siblings, leaf_index) = compute_merkle_root_with_proof(&hashes, idx);
            assert_eq!(proof_root, root);
            assert!(
                verify_merkle_inclusion(root, hashes[idx], &siblings, leaf_index),
                "proof failed for index {idx}"
            );
        }
    }

    fn leaves(n: usize) -> Vec<Hash> {
        (0..n).map(|i| Hash::from_bytes(&i.to_le_bytes())).collect()
    }

    /// Every range in every tree shape up to 24 leaves round-trips, and the
    /// proof never exceeds the `2 * depth` bound. Exhaustive rather than
    /// randomized: the flank walk's off-by-ones live at the tree-shape
    /// boundaries (odd leaf counts, ranges touching the padding), and there
    /// are few enough of those to enumerate.
    #[test]
    fn range_proof_round_trips_for_every_small_range() {
        for n in 1..=24usize {
            let hashes = leaves(n);
            let root = compute_merkle_root(&hashes);
            let depth = n.next_power_of_two().trailing_zeros() as usize;
            for lo in 0..=n {
                for hi in lo..=n {
                    let proof = compute_range_proof(&hashes, lo, hi);
                    assert!(
                        proof.len() <= 2 * depth,
                        "n={n} [{lo},{hi}): proof {} exceeds 2*{depth}",
                        proof.len(),
                    );
                    assert!(
                        verify_range_inclusion(root, &hashes[lo..hi], lo, n, &proof),
                        "n={n} [{lo},{hi}) failed to verify",
                    );
                }
            }
        }
    }

    /// A range covering the whole window needs no proof at all: the left
    /// flank never exists (`a` stays 0) and every right flank lies in the
    /// padding. This is the steady-state contribution shape, so it is the
    /// case worth pinning explicitly.
    #[test]
    fn full_width_range_needs_no_proof() {
        for n in 1..=64usize {
            let hashes = leaves(n);
            let root = compute_merkle_root(&hashes);
            let proof = compute_range_proof(&hashes, 0, n);
            assert!(proof.is_empty(), "n={n} produced {} nodes", proof.len());
            assert!(verify_range_inclusion(root, &hashes, 0, n, &[]));
        }
    }

    /// A range ending at the leaf count carries only left flanks, so its
    /// proof is bounded by `depth` rather than `2 * depth` — the shape a
    /// contribution takes whenever the per-chunk cap doesn't bind.
    #[test]
    fn suffix_range_carries_only_left_flanks() {
        for n in 1..=64usize {
            let hashes = leaves(n);
            let depth = n.next_power_of_two().trailing_zeros() as usize;
            for lo in 0..n {
                let proof = compute_range_proof(&hashes, lo, n);
                assert!(
                    proof.len() <= depth,
                    "n={n} [{lo},{n}): proof {} exceeds {depth}",
                    proof.len(),
                );
            }
        }
    }

    /// A single-leaf range agrees with the per-leaf inclusion path: same
    /// root, and both verifiers accept.
    #[test]
    fn single_leaf_range_agrees_with_inclusion_proof() {
        for n in 1..=24usize {
            let hashes = leaves(n);
            let root = compute_merkle_root(&hashes);
            for idx in 0..n {
                let (proof_root, siblings, leaf_index) =
                    compute_merkle_root_with_proof(&hashes, idx);
                assert_eq!(proof_root, root);
                assert!(verify_merkle_inclusion(
                    root,
                    hashes[idx],
                    &siblings,
                    leaf_index
                ));

                let range = compute_range_proof(&hashes, idx, idx + 1);
                assert!(verify_range_inclusion(
                    root,
                    &hashes[idx..=idx],
                    idx,
                    n,
                    &range
                ));
            }
        }
    }

    #[test]
    fn range_proof_rejects_tampering() {
        let hashes = leaves(13);
        let root = compute_merkle_root(&hashes);
        let (lo, hi) = (3usize, 10usize);
        let proof = compute_range_proof(&hashes, lo, hi);
        assert!(verify_range_inclusion(
            root,
            &hashes[lo..hi],
            lo,
            13,
            &proof
        ));

        // A mutated leaf inside the run.
        let mut tampered = hashes[lo..hi].to_vec();
        tampered[2] = Hash::from_bytes(b"tampered");
        assert!(!verify_range_inclusion(root, &tampered, lo, 13, &proof));

        // Reordering inside the run — contiguity is structural, so the
        // recomputed root simply misses.
        let mut swapped = hashes[lo..hi].to_vec();
        swapped.swap(0, 1);
        assert!(!verify_range_inclusion(root, &swapped, lo, 13, &proof));

        // A mutated flank node.
        let mut bad_proof = proof.clone();
        bad_proof[0] = Hash::from_bytes(b"tampered");
        assert!(!verify_range_inclusion(
            root,
            &hashes[lo..hi],
            lo,
            13,
            &bad_proof
        ));

        // The right run at the wrong offset.
        assert!(!verify_range_inclusion(
            root,
            &hashes[lo..hi],
            lo + 1,
            13,
            &proof
        ));

        // A proof with trailing junk is not silently accepted.
        let mut padded = proof.clone();
        padded.push(Hash::from_bytes(b"extra"));
        assert!(!verify_range_inclusion(
            root,
            &hashes[lo..hi],
            lo,
            13,
            &padded
        ));

        // A truncated proof.
        assert!(!verify_range_inclusion(
            root,
            &hashes[lo..hi],
            lo,
            13,
            &proof[..proof.len() - 1]
        ));

        // A run claiming to run past the tree.
        assert!(!verify_range_inclusion(
            root,
            &hashes[lo..hi],
            8,
            13,
            &proof
        ));
    }

    #[test]
    fn empty_range_verifies_against_empty_proof() {
        assert!(verify_range_inclusion(Hash::ZERO, &[], 0, 0, &[]));
        assert!(verify_range_inclusion(
            Hash::from_bytes(b"anything"),
            &[],
            7,
            13,
            &[]
        ));
        // ...but not against a proof that claims work was needed.
        assert!(!verify_range_inclusion(
            Hash::ZERO,
            &[],
            0,
            0,
            &[Hash::ZERO]
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// Round-trip over wider trees than the exhaustive sweep reaches,
        /// with leaf content the prover doesn't control.
        #[test]
        fn range_proof_round_trips(
            content in prop::collection::vec(any::<u64>(), 1..=200usize),
            lo_frac in 0..1000u32,
            hi_frac in 0..1000u32,
        ) {
            let hashes: Vec<Hash> = content
                .iter()
                .map(|v| Hash::from_bytes(&v.to_le_bytes()))
                .collect();
            let n = hashes.len();
            let mut lo = (lo_frac as usize * n) / 1000;
            let mut hi = (hi_frac as usize * n) / 1000;
            if lo > hi {
                std::mem::swap(&mut lo, &mut hi);
            }

            let root = compute_merkle_root(&hashes);
            let depth = n.next_power_of_two().trailing_zeros() as usize;
            let proof = compute_range_proof(&hashes, lo, hi);
            prop_assert!(proof.len() <= 2 * depth);
            prop_assert!(verify_range_inclusion(root, &hashes[lo..hi], lo, n, &proof));
        }

        /// Any leaf mutation inside the run breaks verification.
        #[test]
        fn range_proof_rejects_mutated_leaf(
            content in prop::collection::vec(any::<u64>(), 2..=120usize),
            lo_frac in 0..1000u32,
            hi_frac in 0..1000u32,
            mutate_at in 0..1000u32,
        ) {
            let hashes: Vec<Hash> = content
                .iter()
                .map(|v| Hash::from_bytes(&v.to_le_bytes()))
                .collect();
            let n = hashes.len();
            let mut lo = (lo_frac as usize * n) / 1000;
            let mut hi = (hi_frac as usize * n) / 1000;
            if lo > hi {
                std::mem::swap(&mut lo, &mut hi);
            }
            prop_assume!(hi > lo);

            let root = compute_merkle_root(&hashes);
            let proof = compute_range_proof(&hashes, lo, hi);
            let mut run = hashes[lo..hi].to_vec();
            let at = (mutate_at as usize) % run.len();
            run[at] = Hash::from_bytes(b"mutated");
            prop_assert!(!verify_range_inclusion(root, &run, lo, n, &proof));
        }
    }

    #[test]
    fn test_inclusion_proof_tampered_rejected() {
        let hashes: Vec<Hash> = (0..8u8).map(|i| Hash::from_bytes(&[i])).collect();
        let (root, siblings, leaf_index) = compute_merkle_root_with_proof(&hashes, 3);

        // Wrong leaf hash should fail
        let wrong_leaf = Hash::from_bytes(b"wrong");
        assert!(!verify_merkle_inclusion(
            root, wrong_leaf, &siblings, leaf_index
        ));

        // Wrong root should fail
        let wrong_root = Hash::from_bytes(b"bad_root");
        assert!(!verify_merkle_inclusion(
            wrong_root, hashes[3], &siblings, leaf_index
        ));
    }

    #[test]
    fn test_inclusion_proof_power_of_two() {
        let hashes: Vec<Hash> = (0..8u8).map(|i| Hash::from_bytes(&[i])).collect();
        let root = compute_merkle_root(&hashes);

        for idx in 0..8 {
            let (proof_root, siblings, leaf_index) = compute_merkle_root_with_proof(&hashes, idx);
            assert_eq!(proof_root, root);
            assert_eq!(siblings.len(), 3); // log2(8) = 3
            assert!(verify_merkle_inclusion(
                root,
                hashes[idx],
                &siblings,
                leaf_index
            ));
        }
    }
}
