//! The active shard partition: a binary trie over the `blake3(node_id)`
//! keyspace whose leaves are the live shards.
//!
//! Every node id routes to exactly one shard by walking the trie from the root
//! along its hash bits (most-significant first) until it reaches a leaf —
//! longest-prefix match. A node id's shard prefix is therefore a prefix of its
//! JMT leaf key, so a shard owns a contiguous subtree of the global state tree.

use std::collections::BTreeSet;

use crate::{Address, ShardId};

/// The set of live shards, forming a complete partition of the keyspace: every
/// infinite bit path from the root passes through exactly one leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardTrie {
    leaves: BTreeSet<ShardId>,
}

impl ShardTrie {
    /// The single-shard trie: the root owns the whole keyspace.
    #[must_use]
    pub fn single() -> Self {
        Self {
            leaves: BTreeSet::from([ShardId::ROOT]),
        }
    }

    /// A uniform trie with `2^depth` leaves, all at `depth`.
    #[must_use]
    pub fn uniform(depth: u32) -> Self {
        let count = 1u64 << depth;
        Self {
            leaves: (0..count).map(|p| ShardId::leaf(depth, p)).collect(),
        }
    }

    /// A uniform trie with `count` leaves.
    ///
    /// # Panics
    /// Panics if `count` is not a power of two (a uniform binary trie only has
    /// power-of-two leaf counts).
    #[must_use]
    pub fn uniform_from_count(count: u64) -> Self {
        assert!(count > 0, "shard count must be positive");
        assert!(
            count.is_power_of_two(),
            "uniform shard count must be a power of two, got {count}"
        );
        Self::uniform(count.trailing_zeros())
    }

    /// Build a trie directly from a leaf set. The caller asserts the leaves
    /// form a complete partition.
    #[must_use]
    pub fn from_leaves(leaves: impl IntoIterator<Item = ShardId>) -> Self {
        Self {
            leaves: leaves.into_iter().collect(),
        }
    }

    /// The shard owning `prefix`'s key space: the walk on the prefix's own
    /// bits, no hashing — the prefix is the placement.
    ///
    /// # Panics
    /// As [`Self::shard_for`].
    #[must_use]
    pub fn shard_for_prefix(&self, prefix: Address) -> ShardId {
        let bytes = prefix.to_bytes();
        self.walk(u64::from_be_bytes(
            bytes[..8].try_into().expect("an address is 32 bytes"),
        ))
    }

    /// Whether `shard` owns `prefix`'s key space, asked without a trie.
    ///
    /// [`Self::shard_for_prefix`] descends by the prefix's own bits, so a
    /// leaf at depth `d` owns exactly the prefixes whose first `d` bits
    /// are its path. Answering it this way holds for any shard *while it
    /// was a leaf*, which is what lets a consumer ask about one no live
    /// trie carries — a shard that has since split or merged away, whose
    /// keyspace a walk can only attribute to its successor.
    ///
    /// A caller asking about a departed shard is therefore asking a
    /// question about a span of time as much as about a prefix, and owes
    /// itself the check that the span is one the shard was live for.
    ///
    /// # Panics
    /// As [`Self::shard_for_prefix`].
    #[must_use]
    pub fn shard_owns_prefix(shard: ShardId, prefix: Address) -> bool {
        let depth = shard.depth();
        let bytes = prefix.to_bytes();
        depth == 0
            || (u64::from_be_bytes(bytes[..8].try_into().expect("an address is 32 bytes"))
                >> (64 - depth))
                == shard.path()
    }

    fn walk(&self, bits: u64) -> ShardId {
        let mut id = ShardId::ROOT;
        loop {
            if self.leaves.contains(&id) {
                return id;
            }
            let depth = id.depth();
            assert!(depth < 64, "shard trie is not a complete partition");
            let bit = (bits >> (63 - depth)) & 1;
            let (left, right) = id.children();
            id = if bit == 0 { left } else { right };
        }
    }

    /// The live shards, in heap-index order.
    pub fn leaves(&self) -> impl ExactSizeIterator<Item = ShardId> + '_ {
        self.leaves.iter().copied()
    }

    /// Number of live shards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Whether the trie has no shards (never true for a valid partition).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Whether `shard` is a live leaf.
    #[must_use]
    pub fn contains(&self, shard: ShardId) -> bool {
        self.leaves.contains(&shard)
    }

    /// Split a leaf into its two children, returning them.
    ///
    /// # Panics
    /// Panics if `shard` is not a live leaf.
    pub fn split(&mut self, shard: ShardId) -> (ShardId, ShardId) {
        assert!(self.leaves.remove(&shard), "split of non-leaf {shard:?}");
        let (left, right) = shard.children();
        self.leaves.insert(left);
        self.leaves.insert(right);
        (left, right)
    }

    /// Merge two sibling leaves back into their parent, returning it.
    ///
    /// # Panics
    /// Panics if the two shards are not live sibling leaves.
    pub fn merge(&mut self, left: ShardId, right: ShardId) -> ShardId {
        assert_eq!(
            left.sibling(),
            Some(right),
            "{left:?} and {right:?} are not siblings"
        );
        assert!(self.leaves.remove(&left), "merge of non-leaf {left:?}");
        assert!(self.leaves.remove(&right), "merge of non-leaf {right:?}");
        let parent = left.parent().expect("non-root leaf has a parent");
        self.leaves.insert(parent);
        parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AddressClass;
    use crate::test_utils::test_prefix;

    #[test]
    fn single_routes_everything_to_root() {
        let trie = ShardTrie::single();
        assert_eq!(trie.len(), 1);
        assert_eq!(
            trie.shard_for_prefix(Address::new([1; 31], AddressClass::Component)),
            ShardId::ROOT
        );
        assert_eq!(
            trie.shard_for_prefix(Address::new([0xff; 31], AddressClass::Component)),
            ShardId::ROOT
        );
    }

    #[test]
    fn uniform_partitions_by_top_bits() {
        let trie = ShardTrie::uniform(1);
        assert_eq!(trie.len(), 2);
        // Every owner lands on one of the two depth-1 leaves, by its MSB.
        for seed in 0u8..32 {
            let shard = trie.shard_for_prefix(Address::new([seed; 31], AddressClass::Component));
            assert_eq!(shard.depth(), 1);
            assert!(trie.contains(shard));
        }
    }

    #[test]
    fn uniform_from_count_requires_power_of_two() {
        assert_eq!(ShardTrie::uniform_from_count(4).len(), 4);
        assert_eq!(ShardTrie::uniform_from_count(1), ShardTrie::single());
    }

    #[test]
    fn split_then_merge_round_trips() {
        let mut trie = ShardTrie::single();
        let (l, r) = trie.split(ShardId::ROOT);
        assert_eq!(trie.len(), 2);
        assert!(trie.contains(l) && trie.contains(r));
        let parent = trie.merge(l, r);
        assert_eq!(parent, ShardId::ROOT);
        assert_eq!(trie, ShardTrie::single());
    }

    #[test]
    fn shard_for_prefix_walks_the_prefix_bits_directly() {
        // The routed shard's path equals the top `depth` bits of the
        // prefix itself — no hashing, the prefix is the placement.
        let trie = ShardTrie::uniform(3);
        for seed in [0x00u8, 0x5A, 0xFF] {
            let prefix = test_prefix(seed);
            let shard = trie.shard_for_prefix(prefix);
            let bits = u64::from_be_bytes(prefix.to_bytes()[..8].try_into().unwrap());
            assert_eq!(shard.path(), bits >> (64 - 3));
        }
    }

    /// The trie-free predicate answers exactly what the walk answers, on
    /// every partition and every prefix: the owner owns it, and no other
    /// leaf does. It feeds the split-boundary fence, so agreement with
    /// the walk is not something to take on the derivation's word.
    #[test]
    fn ownership_without_a_trie_agrees_with_the_walk() {
        use proptest::prelude::*;

        proptest!(|(splits in prop::collection::vec(0usize..8, 0..8), raw in any::<[u8; 31]>())| {
            // Splitting an arbitrary leaf each round reaches partitions
            // at mixed depths, not only the uniform ones.
            let mut trie = ShardTrie::single();
            for pick in splits {
                let leaf = trie.leaves().nth(pick % trie.len()).expect("non-empty");
                trie.split(leaf);
            }

            let prefix = Address::new(raw, AddressClass::Component);
            let owner = trie.shard_for_prefix(prefix);
            prop_assert!(ShardTrie::shard_owns_prefix(owner, prefix));
            for leaf in trie.leaves() {
                prop_assert_eq!(
                    ShardTrie::shard_owns_prefix(leaf, prefix),
                    leaf == owner,
                    "only the owning leaf owns the prefix",
                );
            }
        });
    }

    /// A departed shard still owns what it owned: the predicate answers
    /// about the ancestor a split replaced, which is the case the walk
    /// cannot reach because the trie no longer carries it.
    #[test]
    fn ownership_outlives_the_shard_in_the_trie() {
        let mut trie = ShardTrie::single();
        let prefix = Address::new([0x5A; 31], AddressClass::Component);
        let departed = trie.shard_for_prefix(prefix);

        trie.split(departed);
        let successor = trie.shard_for_prefix(prefix);

        assert_ne!(successor, departed, "the split replaced the owner");
        assert!(ShardTrie::shard_owns_prefix(departed, prefix));
        assert!(ShardTrie::shard_owns_prefix(successor, prefix));
    }

    #[test]
    fn shard_for_matches_a_uniform_leaf_prefix() {
        // The routed shard's path equals the owner prefix's top `depth` bits.
        let trie = ShardTrie::uniform(3);
        let owner = test_prefix(0xab);
        let shard = trie.shard_for_prefix(owner);
        let bits = u64::from_be_bytes(owner.to_bytes()[..8].try_into().unwrap());
        assert_eq!(shard.path(), bits >> (64 - 3));
    }

    #[test]
    fn non_uniform_split_to_three_routes_by_longest_prefix() {
        // 1 shard → split the root → 2 → split one child → 3 leaves at mixed
        // depths: a non-power-of-two partition reached by surgical splits.
        let mut trie = ShardTrie::single();
        let (left, right) = trie.split(ShardId::ROOT);
        let (left0, left1) = trie.split(left);
        assert_eq!(trie.len(), 3);
        assert_eq!(right, ShardId::leaf(1, 1));
        assert_eq!(left0, ShardId::leaf(2, 0));
        assert_eq!(left1, ShardId::leaf(2, 1));

        // Every owner resolves to exactly one leaf by its longest matching
        // prefix: top bit 1 → the depth-1 leaf; top bit 0 → the depth-2 leaf
        // chosen by the second bit.
        for seed in 0u8..=255 {
            let owner = test_prefix(seed);
            let bits = u64::from_be_bytes(owner.to_bytes()[..8].try_into().unwrap());
            let expected = if (bits >> 63) & 1 == 1 {
                ShardId::leaf(1, 1)
            } else {
                ShardId::leaf(2, (bits >> 62) & 1)
            };
            assert_eq!(trie.shard_for_prefix(owner), expected, "seed {seed}");
        }

        // Merging the two depth-2 leaves restores the 2-shard partition.
        assert_eq!(trie.merge(left0, left1), ShardId::leaf(1, 0));
        assert_eq!(trie.len(), 2);
    }
}
