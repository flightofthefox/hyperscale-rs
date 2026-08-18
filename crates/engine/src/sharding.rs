//! Shard assignment and write filtering for [`StateWrites`].
//!
//! A substate key carries its owner prefix — the identity leaf's routing
//! half — so shard assignment is a prefix walk over the shard trie and
//! nothing else. Genesis replicates the stdlib package to every shard's
//! substate store for read availability, but each shard's prefix-rooted
//! JMT must contain only its own subtree, so what a shard commits is
//! filtered here first.

use hyperscale_types::{Hash, SettledWrites, ShardId, ShardTrie, StateWrites, WritesRoot};

use crate::executor::protocol_hash;

/// Filter genesis writes to the cells whose owner prefix routes to
/// `local_shard`, for building that shard's prefix-rooted JMT.
///
/// The stdlib package is replicated to every shard's substate store for
/// read availability, but the prefix-rooted JMT must contain only this
/// shard's subtree — so the committed `state_root` is exactly the global
/// tree's node at the shard prefix. Single-shard deployments root at the
/// empty prefix, where every cell routes to the one shard and this is
/// the identity filter.
#[must_use]
pub fn filter_genesis_writes_for_shard(
    merged: &SettledWrites,
    local_shard: ShardId,
    shard_trie: &ShardTrie,
) -> SettledWrites {
    SettledWrites::from_parts(
        merged
            .cells()
            .iter()
            .filter(|(key, _)| shard_trie.shard_for_prefix(key.owner) == local_shard)
            .map(|(key, change)| (*key, change.clone()))
            .collect(),
        merged
            .entries()
            .iter()
            .filter(|(key, _)| shard_trie.shard_for_prefix(key.owner) == local_shard)
            .map(|(key, change)| (*key, change.clone()))
            .collect(),
    )
}

/// Filter [`StateWrites`] for a single shard.
///
/// A substate key carries its owner prefix — the identity leaf's routing
/// half — so shard assignment is a prefix walk and nothing else.
#[must_use]
pub fn filter_writes_for_shard(
    writes: &StateWrites,
    local_shard: ShardId,
    shard_trie: &ShardTrie,
) -> StateWrites {
    let mut filtered = StateWrites::default();
    for (key, change) in &writes.cells {
        if shard_trie.shard_for_prefix(key.owner) == local_shard {
            filtered.cells.insert(*key, change.clone());
        }
    }
    for (key, movement) in &writes.movements {
        if shard_trie.shard_for_prefix(key.owner) == local_shard {
            filtered.movements.insert(*key, *movement);
        }
    }
    for (key, change) in &writes.entries {
        if shard_trie.shard_for_prefix(key.owner) == local_shard {
            filtered.entries.insert(*key, change.clone());
        }
    }
    filtered
}

/// The `writes_root` for a [`GlobalReceipt`](hyperscale_types::GlobalReceipt)
/// over the writes the executing shard attests — the shard-projected
/// delta for a batch with cross-shard members, the full fold for a
/// whole-locality batch.
///
/// [`StateWrites`] encodes in canonical key order by construction, so the
/// root is the hash of the encoding — a pure function of content with no
/// sort step. Empty writes commit to [`WritesRoot::ZERO`].
#[must_use]
pub fn writes_root(writes: &StateWrites) -> WritesRoot {
    if writes.is_empty() {
        return WritesRoot::ZERO;
    }
    WritesRoot::from_raw(Hash::from(writes.root(protocol_hash)))
}

#[cfg(test)]
mod tests {
    use hyperscale_types::test_utils::test_principal;
    use hyperscale_types::{ShardId, ShardTrie, StateWrites, SubstateKey, WritesRoot};
    use hyperscale_vm_types::{LocalKey, PrincipalAddr};

    use super::*;

    fn writes(cells: &[(PrincipalAddr, [u8; 16], Vec<u8>)]) -> StateWrites {
        let mut writes = StateWrites::default();
        for (owner, local, value) in cells {
            writes.cells.insert(
                SubstateKey {
                    owner: owner.address(),
                    local: LocalKey(*local),
                },
                Some(value.clone()),
            );
        }
        writes
    }

    // ── writes_root ──────────────────────────────────────────────────────────

    #[test]
    fn writes_root_empty_is_zero() {
        assert_eq!(writes_root(&StateWrites::default()), WritesRoot::ZERO);
    }

    #[test]
    fn writes_root_distinguishes_inputs() {
        let a = writes(&[(test_principal(1), [0; 16], vec![1])]);
        let b = writes(&[(test_principal(2), [0; 16], vec![1])]);
        assert_ne!(writes_root(&a), writes_root(&b));
        assert_eq!(writes_root(&a), writes_root(&a.clone()));
    }

    // ── filter_writes_for_shard ──────────────────────────────────────────────

    #[test]
    fn filter_for_shard_keeps_only_this_shard_prefixes() {
        let trie = ShardTrie::uniform_from_count(2);
        let left = test_principal(0x00);
        let right = test_principal(0xFF);
        assert_ne!(trie.shard_for_prefix(left), trie.shard_for_prefix(right));
        let all = writes(&[(left, [1; 16], vec![1]), (right, [1; 16], vec![2])]);

        let filtered = filter_writes_for_shard(&all, trie.shard_for_prefix(left), &trie);
        assert_eq!(filtered.cells.len(), 1);
        assert_eq!(filtered.cells.keys().next().unwrap().owner, left);
    }

    #[test]
    fn filter_for_single_shard_is_the_identity() {
        let all = writes(&[
            (test_principal(1), [1; 16], vec![1]),
            (test_principal(9), [2; 16], vec![2]),
        ]);
        let filtered =
            filter_writes_for_shard(&all, ShardId::ROOT, &ShardTrie::uniform_from_count(1));
        assert_eq!(filtered, all);
    }
}
