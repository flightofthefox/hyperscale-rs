//! Merging and filtering [`StateWrites`].

use hyperscale_jmt::NibblePath;
use hyperscale_types::{SettledWrites, StateWrites, StoredReceipt, SubstateKey};

/// Extract and merge the writes from stored receipts, resolving what
/// they moved against the state they land on.
///
/// Canonical projection from receipts to JMT/substate-write input.
/// Failed receipts contribute nothing (`ConsensusReceipt::writes`
/// returns `None`).
///
/// Two rules, because a receipt says two kinds of thing. An exclusive
/// write is an absolute and the last one wins, matching commit order — a
/// rule that is only sound while settlement order agrees with execution
/// order, which is what the pre-vote settlement-order gate enforces. A
/// movement is relative and composes with every other movement on the
/// cell, so the pair needs no ordering at all and cannot overwrite each
/// other whichever way round they arrive.
///
/// `prior` reads the state being settled into. It is consulted only for
/// cells something moved and nothing in this batch wrote, which is the
/// one case where the starting value is not already here.
#[must_use]
pub fn merge_writes_from_receipts(
    receipts: &[StoredReceipt],
    prior: &mut dyn FnMut(SubstateKey) -> Option<Vec<u8>>,
) -> SettledWrites {
    let mut merged = StateWrites::default();
    for receipt in receipts {
        if let Some(writes) = receipt.consensus.writes() {
            for (key, change) in &writes.cells {
                merged.cells.insert(*key, change.clone());
                // An exclusive write supersedes what earlier receipts
                // moved: the cell's value is now stated outright.
                merged.movements.remove(key);
            }
            for (key, movement) in &writes.movements {
                let entry = merged.movements.entry(*key).or_default();
                *entry = entry.then(*movement);
            }
        }
    }
    merged.resolve(prior)
}

/// Merge writes in order; later entries win per cell.
#[must_use]
pub fn merge_state_writes(list: &[&StateWrites]) -> StateWrites {
    let mut merged = StateWrites::default();
    for writes in list {
        for (key, change) in &writes.cells {
            merged.cells.insert(*key, change.clone());
            merged.movements.remove(key);
        }
        for (key, movement) in &writes.movements {
            let entry = merged.movements.entry(*key).or_default();
            *entry = entry.then(*movement);
        }
    }
    merged
}

/// Restrict `writes` to the cells whose JMT leaves fall under `prefix` —
/// the subset of a followed chain's block writes that belongs to a store
/// rooted there.
///
/// A substate key's leading bits are its owner prefix — the identity
/// leaf's routing half — so every cell of one owner shares the prefix
/// decision.
#[must_use]
pub fn filter_writes_to_prefix(writes: &SettledWrites, prefix: &NibblePath) -> SettledWrites {
    SettledWrites::from_absolutes(
        writes
            .cells()
            .iter()
            .filter(|(key, _)| key_under_prefix(&key.to_bytes(), prefix))
            .map(|(key, change)| (*key, change.clone()))
            .collect(),
    )
}

/// Whether `key`'s leading bits equal `prefix` — the subtree-membership
/// test shard prefixes partition the keyspace by.
fn key_under_prefix(key: &[u8; 32], prefix: &NibblePath) -> bool {
    (0..prefix.len()).all(|i| {
        let key_bit = (key[usize::from(i / 8)] >> (7 - (i % 8))) & 1;
        prefix.bits_at(i, 1) == key_bit
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use hyperscale_jmt::NibblePath;
    use hyperscale_types::{Address, LocalKey, SubstateKey};

    use super::*;

    fn writes_for(owner: [u8; 16], value: u8) -> SettledWrites {
        SettledWrites::from_absolutes(BTreeMap::from([(
            SubstateKey {
                owner: Address(owner),
                local: LocalKey([1; 16]),
            },
            Some(vec![value]),
        )]))
    }

    fn relative(owner: [u8; 16], value: u8) -> StateWrites {
        let mut writes = StateWrites::default();
        writes.cells.insert(
            SubstateKey {
                owner: Address(owner),
                local: LocalKey([1; 16]),
            },
            Some(vec![value]),
        );
        writes
    }

    #[test]
    fn later_writes_win_per_cell() {
        let merged = merge_state_writes(&[&relative([1; 16], 1), &relative([1; 16], 2)]);
        assert_eq!(merged.cells.len(), 1);
        assert_eq!(merged.cells.values().next().unwrap(), &Some(vec![2]));
    }

    #[test]
    fn prefix_filter_splits_on_the_leading_bit() {
        let low = writes_for([0x00; 16], 1);
        let high = writes_for([0xFF; 16], 2);
        let merged = SettledWrites::from_absolutes(
            low.cells()
                .iter()
                .chain(high.cells())
                .map(|(key, change)| (*key, change.clone()))
                .collect(),
        );

        let mut left = NibblePath::empty();
        left.push_bits(0, 1);
        let mut right = NibblePath::empty();
        right.push_bits(1, 1);
        assert_eq!(filter_writes_to_prefix(&merged, &left), low);
        assert_eq!(filter_writes_to_prefix(&merged, &right), high);
        assert_eq!(
            filter_writes_to_prefix(&merged, &NibblePath::empty()),
            merged
        );
    }
}
