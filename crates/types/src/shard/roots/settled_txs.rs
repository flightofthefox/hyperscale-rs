//! [`SettledTxsRoot`] computation.
//!
//! The root commits the set of transactions a shard settled within its
//! retention window up to a terminal block. A terminating shard carries it
//! on its boundary header; a surviving counterpart fetches the same set and
//! accepts it only when its recomputed root equals the attested one, so the
//! complete set — and therefore the absence of any transaction from it — is
//! authenticated.
//!
//! [`SettledTxsRoot`]: crate::SettledTxsRoot

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::{
    FinalizedWave, Hash, SettledTxsRoot, ShardId, TxHash, TypedHash, Verifiable,
    compute_merkle_root,
};

/// The cross-shard transactions `shard` settled in `certificates`.
///
/// One entry per transaction of each committed finalization whose local
/// execution certificate is keyed on this shard — its block's own shard,
/// `block.header().shard_id()`. **Single-shard transactions are excluded:**
/// a purely local transaction's outcome never rides another shard's
/// finalization, so the split-boundary fence never queries it and the
/// counterpart sweep already skips it. The settled set therefore commits
/// exactly the transactions a surviving counterpart can ask about, keeping
/// it proportional to cross-shard traffic rather than total throughput.
///
/// The consequence of that exclusion is what a chain observer can conclude:
/// a single-shard transaction that settled and one abandoned at a terminal
/// are indistinguishable here, because neither appears. Abandonment is a
/// record of its own, not the absence of one.
#[must_use]
pub fn local_settled_tx_hashes<'a>(
    certificates: impl IntoIterator<Item = &'a Arc<Verifiable<FinalizedWave>>>,
    shard: ShardId,
) -> Vec<TxHash> {
    certificates
        .into_iter()
        .filter(|fw| {
            let wave_id = fw.wave_id();
            wave_id.shard_id() == shard && !wave_id.is_zero()
        })
        .flat_map(|fw| fw.tx_hashes())
        .collect()
}

/// Domain tag separating a settled-transaction merkle leaf from every other
/// leaf preimage the codebase hashes.
const SETTLED_TX_LEAF_TAG: &[u8] = b"hyperscale.settled_tx_leaf.v1";

/// The merkle leaf for one settled transaction.
fn settled_tx_leaf(tx_hash: &TxHash) -> Hash {
    let mut preimage = SETTLED_TX_LEAF_TAG.to_vec();
    preimage.extend_from_slice(tx_hash.as_raw().as_bytes());
    Hash::from_bytes(&preimage)
}

/// Merkle root over a shard's settled transactions.
///
/// The hashes are taken as a set — sorted and deduplicated — so the root is
/// a pure function of the membership, independent of the order they were
/// discovered in. Empty → [`SettledTxsRoot::ZERO`].
#[must_use]
pub fn settled_txs_root_from_hashes<'a>(
    tx_hashes: impl IntoIterator<Item = &'a TxHash>,
) -> SettledTxsRoot {
    let sorted: BTreeSet<&TxHash> = tx_hashes.into_iter().collect();
    if sorted.is_empty() {
        return SettledTxsRoot::ZERO;
    }
    let leaves: Vec<Hash> = sorted.into_iter().map(settled_tx_leaf).collect();
    SettledTxsRoot::from_raw(compute_merkle_root(&leaves))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(seed: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[seed]))
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(
            settled_txs_root_from_hashes(std::iter::empty()),
            SettledTxsRoot::ZERO
        );
    }

    #[test]
    fn order_independent_and_deduplicated() {
        let a = tx(1);
        let b = tx(2);
        let c = tx(3);
        let forward = settled_txs_root_from_hashes([&a, &b, &c]);
        let shuffled = settled_txs_root_from_hashes([&c, &a, &b]);
        let with_dup = settled_txs_root_from_hashes([&c, &a, &b, &a, &c]);
        assert_eq!(forward, shuffled);
        assert_eq!(forward, with_dup);
    }

    #[test]
    fn membership_changes_the_root() {
        let a = tx(1);
        let b = tx(2);
        let just_a = settled_txs_root_from_hashes([&a]);
        let a_and_b = settled_txs_root_from_hashes([&a, &b]);
        assert_ne!(just_a, a_and_b);
        assert_ne!(just_a, SettledTxsRoot::ZERO);
    }
}
