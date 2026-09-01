//! Receipt tree leaves and `global_receipt_root` computation/proof helpers.

use crate::{
    ExecutionOutcome, GlobalReceiptRoot, Hash, ShardId, TxOutcome, compute_merkle_root,
    compute_merkle_root_with_proof,
};

/// Compute the leaf hash for a transaction outcome in the receipt tree.
///
/// - `Succeeded`: `H(tx_hash || receipt_hash)`
/// - `Failed`:    `H(tx_hash || b"FAILED:")` (domain-tagged; canonical hash is implicit)
/// - `Aborted`:   `H(tx_hash || b"ABORTED:")`
///
/// The domain tags ensure the three variants can never collide.
///
/// The attested `work` scalar, any settled fee receipt, and the three
/// lists — the shards the transaction's settlement waits on, what the
/// execution escrowed out, and where those crossings land — extend the
/// leaf under their own domain tags. The vote signature covers only the
/// receipt root, and decoding recomputes that root from the outcomes —
/// so a field outside the leaf would be an aggregator's to forge. Work
/// in particular feeds emission weighting and the reshape load
/// predicate, the awaited shards decide how many certificates it takes
/// to settle the transaction at all, and an escrowed entry is what a
/// consuming shard claims, so all of it must sit under the signed root.
///
/// The lists are led by their three counts. Each entry is fixed-width,
/// which makes one list admit one reading; three lists in a row do not,
/// because a 104-byte escrowed entry carries 32 bytes of manifest-chosen
/// resource address and can spell whatever separates them. The counts
/// fix the split on their own, so the reading never rests on a tag
/// being unspellable.
#[must_use]
pub fn tx_outcome_leaf(outcome: &TxOutcome) -> Hash {
    let base = match outcome.outcome() {
        ExecutionOutcome::Succeeded { receipt_hash } => {
            Hash::from_parts(&[outcome.tx_hash().as_bytes(), receipt_hash.as_bytes()])
        }
        ExecutionOutcome::Failed => Hash::from_parts(&[outcome.tx_hash().as_bytes(), b"FAILED:"]),
        ExecutionOutcome::Aborted => Hash::from_parts(&[outcome.tx_hash().as_bytes(), b"ABORTED:"]),
    };
    let with_work = Hash::from_parts(&[
        base.as_bytes(),
        b"WORK:",
        &outcome.attested_work().to_le_bytes(),
        b"RESERVED:",
        &outcome.declared_work().to_le_bytes(),
    ]);
    let with_fee = outcome.fee_receipt().map_or(with_work, |fee_receipt| {
        Hash::from_parts(&[with_work.as_bytes(), b"FEE:", fee_receipt.as_bytes()])
    });
    let count = |len: usize| u32::try_from(len).unwrap_or(u32::MAX).to_le_bytes();
    let counts: Vec<u8> = [
        count(outcome.counterparts().len()),
        count(outcome.escrowed().len()),
        count(outcome.crossing_targets().len()),
    ]
    .concat();
    let awaited = shard_bytes(outcome.counterparts());
    // Fixed-width per entry, like the shard lists either side of it.
    let escrowed: Vec<u8> = outcome
        .escrowed()
        .iter()
        .flat_map(|entry| {
            let mut bytes = [0u8; 104];
            bytes[..4].copy_from_slice(&entry.node.to_le_bytes());
            bytes[4..8].copy_from_slice(&entry.output.to_le_bytes());
            bytes[8..40].copy_from_slice(&entry.resource.to_bytes());
            bytes[40..56].copy_from_slice(&entry.amount.to_le_bytes());
            bytes[56..].copy_from_slice(&entry.record.to_bytes());
            bytes
        })
        .collect();
    let targets = shard_bytes(outcome.crossing_targets());
    Hash::from_parts(&[
        with_fee.as_bytes(),
        b"LISTS:",
        &counts,
        b"AWAITS:",
        &awaited,
        b"ESCROWED:",
        &escrowed,
        b"CROSSING:",
        &targets,
    ])
}

/// A shard list as fixed-width entries, so the concatenation admits one
/// reading — a variable encoding would let two different sets agree on
/// their bytes.
fn shard_bytes(shards: &[ShardId]) -> Vec<u8> {
    shards
        .iter()
        .flat_map(|shard| {
            let mut bytes = [0u8; 12];
            bytes[..4].copy_from_slice(&shard.depth().to_le_bytes());
            bytes[4..].copy_from_slice(&shard.path().to_le_bytes());
            bytes
        })
        .collect()
}

/// Compute the receipt root from a list of transaction outcomes.
///
/// Uses padded merkle tree (power-of-2 padding with `Hash::ZERO`) so that
/// merkle inclusion proofs have a fixed `ceil(log2(N))` siblings.
///
/// Outcomes must be in tick order (= block order within the tick).
pub fn compute_global_receipt_root(outcomes: &[TxOutcome]) -> GlobalReceiptRoot {
    let leaves: Vec<Hash> = outcomes.iter().map(tx_outcome_leaf).collect();
    GlobalReceiptRoot::from_raw(compute_merkle_root(&leaves))
}

/// Compute receipt root and a merkle inclusion proof for a specific tx.
///
/// Returns `(root, proof_siblings, leaf_index, leaf_hash)`.
///
/// # Panics
///
/// Panics if `tx_index >= outcomes.len()` or `outcomes` is empty.
pub fn compute_global_receipt_root_with_proof(
    outcomes: &[TxOutcome],
    tx_index: usize,
) -> (Hash, Vec<Hash>, u32, Hash) {
    let leaves: Vec<Hash> = outcomes.iter().map(tx_outcome_leaf).collect();

    let leaf_hash = leaves[tx_index];
    let (root, siblings, leaf_index) = compute_merkle_root_with_proof(&leaves, tx_index);
    (root, siblings, leaf_index, leaf_hash)
}

#[cfg(test)]
mod reservation_tests {
    use super::{compute_global_receipt_root, tx_outcome_leaf};
    use crate::{ExecutionOutcome, GlobalReceiptHash, Hash, TxHash, TxOutcome};

    fn outcome(reserved: u64) -> TxOutcome {
        TxOutcome::attesting(
            TxHash::from(Hash::from_bytes(b"tx")),
            ExecutionOutcome::Succeeded {
                receipt_hash: GlobalReceiptHash::ZERO,
            },
            7,
        )
        .reserving(reserved)
    }

    /// What a transaction reserved is signed content like what it cost.
    /// An aggregator that could restate it could hand a block a release
    /// larger than the reservation it settles, and the running drain
    /// total would fall below what is actually in flight.
    #[test]
    fn the_reservation_is_covered_by_the_outcome_leaf() {
        assert_ne!(
            tx_outcome_leaf(&outcome(100)),
            tx_outcome_leaf(&outcome(101))
        );
        assert_ne!(
            compute_global_receipt_root(&[outcome(100)]),
            compute_global_receipt_root(&[outcome(101)])
        );
    }

    /// The two work quantities are separate axes: one shard's share of
    /// what a transaction cost, and what every shard agrees it reserved.
    /// Folding them into one leaf position would let a difference in
    /// either hide a difference in the other.
    #[test]
    fn cost_and_reservation_move_the_leaf_independently() {
        let costed = TxOutcome::attesting(
            TxHash::from(Hash::from_bytes(b"tx")),
            ExecutionOutcome::Succeeded {
                receipt_hash: GlobalReceiptHash::ZERO,
            },
            8,
        )
        .reserving(100);
        assert_ne!(tx_outcome_leaf(&costed), tx_outcome_leaf(&outcome(100)));
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_vm_types::{Address, AddressClass, LocalKey, ResourceAddr, SubstateKey};

    use super::*;
    use crate::{EscrowedValue, GlobalReceiptHash, TxHash};

    fn tx_hash() -> TxHash {
        TxHash::from(Hash::from_bytes(b"leaf-tx"))
    }

    fn escrowed(node: u32) -> EscrowedValue {
        EscrowedValue {
            node,
            output: 0,
            resource: ResourceAddr::new([0xE1; 31]),
            amount: 5,
            record: SubstateKey {
                owner: Address::new([0xC1; 31], AddressClass::Component),
                local: LocalKey([u8::try_from(node).expect("a test node fits a byte"); 16]),
            },
        }
    }

    fn base() -> TxOutcome {
        TxOutcome::attesting(tx_hash(), ExecutionOutcome::Aborted, 7)
    }

    /// The list region is one byte string whatever the split between
    /// its three lists, so the counts have to be what fixes the reading:
    /// three escrowed entries and twenty-six shards are the same 312
    /// bytes of region, and the leaves differ.
    #[test]
    fn two_list_splits_of_equal_length_give_different_leaves() {
        let escrowing = base().escrowing((0..3).map(escrowed));
        let crossing = base().crossing_to((0..26).map(|path| ShardId::leaf(5, path)));
        assert_eq!(
            escrowing.escrowed().len() * 104,
            crossing.crossing_targets().len() * 12,
            "the two regions have to be the same length, or this proves nothing"
        );
        assert_ne!(tx_outcome_leaf(&escrowing), tx_outcome_leaf(&crossing));

        // And the same shards awaited rather than crossed to is a third
        // reading of the same bytes.
        let awaiting = base().awaiting((0..26).map(|path| ShardId::leaf(5, path)));
        assert_ne!(tx_outcome_leaf(&awaiting), tx_outcome_leaf(&crossing));
    }

    /// Every field of an escrowed entry is under the leaf: an aggregator
    /// restating what left, or how much, fails the root recompute.
    #[test]
    fn leaf_covers_what_was_escrowed() {
        let one = base().escrowing([escrowed(1)]);
        let more = base().escrowing([EscrowedValue {
            amount: 6,
            ..escrowed(1)
        }]);
        let elsewhere = base().escrowing([EscrowedValue {
            resource: ResourceAddr::new([0xE2; 31]),
            ..escrowed(1)
        }]);
        let moved = base().escrowing([EscrowedValue {
            record: escrowed(2).record,
            ..escrowed(1)
        }]);
        assert_ne!(tx_outcome_leaf(&one), tx_outcome_leaf(&more));
        assert_ne!(tx_outcome_leaf(&one), tx_outcome_leaf(&elsewhere));
        assert_ne!(tx_outcome_leaf(&one), tx_outcome_leaf(&moved));
        assert_ne!(tx_outcome_leaf(&one), tx_outcome_leaf(&base()));
    }

    /// One form: the builder sorts on the whole entry and keeps one per
    /// edge, so two callers offering the same set in different orders
    /// build the same outcome.
    #[test]
    fn escrowed_entries_take_one_form() {
        let forward = base().escrowing([escrowed(1), escrowed(2)]);
        let backward = base().escrowing([escrowed(2), escrowed(1), escrowed(2)]);
        assert_eq!(forward, backward);
        assert_eq!(forward.escrowed().len(), 2);
    }

    /// `work` is folded into the leaf: outcomes identical but for their
    /// attested work hash to different leaves, so a forged work fails
    /// the receipt-root recompute every EC decode runs.
    #[test]
    fn leaf_covers_attested_work() {
        let outcome = |work| TxOutcome::attesting(tx_hash(), ExecutionOutcome::Aborted, work);
        assert_ne!(
            tx_outcome_leaf(&outcome(7)),
            tx_outcome_leaf(&outcome(8)),
            "work must be covered by the leaf"
        );
    }

    /// The fee-receipt extension composes with the work fold — a forged
    /// work is still caught on outcomes that settle a fee receipt.
    #[test]
    fn leaf_covers_attested_work_with_fee_receipt() {
        let fee = GlobalReceiptHash::from_raw(Hash::from_bytes(b"fee"));
        let outcome = |work| TxOutcome::with_fee(tx_hash(), ExecutionOutcome::Failed, fee, work);
        assert_ne!(
            tx_outcome_leaf(&outcome(7)),
            tx_outcome_leaf(&outcome(8)),
            "work must be covered by the leaf on fee-settling outcomes"
        );
    }
}
