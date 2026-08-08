//! Receipt tree leaves and `global_receipt_root` computation/proof helpers.

use crate::{
    ExecutionOutcome, GlobalReceiptRoot, Hash, TxOutcome, compute_merkle_root,
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
/// The attested `work` scalar and any settled fee receipt extend the
/// leaf under their own domain tags. The vote signature covers only the
/// receipt root, and decoding recomputes that root from the outcomes —
/// so a field outside the leaf would be an aggregator's to forge. Work
/// in particular feeds emission weighting and the reshape load
/// predicate, so it must sit under the signed root.
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
    outcome.fee_receipt().map_or(with_work, |fee_receipt| {
        Hash::from_parts(&[with_work.as_bytes(), b"FEE:", fee_receipt.as_bytes()])
    })
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
    use super::*;
    use crate::{GlobalReceiptHash, TxHash};

    fn tx_hash() -> TxHash {
        TxHash::from(Hash::from_bytes(b"leaf-tx"))
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
