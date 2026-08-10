//! [`TerminalVerdictRoot`] verification.

use thiserror::Error;

use crate::{Hash, TerminalVerdict, TerminalVerdictRoot, Verified, Verify, compute_merkle_root};

/// The root over `verdicts`, in block order. Empty →
/// [`TerminalVerdictRoot::ZERO`]; otherwise the merkle root of each
/// record's own hash.
///
/// A record's leaf covers the shard it answers for, that shard's terminal,
/// and every transaction it names, so two blocks claiming the same root
/// carry the same verdicts.
#[must_use]
pub fn terminal_verdict_root_from_records(verdicts: &[TerminalVerdict]) -> TerminalVerdictRoot {
    if verdicts.is_empty() {
        return TerminalVerdictRoot::ZERO;
    }
    let leaves: Vec<Hash> = verdicts.iter().map(record_leaf).collect();
    TerminalVerdictRoot::from_raw(compute_merkle_root(&leaves))
}

/// One record's leaf: its shard, its terminal, and the transactions it
/// names, in the canonical order the record is built in.
fn record_leaf(verdict: &TerminalVerdict) -> Hash {
    let mut bytes = Vec::with_capacity(16 + verdict.unsettled().len() * 32);
    bytes.extend_from_slice(&verdict.shard().to_le_bytes());
    bytes.extend_from_slice(&verdict.terminal_wt().as_millis().to_le_bytes());
    for tx_hash in verdict.unsettled() {
        bytes.extend_from_slice(tx_hash.as_bytes());
    }
    Hash::from_bytes(&bytes)
}

/// Inputs the [`TerminalVerdictRoot`] verifier reads against.
#[derive(Debug, Clone, Copy)]
pub struct TerminalVerdictRootContext<'a> {
    /// The block's records — each contributes one leaf.
    pub verdicts: &'a [TerminalVerdict],
}

/// Failure modes of [`TerminalVerdictRoot`] verification.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum TerminalVerdictRootVerifyError {
    /// The root computed from the supplied records does not match the
    /// claimed root.
    #[error("computed terminal-verdict root {computed:?} ≠ claimed {expected:?}")]
    Mismatch {
        /// Header's claimed root.
        expected: TerminalVerdictRoot,
        /// Root computed from the supplied records.
        computed: TerminalVerdictRoot,
    },
}

impl Verified<TerminalVerdictRoot> {
    /// Compute the root from `verdicts`. Verified by construction.
    #[must_use]
    pub fn compute(verdicts: &[TerminalVerdict]) -> Self {
        Self::new_unchecked(terminal_verdict_root_from_records(verdicts))
    }
}

impl Verify<&TerminalVerdictRootContext<'_>> for TerminalVerdictRoot {
    type Error = TerminalVerdictRootVerifyError;

    fn verify(
        &self,
        context: &TerminalVerdictRootContext<'_>,
    ) -> Result<Verified<Self>, Self::Error> {
        let computed = terminal_verdict_root_from_records(context.verdicts);
        if computed != *self {
            return Err(TerminalVerdictRootVerifyError::Mismatch {
                expected: *self,
                computed,
            });
        }
        Ok(Verified::new_unchecked(*self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ShardId, TxHash, WeightedTimestamp};

    fn tx(seed: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[seed; 32]))
    }

    fn record(shard: ShardId, seeds: &[u8]) -> TerminalVerdict {
        TerminalVerdict::new(
            shard,
            WeightedTimestamp::from_millis(1_000),
            seeds.iter().copied().map(tx),
        )
    }

    /// A block carrying no records commits nothing, which is a value of
    /// its own rather than the root of an empty tree.
    #[test]
    fn no_records_is_the_zero_root() {
        assert_eq!(
            terminal_verdict_root_from_records(&[]),
            TerminalVerdictRoot::ZERO
        );
    }

    /// The leaf covers what the record claims: change the shard, the
    /// terminal, or the transactions named, and the root moves.
    #[test]
    fn the_root_covers_every_term_of_the_claim() {
        let base = record(ShardId::ROOT, &[1, 2]);
        let root = terminal_verdict_root_from_records(std::slice::from_ref(&base));

        let other_shard = record(ShardId::leaf(1, 0), &[1, 2]);
        assert_ne!(
            root,
            terminal_verdict_root_from_records(std::slice::from_ref(&other_shard)),
        );

        let other_txs = record(ShardId::ROOT, &[1, 3]);
        assert_ne!(
            root,
            terminal_verdict_root_from_records(std::slice::from_ref(&other_txs)),
        );

        let other_terminal = TerminalVerdict::new(
            ShardId::ROOT,
            WeightedTimestamp::from_millis(2_000),
            [tx(1), tx(2)],
        );
        assert_ne!(
            root,
            terminal_verdict_root_from_records(std::slice::from_ref(&other_terminal)),
        );
    }

    /// Verification is the recomputation, so a claimed root the records do
    /// not produce is refused with both figures named.
    #[test]
    fn a_root_the_records_do_not_produce_is_refused() {
        let verdicts = vec![record(ShardId::ROOT, &[1])];
        let claimed = terminal_verdict_root_from_records(&verdicts);
        let context = TerminalVerdictRootContext {
            verdicts: &verdicts,
        };

        assert_eq!(
            claimed.verify(&context).map(Verified::into_inner),
            Ok(claimed)
        );
        assert!(matches!(
            TerminalVerdictRoot::ZERO.verify(&context),
            Err(TerminalVerdictRootVerifyError::Mismatch { .. }),
        ));
    }
}
