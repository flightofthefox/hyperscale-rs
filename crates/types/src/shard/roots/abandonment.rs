//! [`AbandonmentRoot`] verification.

use hyperscale_hbor::to_vec as hbor_to_vec;
use thiserror::Error;

use crate::{AbandonmentRecord, AbandonmentRoot, Hash, Verified, Verify, compute_merkle_root};

/// The root over `records`, in block order. Empty →
/// [`AbandonmentRoot::ZERO`]; otherwise the merkle root of each record's
/// own hash.
///
/// A record's leaf covers its whole encoding — the shard it answers for,
/// its evidence, and every transaction it names with the figures
/// abandoning it takes — so two blocks claiming the same root carry the
/// same records.
#[must_use]
pub fn abandonment_root_from_records(records: &[AbandonmentRecord]) -> AbandonmentRoot {
    if records.is_empty() {
        return AbandonmentRoot::ZERO;
    }
    let leaves: Vec<Hash> = records.iter().map(record_leaf).collect();
    AbandonmentRoot::from_raw(compute_merkle_root(&leaves))
}

/// Domain tag separating an abandonment record's merkle leaf from every
/// other leaf preimage the codebase hashes.
const ABANDONMENT_LEAF_TAG: &[u8] = b"hyperscale.abandonment_leaf.v1";

/// One record's leaf: the tag and its canonical encoding, which covers
/// the shard, the evidence whole, and every figure of every name — a
/// figure the root does not commit to is a figure two bodies can
/// disagree on under one certificate.
///
/// # Panics
///
/// If the record does not encode, which a value built through
/// [`AbandonmentRecord::new`] under its caps always does.
fn record_leaf(record: &AbandonmentRecord) -> Hash {
    let bytes = hbor_to_vec(record).expect("an abandonment record encodes");
    Hash::from_parts(&[ABANDONMENT_LEAF_TAG, &bytes])
}

/// Inputs the [`AbandonmentRoot`] verifier reads against.
#[derive(Debug, Clone, Copy)]
pub struct AbandonmentRootContext<'a> {
    /// The block's records — each contributes one leaf.
    pub records: &'a [AbandonmentRecord],
}

/// Failure modes of [`AbandonmentRoot`] verification.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum AbandonmentRootVerifyError {
    /// The root computed from the supplied records does not match the
    /// claimed root.
    #[error("computed abandonment root {computed:?} ≠ claimed {expected:?}")]
    Mismatch {
        /// Header's claimed root.
        expected: AbandonmentRoot,
        /// Root computed from the supplied records.
        computed: AbandonmentRoot,
    },
}

impl Verified<AbandonmentRoot> {
    /// Compute the root from `records`. Verified by construction.
    #[must_use]
    pub fn compute(records: &[AbandonmentRecord]) -> Self {
        Self::new_unchecked(abandonment_root_from_records(records))
    }
}

impl Verify<&AbandonmentRootContext<'_>> for AbandonmentRoot {
    type Error = AbandonmentRootVerifyError;

    fn verify(&self, context: &AbandonmentRootContext<'_>) -> Result<Verified<Self>, Self::Error> {
        let computed = abandonment_root_from_records(context.records);
        if computed != *self {
            return Err(AbandonmentRootVerifyError::Mismatch {
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
    use crate::{
        AbortCharge, Address, AddressClass, Deadline, Hash, Heard, LocalKey, Probed, Question,
        ShardId, SubstateKey, TransactionDecision, TxHash, UnsettledTx, WeightedTimestamp, Word,
    };

    fn tx(seed: u8) -> UnsettledTx {
        UnsettledTx {
            tx_hash: TxHash::from(Hash::from_bytes(&[seed; 32])),
            deadline: Deadline::of(WeightedTimestamp::from_millis(500)),
            declared_work: 7,
            charge: AbortCharge {
                vault: SubstateKey {
                    owner: Address::new([seed; 31], AddressClass::Component),
                    local: LocalKey([seed; 16]),
                },
                amount: 11,
            },
        }
    }

    fn record(shard: ShardId, seeds: &[u8]) -> AbandonmentRecord {
        AbandonmentRecord::departed(
            shard,
            WeightedTimestamp::from_millis(1_000),
            seeds.iter().copied().map(tx),
        )
    }

    fn root_of(record: &AbandonmentRecord) -> AbandonmentRoot {
        abandonment_root_from_records(std::slice::from_ref(record))
    }

    /// A block carrying no records commits nothing, which is a value of
    /// its own rather than the root of an empty tree.
    #[test]
    fn no_records_is_the_zero_root() {
        assert_eq!(abandonment_root_from_records(&[]), AbandonmentRoot::ZERO);
    }

    /// The leaf covers what the record claims: change the shard, the
    /// moment, or the transactions named, and the root moves.
    #[test]
    fn the_root_covers_every_term_of_the_claim() {
        let base = record(ShardId::ROOT, &[1, 2]);
        let root = root_of(&base);

        assert_ne!(root, root_of(&record(ShardId::leaf(1, 0), &[1, 2])));
        assert_ne!(root, root_of(&record(ShardId::ROOT, &[1, 3])));
        assert_ne!(
            root,
            root_of(&AbandonmentRecord::departed(
                ShardId::ROOT,
                WeightedTimestamp::from_millis(2_000),
                [tx(1), tx(2)],
            )),
        );

        // The figures a name carries are part of the claim: the record
        // licenses an abort that returns exactly this much, at exactly
        // this deadline, burning exactly this out of exactly this vault,
        // so a block restating any of them differently is a different
        // block.
        let restated = |entry: UnsettledTx| {
            root_of(&AbandonmentRecord::departed(
                ShardId::ROOT,
                WeightedTimestamp::from_millis(1_000),
                [entry, tx(2)],
            ))
        };
        assert_ne!(
            root,
            restated(UnsettledTx {
                declared_work: 8,
                ..tx(1)
            })
        );
        assert_ne!(
            root,
            restated(UnsettledTx {
                deadline: Deadline::of(WeightedTimestamp::from_millis(501)),
                ..tx(1)
            })
        );
        assert_ne!(
            root,
            restated(UnsettledTx {
                charge: AbortCharge {
                    amount: 12,
                    ..tx(1).charge
                },
                ..tx(1)
            })
        );
        assert_ne!(
            root,
            restated(UnsettledTx {
                charge: AbortCharge {
                    vault: tx(9).charge.vault,
                    ..tx(1).charge
                },
                ..tx(1)
            })
        );
    }

    /// The arms license different aborts, so two records agreeing on
    /// everything but the evidence are different claims.
    #[test]
    fn every_arm_at_one_moment_gives_its_own_leaf() {
        let moment = WeightedTimestamp::from_millis(1_000);
        let heard = |question, word| {
            AbandonmentRecord::heard(
                ShardId::ROOT,
                Heard {
                    question,
                    word,
                    at: moment,
                },
                [tx(1)],
            )
        };
        let digest = Hash::from_bytes(b"digest");
        let records = [
            AbandonmentRecord::departed(ShardId::ROOT, moment, [tx(1)]),
            heard(
                Question::Verdict,
                Word::Refused {
                    decision: TransactionDecision::Reject,
                    digest,
                },
            ),
            heard(
                Question::Verdict,
                Word::Refused {
                    decision: TransactionDecision::Aborted,
                    digest,
                },
            ),
            heard(Question::Verdict, Word::Accepted { digest }),
            heard(Question::Cell(Probed::Core), Word::Absent),
            heard(Question::Cell(Probed::Delivery), Word::Absent),
            heard(Question::Cell(Probed::Claim), Word::Absent),
        ];
        let mut roots: Vec<AbandonmentRoot> = records.iter().map(root_of).collect();
        roots.sort_unstable();
        roots.dedup();
        assert_eq!(roots.len(), records.len());
    }

    /// Verification is the recomputation, so a claimed root the records do
    /// not produce is refused with both figures named.
    #[test]
    fn a_root_the_records_do_not_produce_is_refused() {
        let records = vec![record(ShardId::ROOT, &[1])];
        let claimed = abandonment_root_from_records(&records);
        let context = AbandonmentRootContext { records: &records };

        assert_eq!(
            claimed.verify(&context).map(Verified::into_inner),
            Ok(claimed)
        );
        assert!(matches!(
            AbandonmentRoot::ZERO.verify(&context),
            Err(AbandonmentRootVerifyError::Mismatch { .. }),
        ));
    }
}
