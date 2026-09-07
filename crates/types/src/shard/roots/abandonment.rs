//! [`AbandonmentRoot`] verification.

use thiserror::Error;

use crate::{AbandonmentRecord, AbandonmentRoot, Hash, Verified, Verify, compute_merkle_root};

/// The root over `records`, in block order. Empty →
/// [`AbandonmentRoot::ZERO`]; otherwise the merkle root of each record's
/// own hash.
///
/// A record's leaf covers the shard it answers for, the kind and moment
/// of its evidence, and every transaction it names with the figures
/// abandoning it takes, so two blocks claiming the same root carry the
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

/// One record's leaf: its shard, its evidence's arm and moment, and each
/// transaction it names with its deadline, its reservation and its
/// charge, in the canonical order the record is built in.
///
/// The arm is a byte of its own because the arms license different
/// aborts, and a leaf naming only the moment would let one pass as the
/// other. The charge is under the leaf because a figure the root does
/// not commit to is a figure two bodies can disagree on under one
/// certificate.
fn record_leaf(record: &AbandonmentRecord) -> Hash {
    let mut bytes = ABANDONMENT_LEAF_TAG.to_vec();
    bytes.reserve(17 + record.unsettled().len() * 112);
    bytes.extend_from_slice(&record.shard().to_le_bytes());
    bytes.push(record.evidence().discriminant());
    bytes.extend_from_slice(&record.evidence().moment().as_millis().to_le_bytes());
    for entry in record.unsettled() {
        bytes.extend_from_slice(entry.tx_hash.as_bytes());
        bytes.extend_from_slice(&entry.deadline.at().as_millis().to_le_bytes());
        bytes.extend_from_slice(&entry.declared_work.to_le_bytes());
        bytes.extend_from_slice(&entry.charge.vault.to_bytes());
        bytes.extend_from_slice(&entry.charge.amount.to_le_bytes());
    }
    Hash::from_bytes(&bytes)
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
        AbortCharge, Address, AddressClass, CounterpartEvidence, Deadline, LocalKey, ShardId,
        SubstateKey, TxHash, UnsettledTx, WeightedTimestamp,
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
    /// everything but the kind of evidence are different claims.
    #[test]
    fn departed_and_refused_at_one_moment_give_different_leaves() {
        let moment = WeightedTimestamp::from_millis(1_000);
        let departed = AbandonmentRecord::departed(ShardId::ROOT, moment, [tx(1)]);
        let refused = AbandonmentRecord::new(
            ShardId::ROOT,
            CounterpartEvidence::Refused { refused_wt: moment },
            [tx(1)],
        );
        let unclaimed = AbandonmentRecord::new(
            ShardId::ROOT,
            CounterpartEvidence::Unclaimed { probed_wt: moment },
            [tx(1)],
        );
        let lapsed = AbandonmentRecord::lapsed(ShardId::ROOT, moment, [tx(1)]);
        let untaken = AbandonmentRecord::untaken(ShardId::ROOT, moment, [tx(1)]);
        assert_ne!(root_of(&departed), root_of(&refused));
        assert_ne!(root_of(&refused), root_of(&unclaimed));
        assert_ne!(root_of(&departed), root_of(&unclaimed));
        assert_ne!(root_of(&unclaimed), root_of(&lapsed));
        assert_ne!(root_of(&unclaimed), root_of(&untaken));
        assert_ne!(root_of(&lapsed), root_of(&untaken));
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
