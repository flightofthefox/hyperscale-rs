//! What a block commits about a counterpart's chain.
//!
//! A leg's ledger asks a silent counterpart two kinds of question. What
//! does your state hold — a core's committed cell, a delivery's claim —
//! which a proof against one of your commit-proven headers answers. And
//! what did you decide — did your core refuse this transaction — which
//! your certificate answers.
//!
//! Both are facts about somebody else's chain that this chain's records
//! are composed from, so both have to be the chain's own answer rather
//! than a replica's, or two validators compose different records and the
//! vote splits. The proposer puts what its own fetches and its own
//! broadcasts answered into the block, every voter checks the same
//! claims, and every replica folds the same answers at the same height.
//!
//! # Two kinds, verified differently, and deliberately not merged
//!
//! A proof is bound to an anchor and checked against the root of the
//! header that anchor names; a verdict is checked against the committee
//! that signed it. A proof stops being servable once its anchor falls
//! past the retention floor; a certificate stays re-fetchable from anyone
//! holding it. A proof is pulled per voter; a verdict is already pushed
//! to every shard the transaction awaits. And a verdict is
//! self-verifying where a proof needs a root the replica independently
//! holds.
//!
//! So the bytes differ: the section carries a proof whole, and carries a
//! verdict as a *commitment* to one — the certificate's attested digest
//! and the outcome it names. That is what the fold needs, and it is what
//! keeps the largest object on the consensus path out of every block on
//! the refusal path.

use hyperscale_hbor::Hbor;

use crate::{Hash, ShardId, StateProofBundle, TransactionDecision, TxHash, WeightedTimestamp};

/// A counterpart's verdict on a transaction a leg here issued for, as
/// the chain commits to it.
///
/// Never the certificate itself. What a voter needs is the fact — this
/// shard decided this way at this anchor — and a name for the bytes that
/// say so, which it checks against the certificate it already holds. One
/// that holds none defers, exactly as it defers on an anchor it has not
/// proven.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hbor)]
pub struct VerdictClaim {
    /// The counterpart whose certificate attests it.
    pub shard: ShardId,
    /// The transaction it decided.
    pub tx_hash: TxHash,
    /// The certificate's vote anchor: the clock a record restating this
    /// verdict is held to, and what the mirror is keyed on.
    pub anchor_ts: WeightedTimestamp,
    /// What the certificate decided. Only a refusal licenses anything
    /// here — an acceptance settles the transaction and needs no record
    /// — so this is `Reject` or `Aborted`, and the fold refuses anything
    /// else rather than folding a verdict that licenses nothing.
    pub decision: TransactionDecision,
    /// [`ExecutionCertificate::attested_digest`]: the signed identity of
    /// the certificate, which is copy-invariant where its wire hash is
    /// not.
    ///
    /// [`ExecutionCertificate::attested_digest`]: crate::ExecutionCertificate::attested_digest
    pub digest: Hash,
}

impl VerdictClaim {
    /// Whether this is a verdict that licenses a record at all.
    ///
    /// An acceptance is a settlement, and a settlement needs no record —
    /// so a claim naming one commits nothing and is refused where it is
    /// offered rather than folded into an answer nothing reads.
    #[must_use]
    pub const fn refuses(&self) -> bool {
        matches!(
            self.decision,
            TransactionDecision::Reject | TransactionDecision::Aborted
        )
    }
}

/// One claim a block makes about a counterpart's chain.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hbor)]
pub enum CounterpartClaim {
    /// Cells at a commit-proven anchor, with the multiproof over them.
    Cells(StateProofBundle),
    /// A counterpart's decision, named by the certificate that attests it.
    Verdict(VerdictClaim),
}

impl CounterpartClaim {
    /// Whether the claim is in the one form it may take.
    ///
    /// A verdict that licenses nothing is as malformed as a bundle
    /// naming no key: both would cost a block a leaf for an answer
    /// nothing reads.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        match self {
            Self::Cells(bundle) => bundle.is_well_formed(),
            Self::Verdict(verdict) => verdict.refuses(),
        }
    }

    /// The bundle, where the claim is one.
    #[must_use]
    pub const fn cells(&self) -> Option<&StateProofBundle> {
        match self {
            Self::Cells(bundle) => Some(bundle),
            Self::Verdict(_) => None,
        }
    }

    /// The verdict, where the claim is one.
    #[must_use]
    pub const fn verdict(&self) -> Option<&VerdictClaim> {
        match self {
            Self::Verdict(verdict) => Some(verdict),
            Self::Cells(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockHeight, MerkleInclusionProof, StateAnchor, StateRoot};

    fn verdict(decision: TransactionDecision) -> VerdictClaim {
        VerdictClaim {
            shard: ShardId::leaf(1, 1),
            tx_hash: TxHash::from(Hash::from_bytes(b"tx")),
            anchor_ts: WeightedTimestamp::from_millis(9),
            decision,
            digest: Hash::from_bytes(b"digest"),
        }
    }

    /// Only a refusal licenses a record, so only a refusal is a claim
    /// worth a block's leaf.
    #[test]
    fn a_verdict_claim_carries_a_refusal_or_nothing() {
        assert!(CounterpartClaim::Verdict(verdict(TransactionDecision::Reject)).is_well_formed());
        assert!(CounterpartClaim::Verdict(verdict(TransactionDecision::Aborted)).is_well_formed());
        assert!(
            !CounterpartClaim::Verdict(verdict(TransactionDecision::Accept)).is_well_formed(),
            "an acceptance settles the transaction and licenses no record",
        );
    }

    /// The two arms answer for each other's question with `None`, so a
    /// consumer of one never reads the other by accident.
    #[test]
    fn each_arm_answers_only_its_own_question() {
        let bundle = StateProofBundle::new(
            StateAnchor {
                shard: ShardId::ROOT,
                height: BlockHeight::new(1),
                state_root: StateRoot::ZERO,
            },
            WeightedTimestamp::ZERO,
            [],
            MerkleInclusionProof::dummy(),
        );
        let cells = CounterpartClaim::Cells(bundle);
        assert!(cells.cells().is_some() && cells.verdict().is_none());
        let spoken = CounterpartClaim::Verdict(verdict(TransactionDecision::Reject));
        assert!(spoken.verdict().is_some() && spoken.cells().is_none());
    }
}
