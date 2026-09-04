//! A proof of a counterpart's cells, carried by a block.
//!
//! A leg's ledger asks a silent counterpart whether it took what the
//! leg issued: a core's committed cell, a delivery's claim, a core
//! consumer's claim. The answer is a state proof against one of the
//! counterpart's commit-proven headers, and what is done with it — a
//! record offered, a reclaim or a retirement composed — is composed
//! from the ledger, so the answer has to be the chain's rather than a
//! replica's or the vote splits. The proposer puts the proofs its own
//! fetches answered into the block, every voter checks the same bytes
//! against the same commit-proven header, and every replica folds the
//! same answers at the same height.

use hyperscale_hbor::Hbor;

use crate::{
    Inclusion, MAX_COMMITTED_TX_QUERY, MerkleInclusionProof, StateAnchor, StateProofError,
    SubstateKey, WeightedTimestamp,
};

/// One fetch's answer: a multiproof over `keys` against `anchor`.
///
/// The anchor names the shard, the height and the root the proof
/// reconstructs; `anchor_ts` is the anchor's block clock, which a voter
/// holds to the commit-proven header it has for the anchor, so a window
/// read off it at commit is read off chain content and never the
/// proposer's word.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hbor)]
pub struct StateProofBundle {
    /// The commit-proven state the proof reconstructs.
    pub anchor: StateAnchor,
    /// The anchor's parent-QC weighted timestamp: the clock every
    /// window the answer is held to is read against.
    pub anchor_ts: WeightedTimestamp,
    /// The keys the proof answers for, sorted and without repeats.
    #[hbor(max = MAX_COMMITTED_TX_QUERY)]
    pub keys: Vec<SubstateKey>,
    /// The multiproof over every key against the anchor's root.
    pub proof: MerkleInclusionProof,
}

impl StateProofBundle {
    /// A bundle over `keys`, in the one order it may carry them.
    #[must_use]
    pub fn new(
        anchor: StateAnchor,
        anchor_ts: WeightedTimestamp,
        keys: impl IntoIterator<Item = SubstateKey>,
        proof: MerkleInclusionProof,
    ) -> Self {
        let mut keys: Vec<SubstateKey> = keys.into_iter().collect();
        keys.sort_unstable();
        keys.dedup();
        Self {
            anchor,
            anchor_ts,
            keys,
            proof,
        }
    }

    /// Whether the bundle is in the one form it may take: sorted keys,
    /// without repeats, naming something, and no more than the cap.
    ///
    /// A bundle naming no key answers nothing and would cost a block a
    /// leaf for it, so it is not well-formed rather than merely
    /// pointless.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        !self.keys.is_empty()
            && self.keys.len() <= MAX_COMMITTED_TX_QUERY
            && self.keys.windows(2).all(|pair| pair[0] < pair[1])
    }

    /// Each key as the proof attests it under the anchor's root.
    ///
    /// # Errors
    ///
    /// As [`MerkleInclusionProof::inclusions`]: a proof that does not
    /// decode, does not claim every key, or reconstructs another root.
    pub fn inclusions(&self) -> Result<Vec<(SubstateKey, Inclusion)>, StateProofError> {
        self.proof
            .inclusions(self.anchor.state_root, self.anchor.shard, &self.keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Address, AddressClass, BlockHeight, Hash, LocalKey, ShardId, StateRoot};

    fn key(seed: u8) -> SubstateKey {
        SubstateKey {
            owner: Address::new([seed; 31], AddressClass::Component),
            local: LocalKey([seed; 16]),
        }
    }

    fn anchor() -> StateAnchor {
        StateAnchor {
            shard: ShardId::ROOT,
            height: BlockHeight::new(3),
            state_root: StateRoot::from_raw(Hash::from_bytes(b"root")),
        }
    }

    /// One form: whatever order a caller offers, the bundle it builds
    /// is the bundle every other builder would have produced.
    #[test]
    fn a_bundle_is_built_in_its_canonical_order() {
        let jumbled = StateProofBundle::new(
            anchor(),
            WeightedTimestamp::ZERO,
            [key(3), key(1), key(3), key(2)],
            MerkleInclusionProof::dummy(),
        );
        let ordered = StateProofBundle::new(
            anchor(),
            WeightedTimestamp::ZERO,
            [key(1), key(2), key(3)],
            MerkleInclusionProof::dummy(),
        );
        assert_eq!(jumbled, ordered);
        assert!(jumbled.is_well_formed());
    }

    /// Empty, repeating, or out of order is a second form of the same
    /// claim, or no claim at all.
    #[test]
    fn a_bundle_out_of_its_form_is_refused() {
        let over = |keys: Vec<SubstateKey>| StateProofBundle {
            anchor: anchor(),
            anchor_ts: WeightedTimestamp::ZERO,
            keys,
            proof: MerkleInclusionProof::dummy(),
        };
        assert!(!over(Vec::new()).is_well_formed());
        assert!(!over(vec![key(2), key(1)]).is_well_formed());
        assert!(!over(vec![key(1), key(1)]).is_well_formed());
    }
}
