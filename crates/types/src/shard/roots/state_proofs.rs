//! [`StateProofsRoot`]: the root over a block's state-proof bundles,
//! one leaf per bundle.

use hyperscale_hbor::to_vec as hbor_to_vec;

use crate::{Hash, LeafRoot, StateProofBundle, StateProofsRoot};

/// Domain tag separating a state-proof bundle's merkle leaf from every
/// other leaf preimage the codebase hashes.
const STATE_PROOF_LEAF_TAG: &[u8] = b"hyperscale.state_proof_leaf.v1";

impl LeafRoot for StateProofsRoot {
    type Leaf = StateProofBundle;

    const ZERO: Self = Self::ZERO;

    fn from_raw(raw: Hash) -> Self {
        Self::from_raw(raw)
    }

    /// One bundle's leaf: the tag and its canonical encoding — the
    /// anchor it was taken at, every key it answers for and the proof
    /// bytes themselves — so two blocks claiming the same root carry
    /// the same answers.
    ///
    /// # Panics
    ///
    /// If the bundle does not encode, which a value built through
    /// [`StateProofBundle::new`] under its caps always does.
    fn leaf(bundle: &Self::Leaf) -> Hash {
        let bytes = hbor_to_vec(bundle).expect("a state proof bundle encodes");
        Hash::from_parts(&[STATE_PROOF_LEAF_TAG, &bytes])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Address, AddressClass, Anchor, BlockHeight, LocalKey, MerkleInclusionProof, RootMismatch,
        ShardId, StateRoot, SubstateKey, Verified, Verify, WeightedTimestamp,
    };

    fn key(seed: u8) -> SubstateKey {
        SubstateKey {
            owner: Address::new([seed; 31], AddressClass::Component),
            local: LocalKey([seed; 16]),
        }
    }

    fn bundle(height: u64, keys: &[u8], proof: &[u8]) -> StateProofBundle {
        StateProofBundle::new(
            Anchor {
                shard: ShardId::ROOT,
                height: BlockHeight::new(height),
                state_root: StateRoot::from_raw(Hash::from_bytes(b"root")),
                ts: WeightedTimestamp::from_millis(height * 1_000),
            },
            keys.iter().map(|seed| key(*seed)),
            MerkleInclusionProof::new(proof.to_vec()),
        )
    }

    #[test]
    fn an_empty_section_has_the_zero_root() {
        assert_eq!(StateProofsRoot::over(&[]), StateProofsRoot::ZERO);
    }

    /// Every term of a bundle is under its leaf: the anchor, the clock,
    /// the keys and the bytes each move the root.
    #[test]
    fn every_term_of_a_bundle_moves_the_root() {
        let base = StateProofsRoot::over(&[bundle(3, &[1, 2], b"p")]);
        assert_ne!(base, StateProofsRoot::over(&[bundle(4, &[1, 2], b"p")]));
        assert_ne!(base, StateProofsRoot::over(&[bundle(3, &[1], b"p")]));
        assert_ne!(base, StateProofsRoot::over(&[bundle(3, &[1, 2], b"q")]));
        let mut other_clock = bundle(3, &[1, 2], b"p");
        other_clock.anchor.ts = WeightedTimestamp::from_millis(1);
        assert_ne!(base, StateProofsRoot::over(&[other_clock]));
    }

    #[test]
    fn a_claimed_root_verifies_against_its_bundles_and_no_others() {
        let bundles = [bundle(3, &[1], b"p"), bundle(4, &[2], b"q")];
        let claimed = StateProofsRoot::over(&bundles);
        assert_eq!(
            claimed.verify(&bundles[..]).map(Verified::into_inner),
            Ok(claimed)
        );
        assert_eq!(
            StateProofsRoot::ZERO.verify(&bundles[..]),
            Err(RootMismatch {
                expected: StateProofsRoot::ZERO,
                computed: claimed,
            })
        );
    }
}
