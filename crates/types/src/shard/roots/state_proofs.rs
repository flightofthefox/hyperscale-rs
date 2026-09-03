//! [`StateProofsRoot`] verification.

use thiserror::Error;

use crate::{Hash, StateProofBundle, StateProofsRoot, Verified, Verify, compute_merkle_root};

/// The root over `bundles`, in block order. Empty →
/// [`StateProofsRoot::ZERO`]; otherwise the merkle root of each
/// bundle's own leaf.
///
/// A bundle's leaf covers the anchor it was taken at, the anchor's
/// clock, every key it answers for and the proof bytes themselves, so
/// two blocks claiming the same root carry the same answers.
#[must_use]
pub fn state_proofs_root_from_bundles(bundles: &[StateProofBundle]) -> StateProofsRoot {
    if bundles.is_empty() {
        return StateProofsRoot::ZERO;
    }
    let leaves: Vec<Hash> = bundles.iter().map(bundle_leaf).collect();
    StateProofsRoot::from_raw(compute_merkle_root(&leaves))
}

/// Domain tag separating a state-proof bundle's merkle leaf from every
/// other leaf preimage the codebase hashes.
const STATE_PROOF_LEAF_TAG: &[u8] = b"hyperscale.state_proof_leaf.v1";

/// One bundle's leaf: its anchor, the anchor's clock, each key in the
/// canonical order the bundle is built in, and the proof bytes.
fn bundle_leaf(bundle: &StateProofBundle) -> Hash {
    let mut bytes = STATE_PROOF_LEAF_TAG.to_vec();
    bytes.reserve(56 + bundle.keys.len() * 48 + bundle.proof.as_bytes().len());
    bytes.extend_from_slice(&bundle.anchor.shard.to_le_bytes());
    bytes.extend_from_slice(&bundle.anchor.height.inner().to_le_bytes());
    bytes.extend_from_slice(bundle.anchor.state_root.as_raw().as_bytes());
    bytes.extend_from_slice(&bundle.anchor_ts.as_millis().to_le_bytes());
    bytes.extend_from_slice(&(bundle.keys.len() as u64).to_le_bytes());
    for key in &bundle.keys {
        bytes.extend_from_slice(&key.to_bytes());
    }
    bytes.extend_from_slice(bundle.proof.as_bytes());
    Hash::from_bytes(&bytes)
}

/// Inputs the [`StateProofsRoot`] verifier reads against.
#[derive(Debug, Clone, Copy)]
pub struct StateProofsRootContext<'a> {
    /// The block's bundles — each contributes one leaf.
    pub bundles: &'a [StateProofBundle],
}

/// Failure modes of [`StateProofsRoot`] verification.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum StateProofsRootVerifyError {
    /// The root computed from the bundles is not the claimed root.
    #[error("computed state proofs root {computed:?} ≠ claimed {expected:?}")]
    Mismatch {
        /// Header's claimed root.
        expected: StateProofsRoot,
        /// Root computed from the bundles.
        computed: StateProofsRoot,
    },
}

impl Verified<StateProofsRoot> {
    /// Compute the root over `bundles`. Verified by construction.
    #[must_use]
    pub fn compute(bundles: &[StateProofBundle]) -> Self {
        Self::new_unchecked(state_proofs_root_from_bundles(bundles))
    }
}

impl Verify<&StateProofsRootContext<'_>> for StateProofsRoot {
    type Error = StateProofsRootVerifyError;

    fn verify(&self, context: &StateProofsRootContext<'_>) -> Result<Verified<Self>, Self::Error> {
        let computed = state_proofs_root_from_bundles(context.bundles);
        if computed != *self {
            return Err(StateProofsRootVerifyError::Mismatch {
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
        Address, AddressClass, BlockHeight, LocalKey, MerkleInclusionProof, ShardId, StateAnchor,
        StateRoot, SubstateKey, WeightedTimestamp,
    };

    fn key(seed: u8) -> SubstateKey {
        SubstateKey {
            owner: Address::new([seed; 31], AddressClass::Component),
            local: LocalKey([seed; 16]),
        }
    }

    fn bundle(height: u64, keys: &[u8], proof: &[u8]) -> StateProofBundle {
        StateProofBundle::new(
            StateAnchor {
                shard: ShardId::ROOT,
                height: BlockHeight::new(height),
                state_root: StateRoot::from_raw(Hash::from_bytes(b"root")),
            },
            WeightedTimestamp::from_millis(height * 1_000),
            keys.iter().map(|seed| key(*seed)),
            MerkleInclusionProof::new(proof.to_vec()),
        )
    }

    #[test]
    fn an_empty_section_has_the_zero_root() {
        assert_eq!(state_proofs_root_from_bundles(&[]), StateProofsRoot::ZERO);
    }

    /// Every term of a bundle is under its leaf: the anchor, the clock,
    /// the keys and the bytes each move the root.
    #[test]
    fn every_term_of_a_bundle_moves_the_root() {
        let base = state_proofs_root_from_bundles(&[bundle(3, &[1, 2], b"p")]);
        assert_ne!(
            base,
            state_proofs_root_from_bundles(&[bundle(4, &[1, 2], b"p")])
        );
        assert_ne!(
            base,
            state_proofs_root_from_bundles(&[bundle(3, &[1], b"p")])
        );
        assert_ne!(
            base,
            state_proofs_root_from_bundles(&[bundle(3, &[1, 2], b"q")])
        );
        let mut other_clock = bundle(3, &[1, 2], b"p");
        other_clock.anchor_ts = WeightedTimestamp::from_millis(1);
        assert_ne!(base, state_proofs_root_from_bundles(&[other_clock]));
    }

    #[test]
    fn a_claimed_root_verifies_against_its_bundles_and_no_others() {
        let bundles = [bundle(3, &[1], b"p"), bundle(4, &[2], b"q")];
        let claimed = state_proofs_root_from_bundles(&bundles);
        let context = StateProofsRootContext { bundles: &bundles };
        assert_eq!(
            claimed.verify(&context).map(Verified::into_inner),
            Ok(claimed)
        );
        assert!(matches!(
            StateProofsRoot::ZERO.verify(&context),
            Err(StateProofsRootVerifyError::Mismatch { .. })
        ));
    }
}
