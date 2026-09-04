//! [`StateProofsRoot`] verification.

use thiserror::Error;

use crate::{
    CounterpartClaim, Hash, StateProofBundle, StateProofsRoot, TransactionDecision, Verified,
    Verify, compute_merkle_root,
};

/// The root over `claims`, in block order. Empty →
/// [`StateProofsRoot::ZERO`]; otherwise the merkle root of each claim's
/// own leaf.
///
/// A cells claim's leaf covers the anchor it was taken at, the anchor's
/// clock, every key it answers for and the proof bytes themselves; a
/// verdict claim's covers the counterpart, the transaction, the anchor,
/// the decision and the certificate digest. So two blocks claiming the
/// same root carry the same answers.
///
/// The two arms are tagged apart. Their preimages could not collide by
/// length today, but a root is what a signed header commits to, and a
/// separation that holds by arithmetic rather than by construction is
/// one a later field can quietly remove.
#[must_use]
pub fn state_proofs_root_from_bundles(claims: &[CounterpartClaim]) -> StateProofsRoot {
    if claims.is_empty() {
        return StateProofsRoot::ZERO;
    }
    let leaves: Vec<Hash> = claims.iter().map(claim_leaf).collect();
    StateProofsRoot::from_raw(compute_merkle_root(&leaves))
}

/// One claim's leaf, tagged by arm.
fn claim_leaf(claim: &CounterpartClaim) -> Hash {
    match claim {
        CounterpartClaim::Cells(bundle) => bundle_leaf(bundle),
        CounterpartClaim::Verdict(verdict) => {
            let mut bytes = VERDICT_CLAIM_LEAF_TAG.to_vec();
            bytes.extend_from_slice(&verdict.shard.to_le_bytes());
            bytes.extend_from_slice(verdict.tx_hash.as_bytes());
            bytes.extend_from_slice(&verdict.anchor_ts.as_millis().to_le_bytes());
            bytes.push(match verdict.decision {
                TransactionDecision::Accept => 0,
                TransactionDecision::Reject => 1,
                TransactionDecision::Aborted => 2,
            });
            bytes.extend_from_slice(verdict.digest.as_bytes());
            Hash::from_bytes(&bytes)
        }
    }
}

/// Domain tag separating a verdict claim's merkle leaf from a bundle's
/// and from every other leaf preimage the codebase hashes.
const VERDICT_CLAIM_LEAF_TAG: &[u8] = b"hyperscale.verdict_claim_leaf.v1";

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
    pub bundles: &'a [CounterpartClaim],
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
    pub fn compute(bundles: &[CounterpartClaim]) -> Self {
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
        StateRoot, SubstateKey, TxHash, VerdictClaim, WeightedTimestamp,
    };

    fn key(seed: u8) -> SubstateKey {
        SubstateKey {
            owner: Address::new([seed; 31], AddressClass::Component),
            local: LocalKey([seed; 16]),
        }
    }

    fn bundle(height: u64, keys: &[u8], proof: &[u8]) -> CounterpartClaim {
        CounterpartClaim::Cells(StateProofBundle::new(
            StateAnchor {
                shard: ShardId::ROOT,
                height: BlockHeight::new(height),
                state_root: StateRoot::from_raw(Hash::from_bytes(b"root")),
            },
            WeightedTimestamp::from_millis(height * 1_000),
            keys.iter().map(|seed| key(*seed)),
            MerkleInclusionProof::new(proof.to_vec()),
        ))
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
        if let CounterpartClaim::Cells(cells) = &mut other_clock {
            cells.anchor_ts = WeightedTimestamp::from_millis(1);
        }
        assert_ne!(base, state_proofs_root_from_bundles(&[other_clock]));
    }

    /// A verdict's leaf covers every term of it, and the two arms are
    /// tagged apart rather than separated by however their preimages
    /// happen to be shaped.
    #[test]
    fn every_term_of_a_verdict_moves_the_root() {
        let claim = |anchor: u64, decision, digest: &[u8]| {
            CounterpartClaim::Verdict(VerdictClaim {
                shard: ShardId::leaf(1, 1),
                tx_hash: TxHash::from(Hash::from_bytes(b"tx")),
                anchor_ts: WeightedTimestamp::from_millis(anchor),
                decision,
                digest: Hash::from_bytes(digest),
            })
        };
        let base = state_proofs_root_from_bundles(&[claim(9, TransactionDecision::Reject, b"d")]);
        assert_ne!(
            base,
            state_proofs_root_from_bundles(&[claim(10, TransactionDecision::Reject, b"d")]),
        );
        assert_ne!(
            base,
            state_proofs_root_from_bundles(&[claim(9, TransactionDecision::Aborted, b"d")]),
        );
        assert_ne!(
            base,
            state_proofs_root_from_bundles(&[claim(9, TransactionDecision::Reject, b"e")]),
        );
        assert_ne!(
            base,
            state_proofs_root_from_bundles(&[bundle(3, &[1], b"p")]),
            "and a verdict is never a bundle",
        );
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
