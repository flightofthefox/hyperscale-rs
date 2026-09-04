//! Merkle proofs over the JMT state tree, on the wire.

use hyperscale_hbor::Hbor;
use hyperscale_jmt::{Blake3Hasher, ClaimTermination, MultiProof, Tree};
use thiserror::Error;

use crate::{MAX_MERKLE_PROOF_LEN, ShardId, StateRoot, SubstateKey, shard_prefix_path};

/// Merkle multiproof authenticating substates' presence in, or absence
/// from, the JMT state tree.
///
/// Opaque bytes containing an encoded `hyperscale_jmt::MultiProof`.
/// Generation lives in the storage crate, which walks the tree; reading
/// one back against a root is [`Self::inclusions`] here, beside the
/// provisions verifier that consumes the same bytes.
///
/// The proof contains:
/// - Per-claimed-key termination metadata (leaf / empty-subtree / leaf-mismatch)
/// - Sibling hashes for bottom-up verification
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hbor)]
#[hbor(transparent)]
pub struct MerkleInclusionProof(#[hbor(max = MAX_MERKLE_PROOF_LEN)] pub Vec<u8>);

/// What a proof says about one key under the root it reconstructs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inclusion {
    /// A leaf at the key.
    Present,
    /// No leaf at the key: an empty slot, or a leaf for some other key
    /// on the path.
    Absent,
}

/// Why a proof does not answer for the keys it was asked about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StateProofError {
    /// The bytes do not decode as a multiproof.
    #[error("malformed state proof")]
    Malformed,
    /// A key asked about has no claim in the proof.
    #[error("state proof carries no claim for a requested key")]
    MissingClaim,
    /// The proof does not reconstruct the root it was checked against.
    #[error("state proof does not reconstruct the state root")]
    RootMismatch,
}

impl MerkleInclusionProof {
    /// Create a new proof from raw bytes.
    #[must_use]
    pub const fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Each of `keys` as this proof attests it under `shard`'s `root`, in
    /// the order asked.
    ///
    /// The proof's own claims are what is checked: each asked key's
    /// claim is read off the proof and the tree is reconstructed from
    /// exactly those, so a proof that reaches `root` has attested every
    /// termination it carries. A key the proof does not claim, a proof
    /// that does not decode, and one reconstructing another root are
    /// each refused whole — a proof is usable or it is not.
    ///
    /// `shard` is whose root this is, and every claim must sit under its
    /// prefix. A shard's tree is rooted there, so the reconstruction
    /// never reads the bits above it: without the prefix a proof from one
    /// shard answers for another shard's keys, and since an absence
    /// contributes the same empty hash whatever key it names, it answers
    /// them absent.
    ///
    /// # Errors
    ///
    /// As [`StateProofError`] lists them.
    pub fn inclusions(
        &self,
        root: StateRoot,
        shard: ShardId,
        keys: &[SubstateKey],
    ) -> Result<Vec<(SubstateKey, Inclusion)>, StateProofError> {
        let proof = MultiProof::decode(self.as_bytes()).map_err(|_| StateProofError::Malformed)?;
        let mut expected = Vec::with_capacity(keys.len());
        let mut inclusions = Vec::with_capacity(keys.len());
        for key in keys {
            let jmt_key = key.to_bytes();
            let claim = proof
                .claims
                .binary_search_by_key(&jmt_key, |claim| claim.key)
                .ok()
                .map(|at| &proof.claims[at])
                .ok_or(StateProofError::MissingClaim)?;
            let inclusion = match claim.termination {
                ClaimTermination::Leaf => Inclusion::Present,
                ClaimTermination::EmptySubtree | ClaimTermination::LeafMismatch { .. } => {
                    Inclusion::Absent
                }
            };
            expected.push((
                jmt_key,
                (inclusion == Inclusion::Present)
                    .then_some(claim.value_hash)
                    .flatten(),
            ));
            inclusions.push((*key, inclusion));
        }
        let root_bytes: [u8; 32] = *root.as_raw().as_bytes();
        <Tree<Blake3Hasher, 1>>::verify(&proof, root_bytes, &shard_prefix_path(shard), &expected)
            .map_err(|_| StateProofError::RootMismatch)?;
        Ok(inclusions)
    }

    /// Get the raw proof bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Create a dummy (empty) proof for testing.
    #[cfg(any(test, feature = "test-utils"))]
    #[must_use]
    pub const fn dummy() -> Self {
        Self(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use hyperscale_hbor::{
        DecodeError, from_slice as hbor_from_slice, to_vec as hbor_to_vec, varint,
    };
    use hyperscale_jmt::{Key as JmtKey, LeafValue, MemoryStore, NodeKey};

    use super::*;
    use crate::Hash;
    use crate::state_key::jmt_value_hash;
    use crate::test_utils::test_key;

    type Jmt = Tree<Blake3Hasher, 1>;

    /// A one-version tree holding `present`, and a proof over `asked`
    /// against it.
    fn tree_and_proof(
        present: &[SubstateKey],
        asked: &[SubstateKey],
    ) -> (StateRoot, MerkleInclusionProof) {
        let mut store = MemoryStore::new();
        let updates: BTreeMap<JmtKey, Option<LeafValue>> = present
            .iter()
            .map(|key| {
                let value = key.to_bytes().to_vec();
                (
                    key.to_bytes(),
                    Some(LeafValue::new(jmt_value_hash(&value), value.len() as u64)),
                )
            })
            .collect();
        let result = Jmt::apply_updates(&store, None, 1, &updates).unwrap();
        let root = StateRoot::from_raw(Hash::from_hash_bytes(&result.root_hash));
        store.apply(&result);
        let jmt_keys: Vec<JmtKey> = asked.iter().map(SubstateKey::to_bytes).collect();
        let proof = Jmt::prove(&store, &NodeKey::root(1), &jmt_keys).unwrap();
        (root, MerkleInclusionProof::new(proof.encode()))
    }

    /// A key the tree holds reads back present and one it does not
    /// reads back absent, in the order asked, under the root the proof
    /// reconstructs.
    #[test]
    fn inclusions_read_presence_and_absence_off_one_proof() {
        let (held, other, missing) = (test_key(1), test_key(2), test_key(3));
        let (root, proof) = tree_and_proof(&[held, other], &[missing, held]);
        assert_eq!(
            proof
                .inclusions(root, ShardId::ROOT, &[missing, held])
                .unwrap(),
            vec![(missing, Inclusion::Absent), (held, Inclusion::Present)]
        );
    }

    /// A proof is refused whole when it reconstructs another root, when
    /// it carries no claim for a key asked, and when it does not decode.
    #[test]
    fn inclusions_refuse_a_proof_that_does_not_answer() {
        let (held, missing) = (test_key(1), test_key(3));
        let (root, proof) = tree_and_proof(&[held], &[missing]);
        let other_root = StateRoot::from_raw(Hash::from_bytes(b"another state"));
        assert_eq!(
            proof.inclusions(other_root, ShardId::ROOT, &[missing]),
            Err(StateProofError::RootMismatch)
        );
        assert_eq!(
            proof.inclusions(root, ShardId::ROOT, &[missing, held]),
            Err(StateProofError::MissingClaim),
            "the proof was taken over the missing key alone"
        );
        assert_eq!(
            MerkleInclusionProof::new(vec![0xff; 8]).inclusions(root, ShardId::ROOT, &[missing]),
            Err(StateProofError::Malformed)
        );
    }

    #[test]
    fn roundtrip_preserves_bytes() {
        let proof = MerkleInclusionProof::new(vec![0xab; 1024]);
        let bytes = hbor_to_vec(&proof).unwrap();
        let decoded: MerkleInclusionProof = hbor_from_slice(&bytes).unwrap();
        assert_eq!(decoded, proof);
    }

    #[test]
    fn decode_rejects_oversized_proof() {
        let mut buf = Vec::new();
        varint::write(&mut buf, MAX_MERKLE_PROOF_LEN + 1).unwrap();
        buf.extend(std::iter::repeat_n(0u8, MAX_MERKLE_PROOF_LEN + 1));
        let err = hbor_from_slice::<MerkleInclusionProof>(&buf).unwrap_err();
        assert!(matches!(
            err,
            DecodeError::BoundExceeded { max, actual }
                if max == MAX_MERKLE_PROOF_LEN && actual == MAX_MERKLE_PROOF_LEN + 1
        ));
    }
}
