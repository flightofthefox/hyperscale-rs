//! Commitment evidence: proof that a block committed on its shard.
//!
//! A quorum certificate proves *availability*, not canonicality — an
//! f+1..2f committee can certify two blocks at one height without any
//! validator breaking the safe-vote rule. [`CommitProof`] carries the
//! committing *structure* a bare QC cannot forge: a round-contiguous
//! two-chain, which is the HotStuff-2 direct-commit rule, plus a bounded
//! parent-hash ancestry link for a block that committed only as the prefix
//! of a later two-chain after a view change (INV-SHARD-4).
//!
//! Not itself evidence of misbehaviour. Consumers span the adversarial and
//! the ordinary: [`ShardForkProof`](super::evidence::ShardForkProof) pairs
//! two conflicting proofs into a fork accusation, remote-header
//! consumption marks heights commit-proven, and a split child's follower
//! establishes that the parent's terminal block committed before deriving
//! its genesis from it.

use hyperscale_crypto::Verifier;
use sbor::prelude::*;
use thiserror::Error;

use crate::{
    BlockHash, BlockHeader, BlockHeight, CertifiedBlockHeader, ConsensusPublicKey,
    NetworkDefinition, QcContext, QcVerifyError, ShardId, Verify, VoteCount,
};

/// Cap on a [`CommitProof`]'s ancestry-link length.
///
/// A block commits as the prefix of a later two-chain only across a
/// bounded view-change gap (INV-SHARD-4), so the parent-hash link from the
/// directly-committed block down to the proven block is short. Caps
/// verifier work and, once the proof rides gossip, wire decode.
pub const MAX_COMMIT_PROOF_ANCESTRY: usize = 256;

/// A committee resolved for one QC in a [`CommitProof`].
///
/// Signer public keys in committee (bitfield) order, plus the quorum
/// threshold. Produced by [`ShardForkProof::resolve_committees`] from the
/// topology schedule so an off-thread verifier
/// ([`ShardForkProof::verify_resolved`]) runs the signature work without the
/// schedule in hand — the same emitter-resolves pattern the beacon-block
/// verify action uses.
#[derive(Debug, Clone)]
pub struct ResolvedCommittee {
    /// Committee public keys, positionally aligned to the QC's signer
    /// bitfield.
    pub public_keys: Vec<ConsensusPublicKey>,
    /// Quorum threshold for the shard at the QC's window.
    pub quorum_threshold: VoteCount,
}

/// Proof that a specific block committed on its source shard — the
/// artifact a bare QC cannot forge.
///
/// The commit is witnessed by a round-contiguous two-chain: `child`
/// certifies `certified` (`child.parent == certified.hash`,
/// `child.height == certified.height + 1`, `child.round ==
/// certified.round + 1`), which is exactly the HotStuff-2 direct-commit
/// rule. In the common case `certified` *is* the proven block and
/// `ancestry` is empty. When the proven block committed only as the
/// *prefix* of a later two-chain after a view change (INV-SHARD-4),
/// `ancestry` is the parent-hash header chain from `certified`'s parent
/// down to the proven block: each link is pinned by the hash chain
/// descending from the QC-committed `certified`, so no signature is
/// needed below the two-chain — collision resistance carries the rest.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct CommitProof {
    /// Lower block of the committing two-chain — directly committed by
    /// [`Self::child`]. The proven block when [`Self::ancestry`] is empty.
    certified: CertifiedBlockHeader,
    /// Round-contiguous child that commits [`Self::certified`].
    child: CertifiedBlockHeader,
    /// Parent-hash header chain from [`Self::certified`]'s parent down to
    /// the proven block; empty when `certified` is itself the proven
    /// block. `ancestry[0]` is `certified`'s parent; `ancestry[i].hash()
    /// == ancestry[i-1].parent_block_hash()`; the last element is the
    /// proven block.
    ancestry: Vec<BlockHeader>,
}

/// Failure modes of [`CommitProof`] verification.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum CommitProofVerifyError {
    /// A member's QC does not commit its own header (`qc.block_hash !=
    /// header.hash`, or a shard/height mismatch).
    #[error("commit proof header/QC linkage mismatch")]
    Linkage,
    /// The two-chain crosses shards, or an ancestry link does.
    #[error("commit proof spans shards")]
    ShardMismatch,
    /// `child` does not extend `certified` (parent hash or height off).
    #[error("commit proof child does not extend the certified block")]
    NotAChild,
    /// `child.round != certified.round + 1` — not a direct commit.
    #[error("commit proof child is not round-contiguous")]
    NotRoundContiguous,
    /// An ancestry link's hash or height does not chain down from
    /// `certified`.
    #[error("commit proof ancestry link is broken")]
    AncestryBroken,
    /// The ancestry link exceeds [`MAX_COMMIT_PROOF_ANCESTRY`].
    #[error("commit proof ancestry link is too long")]
    AncestryTooLong,
    /// A member QC failed signature verification against its committee.
    #[error("commit proof QC verification failed: {0}")]
    Qc(#[from] QcVerifyError),
}

impl CommitProof {
    /// Build a commit proof from a two-chain and its ancestry link.
    #[must_use]
    pub const fn new(
        certified: CertifiedBlockHeader,
        child: CertifiedBlockHeader,
        ancestry: Vec<BlockHeader>,
    ) -> Self {
        Self {
            certified,
            child,
            ancestry,
        }
    }

    /// A direct-commit proof: `certified` is itself the proven block,
    /// committed by its round-contiguous `child`.
    #[must_use]
    pub const fn direct(certified: CertifiedBlockHeader, child: CertifiedBlockHeader) -> Self {
        Self::new(certified, child, Vec::new())
    }

    /// The shard this proof is on.
    #[must_use]
    pub const fn shard(&self) -> ShardId {
        self.certified.shard_id()
    }

    /// Lower block of the committing two-chain — the branch head this
    /// proof commits (the proven block itself for a direct commit).
    #[must_use]
    pub const fn certified(&self) -> &CertifiedBlockHeader {
        &self.certified
    }

    /// Hash of the proven block — `certified`'s hash for a direct commit,
    /// or the bottom of the ancestry link for a prefix commit.
    #[must_use]
    pub fn proven_block_hash(&self) -> BlockHash {
        self.ancestry
            .last()
            .map_or_else(|| self.certified.block_hash(), BlockHeader::hash)
    }

    /// Height of the proven block.
    #[must_use]
    pub fn proven_height(&self) -> BlockHeight {
        self.ancestry
            .last()
            .map_or_else(|| self.certified.height(), BlockHeader::height)
    }

    /// The two headers carrying QCs, in canonical order. Both
    /// [`ShardForkProof::resolve_committees`] and
    /// [`ShardForkProof::verify_resolved`] iterate QCs through this, so
    /// resolved committees always line up positionally with the QCs they
    /// verify.
    pub(crate) const fn qc_headers(&self) -> [&CertifiedBlockHeader; 2] {
        [&self.certified, &self.child]
    }

    /// Structural checks that need no committee: header/QC linkage, the
    /// round-contiguous two-chain shape, and a well-formed ancestry link.
    pub(crate) fn verify_structure(&self) -> Result<(), CommitProofVerifyError> {
        for ch in self.qc_headers() {
            if ch.qc().block_hash() != ch.block_hash()
                || ch.qc().shard_id() != ch.shard_id()
                || ch.qc().height() != ch.height()
            {
                return Err(CommitProofVerifyError::Linkage);
            }
        }

        if self.child.shard_id() != self.certified.shard_id() {
            return Err(CommitProofVerifyError::ShardMismatch);
        }
        if self.child.header().parent_block_hash() != self.certified.block_hash()
            || self.child.height() != self.certified.height().next()
        {
            return Err(CommitProofVerifyError::NotAChild);
        }
        if self.child.header().round() != self.certified.header().round().next() {
            return Err(CommitProofVerifyError::NotRoundContiguous);
        }

        if self.ancestry.len() > MAX_COMMIT_PROOF_ANCESTRY {
            return Err(CommitProofVerifyError::AncestryTooLong);
        }
        let mut expected_hash = self.certified.header().parent_block_hash();
        let mut expected_height = self.certified.height().prev();
        for link in &self.ancestry {
            if link.shard_id() != self.certified.shard_id() {
                return Err(CommitProofVerifyError::ShardMismatch);
            }
            if link.hash() != expected_hash || expected_height != Some(link.height()) {
                return Err(CommitProofVerifyError::AncestryBroken);
            }
            expected_hash = link.parent_block_hash();
            expected_height = link.height().prev();
        }
        Ok(())
    }

    /// Verify this proof standalone: the two-chain's structure, then both
    /// member QCs against their resolved committees (`[certified, child]`,
    /// positionally aligned to the two-chain).
    ///
    /// [`ShardForkProof`] verifies its two member proofs through the same
    /// checks; this is the entry for a consumer holding a single proof —
    /// a split child's follower establishing that the parent's terminal
    /// block *committed* rather than merely certified, which a bare QC
    /// cannot show.
    ///
    /// # Errors
    ///
    /// A [`CommitProofVerifyError`] naming the failing check.
    pub fn verify_resolved(
        &self,
        verifier: &dyn Verifier,
        network: &NetworkDefinition,
        committees: &[ResolvedCommittee; 2],
    ) -> Result<(), CommitProofVerifyError> {
        self.verify_structure()?;
        self.verify_qcs(verifier, network, committees)
    }

    /// Verify both member QCs against their resolved committees.
    /// `committees` is `[certified_committee, child_committee]`.
    pub(crate) fn verify_qcs(
        &self,
        verifier: &dyn Verifier,
        network: &NetworkDefinition,
        committees: &[ResolvedCommittee],
    ) -> Result<(), CommitProofVerifyError> {
        for (ch, committee) in self.qc_headers().into_iter().zip(committees) {
            let ctx = QcContext {
                network,
                public_keys: &committee.public_keys,
                quorum_threshold: committee.quorum_threshold,
                verifier,
            };
            ch.qc().verify(&ctx)?;
        }
        Ok(())
    }
}
