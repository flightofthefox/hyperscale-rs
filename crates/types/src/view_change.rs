//! View change types for liveness.

use crate::{
    view_change_message, BlockHeight, Hash, QuorumCertificate, ShardGroupId, Signature,
    SignerBitfield, ValidatorId, VotePower,
};
use sbor::prelude::*;

/// A vote to change the view (leader rotation on timeout).
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct ViewChangeVote {
    /// New view/round number being proposed.
    pub new_view: u64,

    /// Current round number (the round we're leaving).
    pub current_round: u64,

    /// Height at which the view change is happening.
    pub height: BlockHeight,

    /// The highest QC this validator has seen.
    pub highest_qc: QuorumCertificate,

    /// Validator casting this vote.
    pub voter: ValidatorId,

    /// Signature on the view change message.
    pub signature: Signature,
}

impl ViewChangeVote {
    /// Create a new view change vote.
    pub fn new(
        height: BlockHeight,
        current_round: u64,
        new_view: u64,
        voter: ValidatorId,
        highest_qc: QuorumCertificate,
        signature: Signature,
    ) -> Self {
        Self {
            new_view,
            current_round,
            height,
            highest_qc,
            voter,
            signature,
        }
    }

    /// Get the vote key (height, new_view) for tracking.
    pub fn vote_key(&self) -> (BlockHeight, u64) {
        (self.height, self.new_view)
    }

    /// Create the canonical message bytes for signing.
    ///
    /// Uses the centralized `view_change_message` function with the
    /// `DOMAIN_VIEW_CHANGE` tag for domain separation.
    ///
    /// Note: View change votes require `shard_group` context which this type
    /// doesn't contain. Use `hyperscale_types::view_change_message()` with
    /// the shard_group from your BFT state.
    pub fn signing_message(&self, shard_group: ShardGroupId) -> Vec<u8> {
        view_change_message(shard_group, self.height, self.new_view)
    }
}

/// Certificate proving 2f+1 validators agreed to change view.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct ViewChangeCertificate {
    /// The new view/round number.
    pub new_view: u64,

    /// Height at which the view change happened.
    pub height: BlockHeight,

    /// The highest QC seen by the validators.
    /// This is the max of all highest_qc values from the votes.
    pub highest_qc: QuorumCertificate,

    /// Hash of the highest QC's block.
    pub highest_qc_block_hash: Hash,

    /// Which validators signed.
    pub signers: SignerBitfield,

    /// Aggregated signature.
    pub aggregated_signature: Signature,

    /// Total voting power represented by this certificate.
    pub voting_power: VotePower,
}

impl ViewChangeCertificate {
    /// Create a genesis view change certificate.
    pub fn genesis() -> Self {
        Self {
            new_view: 0,
            height: BlockHeight(0),
            highest_qc: QuorumCertificate::genesis(),
            highest_qc_block_hash: Hash::ZERO,
            signers: SignerBitfield::empty(),
            aggregated_signature: Signature::zero(),
            voting_power: VotePower(0),
        }
    }

    /// Get the new round (alias for new_view).
    pub fn new_round(&self) -> u64 {
        self.new_view
    }

    /// Get the number of signers.
    pub fn signer_count(&self) -> usize {
        self.signers.count_ones()
    }

    /// Check if this certificate has quorum (> 2/3 voting power).
    pub fn has_quorum(&self, total_power: u64) -> bool {
        VotePower::has_quorum(self.voting_power.0, total_power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Hash, DOMAIN_VIEW_CHANGE};

    #[test]
    fn test_view_change_vote_signing_message() {
        let shard_group = ShardGroupId(1);
        let vote = ViewChangeVote {
            new_view: 5,
            current_round: 4,
            height: BlockHeight(10),
            highest_qc: QuorumCertificate::genesis(),
            voter: ValidatorId(0),
            signature: Signature::zero(),
        };

        let msg = vote.signing_message(shard_group);
        assert!(msg.starts_with(DOMAIN_VIEW_CHANGE));
        assert_eq!(vote.vote_key(), (BlockHeight(10), 5));
    }

    #[test]
    fn test_view_change_certificate_has_hash() {
        let vcc = ViewChangeCertificate {
            new_view: 5,
            height: BlockHeight(10),
            highest_qc: QuorumCertificate::genesis(),
            highest_qc_block_hash: Hash::ZERO,
            signers: SignerBitfield::empty(),
            aggregated_signature: Signature::zero(),
            voting_power: VotePower(0),
        };
        assert_eq!(vcc.highest_qc_block_hash, Hash::ZERO);
    }

    #[test]
    fn test_genesis_view_change_certificate() {
        let vcc = ViewChangeCertificate::genesis();
        assert_eq!(vcc.new_view, 0);
        assert_eq!(vcc.new_round(), 0);
        assert_eq!(vcc.height, BlockHeight(0));
        assert_eq!(vcc.signer_count(), 0);
        assert!(!vcc.has_quorum(4));
    }
}
