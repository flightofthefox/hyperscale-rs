//! Prefix Consensus round-2 vote notification.

use std::sync::Arc;

use sbor::prelude::BasicSbor;

use crate::{MessageClass, NetworkMessage, PcVote2, SpcView, Verifiable};

/// PC round-2 vote sent via unicast to peers in the slot's committee.
///
/// The inner [`PcVote2`] is self-authenticating — it carries the signer
/// id, one BLS signature per prefix of `x`, the round-1 QC the signer
/// is building on, and a length attestation pinning `|x|`. The wrapping
/// `view` rides alongside because PC votes don't carry their SPC view
/// internally. Wire decode lands the wrapper as `Verifiable::Unverified`;
/// local-dispatched sends from a colocated voter preserve
/// `Verifiable::Verified`.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct PcVote2Notification {
    /// SPC view whose inner PC produced this vote.
    pub view: SpcView,
    /// The vote.
    pub vote: Arc<Verifiable<PcVote2>>,
}

impl PcVote2Notification {
    /// Wrap a [`PcVote2`] for notification. Accepts a raw vote or a
    /// `Verified<PcVote2>`.
    #[must_use]
    pub fn new(view: SpcView, vote: impl Into<Arc<Verifiable<PcVote2>>>) -> Self {
        Self {
            view,
            vote: vote.into(),
        }
    }

    /// Get the inner vote (raw view, regardless of verification state).
    #[must_use]
    pub fn vote(&self) -> &PcVote2 {
        self.vote.as_unverified()
    }

    /// Consume and return the inner vote, preserving the verification
    /// marker.
    #[must_use]
    pub fn into_vote(self) -> Arc<Verifiable<PcVote2>> {
        self.vote
    }
}

impl NetworkMessage for PcVote2Notification {
    fn message_type_id() -> &'static str {
        "beacon.pc.vote2"
    }

    fn class() -> MessageClass {
        MessageClass::Consensus
    }
}

#[cfg(test)]
mod tests {
    use sbor::prelude::*;

    use super::*;
    use crate::{
        AggregateSignature, ConsensusSignature, PcCompactVote, PcQc1, PcVector, PositionalBundle,
        SignerBitfield, SpcView, ValidatorId,
    };

    fn sample_qc1() -> PcQc1 {
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        PcQc1::new(
            PcVector::empty(),
            PositionalBundle::new(signers, vec![PcCompactVote::new(0, None)]),
            AggregateSignature::new([0xAA; 96]),
        )
    }

    fn sample_vote() -> PcVote2 {
        PcVote2::new(
            ValidatorId::new(2),
            PcVector::empty(),
            vec![ConsensusSignature::new([0x11; 96])],
            sample_qc1(),
            ConsensusSignature::new([0x22; 96]),
        )
    }

    #[test]
    fn sbor_round_trip() {
        let n =
            PcVote2Notification::new(SpcView::new(2), Arc::new(Verifiable::from(sample_vote())));
        let bytes = basic_encode(&n).unwrap();
        let decoded: PcVote2Notification = basic_decode(&bytes).unwrap();
        assert_eq!(n, decoded);
    }

    #[test]
    fn class_is_consensus() {
        assert_eq!(PcVote2Notification::class(), MessageClass::Consensus);
    }
}
