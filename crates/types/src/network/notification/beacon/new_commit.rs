//! SPC new-commit notification — announces a committed-high triple.

use std::sync::Arc;

use sbor::prelude::BasicSbor;

use crate::{
    ConsensusSignature, DOMAIN_SPC_NEW_COMMIT, Epoch, MessageClass, NetworkDefinition,
    NetworkMessage, Signed, SpcNewCommitMsg, ValidatorId, Verifiable, spc_relay_signing_message,
};

/// Committed-low announcement broadcast within the slot's committee
/// when an SPC participant commits a verifiable low value.
///
/// The inner [`SpcNewCommitMsg`] is self-authenticating via its
/// embedded `PcQc3` — verifiers check the committee aggregate in the
/// proof and that `proof.x_pp() == value`. `sender` +
/// `sender_signature` ride on the wrapper for relay accountability:
/// the signature is a signature under the sender's key over `(network,
/// epoch, view, msg.hash())`, used to key per-`(epoch, view, sender)`
/// pipeline slots. Wire decode lands the inner wrapper as
/// `Verifiable::Unverified`; locally-dispatched sends preserve the
/// `Verified` marker.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct SpcNewCommitNotification {
    /// Epoch the inner SPC instance belongs to.
    pub epoch: Epoch,
    /// Validator relaying this commit — the implicit signer of
    /// `sender_signature`.
    pub sender: ValidatorId,
    /// signature over `spc_relay_signing_message(network,
    /// DOMAIN_SPC_NEW_COMMIT, epoch, msg.view, msg.hash())`.
    pub sender_signature: ConsensusSignature,
    /// The committed new-commit message.
    pub msg: Arc<Verifiable<SpcNewCommitMsg>>,
}

impl SpcNewCommitNotification {
    /// Wrap an [`SpcNewCommitMsg`] for notification with the
    /// relay-attestation sender + signature. The caller produces
    /// `sender_signature` over [`spc_relay_signing_message`] with
    /// [`DOMAIN_SPC_NEW_COMMIT`].
    #[must_use]
    pub fn new(
        epoch: Epoch,
        sender: ValidatorId,
        sender_signature: ConsensusSignature,
        msg: impl Into<Arc<Verifiable<SpcNewCommitMsg>>>,
    ) -> Self {
        Self {
            epoch,
            sender,
            sender_signature,
            msg: msg.into(),
        }
    }

    /// Get the inner message (raw view, regardless of verification
    /// state).
    #[must_use]
    pub fn msg(&self) -> &SpcNewCommitMsg {
        self.msg.as_unverified()
    }

    /// Consume and return the inner message, preserving the
    /// verification marker.
    #[must_use]
    pub fn into_msg(self) -> Arc<Verifiable<SpcNewCommitMsg>> {
        self.msg
    }
}

impl Signed for SpcNewCommitNotification {
    fn signer(&self) -> ValidatorId {
        self.sender
    }

    fn signature(&self) -> &ConsensusSignature {
        &self.sender_signature
    }

    fn signing_message(&self, network: &NetworkDefinition) -> Vec<u8> {
        spc_relay_signing_message(
            network,
            DOMAIN_SPC_NEW_COMMIT,
            self.epoch,
            self.msg.as_unverified().view,
            &self.msg.as_unverified().hash(),
        )
    }
}

impl NetworkMessage for SpcNewCommitNotification {
    fn message_type_id() -> &'static str {
        "beacon.spc.new_commit"
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
        AggregateSignature, Epoch, PcQc2, PcQc3, PcSignerLengths, PcVector, PcXpProof,
        SignerBitfield, SpcView, ValidatorId,
    };

    fn sample_pc_qc3() -> PcQc3 {
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        let qc2 = PcQc2::new(
            PcVector::empty(),
            signers,
            AggregateSignature::new([0x11; 96]),
            PcXpProof::Full,
        );
        PcQc3::new(
            PcVector::empty(),
            qc2,
            None,
            None,
            SignerBitfield::new(4),
            PcSignerLengths::Uniform(0),
            AggregateSignature::new([0x33; 96]),
        )
    }

    fn sample_msg() -> SpcNewCommitMsg {
        SpcNewCommitMsg {
            view: SpcView::new(4),
            value: PcVector::empty(),
            proof: sample_pc_qc3().into(),
        }
    }

    #[test]
    fn sbor_round_trip() {
        let n = SpcNewCommitNotification::new(
            Epoch::new(7),
            ValidatorId::new(3),
            ConsensusSignature::new([0x55; 96]),
            Arc::new(Verifiable::from(sample_msg())),
        );
        let bytes = basic_encode(&n).unwrap();
        let decoded: SpcNewCommitNotification = basic_decode(&bytes).unwrap();
        assert_eq!(n, decoded);
    }

    #[test]
    fn class_is_consensus() {
        assert_eq!(SpcNewCommitNotification::class(), MessageClass::Consensus);
    }
}
