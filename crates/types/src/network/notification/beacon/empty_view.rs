//! SPC empty-view notification.

use std::sync::Arc;

use sbor::prelude::BasicSbor;

use crate::{
    ConsensusSignature, DOMAIN_PC_EMPTY_VIEW, Epoch, MessageClass, NetworkDefinition,
    NetworkMessage, Signed, SpcEmptyViewMsg, ValidatorId, Verifiable, hash_high_value,
    pc_vote_signing_message, skip_target, spc_context,
};

/// SPC empty-view declaration sent via unicast when a participant
/// times out on a view without observing a leader proposal.
///
/// The inner [`SpcEmptyViewMsg`] is content-signed — it carries the
/// signer id and a BLS signature over the canonical empty-view signing
/// bytes (`skip_target` under [`DOMAIN_PC_EMPTY_VIEW`], bound to the
/// epoch via `spc_context`). The wrapper carries `epoch` so the relay
/// edge can reconstruct that signing message and authenticate the
/// signer via the [`Signed`] check before the coordinator keys a
/// per-`(epoch, view, signer)` verification slot — the same relay-edge
/// discipline `new_view` / `new_commit` get. The embedded reported-
/// triple `PcQc3` is still verified asynchronously downstream.
///
/// Wire decode lands the wrapper as `Verifiable::Unverified`;
/// locally-dispatched sends from a colocated signer preserve
/// `Verifiable::Verified` and bypass the relay-edge check.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct SpcEmptyViewMsgNotification {
    /// Epoch the inner SPC instance belongs to. Bound into the signer's
    /// signing message via `spc_context`, so a swap across epochs makes
    /// the relay-edge signature check fail.
    pub epoch: Epoch,
    /// The empty-view message.
    pub msg: Arc<Verifiable<SpcEmptyViewMsg>>,
}

impl SpcEmptyViewMsgNotification {
    /// Wrap an [`SpcEmptyViewMsg`] for notification at `epoch`. Accepts
    /// a raw msg or a `Verified<SpcEmptyViewMsg>` — the wrapper
    /// preserves the marker.
    #[must_use]
    pub fn new(epoch: Epoch, msg: impl Into<Arc<Verifiable<SpcEmptyViewMsg>>>) -> Self {
        Self {
            epoch,
            msg: msg.into(),
        }
    }

    /// Get the inner empty-view message (raw view, regardless of
    /// verification state).
    #[must_use]
    pub fn msg(&self) -> &SpcEmptyViewMsg {
        self.msg.as_unverified()
    }

    /// Consume and return the inner empty-view message, preserving the
    /// verification marker.
    #[must_use]
    pub fn into_msg(self) -> Arc<Verifiable<SpcEmptyViewMsg>> {
        self.msg
    }
}

impl Signed for SpcEmptyViewMsgNotification {
    fn signer(&self) -> ValidatorId {
        self.msg.as_unverified().signer
    }

    fn signature(&self) -> &ConsensusSignature {
        &self.msg.as_unverified().sig
    }

    fn signing_message(&self, network: &NetworkDefinition) -> Vec<u8> {
        let msg = self.msg.as_unverified();
        let value_hash = hash_high_value(&msg.reported.value);
        let target = skip_target(msg.view, msg.reported.view, value_hash);
        let spc_ctx = spc_context(self.epoch);
        pc_vote_signing_message(network, DOMAIN_PC_EMPTY_VIEW, spc_ctx.as_bytes(), &target)
    }
}

impl NetworkMessage for SpcEmptyViewMsgNotification {
    fn message_type_id() -> &'static str {
        "beacon.spc.empty_view"
    }

    fn class() -> MessageClass {
        MessageClass::Consensus
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_crypto::Signer;
    use hyperscale_crypto_bls::{BlsSigner, BlsVerifier};
    use sbor::prelude::*;

    use super::*;
    use crate::{
        AggregateSignature, Epoch, NetworkDefinition, PcQc2, PcQc3, PcSignerLengths, PcVector,
        PcXpProof, SignedContext, SignedVerifyError, SignerBitfield, SpcHighTriple, SpcView,
        ValidatorId, sign_empty_view_msg, spc_context,
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

    fn sample_msg() -> SpcEmptyViewMsg {
        SpcEmptyViewMsg {
            view: SpcView::new(5),
            reported: SpcHighTriple {
                view: SpcView::new(3),
                value: PcVector::empty(),
                proof: sample_pc_qc3().into(),
            },
            signer: ValidatorId::new(2),
            sig: ConsensusSignature::new([0x44; 96]),
        }
    }

    fn signing_key(seed: u64) -> BlsSigner {
        let mut s = [0u8; 32];
        s[..8].copy_from_slice(&seed.to_le_bytes());
        BlsSigner::from_seed(&s)
    }

    /// Build a notification whose inner empty-view is signed by
    /// `signer_key_seed`'s key but claims `claimed_signer` as the signer
    /// id. Honest when the two match; a forged-signer relay when they
    /// don't.
    fn signed_notification(
        epoch: Epoch,
        signer_key_seed: u64,
        claimed_signer: u64,
    ) -> SpcEmptyViewMsgNotification {
        let reported = SpcHighTriple {
            view: SpcView::new(3),
            value: PcVector::empty(),
            proof: sample_pc_qc3().into(),
        };
        let msg = sign_empty_view_msg(
            &signing_key(signer_key_seed),
            ValidatorId::new(claimed_signer),
            &NetworkDefinition::simulator(),
            &spc_context(epoch),
            SpcView::new(5),
            reported,
        )
        .expect("sign");
        SpcEmptyViewMsgNotification::new(epoch, Arc::new(Verifiable::from(msg)))
    }

    #[test]
    fn sbor_round_trip() {
        let n = SpcEmptyViewMsgNotification::new(
            Epoch::new(7),
            Arc::new(Verifiable::from(sample_msg())),
        );
        let bytes = basic_encode(&n).unwrap();
        let decoded: SpcEmptyViewMsgNotification = basic_decode(&bytes).unwrap();
        assert_eq!(n, decoded);
    }

    #[test]
    fn class_is_consensus() {
        assert_eq!(
            SpcEmptyViewMsgNotification::class(),
            MessageClass::Consensus
        );
    }

    /// The `Signed` relay-edge check passes when the signer's own key
    /// produced the embedded sig — the honest path.
    #[test]
    fn signed_signature_verifies_under_signer_key() {
        let n = signed_notification(Epoch::new(7), 2, 2);
        let pk = signing_key(2).public_key();
        assert!(
            n.verify_signature(&SignedContext {
                verifier: &BlsVerifier,
                network: &NetworkDefinition::simulator(),
                public_key: &pk,
            })
            .is_ok()
        );
    }

    /// A wrapper that claims signer 2 but carries a sig from a different
    /// key is rejected at the relay edge — so a peer can't mint (and
    /// squat) validator 2's `(epoch, view, signer)` verification slot
    /// with a forged-signer empty-view.
    #[test]
    fn signed_signature_rejects_forged_signer() {
        let n = signed_notification(Epoch::new(7), 99, 2);
        let honest_pk = signing_key(2).public_key();
        assert_eq!(
            n.verify_signature(&SignedContext {
                verifier: &BlsVerifier,
                network: &NetworkDefinition::simulator(),
                public_key: &honest_pk,
            }),
            Err(SignedVerifyError::InvalidSignature),
        );
    }

    /// A signature bound to one epoch fails to verify when the wrapper
    /// claims another — `spc_context(epoch)` is folded into the signed
    /// bytes.
    #[test]
    fn signed_signature_is_epoch_bound() {
        let mut n = signed_notification(Epoch::new(7), 2, 2);
        n.epoch = Epoch::new(8);
        let pk = signing_key(2).public_key();
        assert_eq!(
            n.verify_signature(&SignedContext {
                verifier: &BlsVerifier,
                network: &NetworkDefinition::simulator(),
                public_key: &pk,
            }),
            Err(SignedVerifyError::InvalidSignature),
        );
    }
}
