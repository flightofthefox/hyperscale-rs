//! ViewChangeCertificate gossip message.

use hyperscale_types::{NetworkMessage, ShardMessage, ViewChangeCertificate};
use sbor::prelude::BasicSbor;

/// Broadcasts view change certificate (2f+1 votes) to synchronize all validators to a new round.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct ViewChangeCertificateGossip {
    /// The view change certificate being gossiped
    pub certificate: ViewChangeCertificate,
}

impl ViewChangeCertificateGossip {
    /// Create a new view change certificate gossip message.
    pub fn new(certificate: ViewChangeCertificate) -> Self {
        Self { certificate }
    }

    /// Get the inner view change certificate.
    pub fn certificate(&self) -> &ViewChangeCertificate {
        &self.certificate
    }

    /// Consume and return the inner view change certificate.
    pub fn into_certificate(self) -> ViewChangeCertificate {
        self.certificate
    }
}

// Network message implementation
impl NetworkMessage for ViewChangeCertificateGossip {
    fn message_type_id() -> &'static str {
        "view_change.certificate"
    }
}

impl ShardMessage for ViewChangeCertificateGossip {}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_types::{
        BlockHeight, QuorumCertificate, Signature, SignerBitfield, VotePower,
    };

    fn make_test_certificate() -> ViewChangeCertificate {
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);

        let highest_qc = QuorumCertificate::genesis();
        let highest_qc_block_hash = highest_qc.block_hash;

        ViewChangeCertificate {
            new_view: 1,
            height: BlockHeight(10),
            highest_qc,
            highest_qc_block_hash,
            aggregated_signature: Signature::zero(),
            signers,
            voting_power: VotePower(3),
        }
    }

    #[test]
    fn test_view_change_certificate_gossip_creation() {
        let cert = make_test_certificate();
        let gossip = ViewChangeCertificateGossip::new(cert.clone());
        assert_eq!(gossip.certificate(), &cert);
    }

    #[test]
    fn test_view_change_certificate_gossip_into_certificate() {
        let cert = make_test_certificate();
        let gossip = ViewChangeCertificateGossip::new(cert.clone());
        let extracted = gossip.into_certificate();
        assert_eq!(extracted, cert);
    }
}
