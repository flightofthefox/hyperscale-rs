//! Execution certificate fetch response for fallback recovery.

use std::sync::Arc;

use hyperscale_hbor::Hbor;

use crate::{ExecutionCertificate, MessageClass, NetworkMessage};

/// Response to an execution certificate fetch request.
///
/// Returns the requested execution certificates from the source shard's cache.
/// `None` means the source shard cannot serve this request (cert not cached,
/// or the block has been pruned). The requester should try a different peer.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct GetExecutionCertsResponse {
    /// The requested execution certificates.
    ///
    /// - `Some(certs)` — successfully found certificates (may be empty if
    ///   no matching ticks were cached).
    /// - `None` — the source shard cannot serve this request.
    ///
    /// `Arc`-wrapped because the server-side `ExecCertStore` holds each
    /// cert behind `Arc` already.
    pub certificates: Option<Vec<Arc<ExecutionCertificate>>>,
}

impl NetworkMessage for GetExecutionCertsResponse {
    fn message_type_id() -> &'static str {
        "execution_cert.response"
    }

    fn class() -> MessageClass {
        MessageClass::CrossShardProgress
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};

    use super::*;

    #[test]
    fn test_hbor_roundtrip_empty() {
        let response = GetExecutionCertsResponse {
            certificates: Some(vec![]),
        };

        let encoded = hbor_to_vec(&response).unwrap();
        let decoded: GetExecutionCertsResponse = hbor_from_slice(&encoded).unwrap();
        assert_eq!(response, decoded);
    }

    #[test]
    fn test_hbor_roundtrip_unavailable() {
        let response = GetExecutionCertsResponse { certificates: None };

        let encoded = hbor_to_vec(&response).unwrap();
        let decoded: GetExecutionCertsResponse = hbor_from_slice(&encoded).unwrap();
        assert_eq!(response, decoded);
    }
}
