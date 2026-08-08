//! Inbound execution-certificate fetch request handling.

use std::collections::HashSet;
use std::sync::Arc;

use hyperscale_execution::ExecCertStore;
use hyperscale_metrics::record_fetch_response_sent;
use hyperscale_storage::{PendingChain, ShardStorage};
use hyperscale_types::network::request::GetExecutionCertsRequest;
use hyperscale_types::network::response::GetExecutionCertsResponse;
use hyperscale_types::{ExecutionCertificate, TxHash, WaveId};

/// Serve an inbound execution-certificate fetch request.
///
/// Two tiers: the in-memory [`ExecCertStore`] (entries live here between
/// EC aggregation and the wave's containing block committing) and chain
/// storage via [`PendingChain`]. Cache eviction happens at wave-cert
/// commit, at which point storage is the authoritative source.
///
/// The request names transactions, and one certificate covers a whole
/// batch of them, so several requested transactions commonly resolve to
/// the same certificate — it is sent once.
pub fn serve_execution_certs_request<S: ShardStorage>(
    pending_chain: &PendingChain<S>,
    exec_cert_store: &ExecCertStore,
    req: &GetExecutionCertsRequest,
) -> GetExecutionCertsResponse {
    let mut certs: Vec<Arc<ExecutionCertificate>> = Vec::new();
    let mut served: HashSet<WaveId> = HashSet::new();
    let mut missing: Vec<TxHash> = Vec::new();
    for &tx_hash in &req.tx_hashes {
        match exec_cert_store.get_for_tx(tx_hash) {
            Some(cert) => {
                if served.insert(cert.wave_id().clone()) {
                    certs.push(Arc::new((**cert).clone()));
                }
            }
            None => missing.push(tx_hash),
        }
    }

    if !missing.is_empty() {
        for cert in pending_chain.execution_certificates_for_txs(&missing) {
            if served.insert(cert.wave_id().clone()) {
                certs.push(Arc::new(cert.into_inner()));
            }
        }
    }

    if certs.is_empty() {
        GetExecutionCertsResponse { certificates: None }
    } else {
        record_fetch_response_sent("exec_cert", certs.len());
        GetExecutionCertsResponse {
            certificates: Some(certs),
        }
    }
}
