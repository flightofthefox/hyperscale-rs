//! Inbound execution-certificate fetch request handling.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hyperscale_execution::ExecCertStore;
use hyperscale_metrics::record_fetch_response_sent;
use hyperscale_storage::{PendingChain, ShardStorage};
use hyperscale_types::network::request::GetExecutionCertsRequest;
use hyperscale_types::network::response::GetExecutionCertsResponse;
use hyperscale_types::{ExecutionCertificate, TickId, TxHash};

/// Serve an inbound execution-certificate fetch request.
///
/// Two tiers: the in-memory [`ExecCertStore`] (entries live here between
/// EC aggregation and the wave's containing block committing) and chain
/// storage via [`PendingChain`]. Cache eviction happens at finalization
/// commit, at which point storage is the authoritative source.
///
/// The request names transactions, and one certificate covers a whole
/// batch of them, so several requested transactions commonly resolve to
/// the same certificate — it is answered once, projected to the
/// transactions that were actually asked about. A requester asks for what
/// it is missing, so that projection is what it needs and no more; the
/// broadcast this request stands in for was projected the same way.
///
/// A certificate this shard did not produce is already a projection and
/// cannot be narrowed further — the sibling nodes to rebuild the root
/// around a smaller set are exactly what it does not carry. It is
/// answered as it stands.
pub fn serve_execution_certs_request<S: ShardStorage>(
    pending_chain: &PendingChain<S>,
    exec_cert_store: &ExecCertStore,
    req: &GetExecutionCertsRequest,
) -> GetExecutionCertsResponse {
    // Certificates in first-asked order, each with the transactions this
    // request named it for.
    let mut asked: HashMap<TickId, (Arc<ExecutionCertificate>, HashSet<TxHash>)> = HashMap::new();
    let mut order: Vec<TickId> = Vec::new();
    let mut missing: Vec<TxHash> = Vec::new();

    for &tx_hash in &req.tx_hashes {
        match exec_cert_store.get_for_tx(tx_hash) {
            Some(cert) => record(&mut asked, &mut order, Arc::new((**cert).clone()), tx_hash),
            None => missing.push(tx_hash),
        }
    }

    if !missing.is_empty() {
        for cert in pending_chain.execution_certificates_for_txs(&missing) {
            let cert = Arc::new(cert.into_inner());
            for tx_hash in &missing {
                if cert.covers(tx_hash) {
                    record(&mut asked, &mut order, Arc::clone(&cert), *tx_hash);
                }
            }
        }
    }

    let certs: Vec<Arc<ExecutionCertificate>> = order
        .into_iter()
        .filter_map(|tick_id| {
            let (cert, txs) = asked.remove(&tick_id)?;
            if cert.is_complete() {
                cert.project_to(&txs).map(Arc::new)
            } else {
                Some(cert)
            }
        })
        .collect();

    if certs.is_empty() {
        GetExecutionCertsResponse { certificates: None }
    } else {
        record_fetch_response_sent("exec_cert", certs.len());
        GetExecutionCertsResponse {
            certificates: Some(certs),
        }
    }
}

/// File `tx_hash` under the certificate answering for it, preserving the
/// order certificates were first asked about.
fn record(
    asked: &mut HashMap<TickId, (Arc<ExecutionCertificate>, HashSet<TxHash>)>,
    order: &mut Vec<TickId>,
    cert: Arc<ExecutionCertificate>,
    tx_hash: TxHash,
) {
    let tick_id = *cert.tick_id();
    asked
        .entry(tick_id)
        .or_insert_with(|| {
            order.push(tick_id);
            (cert, HashSet::new())
        })
        .1
        .insert(tx_hash);
}
