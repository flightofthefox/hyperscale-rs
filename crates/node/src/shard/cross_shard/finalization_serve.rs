//! Inbound finalization fetch request handling.

use std::sync::Arc;

use hyperscale_metrics::record_fetch_response_sent;
use hyperscale_storage::{PendingChain, ShardStorage};
use hyperscale_types::network::request::GetFinalizationsRequest;
use hyperscale_types::network::response::GetFinalizationsResponse;
use hyperscale_types::{Finalization, TickId, Verifiable};
use quick_cache::sync::Cache as QuickCache;

/// Serve an inbound finalization fetch request.
///
/// Two tiers: an in-memory cache (entries live here between EC aggregation
/// and its containing block committing) and chain storage via
/// [`PendingChain`]. Storage holds attestations and per-tx receipts
/// separately; for anything missed by the cache, we reconstruct the full
/// `Finalization` by pulling both halves. Peers requesting
/// finalizations past the cache window must still get a complete answer from
/// durable storage.
///
/// The wire response carries raw `Arc<Finalization>` bodies — the
/// verification marker is process-local and doesn't cross the network.
pub fn serve_finalizations_request<S: ShardStorage>(
    pending_chain: &PendingChain<S>,
    fw_cache: &QuickCache<TickId, Arc<Verifiable<Finalization>>>,
    req: &GetFinalizationsRequest,
) -> GetFinalizationsResponse {
    let mut finalizations: Vec<Arc<Finalization>> = Vec::new();
    let mut missing: Vec<TickId> = Vec::new();
    for id in &req.tick_ids {
        if let Some(fw) = fw_cache.get(id) {
            finalizations.push(Arc::new(fw.as_unverified().clone()));
        } else {
            missing.push(*id);
        }
    }

    if !missing.is_empty() {
        let certs = pending_chain.certificates_batch(&missing);
        for cert in certs {
            if let Some(fw) =
                Finalization::reconstruct(cert, |h| pending_chain.consensus_receipt(h))
            {
                finalizations.push(Arc::new(fw));
            }
        }
    }

    record_fetch_response_sent("finalization", finalizations.len());
    GetFinalizationsResponse::new(finalizations)
}
