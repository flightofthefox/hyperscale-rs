//! Shared execution-certificate store.
//!
//! Single source of truth for aggregated [`ExecutionCertificate`]s during the
//! window between local aggregation/verification and the tick's containing
//! block committing. Held behind an `Arc` so the single-threaded execution
//! coordinator (the sole writer) and the network worker thread (read-only,
//! serving cross-shard EC fetch requests) can share the same map without
//! channel-bouncing or contending on a coordinator lock.
//!
//! Two writers — both inside the coordinator — feed this store:
//!
//! - **Tick-leader path** inserts on local EC aggregation, before the cert is
//!   broadcast to local peers and remote shards.
//! - **Non-leader path** inserts after verifying a local-shard EC received via
//!   broadcast, so any node can serve fallback EC fetches for its own shard.
//!
//! Eviction is lifecycle-driven: entries are dropped in
//! [`ExecutionCoordinator::remove_finalization`] once the tick's containing
//! block commits, at which point the EC is durably available via
//! [`ShardStorage::get_execution_certificates_by_height`] and the network handler
//! falls through to that on cache miss.
//!
//! Mirrors [`hyperscale_mempool::TxStore`] in shape and intent: a primary
//! index keyed by the natural identifier (`TickId` here, `TxHash` there),
//! plus a transaction index — a counterpart fetches an outcome by naming
//! the transaction, having no way to know which certificate carries it.
//!
//! Backed by [`papaya::HashMap`] — a lock-free concurrent map. Reads from the
//! network worker are wait-free in the common case and never contend with
//! the single state-machine writer.
//!
//! [`ExecutionCertificate`]: hyperscale_types::ExecutionCertificate
//! [`ExecutionCoordinator::remove_finalization`]: crate::ExecutionCoordinator::remove_finalization
//! [`ShardStorage::get_execution_certificates_by_height`]: hyperscale_storage::ShardStorage::get_execution_certificates_by_height

use std::sync::Arc;

use hyperscale_types::{ExecutionCertificate, TickId, TxHash, TxOutcome, Verified};
use papaya::HashMap;

/// Shared, content-addressed store of aggregated [`ExecutionCertificate`]s
/// awaiting block commit.
///
/// Read-heavy on the network worker thread (one lookup per inbound EC
/// fetch); writes (insert on aggregation/verification, evict on tick-cert
/// commit) are infrequent and single-threaded (state machine).
pub struct ExecCertStore {
    inner: HashMap<TickId, Arc<Verified<ExecutionCertificate>>>,
    /// Attested transaction → the certificate carrying its outcome.
    by_tx: HashMap<TxHash, TickId>,
}

impl ExecCertStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
            by_tx: HashMap::new(),
        }
    }

    /// Insert a verified execution certificate. Idempotent: re-inserting
    /// the same `TickId` is a no-op (the existing `Arc` is preserved so
    /// callers holding clones keep pointing at the same allocation).
    pub fn insert(&self, cert: Arc<Verified<ExecutionCertificate>>) {
        let tick_id = *cert.tick_id();
        // Index before the primary insert, so a concurrent reader that
        // resolves a transaction always finds the certificate behind it.
        let by_tx = self.by_tx.pin();
        for tx_hash in cert.tx_outcomes().iter().map(TxOutcome::tx_hash) {
            by_tx.insert(tx_hash, tick_id);
        }
        self.inner.pin().get_or_insert_with(tick_id, || cert);
    }

    /// Look up the verified certificate carrying `tx_hash`'s outcome.
    #[must_use]
    pub fn get_for_tx(&self, tx_hash: TxHash) -> Option<Arc<Verified<ExecutionCertificate>>> {
        let tick_id = self.by_tx.pin().get(&tx_hash).copied()?;
        self.get(&tick_id)
    }

    /// Look up a verified execution certificate by `TickId`.
    #[must_use]
    pub fn get(&self, tick_id: &TickId) -> Option<Arc<Verified<ExecutionCertificate>>> {
        self.inner.pin().get(tick_id).cloned()
    }

    /// Drop the entry for `tick_id`, if any, along with its transaction
    /// index entries.
    pub fn evict(&self, tick_id: &TickId) {
        if let Some(cert) = self.inner.pin().remove(tick_id) {
            let by_tx = self.by_tx.pin();
            for tx_hash in cert.tx_outcomes().iter().map(TxOutcome::tx_hash) {
                // A later certificate re-indexing the same transaction owns
                // the entry; only drop the one this certificate still holds.
                if by_tx.get(&tx_hash) == Some(tick_id) {
                    by_tx.remove(&tx_hash);
                }
            }
        }
    }

    /// Number of certificates currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True when the store holds no certificates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for ExecCertStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    use hyperscale_types::{
        AggregateSignature, BlockHeight, GlobalReceiptRoot, ShardId, SignerBitfield,
        WeightedTimestamp,
    };

    use super::*;

    fn cert(block_height: u64) -> Arc<Verified<ExecutionCertificate>> {
        let tick_id = TickId::new(ShardId::ROOT, BlockHeight::new(block_height));
        Arc::new(Verified::new_unchecked_for_test(ExecutionCertificate::new(
            tick_id,
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            vec![],
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        )))
    }

    #[test]
    fn insert_then_get_round_trips() {
        let store = ExecCertStore::new();
        let c = cert(1);
        let id = *c.tick_id();
        store.insert(Arc::clone(&c));
        assert_eq!(store.get(&id).map(|a| *a.tick_id()), Some(id));
    }

    #[test]
    fn insert_is_idempotent() {
        let store = ExecCertStore::new();
        let c = cert(1);
        store.insert(Arc::clone(&c));
        store.insert(Arc::clone(&c));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn evict_removes_only_named_entry() {
        let store = ExecCertStore::new();
        let a = cert(1);
        let b = cert(2);
        store.insert(Arc::clone(&a));
        store.insert(Arc::clone(&b));
        store.evict(a.tick_id());
        assert!(store.get(a.tick_id()).is_none());
        assert!(store.get(b.tick_id()).is_some());
    }

    #[test]
    fn evict_absent_is_noop() {
        let store = ExecCertStore::new();
        let a = cert(1);
        store.evict(a.tick_id());
        assert!(store.is_empty());
    }
}
