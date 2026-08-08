//! Deduplication index for committed artifacts referenced by block contents.
//!
//! The shard consensus layer enforces a single contract: every committed artifact (tx,
//! cert, provision) appears in the chain exactly once. This index is the
//! mechanism — proposers consult it to filter candidates, validators
//! consult it to reject duplicate inclusions.
//!
//! Per-artifact deadline maps bound the index by artifact-specific
//! BFT-attested horizons:
//!
//! - **txs**: each tx's own `end_timestamp_exclusive` (capped by
//!   `MAX_VALIDITY_RANGE` at admission).
//! - **certs**: `vote_anchor_ts + RETENTION_HORIZON` from the tick's local
//!   EC.
//! - **provisions**: `local_committed_ts + RETENTION_HORIZON`, a
//!   conservative surrogate for `source_weighted_ts` (the source block was
//!   committed before we observed these provisions, so
//!   `local_committed_ts >= source_weighted_ts` always).
//!
//! Pruned when `committed_ts >= deadline`. Past expiry, independent rules
//! reject re-inclusion, so the entry is no longer correctness-bearing.
//!
//! Registration is synchronous with shard commit (called from
//! [`crate::coordinator::ShardCoordinator::record_block_committed`]) so the
//! just-committed block's contents are visible to any subsequent
//! `try_propose` in the same tick — closing the on-qc-formed re-inclusion
//! race without a separate bridge buffer.

use std::collections::HashMap;
use std::sync::Arc;

use hyperscale_types::{
    Finalization, ProvisionHash, Provisions, RETENTION_HORIZON, ShardId, Transaction, TxHash,
    Verifiable, WeightedTimestamp,
};

#[allow(clippy::struct_field_names)] // shared `_retention` postfix is the artifact-tier convention
pub struct CommitDedupIndex {
    /// `tx_hash → end_timestamp_exclusive`. Pruned when
    /// `end_timestamp_exclusive <= current_committed_ts`.
    tx_retention: HashMap<TxHash, WeightedTimestamp>,
    /// `tx_hash → vote_anchor_ts + RETENTION_HORIZON` of the finalization
    /// that resolved it. Every transaction a committed finalization
    /// reached a verdict for, under whichever verdict — which is what
    /// makes settlement and abandonment exclusive rather than two rules
    /// that have to agree, and what a tick key could never express, since
    /// a tick can settle in more than one part.
    resolved_tx_retention: HashMap<TxHash, WeightedTimestamp>,
    /// `provision_hash → local_committed_ts + RETENTION_HORIZON`. Pruned
    /// when `deadline <= current_committed_ts`. Past the horizon, every tx
    /// the batch carried has expired its `validity_range` and terminated
    /// everywhere, so no future block can legitimately reference the same
    /// content-addressed batch.
    provision_retention: HashMap<ProvisionHash, WeightedTimestamp>,
    /// `(source_shard, tx_hash) → deadline`, from committed bundle
    /// *content* — the engagement-mirror evidence that a payer shard's
    /// bundle naming a transaction committed locally. Same deadline tier
    /// as `provision_retention`. Fed from `Block::Live` bodies only: a
    /// sealed (synced) manifest carries bundle hashes without content,
    /// so a freshly synced validator lacks this window's pairs and votes
    /// conservatively until it refills — the designed pairing puts the
    /// bundle in the same block as its transactions, so the
    /// committed-earlier arm is the rare mis-pairing remnant.
    provision_tx_retention: HashMap<(ShardId, TxHash), WeightedTimestamp>,
}

impl CommitDedupIndex {
    pub fn new() -> Self {
        Self {
            tx_retention: HashMap::new(),
            resolved_tx_retention: HashMap::new(),
            provision_retention: HashMap::new(),
            provision_tx_retention: HashMap::new(),
        }
    }

    /// Record a block's transactions in the retention lookup. Each entry's
    /// stored value is the tx's `validity_range.end_timestamp_exclusive`.
    pub fn register_committed_txs(&mut self, transactions: &[Arc<Verifiable<Transaction>>]) {
        for tx in transactions {
            let tx_hash = tx.hash();
            let end = tx.validity_range().end_timestamp_exclusive;
            self.tx_retention.entry(tx_hash).or_insert(end);
        }
    }

    /// Record every transaction a block's finalizations resolved. Each
    /// entry's deadline is the resolving tick's local EC
    /// `vote_anchor_ts + RETENTION_HORIZON`.
    pub fn register_committed_certs(&mut self, finalizations: &[Arc<Verifiable<Finalization>>]) {
        for fw in finalizations {
            let deadline = fw.local_ec().deadline();
            for tx_hash in fw.tx_hashes() {
                self.resolved_tx_retention
                    .entry(tx_hash)
                    .or_insert(deadline);
            }
        }
    }

    /// Record a block's provisions in the retention lookup. Anchored on
    /// `local_committed_ts` (a conservative surrogate for
    /// `source_weighted_ts`). Keyed by `ProvisionHash` so the caller can
    /// source from the block's manifest (which is independent of
    /// `Block::Live`/`Sealed`) rather than depending on `block.provisions()`
    /// (which is empty for `Sealed`).
    pub fn register_committed_provisions(
        &mut self,
        provision_hashes: &[ProvisionHash],
        local_committed_ts: WeightedTimestamp,
    ) {
        let deadline = local_committed_ts.plus(RETENTION_HORIZON);
        for hash in provision_hashes {
            self.provision_retention.entry(*hash).or_insert(deadline);
        }
    }

    /// Record a committed block's bundle content in the engagement-mirror
    /// lookup: every `(source_shard, tx_hash)` pair a committed bundle
    /// names, under the provisions deadline tier.
    pub fn register_committed_provision_txs(
        &mut self,
        batches: &[Arc<Verifiable<Provisions>>],
        local_committed_ts: WeightedTimestamp,
    ) {
        let deadline = local_committed_ts.plus(RETENTION_HORIZON);
        for batch in batches {
            let source = batch.source_shard();
            for entry in batch.transactions() {
                self.provision_tx_retention
                    .entry((source, entry.tx_hash))
                    .or_insert(deadline);
            }
        }
    }

    /// Drop retention-lookup entries past their deadline. `now` is the
    /// `weighted_timestamp` of the latest committed block. Past expiry,
    /// independent rules (tx validity check; finalization-deadline) reject any
    /// re-inclusion, so the entry is no longer correctness-bearing.
    pub fn prune(&mut self, now: WeightedTimestamp) {
        self.tx_retention.retain(|_, end| *end > now);
        self.resolved_tx_retention
            .retain(|_, deadline| *deadline > now);
        self.provision_retention
            .retain(|_, deadline| *deadline > now);
        self.provision_tx_retention
            .retain(|_, deadline| *deadline > now);
    }

    pub fn contains_tx(&self, tx_hash: &TxHash) -> bool {
        self.tx_retention.contains_key(tx_hash)
    }

    /// Whether a committed finalization already reached a verdict for
    /// `tx_hash`, within the retention window.
    pub fn contains_resolved_tx(&self, tx_hash: &TxHash) -> bool {
        self.resolved_tx_retention.contains_key(tx_hash)
    }

    pub fn contains_provision(&self, provision_hash: &ProvisionHash) -> bool {
        self.provision_retention.contains_key(provision_hash)
    }

    /// Whether a committed bundle from `source` named `tx_hash` within
    /// the retention window — the engagement mirror's committed arm.
    pub fn contains_provision_tx(&self, source: ShardId, tx_hash: TxHash) -> bool {
        self.provision_tx_retention.contains_key(&(source, tx_hash))
    }

    pub fn tx_retention_len(&self) -> usize {
        self.tx_retention.len()
    }

    pub fn resolved_tx_retention_len(&self) -> usize {
        self.resolved_tx_retention.len()
    }

    pub fn provision_retention_len(&self) -> usize {
        self.provision_retention.len()
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::test_utils::{
        install_stub_vm_statics, make_finalization, stub_transaction, test_prefix,
    };
    use hyperscale_types::{
        BlockHeight, Hash, MerkleInclusionProof, ProvisionEntry, Provisions, RevealChain, ShardId,
        TimestampRange, TransactionDecision,
    };

    use super::*;

    /// Build a test tx whose `validity_range.end_timestamp_exclusive == end_ms`.
    fn tx_with_end(seed: u8, end_ms: u64) -> Arc<Verifiable<Transaction>> {
        install_stub_vm_statics();
        let range = TimestampRange::new(
            WeightedTimestamp::ZERO,
            WeightedTimestamp::from_millis(end_ms),
        );
        Arc::new(Verifiable::from(stub_transaction(
            test_prefix(seed),
            &[test_prefix(seed)],
            1_000,
            range,
        )))
    }

    fn make_fw(height: u64) -> Arc<Verifiable<Finalization>> {
        Arc::new(
            make_finalization(
                BlockHeight::new(height),
                TxHash::from(Hash::from_bytes(
                    &[u8::try_from(height).unwrap_or(u8::MAX); 32],
                )),
                TransactionDecision::Accept,
            )
            .into(),
        )
    }

    fn make_provisions(seed: u8) -> Arc<Provisions> {
        let tx_hash = TxHash::from(Hash::from_bytes(&[seed; 32]));
        Arc::new(Provisions::new(
            ShardId::leaf(1, 0),
            ShardId::leaf(1, 1),
            BlockHeight::new(u64::from(seed)),
            WeightedTimestamp::ZERO,
            RevealChain::ZERO,
            MerkleInclusionProof::dummy(),
            vec![ProvisionEntry::new(tx_hash, vec![])],
        ))
    }

    // ─── Txs ────────────────────────────────────────────────────────────

    #[test]
    fn register_txs_populates_retention() {
        let mut idx = CommitDedupIndex::new();
        let tx = tx_with_end(1, 60_000);
        let tx_hash = tx.hash();
        idx.register_committed_txs(std::slice::from_ref(&tx));
        assert!(idx.contains_tx(&tx_hash));
        assert_eq!(idx.tx_retention_len(), 1);
    }

    #[test]
    fn prune_drops_txs_past_their_end_exclusive() {
        let mut idx = CommitDedupIndex::new();
        let early = tx_with_end(1, 100);
        let later = tx_with_end(2, 900);
        let early_hash = early.hash();
        let later_hash = later.hash();
        idx.register_committed_txs(&[early, later]);

        idx.prune(WeightedTimestamp::from_millis(500));

        assert!(!idx.contains_tx(&early_hash));
        assert!(idx.contains_tx(&later_hash));
    }

    // ─── Resolutions ────────────────────────────────────────────────────

    /// A committed finalization records every transaction it reached a
    /// verdict for, so a later block naming one of them under a different
    /// tick is refusable. Identity is the transaction and only the
    /// transaction: a tick can settle in more than one part, so its id
    /// answers no question this index is asked.
    #[test]
    fn register_certs_records_what_they_resolved() {
        // make_finalization sets vote_anchor_ts = block_height + 1, so the
        // deadline is that plus RETENTION_HORIZON.
        let mut idx = CommitDedupIndex::new();
        let fw = make_fw(1);
        let tx_hash = fw.tx_hashes().next().expect("a tick names its members");
        idx.register_committed_certs(std::slice::from_ref(&fw));
        assert!(idx.contains_resolved_tx(&tx_hash));

        idx.prune(WeightedTimestamp::ZERO);
        assert!(
            idx.contains_resolved_tx(&tx_hash),
            "still within the window"
        );

        idx.prune(
            fw.local_ec()
                .deadline()
                .plus(std::time::Duration::from_millis(1)),
        );
        assert!(!idx.contains_resolved_tx(&tx_hash));
    }

    // ─── Provisions ─────────────────────────────────────────────────────

    #[test]
    fn register_provisions_populates_retention() {
        let mut idx = CommitDedupIndex::new();
        let p = make_provisions(1);
        idx.register_committed_provisions(&[p.hash()], WeightedTimestamp::from_millis(1_000));
        assert!(idx.contains_provision(&p.hash()));
        assert_eq!(idx.provision_retention_len(), 1);
    }

    #[test]
    fn prune_drops_provisions_past_their_deadline() {
        let mut idx = CommitDedupIndex::new();
        let p = make_provisions(1);
        let now = WeightedTimestamp::from_millis(1_000);
        idx.register_committed_provisions(&[p.hash()], now);

        idx.prune(now);
        assert!(idx.contains_provision(&p.hash()));

        let past = now
            .plus(RETENTION_HORIZON)
            .plus(std::time::Duration::from_millis(1));
        idx.prune(past);
        assert!(!idx.contains_provision(&p.hash()));
    }
}
