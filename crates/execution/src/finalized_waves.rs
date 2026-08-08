//! Terminal-state lookup for finalized waves.
//!
//! A wave lands here after its local EC is aggregated and every remote shard
//! has attested coverage — at that point the wave has a [`WaveCertificate`]
//! and its receipts are ready for block inclusion. Entries are removed by the
//! coordinator once the containing wave-cert block commits; until then the
//! store answers tx-membership and wave-id-hash lookups for peers that need
//! to fetch the finalized data to vote.
//!
//! This is write-once, read-many — [`WaveRegistry`](crate::waves::WaveRegistry)
//! owns the mutable in-flight lifecycle (waves, vote trackers, retries) and
//! hands waves off to this store at the moment of finalization.
//!
//! The underlying map is a `BTreeMap<WaveId, Arc<FinalizedWave>>` so
//! iteration is deterministic — load-bearing for simulation determinism and
//! for proposal building, which iterates the store to include finalized
//! waves in block order. Beside it, a `TxHash → WaveId` index answers the
//! questions that are per transaction rather than per wave: whether a
//! transaction has reached terminal state, which certificate carries it,
//! and what a sync requester already holds.
//!
//! Held behind an `RwLock` so an `Arc<FinalizedWaveStore>` can be shared
//! across every same-shard vnode's `ExecutionCoordinator`. In practice
//! the pinned thread serializes every write, so the lock never contends
//! — it exists to satisfy the type system around shared mutability.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, PoisonError, RwLock};

use hyperscale_types::{
    BloomFilter, DEFAULT_FPR, FinalizedWave, TxHash, Verifiable, WaveCertificate, WaveId,
};

/// Waves by id, plus the transaction index derived from them. Both live
/// under one lock so a reader can never observe a transaction indexed
/// against a wave the map has already dropped.
struct Inner {
    waves: BTreeMap<WaveId, Arc<Verifiable<FinalizedWave>>>,
    by_tx: HashMap<TxHash, WaveId>,
}

/// Per-shard finalized-wave store. See module docs for lifecycle.
///
/// Stored values are [`Verifiable<FinalizedWave>`] in the
/// [`Verifiable::Verified`] variant. Holding the `Block::Live.certificates`
/// transport shape directly lets the proposal-build path source verifiable
/// arcs without a per-extraction conversion; the typed-gate `insert` is
/// the single place where the conversion (and body clone) happens.
pub struct FinalizedWaveStore {
    inner: RwLock<Inner>,
}

impl Default for FinalizedWaveStore {
    fn default() -> Self {
        Self::new()
    }
}

impl FinalizedWaveStore {
    /// Construct an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                waves: BTreeMap::new(),
                by_tx: HashMap::new(),
            }),
        }
    }

    /// True if no finalized waves are currently tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .waves
            .is_empty()
    }

    /// Record a newly-finalized wave under its `WaveId`. Callers wrap
    /// their upstream [`Verified<FinalizedWave>`] into
    /// [`Verifiable::Verified`] before insertion so the same `Arc` can be
    /// shared with downstream `FinalizedWavesAdmitted` consumers without
    /// re-cloning. The store enforces — by virtue of its `Verifiable`
    /// argument type and the typed gates the caller went through to
    /// produce it — that every value held here is in the
    /// `Verifiable::Verified` variant.
    pub fn insert(&self, wave_id: WaveId, fw: Arc<Verifiable<FinalizedWave>>) {
        debug_assert!(
            fw.verified().is_some(),
            "FinalizedWaveStore invariant: only Verifiable::Verified entries are admitted"
        );
        let mut inner = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        for tx_hash in fw.tx_hashes() {
            inner.by_tx.insert(tx_hash, wave_id.clone());
        }
        inner.waves.insert(wave_id, fw);
    }

    /// Remove the entry for `wave_id`, if any. No-op when absent (sync
    /// paths may remove a wave the local node never aggregated).
    pub fn remove(&self, wave_id: &WaveId) {
        let mut inner = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        let Some(fw) = inner.waves.remove(wave_id) else {
            return;
        };
        for tx_hash in fw.tx_hashes() {
            // A later wave re-indexing the same transaction would own the
            // entry; only drop the one this wave still holds.
            if inner.by_tx.get(&tx_hash) == Some(wave_id) {
                inner.by_tx.remove(&tx_hash);
            }
        }
    }

    /// All finalized waves in `WaveId` order. Used by the proposer to
    /// include finalized waves in the next block.
    #[must_use]
    pub fn all_waves(&self) -> Vec<Arc<Verifiable<FinalizedWave>>> {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .waves
            .values()
            .map(Arc::clone)
            .collect()
    }

    /// Lookup by `WaveId`. Peers reference waves by id in fetch requests,
    /// so this is the primary ingress lookup for serving finalized-wave data.
    #[must_use]
    pub fn get(&self, wave_id: &WaveId) -> Option<Arc<Verifiable<FinalizedWave>>> {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .waves
            .get(wave_id)
            .map(Arc::clone)
    }

    /// Certificate containing `tx_hash`, if any. Used to answer
    /// terminal-state queries for a single transaction (e.g. RPC, mempool
    /// status). Returns `None` once the wave has been removed — callers
    /// then fall back to persisted storage.
    #[must_use]
    pub fn get_certificate_for_tx(&self, tx_hash: TxHash) -> Option<Arc<WaveCertificate>> {
        let inner = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        let wave_id = inner.by_tx.get(&tx_hash)?;
        inner
            .waves
            .get(wave_id)
            .map(|fw| Arc::clone(fw.certificate()))
    }

    /// Whether `tx_hash` is part of any currently-tracked finalized wave.
    #[must_use]
    pub fn is_finalized(&self, tx_hash: TxHash) -> bool {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .by_tx
            .contains_key(&tx_hash)
    }

    /// Every transaction in a tracked finalized wave.
    ///
    /// The node passes this to shard consensus for conflict filtering — a transaction
    /// whose wave is already finalized should not be re-proposed.
    #[must_use]
    pub fn all_tx_hashes(&self) -> HashSet<TxHash> {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .by_tx
            .keys()
            .copied()
            .collect()
    }

    /// Whether a wave with this `WaveId` is tracked. Used by debug/query
    /// paths to distinguish "wave is finalized" from "wave has no tracker".
    #[must_use]
    pub fn contains(&self, wave_id: &WaveId) -> bool {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .waves
            .contains_key(wave_id)
    }

    /// Number of finalized waves currently tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .waves
            .len()
    }

    /// Build a bloom filter over every transaction in a tracked wave. Sync
    /// inventory attaches this to `GetBlockRequest` so the responder can
    /// elide the finalized waves the requester already has — a wave whose
    /// every transaction is in the filter.
    #[must_use]
    pub fn cert_bloom_snapshot(&self) -> Option<BloomFilter<TxHash>> {
        // Snapshot ids under the lock, build the bloom after release so
        // we don't hold the read guard across the heavier filter inserts.
        let tx_hashes: Vec<TxHash> = self
            .inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .by_tx
            .keys()
            .copied()
            .collect();
        let mut bf = BloomFilter::with_capacity(tx_hashes.len(), DEFAULT_FPR)?;
        for tx_hash in &tx_hashes {
            bf.insert(tx_hash);
        }
        Some(bf)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use hyperscale_types::{
        AggregateSignature, BlockHeight, ExecutionCertificate, ExecutionOutcome, GlobalReceiptHash,
        GlobalReceiptRoot, Hash, ShardId, SignerBitfield, TxHash, TxOutcome, Verified,
        WeightedTimestamp,
    };

    use super::*;

    fn make_wave_id(block_height: u64) -> WaveId {
        WaveId::new(
            ShardId::ROOT,
            BlockHeight::new(block_height),
            BTreeSet::new(),
        )
    }

    fn make_finalized_wave(
        block_height: u64,
        tx_hashes: &[TxHash],
    ) -> (WaveId, Arc<Verifiable<FinalizedWave>>) {
        let wave_id = make_wave_id(block_height);
        let tx_outcomes: Vec<TxOutcome> = tx_hashes
            .iter()
            .map(|h| {
                TxOutcome::new(
                    *h,
                    ExecutionOutcome::Succeeded {
                        receipt_hash: GlobalReceiptHash::ZERO,
                    },
                )
            })
            .collect();
        let ec = ExecutionCertificate::new(
            wave_id.clone(),
            WeightedTimestamp::ZERO,
            GlobalReceiptRoot::ZERO,
            tx_outcomes,
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );
        let cert = WaveCertificate::new(wave_id.clone(), vec![Arc::new(ec)]);
        // Lookups in this module only inspect the certificate's outcomes; an
        // empty receipts vector is fine for the store's contract.
        let verified = Verified::new_unchecked_for_test(FinalizedWave::new(Arc::new(cert), vec![]));
        let fw = Arc::new(verified.into());
        (wave_id, fw)
    }

    #[test]
    fn empty_store_reports_no_finalized_state() {
        let store = FinalizedWaveStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert!(!store.is_finalized(TxHash::from(Hash::from_bytes(b"anything"))));
        assert!(store.all_tx_hashes().is_empty());
        assert!(store.all_waves().is_empty());
    }

    #[test]
    fn insert_then_lookup_by_tx_hash() {
        let store = FinalizedWaveStore::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx1"));
        let (wid, fw) = make_finalized_wave(1, &[tx]);

        store.insert(wid.clone(), fw);

        assert!(store.is_finalized(tx));
        assert!(store.contains(&wid));
        assert_eq!(store.len(), 1);
        let cert = store.get_certificate_for_tx(tx).expect("cert present");
        assert_eq!(cert.wave_id(), &wid);
    }

    #[test]
    fn lookup_by_wave_id_matches_inserted_wave() {
        let store = FinalizedWaveStore::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx1"));
        let (wid, fw) = make_finalized_wave(1, &[tx]);

        store.insert(wid.clone(), fw);

        let looked_up = store.get(&wid).expect("wave present by id");
        assert_eq!(looked_up.certificate().wave_id(), &wid);

        // Unknown id returns None.
        assert!(store.get(&make_wave_id(99)).is_none());
    }

    #[test]
    fn all_tx_hashes_flattens_across_waves() {
        let store = FinalizedWaveStore::new();
        let a = TxHash::from(Hash::from_bytes(b"a"));
        let b = TxHash::from(Hash::from_bytes(b"b"));
        let c = TxHash::from(Hash::from_bytes(b"c"));
        let (wid1, fw1) = make_finalized_wave(1, &[a, b]);
        let (wid2, fw2) = make_finalized_wave(2, &[c]);

        store.insert(wid1, fw1);
        store.insert(wid2, fw2);

        let all = store.all_tx_hashes();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&a));
        assert!(all.contains(&b));
        assert!(all.contains(&c));
    }

    #[test]
    fn remove_drops_only_the_named_wave() {
        let store = FinalizedWaveStore::new();
        let tx1 = TxHash::from(Hash::from_bytes(b"tx1"));
        let tx2 = TxHash::from(Hash::from_bytes(b"tx2"));
        let (wid1, fw1) = make_finalized_wave(1, &[tx1]);
        let (wid2, fw2) = make_finalized_wave(2, &[tx2]);

        store.insert(wid1.clone(), fw1);
        store.insert(wid2.clone(), fw2);

        store.remove(&wid1);

        assert!(!store.contains(&wid1));
        assert!(store.contains(&wid2));
        assert!(!store.is_finalized(tx1));
        assert!(store.is_finalized(tx2));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn cert_bloom_snapshot_contains_every_tracked_transaction() {
        let store = FinalizedWaveStore::new();
        let tx1 = TxHash::from(Hash::from_bytes(b"tx1"));
        let tx2 = TxHash::from(Hash::from_bytes(b"tx2"));
        let (wid1, fw1) = make_finalized_wave(1, &[tx1]);
        let (wid2, fw2) = make_finalized_wave(2, &[tx2]);
        store.insert(wid1, fw1);
        store.insert(wid2, fw2);

        let bf = store.cert_bloom_snapshot().expect("sizing ok");
        assert!(bf.contains(&tx1));
        assert!(bf.contains(&tx2));
        // Untracked transaction: exercises the filter's zero region.
        assert!(!bf.contains(&TxHash::from(Hash::from_bytes(b"absent"))));
    }

    /// Removing one wave leaves another wave's transactions indexed —
    /// the index is per wave, not a single shared set.
    #[test]
    fn remove_drops_only_the_removed_wave_from_the_tx_index() {
        let store = FinalizedWaveStore::new();
        let a = TxHash::from(Hash::from_bytes(b"a"));
        let b = TxHash::from(Hash::from_bytes(b"b"));
        let (wid1, fw1) = make_finalized_wave(1, &[a]);
        let (wid2, fw2) = make_finalized_wave(2, &[b]);
        store.insert(wid1.clone(), fw1);
        store.insert(wid2, fw2);

        store.remove(&wid1);

        assert!(!store.is_finalized(a));
        assert!(store.is_finalized(b));
        assert!(store.get_certificate_for_tx(a).is_none());
        assert!(store.get_certificate_for_tx(b).is_some());
        assert_eq!(store.all_tx_hashes(), HashSet::from([b]));
    }

    #[test]
    fn remove_absent_wave_is_noop() {
        let store = FinalizedWaveStore::new();
        let missing = make_wave_id(42);
        // No panic, no state change.
        store.remove(&missing);
        assert!(store.is_empty());
    }

    #[test]
    fn all_waves_iterates_in_wave_id_order() {
        let store = FinalizedWaveStore::new();
        let (wid_high, fw_high) = make_finalized_wave(5, &[TxHash::from(Hash::from_bytes(b"hi"))]);
        let (wid_low, fw_low) = make_finalized_wave(1, &[TxHash::from(Hash::from_bytes(b"lo"))]);

        store.insert(wid_high, fw_high);
        store.insert(wid_low, fw_low);

        let waves = store.all_waves();
        assert_eq!(waves.len(), 2);
        // BTreeMap iteration is ordered by key; lower block_height comes first.
        assert_eq!(waves[0].certificate().wave_id().block_height().inner(), 1);
        assert_eq!(waves[1].certificate().wave_id().block_height().inner(), 5);
    }
}
