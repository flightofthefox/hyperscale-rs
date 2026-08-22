//! Provision absorption and readiness tracking for cross-shard
//! transactions.
//!
//! Three correlated maps:
//!
//! - [`verified`](ProvisioningTracker::verified) — committed entry lists
//!   keyed by `tx_hash`, one `Arc<Vec<SubstateEntry>>` per source shard
//!   contribution. Feeds the cross-shard dispatch action for each tx.
//! - `required` — the set of remote shards each cross-shard tx needs
//!   provisions from. Populated when the tx's tick is created.
//! - `received` — the set of remote shards whose provisions have actually
//!   landed. Populated by [`absorb_provisions`](ProvisioningTracker::absorb_provisions).
//!
//! A tx is fully provisioned when `required ⊆ received`; that predicate is
//! surfaced as [`is_fully_provisioned`](ProvisioningTracker::is_fully_provisioned)
//! so callers never inspect the underlying sets.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use hyperscale_types::{
    Provisions, RETENTION_HORIZON, ShardId, SubstateEntry, TxHash, Verified, WeightedTimestamp,
};

/// The environment a source block's bundle carries for the transactions
/// that block committed: the clock, checked against the commit-proven
/// source header at verification.
///
/// One field, and it stays a record because what a bundle carries about
/// the environment is a set rather than a value — a seed is the
/// beacon's, so it needs no carrying, and what else might is answered
/// here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceAnchor {
    /// The source block's parent-QC weighted timestamp.
    pub clock: WeightedTimestamp,
}

pub struct ProvisioningTracker {
    /// Verified provisions keyed by `tx_hash`. Written when provisions are
    /// absorbed; read when a cross-shard tick dispatches. Cleared on the
    /// terminal-state path ([`remove_tx`]) when a finalization
    /// commits, and swept by [`gc_stale_provisions`] for txs whose
    /// retention horizon elapsed without ever finalizing.
    verified: HashMap<TxHash, Vec<Arc<Vec<SubstateEntry>>>>,

    /// Remote shards each cross-shard tx needs provisions from. Populated
    /// at tick creation.
    required: HashMap<TxHash, BTreeSet<ShardId>>,

    /// Remote shards whose provisions have been received, each with the
    /// environment anchor its bundle carried. Populated by
    /// [`absorb_provisions`]. The payer shard's entry is the clock and
    /// draw a non-payer participant executes under.
    received: HashMap<TxHash, BTreeMap<ShardId, SourceAnchor>>,

    /// The payer shard of each cross-shard transaction whose payer is
    /// remote, recorded at tick creation beside `required`. Resolves
    /// which `received` entry carries the transaction's environment
    /// without re-deriving topology at dispatch.
    payer_shards: HashMap<TxHash, ShardId>,

    /// Per-tx retention deadline = the latest `now + RETENTION_HORIZON`
    /// observed at any insert point that touches the tx. Past the
    /// deadline the tx is provably terminal everywhere — every shard
    /// has either committed an EC for it or its `validity_range` has
    /// expired and any tick that admitted it has timed out. Anchored
    /// on BFT-attested `committed_ts`, matching the sender-side
    /// deadline used by [`OutboundProvisionTracker`](hyperscale_provisions::OutboundProvisionTracker).
    deadlines: HashMap<TxHash, WeightedTimestamp>,

    /// Latest BFT-attested local-commit weighted timestamp seen via
    /// [`advance_clock`]. Drives deadline stamping deterministically
    /// across validators.
    now: WeightedTimestamp,
}

impl ProvisioningTracker {
    pub fn new() -> Self {
        Self {
            verified: HashMap::new(),
            required: HashMap::new(),
            received: HashMap::new(),
            payer_shards: HashMap::new(),
            deadlines: HashMap::new(),
            now: WeightedTimestamp::ZERO,
        }
    }

    /// Update the shard consensus-attested local-commit clock used for deadline
    /// stamping. Called once per `on_block_committed`. Monotone — out-of-order
    /// or stale calls are ignored.
    pub fn advance_clock(&mut self, now: WeightedTimestamp) {
        if now > self.now {
            self.now = now;
        }
    }

    /// Stamp `tx_hash` with a deadline of `self.now + RETENTION_HORIZON`,
    /// taking the latest of any existing entry. Idempotent re-stamping
    /// only ever extends the deadline forward, so a late-arriving
    /// provision never causes earlier eviction than its predecessor.
    fn stamp_deadline(&mut self, tx_hash: TxHash) {
        let deadline = self.now.plus(RETENTION_HORIZON);
        self.deadlines
            .entry(tx_hash)
            .and_modify(|d| {
                if deadline > *d {
                    *d = deadline;
                }
            })
            .or_insert(deadline);
    }

    // ─── Required / received ────────────────────────────────────────────

    /// Record the remote shards `tx_hash` needs provisions from. Overwrites
    /// any previous entry — callers set this once per tick creation.
    pub fn record_required(&mut self, tx_hash: TxHash, remote_shards: BTreeSet<ShardId>) {
        self.required.insert(tx_hash, remote_shards);
        self.stamp_deadline(tx_hash);
    }

    /// Record the remote payer shard of a cross-shard transaction, so
    /// dispatch can read the transaction's environment off the payer's
    /// bundle.
    pub fn record_payer_shard(&mut self, tx_hash: TxHash, payer_shard: ShardId) {
        self.payer_shards.insert(tx_hash, payer_shard);
    }

    /// Whether provisions from `shard` have been absorbed for `tx_hash`.
    /// For a transaction's payer shard this is the transaction commit
    /// proof held: absorption admits a bundle only against a
    /// commit-proven source header, committed into the local chain.
    #[must_use]
    pub fn has_received_from(&self, tx_hash: TxHash, shard: ShardId) -> bool {
        self.received
            .get(&tx_hash)
            .is_some_and(|received| received.contains_key(&shard))
    }

    /// The environment carried by the remote payer's bundle: the
    /// payer-shard committing block's parent-QC weighted timestamp and
    /// reveal chain. `None` when the payer is local (the tick
    /// block is the anchor) or the bundle has not been absorbed.
    #[must_use]
    pub fn payer_anchor(&self, tx_hash: TxHash) -> Option<SourceAnchor> {
        let payer = self.payer_shards.get(&tx_hash)?;
        self.received.get(&tx_hash)?.get(payer).copied()
    }

    /// Whether every required provision for `tx_hash` has been received.
    /// Returns `false` for txs with no recorded requirements (single-shard
    /// txs or txs we aren't tracking). A recorded empty requirement is
    /// immediately satisfied — the dependency-free cross-shard leg that
    /// dispatches without waiting.
    pub fn is_fully_provisioned(&self, tx_hash: TxHash) -> bool {
        self.required.get(&tx_hash).is_some_and(|required| {
            required.is_empty()
                || self.received.get(&tx_hash).is_some_and(|received| {
                    required.iter().all(|shard| received.contains_key(shard))
                })
        })
    }

    // ─── Batch absorption ───────────────────────────────────────────────

    /// Absorb a committed provisions. Appends each tx's entry list to the
    /// verified map (one entry list per source-shard contribution) and
    /// records `provisions.source_shard` under `received[tx_hash]`.
    ///
    /// Returns the `tx_hash`es touched — the caller uses these to compute
    /// which local ticks are affected and to drive the dispatch check.
    /// Preserves iteration order of `provisions.transactions` (callers sort
    /// batches upstream for determinism).
    pub fn absorb_provisions(&mut self, provisions: &Verified<Provisions>) -> Vec<TxHash> {
        let mut touched = Vec::with_capacity(provisions.transactions().len());
        let source_shard = provisions.source_shard();
        let anchor = SourceAnchor {
            clock: provisions.source_block_ts(),
        };
        for tx_entry in provisions.transactions() {
            let tx_hash = tx_entry.tx_hash;
            let entries = Arc::new(tx_entry.entries.clone());
            self.verified.entry(tx_hash).or_default().push(entries);
            self.received
                .entry(tx_hash)
                .or_default()
                .insert(source_shard, anchor);
            self.stamp_deadline(tx_hash);
            touched.push(tx_hash);
        }
        touched
    }

    // ─── Terminal cleanup ───────────────────────────────────────────────

    /// Drop all state for `tx_hash` across every owned map. Called when a
    /// finalization commits and the transaction reaches terminal state.
    pub fn remove_tx(&mut self, tx_hash: TxHash) {
        self.verified.remove(&tx_hash);
        self.required.remove(&tx_hash);
        self.received.remove(&tx_hash);
        self.payer_shards.remove(&tx_hash);
        self.deadlines.remove(&tx_hash);
    }

    /// Drop tracker state for txs whose retention horizon elapsed without
    /// reaching finalization. Past `now + RETENTION_HORIZON` from the
    /// latest insert touching the tx, the tx is provably terminal everywhere
    /// — every shard has either committed an EC for it or its
    /// `validity_range` has expired and any tick that admitted it has timed
    /// out — so no future local tick can still consume the verified
    /// provisions. Returns the number of txs swept.
    pub fn gc_stale_provisions(&mut self, now_ts: WeightedTimestamp) -> usize {
        let stale: Vec<TxHash> = self
            .deadlines
            .iter()
            .filter(|(_, deadline)| **deadline <= now_ts)
            .map(|(tx, _)| *tx)
            .collect();
        let count = stale.len();
        for tx in stale {
            self.remove_tx(tx);
        }
        count
    }

    // ─── Accessors ──────────────────────────────────────────────────────

    /// Verified provision entries for `tx_hash`, one slice element per
    /// source-shard contribution. Threaded into
    /// [`TickState::dispatch_if_ready`](crate::tick_state::TickState::dispatch_if_ready)
    /// so the tick can assemble cross-shard execution requests with a
    /// per-tx lookup against committed provisions.
    pub fn provisions_for(&self, tx_hash: TxHash) -> Option<&[Arc<Vec<SubstateEntry>>]> {
        self.verified.get(&tx_hash).map(Vec::as_slice)
    }

    pub fn verified_len(&self) -> usize {
        self.verified.len()
    }

    pub fn required_len(&self) -> usize {
        self.required.len()
    }

    pub fn received_len(&self) -> usize {
        self.received.len()
    }
}

#[cfg(test)]
impl ProvisioningTracker {
    /// Test-only door for seeding `verified` directly. Production code
    /// populates this map via [`Self::absorb_provisions`]; tests that only
    /// exercise the dispatch lookup don't need to construct full
    /// [`Provisions`](hyperscale_types::Provisions) batches.
    pub fn seed_provisions(&mut self, tx_hash: TxHash, entries: Vec<Arc<Vec<SubstateEntry>>>) {
        self.verified.insert(tx_hash, entries);
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::{BlockHeight, Hash, MerkleInclusionProof, ProvisionEntry};

    use super::*;

    fn shard(n: u64) -> ShardId {
        ShardId::leaf(2, n)
    }

    fn make_provisions_at(
        source: ShardId,
        block_height: BlockHeight,
        anchor: SourceAnchor,
        tx_hashes: Vec<TxHash>,
    ) -> Verified<Provisions> {
        let transactions: Vec<ProvisionEntry> = tx_hashes
            .into_iter()
            .map(|tx_hash| ProvisionEntry::new(tx_hash, vec![]))
            .collect();
        Verified::<Provisions>::new_unchecked_for_test(Provisions::new(
            source,
            ShardId::leaf(2, 0),
            block_height,
            anchor.clock,
            MerkleInclusionProof::dummy(),
            transactions,
        ))
    }

    fn make_provisions(
        source: ShardId,
        block_height: BlockHeight,
        tx_hashes: Vec<TxHash>,
    ) -> Verified<Provisions> {
        make_provisions_at(source, block_height, anchor(0), tx_hashes)
    }

    fn anchor(clock_ms: u64) -> SourceAnchor {
        SourceAnchor {
            clock: WeightedTimestamp::from_millis(clock_ms),
        }
    }

    #[test]
    fn fresh_tracker_reports_no_state() {
        let t = ProvisioningTracker::new();
        assert_eq!(t.verified_len(), 0);
        assert_eq!(t.required_len(), 0);
        assert_eq!(t.received_len(), 0);
        assert!(!t.is_fully_provisioned(TxHash::from(Hash::from_bytes(b"missing"))));
    }

    #[test]
    fn has_received_from_tracks_absorbed_sources() {
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));
        assert!(!t.has_received_from(tx, shard(1)));

        // An empty-entry bundle — the payer shard's engagement evidence
        // for a commutative leg — counts exactly like a state-carrying
        // one: absorption is the commitment axis.
        let batch = make_provisions(shard(1), BlockHeight::new(5), vec![tx]);
        t.absorb_provisions(&batch);
        assert!(t.has_received_from(tx, shard(1)));
        assert!(!t.has_received_from(tx, shard(2)));
    }

    #[test]
    fn is_fully_provisioned_requires_required_subset_of_received() {
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));
        t.record_required(tx, [shard(1), shard(2)].into_iter().collect());

        assert!(!t.is_fully_provisioned(tx));

        // Only shard 1 landed.
        let batch1 = make_provisions(shard(1), BlockHeight::new(5), vec![tx]);
        t.absorb_provisions(&batch1);
        assert!(!t.is_fully_provisioned(tx));

        // Shard 2 lands → fully provisioned.
        let batch2 = make_provisions(shard(2), BlockHeight::new(5), vec![tx]);
        t.absorb_provisions(&batch2);
        assert!(t.is_fully_provisioned(tx));
    }

    #[test]
    fn an_empty_requirement_is_immediately_satisfied() {
        // The dependency-free cross-shard leg: requirements recorded as
        // the empty set dispatch without any provision landing.
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"delta-only"));
        t.record_required(tx, BTreeSet::new());
        assert!(t.is_fully_provisioned(tx));
    }

    #[test]
    fn is_fully_provisioned_false_without_required_entry() {
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));
        // Absorbed provisions records `received[tx]` but there's no `required` —
        // the query must not report fully-provisioned just because
        // anything landed.
        let provisions = make_provisions(shard(1), BlockHeight::new(5), vec![tx]);
        t.absorb_provisions(&provisions);
        assert!(!t.is_fully_provisioned(tx));
    }

    #[test]
    fn absorb_provisions_returns_touched_tx_hashes_in_order() {
        let mut t = ProvisioningTracker::new();
        let tx_a = TxHash::from(Hash::from_bytes(b"a"));
        let tx_b = TxHash::from(Hash::from_bytes(b"b"));
        let provisions = make_provisions(shard(1), BlockHeight::new(5), vec![tx_a, tx_b]);
        let touched = t.absorb_provisions(&provisions);
        assert_eq!(touched, vec![tx_a, tx_b]);
    }

    #[test]
    fn absorb_provisions_populates_verified_and_received_maps() {
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));
        let provisions = make_provisions(shard(1), BlockHeight::new(5), vec![tx]);
        t.absorb_provisions(&provisions);

        assert_eq!(t.verified_len(), 1);
        assert_eq!(t.received_len(), 1);
        assert_eq!(
            t.verified.get(&tx).map_or(0, Vec::len),
            1,
            "one provision recorded per provisions entry"
        );
    }

    #[test]
    fn absorb_multiple_batches_for_same_tx_accumulates() {
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));
        t.absorb_provisions(&make_provisions(shard(1), BlockHeight::new(5), vec![tx]));
        t.absorb_provisions(&make_provisions(shard(2), BlockHeight::new(5), vec![tx]));

        // Two distinct source shards → two verified entry lists and two
        // received entries.
        assert_eq!(t.verified.get(&tx).map_or(0, Vec::len), 2);
        assert_eq!(
            t.received.get(&tx).map_or(0, BTreeMap::len),
            2,
            "received set contains both source shards"
        );
    }

    #[test]
    fn payer_anchor_reads_the_payer_bundles_clock_and_draw() {
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));
        t.record_required(tx, [shard(1), shard(2)].into_iter().collect());
        t.record_payer_shard(tx, shard(2));

        // A read-set owner's bundle lands first: no payer anchor yet.
        t.absorb_provisions(&make_provisions_at(
            shard(1),
            BlockHeight::new(5),
            anchor(7_000),
            vec![tx],
        ));
        assert_eq!(t.payer_anchor(tx), None);

        // The payer's bundle carries the committing block's clock and
        // reveal chain — the environment every participant executes the
        // transaction under.
        t.absorb_provisions(&make_provisions_at(
            shard(2),
            BlockHeight::new(9),
            anchor(9_500),
            vec![tx],
        ));
        assert_eq!(t.payer_anchor(tx), Some(anchor(9_500)));

        t.remove_tx(tx);
        assert_eq!(t.payer_anchor(tx), None);
    }

    #[test]
    fn remove_tx_drops_state_from_every_owned_map() {
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));
        t.record_required(tx, std::iter::once(shard(1)).collect());
        let provisions = make_provisions(shard(1), BlockHeight::new(5), vec![tx]);
        t.absorb_provisions(&provisions);
        assert!(t.is_fully_provisioned(tx));

        t.remove_tx(tx);

        assert!(!t.is_fully_provisioned(tx));
        assert_eq!(t.verified_len(), 0);
        assert_eq!(t.required_len(), 0);
        assert_eq!(t.received_len(), 0);
    }

    #[test]
    fn record_required_overwrites_existing_entry() {
        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));
        t.record_required(tx, std::iter::once(shard(1)).collect());
        // Re-record with a different requirement set.
        t.record_required(tx, [shard(1), shard(2)].into_iter().collect());
        assert_eq!(t.required.get(&tx).map_or(0, BTreeSet::len), 2);
    }

    #[test]
    fn gc_stale_provisions_evicts_past_horizon_and_keeps_fresh() {
        use hyperscale_types::RETENTION_HORIZON;

        let mut t = ProvisioningTracker::new();
        let tx_old = TxHash::from(Hash::from_bytes(b"old"));
        let tx_fresh = TxHash::from(Hash::from_bytes(b"fresh"));

        // Old tx absorbed at clock = ms(1_000).
        t.advance_clock(WeightedTimestamp::from_millis(1_000));
        t.record_required(tx_old, std::iter::once(shard(1)).collect());
        t.absorb_provisions(&make_provisions(
            shard(1),
            BlockHeight::new(5),
            vec![tx_old],
        ));

        // Fresh tx absorbed at clock = ms(60_000).
        t.advance_clock(WeightedTimestamp::from_millis(60_000));
        t.record_required(tx_fresh, std::iter::once(shard(1)).collect());
        t.absorb_provisions(&make_provisions(
            shard(1),
            BlockHeight::new(6),
            vec![tx_fresh],
        ));

        let horizon_ms = u64::try_from(RETENTION_HORIZON.as_millis()).unwrap_or(u64::MAX);
        // Past tx_old's deadline (1_000 + horizon) but not tx_fresh's
        // (60_000 + horizon).
        let now = WeightedTimestamp::from_millis(1_000 + horizon_ms + 1);
        assert!(now.as_millis() < 60_000 + horizon_ms);

        let evicted = t.gc_stale_provisions(now);
        assert_eq!(evicted, 1);

        assert!(!t.verified.contains_key(&tx_old));
        assert!(!t.received.contains_key(&tx_old));
        assert!(!t.required.contains_key(&tx_old));
        assert!(t.verified.contains_key(&tx_fresh));
        assert!(t.received.contains_key(&tx_fresh));
        assert!(t.required.contains_key(&tx_fresh));
    }

    #[test]
    fn gc_stale_provisions_late_insert_extends_deadline() {
        use hyperscale_types::RETENTION_HORIZON;

        let mut t = ProvisioningTracker::new();
        let tx = TxHash::from(Hash::from_bytes(b"tx"));

        // First insert at clock = ms(1_000) → deadline = 1_000 + horizon.
        t.advance_clock(WeightedTimestamp::from_millis(1_000));
        t.absorb_provisions(&make_provisions(shard(1), BlockHeight::new(5), vec![tx]));

        // Second insert at clock = ms(60_000) → deadline extended to
        // 60_000 + horizon.
        t.advance_clock(WeightedTimestamp::from_millis(60_000));
        t.absorb_provisions(&make_provisions(shard(2), BlockHeight::new(5), vec![tx]));

        let horizon_ms = u64::try_from(RETENTION_HORIZON.as_millis()).unwrap_or(u64::MAX);
        // Past the FIRST deadline but not the SECOND. Entry must survive.
        let now = WeightedTimestamp::from_millis(1_000 + horizon_ms + 1);
        assert_eq!(t.gc_stale_provisions(now), 0);
        assert!(t.verified.contains_key(&tx));
    }
}
