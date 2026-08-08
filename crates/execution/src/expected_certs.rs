//! Timeout-driven fallback detection for expected execution certificates.
//!
//! A remote shard's committed block header lists the cross-shard
//! transactions in that block. For each one we are party to we expect an
//! aggregated execution certificate carrying that shard's outcome within a
//! bounded window. If it doesn't land in time, we fall back to explicitly
//! fetching it from the source shard's committee.
//!
//! ## Key type
//!
//! Expectations and fulfilments are both keyed by `(source_shard, tx_hash)`
//! — the question actually being asked, which is per transaction. The
//! certificate's own identity is not what a requester holds: a header names
//! transactions, and which certificate the source shard puts each one in is
//! the source shard's business. One arriving certificate therefore fulfils
//! every expectation for the transactions it covers.
//!
//! ## Deadlines
//!
//! All deadlines anchor on the committing QC's `weighted_timestamp` (passed
//! in as `now_ts`), so the window is independent of local block production
//! rate and is identical across validators.
//!
//! - `EXEC_CERT_FALLBACK_TIMEOUT`: age at which the first fallback fetch fires.
//! - `EXEC_CERT_RETRY_INTERVAL`: cooldown between repeated fetches once the
//!   first has fired.
//!
//! ## Fulfilled-tombstone lifetime
//!
//! **Primary signal — state-based**: a fulfilled entry is dropped by
//! [`on_txs_terminated`](ExpectedCertTracker::on_txs_terminated), hooked
//! into `remove_finalization`. A transaction that has reached terminal
//! state locally needs no shard's outcome, so the whole entry goes at once
//! — footprint tracks in-flight work, not gossip windows.
//!
//! **Backstop — time-based**: each entry also carries a deadline
//! (`vote_anchor_ts + RETENTION_HORIZON`), pruned by
//! [`prune_fulfilled`](ExpectedCertTracker::prune_fulfilled). This
//! catches a specific late-arrival race: state-based drain runs at
//! `remove_finalization`, after which the wave is gone. If a
//! duplicate header then arrives within the gossip window, `register`
//! re-creates an expectation, the fallback fetch returns the EC,
//! `mark_fulfilled` re-creates the tombstone — but no future
//! `remove_finalization` will fire for those txs. The deadline evicts it.
//!
//! Retention pruning against the transactions still awaiting coverage is
//! orchestrated by the coordinator via
//! [`retain_if_tx_needed`](ExpectedCertTracker::retain_if_tx_needed),
//! because the tracker cannot see the wave set.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Duration;

use hyperscale_types::{ShardId, TxHash, WAVE_TIMEOUT, WeightedTimestamp};

/// How long to wait before the first fallback request. Anchored on the
/// committing QC's `weighted_timestamp_ms`, so the window stays meaningful
/// regardless of local block production rate. Sized comfortably below
/// `WAVE_TIMEOUT` so fallback fetches rescue missing ECs before the wave
/// aborts.
const EXEC_CERT_FALLBACK_TIMEOUT: Duration = Duration::from_secs(5);

/// Interval between repeated fallback requests for the same cert.
const EXEC_CERT_RETRY_INTERVAL: Duration = Duration::from_secs(10);

/// Grace window during which a freshly-registered expectation is retained
/// even when no local wave holds its transaction yet. Remote committed-block
/// headers can arrive ahead of the local block that commits the same
/// transaction; without this window, the registration is silently pruned by
/// `retain_if_tx_needed` and the EC never gets fetched. Sized to comfortably
/// exceed the worst-case lag between receiving the remote header and
/// committing the local block referencing the same cross-shard tx.
const EXPECTED_RETENTION_GRACE: Duration = WAVE_TIMEOUT;

/// One shard's outcome for one transaction — what an expectation is for.
type ExpectedCertKey = (ShardId, TxHash);

/// Per-expectation bookkeeping.
#[derive(Debug, Clone)]
struct ExpectedEntry {
    /// Local weighted timestamp when we first learned about this cert.
    discovered_at: WeightedTimestamp,
    /// Local weighted timestamp when we last sent a fallback request.
    /// `None` means never requested.
    last_requested_at: Option<WeightedTimestamp>,
}

/// Per-transaction record of which shards' outcomes have already arrived.
#[derive(Debug, Clone)]
struct FulfilledEntry {
    /// Shards whose certificate for this transaction we have ingested.
    /// Suppresses re-registration from a duplicate header.
    shards: BTreeSet<ShardId>,
    /// The latest `vote_anchor_ts + RETENTION_HORIZON` across the
    /// certificates recorded here. Backstop for the late-re-registration
    /// race documented in the module-level fulfilled-tombstone lifetime
    /// section.
    deadline: WeightedTimestamp,
}

pub struct ExpectedCertTracker {
    expected: HashMap<ExpectedCertKey, ExpectedEntry>,
    /// Keyed by transaction alone: a transaction reaching terminal state
    /// retires every shard's outcome for it at once, which is exactly what
    /// [`on_txs_terminated`](Self::on_txs_terminated) is told about.
    fulfilled: HashMap<TxHash, FulfilledEntry>,
}

impl ExpectedCertTracker {
    pub fn new() -> Self {
        Self {
            expected: HashMap::new(),
            fulfilled: HashMap::new(),
        }
    }

    /// Register an expected outcome from `source_shard` for `tx_hash`.
    ///
    /// Idempotent: re-registering an active expectation does not reset the
    /// discovery timestamp. Skipped entirely when that shard's outcome has
    /// already been ingested — guards against late-arriving duplicate
    /// headers re-opening a closed expectation.
    pub fn register(&mut self, source_shard: ShardId, tx_hash: TxHash, now_ts: WeightedTimestamp) {
        if self.is_fulfilled(source_shard, tx_hash) {
            return;
        }
        self.expected
            .entry((source_shard, tx_hash))
            .or_insert(ExpectedEntry {
                discovered_at: now_ts,
                last_requested_at: None,
            });
    }

    /// Record that `source_shard`'s certificate arrived, covering
    /// `tx_hashes` — the EC's `tx_outcomes`'s `tx_hash` set. Clears every
    /// expectation it answers for. `deadline` is the EC's own
    /// `vote_anchor_ts + RETENTION_HORIZON`, used as a backstop by
    /// [`prune_fulfilled`](Self::prune_fulfilled).
    ///
    /// Returns `true` if at least one active expectation was cleared.
    pub fn mark_fulfilled(
        &mut self,
        source_shard: ShardId,
        tx_hashes: impl IntoIterator<Item = TxHash>,
        deadline: WeightedTimestamp,
    ) -> bool {
        let mut cleared = false;
        for tx_hash in tx_hashes {
            cleared |= self.expected.remove(&(source_shard, tx_hash)).is_some();
            let entry = self.fulfilled.entry(tx_hash).or_insert(FulfilledEntry {
                shards: BTreeSet::new(),
                deadline,
            });
            entry.shards.insert(source_shard);
            // A second certificate for the same transaction extends the
            // backstop rather than shortening it: the tombstone has to
            // outlive every certificate it stands in for.
            entry.deadline = entry.deadline.max(deadline);
        }
        cleared
    }

    /// Drop the records for `tx_hashes` that just reached terminal state (a
    /// finalized local wave landed in a committed block). No shard's outcome
    /// for a terminal transaction is wanted anymore.
    pub fn on_txs_terminated(&mut self, tx_hashes: impl IntoIterator<Item = TxHash>) {
        for tx_hash in tx_hashes {
            self.fulfilled.remove(&tx_hash);
        }
    }

    /// Backstop sweep: drop fulfilled tombstones whose deadline has
    /// elapsed. Catches the late-re-registration race — see the
    /// module-level fulfilled-tombstone lifetime section.
    pub fn prune_fulfilled(&mut self, now_ts: WeightedTimestamp) {
        self.fulfilled.retain(|_, entry| entry.deadline > now_ts);
    }

    /// Drive the timeout state machine. Returns `(source_shard, tx_hash,
    /// is_retry)` for each expectation on a transaction in `txs_needed` that
    /// has crossed either the initial or the retry deadline at `now_ts`.
    /// Records `last_requested_at = now_ts` on each returned entry so the
    /// retry cooldown starts ticking.
    ///
    /// A source header names every cross-shard transaction in its block,
    /// including ones bound for other shards, so `txs_needed` is what keeps
    /// this from fetching other shards' business. The cadence is only
    /// stamped on entries actually returned: an expectation held back
    /// because our own block committing the transaction hasn't landed yet
    /// still gets its full initial window once it does.
    pub fn check_timeouts(
        &mut self,
        txs_needed: &HashSet<TxHash>,
        now_ts: WeightedTimestamp,
    ) -> Vec<(ShardId, TxHash, bool)> {
        let mut fetches = Vec::new();
        for (&(source_shard, tx_hash), entry) in &mut self.expected {
            if !txs_needed.contains(&tx_hash) {
                continue;
            }
            let should_request = match entry.last_requested_at {
                None => now_ts.elapsed_since(entry.discovered_at) >= EXEC_CERT_FALLBACK_TIMEOUT,
                Some(last) => now_ts.elapsed_since(last) >= EXEC_CERT_RETRY_INTERVAL,
            };
            if should_request {
                let is_retry = entry.last_requested_at.is_some();
                entry.last_requested_at = Some(now_ts);
                fetches.push((source_shard, tx_hash, is_retry));
            }
        }
        fetches
    }

    /// Eager-fetch every expectation that hasn't been requested yet,
    /// bypassing the timeout window. The timeout in [`Self::check_timeouts`]
    /// is measured against the committed-block weighted timestamp, which
    /// stops advancing while the shard is stalled on the very certs these
    /// fetches recover; a commit-independent caller flushes through here so
    /// the fallback still fires. Records `last_requested_at` so the `io_loop`
    /// owns retries from this point.
    pub fn flush_all(
        &mut self,
        txs_needed: &HashSet<TxHash>,
        now_ts: WeightedTimestamp,
    ) -> Vec<ExpectedCertKey> {
        let mut fetches = Vec::new();
        for (&key, entry) in &mut self.expected {
            if entry.last_requested_at.is_some() || !txs_needed.contains(&key.1) {
                continue;
            }
            entry.last_requested_at = Some(now_ts);
            fetches.push(key);
        }
        fetches
    }

    /// Drop expectations for transactions no outstanding local wave is
    /// waiting on. The coordinator computes the set from `WaveRegistry` and
    /// passes it in — the tracker has no view of waves.
    pub fn retain_if_tx_needed(&mut self, txs_needed: &HashSet<TxHash>, now_ts: WeightedTimestamp) {
        // Retain expectations whose transaction is still held by a local
        // wave OR whose registration is recent enough that the local block
        // committing it may not have landed yet. Without the grace window,
        // a remote header arriving slightly ahead of our own commit of the
        // same transaction is silently pruned and the EC never gets
        // fetched.
        self.expected.retain(|(_, tx_hash), entry| {
            txs_needed.contains(tx_hash)
                || now_ts.elapsed_since(entry.discovered_at) < EXPECTED_RETENTION_GRACE
        });
    }

    /// Drop every active expectation, returning its keys so the caller can
    /// abandon their in-flight fallback fetches. Used when the local chain
    /// terminates at a reshape boundary — no local wave can consume a
    /// fetched EC anymore. Fulfilled tombstones stay; they only suppress
    /// re-registration.
    pub fn drain_expected(&mut self) -> Vec<ExpectedCertKey> {
        self.expected.drain().map(|(key, _)| key).collect()
    }

    /// Retro-stamp `discovered_at == ZERO` entries with `now_ts`.
    /// Remote headers can register expectations before our first local
    /// commit; without this, every such entry would report a ~57-year age
    /// on the next commit and trigger a fallback fetch storm.
    pub fn retro_stamp_zero_timestamps(&mut self, now_ts: WeightedTimestamp) {
        for entry in self.expected.values_mut() {
            if entry.discovered_at == WeightedTimestamp::ZERO {
                entry.discovered_at = now_ts;
            }
        }
    }

    /// Whether `source_shard`'s outcome for `tx_hash` has already been
    /// ingested (a [`mark_fulfilled`](Self::mark_fulfilled) tombstone
    /// exists). Read when a terminated partner's settled set arrives, to
    /// decide which of the certificates it owes us to fetch.
    #[must_use]
    pub fn is_fulfilled(&self, source_shard: ShardId, tx_hash: TxHash) -> bool {
        self.fulfilled
            .get(&tx_hash)
            .is_some_and(|entry| entry.shards.contains(&source_shard))
    }

    pub fn expected_len(&self) -> usize {
        self.expected.len()
    }

    pub fn fulfilled_len(&self) -> usize {
        self.fulfilled.len()
    }
}

#[cfg(test)]
impl ExpectedCertTracker {
    fn is_expected(&self, source_shard: ShardId, tx_hash: TxHash) -> bool {
        self.expected.contains_key(&(source_shard, tx_hash))
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::Hash;
    use proptest::collection::vec as prop_vec;

    use super::*;

    fn shard(index: u8) -> ShardId {
        ShardId::leaf(2, u64::from(index))
    }

    fn tx(seed: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[seed]))
    }

    /// Every transaction is one a local wave awaits, unless a test says
    /// otherwise — the fetch gate is exercised on its own below.
    fn all() -> HashSet<TxHash> {
        (0u8..=255).map(tx).collect()
    }

    fn ms(value: u64) -> WeightedTimestamp {
        WeightedTimestamp::from_millis(value)
    }

    #[test]
    fn register_inserts_expectation_with_discovery_timestamp() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(1000));

        assert!(t.is_expected(shard(1), tx(5)));
        assert_eq!(t.expected_len(), 1);
    }

    #[test]
    fn drain_expected_empties_and_returns_keys() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(1000));
        t.register(shard(1), tx(6), ms(1000));

        let mut drained = t.drain_expected();
        drained.sort();
        let mut registered = vec![(shard(1), tx(5)), (shard(1), tx(6))];
        registered.sort();
        assert_eq!(drained, registered);
        assert_eq!(t.expected_len(), 0);
        assert!(!t.is_expected(shard(1), tx(5)));
    }

    #[test]
    fn register_is_idempotent_and_does_not_reset_discovery() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(1_000));
        // Re-register far later; the original discovery anchor must stand,
        // so the fallback still fires relative to the first sighting.
        t.register(shard(1), tx(5), ms(9_000));

        assert_eq!(t.expected_len(), 1);
        let fetches = t.check_timeouts(&all(), ms(6_000));
        assert_eq!(fetches.len(), 1);
    }

    #[test]
    fn register_skipped_when_already_fulfilled() {
        let mut t = ExpectedCertTracker::new();
        t.mark_fulfilled(shard(1), std::iter::once(tx(5)), ms(60_000));

        t.register(shard(1), tx(5), ms(1_000));

        assert!(!t.is_expected(shard(1), tx(5)));
        assert_eq!(t.expected_len(), 0);
    }

    #[test]
    fn mark_fulfilled_returns_true_when_clearing_active_expectation() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(1_000));

        assert!(t.mark_fulfilled(shard(1), std::iter::once(tx(5)), ms(60_000)));
        assert_eq!(t.expected_len(), 0);
        assert!(t.is_fulfilled(shard(1), tx(5)));
    }

    #[test]
    fn mark_fulfilled_returns_false_when_no_expectation_was_active() {
        let mut t = ExpectedCertTracker::new();
        assert!(!t.mark_fulfilled(shard(1), std::iter::once(tx(5)), ms(60_000)));
        assert!(t.is_fulfilled(shard(1), tx(5)));
    }

    /// One certificate covers every transaction of its batch, so it clears
    /// every expectation for those transactions at once. This is the whole
    /// reason the tracker keys by transaction rather than by certificate.
    #[test]
    fn one_certificate_fulfils_every_transaction_it_covers() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(1), ms(0));
        t.register(shard(1), tx(2), ms(0));
        t.register(shard(1), tx(3), ms(0));

        assert!(t.mark_fulfilled(shard(1), [tx(1), tx(2), tx(3)], ms(60_000)));

        assert_eq!(t.expected_len(), 0);
        assert!(t.check_timeouts(&all(), ms(50_000)).is_empty());
    }

    /// A transaction spanning two counterparts needs an outcome from each.
    /// One shard reporting leaves the other's expectation standing.
    #[test]
    fn fulfilment_is_per_shard() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));
        t.register(shard(2), tx(5), ms(0));

        t.mark_fulfilled(shard(1), std::iter::once(tx(5)), ms(60_000));

        assert!(t.is_fulfilled(shard(1), tx(5)));
        assert!(!t.is_fulfilled(shard(2), tx(5)));
        assert!(t.is_expected(shard(2), tx(5)));
        let fetches = t.check_timeouts(&all(), ms(5_000));
        assert_eq!(fetches, vec![(shard(2), tx(5), false)]);
    }

    /// A transaction that has reached terminal state locally needs no
    /// shard's outcome, so terminating it drops every shard's record at
    /// once rather than one per certificate.
    #[test]
    fn on_txs_terminated_drops_every_shards_record_for_the_transaction() {
        let mut t = ExpectedCertTracker::new();
        t.mark_fulfilled(shard(1), std::iter::once(tx(5)), ms(60_000));
        t.mark_fulfilled(shard(2), std::iter::once(tx(5)), ms(60_000));
        t.mark_fulfilled(shard(1), std::iter::once(tx(6)), ms(60_000));

        t.on_txs_terminated(std::iter::once(tx(5)));

        assert!(!t.is_fulfilled(shard(1), tx(5)));
        assert!(!t.is_fulfilled(shard(2), tx(5)));
        assert!(t.is_fulfilled(shard(1), tx(6)), "unrelated tx survives");
        assert_eq!(t.fulfilled_len(), 1);
    }

    /// A second certificate for the same transaction extends the backstop
    /// rather than shortening it — the record has to outlive every
    /// certificate it stands in for.
    #[test]
    fn a_second_certificate_extends_the_backstop() {
        let mut t = ExpectedCertTracker::new();
        t.mark_fulfilled(shard(1), std::iter::once(tx(5)), ms(70_000));
        t.mark_fulfilled(shard(2), std::iter::once(tx(5)), ms(10_000));

        t.prune_fulfilled(ms(60_000));
        assert_eq!(t.fulfilled_len(), 1, "the later deadline governs");

        t.prune_fulfilled(ms(70_000));
        assert_eq!(t.fulfilled_len(), 0);
    }

    /// A certificate with no outcomes answers for no transaction, so it
    /// clears no expectation and records nothing.
    #[test]
    fn mark_fulfilled_with_no_outcomes_records_nothing() {
        let mut t = ExpectedCertTracker::new();
        assert!(!t.mark_fulfilled(shard(1), std::iter::empty(), ms(60_000)));
        assert_eq!(t.fulfilled_len(), 0);
    }

    #[test]
    fn check_timeouts_fires_after_initial_window_and_records_request_ts() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));

        assert!(t.check_timeouts(&all(), ms(4_999)).is_empty());
        let fetches = t.check_timeouts(&all(), ms(5_000));
        assert_eq!(fetches, vec![(shard(1), tx(5), false)]);
        // The request timestamp is recorded, so an immediate re-poll is
        // inside the retry cooldown and emits nothing.
        assert!(t.check_timeouts(&all(), ms(5_001)).is_empty());
    }

    #[test]
    fn check_timeouts_respects_retry_interval_after_first_request() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));
        t.check_timeouts(&all(), ms(5_000));

        assert!(t.check_timeouts(&all(), ms(14_999)).is_empty());
        let fetches = t.check_timeouts(&all(), ms(15_000));
        assert_eq!(fetches, vec![(shard(1), tx(5), true)], "flagged as retry");
    }

    /// A transaction no local wave holds is not ours to fetch — a source
    /// header names every cross-shard transaction in its block, including
    /// ones bound elsewhere.
    #[test]
    fn check_timeouts_skips_a_transaction_no_local_wave_holds() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));

        assert!(t.check_timeouts(&HashSet::new(), ms(5_000)).is_empty());
    }

    /// Holding a fetch back must not consume its cadence: a transaction our
    /// own block commits after the source's header arrives still gets its
    /// full initial window, rather than dropping straight into the retry
    /// interval having never been requested.
    #[test]
    fn a_held_back_fetch_keeps_its_initial_window() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));

        // Past the initial deadline, but not yet ours.
        assert!(t.check_timeouts(&HashSet::new(), ms(5_000)).is_empty());

        // Our block commits it. The very next poll fires, and as an initial
        // request rather than a retry.
        let fetches = t.check_timeouts(&HashSet::from([tx(5)]), ms(5_001));
        assert_eq!(fetches, vec![(shard(1), tx(5), false)]);
    }

    #[test]
    fn check_timeouts_fires_at_exactly_the_fallback_deadline() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));
        let boundary = ms(u64::try_from(EXEC_CERT_FALLBACK_TIMEOUT.as_millis()).unwrap());
        assert_eq!(t.check_timeouts(&all(), boundary).len(), 1);
    }

    #[test]
    fn check_timeouts_can_mix_initial_and_retry_emissions_in_one_call() {
        let mut t = ExpectedCertTracker::new();
        let early = tx(1);
        let late = tx(2);
        t.register(shard(1), early, ms(0));
        // Fire the early one so it enters the retry cadence.
        t.check_timeouts(&all(), ms(5_000));
        // Register the second after that, so at t=15_000 the first is due
        // for a retry and the second for its initial request.
        t.register(shard(1), late, ms(9_000));

        let mut fetches = t.check_timeouts(&all(), ms(15_000));
        fetches.sort();
        let mut expected = vec![(shard(1), early, true), (shard(1), late, false)];
        expected.sort();
        assert_eq!(fetches, expected);
    }

    #[test]
    fn retain_if_tx_needed_drops_expectations_whose_tx_is_no_longer_tracked() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));

        // Past the grace window with the tx absent from every local wave.
        let past_grace = ms(u64::try_from(EXPECTED_RETENTION_GRACE.as_millis()).unwrap() + 1);
        t.retain_if_tx_needed(&HashSet::new(), past_grace);

        assert_eq!(t.expected_len(), 0);
    }

    #[test]
    fn retain_if_tx_needed_keeps_a_tx_a_local_wave_still_awaits() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));

        let past_grace = ms(u64::try_from(EXPECTED_RETENTION_GRACE.as_millis()).unwrap() + 1);
        t.retain_if_tx_needed(&HashSet::from([tx(5)]), past_grace);

        assert!(t.is_expected(shard(1), tx(5)));
    }

    /// The grace window covers a remote header arriving ahead of our own
    /// commit of the same transaction: it is not in any local wave yet, and
    /// pruning it would leave the certificate unfetched.
    #[test]
    fn retain_if_tx_needed_keeps_recent_unneeded_expectation() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));

        t.retain_if_tx_needed(&HashSet::new(), ms(1_000));

        assert!(t.is_expected(shard(1), tx(5)));
    }

    #[test]
    fn retain_if_tx_needed_prunes_at_exactly_grace_boundary() {
        let mut t = ExpectedCertTracker::new();
        t.register(shard(1), tx(5), ms(0));
        let boundary = ms(u64::try_from(EXPECTED_RETENTION_GRACE.as_millis()).unwrap());
        t.retain_if_tx_needed(&HashSet::new(), boundary);
        assert_eq!(t.expected_len(), 0);
    }

    #[test]
    fn prune_fulfilled_drops_entries_past_their_deadline() {
        let mut t = ExpectedCertTracker::new();
        t.mark_fulfilled(shard(1), std::iter::once(tx(1)), ms(1_000));
        t.mark_fulfilled(shard(1), std::iter::once(tx(2)), ms(60_000));

        t.prune_fulfilled(ms(2_000));

        assert!(!t.is_fulfilled(shard(1), tx(1)));
        assert!(t.is_fulfilled(shard(1), tx(2)));
    }

    #[test]
    fn prune_fulfilled_evicts_at_exactly_the_deadline() {
        // Deadline check is `deadline > now_ts` — strictly greater. At
        // equality the entry is dropped.
        let mut t = ExpectedCertTracker::new();
        t.mark_fulfilled(shard(1), std::iter::once(tx(1)), ms(1_000));
        t.prune_fulfilled(ms(1_000));
        assert_eq!(t.fulfilled_len(), 0);
    }

    #[test]
    fn register_succeeds_after_fulfilled_record_drains() {
        // Once on_txs_terminated drops a record, a duplicate header
        // arriving later is allowed to re-register the expectation. The
        // deadline backstop on `prune_fulfilled` exists precisely because
        // this re-registration path can recreate a record that no future
        // termination will drain.
        let mut t = ExpectedCertTracker::new();
        t.mark_fulfilled(shard(1), std::iter::once(tx(5)), ms(60_000));
        t.on_txs_terminated(std::iter::once(tx(5)));

        t.register(shard(1), tx(5), ms(70_000));

        assert!(t.is_expected(shard(1), tx(5)));
    }

    #[test]
    fn retro_stamp_updates_zero_entries_and_leaves_others_intact() {
        let mut t = ExpectedCertTracker::new();
        let zero_anchored = tx(1);
        let stamped = tx(2);
        t.register(shard(1), zero_anchored, WeightedTimestamp::ZERO);
        t.register(shard(1), stamped, ms(9_000));

        t.retro_stamp_zero_timestamps(ms(10_000));

        // Cross the stamped entry's deadline first (registered at 9_000,
        // deadline at 14_000) while the retro-stamped entry is still fresh
        // (its new anchor is 10_000, deadline 15_000).
        let fetches = t.check_timeouts(&all(), ms(14_000));
        assert_eq!(fetches, vec![(shard(1), stamped, false)]);

        // And finally cross the retro-stamped entry's deadline.
        let fetches = t.check_timeouts(&all(), ms(15_000));
        assert_eq!(fetches, vec![(shard(1), zero_anchored, false)]);
    }

    // ─── Property test ──────────────────────────────────────────────────

    use proptest::prelude::*;

    // For any sequence of register/fulfill events, a key that ends up in
    // the fulfilled set never produces a fallback fetch for itself on any
    // check_timeouts call after fulfillment.
    proptest! {
        #[test]
        fn fulfilled_before_deadline_never_triggers_fallback(
            seeds in prop_vec(0u8..20, 1..10),
            fulfill_indices in prop_vec(0usize..100, 0..10),
            timeouts in prop_vec(0u64..100_000, 1..10),
        ) {
            let mut t = ExpectedCertTracker::new();
            let source = shard(1);

            // Register expectations at t=0 so deadlines are all crossed
            // well before the latest poll time.
            let txs: Vec<TxHash> = seeds.iter().map(|s| tx(*s)).collect();
            for tx_hash in &txs {
                t.register(source, *tx_hash, ms(0));
            }

            // Fulfill a subset BEFORE any deadline could fire. Using a
            // 60s deadline keeps the records alive past every poll.
            for idx in &fulfill_indices {
                let tx_hash = txs[idx % txs.len()];
                t.mark_fulfilled(source, std::iter::once(tx_hash), ms(60_000));
            }

            // Run check_timeouts at a range of later timestamps.
            for now_ms in &timeouts {
                for (fetch_shard, fetch_tx, _) in t.check_timeouts(&all(), ms(*now_ms)) {
                    prop_assert!(
                        !t.is_fulfilled(fetch_shard, fetch_tx),
                        "fallback fetch emitted for a fulfilled key: {fetch_shard:?} {fetch_tx:?}"
                    );
                }
            }
        }
    }
}
