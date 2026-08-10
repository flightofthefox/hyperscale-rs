//! What this shard has committed and not yet resolved.
//!
//! A committed transaction is owed exactly one outcome, and until a
//! certificate carries that outcome the transaction is in flight: its fee
//! reservation is engaged, its work counts against the drain, and the
//! shards party to it are still expected to certify it. This is the list
//! of those, and it is a fold over committed blocks rather than a
//! projection of live execution state — entries insert when a block
//! commits a transaction and release when a committed block carries the
//! outcome resolving it, so every replica's ledger is identical at equal
//! committed frontiers.
//!
//! That is the whole reason it exists apart from [`TickRegistry`]. Tick
//! state is what a node is *working on*, and it does not survive a
//! restart: a shard whose replicas all came back cannot name what it
//! committed and never finished, so it can neither finish it nor abort
//! it. A ledger folded from the chain can be rebuilt from the chain.
//!
//! [`TickRegistry`]: crate::ticks::TickRegistry

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use hyperscale_types::{
    Finalization, MAX_FINALIZATION_DELAY, MAX_VALIDITY_RANGE, RETENTION_HORIZON, ShardId,
    Transaction, TxHash, Verifiable, WeightedTimestamp,
};

/// One committed transaction's outstanding account.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Owed {
    /// The moment past which the transaction can no longer finalize
    /// anywhere: the last block that could have included it, plus the
    /// longest a cross-shard transaction can take to finalize.
    deadline: WeightedTimestamp,
    /// The reservation its committing block took against the drain, held
    /// here because an abandonment has no execution to read it from and
    /// must release exactly what was taken.
    declared_work: u64,
    /// The shards party to it, as its committing block's topology
    /// assigned them.
    ///
    /// Held for the split-boundary fence, which asks a different question
    /// from coverage: not "whose verdict does this need" — an abort is
    /// dominant and needs nobody's — but "did a terminating shard settle
    /// this before it went". An abandonment carries no counterpart
    /// certificate to read that off, so it reads it here.
    ///
    /// A rebuilt entry carries them too: the replay re-drives the
    /// ordinary commit path over the stored blocks, so each transaction's
    /// participants resolve against the topology its own block anchored
    /// in, exactly as they did the first time.
    participants: BTreeSet<ShardId>,
    /// Whether a tick of this shard's took the transaction as a member.
    ///
    /// What it answers is whether a certificate of ours is out where a
    /// counterpart could settle against it, which is what decides whether
    /// this shard may speak for the transaction alone. The account is
    /// where it belongs rather than the tick that produced it: the
    /// certificate outlives the tick, and a shard that could not say
    /// whether it had issued one would have to assume it had.
    certified: bool,
}

/// Committed-but-unresolved transactions, each against its deadline and
/// the reservation it holds.
#[derive(Debug, Default)]
pub struct UnresolvedTxs {
    owed: BTreeMap<TxHash, Owed>,
    /// Where each departed participant's chain ended, for the entries
    /// whose fate only that shard's settled set can decide. Held against
    /// the schedule window that proves the terminal, which is retained on
    /// a frontier of its own.
    departed: BTreeMap<ShardId, WeightedTimestamp>,
}

impl UnresolvedTxs {
    /// Record what a committed block puts in flight.
    ///
    /// One entry point for a live commit and for a replay of the chain
    /// alike, so the deadline rule and the participant set are each
    /// written once and a rebuilt ledger cannot disagree with the one it
    /// rebuilds.
    ///
    /// Idempotent per transaction: a hash cannot commit twice within its
    /// own validity window, and re-registering one must not move the
    /// deadline it was admitted under.
    pub fn register_committed<'a>(
        &mut self,
        txs: impl IntoIterator<Item = (&'a Arc<Verifiable<Transaction>>, BTreeSet<ShardId>)>,
    ) {
        for (tx, participants) in txs {
            let owed = Owed {
                deadline: tx
                    .validity_range()
                    .end_timestamp_exclusive
                    .plus(MAX_FINALIZATION_DELAY),
                declared_work: tx.work(),
                participants,
                certified: false,
            };
            self.owed.entry(tx.hash()).or_insert(owed);
        }
    }

    /// Record that a tick of this shard's has taken `tx_hash` as a member,
    /// and so will speak for it in a certificate a counterpart can settle
    /// against.
    pub fn certify(&mut self, tx_hash: TxHash) {
        if let Some(owed) = self.owed.get_mut(&tx_hash) {
            owed.certified = true;
        }
    }

    /// Whether a certificate of this shard's covers `tx_hash` — the
    /// question that decides whether a verdict on it is this shard's alone
    /// to reach. False for a transaction this ledger does not hold, which
    /// is the same answer it gives for one no tick ever took.
    #[must_use]
    pub fn is_certified(&self, tx_hash: TxHash) -> bool {
        self.owed.get(&tx_hash).is_some_and(|owed| owed.certified)
    }

    /// Record where a departed participant's chain ended.
    ///
    /// Idempotent on the figure, which is a property of the schedule
    /// rather than of when this shard got around to reading it.
    pub fn record_terminal(&mut self, shard: ShardId, cut: WeightedTimestamp) {
        self.departed.entry(shard).or_insert(cut);
    }

    /// The participating shards, other than this one, no terminal is
    /// recorded for — every shard the caller still has to ask the schedule
    /// about.
    #[must_use]
    pub fn unstamped_participants(&self, local_shard: ShardId) -> BTreeSet<ShardId> {
        self.owed
            .values()
            .flat_map(|owed| owed.participants.iter().copied())
            .filter(|shard| *shard != local_shard && !self.departed.contains_key(shard))
            .collect()
    }

    /// The shards party to `tx_hash`, for the fence to ask its question
    /// about and for the abandonment path to ask whether any of them has
    /// left. Empty only for a transaction this ledger does not hold: a
    /// rebuilt entry carries its participants, since the replay re-drives
    /// the ordinary commit path.
    pub fn participants(&self, tx_hash: TxHash) -> impl Iterator<Item = ShardId> + '_ {
        self.owed
            .get(&tx_hash)
            .into_iter()
            .flat_map(|owed| owed.participants.iter().copied())
    }

    /// Drop what a committed block's finalizations resolve. Every verdict
    /// arrives this way — accepted, refused, or aborted — so one release
    /// path covers them all.
    pub fn release_resolved(&mut self, finalizations: &[Arc<Verifiable<Finalization>>]) {
        for finalization in finalizations {
            for tx_hash in finalization.tx_hashes() {
                self.owed.remove(&tx_hash);
            }
        }
    }

    /// The transactions this shard can no longer finalize, each with the
    /// reservation it still holds.
    ///
    /// Read off committed content alone — the ledger is a fold over
    /// committed blocks and `now` is the committed weighted timestamp —
    /// so every replica at the same frontier names the same set, which is
    /// what lets a committee sign the abort it composes.
    #[must_use]
    pub fn past_deadline(&self, now: WeightedTimestamp) -> Vec<(TxHash, u64)> {
        self.owed
            .iter()
            .filter(|(_, owed)| now >= owed.deadline)
            .map(|(tx_hash, owed)| (*tx_hash, owed.declared_work))
            .collect()
    }

    /// Drop the entries nothing can still decide.
    ///
    /// An entry lives as long as the question that decides it is still
    /// open, and there are two such questions. A transaction no
    /// certificate of ours covers is decided by its own deadline: nothing
    /// anywhere can settle it, so this shard abandons it, and the window
    /// from the deadline to `MAX_VALIDITY_RANGE` past it is the room to
    /// get that abandonment committed.
    ///
    /// One a certificate of ours does cover is decided by a participant's
    /// settled set, which reads for `RETENTION_HORIZON` past that shard's
    /// terminal and never again. So it lives while some participant can
    /// still answer — one still running, whose certificate can yet arrive
    /// or whose own terminal can yet let the set speak, or one departed
    /// within that window. A clock of this transaction's own has nothing
    /// to say about when its counterpart leaves, which is why one cannot
    /// be what ends the entry.
    pub fn prune(&mut self, now: WeightedTimestamp, local_shard: ShardId) {
        let departed = &self.departed;
        self.owed.retain(|_, owed| {
            let answerable = owed.participants.iter().any(|shard| {
                *shard != local_shard
                    && departed
                        .get(shard)
                        .is_none_or(|cut| now <= cut.plus(RETENTION_HORIZON))
            });
            if owed.certified && answerable {
                return true;
            }
            owed.deadline.plus(MAX_VALIDITY_RANGE) > now
        });

        let owed = &self.owed;
        self.departed.retain(|shard, _| {
            owed.values()
                .any(|entry| entry.participants.contains(shard))
        });
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.owed.len()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use hyperscale_types::test_utils::{make_finalization, stub_transaction};
    use hyperscale_types::{
        Address, BlockHeight, TimestampRange, TransactionDecision, Verified, WeightedTimestamp,
    };

    use super::*;

    const PAYER: [u8; 16] = [0xAA; 16];
    const LOCAL: ShardId = ShardId::leaf(1, 0);
    const PARTNER: ShardId = ShardId::leaf(1, 1);

    fn tx(seed: u8, end_ms: u64) -> Arc<Verifiable<Transaction>> {
        let validity = TimestampRange::new(
            WeightedTimestamp::ZERO,
            WeightedTimestamp::from_millis(end_ms),
        );
        Arc::new(Verifiable::from(Verified::new_unchecked_for_test(
            stub_transaction(PAYER, &[Address([seed; 16]).0], 1_000, validity),
        )))
    }

    fn ms(v: u64) -> WeightedTimestamp {
        WeightedTimestamp::from_millis(v)
    }

    /// A committed transaction is owed an outcome from the moment its
    /// block commits until a committed block carries one.
    #[test]
    fn a_committed_transaction_is_owed_an_outcome_until_one_commits() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(1, 60_000);
        ledger.register_committed(std::iter::once((&tx, BTreeSet::new())));
        assert_eq!(ledger.len(), 1);

        let resolved =
            make_finalization(BlockHeight::new(1), tx.hash(), TransactionDecision::Accept);
        ledger.release_resolved(&[Arc::new(Verifiable::from(resolved))]);
        assert_eq!(ledger.len(), 0);
    }

    /// Every verdict releases, not only acceptance — an abort resolves a
    /// transaction exactly as a settlement does, which is what lets one
    /// certificate answer for both.
    #[test]
    fn an_abort_releases_as_a_settlement_does() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(2, 60_000);
        ledger.register_committed(std::iter::once((&tx, BTreeSet::new())));

        let aborted =
            make_finalization(BlockHeight::new(1), tx.hash(), TransactionDecision::Aborted);
        ledger.release_resolved(&[Arc::new(Verifiable::from(aborted))]);
        assert_eq!(ledger.len(), 0);
    }

    /// Re-registering a transaction leaves the deadline it was admitted
    /// under alone: the fold has to be identical on a replica that sees
    /// the block once and one that replays it.
    #[test]
    fn re_registering_does_not_move_the_deadline() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(3, 60_000);
        ledger.register_committed(std::iter::once((&tx, BTreeSet::new())));
        ledger.register_committed(std::iter::once((&tx, BTreeSet::new())));

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        assert_eq!(ledger.len(), 1, "one entry, not two");
        assert!(
            ledger
                .past_deadline(deadline.minus(Duration::from_millis(1)))
                .is_empty(),
            "and the deadline is still the one it was admitted under",
        );
        assert_eq!(ledger.past_deadline(deadline), vec![(tx.hash(), tx.work())]);
    }

    /// A transaction becomes abandonable at its own deadline and not
    /// before, carrying the reservation its committing block took.
    #[test]
    fn a_transaction_is_abandonable_at_its_deadline_and_not_before() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(4, 60_000);
        ledger.register_committed(std::iter::once((&tx, BTreeSet::new())));

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        assert!(
            ledger
                .past_deadline(deadline.minus(Duration::from_millis(1)))
                .is_empty(),
            "merely slow is not abandonable",
        );
        assert_eq!(
            ledger.past_deadline(deadline),
            vec![(tx.hash(), tx.work())],
            "at the deadline, named with what it reserved",
        );
    }

    /// An entry survives its deadline: the window past it is the room the
    /// shard has to get the abort committed. It goes once no block could
    /// still reference the transaction at all.
    #[test]
    fn an_entry_outlives_its_deadline_and_not_the_retention_window() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(7, 60_000);
        ledger.register_committed(std::iter::once((&tx, BTreeSet::new())));

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        ledger.prune(deadline, LOCAL);
        assert_eq!(ledger.len(), 1, "still the shard's to resolve");

        ledger.prune(deadline.plus(MAX_VALIDITY_RANGE), LOCAL);
        assert_eq!(ledger.len(), 0, "past every window that could carry it");
    }

    /// A transaction this shard has spoken for is not this shard's to end
    /// on a clock. Its counterpart holds a certificate it can settle
    /// against for as long as it runs, so the entry outlives the window a
    /// deadline would have closed.
    #[test]
    fn a_certified_straddler_outlives_the_deadline_window() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(8, 60_000);
        ledger.register_committed(std::iter::once((&tx, BTreeSet::from([LOCAL, PARTNER]))));
        ledger.certify(tx.hash());

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        ledger.prune(deadline.plus(MAX_VALIDITY_RANGE), LOCAL);
        assert_eq!(ledger.len(), 1, "the counterpart can still settle it");

        ledger.prune(ms(600_000), LOCAL);
        assert_eq!(
            ledger.len(),
            1,
            "and no clock of this shard's says otherwise"
        );
    }

    /// It goes when the last participant that could have answered stops
    /// being able to: a departed shard's settled set reads for
    /// `RETENTION_HORIZON` past its terminal, and nothing decides the
    /// transaction after that.
    #[test]
    fn a_certified_straddler_goes_when_its_last_participant_falls_silent() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(9, 60_000);
        ledger.register_committed(std::iter::once((&tx, BTreeSet::from([LOCAL, PARTNER]))));
        ledger.certify(tx.hash());

        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut);

        ledger.prune(cut.plus(RETENTION_HORIZON), LOCAL);
        assert_eq!(ledger.len(), 1, "the set still reads at the horizon");

        ledger.prune(
            cut.plus(RETENTION_HORIZON).plus(Duration::from_millis(1)),
            LOCAL,
        );
        assert_eq!(ledger.len(), 0, "and never again past it");
    }

    /// One participant falling silent is not enough while another can
    /// still answer.
    #[test]
    fn a_straddler_waits_on_whichever_participant_can_still_answer() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(10, 60_000);
        let third = ShardId::leaf(2, 2);
        ledger.register_committed(std::iter::once((
            &tx,
            BTreeSet::from([LOCAL, PARTNER, third]),
        )));
        ledger.certify(tx.hash());

        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut);
        ledger.prune(cut.plus(RETENTION_HORIZON).plus(MAX_VALIDITY_RANGE), LOCAL);
        assert_eq!(ledger.len(), 1, "the third shard is still running");
    }

    /// A transaction that never left this shard is decided by its own
    /// deadline whatever this shard said about it: there is no counterpart
    /// holding a certificate of ours, so there is nobody to wait for.
    #[test]
    fn a_certificate_over_a_local_transaction_holds_nothing_open() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(11, 60_000);
        ledger.register_committed(std::iter::once((&tx, BTreeSet::from([LOCAL]))));
        ledger.certify(tx.hash());

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        ledger.prune(deadline.plus(MAX_VALIDITY_RANGE), LOCAL);
        assert_eq!(ledger.len(), 0, "nobody to wait for");
    }

    /// The certification is the account's, not the tick's: a transaction
    /// the ledger does not hold reads as uncertified, which is the answer
    /// that lets its holder abandon it.
    #[test]
    fn an_unheld_transaction_is_uncertified() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(12, 60_000);
        assert!(!ledger.is_certified(tx.hash()));

        ledger.register_committed(std::iter::once((&tx, BTreeSet::new())));
        assert!(!ledger.is_certified(tx.hash()), "committed says nothing");
        ledger.certify(tx.hash());
        assert!(ledger.is_certified(tx.hash()));
    }

    /// Only the participants nothing is recorded for, and never this
    /// shard: the caller asks the schedule about each, and this shard's
    /// own terminal is not a thing its ledger outlives.
    #[test]
    fn unstamped_participants_names_what_is_left_to_ask_about() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(13, 60_000);
        let third = ShardId::leaf(2, 2);
        ledger.register_committed(std::iter::once((
            &tx,
            BTreeSet::from([LOCAL, PARTNER, third]),
        )));

        assert_eq!(
            ledger.unstamped_participants(LOCAL),
            BTreeSet::from([PARTNER, third]),
        );
        ledger.record_terminal(PARTNER, ms(500_000));
        assert_eq!(
            ledger.unstamped_participants(LOCAL),
            BTreeSet::from([third])
        );
    }
}
