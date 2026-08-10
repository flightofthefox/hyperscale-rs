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
    Address, Finalization, MAX_FINALIZATION_DELAY, MAX_VALIDITY_RANGE, RETENTION_HORIZON, ShardId,
    ShardTrie, Transaction, TxHash, Verifiable, WeightedTimestamp,
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
    /// The frontier its committing block anchored at, which dates the
    /// question below: a shard that left before this never held the
    /// transaction, whatever its keyspace covers now.
    committed_ts: WeightedTimestamp,
    /// The owner prefixes it reaches outside this shard.
    ///
    /// Who was party to the transaction is a question about these and the
    /// trie of the moment, and the trie of the moment is not something a
    /// rebuild can recover — windows evict, and a shard that has since
    /// split answers for a keyspace it no longer owns. The prefixes are
    /// the transaction's own and its body outlives every window, so a
    /// rebuild reaches the same set from the same block at any distance.
    remote_prefixes: BTreeSet<Address>,
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
    /// alike, so every term is written once and a rebuilt ledger cannot
    /// disagree with the one it rebuilds. Each is a function of the
    /// transaction body and this shard's own identity, both of which
    /// outlive any window, so the two agree at any distance.
    ///
    /// Idempotent per transaction: a hash cannot commit twice within its
    /// own validity window, and re-registering one must not move the
    /// deadline it was admitted under.
    pub fn register_committed<'a>(
        &mut self,
        local_shard: ShardId,
        committed_ts: WeightedTimestamp,
        txs: impl IntoIterator<Item = &'a Arc<Verifiable<Transaction>>>,
    ) {
        for tx in txs {
            let owed = Owed {
                committed_ts,
                deadline: tx
                    .validity_range()
                    .end_timestamp_exclusive
                    .plus(MAX_FINALIZATION_DELAY),
                declared_work: tx.work(),
                remote_prefixes: tx
                    .routing()
                    .all_prefixes()
                    .into_iter()
                    .filter(|prefix| !ShardTrie::shard_owns_prefix(local_shard, *prefix))
                    .collect(),
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

    /// Whether a terminal is already recorded for `shard`.
    #[must_use]
    pub fn knows_terminal(&self, shard: ShardId) -> bool {
        self.departed.contains_key(&shard)
    }

    /// When the shard that held `prefix` when the transaction committed
    /// left, if it has.
    ///
    /// The earliest recorded terminal after the transaction's own commit:
    /// earlier ones belong to shards that were already gone and so never
    /// held it, and later ones to successors that never did either. `None`
    /// while the prefix is still owned by the shard that owned it then.
    fn departure_over(&self, owed: &Owed, prefix: Address) -> Option<WeightedTimestamp> {
        self.departed
            .iter()
            .filter(|(shard, _)| ShardTrie::shard_owns_prefix(**shard, prefix))
            .filter(|(_, cut)| **cut > owed.committed_ts)
            .map(|(_, cut)| *cut)
            .min()
    }

    /// The shards that could hold a certificate of ours for `tx_hash` —
    /// the ones a settlement of it would need, and so the ones the fence
    /// puts its question to.
    ///
    /// Each remote prefix contributes the shard that owns it under `trie`
    /// and, where the prefix changed hands after the transaction
    /// committed, the departed shard that owned it then. Empty for a
    /// transaction this ledger does not hold, and for one that never left
    /// this shard.
    #[must_use]
    pub fn counterparts(&self, tx_hash: TxHash, trie: &ShardTrie) -> BTreeSet<ShardId> {
        let Some(owed) = self.owed.get(&tx_hash) else {
            return BTreeSet::new();
        };
        let mut shards = BTreeSet::new();
        for prefix in &owed.remote_prefixes {
            shards.insert(trie.shard_for_prefix(*prefix));
            shards.extend(
                self.departed
                    .iter()
                    .filter(|(shard, cut)| {
                        ShardTrie::shard_owns_prefix(**shard, *prefix) && **cut > owed.committed_ts
                    })
                    .map(|(shard, _)| *shard),
            );
        }
        shards
    }

    /// Whether some shard that could hold a certificate of ours for
    /// `tx_hash` has left, which is what puts a settlement of it out of
    /// reach: the rest of its coverage will never arrive.
    #[must_use]
    pub fn a_counterpart_has_left(&self, tx_hash: TxHash) -> bool {
        self.owed.get(&tx_hash).is_some_and(|owed| {
            owed.remote_prefixes
                .iter()
                .any(|prefix| self.departure_over(owed, *prefix).is_some())
        })
    }

    /// Whether `tx_hash` reaches beyond this shard at all. A transaction
    /// that does not has no counterpart to hold a certificate of ours,
    /// whatever this shard has said about it.
    #[must_use]
    pub fn reaches_beyond(&self, tx_hash: TxHash) -> bool {
        self.owed
            .get(&tx_hash)
            .is_some_and(|owed| !owed.remote_prefixes.is_empty())
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

    /// The transactions a verdict of this shard's can still speak for,
    /// each with the reservation it holds.
    ///
    /// Bounded at both ends. It opens at the transaction's deadline, past
    /// which no shard can finalize it. It closes a `MAX_VALIDITY_RANGE`
    /// later, which is the room the shard has to get an abandonment
    /// committed — and, more than that, is what keeps a verdict from being
    /// composed on an event far from the transaction's own life.
    ///
    /// The upper bound is an atomicity property rather than bookkeeping.
    /// Composing an abandonment discards the tick holding the member, and
    /// the tick discarded is the one that would have settled it; a window
    /// reaching a counterpart's departure would spend settlements that had
    /// already closed. The entry outlives this window so its reservation
    /// stays accountable, and offering it again here is exactly what that
    /// must not cost.
    ///
    /// Read off committed content alone — the ledger is a fold over
    /// committed blocks and `now` is the committed weighted timestamp —
    /// so every replica at the same frontier names the same set, which is
    /// what lets a committee sign the abort it composes.
    #[must_use]
    pub fn past_deadline(&self, now: WeightedTimestamp) -> Vec<(TxHash, u64)> {
        self.owed
            .iter()
            .filter(|(_, owed)| {
                now >= owed.deadline && now < owed.deadline.plus(MAX_VALIDITY_RANGE)
            })
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
    ///
    /// Returns the transactions dropped because every counterpart has
    /// fallen silent — the ones whose fate is settled by nobody rather
    /// than decided by anybody. Their reservations are owed to a
    /// settlement that provably cannot arrive, and whatever this shard
    /// still holds against them is holding it for nothing.
    pub fn prune(&mut self, now: WeightedTimestamp) -> Vec<TxHash> {
        let mut unanswerable = Vec::new();
        let kept: BTreeMap<TxHash, Owed> = std::mem::take(&mut self.owed)
            .into_iter()
            .filter(|(tx_hash, owed)| {
                let answerable = owed.remote_prefixes.iter().any(|prefix| {
                    self.departure_over(owed, *prefix)
                        .is_none_or(|cut| now <= cut.plus(RETENTION_HORIZON))
                });
                // Having counterparts at all is what makes silence mean
                // something: a transaction that never left this shard has
                // nobody to fall silent, and its own deadline decides it
                // as it decides any other.
                if owed.certified && !owed.remote_prefixes.is_empty() {
                    if answerable {
                        return true;
                    }
                    // Our certificate is out there and no shard is left to
                    // combine it with.
                    unanswerable.push(*tx_hash);
                    return false;
                }
                owed.deadline.plus(MAX_VALIDITY_RANGE) > now
            })
            .collect();
        self.owed = kept;

        // A terminal is what tells a prefix's owner apart from its
        // successor, so one still covering a live entry stays: dropping
        // it would read the departed counterpart as the shard that holds
        // the keyspace now, and hold the entry open against a shard that
        // was never party to it.
        let owed = &self.owed;
        self.departed.retain(|shard, cut| {
            owed.values().any(|entry| {
                *cut > entry.committed_ts
                    && entry
                        .remote_prefixes
                        .iter()
                        .any(|prefix| ShardTrie::shard_owns_prefix(*shard, *prefix))
            })
        });

        unanswerable
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
        BlockHeight, TimestampRange, TransactionDecision, Verified, WeightedTimestamp,
    };

    use super::*;

    /// This shard owns the prefixes whose leading bit is zero, so an
    /// address is remote or local by its top byte and nothing else.
    const LOCAL: ShardId = ShardId::leaf(1, 0);
    const HERE: u8 = 0x11;
    const AWAY: u8 = 0xAA;

    /// The depth-1 shard owning every `AWAY`-topped prefix.
    const PARTNER: ShardId = ShardId::leaf(1, 1);

    /// A transaction paying from `payer` and touching `also`, both given
    /// as the byte an address repeats.
    fn tx_over(payer: u8, also: u8, end_ms: u64) -> Arc<Verifiable<Transaction>> {
        let validity = TimestampRange::new(
            WeightedTimestamp::ZERO,
            WeightedTimestamp::from_millis(end_ms),
        );
        Arc::new(Verifiable::from(Verified::new_unchecked_for_test(
            stub_transaction([payer; 16], &[[also; 16]], 1_000, validity),
        )))
    }

    /// A straddler: payer here, the rest of it away.
    fn tx(seed: u8, end_ms: u64) -> Arc<Verifiable<Transaction>> {
        tx_over(HERE, AWAY.wrapping_add(seed) | 0x80, end_ms)
    }

    fn ms(v: u64) -> WeightedTimestamp {
        WeightedTimestamp::from_millis(v)
    }

    fn commit(ledger: &mut UnresolvedTxs, tx: &Arc<Verifiable<Transaction>>) {
        ledger.register_committed(LOCAL, WeightedTimestamp::ZERO, std::iter::once(tx));
    }

    /// A committed transaction is owed an outcome from the moment its
    /// block commits until a committed block carries one.
    #[test]
    fn a_committed_transaction_is_owed_an_outcome_until_one_commits() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(1, 60_000);
        commit(&mut ledger, &tx);
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
        commit(&mut ledger, &tx);

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
        commit(&mut ledger, &tx);
        commit(&mut ledger, &tx);

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
        commit(&mut ledger, &tx);

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
        commit(&mut ledger, &tx);

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        ledger.prune(deadline);
        assert_eq!(ledger.len(), 1, "still the shard's to resolve");

        ledger.prune(deadline.plus(MAX_VALIDITY_RANGE));
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
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        ledger.prune(deadline.plus(MAX_VALIDITY_RANGE));
        assert_eq!(ledger.len(), 1, "the counterpart can still settle it");

        ledger.prune(ms(600_000));
        assert_eq!(
            ledger.len(),
            1,
            "and no clock of this shard's says otherwise"
        );
    }

    /// Outliving the deadline window is not the same as being abandonable
    /// through it. The entry stays so its reservation stays accountable;
    /// offering it to a verdict again would spend the tick that is still
    /// the transaction's best chance of settling.
    #[test]
    fn an_entry_outliving_the_deadline_window_is_no_longer_a_candidate() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(15, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        assert_eq!(
            ledger.past_deadline(deadline),
            vec![(tx.hash(), tx.work())],
            "at its deadline it is the shard's to speak for",
        );
        assert_eq!(
            ledger.past_deadline(
                deadline
                    .plus(MAX_VALIDITY_RANGE)
                    .minus(Duration::from_millis(1))
            ),
            vec![(tx.hash(), tx.work())],
            "and stays so for the room to get that committed",
        );

        let past = deadline.plus(MAX_VALIDITY_RANGE);
        assert!(
            ledger.past_deadline(past).is_empty(),
            "past the window no verdict of this shard's speaks for it",
        );
        ledger.prune(past);
        assert_eq!(ledger.len(), 1, "though the account still owes it");
    }

    /// It goes when the last counterpart that could have answered stops
    /// being able to: a departed shard's settled set reads for
    /// `RETENTION_HORIZON` past its terminal, and nothing decides the
    /// transaction after that.
    #[test]
    fn a_certified_straddler_goes_when_its_last_counterpart_falls_silent() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(9, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut);

        ledger.prune(cut.plus(RETENTION_HORIZON));
        assert_eq!(ledger.len(), 1, "the set still reads at the horizon");

        ledger.prune(cut.plus(RETENTION_HORIZON).plus(Duration::from_millis(1)));
        assert_eq!(ledger.len(), 0, "and never again past it");
    }

    /// The entry names itself on the way out, so its holder can let go of
    /// what it was keeping for a settlement that cannot arrive.
    #[test]
    fn a_strand_whose_counterparts_all_fell_silent_names_itself() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(16, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut);
        assert!(
            ledger.prune(cut.plus(RETENTION_HORIZON)).is_empty(),
            "while the set still reads, the strand is nobody's to release",
        );
        assert_eq!(
            ledger.prune(cut.plus(RETENTION_HORIZON).plus(Duration::from_millis(1))),
            vec![tx.hash()],
            "past it, nothing can settle it and the strand is named",
        );
    }

    /// A transaction that never left this shard has no counterpart to
    /// fall silent, so it is never named as a strand however long this
    /// shard has spoken for it — its own deadline ends it, as ever.
    #[test]
    fn a_local_transaction_is_never_a_silenced_strand() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx_over(HERE, HERE.wrapping_add(1), 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        assert!(
            ledger.prune(deadline).is_empty(),
            "not yet past its own window",
        );
        assert!(
            ledger.prune(deadline.plus(MAX_VALIDITY_RANGE)).is_empty(),
            "and gone at that window's end without ever being a strand",
        );
        assert_eq!(ledger.len(), 0);
    }

    /// One counterpart falling silent is not enough while another can
    /// still answer.
    #[test]
    fn a_straddler_waits_on_whichever_counterpart_can_still_answer() {
        let mut ledger = UnresolvedTxs::default();
        // Two remote prefixes under different depth-2 shards: `0b10…`
        // and `0b11…`.
        let tx = tx_over(HERE, 0xC0, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let cut = ms(500_000);
        ledger.record_terminal(ShardId::leaf(2, 2), cut);
        ledger.prune(cut.plus(RETENTION_HORIZON).plus(MAX_VALIDITY_RANGE));
        assert_eq!(ledger.len(), 1, "the other shard is still running");
    }

    /// A transaction that never left this shard is decided by its own
    /// deadline whatever this shard said about it: there is no counterpart
    /// holding a certificate of ours, so there is nobody to wait for.
    #[test]
    fn a_certificate_over_a_local_transaction_holds_nothing_open() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx_over(HERE, HERE.wrapping_add(1), 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());
        assert!(!ledger.reaches_beyond(tx.hash()), "nothing of it is remote");

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        ledger.prune(deadline.plus(MAX_VALIDITY_RANGE));
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

        commit(&mut ledger, &tx);
        assert!(!ledger.is_certified(tx.hash()), "committed says nothing");
        ledger.certify(tx.hash());
        assert!(ledger.is_certified(tx.hash()));
    }

    /// A counterpart is whoever owns the keyspace the transaction reaches
    /// into: the shard holding it now, and any that held it and left
    /// since the transaction committed.
    #[test]
    fn counterparts_name_the_shard_holding_the_keyspace_and_the_one_that_left_it() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(13, 60_000);
        commit(&mut ledger, &tx);

        let live = ShardTrie::uniform(1);
        assert_eq!(
            ledger.counterparts(tx.hash(), &live),
            BTreeSet::from([PARTNER]),
            "one shard owns the whole remote side",
        );
        assert!(!ledger.a_counterpart_has_left(tx.hash()));

        // The partner splits. Its keyspace passes to a child, and both
        // answer for the transaction — the child owns it now, the parent
        // held it when our certificate went out.
        ledger.record_terminal(PARTNER, ms(500_000));
        let split = ShardTrie::from_leaves([LOCAL, ShardId::leaf(2, 2), ShardId::leaf(2, 3)]);
        let after = ledger.counterparts(tx.hash(), &split);
        assert!(after.contains(&PARTNER), "the shard that held it then");
        assert_eq!(after.len(), 2, "and the one that holds it now");
        assert!(ledger.a_counterpart_has_left(tx.hash()));
    }

    /// A shard that left before the transaction committed never held it,
    /// whatever its keyspace covers now — so its terminal says nothing
    /// about this transaction's fate.
    #[test]
    fn a_terminal_older_than_the_transaction_is_not_its_counterpart_leaving() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(14, 60_000);
        ledger.register_committed(LOCAL, ms(600_000), std::iter::once(&tx));
        ledger.certify(tx.hash());
        ledger.record_terminal(PARTNER, ms(500_000));

        assert!(
            !ledger.a_counterpart_has_left(tx.hash()),
            "the shard owning the prefix at commit is the successor, still running",
        );
    }
}
