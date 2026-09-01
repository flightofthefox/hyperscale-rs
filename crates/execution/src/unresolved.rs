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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use hyperscale_types::{
    AbandonmentRecord, AbortCharge, Address, Finalization, MAX_VALIDITY_RANGE, ShardId, ShardTrie,
    Transaction, TxHash, Unsettleable, UnsettledTx, Verifiable, Verified, WeightedTimestamp,
};

/// One transaction the ledger will let a tick abandon, with everything
/// that abandonment states: it releases the reservation its committing
/// block took and settles the charge its class fixes, and neither is
/// readable from an execution it never had.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Abandonable {
    /// The transaction.
    pub tx_hash: TxHash,
    /// The reservation to return.
    pub declared_work: u64,
    /// The burn to settle, on the shard holding the vault it names.
    pub charge: AbortCharge,
}

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
    /// What an abort of it burns, held here for the same reason: an
    /// abandonment never reaches an engine, so the charge its verdict
    /// settles has to be readable without one.
    charge: AbortCharge,
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
    /// Whether this shard ran only a leg of the transaction: it froze
    /// divided with this shard outside the core set.
    ///
    /// A leg's tick attests it and its certificate settles alone, so the
    /// entry is never abandoned and its own finalization bears no verdict
    /// on the transaction. What the entry is for is the reclaim: it holds
    /// the terms a reclaim states, and it lives — on the transaction's
    /// own clock — exactly as long as the record cell it would take back.
    leg: bool,
    /// Whether a tick of this shard's has admitted the reclaim of a leg
    /// entry, so the finalization naming the hash next is the
    /// reclaim's. Meaningless off a leg entry.
    reclaim_admitted: bool,
    /// The departed shard a committed record says left this transaction
    /// unsettled.
    ///
    /// The evidence that nothing can settle it, in the one form that
    /// outlives the settled set it was read from. Where the deadline
    /// window says how long this shard may speak for a transaction on its
    /// own clock, this says it may speak whatever the clock reads: no
    /// counterpart is left to contradict it, and the chain says so.
    ///
    /// It also decides how long the entry lives. A covered entry is
    /// abandonable from the moment the record commits, so what it waits on
    /// is a block carrying the abort — and the departure that covered it
    /// is the one clock both the entry and the record are stated in.
    unsettled_by: Option<ShardId>,
}

/// Where a departed participant's chain ended, and how long what it left
/// behind can still be read.
#[derive(Debug, Clone, Copy)]
struct Departure {
    /// The terminal cut — what dates the departure against a
    /// transaction's own commit frontier.
    cut: WeightedTimestamp,
    /// The handoff-anchored terminal-evidence expiry, `None` while the
    /// beacon has not stamped the handoff complete. The entries this
    /// shard holds against the departed shard live to exactly here,
    /// because this is when its settled set stops answering: a shorter
    /// life would strand them before the evidence that decides them is
    /// even attested, and a longer one would hold them past any answer.
    /// An open window holds them — the coordinator re-stamps on every
    /// commit, so the expiry lands within a commit of the beacon's stamp.
    readable_until: Option<WeightedTimestamp>,
}

/// A transaction the ledger let go of without an outcome, because no
/// counterpart is left to reach one with.
///
/// The reservation it took against the drain is not returned — only a
/// committed certificate does that, and none is coming — so this is the
/// leak, and `covered_by_record` is what says which kind. Covered means a
/// record had established the abort was safe and the chain did not get one
/// committed before the entry ran out of counterparts. Uncovered means no
/// record ever named it: either the departed counterpart settled it before
/// it left, in which case its certificate was the only resolution there
/// was, or the evidence never arrived to write a record from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unanswerable {
    /// The transaction dropped.
    pub tx_hash: TxHash,
    /// Whether a committed record had named it unsettled by a departed
    /// counterpart.
    pub covered_by_record: bool,
}

/// What a leg entry keeps for the reclaim and the refusal mirror.
#[derive(Debug, Clone)]
struct Leg {
    body: Arc<Verified<Transaction>>,
    core: BTreeSet<ShardId>,
}

/// Committed-but-unresolved transactions, each against its deadline and
/// the reservation it holds.
#[derive(Debug, Default)]
pub struct UnresolvedTxs {
    owed: BTreeMap<TxHash, Owed>,
    /// What a leg entry keeps beside its account: the body, for the
    /// reclaim, which derives from the transaction's legs and crossings
    /// after the candidate pool let the body go; and the core set, which
    /// says whose refusal is the transaction's. Bounded by the entries'
    /// own horizon, and dropped with them.
    legs: HashMap<TxHash, Leg>,
    /// Where each departed participant's chain ended, for the entries
    /// whose fate only that shard's settled set can decide. Held against
    /// the schedule window that proves the terminal, which is retained on
    /// a frontier of its own.
    departed: BTreeMap<ShardId, Departure>,
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
            let UnsettledTx {
                deadline,
                declared_work,
                charge,
                ..
            } = UnsettledTx::for_transaction(tx);
            let owed = Owed {
                committed_ts,
                deadline,
                declared_work,
                charge,
                remote_prefixes: tx
                    .routing()
                    .all_prefixes()
                    .into_iter()
                    .filter(|prefix| !ShardTrie::shard_owns_prefix(local_shard, *prefix))
                    .collect(),
                certified: false,
                leg: false,
                reclaim_admitted: false,
                unsettled_by: None,
            };
            self.owed.entry(tx.hash()).or_insert(owed);
        }
    }

    /// Record that this shard runs only a leg of `tx_hash`.
    ///
    /// Read off the classification its committing block froze, at the
    /// same commit, so a rebuilt ledger marks the same entries: the
    /// freeze is a function of the block and the placement it committed
    /// under, and the replay re-freezes both.
    pub fn mark_leg(
        &mut self,
        tx_hash: TxHash,
        body: Arc<Verified<Transaction>>,
        core: BTreeSet<ShardId>,
    ) {
        if let Some(owed) = self.owed.get_mut(&tx_hash) {
            owed.leg = true;
            self.legs.insert(tx_hash, Leg { body, core });
        }
    }

    /// The core set of a leg entry — whose refusal is the transaction's.
    /// `None` for anything but a leg entry this ledger holds.
    #[must_use]
    pub fn leg_core(&self, tx_hash: TxHash) -> Option<&BTreeSet<ShardId>> {
        self.legs.get(&tx_hash).map(|leg| &leg.core)
    }

    /// The figures a record naming a leg entry restates, for one no
    /// record covers yet. `None` for anything else: a record is composed
    /// once per entry.
    #[must_use]
    pub fn unsettled_leg_figures(&self, tx_hash: TxHash) -> Option<UnsettledTx> {
        self.owed
            .get(&tx_hash)
            .filter(|owed| owed.leg && owed.unsettled_by.is_none())
            .map(|owed| UnsettledTx {
                tx_hash,
                deadline: owed.deadline,
                declared_work: owed.declared_work,
                charge: owed.charge,
            })
    }

    /// Whether this ledger still holds `tx_hash`.
    #[must_use]
    pub fn contains(&self, tx_hash: TxHash) -> bool {
        self.owed.contains_key(&tx_hash)
    }

    /// The leg entries a committed record has licensed a reclaim of and
    /// no tick has taken yet, each with the body the reclaim derives
    /// from.
    ///
    /// Read off committed content alone, like [`Self::past_deadline`],
    /// so every replica at the same frontier composes the same reclaims.
    /// Never a clock reading: a record is the only thing that puts an
    /// entry here, and a record carries evidence.
    #[must_use]
    pub fn reclaimable(&self) -> Vec<(TxHash, Arc<Verified<Transaction>>)> {
        self.owed
            .iter()
            .filter(|(_, owed)| owed.leg && !owed.reclaim_admitted && owed.unsettled_by.is_some())
            .filter_map(|(tx_hash, _)| Some((*tx_hash, Arc::clone(&self.legs.get(tx_hash)?.body))))
            .collect()
    }

    /// Record that a tick of this shard's has admitted the reclaim of
    /// `tx_hash`, so the finalization naming the hash next is the
    /// reclaim's and releases the entry.
    pub fn admit_reclaim(&mut self, tx_hash: TxHash) {
        if let Some(owed) = self.owed.get_mut(&tx_hash) {
            owed.reclaim_admitted = true;
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

    /// Take what a committed block says departed shards left unsettled.
    ///
    /// A record names transactions its shard did not settle before it
    /// went, which is what puts a settlement of them out of reach for
    /// good. A name this ledger holds is marked; one it does not is
    /// inserted from the record itself, which carries every term
    /// abandoning it takes.
    ///
    /// That insertion is what keeps a rebuild from falling short. The
    /// entry's life is its counterpart's clock while the replay window is
    /// measured in the transaction's, so a restart between the commit and
    /// the record's landing comes back holding the record and not the
    /// entry — and a replica that only marked would name a smaller
    /// abandonable set than its peers at the same frontier, and could not
    /// sign the tick they compose.
    ///
    /// A reconstructed entry is `certified`, because a record names
    /// nothing else, and reaches no prefixes: the questions that read them
    /// — who else is party, whether anyone can still settle — are the ones
    /// a covered entry is never asked, since the record has already
    /// answered them.
    ///
    /// Returns how many it reconstructed, which is how far short this
    /// replica's replay window fell.
    pub fn record_abandonment_records(&mut self, verdicts: &[AbandonmentRecord]) -> usize {
        let mut reconstructed = 0usize;
        for verdict in verdicts {
            for entry in verdict.unsettled() {
                if let Some(owed) = self.owed.get_mut(&entry.tx_hash) {
                    owed.unsettled_by = Some(verdict.shard());
                    continue;
                }
                reconstructed = reconstructed.saturating_add(1);
                self.owed.insert(
                    entry.tx_hash,
                    Owed {
                        deadline: entry.deadline,
                        declared_work: entry.declared_work,
                        charge: entry.charge,
                        // The record dates it no later than the moment its
                        // evidence was taken at, which is the one bound on
                        // its commit the record itself establishes.
                        committed_ts: verdict.evidence().moment(),
                        remote_prefixes: BTreeSet::new(),
                        certified: true,
                        // A leg entry lives inside the replay window —
                        // its horizon is the transaction's own — so a
                        // replay registers and marks it before any record
                        // naming it lands. A replica that still meets one
                        // first reads the mark off the arm: only a leg is
                        // ever refused or unclaimed, and an entry rebuilt
                        // that way has no body to reclaim with, so it
                        // waits out its horizon rather than abandoning
                        // what the record licenses a reclaim of.
                        leg: !matches!(verdict.evidence(), Unsettleable::Departed { .. }),
                        reclaim_admitted: false,
                        unsettled_by: Some(verdict.shard()),
                    },
                );
            }
        }
        reconstructed
    }

    /// The transactions this ledger still owes an outcome for that
    /// `shard` was party to, for a shard that left at `cut`, each with the
    /// terms a record naming it must state.
    ///
    /// Only certified ones: a transaction no certificate of ours covers
    /// is decided by its own deadline and needs no record to speak for
    /// it. Only ones committed before the cut, since a shard that had
    /// already gone was never party to what came after. And only ones no
    /// record covers yet, so a departure is answered once.
    #[must_use]
    pub fn outstanding_with(&self, shard: ShardId, cut: WeightedTimestamp) -> Vec<UnsettledTx> {
        self.owed
            .iter()
            .filter(|(_, owed)| {
                owed.certified && owed.unsettled_by.is_none() && cut > owed.committed_ts
            })
            .filter(|(_, owed)| {
                owed.remote_prefixes
                    .iter()
                    .any(|prefix| ShardTrie::shard_owns_prefix(shard, *prefix))
            })
            .map(|(tx_hash, owed)| UnsettledTx {
                tx_hash: *tx_hash,
                deadline: owed.deadline,
                declared_work: owed.declared_work,
                charge: owed.charge,
            })
            .collect()
    }

    /// Whether a committed record has established that `tx_hash` can
    /// never settle — the question the split-boundary fence otherwise
    /// puts to a settled set that expires.
    #[must_use]
    pub fn is_unsettled_by_departed(&self, tx_hash: TxHash) -> bool {
        self.owed
            .get(&tx_hash)
            .is_some_and(|owed| owed.unsettled_by.is_some())
    }

    /// Record where a departed participant's chain ended, and when what it
    /// left stops being readable.
    ///
    /// Idempotent on the cut, which is a property of the schedule rather
    /// than of when this shard got around to reading it. The expiry fills
    /// in when the caller learns it — the beacon stamps the handoff
    /// complete some epochs after the cut — and never moves once set.
    pub fn record_terminal(
        &mut self,
        shard: ShardId,
        cut: WeightedTimestamp,
        readable_until: Option<WeightedTimestamp>,
    ) {
        let entry = self.departed.entry(shard).or_insert(Departure {
            cut,
            readable_until,
        });
        if entry.readable_until.is_none() {
            entry.readable_until = readable_until;
        }
    }

    /// When the shard that held `prefix` when the transaction committed
    /// left, if it has.
    ///
    /// The earliest recorded terminal after the transaction's own commit:
    /// earlier ones belong to shards that were already gone and so never
    /// held it, and later ones to successors that never did either. `None`
    /// while the prefix is still owned by the shard that owned it then.
    fn departure_over(&self, owed: &Owed, prefix: Address) -> Option<Departure> {
        self.departed
            .iter()
            .filter(|(shard, _)| ShardTrie::shard_owns_prefix(**shard, prefix))
            .filter(|(_, departure)| departure.cut > owed.committed_ts)
            .map(|(_, departure)| *departure)
            .min_by_key(|departure| departure.cut)
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
                    .filter(|(shard, departure)| {
                        ShardTrie::shard_owns_prefix(**shard, *prefix)
                            && departure.cut > owed.committed_ts
                    })
                    .map(|(shard, _)| *shard),
            );
        }
        shards
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
    ///
    /// A leg's own finalization bears no verdict on the transaction and
    /// releases nothing: the entry stays for the reclaim, whose
    /// finalization is the one that releases it, and otherwise lives to
    /// its own horizon.
    pub fn release_resolved(&mut self, finalizations: &[Arc<Verifiable<Finalization>>]) {
        for finalization in finalizations {
            for tx_hash in finalization.tx_hashes() {
                let bears_verdict = self
                    .owed
                    .get(&tx_hash)
                    .is_some_and(|owed| !owed.leg || owed.reclaim_admitted);
                if bears_verdict {
                    self.owed.remove(&tx_hash);
                    self.legs.remove(&tx_hash);
                }
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
    ///
    /// A leg entry is never here: its tick attested it and its
    /// certificate settled alone, so there is nothing to abandon. What a
    /// record licenses on one is a reclaim.
    #[must_use]
    pub fn past_deadline(&self, now: WeightedTimestamp) -> Vec<Abandonable> {
        self.owed
            .iter()
            .filter(|(_, owed)| !owed.leg)
            .filter(|(_, owed)| {
                now >= owed.deadline
                    && (owed.unsettled_by.is_some() || now < owed.deadline.plus(MAX_VALIDITY_RANGE))
            })
            .map(|(tx_hash, owed)| Abandonable {
                tx_hash: *tx_hash,
                declared_work: owed.declared_work,
                charge: owed.charge,
            })
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
    /// settled set, which reads to that shard's terminal-evidence expiry
    /// and never again. So it lives while some participant can
    /// still answer — one still running, whose certificate can yet arrive
    /// or whose own terminal can yet let the set speak, or one departed
    /// within that window. A clock of this transaction's own has nothing
    /// to say about when its counterpart leaves, which is why one cannot
    /// be what ends the entry.
    ///
    /// One a committed record already decided waits on nothing but a block
    /// carrying its abort, so what it lives against is the departure the
    /// record names rather than any counterpart's answerability. That is
    /// the same window the record was composed in, it is the one an entry
    /// reconstructed from a record has, and it is finite where a live
    /// counterpart's is not — which is what lets a replay floor reach
    /// every record still owed a verdict.
    ///
    /// A leg entry has a clock of its own, and it is the transaction's:
    /// `deadline + MAX_VALIDITY_RANGE`, which is the validity end plus the
    /// retention horizon — the moment the record cell it would reclaim
    /// sweeps. Past it there is nothing to take back, whatever evidence
    /// arrives, so the entry goes on that reading alone. Dropping gives a
    /// reclaim up; it never licenses one. A record's evidence does not
    /// extend it, and no counterpart's silence shortens it: a reclaim
    /// waits on a record, and the record's arms are the evidence, not the
    /// counterpart's answerability.
    ///
    /// Returns the transactions dropped because every counterpart has
    /// fallen silent. Each carries whether a committed record had covered
    /// it, which separates a chain that ran out of room to commit the abort
    /// from one that never had the evidence to compose it. A leg entry
    /// dropped at its horizon is not among them: its reservation came
    /// back with its own finalization, so nothing leaks with it.
    pub fn prune(&mut self, now: WeightedTimestamp) -> Vec<Unanswerable> {
        let mut unanswerable = Vec::new();
        let kept: BTreeMap<TxHash, Owed> = std::mem::take(&mut self.owed)
            .into_iter()
            .filter(|(tx_hash, owed)| {
                if owed.leg {
                    return owed.deadline.plus(MAX_VALIDITY_RANGE) > now;
                }
                if let Some(shard) = owed.unsettled_by {
                    if self.departed.get(&shard).is_some_and(|departure| {
                        departure.readable_until.is_none_or(|until| now <= until)
                    }) {
                        return true;
                    }
                    unanswerable.push(Unanswerable {
                        tx_hash: *tx_hash,
                        covered_by_record: true,
                    });
                    return false;
                }
                let answerable = owed.remote_prefixes.iter().any(|prefix| {
                    self.departure_over(owed, *prefix).is_none_or(|departure| {
                        departure.readable_until.is_none_or(|until| now <= until)
                    })
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
                    unanswerable.push(Unanswerable {
                        tx_hash: *tx_hash,
                        covered_by_record: false,
                    });
                    return false;
                }
                owed.deadline.plus(MAX_VALIDITY_RANGE) > now
            })
            .collect();
        self.owed = kept;
        let owed = &self.owed;
        self.legs.retain(|tx_hash, _| owed.contains_key(tx_hash));

        // A terminal is what tells a prefix's owner apart from its
        // successor, so one still covering a live entry stays: dropping
        // it would read the departed counterpart as the shard that holds
        // the keyspace now, and hold the entry open against a shard that
        // was never party to it. One a record names stays for a second
        // reason — it is the clock the covered entry lives against, so
        // dropping it would retire the entry on the next pass.
        let owed = &self.owed;
        self.departed.retain(|shard, departure| {
            owed.values().any(|entry| {
                entry.unsettled_by == Some(*shard)
                    || (departure.cut > entry.committed_ts
                        && entry
                            .remote_prefixes
                            .iter()
                            .any(|prefix| ShardTrie::shard_owns_prefix(*shard, *prefix)))
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

    use hyperscale_types::test_utils::{
        make_finalization, stub_transaction, test_prefix, test_principal,
    };
    use hyperscale_types::{
        BlockHeight, EPOCH_DURATION, EpochWindows, MAX_FINALIZATION_DELAY, TimestampRange,
        TransactionDecision, UnsettledTx, Verified, WeightedTimestamp,
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
            stub_transaction(test_principal(payer), &[test_prefix(also)], 1_000, validity),
        )))
    }

    /// A straddler: payer here, the rest of it away.
    fn tx(seed: u8, end_ms: u64) -> Arc<Verifiable<Transaction>> {
        tx_over(HERE, AWAY.wrapping_add(seed) | 0x80, end_ms)
    }

    fn ms(v: u64) -> WeightedTimestamp {
        WeightedTimestamp::from_millis(v)
    }

    /// When a departure at `cut` stops answering, on the production epoch
    /// grid — the same derivation the commit path stamps departures with.
    fn expiry(cut: WeightedTimestamp) -> WeightedTimestamp {
        EpochWindows::new(EPOCH_DURATION.as_secs() * 1000).terminal_evidence_expiry(cut)
    }

    /// The body a leg entry keeps for its reclaim.
    fn body(tx: &Arc<Verifiable<Transaction>>) -> Arc<Verified<Transaction>> {
        Arc::new(Verified::new_unchecked_for_test(tx.as_unverified().clone()))
    }

    fn commit(ledger: &mut UnresolvedTxs, tx: &Arc<Verifiable<Transaction>>) {
        ledger.register_committed(LOCAL, WeightedTimestamp::ZERO, std::iter::once(tx));
    }

    /// A record's name for `tx`, stating the terms a committing block
    /// would have registered for it.
    fn names(tx: &Arc<Verifiable<Transaction>>) -> UnsettledTx {
        UnsettledTx {
            tx_hash: tx.hash(),
            deadline: tx
                .validity_range()
                .end_timestamp_exclusive
                .plus(MAX_FINALIZATION_DELAY),
            declared_work: tx.work(),
            charge: charge(tx),
        }
    }

    /// What abandoning `tx` states: its reservation and its floor.
    fn abandons(tx: &Arc<Verifiable<Transaction>>) -> Abandonable {
        Abandonable {
            tx_hash: tx.hash(),
            declared_work: tx.work(),
            charge: charge(tx),
        }
    }

    /// The burn an abort of `tx` settles.
    fn charge(tx: &Arc<Verifiable<Transaction>>) -> AbortCharge {
        UnsettledTx::for_transaction(tx).charge
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
        assert_eq!(ledger.past_deadline(deadline), vec![abandons(&tx)]);
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
            vec![abandons(&tx)],
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
            vec![abandons(&tx)],
            "at its deadline it is the shard's to speak for",
        );
        assert_eq!(
            ledger.past_deadline(
                deadline
                    .plus(MAX_VALIDITY_RANGE)
                    .minus(Duration::from_millis(1))
            ),
            vec![abandons(&tx)],
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
    /// being able to: a departed shard's settled set reads to its
    /// evidence expiry, and nothing decides the transaction after that.
    #[test]
    fn a_certified_straddler_goes_when_its_last_counterpart_falls_silent() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(9, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut, Some(expiry(cut)));

        ledger.prune(expiry(cut));
        assert_eq!(ledger.len(), 1, "the set still reads at the expiry");

        ledger.prune(expiry(cut).plus(Duration::from_millis(1)));
        assert_eq!(ledger.len(), 0, "and never again past it");
    }

    /// The entry names itself on the way out, so its holder can let go of
    /// what it was keeping for a settlement that cannot arrive — and says
    /// whether a record had covered it, which separates a chain that ran
    /// out of room to commit the abort from one that never had the
    /// evidence to compose it.
    #[test]
    fn a_strand_whose_counterparts_all_fell_silent_names_itself() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(16, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut, Some(expiry(cut)));
        assert!(
            ledger.prune(expiry(cut)).is_empty(),
            "while the set still reads, the strand is nobody's to release",
        );
        assert_eq!(
            ledger.prune(expiry(cut).plus(Duration::from_millis(1))),
            vec![Unanswerable {
                tx_hash: tx.hash(),
                covered_by_record: false,
            }],
            "past it, nothing can settle it and the strand is named",
        );
    }

    /// A strand a record had covered is dropped for the same reason but is
    /// a different failure: the abort was licensed and the chain never got
    /// one committed. Only one of the two says the evidence path is
    /// working, so the holder has to be able to tell them apart.
    #[test]
    fn a_covered_strand_names_the_record_that_licensed_its_abort() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(18, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut, Some(expiry(cut)));
        assert_eq!(
            ledger.record_abandonment_records(&[AbandonmentRecord::departed(
                PARTNER,
                cut,
                [names(&tx)]
            )]),
            0,
            "the ledger holds the transaction the record names",
        );

        assert_eq!(
            ledger.prune(expiry(cut).plus(Duration::from_millis(1))),
            vec![Unanswerable {
                tx_hash: tx.hash(),
                covered_by_record: true,
            }],
        );
    }

    /// A record naming what the ledger does not hold rebuilds it. That is
    /// a rebuild that came back holding the record and not the entry, and
    /// what the record carries is exactly what the entry would have said,
    /// so the replica reaches the same abandonable set as its peers rather
    /// than a smaller one.
    #[test]
    fn a_record_naming_an_unheld_transaction_rebuilds_it() {
        let mut ledger = UnresolvedTxs::default();
        let (one, two) = (tx(19, 60_000), tx(20, 60_000));
        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut, Some(expiry(cut)));

        assert_eq!(
            ledger.record_abandonment_records(&[AbandonmentRecord::departed(
                PARTNER,
                cut,
                [names(&one), names(&two)],
            )]),
            2,
            "neither was held, so both are rebuilt",
        );
        assert_eq!(ledger.len(), 2);
        assert!(ledger.is_unsettled_by_departed(one.hash()));

        // And each is abandonable on the record's own terms, which is the
        // whole point of it carrying them.
        let past = ms(60_000)
            .plus(MAX_FINALIZATION_DELAY)
            .plus(MAX_VALIDITY_RANGE);
        let mut offered = ledger.past_deadline(past);
        offered.sort_unstable_by_key(|entry| entry.tx_hash);
        let mut expected = vec![abandons(&one), abandons(&two)];
        expected.sort_unstable_by_key(|entry| entry.tx_hash);
        assert_eq!(offered, expected);
    }

    /// A rebuilt entry lives against the departure that named it, not
    /// against a transaction clock it has nothing to say about. It waits
    /// on a block carrying its abort, and stops when the departure it was
    /// written against stops answering.
    #[test]
    fn a_rebuilt_entry_lives_against_the_departure_that_named_it() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(21, 60_000);
        let cut = ms(500_000);
        ledger.record_terminal(PARTNER, cut, Some(expiry(cut)));
        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            cut,
            [names(&tx)],
        )]);

        assert!(
            ledger.prune(cut).is_empty(),
            "the transaction's own deadline is long past, and decides nothing here",
        );
        assert_eq!(ledger.len(), 1);

        assert_eq!(
            ledger.prune(expiry(cut).plus(Duration::from_millis(1))),
            vec![Unanswerable {
                tx_hash: tx.hash(),
                covered_by_record: true,
            }],
        );
    }

    /// A committed record is what lets a verdict outlive the deadline
    /// window. Past that window the shard stops speaking for a
    /// transaction on its own clock; a record says no counterpart is left
    /// to contradict it, and the chain is where that is written.
    #[test]
    fn a_record_reopens_the_window_a_deadline_closed() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(17, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        let past = deadline.plus(MAX_VALIDITY_RANGE);
        assert!(
            ledger.past_deadline(past).is_empty(),
            "on its own clock the shard has stopped speaking for it",
        );

        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            ms(500_000),
            [names(&tx)],
        )]);
        assert_eq!(
            ledger.past_deadline(past),
            vec![abandons(&tx)],
            "the record says nothing can settle it, so the shard may",
        );
        assert!(ledger.is_unsettled_by_departed(tx.hash()));
    }

    /// A record still opens nothing before the transaction's own
    /// deadline: until then it may yet finalize somewhere, and the record
    /// speaks only to what a departed shard did.
    #[test]
    fn a_record_does_not_reach_back_before_the_deadline() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(18, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());
        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            ms(500_000),
            [names(&tx)],
        )]);

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        assert!(
            ledger
                .past_deadline(deadline.minus(Duration::from_millis(1)))
                .is_empty(),
            "merely covered is not yet abandonable",
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
        ledger.record_terminal(ShardId::leaf(2, 2), cut, Some(expiry(cut)));
        ledger.prune(expiry(cut).plus(MAX_VALIDITY_RANGE));
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

        // The partner splits. Its keyspace passes to a child, and both
        // answer for the transaction — the child owns it now, the parent
        // held it when our certificate went out.
        ledger.record_terminal(PARTNER, ms(500_000), Some(expiry(ms(500_000))));
        let split = ShardTrie::from_leaves([LOCAL, ShardId::leaf(2, 2), ShardId::leaf(2, 3)]);
        let after = ledger.counterparts(tx.hash(), &split);
        assert!(after.contains(&PARTNER), "the shard that held it then");
        assert_eq!(after.len(), 2, "and the one that holds it now");
    }

    /// A shard that left before the transaction committed never held it,
    /// whatever its keyspace covers now — so its terminal says nothing
    /// about this transaction's fate, and cannot be the silence that
    /// strands it.
    #[test]
    fn a_terminal_older_than_the_transaction_is_not_its_counterpart_leaving() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(14, 60_000);
        ledger.register_committed(LOCAL, ms(600_000), std::iter::once(&tx));
        ledger.certify(tx.hash());
        let stale = ms(500_000);
        ledger.record_terminal(PARTNER, stale, Some(expiry(stale)));

        assert!(
            ledger
                .prune(expiry(stale).plus(MAX_VALIDITY_RANGE))
                .is_empty(),
            "the shard owning the prefix at commit is the successor, still running",
        );
        assert_eq!(ledger.len(), 1, "so nothing has fallen silent on it");
    }

    /// A leg's own finalization bears no verdict on the transaction, so
    /// it releases nothing, and the entry is never abandoned — not at its
    /// deadline, and not when a committed record names it.
    #[test]
    fn a_leg_entry_outlives_its_own_finalization_and_is_never_abandoned() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(4, 60_000);
        commit(&mut ledger, &tx);
        ledger.mark_leg(tx.hash(), body(&tx), BTreeSet::from([PARTNER]));
        ledger.certify(tx.hash());

        let own = make_finalization(BlockHeight::new(1), tx.hash(), TransactionDecision::Accept);
        ledger.release_resolved(&[Arc::new(Verifiable::from(own))]);
        assert_eq!(ledger.len(), 1, "the leg's finalization decides nothing");

        let past = ms(60_000)
            .plus(MAX_FINALIZATION_DELAY)
            .plus(Duration::from_secs(1));
        assert!(
            ledger.past_deadline(past).is_empty(),
            "a leg is never abandoned"
        );

        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            ms(1_000),
            [names(&tx)],
        )]);
        assert!(
            ledger.past_deadline(past).is_empty(),
            "a record licenses a reclaim of it, never an abort"
        );
        assert_eq!(ledger.len(), 1);
    }

    /// A committed record is what makes a leg entry reclaimable — never a
    /// clock — and the reclaim's own finalization is what releases it,
    /// body and all.
    #[test]
    fn a_record_licenses_the_reclaim_and_its_finalization_releases_the_entry() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(6, 60_000);
        commit(&mut ledger, &tx);
        ledger.mark_leg(tx.hash(), body(&tx), BTreeSet::from([PARTNER]));
        ledger.certify(tx.hash());
        assert!(
            ledger.reclaimable().is_empty(),
            "nothing is reclaimed on a clock"
        );

        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            ms(1_000),
            [names(&tx)],
        )]);
        let reclaimable = ledger.reclaimable();
        assert_eq!(
            reclaimable.len(),
            1,
            "a committed record licenses the reclaim"
        );
        assert_eq!(reclaimable[0].0, tx.hash());
        assert_eq!(
            reclaimable[0].1.hash(),
            tx.hash(),
            "with the body the reclaim derives from"
        );

        ledger.admit_reclaim(tx.hash());
        assert!(ledger.reclaimable().is_empty(), "a tick has taken it");
        let reclaim =
            make_finalization(BlockHeight::new(9), tx.hash(), TransactionDecision::Accept);
        ledger.release_resolved(&[Arc::new(Verifiable::from(reclaim))]);
        assert_eq!(ledger.len(), 0, "the reclaim's finalization releases it");
        assert!(ledger.legs.is_empty(), "and the body goes with it");
    }

    /// A refusal names only legs, so a replica meeting one for a
    /// transaction it does not hold rebuilds a leg entry: never
    /// abandoned, and — with no body to derive the reclaim from — never
    /// reclaimed here either. It waits out its horizon.
    #[test]
    fn a_refusal_record_naming_an_unheld_transaction_rebuilds_a_leg_entry() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(7, 60_000);
        ledger.record_abandonment_records(&[AbandonmentRecord::refused(
            PARTNER,
            ms(1_000),
            [names(&tx)],
        )]);
        assert_eq!(ledger.len(), 1);
        let past = ms(60_000)
            .plus(MAX_FINALIZATION_DELAY)
            .plus(Duration::from_secs(1));
        assert!(ledger.past_deadline(past).is_empty(), "never abandoned");
        assert!(
            ledger.reclaimable().is_empty(),
            "and nothing to reclaim with"
        );
        let horizon = ms(60_000)
            .plus(MAX_FINALIZATION_DELAY)
            .plus(MAX_VALIDITY_RANGE);
        assert!(ledger.prune(horizon).is_empty());
        assert_eq!(ledger.len(), 0, "gone at its horizon");
    }

    /// A leg entry lives on the transaction's own clock, to the moment
    /// the record cell it would reclaim sweeps — whether or not a record
    /// has named it, and whatever its counterparts are doing.
    #[test]
    fn a_leg_entry_lives_to_the_record_cells_own_horizon() {
        let horizon = ms(60_000)
            .plus(MAX_FINALIZATION_DELAY)
            .plus(MAX_VALIDITY_RANGE);
        for covered in [false, true] {
            let mut ledger = UnresolvedTxs::default();
            let tx = tx(5, 60_000);
            commit(&mut ledger, &tx);
            ledger.mark_leg(tx.hash(), body(&tx), BTreeSet::from([PARTNER]));
            ledger.certify(tx.hash());
            if covered {
                ledger.record_abandonment_records(&[AbandonmentRecord::departed(
                    PARTNER,
                    ms(1_000),
                    [names(&tx)],
                )]);
            }
            assert!(
                ledger
                    .prune(horizon.minus(Duration::from_millis(1)))
                    .is_empty()
            );
            assert_eq!(
                ledger.len(),
                1,
                "covered={covered}: readable until the sweep"
            );
            assert!(
                ledger.prune(horizon).is_empty(),
                "covered={covered}: a leg dropped at its horizon leaks no reservation"
            );
            assert_eq!(ledger.len(), 0, "covered={covered}: gone with the cell");
        }
    }
}
