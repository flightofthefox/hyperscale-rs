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

use hyperscale_engine::legs::Classified;
use hyperscale_types::{
    AbandonmentRecord, AbortCharge, Absence, Address, BlockHeight, ClaimProof, CounterpartEvidence,
    Finalization, Probed, ShardId, ShardTrie, StateAnchor, SubstateKey, Transaction,
    TransactionDecision, TxHash, TxResolution, UnsettledTx, Verifiable, Verified,
    WeightedTimestamp, delivery_window_close, leg_entry_horizon, verdict_window_close,
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

/// What this shard has asked a silent counterpart about one transaction:
/// a proof of `key` against `anchor`, whose block sits at `probed_wt`;
/// `floor` is the anchor the answer is held to, kept beside it so the
/// answer can be dated without the ledger.
///
/// A probe is the fetch and nothing more. Its answer is read off the
/// block that carries the proof, by every replica alike, so the probe
/// stays only to keep the question from being asked twice of one
/// header: `answered` once the fetch returned and its bytes are waiting
/// to be offered in a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Probe {
    pub anchor: StateAnchor,
    pub key: SubstateKey,
    pub probed_wt: WeightedTimestamp,
    pub floor: WeightedTimestamp,
    pub probed: Probed,
    pub answered: bool,
}

/// What a proof the chain committed said about one counterpart cell of
/// one transaction — the fold every replica reaches at the same height,
/// and what the record arms are offered from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// The core took it: its certificate is what speaks next.
    Committed,
    /// The counterpart had not taken it past the floor: what an
    /// `Unclaimed` or `Lapsed` record is offered from.
    Absent(Absence),
    /// The consumer's claim is present: what a `Claimed` record is
    /// offered from, licensing the retirement of the record here.
    Present(ClaimProof),
}

/// Let go of every fetch `owed` still has out, so a counterpart that
/// never serves the height does not pin the slot.
fn release_fetches(released: &mut Vec<(StateAnchor, SubstateKey)>, owed: &Owed) {
    released.extend(
        owed.heard
            .probes
            .values()
            .filter(|probe| !probe.answered)
            .map(|probe| (probe.anchor, probe.key)),
    );
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
    /// Whether a committed finalization of this shard's settled the
    /// transaction's price: a leg's own, which burned it inside its
    /// writes, or the verdict that made an issuer a remainder.
    ///
    /// A committed fact rather than the tick's admission, which is what
    /// [`Self::certified`] records: a tick discarded before its
    /// finalization commits burned nothing, and a reclaim reading the
    /// admission would charge nothing either. Meaningless off a leg
    /// entry.
    charged: bool,
    /// What this shard's part in the transaction is, which decides what
    /// the entry waits on and what ends it.
    kind: Kind,
    /// Which tick of this shard's has taken a leg entry's records — a
    /// reclaim or a retirement — so the finalization naming the hash
    /// next is that member's. Meaningless off a leg entry.
    taken: Option<Taken>,
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
    /// What that record established — a refusal, a departure, an
    /// absence or a lapse — which is what the reclaim it licenses says
    /// the transaction's fate was.
    evidence: Option<CounterpartEvidence>,
    /// The consumer shards a committed record says claimed what this
    /// entry issued. Once every consumer has, the records here have
    /// nothing left to hold and the retirement is licensed.
    claimed_by: BTreeSet<ShardId>,
    /// What this shard has heard from the counterparts of the
    /// transaction, mirrored off their certificates as they arrive.
    ///
    /// Held on the entry rather than beside the ledger because that is
    /// the lifetime: a mirror speaks only for a transaction still owed
    /// an outcome here, and the entry going is what makes it moot. A
    /// second home would have to be reclaimed on its own rule, and two
    /// rules for one fact are two answers to when it stops being true.
    heard: Heard,
}

/// What this coordinator holds of its own about one transaction's
/// counterparts.
///
/// What a counterpart *said* is not here: a refusal, an absence and a
/// claim are each asked about by the vote fence too, so they live in the
/// [`CounterpartMirror`] both consumers read. What is here is this
/// coordinator's own working state around them — who has answered, what
/// it has asked, and which questions are closed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Heard {
    /// The core shards whose certificates accepted it. A core shard's
    /// tick closes on every other core shard's certificate, so one
    /// saying it succeeded is not the transaction accepted — that is
    /// every core shard saying so, and this is the count.
    accepted: BTreeSet<ShardId>,
    /// What this validator has asked a silent counterpart, by cell: its
    /// own fetch and nothing more, so one header is asked once.
    probes: BTreeMap<(ShardId, Probed), Probe>,
    /// The cells the chain has answered, whichever way. A closed
    /// question is never asked again, and what the answer *was* is the
    /// mirror's to say.
    answered: BTreeSet<(ShardId, Probed)>,
}

/// What a tick of this shard's composed over a leg entry's records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Taken {
    /// The reclaim, on a record that says the counterpart never will.
    Reclaim,
    /// The retirement, on a record that says every consumer claimed.
    Retire,
}

impl Owed {
    /// Whether `shard`, leaving at `cut`, was party to this entry: it
    /// owned one of the entry's remote prefixes, and the transaction
    /// committed before it left.
    fn party_to(&self, shard: ShardId, cut: WeightedTimestamp) -> bool {
        cut > self.committed_ts
            && self
                .remote_prefixes
                .iter()
                .any(|prefix| ShardTrie::shard_owns_prefix(shard, *prefix))
    }
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

/// What this shard's part in a transaction is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// This shard's verdict is the transaction's, or a share of it: the
    /// entry is abandonable at its deadline and released by that verdict.
    Whole,
    /// This shard only delivers for the transaction: a leg outside the
    /// core that bears no verdict and issues nothing, admissible to the
    /// delivery window's close, which is its deadline. Abandoned there
    /// like a whole entry — and, unlike one, out of any tick still
    /// holding it, since past the close the crossing it would claim
    /// lapses and its issuer may reclaim it.
    Delivery,
    /// This shard ran only a leg of the transaction: it froze divided
    /// with this shard outside the core set.
    ///
    /// A leg's tick attests it and its certificate settles alone, so the
    /// entry is never abandoned and its own finalization bears no verdict
    /// on the transaction. What the entry is for is the reclaim: it holds
    /// the terms a reclaim states, and it lives — on the transaction's
    /// own clock — exactly as long as the record cell it would take back.
    Leg,
    /// This shard's own verdict already resolved the transaction and the
    /// entry stays for the reclaim alone: it issued crossings that
    /// deliveries elsewhere still owe a claim for. Held as a leg entry
    /// from then on — never abandoned, probed past the deadline,
    /// released by the reclaim — but named by no departure record: a
    /// departed deliverer's successor still delivers, and only the lapse
    /// says a delivery never will.
    Remainder,
}

impl Kind {
    /// Whether the entry bears no verdict of its own and is held for a
    /// reclaim: a leg's, or a resolved issuer's remainder.
    const fn is_leg(self) -> bool {
        matches!(self, Self::Leg | Self::Remainder)
    }
}

/// What an entry that may be reclaimed keeps beside its account: a leg
/// entry's, for the reclaim and the refusal mirror, or an issuer's, for
/// the reclaim of what its deliveries never claimed.
#[derive(Debug, Clone)]
struct Kept {
    body: Arc<Verified<Transaction>>,
    /// The classification the committing block froze, which a reclaim
    /// reads its edges and its scope from.
    classified: Classified,
    /// Whose refusal is the transaction's. Empty for an issuer in the
    /// core, whose verdict is its own.
    core: BTreeSet<ShardId>,
    /// The claim cells deliveries elsewhere write for the crossings this
    /// shard issued, each under the shard that was to deliver it when
    /// the transaction committed — what a lapse probe asks about once
    /// the delivery window has closed. The cell follows its prefix to a
    /// departed deliverer's successor; the prober resolves that off the
    /// trie, since the ledger holds no topology.
    deliveries: Vec<(ShardId, SubstateKey)>,
    /// The claim cells core consumers write for the crossings a leg
    /// here issued, each under the shard holding the consumer's target
    /// — what a presence probe asks the core about past the deadline,
    /// and what a `Claimed` record answers for.
    claims: Vec<(ShardId, SubstateKey)>,
}

/// A leg entry a committed record has licensed the retirement of: every
/// consumer of what it issued has claimed, so its records have nothing
/// left to hold.
#[derive(Debug, Clone)]
pub struct Retirable {
    /// The transaction.
    pub tx_hash: TxHash,
    /// Its body, which the retirement's edges derive from.
    pub body: Arc<Verified<Transaction>>,
    /// The classification its committing block froze.
    pub classified: Classified,
}

/// A leg entry a committed record has licensed the reclaim of, with what
/// the reclaim is composed from.
#[derive(Debug, Clone)]
pub struct Reclaimable {
    /// The transaction.
    pub tx_hash: TxHash,
    /// Its body, which the reclaim's edges derive from.
    pub body: Arc<Verified<Transaction>>,
    /// The classification its committing block froze.
    pub classified: Classified,
    /// Whether a committed finalization of this shard's settled the
    /// price, so the reclaim charges nothing.
    pub charged: bool,
}

/// A leg entry whose counterparts have been silent past the
/// transaction's deadline — what a probe of a core's committed set, or
/// of a delivering shard's claim cell, asks about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probeable {
    /// The transaction.
    pub tx_hash: TxHash,
    /// Its deadline: the earliest core anchor at which absence means
    /// anything, and the figure a core's answer is dated against.
    pub deadline: WeightedTimestamp,
    /// Its validity end, which names the committed cell the core would
    /// have written and dates the lapse a delivery's answer is held to.
    pub validity_end: WeightedTimestamp,
    /// The core set. Any one core shard's absence suffices — no core
    /// shard finalizes without every other's certificate — so the probe
    /// goes to one of them. Empty for a shape with no core, whose
    /// counterparts are deliveries alone.
    pub core: BTreeSet<ShardId>,
    /// The claim cells deliveries elsewhere write for what this leg
    /// issued, each under the shard that was to deliver it at commit.
    /// Asked about past the lapse, there and on whatever shard holds
    /// the cell's prefix by then.
    pub deliveries: Vec<(ShardId, SubstateKey)>,
    /// The claim cells core consumers write for what this leg issued,
    /// each under the shard holding the consumer's target. Asked about
    /// past the deadline, there and on whatever shard holds the cell's
    /// prefix by then: present says the core took it, and absent, where
    /// the core is one shard, that it never will.
    pub claims: Vec<(ShardId, SubstateKey)>,
}

/// Committed-but-unresolved transactions, each against its deadline and
/// the reservation it holds.
#[derive(Debug, Default)]
pub struct UnresolvedTxs {
    owed: BTreeMap<TxHash, Owed>,
    /// What a leg entry or an issuer keeps beside its account: the body,
    /// for the reclaim, which derives from the transaction's legs and
    /// crossings after the candidate pool let the body go; the core set,
    /// which says whose refusal is the transaction's; and the claim
    /// cells its deliveries owe. Bounded by the entries' own horizon,
    /// and dropped with them.
    kept: HashMap<TxHash, Kept>,
    /// Where each departed participant's chain ended, for the entries
    /// whose fate only that shard's settled set can decide. Held against
    /// the schedule window that proves the terminal, which is retained on
    /// a frontier of its own.
    departed: BTreeMap<ShardId, Departure>,
    /// Fetches that were still out when the question they asked stopped
    /// mattering — the entry released, or the chain answering first.
    ///
    /// A probe is this validator's own fetch, so releasing it is an
    /// action rather than a fact, and the ledger states which ones it
    /// let go rather than leaving the caller to diff for them. Drained
    /// each commit; nothing reads it twice.
    released_fetches: Vec<(StateAnchor, SubstateKey)>,
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
                charged: false,
                kind: Kind::Whole,
                taken: None,
                unsettled_by: None,
                evidence: None,
                claimed_by: BTreeSet::new(),
                heard: Heard::default(),
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
        classified: Classified,
        deliveries: Vec<(ShardId, SubstateKey)>,
        claims: Vec<(ShardId, SubstateKey)>,
    ) {
        if let Some(owed) = self.owed.get_mut(&tx_hash) {
            owed.kind = Kind::Leg;
            let core = classified.core().clone();
            self.kept.insert(
                tx_hash,
                Kept {
                    body,
                    classified,
                    core,
                    deliveries,
                    claims,
                },
            );
        }
    }

    /// Record that this shard issues crossings of `tx_hash` that
    /// deliveries elsewhere consume, on a member whose verdict is its own
    /// — one in the core. Nothing changes until that verdict commits:
    /// the entry is abandonable and released like any other, and if it
    /// accepted, [`Self::release_resolved`] keeps it on as a remainder
    /// for the reclaim of whatever its deliveries never claim.
    pub fn mark_issuer(
        &mut self,
        tx_hash: TxHash,
        body: Arc<Verified<Transaction>>,
        classified: Classified,
        deliveries: Vec<(ShardId, SubstateKey)>,
    ) {
        if deliveries.is_empty() || !self.owed.contains_key(&tx_hash) {
            return;
        }
        self.kept.insert(
            tx_hash,
            Kept {
                body,
                classified,
                core: BTreeSet::new(),
                deliveries,
                claims: Vec::new(),
            },
        );
    }

    /// Record that this shard only delivers for `tx_hash`: it runs a leg
    /// outside the core that bears no verdict and issues nothing.
    ///
    /// Not a leg entry — there is nothing to reclaim and no core whose
    /// refusal is its own — but an ordinary one on a later clock: the
    /// delivery window's close rather than the transaction's deadline.
    /// A delivery never run by then is abandoned there like any other
    /// unresolved entry, returning its reservation; one that ran is
    /// released by its own finalization.
    pub fn mark_delivery(&mut self, tx_hash: TxHash, validity_end: WeightedTimestamp) {
        if let Some(owed) = self.owed.get_mut(&tx_hash) {
            owed.kind = Kind::Delivery;
            owed.deadline = delivery_window_close(validity_end);
        }
    }

    /// The core set of a leg entry — whose refusal is the transaction's.
    /// `None` for anything but a leg entry this ledger holds.
    #[must_use]
    pub fn leg_core(&self, tx_hash: TxHash) -> Option<&BTreeSet<ShardId>> {
        self.kept.get(&tx_hash).map(|leg| &leg.core)
    }

    /// The figures a record naming a leg entry restates, for one no
    /// record covers yet. `None` for anything else: a record is composed
    /// once per entry.
    #[must_use]
    pub fn unsettled_leg_figures(&self, tx_hash: TxHash) -> Option<UnsettledTx> {
        self.owed
            .get(&tx_hash)
            .filter(|owed| owed.kind.is_leg() && owed.unsettled_by.is_none())
            .map(|owed| UnsettledTx {
                tx_hash,
                deadline: owed.deadline,
                declared_work: owed.declared_work,
                charge: owed.charge,
            })
    }

    /// Whether `shard` is one of the transaction's core — whose refusal
    /// is the transaction's, and whose word is worth mirroring at all.
    #[must_use]
    pub fn core_holds(&self, tx_hash: TxHash, shard: ShardId) -> bool {
        self.leg_core(tx_hash)
            .is_some_and(|core| core.contains(&shard))
    }

    /// Mirror a core shard's acceptance, and say whether it was the last
    /// the transaction was waiting on.
    ///
    /// A core shard's tick closes on every other core shard's
    /// certificate, so one saying it succeeded is not the transaction
    /// accepted: that is every core shard saying so.
    pub fn record_acceptance(&mut self, tx_hash: TxHash, shard: ShardId) -> bool {
        let Some(core_len) = self
            .leg_core(tx_hash)
            .filter(|core| core.contains(&shard))
            .map(BTreeSet::len)
        else {
            return false;
        };
        self.owed.get_mut(&tx_hash).is_some_and(|owed| {
            owed.heard.accepted.insert(shard) && owed.heard.accepted.len() == core_len
        })
    }

    /// Whether the chain has already answered what `probed` asks of
    /// `shard` about `tx_hash`. An answered question is never asked
    /// again.
    #[must_use]
    pub fn answered(&self, tx_hash: TxHash, shard: ShardId, probed: Probed) -> bool {
        self.owed
            .get(&tx_hash)
            .is_some_and(|owed| owed.heard.answered.contains(&(shard, probed)))
    }

    /// Whether a probe of that cell is already out, or already answered
    /// at `height` or newer.
    ///
    /// A question in flight is left alone: a core's header lands every
    /// block, and moving the probe to each new one abandons the fetch
    /// before its answer returns. One whose fetch has answered is moved
    /// on, which is how a cell the chain read absent is asked again — at
    /// a newer header, not at the same one every block.
    #[must_use]
    pub fn probe_stands(
        &self,
        tx_hash: TxHash,
        shard: ShardId,
        probed: Probed,
        height: BlockHeight,
    ) -> bool {
        self.owed
            .get(&tx_hash)
            .and_then(|owed| owed.heard.probes.get(&(shard, probed)))
            .is_some_and(|probe| !probe.answered || probe.anchor.height >= height)
    }

    /// Put a question to a counterpart, replacing whatever stood.
    pub fn record_probe(&mut self, tx_hash: TxHash, shard: ShardId, probe: Probe) {
        if let Some(owed) = self.owed.get_mut(&tx_hash) {
            owed.heard.probes.insert((shard, probe.probed), probe);
        }
    }

    /// Mark every probe this proof spoke to as answered, and name the
    /// transactions it answered for with the clock its header sat at.
    ///
    /// The fetch is only how the proposer comes by the bytes: nothing is
    /// decided here, since the answer is the chain's once a block carries
    /// the proof.
    pub fn mark_probes_answered(
        &mut self,
        anchor: StateAnchor,
        keys: &[SubstateKey],
    ) -> (BTreeSet<TxHash>, Option<WeightedTimestamp>) {
        let mut answered = BTreeSet::new();
        let mut anchor_ts = None;
        for (&tx_hash, owed) in &mut self.owed {
            for probe in owed.heard.probes.values_mut() {
                if probe.anchor == anchor && keys.contains(&probe.key) {
                    probe.answered = true;
                    answered.insert(tx_hash);
                    anchor_ts = Some(probe.probed_wt);
                }
            }
        }
        (answered, anchor_ts)
    }

    /// Close the question `probed` asks of `shard` about `tx_hash`,
    /// releasing any fetch of this validator's still out for it.
    ///
    /// First proof wins: `false` says the cell was already answered, and
    /// a later proof adds nothing.
    pub fn close_question(&mut self, tx_hash: TxHash, shard: ShardId, probed: Probed) -> bool {
        let Some(owed) = self.owed.get_mut(&tx_hash) else {
            return false;
        };
        if let Some(probe) = owed.heard.probes.remove(&(shard, probed))
            && !probe.answered
        {
            self.released_fetches.push((probe.anchor, probe.key));
        }
        self.owed
            .get_mut(&tx_hash)
            .expect("held above")
            .heard
            .answered
            .insert((shard, probed))
    }

    /// The fetches the ledger has let go since this was last asked.
    pub fn take_released_fetches(&mut self) -> Vec<(StateAnchor, SubstateKey)> {
        std::mem::take(&mut self.released_fetches)
    }

    /// The leg entries whose counterparts may now be asked whether they
    /// took the transaction: past the deadline and covered by no record
    /// yet.
    ///
    /// The deadline gates the probe and nothing else. Before it the core
    /// may still legitimately commit, so absence says nothing; past it
    /// absence is proof, and the proof is what a record is composed on.
    /// A delivery's answer is held to a later floor still, the lapse,
    /// which the prober applies to the header it reads. Read off
    /// committed content alone, like [`Self::past_deadline`], so every
    /// replica at the same frontier asks about the same set.
    #[must_use]
    pub fn probeable(&self, now: WeightedTimestamp) -> Vec<Probeable> {
        self.cells()
            .into_iter()
            .filter(|entry| now >= entry.deadline)
            .collect()
    }

    /// Every leg entry no record has answered for, with the counterpart
    /// cells its questions are asked of, whatever the clock: what a
    /// proof the chain committed is read against, since the proof's
    /// own anchor says whether the window was open.
    #[must_use]
    pub fn cells(&self) -> Vec<Probeable> {
        self.owed
            .iter()
            .filter(|(_, owed)| owed.kind.is_leg() && owed.unsettled_by.is_none())
            .filter_map(|(tx_hash, owed)| {
                let leg = self.kept.get(tx_hash)?;
                Some(Probeable {
                    tx_hash: *tx_hash,
                    deadline: owed.deadline,
                    validity_end: leg.body.validity_range().end_timestamp_exclusive,
                    core: leg.core.clone(),
                    deliveries: leg.deliveries.clone(),
                    claims: leg.claims.clone(),
                })
            })
            .collect()
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
    ///
    /// Each carries whether a committed finalization of this shard's
    /// settled the price — a leg that ran settled it inside its own
    /// certificate, and one that never ran, or whose tick was discarded
    /// before its finalization committed, owes it on the reclaim's.
    #[must_use]
    pub fn reclaimable(&self) -> Vec<Reclaimable> {
        self.owed
            .iter()
            .filter(|(_, owed)| {
                owed.kind.is_leg() && owed.taken.is_none() && owed.unsettled_by.is_some()
            })
            .filter_map(|(tx_hash, owed)| {
                let kept = self.kept.get(tx_hash)?;
                Some(Reclaimable {
                    tx_hash: *tx_hash,
                    body: Arc::clone(&kept.body),
                    classified: kept.classified.clone(),
                    charged: owed.charged,
                })
            })
            .collect()
    }

    /// Record that a tick of this shard's has admitted the reclaim of
    /// `tx_hash`, so the finalization naming the hash next is the
    /// reclaim's and releases the entry.
    pub fn admit_reclaim(&mut self, tx_hash: TxHash) {
        if let Some(owed) = self.owed.get_mut(&tx_hash) {
            owed.taken = Some(Taken::Reclaim);
        }
    }

    /// The leg entries committed records have licensed the retirement of
    /// and no tick has taken yet: every consumer of what each issued —
    /// a core's, a delivery's — is on record as having claimed, no
    /// record covers the entry as unsettled, and nothing is taking it
    /// back. Read off committed content alone, like
    /// [`Self::reclaimable`], so every replica composes the same.
    #[must_use]
    pub fn retirable(&self) -> Vec<Retirable> {
        self.owed
            .iter()
            .filter(|(_, owed)| {
                owed.kind.is_leg()
                    && owed.taken.is_none()
                    && owed.unsettled_by.is_none()
                    && !owed.claimed_by.is_empty()
            })
            .filter_map(|(tx_hash, owed)| {
                let kept = self.kept.get(tx_hash)?;
                let consumers: BTreeSet<ShardId> = kept
                    .claims
                    .iter()
                    .chain(&kept.deliveries)
                    .map(|(shard, _)| *shard)
                    .collect();
                (!consumers.is_empty() && consumers.is_subset(&owed.claimed_by)).then(|| {
                    Retirable {
                        tx_hash: *tx_hash,
                        body: Arc::clone(&kept.body),
                        classified: kept.classified.clone(),
                    }
                })
            })
            .collect()
    }

    /// Record that a tick of this shard's has admitted the retirement
    /// of `tx_hash`'s records, so the finalization naming the hash next
    /// is the retirement's and releases the entry.
    pub fn admit_retire(&mut self, tx_hash: TxHash) {
        if let Some(owed) = self.owed.get_mut(&tx_hash) {
            owed.taken = Some(Taken::Retire);
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
                // The settling arm: a consumer claimed, and the entry
                // holds its records for the retirement rather than for a
                // reclaim. A name this ledger does not hold is left
                // alone — nothing here is owed for it.
                if !verdict.evidence().abandons() {
                    if let Some(owed) = self.owed.get_mut(&entry.tx_hash) {
                        owed.claimed_by.insert(verdict.shard());
                    }
                    continue;
                }
                if let Some(owed) = self.owed.get_mut(&entry.tx_hash) {
                    owed.unsettled_by = Some(verdict.shard());
                    owed.evidence = Some(verdict.evidence());
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
                        charged: false,
                        // A leg entry lives inside the replay window —
                        // its horizon is the transaction's own — so a
                        // replay registers and marks it before any record
                        // naming it lands. A replica that still meets one
                        // first reads the mark off the arm: only a leg is
                        // ever refused or unclaimed, and an entry rebuilt
                        // that way has no body to reclaim with, so it
                        // waits out its horizon rather than abandoning
                        // what the record licenses a reclaim of.
                        kind: if matches!(verdict.evidence(), CounterpartEvidence::Departed { .. })
                        {
                            Kind::Whole
                        } else {
                            Kind::Leg
                        },
                        taken: None,
                        unsettled_by: Some(verdict.shard()),
                        evidence: Some(verdict.evidence()),
                        claimed_by: BTreeSet::new(),
                        heard: Heard::default(),
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
    /// already gone was never party to what came after. Only ones no
    /// record covers yet, so a departure is answered once. And never a
    /// remainder: its verdict is in, and a departed deliverer's
    /// successor still delivers what it was owed — only the lapse says a
    /// delivery never will.
    #[must_use]
    pub fn outstanding_with(&self, shard: ShardId, cut: WeightedTimestamp) -> Vec<UnsettledTx> {
        self.owed
            .iter()
            .filter(|(_, owed)| {
                owed.certified
                    && owed.kind != Kind::Remainder
                    && owed.unsettled_by.is_none()
                    && owed.party_to(shard, cut)
            })
            .map(|(tx_hash, owed)| UnsettledTx {
                tx_hash: *tx_hash,
                deadline: owed.deadline,
                declared_work: owed.declared_work,
                charge: owed.charge,
            })
            .collect()
    }

    /// The transactions this ledger holds that `shard`, leaving at `cut`,
    /// was party to: what a departure record naming this shard's
    /// business with it may name, and nothing else. Wider than
    /// [`Self::outstanding_with`] — an entry no certificate covers, or one
    /// a record already answered, is still one the shard was party to —
    /// so a voter reading it refuses only a stranger.
    #[must_use]
    pub fn party_to(&self, shard: ShardId, cut: WeightedTimestamp) -> BTreeSet<TxHash> {
        self.owed
            .iter()
            .filter(|(_, owed)| owed.party_to(shard, cut))
            .map(|(tx_hash, _)| *tx_hash)
            .collect()
    }

    /// Whether a committed record has established that `tx_hash` can
    /// never settle — the question the split-boundary fence otherwise
    /// puts to a settled set that expires.
    /// Whether this shard only delivers for `tx_hash`.
    #[must_use]
    pub fn is_delivery(&self, tx_hash: TxHash) -> bool {
        self.owed
            .get(&tx_hash)
            .is_some_and(|owed| owed.kind == Kind::Delivery)
    }

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
        self.departed.entry(shard).or_insert(Departure {
            cut,
            readable_until: None,
        });
        if let Some(until) = readable_until {
            self.stamp_terminal(shard, until);
        }
    }

    /// Give a departure this ledger holds open its expiry, once. A
    /// departure not held is not invented here: the cut is the schedule's
    /// to state, and [`Self::record_terminal`] is where it is read.
    pub fn stamp_terminal(&mut self, shard: ShardId, readable_until: WeightedTimestamp) {
        if let Some(departure) = self.departed.get_mut(&shard)
            && departure.readable_until.is_none()
        {
            departure.readable_until = Some(readable_until);
        }
    }

    /// The departures this ledger holds with no expiry yet.
    ///
    /// An open window is the transient case — the beacon stamps the
    /// handoff some epochs after the cut — but the schedule only lists a
    /// departure while a retained window carries the shard, and the
    /// stamp lands on the head's boundary record, which outlives that
    /// window. A departure recorded before its window went and stamped
    /// after has to be asked about by name, or it and every entry a
    /// record covers against it hold each other open for good.
    #[must_use]
    pub fn unstamped_departures(&self) -> Vec<ShardId> {
        self.departed
            .iter()
            .filter(|(_, departure)| departure.readable_until.is_none())
            .map(|(shard, _)| *shard)
            .collect()
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

    /// What a committed block's finalizations settle about the
    /// transactions they name, for the status each is reported under.
    ///
    /// A name that decides is the transaction's verdict — a whole
    /// member's, or a failed leg's — except a deciding success on a leg
    /// entry, which is the reclaim of what the leg issued: the
    /// transaction did not happen, refused where the record that
    /// licensed the reclaim was a refusal and aborted where the core
    /// never took it. A lapse reclaim says nothing of the transaction,
    /// since its core accepted, and the core's certificates report
    /// that. A name that decides nothing is a leg finalizing here; a
    /// delivery that succeeded claimed what an accepted core issued,
    /// which is the verdict.
    ///
    /// Read before the same finalizations release the entries they
    /// name, since what a name means is a property of the entry.
    #[must_use]
    pub fn resolutions_of(
        &self,
        finalizations: &[Arc<Verifiable<Finalization>>],
    ) -> Vec<(TxHash, TxResolution)> {
        let mut resolutions = Vec::new();
        for finalization in finalizations {
            let deciding: BTreeSet<TxHash> = finalization.deciding_tx_hashes().collect();
            for (tx_hash, decision) in finalization.tx_decisions() {
                let owed = self.owed.get(&tx_hash);
                let accepted = decision == TransactionDecision::Accept;
                let retired = owed
                    .is_some_and(|owed| owed.kind.is_leg() && owed.taken == Some(Taken::Retire));
                let resolution = if retired {
                    // The retirement: every consumer claimed, so the
                    // transaction was accepted, whatever this member's
                    // own outcome says of the records.
                    TxResolution::Decided(TransactionDecision::Accept)
                } else if !deciding.contains(&tx_hash) {
                    if accepted && owed.is_some_and(|owed| owed.kind == Kind::Delivery) {
                        TxResolution::Decided(TransactionDecision::Accept)
                    } else {
                        TxResolution::LegFinalized
                    }
                } else if accepted && owed.is_some_and(|owed| owed.kind.is_leg()) {
                    match owed.and_then(|owed| owed.evidence) {
                        Some(CounterpartEvidence::Refused { .. }) => {
                            TxResolution::Decided(TransactionDecision::Reject)
                        }
                        Some(
                            CounterpartEvidence::Departed { .. }
                            | CounterpartEvidence::Unclaimed { .. }
                            | CounterpartEvidence::Untaken { .. },
                        ) => TxResolution::Decided(TransactionDecision::Aborted),
                        Some(
                            CounterpartEvidence::Lapsed { .. }
                            | CounterpartEvidence::Claimed { .. },
                        )
                        | None => {
                            continue;
                        }
                    }
                } else {
                    TxResolution::Decided(decision)
                };
                resolutions.push((tx_hash, resolution));
            }
        }
        resolutions
    }

    /// Drop what a committed block's finalizations resolve. Every verdict
    /// arrives this way — accepted, refused, or aborted — so one release
    /// path covers them all.
    ///
    /// A leg entry is released by a finalization that decides the
    /// transaction, or by the retirement's, which decides nothing and
    /// is the last word here all the same: a leg that succeeded bears
    /// no verdict, so the entry stays for the member a tick takes its
    /// records with — the reclaim, whose finalization decides it, or
    /// the retirement. A leg that failed is the transaction's end on
    /// this shard — it issued nothing, so there is nothing to reclaim —
    /// and its own finalization releases it.
    pub fn release_resolved(&mut self, finalizations: &[Arc<Verifiable<Finalization>>]) {
        for finalization in finalizations {
            let deciding: BTreeSet<TxHash> = finalization.deciding_tx_hashes().collect();
            for (tx_hash, decision) in finalization.tx_decisions() {
                let Some(owed) = self.owed.get_mut(&tx_hash) else {
                    continue;
                };
                let retired = owed.taken == Some(Taken::Retire);
                if owed.kind.is_leg() && !deciding.contains(&tx_hash) && !retired {
                    // The leg ran and its certificate burned the price
                    // inside its writes: what a reclaim of it charges
                    // nothing for.
                    owed.charged = true;
                    continue;
                }
                // An issuer that accepted has crossings out that its
                // deliveries owe a claim for: its verdict resolves the
                // transaction, and the entry stays on as a leg entry for
                // the reclaim alone. One that refused issued nothing.
                let issued = owed.kind == Kind::Whole
                    && decision == TransactionDecision::Accept
                    && self
                        .kept
                        .get(&tx_hash)
                        .is_some_and(|kept| !kept.deliveries.is_empty());
                if issued {
                    owed.kind = Kind::Remainder;
                    owed.charged = true;
                } else {
                    if let Some(gone) = self.owed.remove(&tx_hash) {
                        release_fetches(&mut self.released_fetches, &gone);
                    }
                    self.kept.remove(&tx_hash);
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
            .filter(|(_, owed)| !owed.kind.is_leg())
            .filter(|(_, owed)| {
                now >= owed.deadline
                    && (owed.unsettled_by.is_some() || now < verdict_window_close(owed.deadline))
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
    /// `deadline + 2 * MAX_VALIDITY_RANGE`, which is the validity end plus
    /// the escrow grace — the moment the claim cell both its members are
    /// proved against sweeps, one validity range past the lapse a
    /// delivery's absence is proved at, so a lapse proved at the earliest
    /// has a validity range to become a committed reclaim. Past it there is nothing to take
    /// back, whatever evidence arrives, so the entry goes on that reading
    /// alone. Dropping gives a reclaim up; it never licenses one. A
    /// record's evidence does not extend it, and no counterpart's silence
    /// shortens it: a reclaim waits on a record, and the record's arms
    /// are the evidence, not the counterpart's answerability.
    ///
    /// Returns the transactions dropped because every counterpart has
    /// fallen silent. Each carries whether a committed record had covered
    /// it, which separates a chain that ran out of room to commit the abort
    /// from one that never had the evidence to compose it. A leg entry
    /// dropped at its horizon is not among them: its reservation came
    /// back with its own finalization, so nothing leaks with it.
    pub fn prune(&mut self, now: WeightedTimestamp) -> Vec<Unanswerable> {
        let mut unanswerable = Vec::new();
        let (kept, dropped): (BTreeMap<TxHash, Owed>, BTreeMap<TxHash, Owed>) =
            std::mem::take(&mut self.owed)
                .into_iter()
                .partition(|(tx_hash, owed)| {
                    // A leg entry goes at its horizon, where the claim
                    // cell both its members are proved against is swept:
                    // past it neither the reclaim nor the retirement can
                    // be composed, whatever evidence lands. Short of it
                    // only the finalization that decides it ends it.
                    if owed.kind.is_leg() {
                        return leg_entry_horizon(owed.deadline) > now;
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
                    verdict_window_close(owed.deadline) > now
                });
        self.owed = kept;
        for owed in dropped.values() {
            release_fetches(&mut self.released_fetches, owed);
        }
        let owed = &self.owed;
        self.kept.retain(|tx_hash, _| owed.contains_key(tx_hash));

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
        make_finalization, make_leg_finalization, make_undecided_finalization, stub_transaction,
        test_prefix, test_principal,
    };
    use hyperscale_types::{
        BlockHeight, EPOCH_DURATION, EpochWindows, LocalKey, MAX_FINALIZATION_DELAY,
        MAX_VALIDITY_RANGE, TimestampRange, UnsettledTx, Verified, WeightedTimestamp,
    };

    use super::*;

    /// This shard owns the prefixes whose leading bit is zero, so an
    /// address is remote or local by its top byte and nothing else.
    const LOCAL: ShardId = ShardId::leaf(1, 0);
    const HERE: u8 = 0x11;
    const AWAY: u8 = 0xAA;

    /// The depth-1 shard owning every `AWAY`-topped prefix.
    const PARTNER: ShardId = ShardId::leaf(1, 1);

    /// A shape frozen divided with an inbound leg on `LOCAL` feeding a
    /// core on `PARTNER`.
    fn classified() -> Classified {
        use hyperscale_types::ShardTrie;
        use hyperscale_vm_types::LegRole;

        use crate::fixtures::leg;
        let legs = [
            leg(0, LegRole::Inbound, &[]),
            leg(2, LegRole::Core, &[(0, 0)]),
        ];
        let classified = Classified::freeze(&legs, &[], &ShardTrie::uniform(1));
        assert_eq!(classified.core(), &BTreeSet::from([PARTNER]));
        classified
    }

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

    /// What abandoning `tx` states: its reservation and its price.
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

    /// A departure held with no expiry holds the entry a record covers
    /// against it, however far the clock runs; a stamp landing later
    /// gives both their end, and the entry retires as covered once it is
    /// past.
    #[test]
    fn a_departure_stamped_late_still_retires_what_it_covers() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(4, 60_000);
        commit(&mut ledger, &tx);
        ledger.certify(tx.hash());
        let cut = ms(100_000);
        ledger.record_terminal(PARTNER, cut, None);
        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            cut,
            [names(&tx)],
        )]);
        assert_eq!(ledger.unstamped_departures(), vec![PARTNER]);

        let far = expiry(cut).plus(EPOCH_DURATION * 100);
        assert!(
            ledger.prune(far).is_empty(),
            "an open window holds the covered entry"
        );
        assert_eq!(ledger.len(), 1);

        ledger.stamp_terminal(PARTNER, expiry(cut));
        assert!(ledger.unstamped_departures().is_empty());
        let dropped = ledger.prune(far);
        assert_eq!(
            dropped,
            vec![Unanswerable {
                tx_hash: tx.hash(),
                covered_by_record: true,
            }],
            "past the stamp the covered entry retires"
        );
        assert_eq!(ledger.len(), 0);
        assert!(
            ledger.unstamped_departures().is_empty(),
            "and the departure goes with the last entry naming it"
        );

        ledger.stamp_terminal(PARTNER, cut);
        assert!(
            ledger.unstamped_departures().is_empty(),
            "a stamp for a departure not held invents nothing"
        );
    }

    /// A delivery-only entry runs on the delivery window's clock: not
    /// abandoned at the transaction's deadline, abandoned at the window's
    /// close if it never ran, released by its own finalization if it did,
    /// and never a leg — nothing to reclaim, nothing to probe.
    #[test]
    fn a_delivery_entry_lives_to_the_windows_close() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(4, 60_000);
        commit(&mut ledger, &tx);
        ledger.mark_delivery(tx.hash(), ms(60_000));
        assert!(ledger.is_delivery(tx.hash()));
        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        let close = delivery_window_close(ms(60_000));

        assert!(
            ledger.past_deadline(deadline).is_empty(),
            "the transaction's deadline abandons no delivery"
        );
        assert!(
            ledger.probeable(close).is_empty(),
            "and nothing probes for it"
        );
        let abandonable = ledger.past_deadline(close);
        assert_eq!(
            abandonable.len(),
            1,
            "the window's close abandons one never run"
        );
        assert_eq!(abandonable[0].tx_hash, tx.hash());

        let mut delivered = UnresolvedTxs::default();
        commit(&mut delivered, &tx);
        delivered.mark_delivery(tx.hash(), ms(60_000));
        delivered.certify(tx.hash());
        let own = make_finalization(BlockHeight::new(1), tx.hash(), TransactionDecision::Accept);
        delivered.release_resolved(&[Arc::new(Verifiable::from(own))]);
        assert_eq!(
            delivered.len(),
            0,
            "a delivery's own finalization releases it"
        );
    }

    /// A leg entry is probeable from its deadline and not a moment
    /// before, and only while no record covers it; an entry this shard
    /// ran whole is never probed, since nothing it awaits is a core.
    #[test]
    fn a_leg_entry_is_probeable_past_its_deadline_until_a_record_covers_it() {
        let mut ledger = UnresolvedTxs::default();
        let leg = tx(4, 60_000);
        let whole = tx(5, 60_000);
        commit(&mut ledger, &leg);
        commit(&mut ledger, &whole);
        ledger.mark_leg(leg.hash(), body(&leg), classified(), Vec::new(), Vec::new());
        ledger.certify(leg.hash());
        ledger.certify(whole.hash());

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        assert!(
            ledger
                .probeable(deadline.minus(Duration::from_millis(1)))
                .is_empty(),
            "before the deadline the core may still commit"
        );
        assert_eq!(
            ledger.probeable(deadline),
            vec![Probeable {
                tx_hash: leg.hash(),
                deadline,
                validity_end: ms(60_000),
                core: BTreeSet::from([PARTNER]),
                deliveries: Vec::new(),
                claims: Vec::new(),
            }],
            "at the deadline the leg is probeable and the whole entry is not"
        );

        ledger.record_abandonment_records(&[AbandonmentRecord::unclaimed(
            PARTNER,
            deadline,
            [names(&leg)],
        )]);
        assert!(
            ledger.probeable(deadline).is_empty(),
            "a covered entry is asked about once"
        );
        assert_eq!(
            ledger.reclaimable().len(),
            1,
            "and the record licenses the reclaim"
        );
    }

    /// A leg whose counterpart is a delivery is probeable past the
    /// deadline like any other, and carries the claim cells the probe
    /// asks about; a record over the lapse covers it and licenses the
    /// reclaim.
    #[test]
    fn a_leg_delivered_elsewhere_is_probeable_with_its_claims() {
        let mut ledger = UnresolvedTxs::default();
        let leg = tx(6, 60_000);
        commit(&mut ledger, &leg);
        let claim = SubstateKey {
            owner: test_prefix(AWAY),
            local: LocalKey([0xC1; 16]),
        };
        ledger.mark_leg(
            leg.hash(),
            body(&leg),
            classified(),
            vec![(PARTNER, claim)],
            Vec::new(),
        );
        ledger.certify(leg.hash());

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        assert_eq!(
            ledger.probeable(deadline),
            vec![Probeable {
                tx_hash: leg.hash(),
                deadline,
                validity_end: ms(60_000),
                core: BTreeSet::from([PARTNER]),
                deliveries: vec![(PARTNER, claim)],
                claims: Vec::new(),
            }],
        );
        ledger.record_abandonment_records(&[AbandonmentRecord::lapsed(
            PARTNER,
            deadline.plus(MAX_VALIDITY_RANGE),
            [names(&leg)],
        )]);
        assert!(ledger.probeable(deadline).is_empty(), "covered once");
        assert_eq!(
            ledger.reclaimable().len(),
            1,
            "and the lapse licenses the reclaim"
        );
    }

    /// An issuer in the core is released by its own verdict like any
    /// entry — unless it accepted with deliveries owed, when it stays on
    /// as a remainder: never abandoned, named by no departure, probed
    /// past the deadline for its claims, reclaimed on a lapse record,
    /// and released by the reclaim's finalization.
    #[test]
    fn an_issuer_that_accepted_stays_on_as_a_remainder_for_its_deliveries() {
        let claim = SubstateKey {
            owner: test_prefix(AWAY),
            local: LocalKey([0xC2; 16]),
        };
        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        let resolved = |decision| {
            let tx = tx(8, 60_000);
            let mut ledger = UnresolvedTxs::default();
            commit(&mut ledger, &tx);
            ledger.mark_issuer(tx.hash(), body(&tx), classified(), vec![(PARTNER, claim)]);
            ledger.certify(tx.hash());
            let finalization = make_finalization(BlockHeight::new(1), tx.hash(), decision);
            ledger.release_resolved(&[Arc::new(Verifiable::from(finalization))]);
            (ledger, tx)
        };

        let (ledger, _) = resolved(TransactionDecision::Reject);
        assert_eq!(ledger.len(), 0, "a refusal issued nothing to reclaim");

        let (mut ledger, tx) = resolved(TransactionDecision::Accept);
        assert_eq!(ledger.len(), 1, "an acceptance stays for the reclaim");
        assert!(ledger.past_deadline(deadline).is_empty(), "never abandoned");
        assert!(
            ledger.outstanding_with(PARTNER, ms(70_000)).is_empty(),
            "named by no departure: the successor still delivers"
        );
        assert_eq!(
            ledger.probeable(deadline),
            vec![Probeable {
                tx_hash: tx.hash(),
                deadline,
                validity_end: ms(60_000),
                core: BTreeSet::new(),
                deliveries: vec![(PARTNER, claim)],
                claims: Vec::new(),
            }],
        );
        ledger.record_abandonment_records(&[AbandonmentRecord::lapsed(
            PARTNER,
            deadline.plus(MAX_VALIDITY_RANGE),
            [names(&tx)],
        )]);
        let reclaims = ledger.reclaimable();
        assert_eq!(reclaims.len(), 1);
        assert!(
            reclaims[0].charged,
            "the issuer ran, so its reclaim is charged nothing"
        );
        ledger.admit_reclaim(tx.hash());
        let finalization =
            make_finalization(BlockHeight::new(2), tx.hash(), TransactionDecision::Accept);
        ledger.release_resolved(&[Arc::new(Verifiable::from(finalization))]);
        assert_eq!(ledger.len(), 0, "the reclaim's finalization releases it");
    }

    /// A leg's own finalization bears no verdict on the transaction, so
    /// it releases nothing, and the entry is never abandoned — not at its
    /// deadline, and not when a committed record names it.
    #[test]
    fn a_leg_entry_outlives_its_own_finalization_and_is_never_abandoned() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(4, 60_000);
        commit(&mut ledger, &tx);
        ledger.mark_leg(tx.hash(), body(&tx), classified(), Vec::new(), Vec::new());
        ledger.certify(tx.hash());

        let own = make_leg_finalization(BlockHeight::new(1), tx.hash());
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

    /// A reclaim charges the price only where no committed finalization
    /// of this shard's did: a leg admitted to a tick that was discarded
    /// before its finalization committed burned nothing, so its reclaim
    /// carries the price; once the leg's own finalization commits, the
    /// price is settled and the reclaim charges nothing.
    #[test]
    fn a_reclaim_charges_what_no_committed_finalization_settled() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(7, 60_000);
        commit(&mut ledger, &tx);
        ledger.mark_leg(tx.hash(), body(&tx), classified(), Vec::new(), Vec::new());
        ledger.certify(tx.hash());
        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            ms(1_000),
            [names(&tx)],
        )]);
        assert!(
            !ledger.reclaimable()[0].charged,
            "admission to a tick settles nothing"
        );

        let own = make_leg_finalization(BlockHeight::new(1), tx.hash());
        ledger.release_resolved(&[Arc::new(Verifiable::from(own))]);
        assert!(
            ledger.reclaimable()[0].charged,
            "the leg's committed finalization settled the price"
        );
    }

    /// What a finalization's name means is a property of the entry it
    /// names: a leg's own success is the leg finalizing, its failure the
    /// verdict, and a deciding success on a leg entry the reclaim —
    /// whose meaning is the record's: refused, or never taken.
    #[test]
    fn a_finalizations_name_resolves_by_the_entry_it_names() {
        let fw = |ledger: &UnresolvedTxs, finalization: Finalization| {
            ledger.resolutions_of(&[Arc::new(Verifiable::from(finalization))])
        };
        let h = BlockHeight::new(1);

        let mut ledger = UnresolvedTxs::default();
        let whole = tx(1, 60_000);
        commit(&mut ledger, &whole);
        assert_eq!(
            fw(
                &ledger,
                make_finalization(h, whole.hash(), TransactionDecision::Reject)
            ),
            vec![(
                whole.hash(),
                TxResolution::Decided(TransactionDecision::Reject)
            )],
            "a whole member's verdict is the transaction's"
        );

        let leg = tx(2, 60_000);
        commit(&mut ledger, &leg);
        ledger.mark_leg(leg.hash(), body(&leg), classified(), Vec::new(), Vec::new());
        assert_eq!(
            fw(&ledger, make_leg_finalization(h, leg.hash())),
            vec![(leg.hash(), TxResolution::LegFinalized)],
            "a leg's success is its own state"
        );
        assert_eq!(
            fw(
                &ledger,
                make_finalization(h, leg.hash(), TransactionDecision::Reject)
            ),
            vec![(
                leg.hash(),
                TxResolution::Decided(TransactionDecision::Reject)
            )],
            "a leg's failure is the verdict"
        );
        let reclaim =
            make_finalization(BlockHeight::new(9), leg.hash(), TransactionDecision::Accept);
        assert!(
            fw(&ledger, reclaim.clone()).is_empty(),
            "a deciding success on a leg entry no record covers says nothing"
        );
        ledger.record_abandonment_records(&[AbandonmentRecord::refused(
            PARTNER,
            ms(70_000),
            [names(&leg)],
        )]);
        assert_eq!(
            fw(&ledger, reclaim.clone()),
            vec![(
                leg.hash(),
                TxResolution::Decided(TransactionDecision::Reject)
            )],
            "the reclaim of a refused leg reports the refusal"
        );
        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            ms(1_000),
            [names(&leg)],
        )]);
        assert_eq!(
            fw(&ledger, reclaim.clone()),
            vec![(
                leg.hash(),
                TxResolution::Decided(TransactionDecision::Aborted)
            )],
            "the reclaim of a leg its core never took reports an abort"
        );
        ledger.record_abandonment_records(&[AbandonmentRecord::lapsed(
            PARTNER,
            ms(200_000),
            [names(&leg)],
        )]);
        assert!(
            fw(&ledger, reclaim).is_empty(),
            "a lapse reclaim says nothing: the core accepted, and its certificates say so"
        );

        let delivery = tx(3, 60_000);
        commit(&mut ledger, &delivery);
        ledger.mark_delivery(delivery.hash(), ms(60_000));
        assert_eq!(
            fw(
                &ledger,
                make_undecided_finalization(h, delivery.hash(), TransactionDecision::Accept)
            ),
            vec![(
                delivery.hash(),
                TxResolution::Decided(TransactionDecision::Accept)
            )],
            "a delivery that succeeded claimed what an accepted core issued"
        );
        assert_eq!(
            fw(
                &ledger,
                make_undecided_finalization(h, delivery.hash(), TransactionDecision::Reject)
            ),
            vec![(delivery.hash(), TxResolution::LegFinalized)],
            "a delivery that failed decides nothing, and the value waits for a later claim"
        );
    }

    /// A committed `Claimed` record is what licenses retiring a leg's
    /// records — never a clock, never a certificate — once every
    /// consumer is on record; the retirement's own finalization
    /// releases the entry, and a claimed entry is neither reclaimable
    /// nor abandonable meanwhile.
    #[test]
    fn a_claimed_record_licenses_the_retirement_and_its_finalization_releases_the_entry() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(8, 60_000);
        commit(&mut ledger, &tx);
        let claim = SubstateKey {
            owner: test_prefix(AWAY),
            local: LocalKey([0x77; 16]),
        };
        ledger.mark_leg(
            tx.hash(),
            body(&tx),
            classified(),
            Vec::new(),
            vec![(PARTNER, claim)],
        );
        ledger.certify(tx.hash());
        assert!(ledger.retirable().is_empty(), "nothing retires on a clock");

        ledger.record_abandonment_records(&[AbandonmentRecord::claimed(
            PARTNER,
            ms(70_000),
            [names(&tx)],
        )]);
        let retirable = ledger.retirable();
        assert_eq!(retirable.len(), 1, "every consumer claimed");
        assert_eq!(retirable[0].tx_hash, tx.hash());
        assert!(
            ledger.reclaimable().is_empty(),
            "a claim is a settlement, not evidence for a reclaim"
        );
        assert!(
            ledger.past_deadline(ms(200_000)).is_empty(),
            "and abandons nothing"
        );

        ledger.admit_retire(tx.hash());
        assert!(ledger.retirable().is_empty(), "a tick has taken it");
        let retirement = make_leg_finalization(BlockHeight::new(9), tx.hash());
        assert_eq!(
            ledger.resolutions_of(&[Arc::new(Verifiable::from(retirement.clone()))]),
            vec![(
                tx.hash(),
                TxResolution::Decided(TransactionDecision::Accept)
            )],
            "the retirement says every consumer claimed: the transaction was accepted"
        );
        ledger.release_resolved(&[Arc::new(Verifiable::from(retirement))]);
        assert_eq!(ledger.len(), 0, "the retirement's finalization releases it");
    }

    /// A leg that failed is the transaction's end on this shard: its
    /// finalization decides, and with nothing issued there is nothing
    /// to reclaim, so the entry goes with it — body and all — and no
    /// record can license a reclaim of it afterwards.
    #[test]
    fn a_failed_legs_own_finalization_releases_its_entry() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(5, 60_000);
        commit(&mut ledger, &tx);
        ledger.mark_leg(tx.hash(), body(&tx), classified(), Vec::new(), Vec::new());
        ledger.certify(tx.hash());

        let own = make_finalization(BlockHeight::new(1), tx.hash(), TransactionDecision::Reject);
        ledger.release_resolved(&[Arc::new(Verifiable::from(own))]);
        assert_eq!(ledger.len(), 0, "a failed leg's finalization decides it");
        assert!(ledger.kept.is_empty(), "and the body goes with it");

        ledger.record_abandonment_records(&[AbandonmentRecord::departed(
            PARTNER,
            ms(1_000),
            [names(&tx)],
        )]);
        assert!(
            ledger.reclaimable().is_empty(),
            "a record naming it afterwards rebuilds an entry with no body to reclaim from"
        );
    }

    /// A committed record is what makes a leg entry reclaimable — never a
    /// clock — and the reclaim's own finalization is what releases it,
    /// body and all.
    #[test]
    fn a_record_licenses_the_reclaim_and_its_finalization_releases_the_entry() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(6, 60_000);
        commit(&mut ledger, &tx);
        ledger.mark_leg(tx.hash(), body(&tx), classified(), Vec::new(), Vec::new());
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
        assert_eq!(reclaimable[0].tx_hash, tx.hash());
        assert_eq!(
            reclaimable[0].body.hash(),
            tx.hash(),
            "with the body the reclaim derives from"
        );

        ledger.admit_reclaim(tx.hash());
        assert!(ledger.reclaimable().is_empty(), "a tick has taken it");
        let reclaim =
            make_finalization(BlockHeight::new(9), tx.hash(), TransactionDecision::Accept);
        ledger.release_resolved(&[Arc::new(Verifiable::from(reclaim))]);
        assert_eq!(ledger.len(), 0, "the reclaim's finalization releases it");
        assert!(ledger.kept.is_empty(), "and the body goes with it");
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
            .plus(MAX_VALIDITY_RANGE * 2);
        assert!(ledger.prune(horizon).is_empty());
        assert_eq!(ledger.len(), 0, "gone at its horizon");
    }

    /// A leg entry dies where its evidence does: at its horizon the claim
    /// cell both its members are proved against is swept, so neither the
    /// reclaim nor the retirement can be composed past it, whatever a
    /// record says. Short of it only the finalization that decides it
    /// ends it, and a record covering it neither extends nor shortens it.
    #[test]
    fn a_leg_entry_dies_where_its_evidence_does() {
        let horizon = leg_entry_horizon(ms(60_000).plus(MAX_FINALIZATION_DELAY));
        for covered in [false, true] {
            let mut ledger = UnresolvedTxs::default();
            let tx = tx(5, 60_000);
            commit(&mut ledger, &tx);
            ledger.mark_leg(tx.hash(), body(&tx), classified(), Vec::new(), Vec::new());
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
                "covered={covered}: it stands short of its horizon"
            );
            assert!(
                ledger.prune(horizon).is_empty(),
                "covered={covered}: and leaks no reservation going"
            );
            assert_eq!(ledger.len(), 0, "covered={covered}: gone at its horizon");
            assert!(
                ledger.kept.is_empty(),
                "covered={covered}: and the body goes with it"
            );
        }
    }
}
