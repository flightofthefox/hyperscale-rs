//! What counterparts have said about the transactions in flight here,
//! and what this shard still asks them.
//!
//! One account of the exchange: the ledger of what this shard owes an
//! outcome for, the mirror of what counterparts were heard to say —
//! shared with the vote fence, which checks a record against exactly
//! what was offered from — the questions put to silent counterparts,
//! and the proofs fetched back to offer in a block. Everything here is
//! folded from committed content or from a certificate every replica
//! hears the same way, so replicas at one frontier hold one account.
//! The tick machine reads the ledger through it and decides what to do
//! with a strand nobody can answer for.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use hyperscale_core::{Action, FetchIds, FetchRequest, ProtocolEvent};
use hyperscale_metrics::{record_rebuilt_verdict_entry, record_reclaim_probe_answered};
use hyperscale_storage::committed_tx_cell_key;
use hyperscale_types::{
    AbandonmentRecord, Anchor, Block, BlockHeight, CounterpartEvidence, CounterpartMirror,
    Deadline, ExecutionCertificate, Heard, Inclusion, MAX_ABANDONMENT_RECORDS_PER_BLOCK,
    MAX_STATE_PROOFS_PER_BLOCK, MAX_UNSETTLED_PER_BLOCK, MerkleInclusionProof, Probed,
    ProvenAnchors, Question, SettledTxSet, ShardId, ShardTrie, Spoken, StateProofBundle,
    SubstateKey, TerminalEvidence, TopologySchedule, TransactionDecision, TxHash, TxResolution,
    UnsettledTx, Verifiable, Verified, WeightedTimestamp, Word,
};
use hyperscale_vm_effects::CrossingCell;

use crate::unresolved::{Probeable, Released, Unanswerable, UnresolvedTxs};

/// One counterpart cell a leg entry asks about: the shard holding it,
/// the cell, the anchor an answer is held to, and which question it is.
type CounterpartCell = (ShardId, SubstateKey, Probed);

/// What counterparts were heard to say, by shard and question, then by
/// the moment and word of each answer: the grouping a block's records
/// are composed from.
type HeardByQuestion =
    BTreeMap<(ShardId, Question), BTreeMap<(WeightedTimestamp, Word), Vec<UnsettledTx>>>;

/// Every cell `entry` asks a counterpart about, under `trie`: each other
/// core shard's committed cell, each delivery's claim on the shard that
/// was to deliver it and on whatever shard holds the cell's prefix now,
/// and each core consumer's claim. The one enumeration the prober and
/// the commit-time fold both read, so what is asked and what is answered
/// are the same cells.
///
/// `local` is never asked about: a core member holds the core it is part
/// of, itself included, because the arity an absent committed cell is
/// read against is the core's own — but what it has committed is not
/// something it fetches a proof of.
fn counterpart_cells(entry: &Probeable, local: ShardId, trie: &ShardTrie) -> Vec<CounterpartCell> {
    // The committed cell is asked about only where its absence would
    // answer: a core of more than one shard, whose shards settle on each
    // other's certificates with no clock. A core of one shard answers
    // through its claim, which the deadline fences.
    //
    // Every core shard is asked, because any one of them absent is the
    // whole answer — a core that one of its shards never included can
    // never settle — while the shards that did include say only that a
    // sibling is still pending. Asking the lowest alone leaves the
    // crossing stranded whenever that shard is the one that included.
    // Nothing is asked before the deadline, so a core that settles pays
    // for none of this.
    let core_answers = Probed::Core
        .read(Inclusion::Absent, entry.core.len())
        .is_some();
    let core = entry
        .core
        .iter()
        .filter(move |&&shard| core_answers && shard != local)
        .map(|&shard| {
            (
                shard,
                committed_tx_cell_key(shard, entry.tx_hash, entry.deadline.validity_end()),
                Probed::Core,
            )
        });
    let deliveries = entry.deliveries.iter().flat_map(|&(delivered_by, claim)| {
        BTreeSet::from([delivered_by, trie.shard_for_prefix(claim.owner)])
            .into_iter()
            .map(move |shard| (shard, claim, Probed::Delivery))
    });
    let claims = entry.claims.iter().flat_map(|&(shard, claim)| {
        BTreeSet::from([shard, trie.shard_for_prefix(claim.owner)])
            .into_iter()
            .map(move |shard| (shard, claim, Probed::Claim))
    });
    core.chain(deliveries).chain(claims).collect()
}

/// An inherited escrow record, and where its one question stands.
///
/// One question, to one shard: is the claim cell this record names
/// present? So the bookkeeping is one height rather than the ledger's
/// `Asked` — there is no set of counterparts to track, no fetch release
/// to sequence, and no record to compose, because the answer is a state
/// proof and a state proof is block content already.
#[derive(Debug, Clone)]
pub struct Inherited {
    /// The record leaf, which carries the claim key, the issuing
    /// transaction and the expiry every window is read off.
    pub cell: CrossingCell,
    /// The newest counterpart header the claim has been asked at, so
    /// the question is not re-sent at the same one every block.
    asked_at: Option<BlockHeight>,
    /// What a committed proof said, once one has said anything:
    /// present at any anchor, absent only past the lapse.
    pub answer: Option<Word>,
}

impl Inherited {
    /// The record as it arrives at a seat: undisposed, unasked.
    #[must_use]
    pub const fn seated(cell: CrossingCell) -> Self {
        Self {
            cell,
            asked_at: None,
            answer: None,
        }
    }

    /// The deadline every window this record is read against derives
    /// from — the producing intent's, recovered from the expiry the leaf
    /// states.
    #[must_use]
    pub const fn deadline(&self) -> Deadline {
        Deadline::from_expiry(self.cell.expiry_ms)
    }
}

/// Whether any undisposed inherited record is still waiting on `key`.
///
/// Read off the keys rather than off a name, because an inherited
/// record names no transaction this chain committed.
fn awaited_by(inherited: &BTreeMap<SubstateKey, Inherited>, key: SubstateKey) -> bool {
    inherited
        .values()
        .any(|record| record.answer.is_none() && record.cell.consumer_claim == key)
}

/// What a commit folded, and what it could not answer for.
pub struct Committed {
    /// The fetches it releases and the probes it opens.
    pub actions: Vec<Action>,
    /// The transactions let go of because every counterpart has fallen
    /// silent — the tick machine's to discard.
    pub unanswerable: Vec<Unanswerable>,
}

/// What this validator holds to offer in a block it proposes.
pub struct Offers {
    /// Proofs its own fetches answered that no block has carried yet,
    /// in the one order a block carries them.
    pub state_proofs: Vec<StateProofBundle>,
    /// The records it has evidence for and has not yet written down.
    pub abandonment_records: Vec<AbandonmentRecord>,
}

pub struct Counterparts {
    local_shard: ShardId,

    /// Committed transactions still owed an outcome, folded from the
    /// chain rather than read off live tick state — the only account of
    /// what this shard has in flight that can be rebuilt after losing
    /// that state.
    pub(crate) ledger: UnresolvedTxs,

    /// What counterparts have said about the transactions legs here
    /// issued for, shared with the shard coordinator's vote fence: a
    /// core's refusal, a proved absence, a consumer's claim. This
    /// account is the only writer, and the only one that says what
    /// to drop — the ledger above is what an entry there speaks for.
    ///
    /// One mirror, because the fence checks a record against exactly
    /// what was offered from, and two copies could answer differently.
    pub(crate) mirror: Arc<CounterpartMirror>,

    /// Commit-proven remote source blocks, shared with the shard
    /// coordinator, which owns the mirror and feeds it off
    /// `RemoteHeaderCommitted`.
    ///
    /// A cross-shard EC is consumable only against a proven source block
    /// — a bare QC certifies availability, and an f+1..2f corrupt
    /// committee can certify a sibling that never commits and export ECs
    /// computed from it. The same anchors are what a probe of a
    /// counterpart's committed set is taken against, and what this
    /// validator's vote fence holds a block's state proofs to: one
    /// mirror, so a bundle cannot pass the fence at an anchor no prober
    /// here would have chosen.
    pub(crate) proven_anchors: Arc<ProvenAnchors>,

    /// The proofs this validator's own fetches answered, each with the
    /// transactions whose probes it spoke to, held to offer in a block
    /// this validator proposes: a proof is committed content, folded by
    /// every replica at the same height, and the fetch is only how the
    /// proposer comes by the bytes. A bundle leaves when a block
    /// carries it, or when every transaction it answered for is gone.
    fetched: BTreeMap<StateProofBundle, BTreeSet<TxHash>>,

    /// The questions this validator has answered for itself, off a proof
    /// it fetched and verified.
    ///
    /// Kept apart from the mirror, which stays fed by committed content
    /// alone so two validators at one committed height compose the same
    /// records. What this licenses is narrower: not asking a counterpart
    /// again for an answer already in hand.
    verified: BTreeSet<(TxHash, ShardId, Probed)>,

    /// The escrow records this shard inherited with a prefix, each still
    /// undisposed, by cell key.
    ///
    /// A seat's store arrives holding value its predecessors escrowed
    /// and nothing else names it: its ledger begins empty, no body
    /// arrives with the leaves, and the chains that issued the crossings
    /// have ended, so no counterpart record will ever be composed about
    /// them. What is left is the claim cell each record names, and the
    /// leaf carries every term needed to ask about it. Held here rather
    /// than beside the tick machine because the question is a
    /// counterpart's to answer, like every other question in this file.
    pub(crate) inherited: BTreeMap<SubstateKey, Inherited>,
    /// Fetches the ledger let go of since the last commit — a question
    /// the chain answered first, or one whose entry is gone — released
    /// as one abandon at the commit, so a counterpart that never serves
    /// the height does not pin the slot.
    released_fetches: Vec<(Anchor, SubstateKey)>,
}

impl Counterparts {
    #[must_use]
    pub fn new(
        local_shard: ShardId,
        proven_anchors: Arc<ProvenAnchors>,
        mirror: Arc<CounterpartMirror>,
    ) -> Self {
        Self {
            local_shard,
            ledger: UnresolvedTxs::default(),
            mirror,
            proven_anchors,
            fetched: BTreeMap::new(),
            verified: BTreeSet::new(),
            inherited: BTreeMap::new(),
            released_fetches: Vec::new(),
        }
    }

    /// As [`Self::new`], holding the escrow records a reshape seat
    /// imported with its prefix. A leaf that does not decode is one no
    /// disposal could be composed from, so it is dropped rather than
    /// held.
    pub fn seated(
        local_shard: ShardId,
        proven_anchors: Arc<ProvenAnchors>,
        evidence: Arc<CounterpartMirror>,
        records: &[(SubstateKey, Vec<u8>)],
    ) -> Self {
        let mut counterparts = Self::new(local_shard, proven_anchors, evidence);
        counterparts.inherited = records
            .iter()
            .filter_map(|(key, value)| {
                Some((*key, Inherited::seated(CrossingCell::from_bytes(value)?)))
            })
            .collect();
        counterparts
    }

    /// Fold what a committed block says about counterparts — the proofs
    /// and verdict records it carries, the entries its certificates
    /// resolve, the departures the schedule now proves — let go of what
    /// no window can still answer, and ask what the block's clock opens.
    ///
    /// `trie` is the block's committee's, which says who was party to
    /// each transaction, and `now` the committed clock every deadline is
    /// read against.
    pub fn on_commit(
        &mut self,
        trie: &ShardTrie,
        topology_schedule: &TopologySchedule,
        block: &Block,
        now: WeightedTimestamp,
    ) -> Committed {
        self.gc_settled_sets(topology_schedule, now);
        // A proof the chain now carries is everybody's: its answers are
        // folded here, and nothing offers it again.
        for bundle in block.state_proofs() {
            self.fetched.remove(bundle);
        }
        let mut actions = self.fold_state_proofs(trie, block);
        actions.extend(self.fold_verdict_records(block));
        // Every verdict this block carries resolves its transactions,
        // whichever way it went; what is left past every window that
        // could still carry one is nobody's to resolve.
        let released = self.ledger.release_resolved(block.certificates());
        self.released_fetches.extend(released);
        // What the block writes down about departed shards, before the
        // prune below reads what is still answerable.
        let rebuilt = self
            .ledger
            .record_abandonment_records(self.local_shard, block.abandonment_records());
        for _ in 0..rebuilt {
            record_rebuilt_verdict_entry();
        }
        self.cover_recorded(block);
        self.stamp_departures(topology_schedule, now);
        let pruned = self.ledger.prune(now);
        self.released_fetches.extend(pruned.released);
        actions.extend(self.release_answered_fetches());
        // The committed clock is what opens a leg's deadline, so the
        // cores gone silent past it are asked here.
        actions.extend(self.probe(trie, now));
        Committed {
            actions,
            unanswerable: pruned.unanswerable,
        }
    }

    /// Record a departed shard's settled set beside what this ledger
    /// says the shard was party to: a departure record may name only
    /// these, and the fence reads it from the same mirror.
    pub fn on_settled(&self, shard: ShardId, settled: SettledTxSet) {
        let parties = self.ledger.party_to(shard, settled.terminal_wt);
        self.mirror.record_settled(shard, settled, parties);
    }

    /// The departed shards whose settled sets the fence will need and
    /// this validator does not hold: every shard the schedule still
    /// routes to whose boundary record attests a settled root, until the
    /// handoff-anchored evidence window has closed at `now` (an
    /// unstamped window is open). Re-derived whole on every beacon fold,
    /// so a set acquired, a window closed or a shard evicted simply
    /// drops out.
    #[must_use]
    pub fn wanted_settled_sets(
        &self,
        topology_schedule: &TopologySchedule,
        now: WeightedTimestamp,
    ) -> Vec<TerminalEvidence> {
        let head = topology_schedule.head();
        topology_schedule
            .routable_shards()
            .into_iter()
            .filter(|&shard| shard != self.local_shard)
            .filter(|&shard| !self.mirror.with_settled(|sets| sets.contains_key(&shard)))
            .filter(|&shard| {
                topology_schedule
                    .handoff_evidence_expiry(shard)
                    .is_none_or(|expiry| now <= expiry)
            })
            .filter_map(|shard| {
                let anchor = head.boundary(shard)?;
                let attested_root = anchor.terminal_roots?.settled_txs;
                let terminal_wt = topology_schedule.terminal_cut_wt(shard)?;
                Some(TerminalEvidence {
                    shard,
                    height: anchor.height,
                    block_hash: anchor.block_hash,
                    terminal_wt,
                    attested_root,
                })
            })
            .collect()
    }

    /// What this validator holds to offer in a block it proposes.
    #[must_use]
    pub fn offers(&self) -> Offers {
        Offers {
            state_proofs: self.state_proofs(),
            abandonment_records: self.abandonment_records(),
        }
    }

    /// Ask each silent counterpart whether it took the transaction a
    /// leg here issued for, once the transaction's deadline has passed.
    ///
    /// The deadline gates the probe and never the reclaim: absence at a
    /// block past the floor is the evidence, and before it the
    /// counterpart may still legitimately act. A core of more than one
    /// shard is asked about the transaction's committed cell past the
    /// deadline, and every core shard is asked: any one of them absent
    /// is the whole answer, while the shards that did include say only
    /// that a sibling is pending, so asking one alone strands the
    /// crossing whenever that shard is the one that included. A core of
    /// one shard writes no cell and is asked about its consumer's claim
    /// instead. A delivering shard is asked about the crossing's claim
    /// cell past the lapse, the delivery window's close plus the
    /// finalization delay, since a delivery admitted under the close has
    /// claimed by then or never will. Each is asked against the newest commit-proven
    /// header of that shard inside its window — at or past its floor and
    /// short of the probed cell's own sweep, since a proof against a
    /// swept cell is a true proof of nothing — which is the header the
    /// shard is likeliest to still serve, a proof being taken from a
    /// bounded history behind its tip. A shard whose header has not
    /// reached here yet is asked when it does, and one whose every held
    /// header is past the window is not asked at all: the entry then
    /// waits out its horizon.
    ///
    /// A delivering shard that departs at a reshape may leave no header
    /// past the lapse at all, so the claim cell is asked about wherever
    /// its prefix sits: on the shard that was to deliver it and on the
    /// shard the trie names for its owner now, which is the successor
    /// holding the departed chain's cells. Both are asked rather than
    /// the trie's answer alone, because the vote fence checks a record
    /// against the voter's own proof of the shard it names, and two
    /// validators straddling the cut would otherwise prove different
    /// shards and never both vote one record.
    ///
    /// The cell is named from signed content and the counterpart shard
    /// alone, so nothing but the header and the proof is fetched.
    pub fn probe(&mut self, trie: &ShardTrie, now: WeightedTimestamp) -> Vec<Action> {
        let mut wanted: BTreeMap<Anchor, Vec<SubstateKey>> = BTreeMap::new();
        for entry in self.ledger.probeable(now) {
            for (shard, key, probed) in counterpart_cells(&entry, self.local_shard, trie) {
                // The chain has answered: nothing is asked again.
                if self.ledger.answered(entry.tx_hash, shard, probed)
                    || self.verified.contains(&(entry.tx_hash, shard, probed))
                {
                    continue;
                }
                // The newest header an absence would answer at: the one
                // the shard is likeliest to still serve, since a proof is
                // taken from a bounded history behind its tip. Where no
                // header answers one — the cue fired before the window
                // opened — the newest held will do, because the reading
                // the cue is after is a presence, and a presence answers
                // wherever it was taken.
                let Some(anchor) = self
                    .proven_anchors
                    .newest_licensed(shard, |ts| probed.absence_answers_at(ts, entry.deadline))
                    .or_else(|| self.proven_anchors.newest_licensed(shard, |_| true))
                else {
                    continue;
                };
                // A question in flight is left alone: a core's header
                // lands every block, and moving the probe to each new
                // one abandons the fetch before its answer returns. A
                // probe whose fetch has answered is moved on, which is
                // how a claim the chain read absent is asked again —
                // at a newer header, not of the same one every block.
                if self
                    .ledger
                    .probe_stands(entry.tx_hash, shard, probed, anchor.height)
                {
                    continue;
                }
                self.ledger
                    .record_probe(entry.tx_hash, shard, probed, anchor, key);
                wanted.entry(anchor).or_default().push(key);
            }
        }
        // The records this shard inherited ask one question each, of
        // whoever holds the claim's prefix now. Nothing is asked of a
        // claim this shard holds itself — the tick reads that cell
        // directly — and nothing is asked twice at one header.
        for record in self.inherited.values_mut() {
            if record.answer.is_some() {
                continue;
            }
            let claim = record.cell.consumer_claim;
            let shard = trie.shard_for_prefix(claim.owner);
            if shard == self.local_shard {
                continue;
            }
            let Some(anchor) = self.proven_anchors.newest_licensed(shard, |_| true) else {
                continue;
            };
            if record.asked_at.is_some_and(|asked| asked >= anchor.height) {
                continue;
            }
            record.asked_at = Some(anchor.height);
            wanted.entry(anchor).or_default().push(claim);
        }
        wanted
            .into_iter()
            .map(|(anchor, keys)| {
                Action::Fetch(FetchRequest::StateProof {
                    anchor,
                    keys,
                    preferred: None,
                    class: None,
                })
            })
            .collect()
    }

    /// Keep a fetched proof for a block this validator proposes.
    ///
    /// The fetch is only how the proposer comes by the bytes: nothing is
    /// read off the answer here, since the answer is the chain's once a
    /// block carries the proof and every replica folds it there. The
    /// probes the proof spoke to are marked answered, so the question is
    /// not put to the same header again, and the bundle is kept beside
    /// the transactions it answered for, dated to the clock the probe
    /// read off the header.
    pub fn on_proof_fetched(
        &mut self,
        anchor: Anchor,
        keys: Vec<SubstateKey>,
        proof: MerkleInclusionProof,
    ) {
        let answered = self.ledger.mark_probes_answered(anchor, &keys);
        // An answer nothing here asked about is nobody's to commit. An
        // inherited record names no transaction, so what it wants is
        // read off the keys.
        let inherited_wants = keys.iter().any(|key| awaited_by(&self.inherited, *key));
        if answered.is_empty() && !inherited_wants {
            return;
        }
        let bundle = StateProofBundle::new(anchor, keys, proof);
        // What the proof settles for this validator, on the rule a
        // committed proof is read by. A question it answers is not put
        // again; what the answer means is still the chain's to say, and
        // until a block carries this bundle nothing here has composed
        // anything from it.
        if let Ok(inclusions) = bundle.inclusions() {
            for entry in &answered {
                let Some(&(_, inclusion)) = inclusions.iter().find(|(key, _)| *key == entry.key)
                else {
                    continue;
                };
                if entry.probed.licenses(anchor.ts, entry.deadline, inclusion)
                    && entry.probed.read(inclusion, entry.core).is_some()
                {
                    self.verified
                        .insert((entry.tx_hash, entry.shard, entry.probed));
                }
            }
        }
        self.fetched
            .entry(bundle)
            .or_default()
            .extend(answered.iter().map(|entry| entry.tx_hash));
    }

    /// Fold the proofs a committed block carries into the answers every
    /// replica holds, and hand each to the vote fence.
    ///
    /// A bundle answers every cell of the ledger's on the anchor's shard
    /// whose window the anchor's clock sits inside — whether or not this
    /// replica had a probe out, and wherever its own probe sat — so a
    /// replica that never fetched reads the same answer as the one that
    /// did. A key found present means the counterpart took the
    /// transaction, and its own certificate speaks for it next: a
    /// refusal there is mirrored on arrival, and an acceptance is what
    /// settles the record held for the consumer's claim. A core
    /// consumer's claim absent on a core of more than one shard says
    /// only that a sibling is pending, and is asked again at the next
    /// header.
    /// The first proof to answer a cell is the answer; a later one adds
    /// nothing. The hand-off is a continuation emitted here rather than
    /// a map the fence reads later, so an answer is never collected
    /// before it is drained.
    fn fold_state_proofs(&mut self, trie: &ShardTrie, block: &Block) -> Vec<Action> {
        if block.state_proofs().is_empty() {
            return Vec::new();
        }
        let cells: Vec<(Probeable, Vec<CounterpartCell>)> = self
            .ledger
            .cells()
            .into_iter()
            .map(|entry| {
                let cells = counterpart_cells(&entry, self.local_shard, trie);
                (entry, cells)
            })
            .collect();
        let mut actions = Vec::new();
        for bundle in block.state_proofs() {
            actions.extend(self.fold_cells(bundle, &cells));
            self.fold_inherited(bundle, trie);
        }
        actions
    }

    /// Read a committed proof against the claims the inherited records
    /// are waiting on.
    ///
    /// Judged by the same per-word rule the ledger's cells are: a
    /// presence answers wherever it was taken, since the claim cell is
    /// written by the consuming execution and by nothing else; an
    /// absence answers only inside the window it means something in,
    /// which for a record whose consumer's role the leaf does not name
    /// is the lapse — past it no core of any arity can still commit, so
    /// the silence is final.
    fn fold_inherited(&mut self, bundle: &StateProofBundle, trie: &ShardTrie) {
        let Ok(inclusions) = bundle.inclusions() else {
            return;
        };
        for record in self.inherited.values_mut() {
            if record.answer.is_some() {
                continue;
            }
            let claim = record.cell.consumer_claim;
            if trie.shard_for_prefix(claim.owner) != bundle.anchor.shard {
                continue;
            }
            let Some(&(_, inclusion)) = inclusions.iter().find(|(asked, _)| *asked == claim) else {
                continue;
            };
            if !Probed::Delivery.licenses(bundle.anchor.ts, record.deadline(), inclusion) {
                continue;
            }
            record.answer = Some(if inclusion.is_present() {
                Word::Present
            } else {
                Word::Absent
            });
        }
    }

    /// Fold the verdicts a committed block's records restate: each is
    /// the counterpart's own word, folded from the chain, so a replica
    /// that never heard the certificate broadcast holds it from the
    /// block alone.
    fn fold_verdict_records(&self, block: &Block) -> Vec<Action> {
        let mut actions = Vec::new();
        for record in block.abandonment_records() {
            if let CounterpartEvidence::Heard(heard) = record.evidence()
                && heard.question == Question::Verdict
            {
                for entry in record.unsettled() {
                    actions.extend(self.fold_verdict(record.shard(), entry.tx_hash, heard));
                }
            }
        }
        actions
    }

    /// Fold one bundle's answers into the questions the ledger is
    /// waiting on.
    fn fold_cells(
        &mut self,
        bundle: &StateProofBundle,
        cells: &[(Probeable, Vec<CounterpartCell>)],
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        let inclusions = match bundle.inclusions() {
            Ok(inclusions) => inclusions,
            Err(error) => {
                tracing::error!(
                    shard = ?bundle.anchor.shard,
                    height = bundle.anchor.height.inner(),
                    %error,
                    "A committed state proof does not answer for its keys"
                );
                return actions;
            }
        };
        for (entry, cells) in cells {
            for &(shard, key, probed) in cells {
                if shard != bundle.anchor.shard {
                    continue;
                }
                let Some(&(_, inclusion)) = inclusions.iter().find(|(asked, _)| *asked == key)
                else {
                    continue;
                };
                // Judged per word, not per anchor: an absence is
                // read only inside its window, a presence wherever
                // it was taken.
                if !probed.licenses(bundle.anchor.ts, entry.deadline, inclusion) {
                    continue;
                }
                // Read by the one arity rule, whoever fetched the
                // proof: a probe never sent may still be answered
                // by a proof a block carries.
                let Some(inclusion) = probed.read(inclusion, entry.core.len()) else {
                    continue;
                };
                // The question is answered, and a fetch still out
                // for it is released with it.
                let Some(Released(released)) =
                    self.ledger.close_question(entry.tx_hash, shard, probed)
                else {
                    continue;
                };
                self.released_fetches.extend(released);
                record_reclaim_probe_answered(inclusion.is_present());
                let tx_hash = entry.tx_hash;
                match inclusion {
                    // The counterpart took it, and its certificate
                    // says how. Its broadcast may have missed this
                    // shard, so it is fetched rather than waited for.
                    Inclusion::Present(_) => {
                        actions.push(Action::Fetch(FetchRequest::ExecutionCerts {
                            source_shard: shard,
                            tx_hash,
                            preferred: None,
                            class: None,
                        }));
                    }
                    Inclusion::Absent => {
                        self.mirror.record(
                            tx_hash,
                            shard,
                            Heard {
                                question: Question::Cell(probed),
                                word: Word::Absent,
                                at: bundle.anchor.ts,
                            },
                        );
                    }
                }
                // A claim cell present is the settling word: that
                // cell is written by the consuming execution and by
                // nothing else, so its presence is the consumer
                // holding the crossing. A committed cell present
                // says only that the core committed the
                // transaction, which settles no record — its
                // certificate speaks to that, and is fetched above.
                if inclusion.is_present() && matches!(probed, Probed::Claim | Probed::Delivery) {
                    self.mirror.record(
                        tx_hash,
                        shard,
                        Heard {
                            question: Question::Cell(probed),
                            word: Word::Present,
                            at: bundle.anchor.ts,
                        },
                    );
                }
            }
        }
        actions
    }

    /// Write what the block's abandoning records cover into the mirror
    /// the gate and the fence read: neither asks a settled set about a
    /// transaction the chain has established no counterpart can settle.
    fn cover_recorded(&self, block: &Block) {
        for record in block.abandonment_records() {
            if record.evidence().abandons() {
                for tx_hash in record.tx_hashes() {
                    self.mirror.cover(tx_hash);
                }
            }
        }
    }

    /// Fold what `shard` said of `tx_hash` into the mirror the vote
    /// fence reads, and tell the mempool.
    ///
    /// Fed from two directions and read the same way from both: a
    /// certificate arriving by broadcast, and a record the chain
    /// committed, which is the counterpart's own word folded from the
    /// chain rather than from whatever this replica happened to hear —
    /// so a replica that came up between a core's verdict and the
    /// record's proposal holds the same answer its peers do. First
    /// write wins, as the chain's answer is: a second certificate or
    /// record restates a decision already held.
    ///
    /// Only a word this shard has a use for is kept. A core's refusal
    /// is the transaction's verdict, where a leg here issued for it, and
    /// a core shard's acceptance counts toward the transaction being
    /// accepted, which is every core shard saying so. A consumer's
    /// acceptance keeps nothing: it opens the probe whose answer settles
    /// the record held for its claim.
    pub fn fold_verdict(&self, shard: ShardId, tx_hash: TxHash, heard: Heard) -> Vec<Action> {
        if shard == self.local_shard || heard.question != Question::Verdict {
            return Vec::new();
        }
        let Word::Refused { decision, .. } = heard.word else {
            return Vec::new();
        };
        if self.ledger.core_holds(tx_hash, shard)
            && self.ledger.unsettled_figures(tx_hash).is_some()
            && self.mirror.record(tx_hash, shard, heard)
        {
            return vec![Action::Continuation(ProtocolEvent::TransactionsResolved {
                resolutions: vec![(tx_hash, TxResolution::CoreDecided(decision))],
            })];
        }
        Vec::new()
    }

    /// Fold a counterpart's claiming success: the cue to ask whether it
    /// wrote the claim, and — where the shard is in the core — one more
    /// core shard saying the transaction went through.
    ///
    /// Nothing is written down. A success is not evidence: its own
    /// finalization can still be refused afterwards, so what a record
    /// stands on is the claim cell proved present, and this only opens
    /// the question.
    pub fn fold_claimed(&mut self, shard: ShardId, tx_hash: TxHash) -> Vec<Action> {
        if shard == self.local_shard {
            return Vec::new();
        }
        if self.ledger.consumer_holds(tx_hash, shard) {
            self.ledger.cue_probe(tx_hash);
        }
        if self.ledger.core_holds(tx_hash, shard) && self.ledger.record_acceptance(tx_hash, shard) {
            return vec![Action::Continuation(ProtocolEvent::TransactionsResolved {
                resolutions: vec![(
                    tx_hash,
                    TxResolution::CoreDecided(TransactionDecision::Accept),
                )],
            })];
        }
        Vec::new()
    }

    /// The proofs this validator's own fetches answered that no block
    /// has carried yet, in the one order a block carries them, under
    /// the block's cap.
    fn state_proofs(&self) -> Vec<StateProofBundle> {
        self.fetched
            .keys()
            .take(MAX_STATE_PROOFS_PER_BLOCK)
            .cloned()
            .collect()
    }

    /// Drop the bundles no transaction they answered for still needs,
    /// and release every fetch the ledger let go — a question the chain
    /// answered first, or one whose entry is gone — so a counterpart
    /// that never serves the height does not pin the slot.
    fn release_answered_fetches(&mut self) -> Vec<Action> {
        // What this validator answered for itself goes with the entry it
        // answered about: past that there is nothing left to not ask.
        let held = &self.ledger;
        self.verified
            .retain(|(tx_hash, _, _)| held.contains(*tx_hash));
        let unresolved = &self.ledger;
        // A bundle is worth carrying while something still wants what it
        // answers: a transaction the ledger owes an outcome for, or an
        // inherited record whose claim it speaks to. The second has no
        // transaction here at all, which is why the keys are read rather
        // than the names.
        let inherited = &self.inherited;
        self.fetched.retain(|bundle, answered| {
            answered.iter().any(|tx_hash| unresolved.contains(*tx_hash))
                || bundle.keys.iter().any(|key| awaited_by(inherited, *key))
        });
        // The one retention rule for what counterparts said: an entry
        // there speaks for a transaction this ledger still owes an
        // outcome for, and the ledger is here.
        self.mirror.retain(&|tx_hash| unresolved.contains(tx_hash));
        let ids = std::mem::take(&mut self.released_fetches);
        if ids.is_empty() {
            Vec::new()
        } else {
            vec![Action::AbandonFetch(FetchIds::StateProofs(ids))]
        }
    }

    /// The records this shard has evidence for and has not yet written
    /// down — what each departed counterpart left of its business here.
    ///
    /// Composed from the settled sets, which is what bounds when this can
    /// speak at all: a set is acquired once the departed shard's terminal
    /// roots are attested and dropped at its evidence expiry, so a record
    /// is only ever offered while the evidence for it is readable, which
    /// is the same window every voter can check it in. Absence from a set
    /// is proof rather than ignorance — the set is complete and
    /// beacon-attested — so a transaction of ours it does not name is one
    /// that shard never settled and now never will.
    ///
    /// Bounded by [`MAX_UNSETTLED_PER_BLOCK`], one budget across every
    /// departure, with the remainder left for the next block.
    ///
    /// Ascending by shard, which is the one order a block may carry them
    /// in.
    fn abandonment_records(&self) -> Vec<AbandonmentRecord> {
        let mut budget = MAX_UNSETTLED_PER_BLOCK;
        // One record per shard and arm, ascending: a departure first,
        // since it covers everything the shard was party to, then one
        // per question for the shards still running, in the order the
        // block carries them.
        let mut records: BTreeMap<(ShardId, Option<Question>), AbandonmentRecord> = BTreeMap::new();
        // The sets are a hash map, so the shards are walked in sorted
        // order rather than its own: which departures the budget reaches
        // must not turn on a per-process iteration order.
        self.mirror.with_settled(|sets| {
            let mut shards: Vec<ShardId> = sets.keys().copied().collect();
            shards.sort_unstable();
            for shard in shards {
                if budget == 0 || records.len() == MAX_ABANDONMENT_RECORDS_PER_BLOCK {
                    break;
                }
                let settled = &sets[&shard];
                let mut unsettled = self.ledger.outstanding_with(shard, settled.terminal_wt);
                unsettled.retain(|entry| !settled.txs.contains(&entry.tx_hash));
                unsettled.truncate(budget);
                if unsettled.is_empty() {
                    continue;
                }
                budget -= unsettled.len();
                let record = AbandonmentRecord::departed(shard, settled.terminal_wt, unsettled);
                records.insert((shard, None), record);
            }
        });
        // What counterparts were heard to say, one record per shard and
        // question, at the shard's earliest anchor: a record states the
        // one moment every name in it was answered at, since one
        // spanning two satisfies the fence's equality check for neither,
        // and the rest waits a block. Nothing is offered beside a
        // departure, which answers for everything the shard was party
        // to. A presence is offered until the chain has it written down
        // and no longer: the mirror lives to the entry, and the entry to
        // the retirement, so a record offered past its own commit could
        // reach a block after the evidence every voter checks it against
        // has gone.
        let mut heard: HeardByQuestion = BTreeMap::new();
        for (tx_hash, shard, word) in self.mirror.all() {
            if matches!(word.word, Word::Present) && !self.ledger.claim_unrecorded(tx_hash, shard) {
                continue;
            }
            let Some(figures) = self.ledger.unsettled_figures(tx_hash) else {
                continue;
            };
            heard
                .entry((shard, word.question))
                .or_default()
                .entry((word.at, word.word))
                .or_default()
                .push(figures);
        }
        for ((shard, question), anchors) in heard {
            if budget == 0 || records.len() == MAX_ABANDONMENT_RECORDS_PER_BLOCK {
                break;
            }
            if records.contains_key(&(shard, None)) {
                continue;
            }
            let Some(((at, word), mut unsettled)) = anchors.into_iter().next() else {
                continue;
            };
            unsettled.truncate(budget);
            budget -= unsettled.len();
            let evidence = Heard { question, word, at };
            records.insert(
                (shard, Some(question)),
                AbandonmentRecord::heard(shard, evidence, unsettled),
            );
        }
        records.into_values().collect()
    }

    /// Record where each departed shard's chain ended, for the entries
    /// whose fate only that shard's settled set can decide.
    ///
    /// Read on every commit, while the schedule still carries the window
    /// that proves the terminal — the account outlives that window, and a
    /// departure it never recorded reads afterwards as a counterpart that
    /// never left. Re-run rather than gated on first sight, because the
    /// expiry is not knowable at the cut: the beacon stamps the handoff
    /// complete some epochs later, and the ledger's entry fills in on the
    /// first commit after the stamp lands.
    pub fn stamp_departures(
        &mut self,
        topology_schedule: &TopologySchedule,
        now: WeightedTimestamp,
    ) {
        for (shard, cut) in topology_schedule.departures_at(now) {
            if shard != self.local_shard {
                self.ledger.record_terminal(
                    shard,
                    cut,
                    topology_schedule.handoff_evidence_expiry(shard),
                );
            }
        }
        // A departure held open is asked about by name on every commit,
        // since the schedule lists it only while a retained window
        // carries the shard and the stamp lands on the head's boundary
        // record, which outlives that window. One whose evidence the
        // schedule no longer reads at all closes now — the same reading
        // the settled sets are dropped on — so an entry a record covers
        // against it retires with the set that could have answered.
        for shard in self.ledger.unstamped_departures() {
            if let Some(expiry) = topology_schedule.handoff_evidence_expiry(shard) {
                self.ledger.stamp_terminal(shard, expiry);
            } else if !topology_schedule.terminal_evidence_readable(shard, now) {
                self.ledger.stamp_terminal(shard, now);
            }
        }
    }

    /// Fold every verdict a certificate carries, before it is routed:
    /// the leg's tick settled long ago, so the certificate routes
    /// nowhere, and what it says is the one thing in it this shard still
    /// has a use for.
    pub fn on_certificate(&mut self, ec: &Arc<Verified<ExecutionCertificate>>) -> Vec<Action> {
        let shard = ec.shard_id();
        let mut actions = Vec::new();
        for (tx_hash, spoken) in ec.verdicts() {
            actions.extend(match spoken {
                Spoken::Refused(heard) => self.fold_verdict(shard, tx_hash, heard),
                Spoken::Claimed { .. } => self.fold_claimed(shard, tx_hash),
            });
        }
        actions
    }

    /// Whether `shard`'s settled set stands in for a commit proof of this
    /// certificate's source block.
    ///
    /// Membership means the transaction's certificate committed in the
    /// departed chain at or before its terminal, and the set itself was
    /// verified against the beacon-attested terminal root — a stronger
    /// statement than a commit proof of one source block, which is
    /// exactly what a departed chain can no longer supply. Every outcome
    /// must be named: one outside the set is a verdict the departed
    /// shard never settled, and a certificate naming nothing gives the
    /// set nothing to vouch for.
    pub fn settled_set_admits(
        &self,
        shard: ShardId,
        cert: &Verifiable<ExecutionCertificate>,
    ) -> bool {
        self.mirror.with_settled(|sets| {
            sets.get(&shard).is_some_and(|settled| {
                let outcomes = cert.tx_outcomes();
                !outcomes.is_empty()
                    && outcomes
                        .iter()
                        .all(|outcome| settled.txs.contains(&outcome.tx_hash()))
            })
        })
    }

    /// Drop settled sets past their evidence window. Past it the gate
    /// rejects any outcome naming the shard regardless of the set, so
    /// retaining it only leaks memory.
    fn gc_settled_sets(&self, topology_schedule: &TopologySchedule, now: WeightedTimestamp) {
        self.mirror
            .retain_departures(&|shard| topology_schedule.terminal_evidence_readable(shard, now));
    }
}
