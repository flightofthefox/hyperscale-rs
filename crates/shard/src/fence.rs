//! The vote fence: what this validator's own evidence says about the
//! claims a block makes on other chains.
//!
//! A block's content rules are deterministic over the block and the
//! committed chain, and every replica reaches the same verdict on them.
//! The fence is the other kind of check: a block claims that a departed
//! shard left a transaction unsettled, that a counterpart said a word
//! about it, that a core's header carries a root, that a predecessor
//! never committed a transaction — and a validator can only hold those
//! claims to what it has itself mirrored. So the fence has three answers
//! rather than two. A claim the evidence contradicts refuses the vote for
//! good; a claim the evidence cannot yet answer withholds it, and names
//! what would let it answer; and a block making no claim this validator
//! cannot attest to passes.
//!
//! Everything the fence reads is a mirror shared with the execution
//! coordinator — what counterparts said, which anchors are commit-proven,
//! what the predecessors answered — so a block passes here exactly when
//! the composer that offered its content would have offered it against
//! the same mirror.

use hyperscale_core::{Action, ProtocolEvent};
use hyperscale_metrics::record_verdict_claim_deferred;
use hyperscale_types::{
    AbandonmentRecord, Block, CounterpartEvidence, CounterpartMirror, Heard, ProvenAnchors,
    Question, SettledSetVerdict, ShardId, StateProofBundle, TopologySchedule, UnsettledTx,
    WeightedTimestamp, Word, settled_set_verdict,
};

use crate::precut::{Precut, PrecutStatus};

/// Why the fence withheld a vote.
#[derive(Debug)]
pub enum Withheld {
    /// This validator's evidence contradicts a claim the block makes;
    /// the block never gets this vote.
    Refused(String),
    /// This validator cannot yet check a claim the block makes; the
    /// block stays pending, and `wanted` asks for what would let it.
    Deferred {
        /// The claim, for the log.
        why: String,
        /// What to ask for, if anything, so the answer arrives.
        wanted: Vec<Action>,
    },
}

impl Withheld {
    const fn deferred(why: String) -> Self {
        Self::Deferred {
            why,
            wanted: Vec::new(),
        }
    }
}

/// The evidence a vote is fenced on, borrowed from the coordinator for
/// one judgment.
pub struct VoteFence<'a> {
    /// What counterparts said, and the departed shards' settled sets.
    pub evidence: &'a CounterpartMirror,
    /// The commit-proven remote headers.
    pub proven_anchors: &'a ProvenAnchors,
    /// The predecessors' answers about transactions opening before the
    /// chain's origin.
    pub precut: &'a Precut,
    /// Where this chain began; content anchored before it belongs to a
    /// predecessor.
    pub cut: WeightedTimestamp,
    /// The shard this validator votes on.
    pub local_shard: ShardId,
}

impl VoteFence<'_> {
    /// Judge every claim `block` makes, in the order the cheapest
    /// evidence answers: its finalizations against the settled sets, its
    /// records against the sets and the mirror, its state proofs against
    /// the proven anchors, and its pre-cut content against the
    /// predecessors' answers.
    ///
    /// # Errors
    ///
    /// The first claim this validator refuses or cannot yet check.
    pub fn judge(&self, schedule: &TopologySchedule, block: &Block) -> Result<(), Withheld> {
        let anchored_wt = block.header().parent_qc().weighted_timestamp();
        self.finalizations(schedule, block, anchored_wt)?;
        self.records(block)?;
        self.state_proofs(block)?;
        self.precut(block)
    }

    /// The split-boundary fence over a block's finalizations.
    ///
    /// Each finalization's claims on its counterparts' settled sets —
    /// a settlement on every shard whose certificate it carries, an
    /// abandonment on every shard a local outcome awaited and never
    /// heard from — are the ones [`hyperscale_types::Finalization::claims`]
    /// derives, so a voter judges a tick by the rule its proposer's gate
    /// composed it under. Record coverage is read off the mirror the
    /// execution coordinator writes at the record's commit, so a replica
    /// that came up between the record and the abandonment holds the
    /// same answer its peers do.
    ///
    /// Past-terminal-ness is read off the **anchored** snapshot at
    /// `anchored_wt` (the block's `parent_qc` weighted timestamp), never
    /// the head, so every replica voting this block reaches the same
    /// verdict. A shard evicted from every retained window is so far past
    /// its terminal that any claim naming it is unreachable everywhere —
    /// refused. A past-terminal shard whose settled set isn't known yet
    /// defers the vote; past the set's evidence window the claim is
    /// categorically unprovable and refuses.
    ///
    /// # Errors
    ///
    /// A claim the sets contradict, or one no held set can answer.
    pub fn finalizations(
        &self,
        schedule: &TopologySchedule,
        block: &Block,
        anchored_wt: WeightedTimestamp,
    ) -> Result<(), Withheld> {
        let claims = block
            .certificates()
            .iter()
            .flat_map(|fw| fw.claims(self.local_shard, |tx_hash| self.evidence.covers(tx_hash)));
        let verdict = self.evidence.with_settled(|settled| {
            settled_set_verdict(settled, schedule, self.local_shard, anchored_wt, claims)
        });
        match verdict {
            SettledSetVerdict::Pass => Ok(()),
            SettledSetVerdict::Reject => Err(Withheld::Refused(
                "finalization names a past-terminal shard that didn't settle it".into(),
            )),
            SettledSetVerdict::Defer => Err(Withheld::deferred(
                "settled set for a past-terminal shard unknown".into(),
            )),
        }
    }

    /// Whether the block's abandonment records are ones this voter can
    /// attest to.
    ///
    /// Each record is held to the evidence it claims, on the arm it
    /// carries. The figures each name restates are not this fence's to
    /// check: they are read off the committed body, which lives in the
    /// store, and so are checked by the delegated verification the vote
    /// also waits on.
    ///
    /// # Errors
    ///
    /// The first record whose evidence this validator contradicts or
    /// has not mirrored.
    pub fn records(&self, block: &Block) -> Result<(), Withheld> {
        block
            .abandonment_records()
            .iter()
            .try_for_each(|verdict| self.record_stands(verdict))
    }

    /// Whether one record's evidence stands for this validator, on the
    /// arm it carries. A departure is checked against this validator's
    /// own ledger and the departed shard's settled set; what a shard was
    /// heard to say is checked against this validator's own mirror of
    /// it, equality on the word and the moment.
    fn record_stands(&self, verdict: &AbandonmentRecord) -> Result<(), Withheld> {
        match verdict.evidence() {
            CounterpartEvidence::Departed { .. } => self.departure_stands(verdict),
            CounterpartEvidence::Heard(heard) => verdict
                .unsettled()
                .iter()
                .try_for_each(|entry| self.heard_stands(verdict.shard(), entry, heard)),
        }
    }

    /// Whether a departure record stands: this validator's own ledger
    /// says the departed shard was party to every name, and the shard's
    /// settled set names none of them. That the schedule attests the
    /// cut it names, inside the evidence window, is admission's rule.
    ///
    /// The set says what the departed shard settled; the ledger says
    /// what it was party to, and a record may name only that — a
    /// stranger to the departed shard is absent from its set trivially,
    /// and abandoning it would charge a payer for a transaction a live
    /// counterpart can still settle. The set is complete and
    /// beacon-attested, so absence from it is proof rather than
    /// ignorance; a voter that has not acquired it defers, since the
    /// record is only proposable inside the window the set can be read
    /// in, so a voter inside it either has the set or is about to.
    fn departure_stands(&self, verdict: &AbandonmentRecord) -> Result<(), Withheld> {
        let shard = verdict.shard();
        let stranger = self.evidence.with_parties(shard, |parties| {
            parties.map(|parties| {
                verdict
                    .tx_hashes()
                    .find(|tx_hash| !parties.contains(tx_hash))
            })
        });
        match stranger {
            None => {
                return Err(Withheld::deferred(format!(
                    "departure record's parties for {shard:?} not yet mirrored"
                )));
            }
            Some(Some(stranger)) => {
                return Err(Withheld::Refused(format!(
                    "abandonment record names {stranger}, which the departed shard {shard:?} \
                     was not party to"
                )));
            }
            Some(None) => {}
        }
        let settled = self.evidence.with_settled(|sets| {
            sets.get(&shard).map(|settled| {
                verdict
                    .tx_hashes()
                    .find(|tx_hash| settled.txs.contains(tx_hash))
            })
        });
        match settled {
            None => Err(Withheld::deferred(format!(
                "settled set of {shard:?} for an abandonment record unknown"
            ))),
            Some(Some(tx_hash)) => Err(Withheld::Refused(format!(
                "abandonment record names {tx_hash}, which its shard {shard:?} settled"
            ))),
            Some(None) => Ok(()),
        }
    }

    /// Whether one name's word stands for this validator: an answer to a
    /// cell question sits inside the window the question is meaningful
    /// in for the name's deadline, and the word and the moment are the
    /// ones this validator itself mirrored — off the counterpart's
    /// certificate, or off the proof the chain committed, which every
    /// replica folds at the same height. A voter holding no mirror
    /// cannot say and defers; one whose mirror disagrees refuses.
    /// Whether a heard answer stands. An acceptance is also a settlement
    /// claim on the shard that spoke it — a chain whose termination is
    /// scheduled can be cut before the finalization its certificate
    /// promises lands — so it is held to the verdict a finalization's
    /// claim would be, at the block's anchor.
    fn heard_stands(
        &self,
        shard: ShardId,
        entry: &UnsettledTx,
        heard: Heard,
    ) -> Result<(), Withheld> {
        if let Question::Cell(probed) = heard.question
            && heard.word != Word::Present
            && !probed.absence_answers_at(heard.at, entry.deadline)
        {
            return Err(Withheld::Refused(format!(
                "abandonment record probes {shard:?} for {} at {:?}, outside the window its \
                 deadline {:?} licenses",
                entry.tx_hash, heard.at, entry.deadline
            )));
        }
        // A presence is compared as a word and not as a moment. Its
        // anchor carries no meaning — the cell is written by the one
        // execution that consumes the crossing, so it is there or it is
        // not, whenever the reading was taken — and two validators that
        // probed at different headers of the same chain read the same
        // fact. Holding them to one header would refuse a record for a
        // race neither of them lost.
        match self.evidence.heard(entry.tx_hash, shard, heard.question) {
            Some(mirrored)
                if mirrored == heard
                    || (heard.word == Word::Present && mirrored.word == Word::Present) => {}
            Some(mirrored) => {
                return Err(Withheld::Refused(format!(
                    "abandonment record restates {heard:?} of {shard:?} for {}, which this \
                     validator reads as {mirrored:?}",
                    entry.tx_hash
                )));
            }
            None => {
                if heard.question == Question::Verdict {
                    record_verdict_claim_deferred();
                }
                return Err(Withheld::deferred(format!(
                    "abandonment record restates an answer of {shard:?} for {} this validator \
                     has not mirrored",
                    entry.tx_hash
                )));
            }
        }
        Ok(())
    }

    /// Whether the block's state proofs name anchors this voter can
    /// check.
    ///
    /// A bundle claims a root and a clock for a counterpart's height.
    /// Both are read off the commit-proven header this voter holds for
    /// it: one that agrees passes to the delegated proof walk, one that
    /// disagrees is refused, and a voter holding no proven header for
    /// the height defers and asks for its commit proof — the same
    /// deferral a provision takes on a source block not yet proven.
    /// The proposer probed at a header it held, so an honest bundle's
    /// anchor reaches every voter in the ordinary course. Every missing
    /// anchor is asked for at once, so one deferral fetches them all.
    ///
    /// # Errors
    ///
    /// A bundle whose anchor this validator's proven header disagrees
    /// with, or the set of anchors it has not proven.
    pub fn state_proofs(&self, block: &Block) -> Result<(), Withheld> {
        let mut wanted = Vec::new();
        for bundle in block.state_proofs() {
            self.anchor_stands(bundle, &mut wanted)?;
        }
        if wanted.is_empty() {
            Ok(())
        } else {
            Err(Withheld::Deferred {
                why: format!(
                    "state proofs name {} heights this validator has not commit-proven",
                    wanted.len()
                ),
                wanted,
            })
        }
    }

    /// Whether a bundle's anchor is one this voter has commit-proven, and
    /// whether every term of it — the root and the clock — agrees with
    /// the header held for it. An anchor not held is asked for.
    fn anchor_stands(
        &self,
        bundle: &StateProofBundle,
        wanted: &mut Vec<Action>,
    ) -> Result<(), Withheld> {
        let anchor = bundle.anchor;
        match self.proven_anchors.at(anchor.shard, anchor.height) {
            Some(held) if held == anchor => Ok(()),
            Some(held) => Err(Withheld::Refused(format!(
                "state proof names an anchor of {:?} at height {} (root {:?}, clock {:?}) this \
                 validator's commit-proven header disagrees with (root {:?}, clock {:?})",
                anchor.shard,
                anchor.height.inner(),
                anchor.state_root,
                anchor.ts,
                held.state_root,
                held.ts,
            ))),
            None => {
                wanted.push(Action::Continuation(ProtocolEvent::CommitProofNeeded {
                    source_shard: anchor.shard,
                    block_height: anchor.height,
                }));
                Ok(())
            }
        }
    }

    /// Which of `block`'s transactions belong to the chain that ran
    /// before this one: those whose validity window opened before the
    /// cut. A certificate anchored before the cut is admission's to
    /// refuse; a transaction is different, since the hazard is only what
    /// the predecessor actually *committed*. One submitted before the
    /// cut and never committed is harmless, and landing it here is its
    /// first inclusion. Refusing the whole class is the safe default a
    /// successor runs under until it can ask the finer question, and the
    /// predecessors' answers are what narrow it: per predecessor, each
    /// absence proven against a `committed_txs_root` this chain
    /// commit-proved.
    ///
    /// Unresolved defers rather than refuses. Every honest validator
    /// reaches the same verdict once the answer lands, so a slow answer
    /// costs a wait; refusing would spend a round on it instead and make
    /// the block look bad rather than early. A proven replay refuses the
    /// block whatever else is outstanding, so the scan runs to the end
    /// rather than deferring on the first unresolved transaction it
    /// meets.
    ///
    /// Provisions are left out. A batch carries its *source* shard's
    /// weighted timestamp where the cut is in this chain's, so a rule
    /// written on that comparison would refuse honest batches near the
    /// boundary — and a pre-cut batch can only provision transactions the
    /// rule above already refuses, so it is inert here.
    ///
    /// # Errors
    ///
    /// A transaction a predecessor committed, or one no predecessor has
    /// answered for yet.
    pub fn precut(&self, block: &Block) -> Result<(), Withheld> {
        let mut deferred = None;
        for tx in block
            .transactions()
            .iter()
            .filter(|tx| tx.validity_range().start_timestamp_inclusive < self.cut)
        {
            match self.precut.status(&tx.hash()) {
                PrecutStatus::Absent => {}
                PrecutStatus::Committed => {
                    return Err(Withheld::Refused(format!(
                        "transaction {} predates this chain's origin and a predecessor \
                         committed it",
                        tx.hash()
                    )));
                }
                PrecutStatus::Unresolved => deferred = Some(tx.hash()),
            }
        }
        deferred.map_or(Ok(()), |tx_hash| {
            Err(Withheld::deferred(format!(
                "pre-cut transaction {tx_hash} unresolved against the predecessors"
            )))
        })
    }
}
