//! Which legs of a transaction this shard runs.
//!
//! Turns a transaction's derived legs plus a placement into the
//! [`LegPlan`] the kernel executes. Pure and node-independent: nothing
//! here consults the executing node, because two shards divide one
//! manifest separately and their answers have to agree, or a crossing is
//! issued that nobody claims. Every placement fact a plan reads is
//! frozen in the [`Classified`] the committing block took, so two
//! replicas planning one tick under different topology heads still plan
//! it the same.
//!
//! The star is sources → middle → sinks and the middle is one unit, so a
//! shard's work for one transaction is its inbound legs, its share of the
//! core if it is in the core set, and its outbound legs — and the only
//! thing any of it waits for is the arrivals its own legs consume. There
//! is no visit index and no re-entry.
//!
//! A leg whose home is in the core set is not a leg. It runs in the core
//! member on its shard, passes its value directly, and is replicated
//! where the core is: a shard's sides and an edge's claimants are both
//! read off which shards run a node, and a leg beside the core on its
//! own shard would otherwise be departed into a record its consumer, on
//! the same shard, could never be handed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use hyperscale_types::{Address, EscrowedValue, ShardId, ShardTrie};
use hyperscale_vm_effects::{CrossingSite, StarShape, star_at};
use hyperscale_vm_kernel::{Crossed, Departure, ExecutionScope, LegPlan, PlanFault};
use hyperscale_vm_types::{Crossing, LegRole, LegShape, ProtocolHasher, SubstateKey};

use crate::sharding::TrieShardResolver;

/// Whether a transaction's legs run where their state lives.
///
/// Frozen onto the transaction when its block commits and carried from
/// there, never re-derived downstream. Only [`decomposes`] says yes, so a
/// bare answer cannot be passed where a frozen one belongs; [`Self::WHOLE`]
/// is always correct and is what every execution ran before anything
/// else could run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decomposed(bool);

impl Decomposed {
    /// The whole shape on every participant.
    pub const WHOLE: Self = Self(false);

    /// Whether the legs run where their state lives.
    #[must_use]
    pub const fn holds(self) -> bool {
        self.0
    }
}

/// The classification frozen onto a transaction when its block
/// committed: whether it divides, each node's settled role, the shards
/// its core sits on, and the trie all of that was read against.
///
/// Taken once, at one placement, and carried from there — every consumer
/// reads this and none re-derives it, so a reshape landing between
/// composition and execution cannot leave one shard running a whole
/// manifest while its counterpart waits to be sent half of it, and two
/// replicas whose topology heads flipped at different moments plan one
/// tick alike. Only [`Self::freeze`] can answer that a transaction
/// divides; [`Self::whole`] is the always-correct answer for a caller
/// with no placement to freeze against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classified {
    decomposed: Decomposed,
    /// The trie the classification was read against, and the one every
    /// plan built from it resolves an owner through.
    trie: Arc<ShardTrie>,
    /// Each node's role once placement settled it: the classifier's,
    /// with a leg whose home is in the core set folded into the core.
    roles: Vec<LegRole>,
    core: BTreeSet<ShardId>,
    delivering: BTreeSet<ShardId>,
    /// The shards that run an outbound leg beside an inbound one: a
    /// swap's caller, which withdraws before the venue and banks after
    /// it. Never a core shard, whose legs are its core member's.
    mixed: BTreeSet<ShardId>,
}

/// Which of a shard's legs a member runs.
///
/// A shard outside the core may hold legs on both sides of it: inbound
/// ones that feed it and outbound ones that consume what it returns. The
/// two cannot be one admission, since the outbound legs wait on a
/// crossing the core issues only after the inbound ones have crossed to
/// it. So a shard's work divides into at most two members — an issuing
/// one that runs its inbound legs, its share of the core, and any
/// outbound leg whose producer runs beside it, and a delivering one that
/// runs the outbound legs whose producers run elsewhere, once their
/// arrivals land — and a shard with legs on one side only has the one.
/// A node's side is decided by where its edges come from, never by its
/// role alone, so a member's arrivals are exactly the edges that cross.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
    /// The inbound legs and the core share: what settles, what issues,
    /// what a verdict is drawn from.
    Issuing,
    /// The outbound legs: a delivery, admitted when what it consumes has
    /// crossed, deciding nothing and reserving nothing.
    Delivering,
}

impl Classified {
    /// Freeze `legs` against `trie`: the classifier's own answer, the
    /// settled roles and the core set beside it. Nothing else enters —
    /// the trie is the one placement fact, and it changes only at a cut,
    /// where the shard leaving it commits nothing more — so every shard
    /// and every replica committing the transaction under one trie
    /// freezes one shape.
    #[must_use]
    pub fn freeze(legs: &[LegShape], owners: &[Address], trie: &ShardTrie) -> Self {
        let trie = Arc::new(trie.clone());
        let star = star_of(legs, &trie);
        let decomposed =
            Decomposed(star.decomposes(legs, owners, &TrieShardResolver { trie: &trie }));
        let core: BTreeSet<ShardId> = star
            .core
            .iter()
            .map(|shard| ShardId::from_heap_index(shard.0))
            .collect();
        let consumers = consumers_of(legs);
        let roles = fold_beside_the_core(legs, &star.roles, &core, &trie, &consumers);
        let (delivering, mixed) = if decomposed.holds() {
            let star = Star::view(legs, &roles, &core, &trie, consumers);
            let (delivers, settles) = star.delivery_sides();
            (
                delivers.difference(&settles).copied().collect(),
                delivers.intersection(&settles).copied().collect(),
            )
        } else {
            (BTreeSet::new(), BTreeSet::new())
        };
        Self {
            decomposed,
            trie,
            roles,
            core,
            delivering,
            mixed,
        }
    }

    /// The whole shape on every participant, with no core set read.
    #[must_use]
    pub fn whole() -> Self {
        Self {
            decomposed: Decomposed::WHOLE,
            trie: Arc::new(ShardTrie::single()),
            roles: Vec::new(),
            core: BTreeSet::new(),
            delivering: BTreeSet::new(),
            mixed: BTreeSet::new(),
        }
    }

    /// The trie this classification was read against.
    #[must_use]
    pub fn trie(&self) -> &ShardTrie {
        &self.trie
    }

    /// Whether `shard` only delivers for this transaction: it sits
    /// outside the core and every leg it runs is a delivery, so nothing
    /// it does bears a verdict or issues a crossing. Such a member is
    /// admissible past the transaction's validity end, since the record
    /// cell it claims is bounded by its own sweep and not by the window.
    #[must_use]
    pub fn delivers_at(&self, shard: ShardId) -> bool {
        self.delivering.contains(&shard)
    }

    /// Whether `shard` runs an outbound leg beside an inbound one, and
    /// so runs this transaction as two members: an issuing one at its
    /// commit, and a delivering one once the core's output has crossed
    /// back to it.
    #[must_use]
    pub fn mixed_at(&self, shard: ShardId) -> bool {
        self.mixed.contains(&shard)
    }

    /// The side `shard`'s first member runs: its delivery where it only
    /// delivers, and everything that issues otherwise. A mixed shard's
    /// delivering member is registered by the issuing one's admission.
    #[must_use]
    pub fn first_side_at(&self, shard: ShardId) -> Side {
        if self.delivers_at(shard) {
            Side::Delivering
        } else {
            Side::Issuing
        }
    }

    /// Whether the legs run where their state lives.
    #[must_use]
    pub const fn decomposed(&self) -> Decomposed {
        self.decomposed
    }

    /// The shards the core's nodes sit on.
    #[must_use]
    pub const fn core(&self) -> &BTreeSet<ShardId> {
        &self.core
    }
}

/// One shard's member of a frozen transaction.
///
/// Every per-member quantity is a function of the frozen classification,
/// where the member runs, which of its shard's legs it takes and what the
/// transaction reaches. Derived once here and asked by name, so a
/// consumer reads the question it means and cannot reach another's answer
/// except through its own name.
///
/// The two that look alike stay apart, and the sets they read are why.
/// [`reaches_beyond`](Self::reaches_beyond) asks whether the transaction
/// touches another shard at all, off the participants;
/// [`abortable`](Self::abortable) asks whether *this member's*
/// settlement waits on another shard, off what it awaits. A leg of a
/// divided transaction reaches beyond and is not abortable — it awaits
/// only itself — and reading either for the other is how a member's
/// writes get held provisional that nothing can retract, or released
/// that something can.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    classified: Classified,
    local: ShardId,
    side: Side,
    /// Every shard the transaction touches. Beside the classification
    /// because a whole shape's member awaits all of them, and no frozen
    /// answer names them.
    participating: BTreeSet<ShardId>,
}

impl Member {
    /// The member `local` runs on `side` of a transaction frozen as
    /// `classified` and reaching `participating`.
    #[must_use]
    pub const fn of(
        classified: Classified,
        local: ShardId,
        side: Side,
        participating: BTreeSet<ShardId>,
    ) -> Self {
        Self {
            classified,
            local,
            side,
            participating,
        }
    }

    /// The member a caller with no placement to freeze against runs:
    /// the whole transaction, on its own shard, reaching nobody else.
    #[must_use]
    pub fn whole(local: ShardId) -> Self {
        Self::of(
            Classified::whole(),
            local,
            Side::Issuing,
            BTreeSet::from([local]),
        )
    }

    /// The classification this member was frozen against.
    #[must_use]
    pub const fn classified(&self) -> &Classified {
        &self.classified
    }

    /// The shard running it.
    #[must_use]
    pub const fn local(&self) -> ShardId {
        self.local
    }

    /// Which of its shard's legs it runs.
    #[must_use]
    pub const fn side(&self) -> Side {
        self.side
    }

    /// Every shard the transaction touches — who this tick's certificate
    /// is owed to, since any of them may need what a member escrowed.
    #[must_use]
    pub const fn reach(&self) -> &BTreeSet<ShardId> {
        &self.participating
    }

    /// Whether the transaction reaches beyond this shard — the fact that
    /// makes its writes provisional and its verdict a counterpart's to
    /// share. Off the participants, never off what this member awaits.
    #[must_use]
    pub fn reaches_beyond(&self) -> bool {
        self.participating.iter().any(|&shard| shard != self.local)
    }

    /// Whether a counterpart's verdict can still discard this member's
    /// effects after it executes. Off what this member awaits, never off
    /// what the transaction reaches.
    #[must_use]
    pub fn abortable(&self) -> bool {
        self.awaited().iter().any(|&shard| shard != self.local)
    }

    /// Whose certificate this member's settlement waits on: the whole
    /// core set for a member of it, every participant for a whole shape,
    /// and itself otherwise.
    #[must_use]
    pub fn awaited(&self) -> BTreeSet<ShardId> {
        if !self.classified.decomposed().holds() {
            self.participating.clone()
        } else if self.in_core() {
            self.classified.core().clone()
        } else {
            BTreeSet::from([self.local])
        }
    }

    /// Whether this shard's certificate decides the transaction: it does
    /// unless the member is a leg, whose transaction its core decides.
    #[must_use]
    pub fn decides(&self) -> bool {
        !self.classified.decomposed().holds() || self.in_core()
    }

    /// Whether the member only delivers: a leg that failed is the
    /// transaction's end on its shard, but a delivery that failed decides
    /// nothing — the value it claims stays in its cell for a later claim.
    #[must_use]
    pub fn delivers(&self) -> bool {
        self.classified.decomposed().holds() && self.side == Side::Delivering
    }

    /// Whether this shard's nodes sit in the core set.
    #[must_use]
    pub fn in_core(&self) -> bool {
        self.classified.core().contains(&self.local)
    }

    /// Whether this shard runs an outbound leg beside an inbound one, and
    /// so runs the transaction as two members.
    #[must_use]
    pub fn runs_both_sides(&self) -> bool {
        self.classified.mixed_at(self.local)
    }

    /// Whether this is the second member its shard runs of the
    /// transaction: a mixed shard's delivering one, whose issuing member
    /// took the block's reservation, settled the price and committed the
    /// signers' nullifiers, so this one does none of those.
    #[must_use]
    pub fn is_second(&self) -> bool {
        self.side == Side::Delivering && self.runs_both_sides()
    }
}

/// What a member runs of its transaction: the shape its committing
/// block froze, or housekeeping on the records a producer here left.
///
/// The two housekeeping arms name cells and not a manifest. That is what
/// lets a shard holding the record and no body compose them — a reshape
/// successor, whose store arrives as a prefix of leaves and whose ledger
/// begins empty — and it is why neither carries the classification the
/// shape arm does: the record leaf says which cells the member touches,
/// and the transaction is a name on the receipt rather than an input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Runs {
    /// The transaction as classified at commit — whole, or the legs
    /// this shard's placement gives it on its side.
    Shape(Member),
    /// No node at all: the records of the crossings a producer here
    /// issued, deleted on the evidence that every consumer claimed them.
    Retire {
        /// The record cells to delete.
        records: Vec<SubstateKey>,
    },
    /// No node at all: records this shard inherited with a prefix, each
    /// decided against the claim cell it names — credited back where
    /// that cell is absent inside the window an absence means something
    /// in, deleted where it is there.
    ///
    /// The one housekeeping member whose evidence is this shard's own
    /// state. The other two rest on a counterpart's committed record;
    /// a record arriving with a prefix has no counterpart left to ask,
    /// and the shard that inherited it holds both halves of the crossing
    /// or cannot decide it at all.
    Inherited {
        /// The record cells to decide.
        records: Vec<SubstateKey>,
    },
    /// No node at all: the crossings a producer here issued, taken back
    /// on the evidence that no consumer ever claimed them.
    Reclaim {
        /// The record cells to credit back and delete.
        records: Vec<SubstateKey>,
        /// Whether this shard's own certificate of the leg settled the
        /// price already. A leg that ran is determined, and burned it
        /// inside its writes at its own finalization; one that never ran
        /// — held for a bundle that never came — owes it still, and the
        /// reclaim's receipt is the one of this shard's left to carry it.
        charged: bool,
    },
}

/// The star `legs` implies under `trie` — the anchored half of the
/// classification.
///
/// It reaches `star_at`, so the write-free demotion is decided once, in
/// the classifier, rather than a second time here.
#[must_use]
pub fn star_of(legs: &[LegShape], trie: &ShardTrie) -> StarShape {
    star_at(legs, &TrieShardResolver { trie })
}

/// The shards the core's nodes sit on.
///
/// A read of [`star_of`] rather than its own fold: two implementations of
/// "which shards is the core on" would disagree exactly where the
/// tie-break bites.
#[must_use]
pub fn core_shards(legs: &[LegShape], trie: &ShardTrie) -> BTreeSet<ShardId> {
    star_of(legs, trie)
        .core
        .iter()
        .map(|shard| ShardId::from_heap_index(shard.0))
        .collect()
}

/// The settled roles with every leg beside the core folded into it: an
/// attesting node on a core shard, an inbound leg on a core shard feeding
/// a core node, an outbound leg on a core shard fed by one. Each then
/// runs in the core member and is replicated where the core is, so
/// nothing is departed between a producer and a consumer that run
/// together.
fn fold_beside_the_core(
    legs: &[LegShape],
    settled: &[LegRole],
    core: &BTreeSet<ShardId>,
    trie: &ShardTrie,
    consumers: &BTreeMap<(u32, u32), u32>,
) -> Vec<LegRole> {
    let role_of = |node: u32| settled.get(node as usize).copied().unwrap_or_default();
    (0u32..)
        .zip(settled.iter().zip(legs))
        .map(|(index, (&role, leg))| {
            if !core.contains(&trie.shard_for_prefix(leg.target)) {
                return role;
            }
            let beside_the_core = match role {
                LegRole::Attesting | LegRole::Core => true,
                LegRole::Inbound => leg_outputs(legs, index)
                    .filter_map(|output| consumers.get(&(index, output)))
                    .any(|&consumer| role_of(consumer) == LegRole::Core),
                LegRole::Outbound => leg
                    .edges
                    .iter()
                    .any(|edge| role_of(edge.source) == LegRole::Core),
            };
            if beside_the_core { LegRole::Core } else { role }
        })
        .collect()
}

/// Every `(source, output)` edge's consumer.
fn consumers_of(legs: &[LegShape]) -> BTreeMap<(u32, u32), u32> {
    legs.iter()
        .enumerate()
        .flat_map(|(index, leg)| {
            let index = u32::try_from(index).expect("a manifest has fewer than u32 nodes");
            leg.edges
                .iter()
                .map(move |edge| ((edge.source, edge.output), index))
        })
        .collect()
}

/// The outputs of `node` some edge consumes.
fn leg_outputs(legs: &[LegShape], node: u32) -> impl Iterator<Item = u32> + '_ {
    legs.iter()
        .flat_map(|leg| leg.edges.iter())
        .filter(move |edge| edge.source == node)
        .map(|edge| edge.output)
}

/// Whether this transaction's legs run where their state lives: the
/// classifier's verdict over `trie`, and nothing about when.
///
/// A shard scheduled to leave the trie divides like any other, in its
/// final window too: a record cell follows its prefix to the successor,
/// a claim or a delivery is a pull on whoever holds the prefix when it is
/// made, and a crossing the delivery window closes on unclaimed is
/// reclaimed on the successor's own proof of its absence. A rule that
/// read the block's window here would flip at the boundary into that
/// window while the trie did not, and two shards committing one
/// transaction on either side of it froze different shapes.
///
/// One conjunct is the planner's own: a sink's edges all come from its
/// own shard or none do, since a sink fed from both sides would need an
/// arrival its own shard's issuing member could only hand it through a
/// bundle to itself. Running whole is always correct, so such a shape
/// takes that.
#[must_use]
pub fn decomposes(legs: &[LegShape], owners: &[Address], trie: &ShardTrie) -> Decomposed {
    let resolver = TrieShardResolver { trie };
    let star = star_at(legs, &resolver);
    Decomposed(
        star.decomposes(legs, owners, &resolver)
            && no_sink_is_fed_from_both_sides(legs, &star, trie),
    )
}

/// Whether every outbound leg's producers all run where it does, or none
/// do.
fn no_sink_is_fed_from_both_sides(legs: &[LegShape], star: &StarShape, trie: &ShardTrie) -> bool {
    let core: BTreeSet<ShardId> = star
        .core
        .iter()
        .map(|shard| ShardId::from_heap_index(shard.0))
        .collect();
    let running = |node: u32| -> BTreeSet<ShardId> {
        match star.roles.get(node as usize).copied().unwrap_or_default() {
            LegRole::Core => core.clone(),
            LegRole::Inbound | LegRole::Outbound | LegRole::Attesting => legs
                .get(node as usize)
                .map(|leg| trie.shard_for_prefix(leg.target))
                .into_iter()
                .collect(),
        }
    };
    legs.iter().zip(&star.roles).all(|(leg, role)| {
        if *role != LegRole::Outbound {
            return true;
        }
        let home = trie.shard_for_prefix(leg.target);
        let beside: BTreeSet<bool> = leg
            .edges
            .iter()
            .map(|edge| running(edge.source).contains(&home))
            .collect();
        beside.len() <= 1
    })
}

/// One value edge whose producer and consumer do not run together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossingEdge {
    /// The shard whose verdict commits the record cell: the producing
    /// node's home.
    pub from: ShardId,
    /// The shards that claim it — every shard running the consumer that
    /// does not also run the producer. Several for a consumer inside a
    /// multi-shard core.
    pub to: BTreeSet<ShardId>,
    /// The producing node.
    pub node: u32,
    /// Which of its outputs the edge carries.
    pub output: u32,
    /// The record cell, under the producing node's target.
    pub record: SubstateKey,
    /// Whether an outbound leg consumes it — a delivery's arrival, which
    /// the delivering member of each claiming shard waits on rather than
    /// the issuing one.
    pub delivers: bool,
}

/// The value edges that cross under `classified`.
///
/// None when the shape runs whole: every participant runs every node, so
/// nothing is handed between them.
#[must_use]
pub fn crossings_of(
    legs: &[LegShape],
    crossings: &[Crossing],
    classified: &Classified,
) -> Vec<CrossingEdge> {
    if !classified.decomposed().holds() {
        return Vec::new();
    }
    let star = Star::of(legs, classified);
    crossings
        .iter()
        .filter_map(|crossing| {
            let consumer = star.consumer_of(crossing.node, crossing.output)?;
            let producer_runs = star.running(crossing.node);
            let to: BTreeSet<ShardId> = star
                .running(consumer)
                .difference(&producer_runs)
                .copied()
                .collect();
            (!to.is_empty()).then(|| CrossingEdge {
                from: star.home(crossing.node),
                to,
                node: crossing.node,
                output: crossing.output,
                record: crossing.record,
                delivers: star.role(consumer) == LegRole::Outbound,
            })
        })
        .collect()
}

/// The claim cells the deliveries of `local`'s issued crossings write,
/// each under the shard that delivers it.
///
/// For every crossing an inbound leg here produces whose consumer is a
/// core node: that consumer's claim cell and the shard holding it —
/// the consumer's own home, since a claim sits under its target
/// wherever else the core runs it. What a presence probe asks the core
/// about once the transaction's deadline has passed: a claim present
/// there says the core took the crossing, and the record here is
/// retired on it. Empty when the shape runs whole.
#[must_use]
pub fn core_claims(
    legs: &[LegShape],
    crossings: &[Crossing],
    classified: &Classified,
    local: ShardId,
) -> Vec<(ShardId, SubstateKey)> {
    if !classified.decomposed().holds() {
        return Vec::new();
    }
    let star = Star::of(legs, classified);
    crossings
        .iter()
        .filter_map(|crossing| {
            let producer = star.leg(crossing.node).ok()?;
            if star.role(crossing.node) == LegRole::Core || star.home(crossing.node) != local {
                return None;
            }
            let consumer = star.consumer_of(crossing.node, crossing.output)?;
            if star.role(consumer) != LegRole::Core || star.running(consumer).contains(&local) {
                return None;
            }
            let claim = CrossingSite::claim(
                &ProtocolHasher,
                star.leg(consumer).ok()?.target,
                producer.intent,
                producer.local,
                crossing.output,
                producer.expiry_ms,
            )
            .key();
            Some((star.home(consumer), claim))
        })
        .collect()
}

/// For every crossing a node here produces — an inbound leg's, or the
/// core's on a core shard — whose consumer is an outbound leg on another
/// shard: that consumer's claim cell and its home.
///
/// A delivery that never claimed leaves exactly this cell absent, which
/// is what a lapse probe asks the delivering shard about, and the
/// crossing is then the producer's to take back. Empty when the shape
/// runs whole, since nothing is then handed between shards.
#[must_use]
pub fn delivered_claims(
    legs: &[LegShape],
    crossings: &[Crossing],
    classified: &Classified,
    local: ShardId,
) -> Vec<(ShardId, SubstateKey)> {
    if !classified.decomposed().holds() {
        return Vec::new();
    }
    let star = Star::of(legs, classified);
    crossings
        .iter()
        .filter_map(|crossing| {
            let producer = star.leg(crossing.node).ok()?;
            if star.role(crossing.node) == LegRole::Outbound || star.home(crossing.node) != local {
                return None;
            }
            let consumer = star.consumer_of(crossing.node, crossing.output)?;
            if star.role(consumer) != LegRole::Outbound {
                return None;
            }
            let home = star.home(consumer);
            if home == local {
                return None;
            }
            let claim = CrossingSite::claim(
                &ProtocolHasher,
                star.leg(consumer).ok()?.target,
                producer.intent,
                producer.local,
                crossing.output,
                producer.expiry_ms,
            )
            .key();
            Some((home, claim))
        })
        .collect()
}

/// What one shard runs of a transaction, and the scope it judges under.
#[derive(Clone, Debug)]
pub struct ShardPlan {
    /// Which nodes this shard runs, what arrives for them, and what
    /// departs from them.
    pub legs: LegPlan,
    /// What this shard judges before any body runs: its own shard for a
    /// leg member, the whole core set for a core member.
    pub scope: ExecutionScope,
}

impl ShardPlan {
    /// The plan every execution ran before there was anything else to
    /// run: nothing skipped, nothing crossing, every owner in scope.
    #[must_use]
    pub fn whole() -> Self {
        Self {
            legs: LegPlan::whole(0),
            scope: ExecutionScope::whole(),
        }
    }
}

/// What a plan cannot be built from.
///
/// Every arm here is reachable only from a malformed input, and each is
/// stated rather than smoothed over: a plan that invented value would
/// credit an execution with something nobody certified, and a plan that
/// dropped one reaches its consumer as a missing producer edge — an
/// outcome priced to nobody.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanDefect {
    /// An edge or a crossing names a node the legs do not have.
    #[error("node {node} is past the manifest")]
    NoSuchNode {
        /// The index named.
        node: u32,
    },
    /// A consumer this shard runs takes an edge from a producer it does
    /// not run, and nothing attested arrived for it.
    #[error("nothing arrived for edge ({node}, {output})")]
    MissingArrival {
        /// The producing node.
        node: u32,
        /// Which of its outputs.
        output: u32,
    },
    /// This shard runs nothing of the transaction.
    #[error("this shard runs no leg of the transaction")]
    NotAParticipant,
    /// What the plan itself refuses: an edge acted on twice, an action
    /// disagreeing with who runs the node, or more crossings than one
    /// outcome can state a verdict for.
    #[error(transparent)]
    Fault(#[from] PlanFault),
}

/// What `local` runs of the transaction, what arrives for it, and what
/// departs from it.
///
/// `arrivals` is what committed bundles attested for the edges this
/// shard's nodes consume; `crossings` is every value edge's record cell
/// as the transaction derives it. Both are read, never derived.
///
/// # Errors
///
/// [`PlanDefect`], on its own terms — never a smaller plan.
pub fn plan_for_shard(
    legs: &[LegShape],
    crossings: &[Crossing],
    arrivals: &[EscrowedValue],
    classified: &Classified,
    local: ShardId,
    side: Side,
) -> Result<ShardPlan, PlanDefect> {
    if !classified.decomposed().holds() {
        return Ok(ShardPlan::whole());
    }
    let star = Star::of(legs, classified);
    let runs_here = |node: u32| star.runs(node, local, side);
    let mut plan = LegPlan::whole(legs.len());
    let mut participant = false;
    for node in 0..star.len() {
        if runs_here(node) {
            participant = true;
        } else {
            plan.skip(node)?;
        }
    }
    if !participant {
        return Err(PlanDefect::NotAParticipant);
    }
    for index in 0..star.len() {
        let consumer = star.leg(index)?;
        if !runs_here(index) {
            continue;
        }
        for edge in &consumer.edges {
            let producer = star.leg(edge.source)?;
            if runs_here(edge.source) {
                continue;
            }
            let arrived = arrivals
                .iter()
                .find(|value| (value.node, value.output) == (edge.source, edge.output))
                .ok_or(PlanDefect::MissingArrival {
                    node: edge.source,
                    output: edge.output,
                })?;
            let claim = CrossingSite::claim(
                &ProtocolHasher,
                consumer.target,
                producer.intent,
                producer.local,
                edge.output,
                producer.expiry_ms,
            );
            plan.arrives(
                edge.source,
                edge.output,
                Crossed {
                    resource: arrived.resource,
                    amount: arrived.amount,
                },
                claim,
            )?;
        }
    }
    for crossing in crossings {
        let producer = star.leg(crossing.node)?;
        if !runs_here(crossing.node) {
            continue;
        }
        let Some(consumer) = star.consumer_of(crossing.node, crossing.output) else {
            continue;
        };
        if runs_here(consumer) {
            continue;
        }
        let record = CrossingSite::record(
            &ProtocolHasher,
            producer.target,
            producer.intent,
            producer.local,
            crossing.output,
            producer.expiry_ms,
        );
        // The consumer's claim, named here because this is the last
        // reader of the manifest that holds both ends of the edge: the
        // record travels with the prefix and the manifest does not, so
        // a successor asking whether the crossing was taken has only
        // what the leaf says.
        let consumer_claim = CrossingSite::claim(
            &ProtocolHasher,
            star.leg(consumer)?.target,
            producer.intent,
            producer.local,
            crossing.output,
            producer.expiry_ms,
        )
        .key();
        plan.departs(
            crossing.node,
            crossing.output,
            Departure {
                site: record,
                consumer_claim,
            },
        )?;
    }
    Ok(ShardPlan {
        legs: plan,
        scope: star.scope_for(local),
    })
}

/// Every record a producer here wrote for a consumer running elsewhere,
/// in edge order.
///
/// What the retirement of this transaction deletes, once a committed
/// record says every such consumer claimed. Which shards those were is
/// the ledger's question, not this one's.
#[must_use]
pub fn records_to_retire(
    legs: &[LegShape],
    crossings: &[Crossing],
    classified: &Classified,
    local: ShardId,
) -> Vec<SubstateKey> {
    issued_from(legs, crossings, classified, local)
}

/// Every record a producer here wrote whose consumer runs elsewhere, in
/// edge order.
///
/// What the reclaim of this transaction credits back, once a committed
/// record says no such consumer can still claim. An inbound leg's
/// crossing is taken back when its core refuses or never answers; a
/// core's, when the delivery it was issued to lapses. Both credit the
/// cell the record names, so neither needs more than the key.
#[must_use]
pub fn records_to_reclaim(
    legs: &[LegShape],
    crossings: &[Crossing],
    classified: &Classified,
    local: ShardId,
) -> Vec<SubstateKey> {
    issued_from(legs, crossings, classified, local)
}

/// The record cells this shard's producing nodes wrote for consumers
/// running elsewhere.
///
/// One fold for both housekeeping members, because they name the same
/// cells and differ only in what a committed record licensed doing with
/// them.
fn issued_from(
    legs: &[LegShape],
    crossings: &[Crossing],
    classified: &Classified,
    local: ShardId,
) -> Vec<SubstateKey> {
    let star = Star::of(legs, classified);
    crossings
        .iter()
        .filter_map(|crossing| {
            let producer = star.leg(crossing.node).ok()?;
            if star.role(crossing.node) == LegRole::Outbound || star.home(crossing.node) != local {
                return None;
            }
            let consumer = star.consumer_of(crossing.node, crossing.output)?;
            if star.running(consumer).contains(&local) {
                return None;
            }
            Some(
                CrossingSite::record(
                    &ProtocolHasher,
                    producer.target,
                    producer.intent,
                    producer.local,
                    crossing.output,
                    producer.expiry_ms,
                )
                .key(),
            )
        })
        .collect()
}

/// The frozen star with the placement facts every question here reads:
/// each node's role and home, the core set, and which shards run which
/// node on which side — all off the classification, none re-derived.
struct Star<'a> {
    legs: &'a [LegShape],
    roles: &'a [LegRole],
    core: &'a BTreeSet<ShardId>,
    trie: &'a Arc<ShardTrie>,
    consumers: BTreeMap<(u32, u32), u32>,
}

impl<'a> Star<'a> {
    fn of(legs: &'a [LegShape], classified: &'a Classified) -> Self {
        Self::view(
            legs,
            &classified.roles,
            &classified.core,
            &classified.trie,
            consumers_of(legs),
        )
    }

    const fn view(
        legs: &'a [LegShape],
        roles: &'a [LegRole],
        core: &'a BTreeSet<ShardId>,
        trie: &'a Arc<ShardTrie>,
        consumers: BTreeMap<(u32, u32), u32>,
    ) -> Self {
        Self {
            legs,
            roles,
            core,
            trie,
            consumers,
        }
    }

    /// The side `node` runs on at `local`: a sink whose producers run
    /// elsewhere is a delivery, waiting on their arrival; everything
    /// else — a source, the core, a sink fed beside itself — issues.
    fn side_of(&self, node: u32, local: ShardId) -> Side {
        let delivers = self.role(node) == LegRole::Outbound
            && self.legs.get(node as usize).is_some_and(|leg| {
                leg.edges
                    .iter()
                    .any(|edge| !self.running(edge.source).contains(&local))
            });
        if delivers {
            Side::Delivering
        } else {
            Side::Issuing
        }
    }

    /// Whether `node` runs in `local`'s member on `side`.
    fn runs(&self, node: u32, local: ShardId, side: Side) -> bool {
        self.running(node).contains(&local) && self.side_of(node, local) == side
    }

    /// The shards running a delivery, and the shards running anything
    /// that issues. A shard in both runs the transaction as two members.
    fn delivery_sides(&self) -> (BTreeSet<ShardId>, BTreeSet<ShardId>) {
        let mut delivers = BTreeSet::new();
        let mut settles = BTreeSet::new();
        for node in 0..self.len() {
            for shard in self.running(node) {
                match (self.role(node), self.side_of(node, shard)) {
                    (LegRole::Attesting, _) => {}
                    (_, Side::Delivering) => {
                        delivers.insert(shard);
                    }
                    (_, Side::Issuing) => {
                        settles.insert(shard);
                    }
                }
            }
        }
        (delivers, settles)
    }

    fn len(&self) -> u32 {
        u32::try_from(self.legs.len()).expect("a manifest has fewer than u32 nodes")
    }

    fn leg(&self, node: u32) -> Result<&'a LegShape, PlanDefect> {
        self.legs
            .get(node as usize)
            .ok_or(PlanDefect::NoSuchNode { node })
    }

    /// The node's settled role. A node past the manifest is core, the
    /// direction every unsure answer takes.
    fn role(&self, node: u32) -> LegRole {
        self.roles.get(node as usize).copied().unwrap_or_default()
    }

    fn home(&self, node: u32) -> ShardId {
        self.legs
            .get(node as usize)
            .map_or(ShardId::ROOT, |leg| self.trie.shard_for_prefix(leg.target))
    }

    /// The shards that run `node`: every core shard for a core node, its
    /// home for a leg.
    fn running(&self, node: u32) -> BTreeSet<ShardId> {
        match self.role(node) {
            LegRole::Core => self.core.clone(),
            LegRole::Inbound | LegRole::Outbound | LegRole::Attesting => {
                BTreeSet::from([self.home(node)])
            }
        }
    }

    fn consumer_of(&self, node: u32, output: u32) -> Option<u32> {
        self.consumers.get(&(node, output)).copied()
    }

    /// What `local` judges: the core set if it is in it, itself
    /// otherwise.
    fn scope_for(&self, local: ShardId) -> ExecutionScope {
        let trie = Arc::clone(self.trie);
        if self.core.contains(&local) {
            let core = self.core.clone();
            ExecutionScope::spanning(move |owner| core.contains(&trie.shard_for_prefix(owner)))
        } else {
            ExecutionScope::spanning(move |owner| trie.shard_for_prefix(owner) == local)
        }
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::{Address, AddressClass, LocalKey, SubstateKey};
    use hyperscale_vm_effects::{Hash32, SubintentHash};
    use hyperscale_vm_types::{ResourceAddr, ValueEdge};

    use super::*;

    const RESOURCE: ResourceAddr = ResourceAddr::new([0xE1; 31]);

    /// An owner whose top bit is `high`, which is what a depth-one trie
    /// splits on.
    fn owner(seed: u8, high: bool) -> Address {
        let mut body = [seed; 31];
        body[0] = if high { 0x80 | seed } else { seed & 0x7F };
        Address::new(body, AddressClass::Component)
    }

    fn cell(owner: Address, slot: u8) -> SubstateKey {
        SubstateKey {
            owner,
            local: LocalKey([slot; 16]),
        }
    }

    fn leg(target: Address, role: LegRole, edges: &[(u32, u32)], local: u32) -> LegShape {
        LegShape {
            target,
            role,
            edges: edges
                .iter()
                .map(|(source, output)| ValueEdge {
                    source: *source,
                    output: *output,
                    non_fungible: false,
                })
                .collect(),
            presents: Vec::new(),
            declares: vec![target],
            intent: SubintentHash(Hash32([7; 32])),
            local,
            expiry_ms: 1_000,
        }
    }

    fn crossing(legs: &[LegShape], node: u32, output: u32) -> Crossing {
        let producer = &legs[node as usize];
        Crossing {
            node,
            output,
            record: CrossingSite::record(
                &ProtocolHasher,
                producer.target,
                producer.intent,
                producer.local,
                output,
                producer.expiry_ms,
            )
            .key(),
        }
    }

    fn arrival(node: u32, output: u32, amount: u128) -> EscrowedValue {
        EscrowedValue {
            node,
            output,
            resource: RESOURCE,
            amount,
            // The planner reads the edge, never the record: which cell a
            // bundle proved is the requirement's business.
            record: cell(owner(0xFF, true), 9),
        }
    }

    fn trie() -> ShardTrie {
        ShardTrie::uniform(1)
    }

    fn low() -> ShardId {
        ShardId::leaf(1, 0)
    }

    fn high() -> ShardId {
        ShardId::leaf(1, 1)
    }

    /// A transfer: sign-in and withdraw on the low shard, deposit on the
    /// high one. The sign-in is the only core, by demotion.
    fn transfer() -> Vec<LegShape> {
        let alice = owner(0x11, false);
        let bob = owner(0x22, true);
        vec![
            leg(alice, LegRole::Attesting, &[], 0),
            leg(alice, LegRole::Inbound, &[], 1),
            leg(bob, LegRole::Outbound, &[(1, 0)], 2),
        ]
    }

    /// A swap: sign-in, withdraw and deposit on the caller's low shard,
    /// the venue on the high one.
    fn swap() -> Vec<LegShape> {
        let alice = owner(0x11, false);
        let venue = owner(0x33, true);
        vec![
            leg(alice, LegRole::Attesting, &[], 0),
            leg(alice, LegRole::Inbound, &[], 1),
            leg(venue, LegRole::Core, &[(1, 0)], 2),
            leg(alice, LegRole::Outbound, &[(2, 0)], 3),
        ]
    }

    fn frozen(legs: &[LegShape]) -> Classified {
        let classified = Classified::freeze(legs, &[], &trie());
        assert!(
            classified.decomposed().holds(),
            "the fixture has to decompose"
        );
        classified
    }

    /// The claim cells a shard's issued crossings are owed by deliveries
    /// elsewhere: a transfer's withdraw is owed the deposit's claim on
    /// the recipient's shard, the recipient's shard issues nothing, and a
    /// swap's withdraw is consumed by the core, which is no delivery.
    #[test]
    fn delivered_claims_name_the_deliveries_of_what_a_shard_issued() {
        let legs = transfer();
        let crossings = [crossing(&legs, 1, 0)];
        let bob = owner(0x22, true);
        let expected = CrossingSite::claim(
            &ProtocolHasher,
            bob,
            legs[1].intent,
            legs[1].local,
            0,
            legs[1].expiry_ms,
        )
        .key();
        assert_eq!(
            delivered_claims(&legs, &crossings, &frozen(&legs), low()),
            vec![(high(), expected)],
        );
        assert!(
            delivered_claims(&legs, &crossings, &frozen(&legs), high()).is_empty(),
            "the delivering shard issued nothing",
        );
        assert!(
            delivered_claims(&legs, &crossings, &Classified::whole(), low()).is_empty(),
            "a whole shape hands nothing between shards",
        );

        let swap = swap();
        let crossings = [crossing(&swap, 1, 0), crossing(&swap, 2, 0)];
        assert!(
            delivered_claims(&swap, &crossings, &frozen(&swap), low()).is_empty(),
            "a crossing the core consumes is answered by the core, not a delivery",
        );
    }

    /// The whole shape on every participant, whatever the trie.
    #[test]
    fn a_whole_transaction_plans_the_whole_shape() {
        let legs = transfer();
        let plan = plan_for_shard(&legs, &[], &[], &Classified::whole(), low(), Side::Issuing)
            .expect("a whole plan needs nothing");
        assert!(plan.legs.is_whole());
        assert!(plan.scope.covers(owner(0x22, true)));
        assert!(crossings_of(&legs, &[crossing(&legs, 1, 0)], &Classified::whole()).is_empty());
    }

    /// A transfer plans one inbound leg on the sender's shard and one
    /// outbound on the recipient's, with one crossing between them.
    #[test]
    fn a_transfer_divides_into_an_inbound_and_an_outbound_leg() {
        let legs = transfer();
        let crossings = vec![crossing(&legs, 1, 0)];
        let divided = frozen(&legs);

        let edges = crossings_of(&legs, &crossings, &divided);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, low());
        assert_eq!(edges[0].to, BTreeSet::from([high()]));
        assert_eq!(core_shards(&legs, &trie()), BTreeSet::from([low()]));

        let sender = plan_for_shard(&legs, &crossings, &[], &divided, low(), Side::Issuing)
            .expect("the sender's legs need no arrival");
        assert!(sender.legs.runs(0) && sender.legs.runs(1) && !sender.legs.runs(2));
        assert!(sender.legs.departure(1, 0).is_some());
        assert!(sender.scope.covers(owner(0x11, false)));
        assert!(!sender.scope.covers(owner(0x22, true)));

        let recipient = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(1, 0, 100)],
            &divided,
            high(),
            Side::Delivering,
        )
        .expect("the recipient's leg has its arrival");
        assert!(!recipient.legs.runs(0) && !recipient.legs.runs(1) && recipient.legs.runs(2));
        assert_eq!(
            recipient.legs.arrival(1, 0).map(|arrival| arrival.crossed),
            Some(Crossed {
                resource: RESOURCE,
                amount: 100
            })
        );
        assert!(recipient.scope.covers(owner(0x22, true)));
        assert!(!recipient.scope.covers(owner(0x11, false)));
    }

    /// A swap plans the sign-in, the withdraw and the deposit on the
    /// caller's shard, the venue on its own, and the core is one shard.
    #[test]
    fn a_swap_keeps_the_core_on_the_venue_alone() {
        let legs = swap();
        let crossings = vec![crossing(&legs, 1, 0), crossing(&legs, 2, 0)];
        let divided = frozen(&legs);
        assert_eq!(core_shards(&legs, &trie()), BTreeSet::from([high()]));

        // The caller runs the transaction as two members: its issuing one
        // signs in and withdraws, waiting on nothing, and its delivering
        // one banks the venue's output once that has crossed back.
        let issuing = plan_for_shard(&legs, &crossings, &[], &divided, low(), Side::Issuing)
            .expect("the caller's issuing legs take no arrival");
        assert!(issuing.legs.runs(0) && issuing.legs.runs(1));
        assert!(!issuing.legs.runs(2) && !issuing.legs.runs(3));
        assert!(issuing.legs.departure(1, 0).is_some());
        assert!(issuing.legs.arrival(2, 0).is_none());
        let delivering = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(2, 0, 90)],
            &divided,
            low(),
            Side::Delivering,
        )
        .expect("the caller's delivering leg has its arrival");
        assert!(delivering.legs.runs(3));
        assert!(!delivering.legs.runs(0) && !delivering.legs.runs(1) && !delivering.legs.runs(2));
        assert!(delivering.legs.arrival(2, 0).is_some());
        assert!(delivering.legs.departure(1, 0).is_none());

        let venue = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(1, 0, 100)],
            &divided,
            high(),
            Side::Issuing,
        )
        .expect("the venue has its arrival");
        assert!(venue.legs.runs(2));
        assert!(!venue.legs.runs(0) && !venue.legs.runs(1) && !venue.legs.runs(3));
        assert!(venue.legs.arrival(1, 0).is_some());
        assert!(venue.legs.departure(2, 0).is_some());
        assert!(venue.scope.covers(owner(0x33, true)));
        assert!(!venue.scope.covers(owner(0x11, false)));
    }

    /// A consumer whose producer runs elsewhere needs its arrival, and a
    /// plan with none is a defect rather than a smaller plan.
    #[test]
    fn a_missing_arrival_is_a_defect() {
        let legs = transfer();
        let crossings = vec![crossing(&legs, 1, 0)];
        let divided = frozen(&legs);
        assert_eq!(
            plan_for_shard(&legs, &crossings, &[], &divided, high(), Side::Delivering).err(),
            Some(PlanDefect::MissingArrival { node: 1, output: 0 }),
        );
    }

    /// A core's crossing to a delivery elsewhere is the core shard's to
    /// take back when the delivery lapses, claimed under the core node's
    /// own target; the caller's shard, which issued the withdraw, takes
    /// back that one and never the venue's.
    #[test]
    fn a_core_shard_reclaims_what_it_issued_to_a_delivery() {
        let legs = swap();
        let crossings = vec![crossing(&legs, 1, 0), crossing(&legs, 2, 0)];
        let divided = frozen(&legs);
        let venue = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(1, 0, 100)],
            &divided,
            high(),
            Side::Issuing,
        )
        .expect("a core issues what it minted");
        assert!(venue.legs.departure(2, 0).is_some());

        let reclaimed = records_to_reclaim(&legs, &crossings, &divided, high());
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(reclaimed[0], crossings[1].record);
        assert_eq!(reclaimed[0].owner, owner(0x33, true));
        assert_eq!(
            delivered_claims(&legs, &crossings, &divided, high())
                .into_iter()
                .map(|(shard, _)| shard)
                .collect::<Vec<_>>(),
            vec![low()],
            "and its deliveries are the caller's shard's to make"
        );
    }

    /// A shard that runs nothing of the transaction is not a participant,
    /// and a crossing naming a node past the manifest is malformed.
    #[test]
    fn a_non_participant_and_a_malformed_crossing_are_defects() {
        let legs = swap();
        let divided = frozen(&legs);
        let elsewhere = ShardTrie::uniform(2);
        let divided_deeper = Classified::freeze(&legs, &[], &elsewhere);
        assert!(divided_deeper.decomposed().holds());
        // Under a four-leaf trie the low owners sit at path 0 and the
        // venue at path 2, so leaf 1 runs nothing.
        assert_eq!(
            plan_for_shard(
                &legs,
                &[],
                &[],
                &divided_deeper,
                ShardId::leaf(2, 1),
                Side::Issuing
            )
            .err(),
            Some(PlanDefect::NotAParticipant),
        );
        let past = Crossing {
            node: 9,
            ..crossing(&legs, 1, 0)
        };
        assert_eq!(
            plan_for_shard(
                &legs,
                &[past],
                &[arrival(2, 0, 90)],
                &divided,
                low(),
                Side::Issuing
            )
            .err(),
            Some(PlanDefect::NoSuchNode { node: 9 }),
        );
    }

    /// The sender takes back exactly what its inbound leg issued, under
    /// its own target, and never the venue's crossing; a shard that
    /// issued nothing has nothing to reclaim.
    #[test]
    fn a_reclaim_takes_back_the_inbound_crossing_alone() {
        let legs = swap();
        let crossings = vec![crossing(&legs, 1, 0), crossing(&legs, 2, 0)];
        let reclaimed = records_to_reclaim(&legs, &crossings, &frozen(&legs), low());
        assert_eq!(
            reclaimed.len(),
            1,
            "the venue's crossing is not the caller's"
        );
        assert_eq!(reclaimed[0], crossings[0].record);
        assert_eq!(reclaimed[0].owner, owner(0x11, false));

        let legs = transfer();
        let crossings = vec![crossing(&legs, 1, 0)];
        assert!(
            records_to_reclaim(&legs, &crossings, &frozen(&legs), high()).is_empty(),
            "the recipient's shard issued nothing"
        );
    }

    /// A leg whose home is a core shard is the core member's: the venue's
    /// output to a recipient on the venue's own shard is passed directly
    /// rather than departed into a record the shard could never be
    /// handed, and the shard has no delivering member at all.
    #[test]
    fn an_outbound_leg_on_a_core_shard_runs_in_the_core_member() {
        let bob = owner(0x22, false);
        let venue = owner(0x33, true);
        let recipient = owner(0x44, true);
        let legs = vec![
            leg(bob, LegRole::Attesting, &[], 0),
            leg(bob, LegRole::Inbound, &[], 1),
            leg(venue, LegRole::Core, &[(1, 0)], 2),
            leg(recipient, LegRole::Outbound, &[(2, 0)], 3),
        ];
        let crossings = [crossing(&legs, 1, 0), crossing(&legs, 2, 0)];
        let classified = frozen(&legs);
        assert!(
            !classified.mixed_at(high()),
            "the core shard runs one member"
        );
        assert!(!classified.delivers_at(high()));
        let edges = crossings_of(&legs, &crossings, &classified);
        assert_eq!(edges.len(), 1, "only the withdraw crosses");
        assert_eq!((edges[0].node, edges[0].output), (1, 0));

        let core = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(1, 0, 5)],
            &classified,
            high(),
            Side::Issuing,
        )
        .expect("the core member runs the venue and the deposit");
        assert!(core.legs.runs(2) && core.legs.runs(3));
        assert!(
            core.legs.departure(2, 0).is_none(),
            "the venue's output stays in the execution"
        );
        assert_eq!(
            plan_for_shard(
                &legs,
                &crossings,
                &[],
                &classified,
                high(),
                Side::Delivering
            )
            .err(),
            Some(PlanDefect::NotAParticipant),
        );
        assert!(records_to_reclaim(&legs, &crossings, &classified, high()).is_empty());
    }

    /// An inbound leg on one shard of a multi-shard core is replicated
    /// with the core: nothing is promised to the other core shard, and
    /// each plans the withdraw beside the venues.
    #[test]
    fn an_inbound_leg_on_a_core_shard_is_replicated_with_the_core() {
        fn owner_at(seed: u8, path: u8) -> Address {
            let mut body = [seed; 31];
            body[0] = (path << 6) | (seed & 0x3F);
            Address::new(body, AddressClass::Component)
        }
        let trie = ShardTrie::uniform(2);
        let (leaf0, leaf1, leaf2) = (
            ShardId::leaf(2, 0),
            ShardId::leaf(2, 1),
            ShardId::leaf(2, 2),
        );
        let legs = vec![
            leg(owner_at(0x11, 0), LegRole::Attesting, &[], 0),
            leg(owner_at(0x11, 0), LegRole::Inbound, &[], 1),
            leg(owner_at(0x12, 0), LegRole::Core, &[(1, 0)], 2),
            leg(owner_at(0x13, 2), LegRole::Core, &[(2, 0)], 3),
            leg(owner_at(0x14, 1), LegRole::Outbound, &[(3, 0)], 4),
        ];
        let crossings = [
            crossing(&legs, 1, 0),
            crossing(&legs, 2, 0),
            crossing(&legs, 3, 0),
        ];
        let classified = Classified::freeze(&legs, &[], &trie);
        assert!(classified.decomposed().holds());
        assert_eq!(classified.core(), &BTreeSet::from([leaf0, leaf2]));
        let edges = crossings_of(&legs, &crossings, &classified);
        assert_eq!(
            edges.iter().map(|edge| edge.node).collect::<Vec<_>>(),
            vec![3],
            "only the second venue's output to the deposit crosses"
        );
        assert_eq!(edges[0].to, BTreeSet::from([leaf1]));
        for shard in [leaf0, leaf2] {
            let plan = plan_for_shard(&legs, &crossings, &[], &classified, shard, Side::Issuing)
                .expect("a core shard plans the withdraw beside the venues");
            assert!(
                plan.legs.runs(0) && plan.legs.runs(1) && plan.legs.runs(2) && plan.legs.runs(3)
            );
            assert!(plan.legs.departure(1, 0).is_none());
            assert!(plan.legs.departure(3, 0).is_some());
        }
    }

    /// A sink fed beside itself issues: an outbound leg whose producer
    /// runs on its own shard's issuing member takes the value directly,
    /// while one fed by the core elsewhere is that shard's delivery.
    #[test]
    fn a_sink_fed_beside_itself_runs_in_the_issuing_member() {
        let alice = owner(0x11, false);
        let carol = owner(0x55, false);
        let venue = owner(0x33, true);
        let legs = vec![
            leg(alice, LegRole::Attesting, &[], 0),
            leg(alice, LegRole::Inbound, &[], 1),
            leg(venue, LegRole::Core, &[(1, 0)], 2),
            leg(alice, LegRole::Outbound, &[(2, 0)], 3),
            leg(alice, LegRole::Inbound, &[], 4),
            leg(carol, LegRole::Outbound, &[(4, 0)], 5),
        ];
        let crossings = [
            crossing(&legs, 1, 0),
            crossing(&legs, 2, 0),
            crossing(&legs, 4, 0),
        ];
        let classified = frozen(&legs);
        assert!(
            classified.mixed_at(low()),
            "the venue's return is a delivery"
        );
        let edges = crossings_of(&legs, &crossings, &classified);
        assert_eq!(
            edges.iter().map(|edge| edge.node).collect::<Vec<_>>(),
            vec![1, 2],
            "the local transfer's edge never crosses"
        );
        let issuing = plan_for_shard(&legs, &crossings, &[], &classified, low(), Side::Issuing)
            .expect("the issuing member runs both withdraws and the local deposit");
        assert!(issuing.legs.runs(1) && issuing.legs.runs(4) && issuing.legs.runs(5));
        assert!(!issuing.legs.runs(3));
        assert!(issuing.legs.departure(1, 0).is_some());
        assert!(issuing.legs.departure(4, 0).is_none());
        let delivering = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(2, 0, 7)],
            &classified,
            low(),
            Side::Delivering,
        )
        .expect("the delivering member runs the venue's return alone");
        assert!(delivering.legs.runs(3) && !delivering.legs.runs(5));
    }

    /// A sink fed from both sides of its own shard runs whole: its
    /// issuing member could only hand it the local edge through a bundle
    /// to itself.
    #[test]
    fn a_sink_fed_from_both_sides_runs_whole() {
        let alice = owner(0x11, false);
        let venue = owner(0x33, true);
        let legs = vec![
            leg(alice, LegRole::Attesting, &[], 0),
            leg(alice, LegRole::Inbound, &[], 1),
            leg(venue, LegRole::Core, &[(1, 0)], 2),
            leg(alice, LegRole::Inbound, &[], 3),
            leg(alice, LegRole::Outbound, &[(2, 0), (3, 0)], 4),
        ];
        assert!(!decomposes(&legs, &[], &trie()).holds());
        let mut one_sided = legs;
        one_sided[4] = leg(alice, LegRole::Outbound, &[(2, 0)], 4);
        one_sided[3] = leg(alice, LegRole::Outbound, &[], 3);
        assert!(decomposes(&one_sided, &[], &trie()).holds());
    }
}
