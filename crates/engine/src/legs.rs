//! Which legs of a transaction this shard runs.
//!
//! Turns a transaction's derived legs plus a placement into the
//! [`LegPlan`] the kernel executes. Pure and node-independent: nothing
//! here consults the executing node, because two shards divide one
//! manifest separately and their answers have to agree, or a crossing is
//! issued that nobody claims.
//!
//! The star is sources → middle → sinks and the middle is one unit, so a
//! shard's work for one transaction is its inbound legs, its share of the
//! core if it is in the core set, and its outbound legs — and the only
//! thing any of it waits for is the arrivals its own legs consume. There
//! is no visit index and no re-entry.

use std::collections::{BTreeMap, BTreeSet};

use hyperscale_types::{EscrowedValue, ShardId, ShardTrie};
use hyperscale_vm_effects::{CrossingSite, StarShape, star_at};
use hyperscale_vm_kernel::{Crossed, ExecutionScope, LegPlan, PlanTooWide, Reclaim};
use hyperscale_vm_types::{Crossing, LegRole, LegShape, ProtocolHasher, SubstateKey};

use crate::sharding::TrieShardResolver;

/// Whether a transaction the classifier says decomposes is run leg by
/// leg.
///
/// A build-time fact, not a strategy threaded through the pipeline:
/// every replica of every shard answers alike, so the frozen
/// classification is the same answer everywhere. Off only under the
/// `whole-shape` feature — the comparison build the decomposition
/// scenarios are held against.
#[must_use]
pub const fn decomposition_enabled() -> bool {
    !cfg!(feature = "whole-shape")
}

/// The trie and the departing set, read together.
///
/// A caller taking one at this anchor and the other at another would
/// divide against one placement and wait against another.
#[derive(Clone, Copy, Debug)]
pub struct Placement<'a> {
    trie: &'a ShardTrie,
    leaving: &'a BTreeSet<ShardId>,
}

impl<'a> Placement<'a> {
    /// One placement: the shard partition and the shards leaving it.
    #[must_use]
    pub const fn new(trie: &'a ShardTrie, leaving: &'a BTreeSet<ShardId>) -> Self {
        Self { trie, leaving }
    }

    /// The partition this placement divides against.
    #[must_use]
    pub const fn trie(&self) -> &'a ShardTrie {
        self.trie
    }
}

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
/// committed: whether it divides, and the shards its core sits on.
///
/// Taken once, at one placement, and carried from there — every consumer
/// reads this and none re-derives it, so a reshape landing between
/// composition and execution cannot leave one shard running a whole
/// manifest while its counterpart waits to be sent half of it. Only
/// [`Self::freeze`] can answer that a transaction divides;
/// [`Self::whole`] is the always-correct answer for a caller with no
/// placement to freeze against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classified {
    decomposed: Decomposed,
    core: BTreeSet<ShardId>,
    delivering: BTreeSet<ShardId>,
}

impl Classified {
    /// Freeze `legs` against `placement`: the classifier's own answer,
    /// and the core set beside it.
    ///
    /// Whether a build runs divided shapes at all is
    /// [`decomposition_enabled`]'s question, asked by the coordinator
    /// that freezes — so this answers honestly wherever it is asked.
    #[must_use]
    pub fn freeze(legs: &[LegShape], placement: Placement<'_>) -> Self {
        let decomposed = decomposes(legs, placement);
        Self {
            decomposed,
            core: core_shards(legs, placement.trie()),
            delivering: if decomposed.holds() {
                delivering_shards(legs, placement.trie())
            } else {
                BTreeSet::new()
            },
        }
    }

    /// The whole shape on every participant, with no core set read.
    #[must_use]
    pub const fn whole() -> Self {
        Self {
            decomposed: Decomposed::WHOLE,
            core: BTreeSet::new(),
            delivering: BTreeSet::new(),
        }
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

/// What a member runs of its transaction: the shape its committing
/// block froze, or the reclaim of what a leg here issued.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Runs {
    /// The transaction as classified at commit — whole, or the legs
    /// this shard's placement gives it.
    Shape(Classified),
    /// No node at all: the crossings an inbound leg here issued, taken
    /// back on the evidence a committed record carries.
    Reclaim,
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

/// The shards outside the core that only deliver.
///
/// Every leg there is an outbound leg or an attesting one that stayed
/// home beside it, and at least one delivers. A shard with an inbound
/// leg issues a crossing under the transaction's window and is not
/// among them, and a core shard bears the verdict.
///
/// Read off [`star_of`]'s settled roles, so the attesting demotion is
/// decided once.
#[must_use]
pub fn delivering_shards(legs: &[LegShape], trie: &ShardTrie) -> BTreeSet<ShardId> {
    let star = star_of(legs, trie);
    let mut delivers = BTreeSet::new();
    let mut settles = BTreeSet::new();
    for (leg, role) in legs.iter().zip(&star.roles) {
        let shard = trie.shard_for_prefix(leg.target);
        match role {
            LegRole::Outbound => {
                delivers.insert(shard);
            }
            LegRole::Attesting => {}
            LegRole::Inbound | LegRole::Core => {
                settles.insert(shard);
            }
        }
    }
    delivers.difference(&settles).copied().collect()
}

/// Whether this transaction's legs run where their state lives.
///
/// Two conjuncts, and running whole is always correct, so an unsure
/// answer takes it. The classifier's verdict is the first. The second is
/// that no leg target sits on a departing shard: a record is written by
/// the issuing shard's verdict and read a round later, and a shard
/// leaving the trie has no round left, so the transaction would settle
/// on the issuer's certificate while the claim it is owed had nowhere to
/// be made.
#[must_use]
pub fn decomposes(legs: &[LegShape], placement: Placement<'_>) -> Decomposed {
    let resolver = TrieShardResolver {
        trie: placement.trie,
    };
    let star = star_at(legs, &resolver);
    let nobody_leaving = legs.iter().all(|leg| {
        !placement
            .leaving
            .contains(&placement.trie.shard_for_prefix(leg.target))
    });
    Decomposed(star.decomposes(legs, &resolver) && nobody_leaving)
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
}

/// The value edges that cross under `decomposed`.
///
/// None when the shape runs whole: every participant runs every node, so
/// nothing is handed between them.
#[must_use]
pub fn crossings_of(
    legs: &[LegShape],
    crossings: &[Crossing],
    decomposed: Decomposed,
    trie: &ShardTrie,
) -> Vec<CrossingEdge> {
    if !decomposed.holds() {
        return Vec::new();
    }
    let star = Star::of(legs, trie);
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
            })
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
            legs: LegPlan::whole(),
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
    /// A producer this shard runs sends an edge off it, and its frame
    /// reserved no single cell for the value to leave from.
    #[error("edge ({node}, {output}) departs from no reserved cell")]
    NoOrigin {
        /// The producing node.
        node: u32,
        /// Which of its outputs.
        output: u32,
    },
    /// This shard runs nothing of the transaction.
    #[error("this shard runs no leg of the transaction")]
    NotAParticipant,
    /// This shard issued nothing it could take back.
    #[error("this shard has no inbound crossing to reclaim")]
    NothingToReclaim,
    /// More crossings than one outcome can state a verdict for.
    #[error(transparent)]
    TooWide(#[from] PlanTooWide),
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
    decomposed: Decomposed,
    trie: &ShardTrie,
    local: ShardId,
) -> Result<ShardPlan, PlanDefect> {
    if !decomposed.holds() {
        return Ok(ShardPlan::whole());
    }
    let star = Star::of(legs, trie);
    let runs_here = |node: u32| star.running(node).contains(&local);
    let mut plan = LegPlan::whole();
    let mut participant = false;
    for node in 0..star.len() {
        if runs_here(node) {
            participant = true;
        } else {
            plan.skip(node);
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
        let origin = crossing.origin.ok_or(PlanDefect::NoOrigin {
            node: crossing.node,
            output: crossing.output,
        })?;
        let record = CrossingSite::record(
            &ProtocolHasher,
            producer.target,
            producer.intent,
            producer.local,
            crossing.output,
            producer.expiry_ms,
        );
        plan.departs(crossing.node, crossing.output, record, origin)?;
    }
    Ok(ShardPlan {
        legs: plan,
        scope: star.scope_for(local),
    })
}

/// What `local` takes back of a transaction whose core will never claim
/// it: every crossing an inbound leg here issued, claimed by that leg's
/// own target.
///
/// Only an inbound leg's crossings are ever reclaimed. Outbound value was
/// issued by a core that already committed, and nobody may take it back.
///
/// # Errors
///
/// [`PlanDefect`]: a crossing naming a node the legs do not have, or a
/// shard that issued nothing it could take back.
pub fn reclaim_for_shard(
    legs: &[LegShape],
    crossings: &[Crossing],
    trie: &ShardTrie,
    local: ShardId,
) -> Result<ShardPlan, PlanDefect> {
    let star = Star::of(legs, trie);
    let mut plan = LegPlan::whole();
    for node in 0..star.len() {
        plan.skip(node);
    }
    let mut reclaimed = false;
    for crossing in crossings {
        let producer = star.leg(crossing.node)?;
        if star.role(crossing.node) != LegRole::Inbound || star.home(crossing.node) != local {
            continue;
        }
        let Some(consumer) = star.consumer_of(crossing.node, crossing.output) else {
            continue;
        };
        if star.running(consumer).contains(&local) {
            continue;
        }
        let claim = CrossingSite::claim(
            &ProtocolHasher,
            producer.target,
            producer.intent,
            producer.local,
            crossing.output,
            producer.expiry_ms,
        );
        plan.reclaims(
            crossing.node,
            crossing.output,
            Reclaim {
                record: crossing.record,
                claim,
            },
        )?;
        reclaimed = true;
    }
    if !reclaimed {
        return Err(PlanDefect::NothingToReclaim);
    }
    Ok(ShardPlan {
        legs: plan,
        scope: star.scope_for(local),
    })
}

/// The anchored star with the placement facts every question here reads:
/// each node's home, the core set, and which shards run which node.
struct Star<'a> {
    legs: &'a [LegShape],
    trie: &'a ShardTrie,
    shape: StarShape,
    core: BTreeSet<ShardId>,
    consumers: BTreeMap<(u32, u32), u32>,
}

impl<'a> Star<'a> {
    fn of(legs: &'a [LegShape], trie: &'a ShardTrie) -> Self {
        let shape = star_of(legs, trie);
        let core = shape
            .core
            .iter()
            .map(|shard| ShardId::from_heap_index(shard.0))
            .collect();
        let consumers = legs
            .iter()
            .enumerate()
            .flat_map(|(index, leg)| {
                let index = u32::try_from(index).expect("a manifest has fewer than u32 nodes");
                leg.edges
                    .iter()
                    .map(move |edge| ((edge.source, edge.output), index))
            })
            .collect();
        Self {
            legs,
            trie,
            shape,
            core,
            consumers,
        }
    }

    fn len(&self) -> u32 {
        u32::try_from(self.legs.len()).expect("a manifest has fewer than u32 nodes")
    }

    fn leg(&self, node: u32) -> Result<&'a LegShape, PlanDefect> {
        self.legs
            .get(node as usize)
            .ok_or(PlanDefect::NoSuchNode { node })
    }

    /// The node's role once placement settled the attesting ones. A node
    /// past the manifest is core, the direction every unsure answer
    /// takes.
    fn role(&self, node: u32) -> LegRole {
        self.shape
            .roles
            .get(node as usize)
            .copied()
            .unwrap_or_default()
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
        let trie = self.trie.clone();
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

    fn crossing(legs: &[LegShape], node: u32, output: u32, origin: bool) -> Crossing {
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
            origin: origin.then(|| cell(producer.target, 1)),
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

    fn decomposed(legs: &[LegShape]) -> Decomposed {
        let leaving = BTreeSet::new();
        let verdict = decomposes(legs, Placement::new(&trie(), &leaving));
        assert!(verdict.holds(), "the fixture has to decompose");
        verdict
    }

    /// The whole shape on every participant, whatever the trie.
    #[test]
    fn a_whole_transaction_plans_the_whole_shape() {
        let legs = transfer();
        let plan = plan_for_shard(&legs, &[], &[], Decomposed::WHOLE, &trie(), low())
            .expect("a whole plan needs nothing");
        assert!(plan.legs.is_whole());
        assert!(plan.scope.covers(owner(0x22, true)));
        assert!(
            crossings_of(
                &legs,
                &[crossing(&legs, 1, 0, true)],
                Decomposed::WHOLE,
                &trie()
            )
            .is_empty()
        );
    }

    /// A transfer plans one inbound leg on the sender's shard and one
    /// outbound on the recipient's, with one crossing between them.
    #[test]
    fn a_transfer_divides_into_an_inbound_and_an_outbound_leg() {
        let legs = transfer();
        let crossings = vec![crossing(&legs, 1, 0, true)];
        let divided = decomposed(&legs);

        let edges = crossings_of(&legs, &crossings, divided, &trie());
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, low());
        assert_eq!(edges[0].to, BTreeSet::from([high()]));
        assert_eq!(core_shards(&legs, &trie()), BTreeSet::from([low()]));

        let sender = plan_for_shard(&legs, &crossings, &[], divided, &trie(), low())
            .expect("the sender's legs need no arrival");
        assert!(sender.legs.runs(0) && sender.legs.runs(1) && !sender.legs.runs(2));
        assert_eq!(
            sender
                .legs
                .departing(1, 0)
                .map(|departure| departure.origin),
            Some(cell(owner(0x11, false), 1)),
        );
        assert!(sender.scope.covers(owner(0x11, false)));
        assert!(!sender.scope.covers(owner(0x22, true)));

        let recipient = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(1, 0, 100)],
            divided,
            &trie(),
            high(),
        )
        .expect("the recipient's leg has its arrival");
        assert!(!recipient.legs.runs(0) && !recipient.legs.runs(1) && recipient.legs.runs(2));
        assert_eq!(
            recipient.legs.arrival(1, 0),
            Some(Crossed {
                resource: RESOURCE,
                amount: 100
            })
        );
        assert!(recipient.legs.claim(1, 0).is_some());
        assert!(recipient.scope.covers(owner(0x22, true)));
        assert!(!recipient.scope.covers(owner(0x11, false)));
    }

    /// A swap plans the sign-in, the withdraw and the deposit on the
    /// caller's shard, the venue on its own, and the core is one shard.
    #[test]
    fn a_swap_keeps_the_core_on_the_venue_alone() {
        let legs = swap();
        let crossings = vec![crossing(&legs, 1, 0, true), crossing(&legs, 2, 0, true)];
        let divided = decomposed(&legs);
        assert_eq!(core_shards(&legs, &trie()), BTreeSet::from([high()]));

        let caller = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(2, 0, 90)],
            divided,
            &trie(),
            low(),
        )
        .expect("the caller's legs have their arrival");
        assert!(caller.legs.runs(0) && caller.legs.runs(1) && !caller.legs.runs(2));
        assert!(caller.legs.runs(3));
        assert!(caller.legs.departing(1, 0).is_some());
        assert!(caller.legs.arrival(2, 0).is_some());

        let venue = plan_for_shard(
            &legs,
            &crossings,
            &[arrival(1, 0, 100)],
            divided,
            &trie(),
            high(),
        )
        .expect("the venue has its arrival");
        assert!(venue.legs.runs(2));
        assert!(!venue.legs.runs(0) && !venue.legs.runs(1) && !venue.legs.runs(3));
        assert!(venue.legs.arrival(1, 0).is_some());
        assert!(venue.legs.departing(2, 0).is_some());
        assert!(venue.scope.covers(owner(0x33, true)));
        assert!(!venue.scope.covers(owner(0x11, false)));
    }

    /// A consumer whose producer runs elsewhere needs its arrival, and a
    /// plan with none is a defect rather than a smaller plan.
    #[test]
    fn a_missing_arrival_is_a_defect() {
        let legs = transfer();
        let crossings = vec![crossing(&legs, 1, 0, true)];
        let divided = decomposed(&legs);
        assert_eq!(
            plan_for_shard(&legs, &crossings, &[], divided, &trie(), high()).err(),
            Some(PlanDefect::MissingArrival { node: 1, output: 0 }),
        );
    }

    /// A departing edge from a producer that reserved no single cell has
    /// nowhere for its value to leave from, and the plan says so rather
    /// than inventing a cell.
    #[test]
    fn a_departure_without_an_origin_is_a_defect() {
        let legs = transfer();
        let crossings = vec![crossing(&legs, 1, 0, false)];
        let divided = decomposed(&legs);
        assert_eq!(
            plan_for_shard(&legs, &crossings, &[], divided, &trie(), low()).err(),
            Some(PlanDefect::NoOrigin { node: 1, output: 0 }),
        );
    }

    /// A shard that runs nothing of the transaction is not a participant,
    /// and a crossing naming a node past the manifest is malformed.
    #[test]
    fn a_non_participant_and_a_malformed_crossing_are_defects() {
        let legs = swap();
        let divided = decomposed(&legs);
        let elsewhere = ShardTrie::uniform(2);
        let leaving = BTreeSet::new();
        let divided_deeper = decomposes(&legs, Placement::new(&elsewhere, &leaving));
        assert!(divided_deeper.holds());
        // Under a four-leaf trie the low owners sit at path 0 and the
        // venue at path 2, so leaf 1 runs nothing.
        assert_eq!(
            plan_for_shard(
                &legs,
                &[],
                &[],
                divided_deeper,
                &elsewhere,
                ShardId::leaf(2, 1)
            )
            .err(),
            Some(PlanDefect::NotAParticipant),
        );
        let past = Crossing {
            node: 9,
            ..crossing(&legs, 1, 0, true)
        };
        assert_eq!(
            plan_for_shard(
                &legs,
                &[past],
                &[arrival(2, 0, 90)],
                divided,
                &trie(),
                low()
            )
            .err(),
            Some(PlanDefect::NoSuchNode { node: 9 }),
        );
    }

    /// A shape reaching a departing shard does not decompose: running
    /// whole is always correct, and a record written for a shard with no
    /// round left would have nowhere to be claimed.
    #[test]
    fn a_shape_reaching_a_departing_shard_runs_whole() {
        let legs = transfer();
        let leaving = BTreeSet::from([high()]);
        assert!(!decomposes(&legs, Placement::new(&trie(), &leaving)).holds());
        let nobody = BTreeSet::new();
        assert!(decomposes(&legs, Placement::new(&trie(), &nobody)).holds());
    }

    /// The sender takes back exactly what its inbound leg issued, under
    /// its own target; the recipient issued nothing it could reclaim.
    #[test]
    fn a_reclaim_takes_back_the_inbound_crossing_alone() {
        let legs = swap();
        let crossings = vec![crossing(&legs, 1, 0, true), crossing(&legs, 2, 0, true)];
        let caller = reclaim_for_shard(&legs, &crossings, &trie(), low())
            .expect("the caller issued")
            .legs;
        let reclaimed: Vec<((u32, u32), Reclaim)> = caller.reclaimed().collect();
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(reclaimed[0].0, (1, 0));
        assert_eq!(reclaimed[0].1.record, crossings[0].record);
        assert_eq!(reclaimed[0].1.claim.key().owner, owner(0x11, false));
        assert!((0..4).all(|node| !caller.runs(node)));

        // The venue's crossing is outbound value: a core committed it,
        // and nobody takes it back.
        assert_eq!(
            reclaim_for_shard(&legs, &crossings, &trie(), high()).err(),
            Some(PlanDefect::NothingToReclaim),
        );
    }
}
