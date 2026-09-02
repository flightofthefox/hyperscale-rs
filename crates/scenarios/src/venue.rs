//! A venue: one pool priced by callers on another shard, and what each
//! of them is left holding.
//!
//! A swap is the shape the leg-local rule exists for. The withdraw runs
//! as a leg on the caller's shard and certifies alone; the pricing runs
//! at the venue, which is the core and the only shard whose verdict
//! decides the swap; the deposit is a delivery back. So the caller is
//! the leg and the venue is the core, exactly as a stake into a remote
//! pool is, and what these scenarios pin is what each side is left
//! holding after the venue accepts, refuses, or is never reached.
//!
//! The pair is XRD against a pool's stake unit. Value enters the world
//! through a mint and nowhere else, so the second side is minted the way
//! anything else is: an account stakes, and what it is handed back is a
//! resource the venue can price.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use hyperscale_engine::XRD;
use hyperscale_engine::genesis::stake_unit;
use hyperscale_types::{
    ComponentAddr, Ed25519PrivateKey, MIN_STAKE_FLOOR, PrincipalAddr, ProtocolHasher, ResourceAddr,
    ShardId, SubstateKey, Transaction, TransactionDecision, TransactionStatus, TxHash,
};
use hyperscale_vm_effects::{InstanceMeta, Value, child_key};
use hyperscale_vm_fixtures::amm;

use crate::support::query::{declared_price, held, held_at, vault_balance};
use crate::support::tx::{
    GENESIS_POOL_ID, account_routing_to_n, build_add_liquidity_tx, build_instantiate_tx,
    build_stake_tx, build_swap_tx, pool_at, validity_around, venue_on,
};
use crate::support::wait::await_tx_terminal;
use crate::support::{Budget, Cluster, epochs};

/// What the liquidity provider is funded with: enough to stake the
/// floor, stock the pool's XRD side, and pay for both.
pub const PROVIDER_FUNDING: u128 = 8 * MIN_STAKE_FLOOR.attos();

/// What each swapping account is funded with.
pub const SWAPPER_FUNDING: u128 = 200_000_000;

/// What one swap pays in.
pub const SWAP_INPUT: u128 = 2_000_000;

/// How many accounts swap against the venue at once.
pub const SWAPPERS: usize = 8;

/// The shard the venue sits on, and so the shard whose cells every swap
/// contends for.
pub const VENUE_SHARD: ShardId = ShardId::leaf(1, 0);

/// The shard the swappers sit on when they share one — not the venue's,
/// so every swap is a crossing and the caller's withdraw is a leg.
pub const SWAPPER_SHARD: ShardId = ShardId::leaf(1, 1);

/// The venue's shard in a four-shard world, and the three the swappers
/// spread across.
///
/// One caller shard measures a round trip; three measure fan-in, which
/// is the shape a hot venue actually takes — a pool priced by users who
/// do not share a shard with it or with each other.
pub const WIDE_VENUE_SHARD: ShardId = ShardId::leaf(2, 0);

/// The caller shards in that world.
#[must_use]
pub fn wide_swapper_shards() -> Vec<ShardId> {
    vec![
        ShardId::leaf(2, 1),
        ShardId::leaf(2, 2),
        ShardId::leaf(2, 3),
    ]
}

/// The provider and the swappers, with the funding each needs.
///
/// Genesis seeds accounts and nothing else, so everything the venue
/// holds is put there by a transaction — which is what makes this a
/// measurement of the ordinary path rather than of a seeded fixture.
#[must_use]
pub fn venue_genesis_accounts() -> Vec<(PrincipalAddr, u128)> {
    venue_genesis_accounts_on(VENUE_SHARD, &[SWAPPER_SHARD])
}

/// The same, for a venue on `venue_shard` called from `caller_shards` —
/// dealt round-robin, so the swappers spread evenly however many shards
/// they have.
#[must_use]
pub fn venue_genesis_accounts_on(
    venue_shard: ShardId,
    caller_shards: &[ShardId],
) -> Vec<(PrincipalAddr, u128)> {
    let mut taken = Vec::new();
    let mut accounts = vec![(grind_onto(venue_shard, &mut taken).1, PROVIDER_FUNDING)];
    accounts.extend(
        swappers_on(caller_shards, &mut taken)
            .into_iter()
            .map(|(_, account)| (account, SWAPPER_FUNDING)),
    );
    accounts
}

/// An account on `shard`, whatever depth the world's partition has: the
/// uniform trie a leaf at that depth belongs to has `2^depth` shards.
pub fn grind_onto(shard: ShardId, taken: &mut Vec<u8>) -> (Ed25519PrivateKey, PrincipalAddr) {
    account_routing_to_n(shard, 1u64 << shard.depth(), taken)
}

/// The accounts that swap, dealt round-robin across the caller shards so
/// the venue's fan-in is as wide as the world it is given.
fn swappers_on(
    caller_shards: &[ShardId],
    taken: &mut Vec<u8>,
) -> Vec<(Ed25519PrivateKey, PrincipalAddr)> {
    (0..SWAPPERS)
        .map(|index| grind_onto(caller_shards[index % caller_shards.len()], taken))
        .collect()
}

/// The cell a venue keeps its reserve of `resource` in.
///
/// A venue's reserves live in the package's own state rather than in the
/// protocol vault slot an address answers for, so nothing that resolves
/// an address reaches them. Reading the reserve is what makes a claim
/// about what a venue holds, where spending only makes one about what it
/// can still afford.
#[must_use]
pub fn reserve_cell(venue: &InstanceMeta, resource: ResourceAddr) -> SubstateKey {
    child_key(
        &ProtocolHasher,
        venue.address(&ProtocolHasher).address(),
        amm::RESERVES,
        &[Value::Address(resource.address()).canonical_bytes()],
    )
}

/// A venue on `shard`, sealed and stocked by a provider of its own.
pub struct StockedVenue {
    /// The venue's metadata, which every call is typed against.
    pub meta: InstanceMeta,
    /// The stake unit the venue prices XRD against.
    pub unit: ResourceAddr,
}

/// Stand a venue up on `shard`: seat its provider, seal it, and stock
/// both sides of its pair.
///
/// # Panics
///
/// Panics if the stake, the seal or the stocking misses its budget.
pub fn stand_up_venue<C: Cluster>(c: &mut C, shard: ShardId, taken: &mut Vec<u8>) -> StockedVenue {
    let (provider_key, provider_account) = grind_onto(shard, taken);
    let pool = pool_at(GENESIS_POOL_ID);
    let unit = stake_unit(pool);
    let meta = venue_on(shard, (*XRD, unit));
    stock_venue(c, &provider_key, provider_account, pool, &meta, unit);
    StockedVenue { meta, unit }
}

/// Seal the venue and stock both sides of its pair.
pub fn stock_venue<C: Cluster>(
    c: &mut C,
    key: &Ed25519PrivateKey,
    account: PrincipalAddr,
    pool: ComponentAddr,
    meta: &InstanceMeta,
    unit: ResourceAddr,
) {
    // The second side of the pair, minted the only way value is: the
    // provider stakes, and the units it is handed are what the venue
    // prices XRD against.
    accepted(
        c,
        build_stake_tx(
            key,
            account,
            pool,
            MIN_STAKE_FLOOR.attos(),
            validity_around(c.now()),
        ),
        "the provider must hold the side it is about to stock",
    );
    // The seal, in a transaction of its own: a call's fence reads a
    // committed leaf, so the venue has to be actual before anything
    // reaches it.
    accepted(
        c,
        build_instantiate_tx(key, std::slice::from_ref(meta), validity_around(c.now())),
        "the venue must be sealed before it can be called",
    );
    // Staking mints through the pool, which sits on the pool's own
    // shard, so the units land a settlement after the stake reports
    // terminal. A status read is not enough to sequence against either:
    // it polls every hosted store, a terminated predecessor's included,
    // and one of those can answer for a transaction whose effects the
    // live shard has yet to apply. Drive to the balance the next call
    // spends instead.
    assert!(
        c.run_until(epochs(8), |c| held(c, account.address(), unit)
            >= MIN_STAKE_FLOOR.attos()),
        "the provider's stake must mint the units its venue is stocked with",
    );
    accepted(
        c,
        build_add_liquidity_tx(
            key,
            account,
            meta,
            (*XRD, unit),
            (MIN_STAKE_FLOOR.attos(), MIN_STAKE_FLOOR.attos()),
            validity_around(c.now()),
        ),
        "the venue must hold both sides before it can quote",
    );
}

/// Submit `tx` and hold the run to its acceptance.
pub fn accepted<C: Cluster>(c: &mut C, tx: Transaction, context: &str) {
    let hash = tx.hash();
    c.submit(Arc::new(tx));
    let status = await_tx_terminal(c, hash, epochs(12));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "{context}; status = {status:?}",
    );
}

/// Submit `tx` and hold the run to a verdict, whichever way it went.
fn terminal<C: Cluster>(c: &mut C, tx: Transaction, budget: Budget) -> Option<TransactionStatus> {
    let hash = tx.hash();
    c.submit(Arc::new(tx));
    await_tx_terminal(c, hash, budget)
}

/// A swap the venue accepts charges its caller its input and one price,
/// exactly.
///
/// The caller's own shard runs the withdraw as a leg and banks the
/// result as a delivery, and the fee is the transaction's rather than
/// either's: what the payer's vault is out is the input it swapped plus
/// one declared price, and a shard that charged per visit would take
/// two. Read as an exact figure rather than a bound, because the failure
/// this pins is a second charge of exactly the same size, which every
/// bound loose enough to survive a re-pricing would admit.
///
/// The venue's reserve of XRD rises by the input: what the caller paid
/// is what the core claimed, and nothing was minted or stranded on the
/// way.
///
/// # Panics
///
/// Panics if the venue misses its budget standing up, if the swap does
/// not accept, if the caller is charged anything but its input and one
/// price, or if the venue's reserve moved by anything but the input.
pub fn a_swap_charges_its_caller_its_input_and_one_price<C: Cluster>(c: &mut C, budget: Budget) {
    let mut taken = Vec::new();
    let venue = stand_up_venue(c, VENUE_SHARD, &mut taken);
    let (caller_key, caller) = grind_onto(SWAPPER_SHARD, &mut taken);

    // A floor the pool clears, so the swap settles and the burn is the
    // success one rather than a refusal's.
    let swap = build_swap_tx(
        &caller_key,
        caller,
        &venue.meta,
        *XRD,
        SWAP_INPUT,
        0,
        validity_around(c.now()),
    );
    let price = declared_price(c, &swap);
    let funded = vault_balance(c, SWAPPER_SHARD, caller);
    let reserve = reserve_cell(&venue.meta, *XRD);
    let stocked = held_at(c, reserve);
    assert!(
        stocked > 0,
        "the venue has to be holding something to price against"
    );

    let status = terminal(c, swap, budget);
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "the swap must accept for its success burn to be the one measured; \
         status = {status:?}",
    );

    // The measurement waits for the delivery, and the wait is the point:
    // the deposit that banks the output lands a hop after the venue's
    // verdict, so a burn charged per visit is not yet visible at the
    // verdict and shows only once the credit does.
    let delivered = c.run_until(budget, |c| held(c, caller.address(), venue.unit) > 0);
    assert!(
        delivered,
        "the swap's output never reached its caller, so the delivery that \
         banks it never ran and what it charges cannot be counted",
    );

    let kept = vault_balance(c, SWAPPER_SHARD, caller);
    assert_eq!(
        funded.saturating_sub(kept),
        SWAP_INPUT + price,
        "a swap costs its caller its input and one price: {funded} before, \
         {kept} after, input {SWAP_INPUT}, price {price}",
    );
    assert_eq!(
        held_at(c, reserve),
        stocked + SWAP_INPUT,
        "the venue claims exactly what the caller paid in",
    );
}

/// A swap the venue refuses gives its caller back what its leg took, and
/// leaves the venue holding what it held.
///
/// The withdraw runs and certifies on the caller's own shard before the
/// venue has said anything, so the caller is the participant left with
/// a leg that already moved value. The venue declines, its certificate
/// is mirrored into a refusal record on the caller's chain, and the
/// record licenses the reclaim that credits the vault from the record
/// cell the leg wrote. What the caller is out afterwards is the price
/// and nothing else.
///
/// Read off the vault rather than by spending: a caller funded for many
/// swaps pays for the next one out of what it kept, so a swap that
/// quietly took the input would still read as a pass. The venue's
/// reserve is read too, because a venue that wrongly kept the input
/// prices the next swap more easily, so a follow-on that accepts would
/// otherwise stand in for the very failure it is meant to rule out.
///
/// # Panics
///
/// Panics if the venue misses its budget standing up, if the refused
/// swap does not refuse, if its caller does not get its input back, if
/// the venue's reserve moved, or if the swap after it does not accept.
pub fn a_swap_the_venue_refuses_gives_its_caller_back_its_leg<C: Cluster>(
    c: &mut C,
    budget: Budget,
) {
    let mut taken = Vec::new();
    let venue = stand_up_venue(c, VENUE_SHARD, &mut taken);
    let (caller_key, caller) = grind_onto(SWAPPER_SHARD, &mut taken);

    // A floor no pool this size can pay: a swap of `SWAP_INPUT` returns
    // less than it took in, so anything above the input declines.
    let refused = build_swap_tx(
        &caller_key,
        caller,
        &venue.meta,
        *XRD,
        SWAP_INPUT,
        SWAP_INPUT * 100,
        validity_around(c.now()),
    );
    let price = declared_price(c, &refused);
    let funded = vault_balance(c, SWAPPER_SHARD, caller);
    let reserve = reserve_cell(&venue.meta, *XRD);
    let stocked = held_at(c, reserve);
    assert!(
        stocked > 0,
        "the venue has to be holding something to price against"
    );

    // The caller's own leg certifies first and reports the swap accepted
    // on its chain, which is a claim about the leg and not the verdict;
    // the venue's refusal follows on the venue's chain, and the reclaim
    // it licenses is a block of the caller's own after that. So nothing
    // is read off the first terminal status: what is asserted is the
    // reclaim, inside the same budget the swap itself was given.
    //
    // The input came back and the price did not: a refusal costs what
    // the success it displaced would have. Asserting the difference
    // rather than an inequality against the input is what makes this
    // separate a reclaim from a price that happens to exceed it.
    let _ = terminal(c, refused, budget);
    let back = c.run_until(budget, |c| {
        funded.saturating_sub(vault_balance(c, SWAPPER_SHARD, caller)) == price
    });
    let kept = vault_balance(c, SWAPPER_SHARD, caller);
    assert!(
        back,
        "a refused swap must give its caller back what its leg took and \
         charge it the price: {funded} before, {kept} after, on an input \
         of {SWAP_INPUT} priced at {price}",
    );
    assert_eq!(
        held_at(c, reserve),
        stocked,
        "a venue that refused a swap holds exactly what it held",
    );

    // That the venue can still price is the weaker claim, and the one
    // that says the swap after this is not running against a wedged
    // pool.
    let again = build_swap_tx(
        &caller_key,
        caller,
        &venue.meta,
        *XRD,
        SWAP_INPUT,
        0,
        validity_around(c.now()),
    );
    let status = terminal(c, again, budget);
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "a refused swap must leave its caller funded and its venue \
         priceable; status = {status:?}",
    );
}

/// A swap refused at its own inbound leg never reaches the venue, and
/// still pays.
///
/// The caller withdraws more than it holds, so the leg fails on the
/// caller's shard before anything crosses: no record cell is written,
/// the venue never runs a thing, its reserve is untouched, and the
/// caller is out the declared price — an attempt the network committed,
/// reserved for and ran a batch for, whatever refused it.
///
/// # Panics
///
/// Panics if the venue misses its budget standing up, if the swap does
/// not refuse, if a record cell was written for it, if the venue's
/// reserve moved, or if the caller is charged anything but the price.
pub fn a_swap_refused_at_its_inbound_leg_never_reaches_the_venue<C: Cluster>(
    c: &mut C,
    budget: Budget,
) {
    let mut taken = Vec::new();
    let venue = stand_up_venue(c, VENUE_SHARD, &mut taken);
    let (caller_key, caller) = grind_onto(SWAPPER_SHARD, &mut taken);

    let funded = vault_balance(c, SWAPPER_SHARD, caller);
    assert!(
        funded > 0,
        "the caller has to hold something for the overdraw to be one"
    );
    let overdrawn = build_swap_tx(
        &caller_key,
        caller,
        &venue.meta,
        *XRD,
        funded * 2,
        0,
        validity_around(c.now()),
    );
    let price = declared_price(c, &overdrawn);
    let record = overdrawn
        .try_derived(c.derivation().as_ref())
        .expect("a scenario fixture derives")
        .crossings
        .first()
        .expect("a swap crosses to its venue")
        .record;
    let reserve = reserve_cell(&venue.meta, *XRD);
    let stocked = held_at(c, reserve);
    assert!(
        stocked > 0,
        "the venue has to be holding something to price against"
    );

    let status = terminal(c, overdrawn, budget);
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Reject))
        ),
        "an overdrawn withdraw refuses on the caller's own shard; status = {status:?}",
    );

    let charged = c.run_until(budget, |c| {
        funded.saturating_sub(vault_balance(c, SWAPPER_SHARD, caller)) == price
    });
    let kept = vault_balance(c, SWAPPER_SHARD, caller);
    assert!(
        charged,
        "a leg refused before it issued still pays the price: {funded} before, \
         {kept} after, price {price}",
    );
    assert!(
        c.substate(SWAPPER_SHARD, record.owner, record.local.0)
            .is_none(),
        "a leg that could not issue writes no record cell",
    );
    assert_eq!(
        held_at(c, reserve),
        stocked,
        "the venue never ran, so it holds exactly what it held",
    );
}

/// What one hot-venue run observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VenueReport {
    /// How many swaps were submitted together.
    pub submitted: usize,
    /// From the first submission to the last settlement.
    pub elapsed: Duration,
    /// The median swap's time from the queue opening to its own
    /// settlement.
    pub latency_p50: Duration,
}

/// Stand a venue up and drive every swapper at it at once.
///
/// # Panics
///
/// Panics if the grow, the seal, the stocking or any swap misses its
/// budget, or if a swap does not accept — a refused swap would mean the
/// pool ran dry or the floor was set wrong, either of which makes the
/// measurement meaningless rather than merely worse.
pub fn hot_venue_clears_swaps<C: Cluster>(c: &mut C, budget: Budget) -> VenueReport {
    hot_venue_clears_swaps_on(c, VENUE_SHARD, &[SWAPPER_SHARD], budget)
}

/// The same, for a venue on `venue_shard` its callers reach from
/// `caller_shards`.
///
/// # Panics
///
/// As [`hot_venue_clears_swaps`], and if a caller shard is the venue's —
/// a swap that never leaves its shard measures no hold worth collapsing.
pub fn hot_venue_clears_swaps_on<C: Cluster>(
    c: &mut C,
    venue_shard: ShardId,
    caller_shards: &[ShardId],
    budget: Budget,
) -> VenueReport {
    assert!(
        c.serves_shard(venue_shard)
            && caller_shards
                .iter()
                .all(|shard| c.serves_shard(*shard) && *shard != venue_shard),
        "the venue and its callers must sit on different shards, or the \
         hold under measurement is a local one",
    );
    let mut taken = Vec::new();
    let venue = stand_up_venue(c, venue_shard, &mut taken);
    let swappers = swappers_on(caller_shards, &mut taken);

    // Every swapper at once: the venue's cells are what they contend
    // for, so what the run measures is how fast that queue drains.
    let start = c.now();
    let mut submissions: Vec<TxHash> = Vec::with_capacity(swappers.len());
    for (key, account) in &swappers {
        let tx = build_swap_tx(
            key,
            *account,
            &venue.meta,
            *XRD,
            SWAP_INPUT,
            0,
            validity_around(c.now()),
        );
        submissions.push(tx.hash());
        c.submit(Arc::new(tx));
    }

    settle_swaps(c, &submissions, start, budget)
}

/// Drive until every swap is terminal, recording each one's first
/// observed settlement, then fold what the queue took.
fn settle_swaps<C: Cluster>(
    c: &mut C,
    submissions: &[TxHash],
    start: Duration,
    budget: Budget,
) -> VenueReport {
    let settled = RefCell::new(BTreeMap::<TxHash, Duration>::new());
    let all = c.run_until(budget, |c| {
        let mut settled = settled.borrow_mut();
        for hash in submissions {
            if !settled.contains_key(hash)
                && c.tx_status(*hash).is_some_and(|status| status.is_final())
            {
                settled.insert(*hash, c.now());
            }
        }
        settled.len() == submissions.len()
    });
    assert!(all, "the venue never cleared its queue within budget");

    for hash in submissions {
        let status = c.tx_status(*hash);
        assert!(
            matches!(
                status,
                Some(TransactionStatus::Completed(TransactionDecision::Accept))
            ),
            "a swap must accept for the run to measure anything; status = {status:?}",
        );
    }

    let settled = settled.into_inner();
    let mut latencies: Vec<Duration> = submissions
        .iter()
        .map(|hash| settled[hash].saturating_sub(start))
        .collect();
    latencies.sort_unstable();
    let last = settled.values().max().copied().unwrap_or(start);
    VenueReport {
        submitted: submissions.len(),
        elapsed: last.saturating_sub(start),
        latency_p50: latencies[latencies.len() / 2],
    }
}
