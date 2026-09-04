//! A divided shape meeting a reshape: the shard a leg or a core sits on
//! leaving the trie while the shape is in flight.
//!
//! A shard scheduled to leave divides like any other, to its terminal:
//! classification is the shape's legs against the committing block's
//! trie and nothing else, so a pending reshape changes what a member
//! runs not at all. A record cell follows its prefix to the successor,
//! and a claim or a delivery is a pull on whoever holds the prefix when
//! it is made. What these scenarios pin is that nothing in flight across
//! the cut is stranded by it — a transfer the leaving shard included settles
//! there, one it never included is delivered by its successor or refused
//! at the payer's deadline, and a call into a component on the leaving
//! shard keeps clearing from admission through the cut and from the
//! successor after it.

use hyperscale_engine::XRD;
use hyperscale_types::{
    BlockHeight, Ed25519PrivateKey, PrincipalAddr, ShardId, SubstateKey, TransactionDecision,
    TransactionStatus, TxHash, WeightedTimestamp, WorkInFlight, delivery_window_close,
};

use crate::reshape::split_lifecycle;
use crate::route::{FIRST_VENUE_SHARD, ROUTE_INPUT, SECOND_VENUE_SHARD, TRADER_SHARD};
use crate::straddler::{
    STRADDLER_PAYMENT, cast_splitter_vote, cast_threshold_vote, isolate_ec_intake,
    split_bytes_over, straddler_split_bytes, vote_splitter_down_to,
};
use crate::support::conservation::{Charges, World};
use crate::support::query::{
    anchored_genesis_height, held, held_at, merge_keeper_count, split_admitted,
};
use crate::support::tx::{
    MERGE_STRADDLER_LEFT, MERGE_STRADDLER_SURVIVOR, STRADDLER_SPLITTER, STRADDLER_SURVIVOR,
    build_route_tx, build_swap_tx, build_transfer_tx, fixture_flash_bytes,
    merge_survivor_ballast_accounts, merge_train_setup, quarter_ballast_over,
    split_ballast_accounts_over, split_train_setup, validity_around,
};
use crate::support::wait::{
    await_anchor_seeded, await_merge_keeper_count, await_serves, await_split_admitted,
    await_tx_terminal,
};
use crate::support::{Budget, Cluster, FaultableCluster, epochs};
use crate::venue::{
    PROVIDER_FUNDING, SWAP_INPUT, SWAPPER_FUNDING, StockedVenue, grind_onto, reserve_cell,
    stand_up_venue, swappers_on, venue_genesis_accounts_on,
};

/// The most transfers the train carries into the splitter, and what the
/// funding covers: enough to reach the admission at a few submissions
/// per epoch with several to spare past it, one payer each.
pub const SPLIT_TRAIN: usize = 48;

/// The most transfers the train carries into the merging shard: enough
/// to reach the pairing from the grow at a few submissions per epoch
/// with several to spare past the gate, one payer each.
pub const MERGE_TRAIN: usize = 32;

/// Submissions per epoch: slow enough that the funded train outlasts a
/// reshape's lead and the admission it leads to, with transfers on both
/// sides of the fold.
const TRAIN_PER_EPOCH: u64 = 2;

/// Transfers the train keeps sending once the reshape's gate has
/// drained: enough to reach the leaving shard's coast, where it includes
/// nothing.
const PAST_THE_GATE: usize = 6;

/// Where the leaving shard stood when a transfer into it was submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No reshape pending: the transfer divides and its delivery lands.
    Live,
    /// The reshape admitted and pending — a split admitted, a merge
    /// paired: the shard is departing, still including, and the transfer
    /// divides and settles as it would anywhere.
    Departing,
    /// The gate drained: the reshape no longer pends, the shard still
    /// includes for a while, then coasts on empty blocks to its terminal.
    /// A transfer it included settles; one it never included is accepted
    /// on the payer's chain and delivered by the shard's successor once
    /// the cut has landed the recipient's prefix there, or, where the
    /// delivery window closes first, reclaimed on the successor's proof
    /// that it never was.
    Draining,
}

/// The byte skew for [`a_departing_venue_clears_swaps_and_carries_on`]:
/// the survivor holds the fixture flash, so the splitter's ballast leads
/// that rather than the protocol's alone.
#[must_use]
pub fn departing_venue_ballast() -> Vec<(PrincipalAddr, u128)> {
    split_ballast_accounts_over(fixture_flash_bytes())
}

/// The reshape trigger [`a_departing_venue_clears_swaps_and_carries_on`]
/// arms at genesis: above each child of the ballasted root and below the
/// root itself, on the fixture flash's scale.
#[must_use]
pub fn departing_venue_split_bytes() -> u64 {
    fixture_flash_bytes() + 30_000
}

/// Genesis funding for the departing-route scenarios.
///
/// On the four-shard route topology: ballast putting the first venue's
/// quarter alone over the threshold the scenarios vote in, a provider on
/// each venue's shard, and the trader on its own, ground in the order
/// [`departing_route`] stands them up.
#[must_use]
pub fn departing_route_genesis_accounts() -> Vec<(PrincipalAddr, u128)> {
    let mut accounts = quarter_ballast_over(FIRST_VENUE_SHARD, fixture_flash_bytes());
    let mut taken = Vec::new();
    accounts.push((
        grind_onto(FIRST_VENUE_SHARD, &mut taken).1,
        PROVIDER_FUNDING,
    ));
    accounts.push((
        grind_onto(SECOND_VENUE_SHARD, &mut taken).1,
        PROVIDER_FUNDING,
    ));
    accounts.push((grind_onto(TRADER_SHARD, &mut taken).1, SWAPPER_FUNDING));
    accounts
}

/// Genesis funding for [`a_train_into_a_splitter_strands_nothing`].
#[must_use]
pub fn split_train_genesis_accounts() -> Vec<(PrincipalAddr, u128)> {
    split_train_setup(SPLIT_TRAIN).accounts
}

/// Genesis funding for [`a_train_into_a_merging_shard_strands_nothing`].
#[must_use]
pub fn merge_train_genesis_accounts() -> Vec<(PrincipalAddr, u128)> {
    merge_train_setup(MERGE_TRAIN).accounts
}

/// A venue whose shard is leaving keeps clearing swaps, and clears them
/// from its child once it has left.
///
/// The venue sits on the splitter and its callers on the survivor. With
/// the split admitted the venue's shard is departing, and every swap
/// reaching it divides and settles as it would anywhere: the callers pay
/// their input and one price, the venue claims the inputs. Then the cut:
/// the venue's cells land under a child, the reserve reads there at
/// exactly what the swaps left in it, and a swap after the cut divides
/// against the child and settles like any other.
///
/// # Panics
///
/// Panics if the venue misses its budget standing up, if the split is
/// not admitted or executed within budget, if any swap fails to accept,
/// if the reserve does not carry across the cut, or if either side of
/// the pair is not conserved.
pub fn a_departing_venue_clears_swaps_and_carries_on(c: &mut impl Cluster, budget: Budget) {
    let venue_shard = STRADDLER_SPLITTER;
    let caller_shard = STRADDLER_SURVIVOR;
    split_lifecycle(c);
    assert!(
        c.serves_shard(venue_shard) && c.serves_shard(caller_shard),
        "the grow must seat the venue's shard and its callers'",
    );
    let mut taken = Vec::new();
    let venue = stand_up_venue(c, venue_shard, &mut taken);
    let swappers = swappers_on(&[caller_shard], &mut taken);
    let reserve = reserve_cell(&venue.meta, *XRD);
    let stocked = held_at(c, reserve);
    assert!(
        stocked > 0,
        "the venue has to be holding something to price against"
    );

    let holders: Vec<_> = swappers
        .iter()
        .map(|(_, account)| account.address())
        .collect();
    let xrd = World::open(c, *XRD, holders.iter().copied(), [reserve]);
    let units = World::open(
        c,
        venue.unit,
        holders,
        [reserve_cell(&venue.meta, venue.unit)],
    );
    let mut charges = Charges::default();

    // The venue's shard is leaving from here to the cut.
    vote_splitter_down_to(c, split_bytes_over(fixture_flash_bytes()));

    let (leaving, later) = swappers.split_at(swappers.len() - 1);
    let mut submitted: Vec<TxHash> = Vec::new();
    for (key, caller) in leaving {
        let swap = build_swap_tx(
            key,
            *caller,
            &venue.meta,
            *XRD,
            SWAP_INPUT,
            0,
            validity_around(c.now()),
        );
        submitted.push(charges.submit(c, swap));
    }
    for hash in &submitted {
        let status = await_tx_terminal(c, *hash, budget);
        assert!(
            matches!(
                status,
                Some(TransactionStatus::Completed(TransactionDecision::Accept))
            ),
            "a swap reaching a departing venue must still settle; status = {status:?}",
        );
    }
    let claimed = stocked + SWAP_INPUT * u128::try_from(leaving.len()).expect("a few swaps");
    assert!(
        c.run_until(budget, |c| held_at(c, reserve) == claimed),
        "the departing venue must claim every input: reserve {} against {claimed}",
        held_at(c, reserve),
    );

    // The cut. The venue's prefix lands under one of the children, and
    // the reserve reads there at what the swaps left.
    let (left, right) = venue_shard.children();
    assert!(
        await_serves(c, left, epochs(28)) && await_serves(c, right, epochs(28)),
        "both splitter children must be served within budget",
    );
    assert!(
        await_anchor_seeded(c, left, epochs(6)),
        "the beacon must compose the split children's anchor",
    );
    assert!(
        anchored_genesis_height(c, left).is_some(),
        "the children's seeded genesis pins the venue's cells under a child",
    );
    assert_eq!(
        held_at(c, reserve),
        claimed,
        "the venue's reserve must carry across the cut untouched",
    );

    swap_against_the_child(c, &venue, &later[0], &mut charges, claimed, budget);

    xrd.assert_settles_within(c, &charges, budget, "swaps across the venue's split");
    units.assert_settles_within(
        c,
        &Charges::default(),
        budget,
        "swaps across the venue's split",
    );
}

/// The ballast [`a_leg_issued_on_a_departing_shard_reaches_its_venue`]
/// arms its trigger over.
///
/// The callers' shard carries the lead, so the flash-holding venue shard
/// stays under the threshold the scenario votes in.
#[must_use]
pub fn departing_caller_ballast() -> Vec<(PrincipalAddr, u128)> {
    split_ballast_accounts_over(fixture_flash_bytes())
}

/// A venue that stays, callers on the shard that is leaving, and the
/// worlds a swap between them can reach.
struct DepartingCallers {
    venue: StockedVenue,
    swappers: Vec<(Ed25519PrivateKey, PrincipalAddr)>,
    reserve: SubstateKey,
    stocked: u128,
    xrd: World,
    units: World,
}

/// Stand the venue up on the survivor, the callers on the splitter, and
/// vote the splitter down so the callers' shard alone is leaving.
///
/// # Panics
///
/// Panics if either shard is unserved, if the venue misses its budget
/// standing up or is holding nothing, or if the split is not admitted on
/// the callers' shard or is admitted on the venue's.
fn departing_callers<C: Cluster>(c: &mut C) -> DepartingCallers {
    let (venue_shard, caller_shard) = (STRADDLER_SURVIVOR, STRADDLER_SPLITTER);
    split_lifecycle(c);
    let set = stock_callers_against(c, venue_shard, caller_shard);
    // The callers' shard is leaving from here to the cut; the venue's
    // stays under the threshold with the flash on it.
    vote_splitter_down_to(c, split_bytes_over(fixture_flash_bytes()));
    assert!(
        await_split_admitted(c, caller_shard, epochs(20)),
        "the callers' shard must admit the split",
    );
    assert!(
        !split_admitted(c, venue_shard),
        "the venue's shard must not split",
    );
    set
}

/// Stand the venue up on a surviving quarter and the callers on the
/// merge-left child, then wait for the light pair to pair its keepers so
/// the callers' shard is the one leaving.
///
/// # Panics
///
/// Panics if a quarter is unserved, if the venue misses its budget
/// standing up or is holding nothing, or if the light pair does not pair
/// a keeper quorum within budget.
fn merging_callers<C: Cluster>(c: &mut C) -> DepartingCallers {
    let (venue_shard, caller_shard) = (MERGE_STRADDLER_SURVIVOR, MERGE_STRADDLER_LEFT);
    assert!(
        (0..4).all(|path| await_serves(c, ShardId::leaf(2, path), epochs(4))),
        "the grown four-shard topology must seat every quarter",
    );
    let set = stock_callers_against(c, venue_shard, caller_shard);
    // The light merging pair asserts its merge from the genesis byte
    // skew, so the callers' shard is leaving from the pairing to the cut.
    let parent = caller_shard.parent().expect("a depth-2 leaf has a parent");
    assert!(
        await_merge_keeper_count(c, parent, 3, epochs(24)),
        "the light merging pair must pair a keeper quorum within budget",
    );
    assert!(
        c.serves_shard(caller_shard),
        "the callers' shard must still be committing when the swaps go",
    );
    set
}

/// Stand a venue up on `venue_shard` with its callers on `caller_shard`,
/// and open the worlds a swap between them can reach.
///
/// # Panics
///
/// Panics if either shard is unserved, or if the venue misses its budget
/// standing up or is holding nothing to price against.
fn stock_callers_against<C: Cluster>(
    c: &mut C,
    venue_shard: ShardId,
    caller_shard: ShardId,
) -> DepartingCallers {
    assert!(
        c.serves_shard(venue_shard) && c.serves_shard(caller_shard),
        "the grow must seat the venue's shard and its callers'",
    );
    let mut taken = Vec::new();
    let venue = stand_up_venue(c, venue_shard, &mut taken);
    let swappers = swappers_on(&[caller_shard], &mut taken);
    let reserve = reserve_cell(&venue.meta, *XRD);
    let stocked = held_at(c, reserve);
    assert!(
        stocked > 0,
        "the venue has to be holding something to price against"
    );
    let holders: Vec<_> = swappers
        .iter()
        .map(|(_, account)| account.address())
        .collect();
    let xrd = World::open(c, *XRD, holders.iter().copied(), [reserve]);
    let units = World::open(
        c,
        venue.unit,
        holders,
        [reserve_cell(&venue.meta, venue.unit)],
    );
    DepartingCallers {
        venue,
        swappers,
        reserve,
        stocked,
        xrd,
        units,
    }
}

/// A caller whose own shard is leaving still reaches a venue that stays,
/// and its successor holds what the swap returned.
///
/// The mirror of [`a_departing_venue_clears_swaps_and_carries_on`]: there
/// the core's shard leaves, here the issuer's. With the reshape pending
/// the callers' shard is departing, and a swap issued there divides as it
/// would anywhere — the caller's leg pays, its crossing reaches the
/// venue, and the venue claims the input and returns the units. Then the
/// cut, which `cut` waits for: the callers' cells land on the successor,
/// the units read there at exactly what the swaps returned, and a swap
/// issued from the successor settles like any other.
///
/// # Panics
///
/// Panics if any swap fails to accept, if the venue does not claim every
/// input, if a caller's output does not carry across the cut, or if
/// either side of the pair is not conserved.
fn swaps_across_the_callers_cut<C: Cluster>(
    c: &mut C,
    set: DepartingCallers,
    cut: impl FnOnce(&mut C),
    budget: Budget,
) {
    let DepartingCallers {
        venue,
        swappers,
        reserve,
        stocked,
        xrd,
        units,
    } = set;
    let mut charges = Charges::default();

    let (leaving, later) = swappers.split_at(swappers.len() - 1);
    let mut submitted: Vec<TxHash> = Vec::new();
    for (key, caller) in leaving {
        let swap = build_swap_tx(
            key,
            *caller,
            &venue.meta,
            *XRD,
            SWAP_INPUT,
            0,
            validity_around(c.now()),
        );
        submitted.push(charges.submit(c, swap));
    }
    for hash in &submitted {
        let status = await_tx_terminal(c, *hash, budget);
        assert!(
            matches!(
                status,
                Some(TransactionStatus::Completed(TransactionDecision::Accept))
            ),
            "a swap issued on a departing shard must still settle; status = {status:?}",
        );
    }
    let claimed = stocked + SWAP_INPUT * u128::try_from(leaving.len()).expect("a few swaps");
    assert!(
        c.run_until(budget, |c| held_at(c, reserve) == claimed),
        "the staying venue must claim every input a departing caller sent: reserve {} against \
         {claimed}",
        held_at(c, reserve),
    );
    // What the issuing side is owed: the venue's return crosses back into
    // the shard that is leaving, and every caller banks it before the cut.
    let callers: Vec<PrincipalAddr> = leaving.iter().map(|(_, caller)| *caller).collect();
    assert!(
        c.run_until(budget, |c| callers.iter().all(|caller| held(
            c,
            caller.address(),
            venue.unit
        ) > 0)),
        "every caller on the departing shard must bank its output before the cut; holdings = {:?}",
        callers
            .iter()
            .map(|caller| held(c, caller.address(), venue.unit))
            .collect::<Vec<_>>(),
    );
    let banked: Vec<u128> = callers
        .iter()
        .map(|caller| held(c, caller.address(), venue.unit))
        .collect();

    cut(c);
    // The successor answers for the prefix once it has adopted it, which
    // is a moment after it serves — so the read is awaited and the
    // assertion is on the figure, not on when it arrives.
    let carried = |c: &C| {
        callers
            .iter()
            .map(|caller| held(c, caller.address(), venue.unit))
            .collect::<Vec<_>>()
    };
    assert!(
        c.run_until(budget, |c| carried(c) == banked),
        "a departing caller's output must carry across its own shard's cut untouched: {:?} \
         against {banked:?}",
        carried(c),
    );

    swap_against_the_child(c, &venue, &later[0], &mut charges, claimed, budget);

    xrd.assert_settles_within(c, &charges, budget, "swaps across the callers' cut");
    units.assert_settles_within(
        c,
        &Charges::default(),
        budget,
        "swaps across the callers' cut",
    );
}

/// A swap issued on a splitting shard reaches its venue, and the child
/// that takes the caller's prefix holds what came back.
///
/// # Panics
///
/// Panics as [`departing_callers`] and [`swaps_across_the_callers_cut`]
/// do.
pub fn a_leg_issued_on_a_departing_shard_reaches_its_venue(c: &mut impl Cluster, budget: Budget) {
    let set = departing_callers(c);
    swaps_across_the_callers_cut(c, set, |c| await_cut(c, STRADDLER_SPLITTER), budget);
}

/// A swap issued on a merging shard reaches its venue, and the parent the
/// pair collapses into holds what came back.
///
/// [`a_leg_issued_on_a_departing_shard_reaches_its_venue`] across the
/// other reshape: the callers sit on the merge-left child, which the
/// grown topology's byte skew pairs with its sibling from the grow alone,
/// and the venue on a surviving quarter. Requires the
/// [`merging_caller_genesis_accounts`] funding on a config grown to four
/// shards.
///
/// # Panics
///
/// Panics as [`merging_callers`] and [`swaps_across_the_callers_cut`] do,
/// and if the merged parent is not served within budget.
pub fn a_leg_issued_on_a_merging_shard_reaches_its_venue(c: &mut impl Cluster, budget: Budget) {
    let parent = MERGE_STRADDLER_LEFT
        .parent()
        .expect("a depth-2 leaf has a parent");
    let set = merging_callers(c);
    swaps_across_the_callers_cut(
        c,
        set,
        |c| {
            assert!(
                await_serves(c, parent, epochs(28)),
                "the merged parent must be served within budget",
            );
        },
        budget,
    );
}

/// Genesis funding for [`a_leg_issued_on_a_merging_shard_reaches_its_venue`].
///
/// The merge topology's byte skew, the venue's provider on a surviving
/// quarter, and the callers on the merge-left child — so the pair that
/// merges is the callers', and the venue's shard never reshapes.
#[must_use]
pub fn merging_caller_genesis_accounts() -> Vec<(PrincipalAddr, u128)> {
    let mut accounts = merge_survivor_ballast_accounts();
    accounts.extend(venue_genesis_accounts_on(
        MERGE_STRADDLER_SURVIVOR,
        &[MERGE_STRADDLER_LEFT],
    ));
    accounts
}

/// Wait for `splitter` to cut: both children served and the beacon
/// carrying their seeded genesis, so the prefixes it held read on the
/// children.
///
/// # Panics
///
/// Panics if a child is unserved within budget or the anchor is never
/// composed.
fn await_cut<C: Cluster>(c: &mut C, splitter: ShardId) {
    let (left, right) = splitter.children();
    assert!(
        await_serves(c, left, epochs(28)) && await_serves(c, right, epochs(28)),
        "both splitter children must be served within budget",
    );
    assert!(
        await_anchor_seeded(c, left, epochs(6)),
        "the beacon must compose the split children's anchor",
    );
    assert!(
        anchored_genesis_height(c, left).is_some(),
        "the children's seeded genesis pins the split shard's cells under a child",
    );
}

/// Two stocked venues, one on a shard about to leave, and the trader on
/// a shard of its own, opened over the worlds a route can reach.
struct DepartingRoute {
    leaving: StockedVenue,
    staying: StockedVenue,
    key: Ed25519PrivateKey,
    trader: PrincipalAddr,
    reserves: [SubstateKey; 2],
    stocked: [u128; 2],
    xrd: World,
    units: World,
}

/// Stand the departing route up on the four-shard route topology: the
/// first venue's shard is voted over the threshold and admitted to
/// split, and both venues are stocked before that. The trader sits on a
/// shard of its own: a leg beside a core node on one shard runs as that
/// core member and waits with it, where the leg under test pays and
/// settles alone.
///
/// # Panics
///
/// Panics if a quarter is unserved, if either venue misses its budget
/// standing up, or if the split is not admitted, or admitted elsewhere.
fn departing_route<C: Cluster>(c: &mut C) -> DepartingRoute {
    let departing = FIRST_VENUE_SHARD;
    assert!(
        (0..4).all(|path| c.serves_shard(ShardId::leaf(2, path))),
        "the grown four-shard topology must seat every quarter",
    );
    let mut taken = Vec::new();
    let leaving = stand_up_venue(c, departing, &mut taken);
    let staying = stand_up_venue(c, SECOND_VENUE_SHARD, &mut taken);
    let (key, trader) = grind_onto(TRADER_SHARD, &mut taken);
    let reserves = [
        reserve_cell(&leaving.meta, *XRD),
        reserve_cell(&staying.meta, *XRD),
    ];
    let stocked = reserves.map(|reserve| held_at(c, reserve));
    assert!(
        stocked.iter().all(|&held| held > 0),
        "both venues have to be holding something to price against"
    );
    let xrd = World::open(c, *XRD, [trader.address()], reserves);
    let units = World::open(
        c,
        leaving.unit,
        [trader.address()],
        [
            reserve_cell(&leaving.meta, leaving.unit),
            reserve_cell(&staying.meta, staying.unit),
        ],
    );
    // The departing venue's shard is leaving from here to the cut: its
    // ballast alone crosses the voted threshold. The vote settles before
    // any certificate channel is cut, since it crosses the same shards.
    cast_threshold_vote(c, split_bytes_over(fixture_flash_bytes()));
    assert!(
        await_split_admitted(c, departing, epochs(20)),
        "only the over-threshold venue shard must admit a split",
    );
    assert!(
        (0..4)
            .map(|path| ShardId::leaf(2, path))
            .filter(|&shard| shard != departing)
            .all(|shard| !split_admitted(c, shard)),
        "no other quarter may split",
    );
    DepartingRoute {
        leaving,
        staying,
        key,
        trader,
        reserves,
        stocked,
        xrd,
        units,
    }
}

/// Submit one route through the departing venue and hold it until both
/// shards have committed it while both are live, the survivor's drain
/// has engaged, and the trader's leg has paid. Returns the route's hash,
/// the survivor's drain before the route, and what the trader holds
/// once its leg has paid.
///
/// # Panics
///
/// Panics if both shards do not commit the route while both are live,
/// if it engages no hold on the survivor, or if the trader's leg does
/// not pay.
fn submit_departing_route<C: Cluster>(
    c: &mut C,
    route: &DepartingRoute,
    charges: &mut Charges,
) -> (TxHash, WorkInFlight, u128, WeightedTimestamp) {
    let (departing, survivor) = (FIRST_VENUE_SHARD, SECOND_VENUE_SHARD);
    let baseline = c
        .committed_work_in_flight(survivor)
        .expect("the survivor must serve a committed tip before the route");
    let validity = validity_around(c.now());
    let tx = build_route_tx(
        &route.key,
        route.trader,
        (&route.leaving.meta, &route.staying.meta),
        *XRD,
        ROUTE_INPUT,
        0,
        validity,
    );
    let hash = charges.submit(c, tx);
    assert!(
        c.run_until(epochs(12), |c| c.chain_fate(survivor, hash).0.is_some()
            && c.chain_fate(departing, hash).0.is_some()),
        "both shards must commit the route while both are live",
    );
    let engaged = c
        .committed_work_in_flight(survivor)
        .expect("the survivor must serve a committed tip once it holds the route");
    assert!(
        engaged > baseline,
        "the route must engage a hold against the survivor's drain, or its release below \
         proves nothing; baseline = {baseline:?}, engaged = {engaged:?}",
    );
    assert!(
        c.run_until(epochs(8), |c| held(c, route.trader.address(), *XRD)
            < SWAPPER_FUNDING - ROUTE_INPUT),
        "the trader's leg must pay before the core is asked anything",
    );
    (
        hash,
        baseline,
        held(c, route.trader.address(), *XRD),
        validity.end_timestamp_exclusive,
    )
}

/// Wait for the departing venue's shard to terminate: both children
/// served.
///
/// # Panics
///
/// Panics if a child is not served within budget.
fn await_departed<C: Cluster>(c: &mut C) {
    let (left, right) = FIRST_VENUE_SHARD.children();
    assert!(
        await_serves(c, left, epochs(28)) && await_serves(c, right, epochs(28)),
        "both splitter children must be served within budget",
    );
}

/// A route through a departing venue releases the surviving venue's
/// hold at the terminal and gives the trader its input back.
///
/// A route's two venues are one core, each awaiting the other's
/// certificate, so a route is the shape that still holds work across a
/// shard's terminal: the trader's leg pays and settles alone, and the
/// two venue ticks wait. With the certificate channel cut both ways
/// neither venue ever holds the other's, the departing venue abandons
/// at its deadline and terminates having settled nothing, and the
/// survivor's tick — fenced on the departing shard's settled set from
/// the admission — stays engaged against the survivor's drain until that
/// set arrives. Then the departure record speaks: the survivor abandons
/// the core member, the drain returns to its baseline, and the trader's
/// crossing is reclaimed. Neither reserve moves.
///
/// # Panics
///
/// Panics as [`departing_route`] and [`submit_departing_route`] do, and
/// if the hold does not return to its baseline or the input to the
/// trader after the terminal, if the route is not abandoned, if a
/// reserve moves, or if either side of the pair is not conserved.
pub fn a_route_into_a_departing_venue_releases_the_survivors_hold<C: FaultableCluster>(c: &mut C) {
    let (departing, survivor) = (FIRST_VENUE_SHARD, SECOND_VENUE_SHARD);
    let route = departing_route(c);
    // Neither venue may hold the other's certificate. Provisions and
    // headers still flow, so each venue commits the route and runs its
    // own core leg, which is the state under test.
    let _ = isolate_ec_intake(c, departing, survivor);
    let _ = isolate_ec_intake(c, survivor, departing);
    let mut charges = Charges::default();
    let (hash, baseline, paid, ..) = submit_departing_route(c, &route, &mut charges);

    // The cut. The departing venue's cells land under a child, and its
    // settled set reaches the survivor.
    await_departed(c);
    assert!(
        c.run_until(epochs(12), |c| c.committed_work_in_flight(survivor)
            == Some(baseline)),
        "the survivor's hold must return to its baseline once the departed venue's settled \
         set has answered; holds {:?} against {baseline:?}",
        c.committed_work_in_flight(survivor),
    );
    // The departure record licenses the trader's reclaim, which reads the
    // record cell the trader's leg wrote — and reads it whenever the
    // record lands, because a record is value and value is not swept on
    // a clock. So the input comes back however long the departure took,
    // which is what makes this readable on either epoch clock: one whose
    // epochs outrun the escrow grace reaches the reclaim just the same.
    assert!(
        c.run_until(epochs(12), |c| held(c, route.trader.address(), *XRD)
            == paid + ROUTE_INPUT),
        "the trader must get the route's input back on the departure record; holds {} \
         against {}",
        held(c, route.trader.address(), *XRD),
        paid + ROUTE_INPUT,
    );
    // The trader's own leg accepted and stays accepted; the core it
    // fed is abandoned on both shards, never settled one-sided.
    for shard in [departing, survivor] {
        let fate = c.chain_fate(shard, hash).1.map(|(_, decision)| decision);
        assert!(
            fate != Some(TransactionDecision::Accept),
            "a venue settled a route its counterpart never certified; {shard} reached {fate:?}",
        );
    }
    for (reserve, before) in route.reserves.into_iter().zip(route.stocked) {
        assert_eq!(
            held_at(c, reserve),
            before,
            "neither venue may have applied its side of a route the other never certified",
        );
    }
    c.clear_drops();
    // And conserved outright: nothing of the route is stranded, because
    // nothing it issued was swept out from under its reclaim.
    assert!(
        c.run_until(epochs(8), |c| route.xrd.settles(c, charges.burned(c))),
        "a route through a departing venue: the world must settle against the burn alone",
    );
    route
        .xrd
        .assert_settled(c, charges.burned(c), "a route through a departing venue");
    charges.assert_each_fits_a_full_block(c);
    route.units.assert_settles_within(
        c,
        &Charges::default(),
        epochs(8),
        "a route through a departing venue",
    );
}

/// A route the departing venue settled is settled by the surviving venue
/// too, from the certificate the departed chain left behind.
///
/// The mirror of
/// [`a_route_into_a_departing_venue_releases_the_survivors_hold`]: the
/// cut takes only the survivor's intake, so the departing venue holds
/// both certificates and settles, applying its half, while the survivor
/// holds only its own and cannot apply until the departed venue's
/// reaches it. What that leaves the survivor is a transaction its
/// counterpart's settled set names as settled — so no record covers it,
/// the fence refuses any abandonment of it, and the only resolution left
/// is the certificate itself, committed on the departed shard's tail
/// chain. The cut lifts once the children seat, so what is measured is
/// a retention limit and not a partition: the survivor asks for the
/// certificate on a whole network, having missed it while the cut was
/// up, and the route banks its output.
///
/// # Panics
///
/// Panics as [`departing_route`] and [`submit_departing_route`] do, and
/// if the departing venue does not settle the route before it leaves,
/// if the survivor never applies what the departed venue settled, if
/// its hold does not return to its baseline, or if either side of the
/// pair is not conserved.
pub fn a_route_the_departing_venue_settled_is_settled_by_the_survivor<C: FaultableCluster>(
    c: &mut C,
) {
    let (departing, survivor) = (FIRST_VENUE_SHARD, SECOND_VENUE_SHARD);
    let route = departing_route(c);
    let _ = isolate_ec_intake(c, survivor, departing);
    let mut charges = Charges::default();
    let (hash, baseline, paid, validity_end) = submit_departing_route(c, &route, &mut charges);
    assert!(
        c.run_until(epochs(12), |c| matches!(
            c.chain_fate(departing, hash).1,
            Some((_, TransactionDecision::Accept))
        )),
        "the departing venue holds both certificates and settles the route before it leaves",
    );
    assert!(
        c.chain_fate(survivor, hash).1.is_none(),
        "the survivor holds only its own certificate and cannot apply yet",
    );

    await_departed(c);
    c.clear_drops();
    assert!(
        c.run_until(epochs(12), |c| matches!(
            c.chain_fate(survivor, hash).1,
            Some((_, TransactionDecision::Accept))
        )),
        "the survivor must recover the departed venue's certificate from its tail chain and \
         apply what it settled",
    );
    assert!(
        c.run_until(epochs(12), |c| c.committed_work_in_flight(survivor)
            == Some(baseline)),
        "the survivor's hold must return to its baseline once it has settled; holds {:?} \
         against {baseline:?}",
        c.committed_work_in_flight(survivor),
    );
    // The output is a delivery to the trader, admissible to the delivery
    // window's close, and the crossing it rides is a record cell swept at
    // the intent's end plus the escrow grace. On a clock the window
    // outlasts, the trader banks it. On a clock whose epochs outrun the
    // window the survivor settles after it: the output's delivery can no
    // longer be admitted and, the survivor having settled past the grace
    // as well, the record it issued is swept under the lapse probe that
    // would have returned it — the output is stranded, and the world is
    // short by exactly it, at most the input the route put in.
    let banked = c.run_until(epochs(8), |c| held(c, route.trader.address(), *XRD) > paid);
    let clock = WeightedTimestamp::ZERO.plus(c.now());
    let burned = charges.burned(c);
    if banked || clock < delivery_window_close(validity_end) {
        assert!(
            banked,
            "the route must bank its output for the trader while its delivery window is open; \
             holds {} against {paid}",
            held(c, route.trader.address(), *XRD),
        );
        route.xrd.assert_settles_within(
            c,
            &charges,
            epochs(8),
            "a route settled across a departure",
        );
    } else {
        assert_eq!(
            held(c, route.trader.address(), *XRD),
            paid,
            "past its delivery window the output never reaches the trader",
        );
        let stranded = route.xrd.before() - route.xrd.held(c) - burned;
        assert!(
            stranded > 0 && stranded <= ROUTE_INPUT,
            "the world must be short by the stranded output alone, at most the input; short by \
             {stranded}",
        );
        charges.assert_each_fits_a_full_block(c);
    }
    route.units.assert_settles_within(
        c,
        &Charges::default(),
        epochs(8),
        "a route settled across a departure",
    );
}

/// A train of transfers into a shard across its split's admission strands
/// nothing: each reaches the fate its phase owes it, and the train's
/// accounts are conserved throughout.
///
/// The payers sit on the survivor and the recipients on the splitter.
/// A transfer every few blocks from before the vote until the splitter
/// has coasted, so the train holds every [`Phase`] — transfers the payer
/// committed before the admission fold, ones committed while the split
/// pended, and ones the coasting splitter never included — and, around the fold, the pair whose payer committed on
/// one side of it and whose delivery landed on the other. A transfer the
/// splitter included settles and credits its recipient once; one it
/// never included credits its recipient exactly when the payer accepted
/// it, by the child's delivery, and never otherwise.
///
/// # Panics
///
/// Panics if the train misses any phase or never reaches the coast, if a
/// transfer's credit disagrees with its verdict, or if the train's
/// accounts are not conserved.
pub fn a_train_into_a_splitter_strands_nothing<C: Cluster>(c: &mut C) {
    let splitter = STRADDLER_SPLITTER;
    let setup = split_train_setup(SPLIT_TRAIN);
    split_lifecycle(c);
    let world = train_world(c, &setup.legs);
    let mut charges = Charges::default();

    let sent = drive_train(
        c,
        splitter,
        &setup.legs,
        &mut charges,
        |c| split_admitted(c, splitter),
        |c| cast_splitter_vote(c, straddler_split_bytes()),
    );

    let children = <[ShardId; 2]>::from(splitter.children());
    assert_train_fates(c, splitter, &children, &sent);
    world.assert_settles_within(c, &charges, epochs(8), "a train across a split's admission");
}

/// A train of transfers into a shard across its merge's pairing strands
/// nothing: each reaches the fate its phase owes it, and the train's
/// accounts are conserved throughout.
///
/// [`a_train_into_a_splitter_strands_nothing`] across the other reshape.
/// The payers sit on the surviving quarter and the recipients on the
/// merge-left child, which the grown topology's byte skew pairs with its
/// sibling from the grow alone. A transfer every few blocks from before
/// the pairing until the merging child has coasted, so the train holds
/// every [`Phase`]: transfers committed before the pairing fold, ones
/// committed while the merge pended, and ones the coasting child never
/// included, which the merged parent delivers once the cut has landed
/// the recipient's prefix there. Requires the [`merge_train_setup`]
/// funding on a config grown to four shards.
///
/// # Panics
///
/// Panics if the grown topology does not seat every quarter, if the
/// train misses any phase or never reaches the coast, if a transfer's
/// credit disagrees with its verdict, or if the train's accounts are not
/// conserved.
pub fn a_train_into_a_merging_shard_strands_nothing<C: Cluster>(c: &mut C) {
    let merging = MERGE_STRADDLER_LEFT;
    let parent = merging.parent().expect("a depth-2 leaf has a parent");
    let setup = merge_train_setup(MERGE_TRAIN);
    assert!(
        (0..4).all(|path| await_serves(c, ShardId::leaf(2, path), epochs(4))),
        "the grown four-shard topology must seat every quarter",
    );
    let world = train_world(c, &setup.legs);
    let mut charges = Charges::default();

    let sent = drive_train(
        c,
        merging,
        &setup.legs,
        &mut charges,
        |c| merge_keeper_count(c, parent).is_some(),
        |_| {},
    );

    assert_train_fates(c, merging, &[parent], &sent);
    world.assert_settles_within(c, &charges, epochs(8), "a train across a merge's pairing");
}

/// Everything a train can reach: each leg's payer and recipient. The
/// ballast holds the byte skew and never spends.
fn train_world<C: Cluster>(
    c: &C,
    legs: &[(Ed25519PrivateKey, PrincipalAddr, PrincipalAddr)],
) -> World {
    World::open(
        c,
        *XRD,
        legs.iter()
            .flat_map(|(_, from, to)| [from.address(), to.address()]),
        [],
    )
}

/// Send `legs` into `terminating` every few blocks until its reshape has
/// drained and several more have gone after it, recording the phase each
/// went in — and sooner than that wherever a phase nothing has covered
/// opens, so the coverage the train exists for does not turn on the
/// reshape's timing. `pending` reads whether the reshape is admitted and
/// pending; `arm` runs once the cadence is measured, before the second
/// leg — where a scenario casts the vote that starts its reshape.
///
/// # Panics
///
/// Panics if the train ends without a transfer sent in every [`Phase`].
fn drive_train<C: Cluster>(
    c: &mut C,
    terminating: ShardId,
    legs: &[(Ed25519PrivateKey, PrincipalAddr, PrincipalAddr)],
    charges: &mut Charges,
    pending: impl Fn(&C) -> bool,
    arm: impl FnOnce(&mut C),
) -> Vec<(TxHash, PrincipalAddr, Phase)> {
    // Block cadence is activity-driven, so the spacing is measured off
    // the first leg rather than assumed.
    let mut legs = legs.iter();
    let mut sent: Vec<(TxHash, PrincipalAddr, Phase)> = Vec::new();
    let mut admitted_once = false;
    let height = |c: &C| {
        c.committed_height(terminating)
            .map_or(0, BlockHeight::inner)
    };
    let before = height(c);
    send_leg(
        c,
        legs.next().expect("a funded leg"),
        charges,
        &mut sent,
        &mut admitted_once,
        &pending,
    );
    c.run_until(epochs(1), |_| false);
    let spacing = (height(c).saturating_sub(before) / TRAIN_PER_EPOCH).max(1);

    arm(c);

    let mut draining = 0;
    for leg in legs {
        if send_leg(c, leg, charges, &mut sent, &mut admitted_once, &pending) == Phase::Draining {
            draining += 1;
            if draining >= PAST_THE_GATE {
                break;
            }
        }
        // The spacing is what the train rides, and phase coverage is
        // what it is for: a phase narrower than the spacing would get no
        // leg at all, and the assertion below would fail on the reshape's
        // timing rather than on anything the train measures. So the wait
        // ends early the moment a phase nothing has been sent in opens.
        let from_height = height(c);
        let covered: Vec<Phase> = sent.iter().map(|(_, _, phase)| *phase).collect();
        let seen_admitted = admitted_once;
        c.run_until(epochs(2), |c| {
            height(c) >= from_height + spacing
                || !covered.contains(&phase_of(pending(c), seen_admitted))
        });
    }
    for phase in [Phase::Live, Phase::Departing, Phase::Draining] {
        assert!(
            sent.iter().any(|(_, _, sent_in)| *sent_in == phase),
            "the train has to hold a transfer sent {phase:?}, or that phase's fate goes unread",
        );
    }
    sent
}

/// Every transfer's fate, once `terminating` has reached its terminal and
/// nothing more can be included: whether it took the transfer is what
/// decides between settling and refusal, and a recipient is credited
/// exactly when its payer's transfer was accepted.
///
/// # Panics
///
/// Panics if a successor is not served within budget, if a transfer sent
/// before the drain was never included, if any transfer reaches a fate
/// its phase does not allow, if a credit disagrees with a verdict, or if
/// the train never reached the coast.
fn assert_train_fates<C: Cluster>(
    c: &mut C,
    terminating: ShardId,
    successors: &[ShardId],
    sent: &[(TxHash, PrincipalAddr, Phase)],
) {
    for &successor in successors {
        assert!(
            await_serves(c, successor, epochs(28)),
            "successor {successor} must be served within budget",
        );
    }
    let mut never_included = 0;
    for (hash, to, phase) in sent {
        let included = c.chain_fate(terminating, *hash).0.is_some();
        assert!(
            included || *phase == Phase::Draining,
            "a transfer sent {phase:?} must be included by the leaving shard",
        );
        never_included += usize::from(!included);
        let taken = if included {
            "included"
        } else {
            "never included"
        };
        let status = await_tx_terminal(c, *hash, epochs(12));
        // The credit is the recipient's chain's to give: the leaving
        // shard's or, for a transfer it never included, whichever
        // successor took the recipient's prefix. A transfer accepted by
        // its payer and by no chain holding the recipient is one whose
        // delivery lapsed, and the reclaim returns the payment — the
        // world's conservation is what reads that.
        let delivered = std::iter::once(terminating)
            .chain(successors.iter().copied())
            .any(|shard| {
                c.chain_fate(shard, *hash)
                    .1
                    .is_some_and(|(_, decision)| decision == TransactionDecision::Accept)
            });
        let credited = match (fate_owed(*phase, included), status, delivered) {
            (
                Fate::Settled | Fate::CarriedOrReclaimed,
                Some(TransactionStatus::Completed(TransactionDecision::Accept)),
                true,
            ) => 10 + STRADDLER_PAYMENT,
            (
                Fate::CarriedOrReclaimed,
                Some(TransactionStatus::Completed(
                    TransactionDecision::Accept | TransactionDecision::Aborted,
                )),
                false,
            ) => 10,
            (owed, other, delivered) => panic!(
                "a transfer sent {phase:?} and {taken} by the leaving shard owes {owed:?} and \
                 reached {other:?}, delivered = {delivered}",
            ),
        };
        assert!(
            c.run_until(epochs(8), |c| held(c, to.address(), *XRD) == credited),
            "a recipient of a transfer sent {phase:?} and {taken} by the leaving shard must \
             hold {credited}; holds {}",
            held(c, to.address(), *XRD),
        );
    }
    assert!(
        never_included > 0,
        "the train has to reach the leaving shard's coast, or nothing here crosses the cut",
    );
}

/// What a transfer's phase leaves open once the leaving shard has
/// terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fate {
    /// Settled on a chain that credited the recipient. The leaving shard
    /// included it and settled it, whatever it was doing at the time.
    Settled,
    /// Carried by the successor that took the recipient's prefix, or, if
    /// the delivery window closed before the successor could serve,
    /// reclaimed on the successor's proof that it never delivered.
    CarriedOrReclaimed,
}

/// The fate a phase owes a transfer the leaving shard did or did not
/// include.
///
/// One shape takes two fates, and it is the one the reshape opens: a
/// transfer the coasting shard never included races the delivery
/// window's close. Everything else settles, so a run that crossed no cut
/// satisfies no disjunction here.
const fn fate_owed(phase: Phase, included: bool) -> Fate {
    match (phase, included) {
        (Phase::Live | Phase::Departing, _) | (Phase::Draining, true) => Fate::Settled,
        (Phase::Draining, false) => Fate::CarriedOrReclaimed,
    }
}

/// Submit one of the train's legs, recording the leaving shard's phase
/// when it went. The reshape shows as pending from its admission until
/// the gate drains, so a shard once admitted and no longer pending is
/// draining.
/// Which phase a shard is in, from whether its reshape pends now and
/// whether one ever has.
///
/// Read without sending anything, so the train can wait on a phase
/// opening as well as record the one a leg went in.
const fn phase_of(pending: bool, admitted_once: bool) -> Phase {
    match (pending, admitted_once) {
        (true, _) => Phase::Departing,
        (false, false) => Phase::Live,
        (false, true) => Phase::Draining,
    }
}

fn send_leg<C: Cluster>(
    c: &mut C,
    (key, from, to): &(Ed25519PrivateKey, PrincipalAddr, PrincipalAddr),
    charges: &mut Charges,
    sent: &mut Vec<(TxHash, PrincipalAddr, Phase)>,
    admitted_once: &mut bool,
    pending: impl Fn(&C) -> bool,
) -> Phase {
    let pending = pending(c);
    *admitted_once |= pending;
    let phase = phase_of(pending, *admitted_once);
    let tx = build_transfer_tx(key, *from, *to, STRADDLER_PAYMENT, validity_around(c.now()));
    sent.push((charges.submit(c, tx), *to, phase));
    phase
}

/// A swap after the cut divides against the venue's child and settles:
/// the caller banks its output and the child claims the input.
fn swap_against_the_child<C: Cluster>(
    c: &mut C,
    venue: &StockedVenue,
    (key, caller): &(Ed25519PrivateKey, PrincipalAddr),
    charges: &mut Charges,
    claimed: u128,
    budget: Budget,
) {
    let swap = build_swap_tx(
        key,
        *caller,
        &venue.meta,
        *XRD,
        SWAP_INPUT,
        0,
        validity_around(c.now()),
    );
    let hash = charges.submit(c, swap);
    let status = await_tx_terminal(c, hash, budget);
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "a swap against the venue's child must settle; status = {status:?}",
    );
    assert!(
        c.run_until(budget, |c| held(c, caller.address(), venue.unit) > 0),
        "the post-cut swap's output never reached its caller",
    );
    assert_eq!(
        held_at(c, reserve_cell(&venue.meta, *XRD)),
        claimed + SWAP_INPUT,
        "the venue's child claims the post-cut input",
    );
}
