//! A divided shape meeting a reshape: the shard a leg or a core sits on
//! leaving the trie while the shape is in flight.
//!
//! A shard scheduled to leave refuses to decompose anything reaching it —
//! a record written for a shard with no round left would have nowhere to
//! be claimed — so from a split's admission to its cut every shape
//! touching the splitter runs whole, and after the cut its cells sit
//! under a child and shapes divide again. What these scenarios pin is
//! that both halves of that rule settle and that the seam between them
//! strands nothing: the classification is taken once per shard at its
//! own commit, and a transfer whose payer committed just before the
//! admission fold and whose delivery lands just after is the shape that
//! would read one answer on one side and the other on the other.

use hyperscale_engine::XRD;
use hyperscale_types::{
    BlockHeight, Ed25519PrivateKey, PrincipalAddr, TransactionDecision, TransactionStatus, TxHash,
};

use crate::reshape::split_lifecycle;
use crate::straddler::{
    STRADDLER_PAYMENT, cast_splitter_vote, split_bytes_over, straddler_split_bytes,
    vote_splitter_down_to,
};
use crate::support::conservation::{Charges, World};
use crate::support::query::{anchored_genesis_height, held, held_at, split_admitted};
use crate::support::tx::{
    STRADDLER_SPLITTER, STRADDLER_SURVIVOR, build_swap_tx, build_transfer_tx, fixture_flash_bytes,
    split_ballast_accounts_over, split_train_setup, validity_around,
};
use crate::support::wait::{await_anchor_seeded, await_serves, await_tx_terminal};
use crate::support::{Budget, Cluster, epochs};
use crate::venue::{SWAP_INPUT, StockedVenue, reserve_cell, stand_up_venue, swappers_on};

/// The most transfers the train carries into the splitter, and what the
/// funding covers: enough to reach the admission at a few submissions
/// per epoch with several to spare past it, one payer each.
pub const SPLIT_TRAIN: usize = 48;

/// Submissions per epoch: slow enough that the funded train outlasts a
/// vote's lead and the admission it leads to, with transfers on both
/// sides of the fold.
const TRAIN_PER_EPOCH: u64 = 2;

/// Transfers the train keeps sending once the splitter's gate has
/// drained: enough to reach its coast, where it includes nothing.
const PAST_THE_GATE: usize = 6;

/// Where the splitter stood when a transfer into it was submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No split admitted: the transfer divides and its delivery lands.
    Live,
    /// The split admitted and pending: the splitter is departing, so the
    /// transfer runs whole on both shards and still settles.
    Departing,
    /// The gate drained: the split no longer pends, the splitter still
    /// includes for a while, then coasts on empty blocks to its terminal.
    /// A transfer it included settles; one it never included is either
    /// refused at the payer's deadline, crediting nobody, or — where the
    /// payer read the splitter as no longer departing and divided it —
    /// accepted on the payer's chain and delivered by the splitter's
    /// child once the cut has landed the recipient's prefix there.
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

/// Genesis funding for [`a_train_into_a_splitter_strands_nothing`].
#[must_use]
pub fn split_train_genesis_accounts() -> Vec<(PrincipalAddr, u128)> {
    split_train_setup(SPLIT_TRAIN).accounts
}

/// A venue whose shard is leaving keeps clearing swaps, and clears them
/// from its child once it has left.
///
/// The venue sits on the splitter and its callers on the survivor. With
/// the split admitted the venue's shard is departing, so every swap
/// reaching it runs whole and still settles: the callers pay their input
/// and one price, the venue claims the inputs. Then the cut: the venue's
/// cells land under a child, the reserve reads there at exactly what the
/// swaps left in it, and a swap after the cut divides against the child
/// and settles like any other.
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

/// A train of transfers into a shard across its split's admission strands
/// nothing: each reaches the fate its phase owes it, and the train's
/// accounts are conserved throughout.
///
/// The payers sit on the survivor and the recipients on the splitter.
/// A transfer every few blocks from before the vote until the splitter
/// has coasted, so the train holds every [`Phase`] — transfers the payer
/// committed divided before the admission fold, ones every shard ran
/// whole while the split pended, and ones the coasting splitter never
/// included — and, around the fold, the pair whose payer committed on
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
    let world = World::open(
        c,
        *XRD,
        setup
            .legs
            .iter()
            .flat_map(|(_, from, to)| [from.address(), to.address()]),
        [],
    );
    let mut charges = Charges::default();

    // Block cadence is activity-driven, so the spacing is measured off
    // the first leg rather than assumed.
    let mut legs = setup.legs.iter();
    let mut sent: Vec<(TxHash, PrincipalAddr, Phase)> = Vec::new();
    let mut admitted_once = false;
    let height = |c: &C| c.committed_height(splitter).map_or(0, BlockHeight::inner);
    let before = height(c);
    send_leg(
        c,
        legs.next().expect("a funded leg"),
        &mut charges,
        &mut sent,
        &mut admitted_once,
    );
    c.run_until(epochs(1), |_| false);
    let spacing = (height(c).saturating_sub(before) / TRAIN_PER_EPOCH).max(1);

    cast_splitter_vote(c, straddler_split_bytes());

    // A transfer every `spacing` blocks until the splitter's gate has
    // drained and several more have gone after it.
    let mut draining = 0;
    for leg in legs {
        if send_leg(c, leg, &mut charges, &mut sent, &mut admitted_once) == Phase::Draining {
            draining += 1;
            if draining >= PAST_THE_GATE {
                break;
            }
        }
        let from_height = height(c);
        c.run_until(epochs(2), |c| height(c) >= from_height + spacing);
    }
    for phase in [Phase::Live, Phase::Departing, Phase::Draining] {
        assert!(
            sent.iter().any(|(_, _, sent_in)| *sent_in == phase),
            "the train has to hold a transfer sent {phase:?}, or that phase's fate goes unread",
        );
    }

    // Every transfer's fate, once the splitter has reached its terminal
    // and nothing more can be included: whether the splitter took it is
    // what decides between settling and refusal.
    let (left, right) = splitter.children();
    assert!(
        await_serves(c, left, epochs(28)) && await_serves(c, right, epochs(28)),
        "both splitter children must be served within budget",
    );
    let mut never_included = 0;
    for (hash, to, phase) in &sent {
        let included = c.chain_fate(splitter, *hash).0.is_some();
        assert!(
            included || *phase == Phase::Draining,
            "a transfer sent {phase:?} must be included by the splitter",
        );
        never_included += usize::from(!included);
        let status = await_tx_terminal(c, *hash, epochs(12));
        let credited = match status {
            Some(TransactionStatus::Completed(TransactionDecision::Accept)) => {
                10 + STRADDLER_PAYMENT
            }
            Some(TransactionStatus::Completed(TransactionDecision::Aborted)) if !included => 10,
            other => panic!(
                "a transfer sent {phase:?} and {} by the splitter reached {other:?}",
                if included {
                    "included"
                } else {
                    "never included"
                },
            ),
        };
        assert!(
            c.run_until(epochs(8), |c| held(c, to.address(), *XRD) == credited),
            "a recipient of a transfer sent {phase:?} and {} by the splitter must hold \
             {credited}; holds {}",
            if included {
                "included"
            } else {
                "never included"
            },
            held(c, to.address(), *XRD),
        );
    }
    assert!(
        never_included > 0,
        "the train has to reach the splitter's coast, or nothing here crosses the cut",
    );
    world.assert_settles_within(c, &charges, epochs(8), "a train across a split's admission");
}

/// Submit one of the train's legs, recording the splitter's phase when it
/// went. The split shows as admitted while it pends and clears when the
/// gate drains, so a shard once admitted and no longer pending is
/// draining.
fn send_leg<C: Cluster>(
    c: &mut C,
    (key, from, to): &(Ed25519PrivateKey, PrincipalAddr, PrincipalAddr),
    charges: &mut Charges,
    sent: &mut Vec<(TxHash, PrincipalAddr, Phase)>,
    admitted_once: &mut bool,
) -> Phase {
    let pending = split_admitted(c, STRADDLER_SPLITTER);
    *admitted_once |= pending;
    let phase = match (pending, *admitted_once) {
        (true, _) => Phase::Departing,
        (false, false) => Phase::Live,
        (false, true) => Phase::Draining,
    };
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
