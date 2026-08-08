//! A claim holds inside the tick that makes it, not only after it.
//!
//! A cross-shard leg's local writes are provisional: the wave can still
//! refuse them, and then they never happened. What keeps a later
//! transaction from reading one is the claim its declared cells put on
//! the tick chain — and a tick is one batch over one overlay, so a
//! transaction sharing the leg's tick reads exactly the same provisional
//! value that a transaction in the next tick is kept away from.
//!
//! Sharing it is worse, in fact. A single-shard transaction's writes are
//! determined at commit and settle unconditionally, so a leg's effect
//! folded into one of them survives the abort that was supposed to
//! discard it — the cross-shard half applied on one side and nowhere
//! else.

mod common;

use common::sim::{ExecutionSim, LEFT, Schedule, cell_of, counter, settle};
use hyperscale_types::Transaction;
use hyperscale_types::test_utils::{test_prefix, test_transaction_with_prefixes};

/// The cell both the crossing and the follower write.
const SHARED: u8 = 7;

/// A cell only the control transaction writes.
const OTHER: u8 = 9;

/// A prefix on the far side of the partition.
const REMOTE: u8 = 200;

/// A transaction declaring cells on both sides of the partition: a
/// cross-shard wave, whose local writes are provisional until it settles.
fn crossing(seed: u8, owner: u8) -> Transaction {
    test_transaction_with_prefixes(
        &[seed, seed ^ 0x33, seed ^ 0xcc],
        &[],
        &[test_prefix(owner), test_prefix(REMOTE)],
    )
}

/// A single-shard transaction over one declared cell.
fn local(seed: u8, owner: u8) -> Transaction {
    test_transaction_with_prefixes(
        &[seed, seed ^ 0x5a, seed ^ 0xa5],
        &[],
        &[test_prefix(owner)],
    )
}

/// The follower is kept out of the tick that carries the crossing, and
/// the control beside it is not: the exclusion is the shared cell, not
/// the fact that a crossing was in the block.
#[test]
fn a_leg_claims_against_the_tick_it_joins() {
    let mut sim = ExecutionSim::with_shards(Schedule::Eager, 2, LEFT);

    let leg = crossing(0, SHARED);
    let leg_hash = leg.hash();
    let follower = local(1, SHARED);
    let follower_hash = follower.hash();
    let control = local(2, OTHER);
    let control_hash = control.hash();

    sim.commit(vec![leg, follower, control], Vec::new());
    sim.drain();

    let wave = sim.wave_of(leg_hash).expect("the crossing has a wave");
    assert!(!wave.is_zero(), "the crossing composes a cross-shard wave");
    assert!(
        !sim.receipts_for(&wave).is_empty(),
        "the crossing itself executes"
    );

    let single = sim.wave_of(follower_hash).expect("the follower has a wave");
    assert!(single.is_zero(), "the follower is single-shard");
    let executed: Vec<_> = sim
        .receipts_for(&single)
        .into_iter()
        .map(|receipt| receipt.tx_hash)
        .collect();
    assert!(
        !executed.contains(&follower_hash),
        "the follower shared a batch with a leg whose writes it could read"
    );
    assert!(
        executed.contains(&control_hash),
        "the control declares nothing the leg claimed and must not wait"
    );

    // Nothing of the crossing is readable while its wave is open, which
    // is exactly what the follower would have folded over.
    assert_eq!(counter(sim.read(cell_of(test_prefix(SHARED)))), 0);

    // Once the wave settles the claim clears, and the follower enters
    // the next tick composed — reading the promoted value, not the one
    // its own block saw.
    let receipts = sim.receipts_for(&wave);
    sim.commit(Vec::new(), vec![settle(&wave, &receipts)]);
    sim.commit(Vec::new(), Vec::new());
    sim.drain();

    assert_eq!(
        counter(sim.read(cell_of(test_prefix(SHARED)))),
        2,
        "the follower must land on top of the crossing, not beside it"
    );
}

/// Two legs of *one* wave sharing a cell are the same hazard, and the
/// wave envelope does not close it.
///
/// A wave settles each member on its own verdict — `TickResolution`
/// carries the aborted set per transaction — so a counterpart can refuse
/// one leg while accepting its sibling. A sibling that folded over the
/// refused leg's writes carries them into state anyway, which is the
/// one-sided settlement the whole cross-shard design exists to prevent.
/// Two crossings with the same remote set land in one wave, so this is
/// the commonest shape rather than an exotic one.
#[test]
fn one_leg_claims_against_its_own_sibling() {
    let mut sim = ExecutionSim::with_shards(Schedule::Eager, 2, LEFT);

    let first = crossing(3, SHARED);
    let first_hash = first.hash();
    let second = crossing(4, SHARED);
    let second_hash = second.hash();

    sim.commit(vec![first, second], Vec::new());
    sim.drain();

    let wave = sim.wave_of(first_hash).expect("the crossing has a wave");
    assert_eq!(
        sim.wave_of(second_hash).as_ref(),
        Some(&wave),
        "two crossings over one remote set share a wave"
    );
    assert_eq!(
        sim.receipts_for(&wave).len(),
        1,
        "one leg claims the cell; its sibling waits for the verdict"
    );
}

/// A shard whose transactions touch nothing in common composes them all
/// into one tick — the claim costs nothing where nothing is shared.
#[test]
fn disjoint_declarations_all_join_one_tick() {
    let mut sim = ExecutionSim::new(Schedule::Eager);

    let txs: Vec<Transaction> = (0..4u8).map(|seed| local(seed, 20 + seed)).collect();
    let hashes: Vec<_> = txs.iter().map(Transaction::hash).collect();
    sim.commit(txs, Vec::new());
    sim.drain();

    let wave = sim.wave_of(hashes[0]).expect("a single-shard wave");
    let executed: Vec<_> = sim
        .receipts_for(&wave)
        .into_iter()
        .map(|receipt| receipt.tx_hash)
        .collect();
    for hash in hashes {
        assert!(executed.contains(&hash), "every disjoint member executes");
    }
    assert_eq!(sim.outputs().len(), 1, "one tick, not four");
}
