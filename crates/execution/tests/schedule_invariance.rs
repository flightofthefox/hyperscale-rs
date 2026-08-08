//! A tick's output is a function of the committed chain, not of local timing.
//!
//! Execution runs behind consensus and at its own pace: a replica under load
//! composes ticks well ahead of running them, and the resolutions those ticks
//! gate are emitted correspondingly later. None of that may reach a receipt.
//! If it did, two honest replicas on the same chain would derive different
//! writes, `reconcile_local_ec_root` would latch `locally_divergent`, and the
//! slower one would quietly stop contributing to every wave it holds.
//!
//! So the lane fixes the committed chain and quantifies over the schedule.

mod common;

use common::sim::{
    CREDIT, ExecutionSim, LEFT, Schedule, amount, cell_of, counter, settle, vault_of,
};
use hyperscale_storage::TickOutput;
use hyperscale_types::test_utils::{test_prefix, test_transaction_with_prefixes};
use hyperscale_types::{BlockHeight, Transaction};

/// A local prefix — under a two-shard partition it routes to [`LEFT`].
const LOCAL: u8 = 7;

/// A prefix on the far side of a two-shard partition.
const REMOTE: u8 = 200;

/// A single-shard transaction: one declared cell, no reads, so it
/// provisions nothing and joins the tick its block commits.
fn local_tx(seed: u8, owner: u8) -> Transaction {
    test_transaction_with_prefixes(
        &[seed, seed ^ 0x5a, seed ^ 0xa5],
        &[],
        &[test_prefix(owner)],
    )
}

/// A transaction declaring cells on both sides of the partition: a
/// cross-shard wave, whose local writes are provisional until it settles.
/// It provisions nothing either, so it joins its block's tick at once.
fn crossing(seed: u8, owner: u8) -> Transaction {
    test_transaction_with_prefixes(
        &[seed, seed ^ 0x33, seed ^ 0xcc],
        &[],
        &[test_prefix(owner), test_prefix(REMOTE)],
    )
}

/// Three blocks each carrying one transaction over one cell, so the second
/// reads what the first wrote and the third reads what both did. A baseline
/// that lost a predecessor shows up as a short count.
fn chained(schedule: Schedule) -> Vec<(BlockHeight, TickOutput)> {
    let mut sim = ExecutionSim::new(schedule);
    for seed in 0..3u8 {
        sim.commit(vec![local_tx(seed, LOCAL)], Vec::new());
    }
    sim.drain();
    assert_eq!(
        counter(sim.read(cell_of(test_prefix(LOCAL)))),
        3,
        "each transaction must have read what its predecessors wrote"
    );
    // The other half of what a receipt says. The counter is an absolute
    // and re-applying one is harmless; the credit states what it moved,
    // so a fold that drops it or applies it twice shows up only here.
    assert_eq!(
        amount(sim.read(vault_of(test_prefix(LOCAL)))),
        3 * CREDIT,
        "one credit per transaction must reach the baseline, exactly once"
    );
    sim.outputs().to_vec()
}

/// Holding every completion back until later blocks have committed changes
/// when ticks run and when the resolutions they gate are emitted, and
/// changes nothing about what they produce.
#[test]
fn tick_outputs_do_not_move_with_execution_lag() {
    let eager = chained(Schedule::Eager);
    assert_eq!(eager.len(), 3, "one tick per block carrying work");
    for lag in 1..=3 {
        assert_eq!(
            chained(Schedule::Lagged(lag)),
            eager,
            "lagging execution by {lag} blocks changed a tick output"
        );
    }
}

/// The same over a cross-shard wave, whose writes are provisional.
///
/// A settlement promotes those entries into the readable fold, so *when*
/// it is applied relative to a later tick's dispatch is exactly the timing
/// that could leak into a receipt. It cannot, because a transaction
/// declaring a claimed cell is kept out of every tick until the claim
/// clears — so no tick ever has a member whose read the promotion moves.
fn with_a_crossing(schedule: Schedule) -> (Vec<(BlockHeight, TickOutput)>, u64) {
    let mut sim = ExecutionSim::with_shards(schedule, 2, LEFT);

    let leg = crossing(0, LOCAL);
    let leg_hash = leg.hash();
    sim.commit(vec![leg], Vec::new());
    sim.drain();
    let wave = sim.wave_of(leg_hash).expect("the crossing has a tick");
    assert_eq!(
        counter(sim.read(cell_of(test_prefix(LOCAL)))),
        0,
        "a provisional write must not be readable"
    );

    // A follower over the same cell: claimed, so it waits.
    sim.commit(vec![local_tx(1, LOCAL)], Vec::new());
    sim.drain();

    // The crossing settles; the follower enters the next tick composed.
    let receipts = sim.receipts_for(&wave);
    assert!(!receipts.is_empty(), "the crossing produced a receipt");
    sim.commit(Vec::new(), vec![settle(&wave, &receipts)]);
    sim.commit(Vec::new(), Vec::new());
    sim.drain();

    assert_eq!(
        amount(sim.settled(vault_of(test_prefix(LOCAL)))),
        CREDIT,
        "the crossing's credit must settle exactly once"
    );
    let settled = counter(sim.settled(cell_of(test_prefix(LOCAL))));
    (sim.outputs().to_vec(), settled)
}

#[test]
fn tick_outputs_do_not_move_with_resolution_timing() {
    let (eager, settled) = with_a_crossing(Schedule::Eager);
    assert_eq!(settled, 1, "the crossing's write settled exactly once");
    for lag in 1..=3 {
        assert_eq!(
            with_a_crossing(Schedule::Lagged(lag)),
            (eager.clone(), settled),
            "lagging execution by {lag} blocks changed a tick output"
        );
    }
}

/// A tick reads its own anchor, and the chain evicts a fold once the base
/// "covers" it — but the base covers it at the height its wave *settled*,
/// which can be above the anchor a queued tick will read from.
///
/// So a tick that runs before the eviction sees its predecessor's write in
/// the fold, and one that runs after sees neither the fold nor a base old
/// enough to hold it. Same committed chain, two answers.
fn evicted_between(schedule: Schedule) -> (u64, u128) {
    let mut sim = ExecutionSim::new(schedule);

    let first = local_tx(0, LOCAL);
    let first_hash = first.hash();
    sim.commit(vec![first], Vec::new());

    // The follower: composed now, run whenever the schedule says.
    sim.commit(vec![local_tx(1, LOCAL)], Vec::new());

    // The predecessor's certificate. Settling it lets the chain evict its
    // fold, and the base only gains the write at this height.
    let wave = sim.wave_of(first_hash).expect("wave assigned");
    let receipts = sim.receipts_for(&wave);
    assert!(!receipts.is_empty(), "the predecessor executed");
    sim.commit(Vec::new(), vec![settle(&wave, &receipts)]);

    sim.drain();
    // Read together: the settled fold and the base both carry the
    // predecessor's contribution for a window, and the reader has to take
    // exactly one of them. An absolute cannot tell the difference; the
    // credit can.
    (
        counter(sim.read(cell_of(test_prefix(LOCAL)))),
        amount(sim.read(vault_of(test_prefix(LOCAL)))),
    )
}

#[test]
fn a_fold_evicted_before_its_reader_runs_does_not_change_the_answer() {
    assert_eq!(
        evicted_between(Schedule::Lagged(1)),
        evicted_between(Schedule::Eager),
        "eviction raced a queued tick: the follower lost its predecessor's write"
    );
}
