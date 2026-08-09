//! What a restart has to rebuild before it can execute again.
//!
//! The committed chain survives a restart and everything execution was
//! tracking does not. Two replicas at one committed frontier still have
//! to compose the same tick and execute it against the same baseline, so
//! whatever the restart lost has to come back out of committed content.

mod common;

use common::sim::{ExecutionSim, Schedule, cell_of, counter, settle};
use hyperscale_types::Transaction;
use hyperscale_types::test_utils::{test_prefix, test_transaction_with_prefixes};

/// The cell the fixture writes.
const CELL: u8 = 7;

fn tx(seed: u8, cell: u8) -> Transaction {
    test_transaction_with_prefixes(&[seed, seed ^ 0x5a, seed ^ 0xa5], &[], &[test_prefix(cell)])
}

/// A tick's writes are readable through the tick chain from the moment
/// it executes and reach the base only when its certificate settles
/// them. A restart in between drops the chain, so the baseline the next
/// tick executes against has to be rebuilt by replaying the blocks that
/// produced it.
#[test]
fn a_restart_rebuilds_the_baseline_an_unsettled_tick_left() {
    let mut sim = ExecutionSim::new(Schedule::Eager);
    sim.commit(vec![tx(0, CELL)], Vec::new());
    sim.drain();

    let cell = cell_of(test_prefix(CELL));
    assert_eq!(
        counter(sim.read(cell)),
        1,
        "fixture precondition: the write is readable through the chain",
    );
    assert_eq!(
        counter(sim.settled(cell)),
        0,
        "fixture precondition: nothing has settled it into the base",
    );

    sim.restart();
    assert_eq!(
        counter(sim.read(cell)),
        1,
        "so the restarted replica executes the next tick against the same baseline",
    );

    // And the tick above it reads that baseline rather than the bare
    // base, which is the whole reason the fold has to come back.
    sim.commit(vec![tx(1, CELL)], Vec::new());
    sim.drain();
    assert_eq!(
        counter(sim.read(cell)),
        2,
        "a tick after the restart chains onto what the replayed one wrote",
    );
}

/// A tick already settled into the base is not folded a second time.
///
/// Replay re-runs every tick in the window, including ones whose
/// certificate has since committed — so the fold and the base both hold
/// their effect, and counting it twice would be as wrong as losing it.
#[test]
fn a_restart_does_not_refold_what_the_base_already_carries() {
    let mut sim = ExecutionSim::new(Schedule::Eager);
    sim.commit(vec![tx(0, CELL)], Vec::new());
    sim.drain();

    let tick = sim.tick_of(tx(0, CELL).hash()).expect("tick assigned");
    let receipts = sim.receipts_for(&tick);
    sim.commit(Vec::new(), vec![settle(&tick, &receipts)]);
    sim.drain();

    let cell = cell_of(test_prefix(CELL));
    assert_eq!(
        counter(sim.settled(cell)),
        1,
        "fixture precondition: the certificate put it in the base",
    );

    sim.restart();
    assert_eq!(
        counter(sim.read(cell)),
        1,
        "a replayed tick the base already carries contributes nothing further",
    );
}
