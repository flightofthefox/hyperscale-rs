//! Two ticks writing one cell settle in the order they executed.
//!
//! A receipt carries an absolute cell value computed from its tick's
//! baseline, and settlement is last writer per cell
//! (`merge_writes_from_receipts`). Those two together are sound only while
//! settlement order agrees with tick order: a later tick's absolute already
//! contains every earlier tick's effect and may overwrite them, while an
//! earlier tick's contains none of the later ones and must not.
//!
//! Nothing about finalization establishes the agreement — two ticks whose
//! leaders differ in speed finalize either way round — so it is the
//! proposal that has to. It does so by construction rather than by a rule:
//! a tick's certificate settles what that tick produced, ticks execute in
//! height order, and `TickId` sorts by height, so the store's own order is
//! settlement order and a proposer has no other list to offer.

mod common;

use common::sim::{ExecutionSim, Schedule, cell_of, counter, settle};
use hyperscale_types::test_utils::{test_prefix, test_transaction_with_prefixes};
use hyperscale_types::{TickId, Transaction};

/// The one cell both transactions write.
const CELL: u8 = 7;

/// A cell nothing else touches.
const OTHER: u8 = 9;

fn tx(seed: u8, cell: u8) -> Transaction {
    test_transaction_with_prefixes(&[seed, seed ^ 0x5a, seed ^ 0xa5], &[], &[test_prefix(cell)])
}

/// Two transactions over one cell in consecutive blocks. The second reads
/// what the first wrote, so its receipt carries the count of both.
fn two_over_one_cell() -> (ExecutionSim, Vec<TickId>) {
    let mut sim = ExecutionSim::new(Schedule::Eager);
    let mut ticks = Vec::new();
    for seed in 0..2u8 {
        let transaction = tx(seed, CELL);
        let hash = transaction.hash();
        sim.commit(vec![transaction], Vec::new());
        sim.drain();
        ticks.push(sim.tick_of(hash).expect("tick assigned"));
    }
    assert_eq!(
        counter(sim.read(cell_of(test_prefix(CELL)))),
        2,
        "the second transaction must have read the first's write"
    );
    (sim, ticks)
}

/// Settling in tick order keeps both writes — the basis of chaining onto
/// absolutes, and the control for everything below.
#[test]
fn settling_in_tick_order_keeps_both_writes() {
    let (mut sim, ticks) = two_over_one_cell();
    for tick in &ticks {
        let receipts = sim.receipts_for(tick);
        sim.commit(Vec::new(), vec![settle(tick, &receipts)]);
    }
    sim.drain();
    assert_eq!(counter(sim.settled(cell_of(test_prefix(CELL)))), 2);
}

/// The reverse is unconstructable rather than refused. Both
/// finalizations are handed over later-first, so the ready set cannot be
/// in tick order by arrival, and the proposal still comes out in it —
/// there is no list a proposer could offer that settles the later tick's
/// absolute ahead of the write it already contains.
#[test]
fn the_proposal_carries_certificates_in_tick_order() {
    let (sim, ticks) = two_over_one_cell();
    for tick in ticks.iter().rev() {
        let receipts = sim.receipts_for(tick);
        sim.admit(settle(tick, &receipts));
    }
    assert_eq!(
        sim.offered_finalizations(),
        ticks,
        "the proposal is in tick order whatever order finalization reached it in"
    );
}

/// Ticks that share no cell land on the same state whichever order
/// settles them: each carries only what it touched, so there is no
/// absolute for the other to revert. The order is unconditional, and this
/// is why paying for it costs nothing.
#[test]
fn ticks_over_disjoint_cells_settle_to_the_same_state_in_any_order() {
    let mut sim = ExecutionSim::new(Schedule::Eager);
    let mut ticks = Vec::new();
    for (seed, cell) in [(0u8, CELL), (1, OTHER)] {
        let transaction = tx(seed, cell);
        let hash = transaction.hash();
        sim.commit(vec![transaction], Vec::new());
        sim.drain();
        ticks.push(sim.tick_of(hash).expect("tick assigned"));
    }

    // Committed in the reverse of tick order.
    for tick in ticks.iter().rev() {
        let receipts = sim.receipts_for(tick);
        sim.commit(Vec::new(), vec![settle(tick, &receipts)]);
    }
    sim.drain();

    assert_eq!(counter(sim.settled(cell_of(test_prefix(CELL)))), 1);
    assert_eq!(counter(sim.settled(cell_of(test_prefix(OTHER)))), 1);
}
