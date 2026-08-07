//! Two waves writing one cell settle in the order they executed.
//!
//! A receipt carries an absolute cell value computed from its tick's
//! baseline, and settlement is last writer per cell
//! (`merge_writes_from_receipts`). Those two together are sound only while
//! settlement order agrees with tick order: a later tick's absolute already
//! contains every earlier tick's effect and may overwrite them, while an
//! earlier tick's contains none of the later ones and must not.
//!
//! Nothing about finalization enforces the agreement — two waves whose
//! leaders differ in speed can finalize either way round — so the ordering
//! is imposed where certificates enter a block: the proposer offers them in
//! tick order, and the pre-vote gate refuses a list that does not.

mod common;

use common::sim::{ExecutionSim, Schedule, cell_of, counter, settle};
use hyperscale_types::test_utils::{test_prefix, test_transaction_with_prefixes};
use hyperscale_types::{Transaction, WaveId};

/// The one cell both transactions write.
const CELL: u8 = 7;

/// A cell nothing else touches.
const OTHER: u8 = 9;

fn tx(seed: u8, cell: u8) -> Transaction {
    test_transaction_with_prefixes(&[seed, seed ^ 0x5a, seed ^ 0xa5], &[], &[test_prefix(cell)])
}

/// Two transactions over one cell in consecutive blocks. The second reads
/// what the first wrote, so its receipt carries the count of both.
fn two_over_one_cell() -> (ExecutionSim, Vec<WaveId>) {
    let mut sim = ExecutionSim::new(Schedule::Eager);
    let mut waves = Vec::new();
    for seed in 0..2u8 {
        let transaction = tx(seed, CELL);
        let hash = transaction.hash();
        sim.commit(vec![transaction], Vec::new());
        sim.drain();
        waves.push(sim.wave_of(hash).expect("wave assigned"));
    }
    assert_eq!(
        counter(sim.read(cell_of(test_prefix(CELL)))),
        2,
        "the second transaction must have read the first's write"
    );
    (sim, waves)
}

/// Settling in tick order keeps both writes — the basis of chaining onto
/// absolutes, and the control for everything below.
#[test]
fn settling_in_tick_order_keeps_both_writes() {
    let (mut sim, waves) = two_over_one_cell();
    for wave in &waves {
        let receipts = sim.receipts_for(wave);
        sim.commit(Vec::new(), vec![settle(wave, &receipts)]);
    }
    sim.drain();
    assert_eq!(counter(sim.settled(cell_of(test_prefix(CELL)))), 2);
}

/// The reverse is refused before it can be voted on: the earlier receipt's
/// absolute was computed before the later transaction existed, and landing
/// it last would revert a committed write.
#[test]
fn a_certificate_settling_ahead_of_its_predecessor_is_refused() {
    let (sim, waves) = two_over_one_cell();
    assert_eq!(
        sim.settles_out_of_order(&[waves[1].clone(), waves[0].clone()]),
        Some(waves[1].clone()),
        "the later tick's certificate may not settle first"
    );
    assert_eq!(
        sim.settles_out_of_order(&[waves[0].clone(), waves[1].clone()]),
        None,
        "tick order is what the rule asks for, not any order"
    );
}

/// Waves that share no cell are unordered — the rule is per cell, not a
/// serialization of the settlement pipeline.
#[test]
fn waves_over_disjoint_cells_settle_in_any_order() {
    let mut sim = ExecutionSim::new(Schedule::Eager);
    let mut waves = Vec::new();
    for (seed, cell) in [(0u8, CELL), (1, OTHER)] {
        let transaction = tx(seed, cell);
        let hash = transaction.hash();
        sim.commit(vec![transaction], Vec::new());
        sim.drain();
        waves.push(sim.wave_of(hash).expect("wave assigned"));
    }
    assert_eq!(
        sim.settles_out_of_order(&[waves[1].clone(), waves[0].clone()]),
        None
    );
}
