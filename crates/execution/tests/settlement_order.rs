//! What settles depends on the order certificates commit in, and it must not.
//!
//! A receipt carries an absolute cell value computed from its tick's
//! baseline, and settlement is last writer per cell
//! (`merge_writes_from_receipts`). Those two together are only sound while
//! settlement order agrees with tick order: a later tick's absolute already
//! contains every earlier tick's effect, so it may overwrite them, but an
//! earlier tick's absolute contains none of the later ones and must not.
//!
//! Nothing enforces the agreement. `get_finalized_waves` hands the proposer
//! everything ready in `WaveId` order and block validity imposes no ordering
//! on `certificates()`, so two waves whose leaders differ in speed can
//! finalize either way round.
//!
//! The test below states the present behaviour rather than the intended one:
//! the two orders disagree, and the losing one drops a committed write. It
//! is the shape a fix has to flip.

mod common;

use common::sim::{ExecutionSim, Schedule, cell_of, counter, settle};
use hyperscale_types::Transaction;
use hyperscale_types::test_utils::{test_prefix, test_transaction_with_prefixes};

/// The one cell both transactions write.
const CELL: u8 = 7;

fn tx(seed: u8) -> Transaction {
    test_transaction_with_prefixes(&[seed, seed ^ 0x5a, seed ^ 0xa5], &[], &[test_prefix(CELL)])
}

/// Two transactions over one cell in consecutive blocks, their certificates
/// committed in `order`. Returns what settled.
///
/// The second reads what the first wrote — that is the chaining working —
/// so its receipt carries the count of both. The first's carries only its
/// own.
fn settled_under(order: [usize; 2]) -> u64 {
    let mut sim = ExecutionSim::new(Schedule::Eager);

    let mut waves = Vec::new();
    for seed in 0..2u8 {
        let transaction = tx(seed);
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

    for index in order {
        let receipts = sim.receipts_for(&waves[index]);
        sim.commit(Vec::new(), vec![settle(&waves[index], &receipts)]);
    }
    sim.drain();
    counter(sim.settled(cell_of(test_prefix(CELL))))
}

/// Settling in tick order keeps both writes: the later absolute subsumes
/// the earlier one, which is the whole basis of chaining onto absolutes.
#[test]
fn settling_in_tick_order_keeps_both_writes() {
    assert_eq!(settled_under([0, 1]), 2);
}

/// Settling in the opposite order drops the later write. The earlier
/// receipt's absolute was computed before the later transaction existed, so
/// landing it last reverts state to a value no longer true.
///
/// Both transactions committed, both executed, both certified — and one is
/// gone from state. A fix makes this equal the tick-order result.
#[test]
fn settling_out_of_tick_order_drops_a_committed_write() {
    assert_eq!(settled_under([1, 0]), 1);
    assert_ne!(
        settled_under([1, 0]),
        settled_under([0, 1]),
        "settlement order must not decide which committed write survives"
    );
}
