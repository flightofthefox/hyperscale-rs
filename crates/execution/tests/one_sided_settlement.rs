//! A transaction refused by one participant settles nowhere.
//!
//! A cross-shard transaction runs at every shard its effects reach, and
//! each shard reports what its own half did. The verdict is the combine of
//! those reports, not any one of them: a leg that completed here while its
//! payer shard found the reservation infeasible describes a transaction
//! that did not happen. Applying this shard's half regardless would move
//! value one-sidedly — the recipient credited, the payer never debited.

mod common;

use common::sim::{
    ExecutionSim, LEFT, Schedule, cell_of, counter, settle, settle_refused_by_counterpart,
};
use hyperscale_types::test_utils::{test_prefix, test_transaction_with_prefixes};
use hyperscale_types::{ShardId, Transaction};

/// The local cell the crossing writes.
const LOCAL: u8 = 7;

/// A prefix on the far side of the partition.
const REMOTE: u8 = 200;

fn crossing(seed: u8) -> Transaction {
    test_transaction_with_prefixes(
        &[seed, seed ^ 0x33, seed ^ 0xcc],
        &[],
        &[test_prefix(LOCAL), test_prefix(REMOTE)],
    )
}

/// Settle a crossing with `refused`, and report what reached state.
fn settled_local_effect(refused: bool) -> u64 {
    let mut sim = ExecutionSim::with_shards(Schedule::Eager, 2, LEFT);

    let leg = crossing(0);
    let hash = leg.hash();
    sim.commit(vec![leg], Vec::new());
    sim.drain();

    let wave = sim.wave_of(hash).expect("the crossing has a wave");
    let receipts = sim.receipts_for(&wave);
    assert!(!receipts.is_empty(), "the local half executed");

    let finalized = if refused {
        settle_refused_by_counterpart(&wave, ShardId::leaf(1, 1), &receipts)
    } else {
        settle(&wave, &receipts)
    };
    sim.commit(Vec::new(), vec![finalized]);
    sim.drain();
    counter(sim.settled(cell_of(test_prefix(LOCAL))))
}

/// The control: with every participant accepting, the local half settles.
#[test]
fn an_accepted_crossing_settles_its_local_half() {
    assert_eq!(settled_local_effect(false), 1);
}

/// The refusal: nothing settles, because nothing happened.
#[test]
fn a_crossing_refused_by_its_counterpart_settles_nothing() {
    assert_eq!(
        settled_local_effect(true),
        0,
        "a shard applied its own half of a transaction the wave refused"
    );
}
