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
    ExecutionSim, FLOOR, LEFT, Schedule, amount, cell_of, charge_of, counter, settle,
    settle_refused_by_counterpart, vault_of,
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

/// Settle a crossing with `refused`, and report what reached state: the
/// cell its effects would have written, the vault they would have
/// credited, and the charge it owes for having been refused.
fn settled_local_effect(refused: bool) -> (u64, u128, u128) {
    let mut sim = ExecutionSim::with_shards(Schedule::Eager, 2, LEFT);

    let leg = crossing(0);
    let hash = leg.hash();
    sim.commit(vec![leg], Vec::new());
    // The payer's leg runs in the tick that attests it, so it waits for
    // the counterpart to commit the transaction and echo that back.
    sim.engage(ShardId::leaf(1, 1), &[hash]);
    sim.drain();

    let wave = sim.wave_of(hash).expect("the crossing joined a tick");
    let receipts = sim.receipts_for(&wave);
    assert!(!receipts.is_empty(), "the local half executed");

    let finalized = if refused {
        settle_refused_by_counterpart(
            &wave,
            ShardId::leaf(1, 1),
            &receipts,
            &sim.charges_for(&wave),
        )
    } else {
        settle(&wave, &receipts)
    };
    sim.commit(Vec::new(), vec![finalized]);
    sim.drain();
    (
        counter(sim.settled(cell_of(test_prefix(LOCAL)))),
        amount(sim.settled(vault_of(test_prefix(LOCAL)))),
        amount(sim.settled(charge_of(test_prefix(LOCAL)))),
    )
}

/// The control: with every participant accepting, the local half settles
/// and there is nothing to charge.
#[test]
fn an_accepted_crossing_settles_its_local_half() {
    assert_eq!(settled_local_effect(false), (1, 1, 0));
}

/// The refusal: none of the effects settle, because nothing happened —
/// and the charge does, because the attempt did.
///
/// The two halves fail in opposite directions and both matter. Settling
/// the effects moves value one-sidedly; dropping the charge with them
/// makes a transaction whose counterpart refuses it free, and leaves the
/// tick chain holding a debit state never takes.
#[test]
fn a_crossing_refused_by_its_counterpart_settles_only_its_charge() {
    assert_eq!(
        settled_local_effect(true),
        (0, 0, FLOOR),
        "a refused crossing must settle its charge and none of its effects"
    );
}
