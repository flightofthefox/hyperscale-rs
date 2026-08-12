//! A replica that restarts rejoins the committee it left.
//!
//! The committed chain survives a restart and everything execution held
//! does not: which tick holds which transaction, and what each tick's
//! baseline was. Both are functions of committed content, and a replica
//! that fails to rebuild them composes a tick its peers do not — which
//! is a fail-stop, not a dropped vote.
//!
//! Sim-only. The restart primitive tears a vnode down and seats it again
//! on the storage it kept, which production would have to do by bouncing
//! a process.

mod support;

use std::sync::Arc;
use std::time::Duration;

use hyperscale_scenarios::tx::{
    build_transfer_tx, genesis_accounts, recipient, sender, validity_around,
};
use hyperscale_scenarios::wait::await_tx_terminal;
use hyperscale_scenarios::{Cluster, FaultableCluster, ScenarioConfig, epochs};
use hyperscale_types::{ShardId, TransactionStatus, TxHash};
use support::SimCluster;

/// Single shard, four-validator committee, resharding disarmed.
const fn one_shard() -> ScenarioConfig {
    ScenarioConfig {
        shard_size: 4,
        vnodes_per_host: 1,
        pool_surplus: 0,
        num_shards: 1,
        split_bytes: u64::MAX,
        latency: Duration::from_millis(150),
    }
}

/// Whether `shard` has committed `tx` and still owes it an outcome —
/// exactly the window a restart has to replay across. Restarting outside
/// it replays nothing and measures nothing.
fn owed_an_outcome(c: &SimCluster, shard: ShardId, tx: TxHash) -> bool {
    let (committed, outcome) = c.chain_fate(shard, tx);
    committed.is_some() && outcome.is_none()
}

/// The chain advances again after `restarted` of four members bounce
/// together, with traffic either side of the bounce.
fn restart_and_advance(restarted: usize) {
    let mut cluster = SimCluster::with_accounts(&one_shard(), 42, &genesis_accounts(8, 1));
    let shard = ShardId::ROOT;
    let (payer, from) = sender(0);

    for index in 0..4u8 {
        let tx = build_transfer_tx(
            &payer,
            from,
            recipient(index),
            10,
            validity_around(cluster.now()),
        );
        cluster.submit(Arc::new(tx));
    }
    assert!(
        cluster.run_until(epochs(8), |c| c
            .committed_height(shard)
            .is_some_and(|h| h.inner() > 3)),
        "the chain must be running before the restart",
    );

    let hosts = cluster.committee_hosts(shard);
    let before = cluster
        .committed_height(shard)
        .expect("the chain is running");
    for &host in hosts.iter().take(restarted) {
        cluster.restart_host(host, shard);
    }

    let target = before.inner() + 5;
    assert!(
        cluster.run_until(epochs(24), |c| c
            .committed_height(shard)
            .is_some_and(|h| h.inner() >= target)),
        "the chain must advance past {target:?} after {restarted} of four restart; \
         it reached {:?}",
        cluster.committed_height(shard),
    );
}

/// A committee keeps committing when part of it restarts at once.
///
/// One of four is carried by the three that stayed up, and the survivor
/// count is what makes that case say so little: the chain commits without
/// the restarted replica, and the commit is what reseats whatever its
/// rebuild missed. At two the quorum needs them back, so anything a
/// restart fails to rebuild stops the shard instead of healing behind it
/// — which is why the sweep runs to the largest minority rather than
/// asserting one restart and calling the path covered.
#[test]
fn a_committee_advances_after_part_of_it_restarts() {
    for restarted in 1..=3 {
        restart_and_advance(restarted);
    }
}

/// Every replica restarting at once is not this test's case.
///
/// Nothing carries the chain: the committee comes back holding its
/// committed tip and the certified block above it is gone from every
/// replica at once, so the QC they all recover is the one below where
/// they stopped. They propose and vote and no quorum forms.
#[test]
#[ignore = "known gap: a committee whose every replica restarts together \
            re-proposes without ever forming a quorum"]
fn a_committee_advances_after_all_of_it_restarts() {
    restart_and_advance(4);
}

/// A restarted member agrees with its peers about what it executed.
///
/// The membership assertion in its strong form. Composing a tick alone
/// aborts the process through `escalate_divergence` — which a replayed
/// replica is exposed to whether or not its vote reaches anyone, since
/// building the vote is what latches the local root the returning
/// certificate is reconciled against. So surviving the run says
/// something; agreeing on the committed state root at one height says
/// the whole of it.
#[test]
fn a_restarted_member_agrees_on_the_state_it_rebuilt() {
    let mut cluster = SimCluster::with_accounts(&one_shard(), 42, &genesis_accounts(8, 1));
    let shard = ShardId::ROOT;
    let (payer, from) = sender(0);

    // Traffic either side of the restart, so the replica comes back with
    // something outstanding and keeps composing afterwards.
    let mut submitted = Vec::new();
    for index in 0..4u8 {
        let tx = build_transfer_tx(
            &payer,
            from,
            recipient(index),
            10,
            validity_around(cluster.now()),
        );
        submitted.push(tx.hash());
        cluster.submit(Arc::new(tx));
    }
    let held = submitted[0];
    assert!(
        cluster.run_until(epochs(8), |c| owed_an_outcome(c, shard, held)),
        "the shard must be holding something for the restart to have to rebuild",
    );

    let host = *cluster
        .committee_hosts(shard)
        .first()
        .expect("the shard has a seated committee");
    cluster.restart_host(host, shard);

    for tx in submitted {
        let status = await_tx_terminal(&mut cluster, tx, epochs(24));
        assert!(
            matches!(status, Some(TransactionStatus::Completed(_))),
            "every transaction must reach an outcome across the restart; status = {status:?}",
        );
    }

    // And the restarted replica computed the same state as a peer that
    // never went down, at a height they both hold.
    let peer = *cluster
        .committee_hosts(shard)
        .iter()
        .find(|&&h| h != host)
        .expect("a peer that never restarted");
    let height = cluster
        .host_committed_height(host, shard)
        .expect("the restarted host is committing")
        .min(
            cluster
                .host_committed_height(peer, shard)
                .expect("the peer is committing"),
        );
    let root_at = |c: &SimCluster, h: usize| {
        c.host_block(h, shard, height)
            .map(|certified| certified.block().header().state_root())
    };
    assert!(
        root_at(&cluster, host).is_some(),
        "the restarted host must hold the height it is compared at",
    );
    assert_eq!(
        root_at(&cluster, host),
        root_at(&cluster, peer),
        "a restarted replica's fold must be the one its committee agreed on",
    );
}
