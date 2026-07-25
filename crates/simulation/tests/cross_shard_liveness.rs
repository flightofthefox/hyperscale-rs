//! Sustained cross-shard traffic must not cost a shard a round.
//!
//! A proposer resolves the reshape load predicate before it builds, and that
//! walk needs every uncommitted ancestor's substate byte delta. Cross-shard
//! blocks are the ones that reliably leave a replica behind on its local
//! execution, so a replica taking its proposer turn mid-catch-up finds the
//! walk unresolvable. It must park and resume when the walk's inputs land —
//! not sit out the view-change timeout, which burns a round on a fallback and
//! costs the shard ~15x its median block spacing.

use std::sync::Arc;
use std::time::Duration;

use hyperscale_scenarios::tx::{
    account_from_seed, build_transfer_tx, signer_from_seed, validity_around,
};
use hyperscale_scenarios::{Cluster, ScenarioConfig};
use hyperscale_storage::ShardChainReader;
use hyperscale_types::{BlockHeight, ShardId};
use radix_common::math::Decimal;
use radix_common::network::NetworkDefinition;

mod support;

use support::SimCluster;

/// Deterministic — the assertion is over every block both children commit, not
/// a particular interleaving, so any seed exercises it.
const SEED: u64 = 42;

/// Accounts the load draws from. Enough that consecutive transfers rarely
/// contend, which would hold them in the ready set rather than run them.
const ACCOUNTS: u8 = 8;

/// Transfers to submit once the topology has settled.
const TRANSFERS: u32 = 24;

/// Simulated time between submissions, and the settle time either side of the
/// run. Comfortably above the wave lifetime so each transfer terminates.
const SPACING: Duration = Duration::from_secs(10);

const fn grow_config() -> ScenarioConfig {
    ScenarioConfig {
        shard_size: 4,
        pool_surplus: 4,
        vnodes_per_host: 1,
        num_shards: 1,
        split_bytes: 0,
        latency: Duration::from_millis(150),
    }
}

/// Fallback blocks each host holds for `shard`, as `(height, round)`.
///
/// Read from whichever host is furthest along: hosts of one shard agree on
/// committed content, so the longest chain answers for all of them.
fn fallbacks(cluster: &SimCluster, shard: ShardId) -> Vec<(u64, u64)> {
    let runner = cluster.runner();
    let Some(storage) = (0..runner.num_hosts())
        .filter_map(|host| runner.hosts_shard(host, shard))
        .max_by_key(|storage| storage.committed_height())
    else {
        return Vec::new();
    };
    let mut found = Vec::new();
    let mut height = BlockHeight::GENESIS.next();
    while height <= storage.committed_height() {
        if let Some(certified) = storage.get_certified_header(height) {
            let header = certified.as_ref().header();
            if header.is_fallback() {
                found.push((header.height().inner(), header.round().inner()));
            }
        }
        height = height.next();
    }
    found
}

#[test]
fn cross_shard_load_costs_no_view_changes() {
    let balances: Vec<_> = (1..=ACCOUNTS)
        .map(|seed| (account_from_seed(seed), Decimal::from(100_000)))
        .collect();
    let mut cluster = SimCluster::with_balances(&grow_config(), SEED, &balances);
    cluster.runner_mut().grow_to(2);

    let (left, right) = ShardId::ROOT.children();
    // Let the reshape drain out of the children's chains, so the traffic below
    // runs against a settled topology and nothing here is attributed to the
    // split's own boundary work.
    let settled_from = {
        let now = cluster.runner().now() + SPACING;
        cluster.runner_mut().run_until(now);
        [left, right].map(|shard| fallbacks(&cluster, shard).len())
    };

    for nonce in 0..TRANSFERS {
        let from = u8::try_from(nonce % u32::from(ACCOUNTS)).expect("modulo fits u8") + 1;
        let to = (from % ACCOUNTS) + 1;
        let transfer = build_transfer_tx(
            &signer_from_seed(from),
            account_from_seed(from),
            account_from_seed(to),
            Decimal::from(1),
            &NetworkDefinition::simulator(),
            nonce,
            validity_around(cluster.now()),
        );
        cluster.submit(Arc::new(transfer));
        let next = cluster.runner().now() + SPACING;
        cluster.runner_mut().run_until(next);
    }

    for (shard, before) in [left, right].into_iter().zip(settled_from) {
        let burned = &fallbacks(&cluster, shard)[before..];
        assert!(
            burned.is_empty(),
            "shard {shard} burned a round under cross-shard load; \
             fallback blocks at (height, round) {burned:?}"
        );
    }
}
