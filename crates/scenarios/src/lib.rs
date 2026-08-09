//! Portable node-behavioral scenarios.
//!
//! A *scenario* is a plain synchronous function over an abstract [`Cluster`]:
//! it drives the cluster from a precondition to a postcondition and asserts the
//! postcondition. The same function body runs on both harnesses — the
//! simulation's logical clock and production's wall-clock QUIC + `RocksDB`
//! cluster — via two thin adaptors that each implement [`Cluster`]. A scenario
//! that passes on one harness and fails on the other is then a real divergence,
//! not a test-authoring artefact.
//!
//! Each module at the crate root is one such scenario (or a small family of
//! them). The harness-agnostic vocabulary they are written against — the
//! [`Cluster`] trait, [`ScenarioConfig`], [`Budget`], and the [`query`],
//! [`wait`], [`tx`], and [`grow_to`] helpers — lives in [`support`]. The two
//! adaptors (`SimCluster`, `ProdCluster`) are supplied by the test crates that
//! depend on this one.

mod support;

mod contention;
mod execution;
mod faults;
mod liveness;
mod multi_vnode;
mod reshape;
mod straddler;
mod transactions;
mod witnesses;

pub use contention::{ContentionReport, cross_shard_fraction, participant_count_sweep};
pub use execution::{
    a_failed_attempt_still_attests_work, a_payer_cannot_spend_one_balance_twice, abort_converges,
    abort_floor_settles_on_deadline, attested_load_reaches_the_beacon,
    cross_shard_credit_survives_a_later_local_credit, cross_shard_transfer, deploy_storm_rides_out,
    events_land_on_their_emitters_home_shard, failure_charges_its_payer, hot_recipient,
    insolvent_payer_engages_nothing, nullifier_race_admits_exactly_one,
    preview_reports_resource_changes, randomness_draw_agrees_across_shards,
    reads_the_committed_baseline, single_transfer, withdrawals_compose_over_one_vault,
    zipf_payments,
};
pub use faults::{
    beacon_lag_drops_skipped_epochs_reveal_chains, beacon_pool_partition_stalls_epoch_production,
    cross_shard_compound_drop_fetch_fallback, cross_shard_exec_cert_drop_fetch_fallback,
    cross_shard_header_fetch_fallback, cross_shard_provisions_drop_fetch_fallback,
    cross_shard_provisions_fetch_with_request_loss,
    cross_shard_provisions_recovers_after_transient_outage,
    cross_shard_transaction_da_fetch_fallback, gossip_drop_engages_fetch_fallback,
    halted_shard_recovers_by_committee_redraw, halted_shard_straddler_atomic,
    inter_shard_partition_strands_ticks_until_it_heals, isolated_validator_still_settles,
    minority_fragment_rejoins_after_partition, partition_halts_and_heals,
    partition_heals_at_exact_quorum,
};
pub use liveness::liveness_baseline;
pub use multi_vnode::multi_vnode_progress;
pub use reshape::{
    MAX_REPLAY_PROBES, grow_reaches_four_shard_topology, grow_reaches_two_shard_topology,
    merge_lifecycle, merge_seats_full_keeper_committee, split_boundary_refuses_a_replay,
    split_lifecycle,
};
pub use straddler::{
    isolate_ec_intake, merge_straddler_atomic, split_straddler_atomic,
    split_straddler_ec_partition_atomic, split_straddler_run,
    split_terminating_payer_releases_its_reservation, straddler_one_sided_count,
    surviving_sibling_split_seats_full_committees,
};
pub use support::{
    Budget, Cluster, FaultHandle, FaultableCluster, ScenarioConfig, epochs, grow_to, query, tx,
    vote_reshape_threshold, wait,
};
pub use transactions::livelock_resolves_promptly;
pub use witnesses::{
    delegation_folds_into_beacon_state, pool_capacity_caps_registrations,
    re_registration_of_a_live_validator_is_a_no_op, register_validator_pools_a_node,
    register_without_capacity_is_rejected, registered_validator_activates_onto_a_shard,
    stake_withdraw_drops_effective_stake, withdrawal_ejects_a_validator_that_a_deposit_reactivates,
};
