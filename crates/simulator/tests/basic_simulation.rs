//! Basic simulation test.
//!
//! Verifies that the simulator can run a simple workload and collect metrics.

use hyperscale_simulator::{Simulator, SimulatorConfig, WorkloadConfig};
use std::time::Duration;

/// Run a simple single-shard simulation.
#[test]
fn test_single_shard_simulation() {
    // Initialize tracing for debugging
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    // Configure simulation: 1 shard, 4 validators
    // With NoContention mode, we need (accounts / 2) >= max_concurrent_transactions
    // to avoid wrap-around contention. With 530 txs over 5s and ~1s finalization time,
    // we could have ~100+ concurrent txs. Use 500 accounts (250 pairs) for headroom.
    let config = SimulatorConfig::new(1, 4)
        .with_accounts_per_shard(500)
        .with_seed(42)
        .with_workload(
            WorkloadConfig::transfers_only()
                .with_batch_size(5)
                // Use smaller batch interval than proposal_interval (100ms)
                // to ensure transactions are available when proposals happen
                .with_batch_interval(Duration::from_millis(50))
                .with_no_contention(), // Zero lock contention
        );

    // Create and initialize simulator
    let mut simulator = Simulator::new(config).expect("Failed to create simulator");
    simulator.initialize();

    // Run for 5 seconds of simulated time
    let report = simulator.run_for(Duration::from_secs(5));

    // Verify we processed some transactions
    println!("\n=== Test Results ===");
    println!("Submitted: {}", report.total_submitted);
    println!("Finalized: {}", report.total_finalized);
    println!("Rejected:  {}", report.total_rejected);
    println!("TPS:       {:.2}", report.average_tps);

    assert!(
        report.total_submitted > 0,
        "Should have submitted transactions"
    );

    // With a well-configured simulation, most transactions should complete
    // (may not be 100% if simulation time is too short for ramp-down)
    let completion_rate = report.total_finalized as f64 / report.total_submitted as f64;
    println!("Completion rate: {:.2}%", completion_rate * 100.0);

    // Account usage should be relatively even with NoContention mode
    let usage_stats = simulator.account_usage_stats();
    println!("Account skew ratio: {:.2}", usage_stats.skew_ratio());
}

/// Run a multi-shard simulation with cross-shard transactions.
#[test]
fn test_multi_shard_simulation() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    // Configure simulation: 2 shards, 3 validators each
    let config = SimulatorConfig::new(2, 3)
        .with_accounts_per_shard(20)
        .with_seed(12345)
        .with_workload(
            WorkloadConfig::transfers_only()
                .with_cross_shard_ratio(0.5) // 50% cross-shard
                .with_batch_size(3)
                .with_batch_interval(Duration::from_millis(500))
                .with_round_robin(), // Minimize contention
        );

    let mut simulator = Simulator::new(config).expect("Failed to create simulator");
    simulator.initialize();

    // Run for 5 seconds
    let report = simulator.run_for(Duration::from_secs(5));

    println!("\n=== Multi-Shard Test Results ===");
    println!("Submitted: {}", report.total_submitted);
    println!("Finalized: {}", report.total_finalized);
    println!("TPS:       {:.2}", report.average_tps);
    println!("P50:       {:?}", report.p50_latency());
    println!("P99:       {:?}", report.p99_latency());

    assert!(
        report.total_submitted > 0,
        "Should have submitted transactions"
    );
}

/// Test that determinism works - same seed produces same results.
#[test]
fn test_determinism() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("warn")
        .with_test_writer()
        .try_init();

    let config = SimulatorConfig::new(1, 4)
        .with_accounts_per_shard(20)
        .with_seed(99999)
        .with_workload(
            WorkloadConfig::transfers_only()
                .with_batch_size(3)
                .with_batch_interval(Duration::from_millis(200)),
        );

    // Run simulation twice with same config
    let mut sim1 = Simulator::new(config.clone()).unwrap();
    sim1.initialize();
    let report1 = sim1.run_for(Duration::from_secs(3));

    let mut sim2 = Simulator::new(config).unwrap();
    sim2.initialize();
    let report2 = sim2.run_for(Duration::from_secs(3));

    // Results should be identical
    assert_eq!(
        report1.total_submitted, report2.total_submitted,
        "Submissions should be deterministic"
    );
    assert_eq!(
        report1.total_finalized, report2.total_finalized,
        "Finalizations should be deterministic"
    );
    assert_eq!(
        report1.total_rejected, report2.total_rejected,
        "Rejections should be deterministic"
    );

    println!("Determinism verified: both runs produced identical results");
    println!("  Submitted: {}", report1.total_submitted);
    println!("  Finalized: {}", report1.total_finalized);
}
