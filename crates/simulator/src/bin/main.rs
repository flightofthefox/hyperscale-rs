//! Hyperscale Deterministic Simulator CLI
//!
//! Run long-running workload simulations with configurable parameters.
//!
//! # Example
//!
//! ```bash
//! # Run a 60-second simulation with 2 shards
//! hyperscale-sim --shards 2 --duration 60
//!
//! # Run with more validators and cross-shard transactions
//! hyperscale-sim -s 4 -v 5 -d 120 --cross-shard-ratio 0.3
//! ```

use clap::Parser;
use hyperscale_simulator::{Simulator, SimulatorConfig, WorkloadConfig};
use std::time::Duration;
use tracing::info;
use tracing_subscriber::EnvFilter;

/// Hyperscale Deterministic Simulator
///
/// Runs long-running workload simulations in a fully deterministic environment.
/// Given the same seed, produces identical results every run.
#[derive(Parser, Debug)]
#[command(name = "hyperscale-sim")]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of shards
    #[arg(short = 's', long, default_value = "1")]
    shards: u32,

    /// Number of validators per shard
    #[arg(short = 'v', long, default_value = "4")]
    validators: u32,

    /// Simulation duration in seconds
    #[arg(short = 'd', long, default_value = "30")]
    duration: u64,

    /// Random seed for deterministic simulation
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Number of accounts per shard
    #[arg(short = 'a', long, default_value = "500")]
    accounts: usize,

    /// Transactions per second target
    #[arg(long, default_value = "100")]
    tps: usize,

    /// Ratio of cross-shard transactions (0.0-1.0)
    #[arg(long, default_value = "0.0")]
    cross_shard_ratio: f64,

    /// Ramp-down time in seconds (wait for in-flight to complete)
    #[arg(long, default_value = "30")]
    ramp_down: u64,

    /// Show livelock analysis at end
    #[arg(long)]
    analyze_livelocks: bool,

    /// Use no-contention account distribution (disjoint pairs)
    #[arg(long)]
    no_contention: bool,
}

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn,hyperscale_simulator=info")),
        )
        .init();

    let args = Args::parse();

    info!(
        shards = args.shards,
        validators = args.validators,
        duration_secs = args.duration,
        seed = args.seed,
        accounts = args.accounts,
        tps = args.tps,
        cross_shard_ratio = args.cross_shard_ratio,
        "Starting simulation"
    );

    // Calculate batch parameters to achieve target TPS
    // batch_interval * batches_per_sec = 1
    // batch_size * batches_per_sec = tps
    // => batch_size = tps * batch_interval
    let batch_interval_ms = 50u64; // 50ms between batches = 20 batches/sec
    let batches_per_sec = 1000 / batch_interval_ms;
    let batch_size = (args.tps as u64 / batches_per_sec).max(1) as usize;

    info!(
        batch_size,
        batch_interval_ms,
        effective_tps = batch_size * batches_per_sec as usize,
        "Workload parameters"
    );

    // Configure workload
    let mut workload = WorkloadConfig::transfers_only()
        .with_batch_size(batch_size)
        .with_batch_interval(Duration::from_millis(batch_interval_ms))
        .with_cross_shard_ratio(args.cross_shard_ratio);

    if args.no_contention {
        workload = workload.with_no_contention();
    }

    // Create simulator config
    let config = SimulatorConfig::new(args.shards, args.validators)
        .with_accounts_per_shard(args.accounts)
        .with_seed(args.seed)
        .with_workload(workload);

    // Create and initialize simulator
    let mut simulator = Simulator::new(config).expect("Failed to create simulator");
    simulator.initialize();

    // Run simulation
    // Total time = submission duration + ramp-down
    let total_duration = Duration::from_secs(args.duration + args.ramp_down);
    let report = simulator.run_for(total_duration);

    // Print report
    report.print_summary();

    // Print summary
    println!("\n=== Simulation Complete ===");
    println!("Submitted:  {}", report.total_submitted);
    println!("Finalized:  {}", report.total_finalized);
    println!("Rejected:   {}", report.total_rejected);
    println!(
        "Completion: {:.2}%",
        report.total_finalized as f64 / report.total_submitted.max(1) as f64 * 100.0
    );
    println!("TPS:        {:.2}", report.average_tps);

    // Livelock analysis
    if args.analyze_livelocks {
        let livelock_report = simulator.analyze_livelocks();
        livelock_report.print_summary();
    }

    // Account usage stats
    let usage = simulator.account_usage_stats();
    println!("\n=== Account Usage ===");
    println!("Skew ratio: {:.2}", usage.skew_ratio());
}
