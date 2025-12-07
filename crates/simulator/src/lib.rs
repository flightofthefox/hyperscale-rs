//! Hyperscale Simulator
//!
//! A long-running workload simulator built on top of the simulation framework.
//! Provides tools for stress testing, performance measurement, and system validation.
//!
//! # Architecture
//!
//! The simulator builds on `hyperscale-simulation` to provide:
//!
//! - **Account Management**: Pre-funded accounts with shard-targeted generation
//! - **Workload Generation**: Configurable transaction generators (transfers, etc.)
//! - **Metrics Collection**: TPS, latency percentiles, lock contention tracking
//! - **Configuration**: Flexible setup for various test scenarios
//!
//! # Example
//!
//! ```ignore
//! use hyperscale_simulator::{Simulator, SimulatorConfig, WorkloadConfig};
//! use std::time::Duration;
//!
//! // Create a simulator with 2 shards, 3 validators each
//! let config = SimulatorConfig::new(2, 3)
//!     .with_accounts_per_shard(100)
//!     .with_workload(WorkloadConfig::transfers_only());
//!
//! let mut simulator = Simulator::new(config, 12345);
//! let report = simulator.run_for(Duration::from_secs(60));
//!
//! println!("TPS: {:.2}", report.average_tps());
//! println!("P99 latency: {:?}", report.p99_latency());
//! ```

pub mod accounts;
pub mod config;
pub mod livelock;
pub mod metrics;
pub mod runner;
pub mod workload;

pub use accounts::{AccountPool, FundedAccount};
pub use config::{AccountDistribution, SimulatorConfig, WorkloadConfig};
pub use livelock::{LivelockAnalyzer, LivelockReport, StuckTransaction};
pub use metrics::{MetricsCollector, SimulationReport};
pub use runner::Simulator;
pub use workload::{TransferWorkload, WorkloadGenerator};
