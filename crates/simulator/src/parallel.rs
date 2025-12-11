//! Parallel simulation orchestrator.
//!
//! Wraps `ParallelSimulator` with workload generation and metrics collection
//! for multi-core performance testing.

use crate::config::SimulatorConfig;
use hyperscale_parallel::{ParallelConfig, ParallelSimulator, SimulationReport};
use hyperscale_spammer::{
    AccountPool, AccountPoolError, SelectionMode, TransferWorkload, WorkloadGenerator,
};
use hyperscale_types::{shard_for_node, RoutableTransaction, ShardGroupId};
use radix_common::network::NetworkDefinition;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::info;

/// Errors from parallel orchestration.
#[derive(Debug, Error)]
pub enum ParallelOrchestratorError {
    #[error("Account pool error: {0}")]
    AccountPool(#[from] AccountPoolError),
}

/// Configuration for parallel orchestrator.
#[derive(Debug, Clone)]
pub struct ParallelOrchestratorConfig {
    /// Number of shards.
    pub num_shards: usize,
    /// Validators per shard.
    pub validators_per_shard: usize,
    /// Accounts per shard for workload generation.
    pub accounts_per_shard: usize,
    /// Target transactions per second.
    pub target_tps: u64,
    /// Duration of transaction submission.
    pub submission_duration: Duration,
    /// Drain duration after submission ends.
    pub drain_duration: Duration,
    /// Cross-shard transaction ratio (0.0 to 1.0).
    pub cross_shard_ratio: f64,
    /// Random seed.
    pub seed: u64,
    /// Account selection mode for workload generation.
    pub selection_mode: SelectionMode,
}

impl Default for ParallelOrchestratorConfig {
    fn default() -> Self {
        Self {
            num_shards: 2,
            validators_per_shard: 4,
            accounts_per_shard: 100,
            target_tps: 100,
            submission_duration: Duration::from_secs(10),
            drain_duration: Duration::from_secs(5),
            cross_shard_ratio: 0.1,
            seed: 42,
            selection_mode: SelectionMode::Random,
        }
    }
}

impl ParallelOrchestratorConfig {
    /// Create a new configuration.
    pub fn new(num_shards: usize, validators_per_shard: usize) -> Self {
        Self {
            num_shards,
            validators_per_shard,
            ..Default::default()
        }
    }

    /// Set the target TPS.
    pub fn with_target_tps(mut self, tps: u64) -> Self {
        self.target_tps = tps;
        self
    }

    /// Set the submission duration.
    pub fn with_submission_duration(mut self, duration: Duration) -> Self {
        self.submission_duration = duration;
        self
    }

    /// Set the drain duration.
    pub fn with_drain_duration(mut self, duration: Duration) -> Self {
        self.drain_duration = duration;
        self
    }

    /// Set the cross-shard ratio.
    pub fn with_cross_shard_ratio(mut self, ratio: f64) -> Self {
        self.cross_shard_ratio = ratio;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set accounts per shard.
    pub fn with_accounts_per_shard(mut self, accounts: usize) -> Self {
        self.accounts_per_shard = accounts;
        self
    }

    /// Set account selection mode.
    pub fn with_selection_mode(mut self, mode: SelectionMode) -> Self {
        self.selection_mode = mode;
        self
    }

    /// Use no-contention mode for zero account conflicts.
    pub fn with_no_contention(self) -> Self {
        self.with_selection_mode(SelectionMode::NoContention)
    }
}

impl From<&SimulatorConfig> for ParallelOrchestratorConfig {
    fn from(config: &SimulatorConfig) -> Self {
        Self {
            num_shards: config.num_shards as usize,
            validators_per_shard: config.validators_per_shard as usize,
            accounts_per_shard: config.accounts_per_shard,
            target_tps: 100, // Default
            submission_duration: Duration::from_secs(10),
            drain_duration: Duration::from_secs(5),
            cross_shard_ratio: config.workload.cross_shard_ratio,
            seed: config.seed,
            selection_mode: config.workload.selection_mode,
        }
    }
}

/// Orchestrates parallel simulation with workload generation.
///
/// Uses `ParallelSimulator` under the hood for multi-core execution via rayon,
/// combined with `AccountPool` and `TransferWorkload` for transaction
/// generation.
pub struct ParallelOrchestrator {
    /// The parallel simulator.
    simulator: ParallelSimulator,
    /// Account pool for transaction generation.
    accounts: AccountPool,
    /// Workload generator.
    workload: TransferWorkload,
    /// Configuration.
    config: ParallelOrchestratorConfig,
    /// RNG for workload generation.
    rng: ChaCha8Rng,
}

impl ParallelOrchestrator {
    /// Create a new parallel orchestrator.
    pub fn new(config: ParallelOrchestratorConfig) -> Result<Self, ParallelOrchestratorError> {
        // Create parallel simulator config
        let parallel_config = ParallelConfig::new(config.num_shards, config.validators_per_shard)
            .with_seed(config.seed)
            .with_drain_duration(config.drain_duration);

        let simulator = ParallelSimulator::new(parallel_config);

        // Generate accounts
        let accounts = AccountPool::generate(config.num_shards as u64, config.accounts_per_shard)?;

        // Create workload generator
        let workload = TransferWorkload::new(NetworkDefinition::simulator())
            .with_cross_shard_ratio(config.cross_shard_ratio)
            .with_selection_mode(config.selection_mode);

        let rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(1));

        info!(
            num_shards = config.num_shards,
            validators_per_shard = config.validators_per_shard,
            accounts_per_shard = config.accounts_per_shard,
            target_tps = config.target_tps,
            "ParallelOrchestrator created"
        );

        Ok(Self {
            simulator,
            accounts,
            workload,
            config,
            rng,
        })
    }

    /// Run the full parallel simulation with simulated time.
    ///
    /// This will:
    /// 1. Initialize the simulator (creates all nodes, runs genesis)
    /// 2. Submit all transactions
    /// 3. Step the simulation until all transactions complete
    /// 4. Return the final report
    ///
    /// Note: Uses target_tps and submission_duration to calculate total transactions.
    /// The simulation runs synchronously using rayon for parallelism.
    pub async fn run(mut self) -> Result<SimulationReport, ParallelOrchestratorError> {
        // Initialize the simulator
        self.simulator.initialize();

        // Calculate total transactions and submission rate
        let total_transactions = (self.config.target_tps as f64
            * self.config.submission_duration.as_secs_f64())
            as usize;

        // Calculate how many transactions to submit per millisecond of simulated time
        let submission_duration_ms = self.config.submission_duration.as_millis() as usize;
        let txs_per_ms = if submission_duration_ms > 0 {
            (total_transactions as f64 / submission_duration_ms as f64).ceil() as usize
        } else {
            total_transactions
        };

        info!(
            target_tps = self.config.target_tps,
            submission_duration_ms,
            total_transactions,
            txs_per_ms,
            "Starting simulation"
        );

        let start_time = Instant::now();

        // Generate all transactions upfront
        let mut transactions: VecDeque<_> = self
            .workload
            .generate_batch(&self.accounts, total_transactions, &mut self.rng)
            .into();

        // Run simulation: submit transactions over time while stepping
        let max_steps = submission_duration_ms + 10_000; // submission + drain
        let mut steps = 0;
        let mut stall_count = 0;
        let mut last_in_flight = 0;
        let mut last_logged_completed = 0u64;
        let progress_interval = (total_transactions / 10).max(100) as u64;

        while steps < max_steps {
            // Submit transactions for this millisecond of simulated time
            if !transactions.is_empty() {
                let to_submit = txs_per_ms.min(transactions.len());
                for _ in 0..to_submit {
                    if let Some(tx) = transactions.pop_front() {
                        self.simulator.submit_transaction(tx);
                    }
                }
            }

            // Step the simulation
            let events = self.simulator.step();
            steps += 1;

            let (submitted, completed, rejected, current_in_flight) = self.simulator.metrics();

            // Check for completion (all submitted and none in flight)
            if transactions.is_empty() && current_in_flight == 0 {
                break;
            }

            // Check for stall (only after all transactions submitted)
            if transactions.is_empty() {
                if current_in_flight == last_in_flight && events == 0 {
                    stall_count += 1;
                    if stall_count >= 1000 {
                        tracing::warn!(
                            in_flight = current_in_flight,
                            steps,
                            "Simulation stalled - no progress"
                        );
                        break;
                    }
                } else {
                    stall_count = 0;
                }
            }
            last_in_flight = current_in_flight;

            // Log progress when completions cross thresholds
            if completed / progress_interval > last_logged_completed / progress_interval {
                let elapsed = start_time.elapsed();
                let current_tps = if elapsed.as_secs_f64() > 0.0 {
                    completed as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                info!(
                    submitted,
                    completed,
                    rejected,
                    in_flight = current_in_flight,
                    simulated_time_ms = self.simulator.simulated_time().as_millis(),
                    tps = format!("{:.1}", current_tps),
                    "Simulation progress"
                );
                last_logged_completed = completed;
            }
        }

        let wall_clock_duration = start_time.elapsed();
        let report = self.simulator.finalize(wall_clock_duration);

        info!(
            completed = report.completed,
            rejected = report.rejected,
            tps = format!("{:.2}", report.tps),
            wall_clock_ms = wall_clock_duration.as_millis(),
            "Simulation complete"
        );

        Ok(report)
    }

    /// Get access to the underlying simulator (for advanced control).
    pub fn simulator(&self) -> &ParallelSimulator {
        &self.simulator
    }

    /// Get mutable access to the underlying simulator.
    pub fn simulator_mut(&mut self) -> &mut ParallelSimulator {
        &mut self.simulator
    }

    /// Determine the target shard for a transaction based on its first declared write.
    fn _get_target_shard(&self, tx: &RoutableTransaction) -> ShardGroupId {
        tx.declared_writes
            .first()
            .map(|node_id| shard_for_node(node_id, self.config.num_shards as u64))
            .unwrap_or(ShardGroupId(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = ParallelOrchestratorConfig::new(2, 4)
            .with_target_tps(500)
            .with_submission_duration(Duration::from_secs(30))
            .with_cross_shard_ratio(0.2);

        assert_eq!(config.num_shards, 2);
        assert_eq!(config.validators_per_shard, 4);
        assert_eq!(config.target_tps, 500);
        assert_eq!(config.submission_duration, Duration::from_secs(30));
        assert_eq!(config.cross_shard_ratio, 0.2);
    }
}
