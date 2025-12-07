//! Configuration types for the simulator.

use hyperscale_simulation::NetworkConfig;
use radix_common::math::Decimal;
use std::time::Duration;

/// Configuration for a simulation run.
#[derive(Clone, Debug)]
pub struct SimulatorConfig {
    /// Number of shards in the network.
    pub num_shards: u32,

    /// Number of validators per shard.
    pub validators_per_shard: u32,

    /// Number of accounts to create per shard.
    pub accounts_per_shard: usize,

    /// Initial XRD balance for each account.
    pub initial_balance: Decimal,

    /// Workload configuration.
    pub workload: WorkloadConfig,

    /// Random seed for deterministic simulation.
    pub seed: u64,
}

impl SimulatorConfig {
    /// Create a new simulator configuration.
    pub fn new(num_shards: u32, validators_per_shard: u32) -> Self {
        Self {
            num_shards,
            validators_per_shard,
            accounts_per_shard: 50,
            initial_balance: Decimal::from(10_000u32),
            workload: WorkloadConfig::default(),
            seed: 12345,
        }
    }

    /// Set the number of accounts per shard.
    pub fn with_accounts_per_shard(mut self, accounts: usize) -> Self {
        self.accounts_per_shard = accounts;
        self
    }

    /// Set the initial balance for accounts.
    pub fn with_initial_balance(mut self, balance: Decimal) -> Self {
        self.initial_balance = balance;
        self
    }

    /// Set the workload configuration.
    pub fn with_workload(mut self, workload: WorkloadConfig) -> Self {
        self.workload = workload;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Total number of validators across all shards.
    pub fn total_validators(&self) -> u32 {
        self.num_shards * self.validators_per_shard
    }

    /// Total number of accounts across all shards.
    pub fn total_accounts(&self) -> usize {
        self.num_shards as usize * self.accounts_per_shard
    }

    /// Convert to a NetworkConfig for the underlying simulation.
    pub fn to_network_config(&self) -> NetworkConfig {
        NetworkConfig {
            num_shards: self.num_shards,
            validators_per_shard: self.validators_per_shard,
            ..Default::default()
        }
    }
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self::new(2, 4)
    }
}

/// Account selection distribution mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AccountDistribution {
    /// Pure random selection - can cause high contention.
    #[default]
    Random,

    /// Round-robin selection - cycles through accounts to minimize contention.
    RoundRobin,

    /// Zipf distribution - realistic "popular accounts" pattern.
    /// The exponent parameter controls skewness (higher = more skewed).
    Zipf {
        /// Zipf exponent (1 = mild skew, 2+ = heavy skew toward hotspots).
        exponent: u32,
    },

    /// Partitioned - each batch uses disjoint account sets (zero contention).
    NoContention,
}

/// Workload configuration.
#[derive(Clone, Debug)]
pub struct WorkloadConfig {
    /// Ratio of transfer transactions (vs other types in future).
    /// 1.0 = all transfers.
    pub transfer_ratio: f64,

    /// Ratio of cross-shard transactions (vs same-shard).
    /// 1.0 = all cross-shard, 0.0 = all same-shard.
    pub cross_shard_ratio: f64,

    /// Number of transactions to generate per batch.
    pub batch_size: usize,

    /// Time between transaction batches (simulated time).
    pub batch_interval: Duration,

    /// Account selection distribution mode.
    pub account_distribution: AccountDistribution,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            transfer_ratio: 1.0,
            cross_shard_ratio: 0.3,
            batch_size: 10,
            batch_interval: Duration::from_millis(500),
            account_distribution: AccountDistribution::default(),
        }
    }
}

impl WorkloadConfig {
    /// Create a transfer-only workload.
    pub fn transfers_only() -> Self {
        Self {
            transfer_ratio: 1.0,
            ..Default::default()
        }
    }

    /// Set the cross-shard ratio.
    pub fn with_cross_shard_ratio(mut self, ratio: f64) -> Self {
        self.cross_shard_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the batch interval.
    pub fn with_batch_interval(mut self, interval: Duration) -> Self {
        self.batch_interval = interval;
        self
    }

    /// Set the account distribution mode.
    pub fn with_account_distribution(mut self, distribution: AccountDistribution) -> Self {
        self.account_distribution = distribution;
        self
    }

    /// Use round-robin account selection (minimal contention).
    pub fn with_round_robin(self) -> Self {
        self.with_account_distribution(AccountDistribution::RoundRobin)
    }

    /// Use Zipf distribution for realistic hotspot patterns.
    pub fn with_zipf(self, exponent: u32) -> Self {
        self.with_account_distribution(AccountDistribution::Zipf { exponent })
    }

    /// Use partitioned accounts for zero contention testing.
    pub fn with_no_contention(self) -> Self {
        self.with_account_distribution(AccountDistribution::NoContention)
    }
}
