//! Workload generation for simulations.
//!
//! Provides various transaction generators for testing the network under load.

mod transfer;

pub use transfer::TransferWorkload;

use crate::accounts::AccountPool;
use hyperscale_types::RoutableTransaction;

/// Trait for generating transaction workloads.
pub trait WorkloadGenerator {
    /// Generate a batch of transactions.
    ///
    /// The generator may modify accounts (e.g., incrementing nonces).
    fn generate_batch(
        &mut self,
        accounts: &mut AccountPool,
        rng: &mut impl rand::Rng,
    ) -> Vec<RoutableTransaction>;

    /// Generate a single transaction.
    fn generate_one(
        &mut self,
        accounts: &mut AccountPool,
        rng: &mut impl rand::Rng,
    ) -> Option<RoutableTransaction>;
}

/// Mixed workload combining multiple transaction types.
pub struct MixedWorkload {
    transfer: TransferWorkload,
    batch_size: usize,
}

impl MixedWorkload {
    /// Create a new mixed workload.
    pub fn new(transfer: TransferWorkload, batch_size: usize) -> Self {
        Self {
            transfer,
            batch_size,
        }
    }

    /// Create a transfer-only workload.
    pub fn transfers_only(cross_shard_ratio: f64, batch_size: usize) -> Self {
        Self {
            transfer: TransferWorkload::new(cross_shard_ratio),
            batch_size,
        }
    }
}

impl WorkloadGenerator for MixedWorkload {
    fn generate_batch(
        &mut self,
        accounts: &mut AccountPool,
        rng: &mut impl rand::Rng,
    ) -> Vec<RoutableTransaction> {
        let mut transactions = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            if let Some(tx) = self.transfer.generate_one(accounts, rng) {
                transactions.push(tx);
            }
        }

        transactions
    }

    fn generate_one(
        &mut self,
        accounts: &mut AccountPool,
        rng: &mut impl rand::Rng,
    ) -> Option<RoutableTransaction> {
        self.transfer.generate_one(accounts, rng)
    }
}
