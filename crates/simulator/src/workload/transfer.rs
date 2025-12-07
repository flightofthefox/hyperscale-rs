//! XRD transfer workload generator.

use crate::accounts::{AccountPool, FundedAccount};
use crate::config::AccountDistribution;
use crate::workload::WorkloadGenerator;
use hyperscale_types::{sign_and_notarize, RoutableTransaction};
use radix_common::constants::XRD;
use radix_common::math::Decimal;
use radix_common::network::NetworkDefinition;
use radix_transactions::builder::ManifestBuilder;
use tracing::warn;

/// Generates XRD transfer transactions.
pub struct TransferWorkload {
    /// Ratio of cross-shard transactions (0.0 to 1.0).
    cross_shard_ratio: f64,

    /// Account selection distribution mode.
    distribution: AccountDistribution,

    /// Transfer amount per transaction.
    amount: Decimal,

    /// Network definition for transaction signing.
    network: NetworkDefinition,
}

impl TransferWorkload {
    /// Create a new transfer workload generator.
    pub fn new(cross_shard_ratio: f64) -> Self {
        Self {
            cross_shard_ratio: cross_shard_ratio.clamp(0.0, 1.0),
            distribution: AccountDistribution::default(),
            amount: Decimal::from(100u32),
            network: NetworkDefinition::simulator(),
        }
    }

    /// Set the account distribution mode.
    pub fn with_distribution(mut self, distribution: AccountDistribution) -> Self {
        self.distribution = distribution;
        self
    }

    /// Set the transfer amount.
    pub fn with_amount(mut self, amount: Decimal) -> Self {
        self.amount = amount;
        self
    }

    /// Generate a same-shard transfer.
    fn generate_same_shard(
        &mut self,
        accounts: &mut AccountPool,
        rng: &mut impl rand::Rng,
    ) -> Option<RoutableTransaction> {
        let (from, to) = accounts.same_shard_pair_with_distribution(rng, self.distribution)?;
        self.build_transfer(from, to)
    }

    /// Generate a cross-shard transfer.
    fn generate_cross_shard(
        &mut self,
        accounts: &mut AccountPool,
        rng: &mut impl rand::Rng,
    ) -> Option<RoutableTransaction> {
        let (mut from, to) = accounts.cross_shard_pair_with_distribution(rng, self.distribution)?;
        let tx = self.build_transfer(&mut from, &to)?;
        // Update the 'from' account nonce in the pool
        accounts.update_account(from);
        Some(tx)
    }

    /// Build a transfer transaction from one account to another.
    fn build_transfer(
        &self,
        from: &mut FundedAccount,
        to: &FundedAccount,
    ) -> Option<RoutableTransaction> {
        // Build manifest: withdraw from sender, deposit to receiver
        let manifest = ManifestBuilder::new()
            .lock_fee(from.address, Decimal::from(10u32))
            .withdraw_from_account(from.address, XRD, self.amount)
            .try_deposit_entire_worktop_or_abort(to.address, None)
            .build();

        // Get and increment nonce
        let nonce = from.next_nonce();

        // Sign and notarize
        let notarized =
            match sign_and_notarize(manifest, &self.network, nonce as u32, &from.keypair) {
                Ok(n) => n,
                Err(e) => {
                    warn!(error = ?e, "Failed to sign transaction");
                    return None;
                }
            };

        // Convert to RoutableTransaction
        let tx: RoutableTransaction = match notarized.try_into() {
            Ok(t) => t,
            Err(e) => {
                warn!(error = ?e, "Failed to convert to RoutableTransaction");
                return None;
            }
        };

        Some(tx)
    }
}

impl WorkloadGenerator for TransferWorkload {
    fn generate_batch(
        &mut self,
        accounts: &mut AccountPool,
        rng: &mut impl rand::Rng,
    ) -> Vec<RoutableTransaction> {
        // Just generate one by one
        let mut txs = Vec::new();
        for _ in 0..10 {
            if let Some(tx) = self.generate_one(accounts, rng) {
                txs.push(tx);
            }
        }
        txs
    }

    fn generate_one(
        &mut self,
        accounts: &mut AccountPool,
        rng: &mut impl rand::Rng,
    ) -> Option<RoutableTransaction> {
        let is_cross_shard =
            accounts.num_shards() >= 2 && rng.gen::<f64>() < self.cross_shard_ratio;

        if is_cross_shard {
            self.generate_cross_shard(accounts, rng)
        } else {
            self.generate_same_shard(accounts, rng)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_generate_same_shard_transfer() {
        let mut accounts = AccountPool::generate(2, 10).unwrap();
        let mut workload = TransferWorkload::new(0.0); // All same-shard
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let tx = workload.generate_one(&mut accounts, &mut rng);
        assert!(tx.is_some(), "Should generate a transaction");

        let tx = tx.unwrap();
        assert!(
            !tx.declared_writes.is_empty(),
            "Transaction should have declared writes"
        );
    }

    #[test]
    fn test_generate_cross_shard_transfer() {
        let mut accounts = AccountPool::generate(2, 10).unwrap();
        let mut workload = TransferWorkload::new(1.0); // All cross-shard
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let tx = workload.generate_one(&mut accounts, &mut rng);
        assert!(tx.is_some(), "Should generate a transaction");

        let tx = tx.unwrap();
        assert!(
            tx.is_cross_shard(2),
            "Transaction should be cross-shard for 2 shards"
        );
    }
}
