//! XRD transfer workload generator.

use crate::accounts::{AccountPool, FundedAccount, SelectionMode};
use crate::workloads::WorkloadGenerator;
use hyperscale_types::{sign_and_notarize, RoutableTransaction, ShardGroupId};
use radix_common::constants::XRD;
use radix_common::math::Decimal;
use radix_common::network::NetworkDefinition;
use radix_transactions::builder::ManifestBuilder;
use rand::{Rng, RngCore};
use tracing::warn;

/// Generates XRD transfer transactions.
pub struct TransferWorkload {
    /// Ratio of cross-shard transactions (0.0 to 1.0).
    cross_shard_ratio: f64,

    /// Account selection mode.
    selection_mode: SelectionMode,

    /// Transfer amount per transaction.
    amount: Decimal,

    /// Network definition for transaction signing.
    network: NetworkDefinition,
}

impl TransferWorkload {
    /// Create a new transfer workload generator.
    pub fn new(network: NetworkDefinition) -> Self {
        Self {
            cross_shard_ratio: 0.3,
            selection_mode: SelectionMode::default(),
            amount: Decimal::from(100u32),
            network,
        }
    }

    /// Set the cross-shard transaction ratio (0.0 to 1.0).
    pub fn with_cross_shard_ratio(mut self, ratio: f64) -> Self {
        self.cross_shard_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Set the account selection mode.
    pub fn with_selection_mode(mut self, mode: SelectionMode) -> Self {
        self.selection_mode = mode;
        self
    }

    /// Set the transfer amount.
    pub fn with_amount(mut self, amount: Decimal) -> Self {
        self.amount = amount;
        self
    }

    /// Generate a same-shard transfer.
    fn generate_same_shard_inner<R: Rng + ?Sized>(
        &self,
        accounts: &AccountPool,
        rng: &mut R,
    ) -> Option<RoutableTransaction> {
        let shard = ShardGroupId(rng.gen_range(0..accounts.num_shards()));
        let shard_accounts = accounts.accounts_for_shard(shard)?;

        if shard_accounts.len() < 2 {
            return None;
        }

        let (idx1, idx2) = self.select_pair_indices(shard_accounts.len(), rng);
        let from = &shard_accounts[idx1];
        let to = &shard_accounts[idx2];

        self.build_transfer(from, to)
    }

    /// Generate a cross-shard transfer.
    fn generate_cross_shard_inner<R: Rng + ?Sized>(
        &self,
        accounts: &AccountPool,
        rng: &mut R,
    ) -> Option<RoutableTransaction> {
        if accounts.num_shards() < 2 {
            return None;
        }

        let shard1 = ShardGroupId(rng.gen_range(0..accounts.num_shards()));
        let mut shard2 = ShardGroupId(rng.gen_range(0..accounts.num_shards()));
        while shard2 == shard1 {
            shard2 = ShardGroupId(rng.gen_range(0..accounts.num_shards()));
        }

        let accounts1 = accounts.accounts_for_shard(shard1)?;
        let accounts2 = accounts.accounts_for_shard(shard2)?;

        if accounts1.is_empty() || accounts2.is_empty() {
            return None;
        }

        let idx1 = self.select_single_index(accounts1.len(), rng);
        let idx2 = self.select_single_index(accounts2.len(), rng);

        let from = &accounts1[idx1];
        let to = &accounts2[idx2];

        self.build_transfer(from, to)
    }

    /// Select a pair of distinct indices.
    fn select_pair_indices<R: Rng + ?Sized>(
        &self,
        num_accounts: usize,
        rng: &mut R,
    ) -> (usize, usize) {
        match self.selection_mode {
            SelectionMode::Random | SelectionMode::RoundRobin | SelectionMode::NoContention => {
                // For RoundRobin and NoContention, the AccountPool handles the
                // stateful selection via atomics. Here we just use random as a fallback
                // for the rare case where workload does its own selection.
                let idx1 = rng.gen_range(0..num_accounts);
                let mut idx2 = rng.gen_range(0..num_accounts);
                while idx2 == idx1 {
                    idx2 = rng.gen_range(0..num_accounts);
                }
                (idx1, idx2)
            }
            SelectionMode::Zipf { exponent } => {
                let idx1 = self.zipf_index(num_accounts, exponent, rng);
                let mut idx2 = self.zipf_index(num_accounts, exponent, rng);
                while idx2 == idx1 {
                    idx2 = self.zipf_index(num_accounts, exponent, rng);
                }
                (idx1, idx2)
            }
        }
    }

    /// Select a single index.
    fn select_single_index<R: Rng + ?Sized>(&self, num_accounts: usize, rng: &mut R) -> usize {
        match self.selection_mode {
            SelectionMode::Random | SelectionMode::RoundRobin | SelectionMode::NoContention => {
                rng.gen_range(0..num_accounts)
            }
            SelectionMode::Zipf { exponent } => self.zipf_index(num_accounts, exponent, rng),
        }
    }

    /// Generate a Zipf-distributed index.
    fn zipf_index<R: Rng + ?Sized>(&self, n: usize, exponent: f64, rng: &mut R) -> usize {
        let exp = exponent.max(1.0);
        let u: f64 = rng.gen();
        let idx = ((n as f64).powf(1.0 - u)).powf(1.0 / exp) as usize;
        idx.min(n - 1)
    }

    /// Build a transfer transaction from one account to another.
    fn build_transfer(
        &self,
        from: &FundedAccount,
        to: &FundedAccount,
    ) -> Option<RoutableTransaction> {
        // Build manifest: withdraw from sender, deposit to receiver
        let manifest = ManifestBuilder::new()
            .lock_fee(from.address, Decimal::from(10u32))
            .withdraw_from_account(from.address, XRD, self.amount)
            .try_deposit_entire_worktop_or_abort(to.address, None)
            .build();

        // Get and increment nonce atomically
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

    /// Generate one transaction (internal helper for trait impl).
    fn generate_one_inner<R: Rng + ?Sized>(
        &self,
        accounts: &AccountPool,
        rng: &mut R,
    ) -> Option<RoutableTransaction> {
        let is_cross_shard =
            accounts.num_shards() >= 2 && rng.gen::<f64>() < self.cross_shard_ratio;

        if is_cross_shard {
            self.generate_cross_shard_inner(accounts, rng)
        } else {
            self.generate_same_shard_inner(accounts, rng)
        }
    }
}

impl WorkloadGenerator for TransferWorkload {
    fn generate_one(
        &self,
        accounts: &AccountPool,
        rng: &mut dyn RngCore,
    ) -> Option<RoutableTransaction> {
        // Wrap the dyn RngCore to get Rng trait
        self.generate_one_inner(accounts, rng)
    }

    fn generate_batch(
        &self,
        accounts: &AccountPool,
        count: usize,
        rng: &mut dyn RngCore,
    ) -> Vec<RoutableTransaction> {
        (0..count)
            .filter_map(|_| self.generate_one_inner(accounts, rng))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_generate_same_shard_transfer() {
        let accounts = AccountPool::generate(2, 10).unwrap();
        let workload =
            TransferWorkload::new(NetworkDefinition::simulator()).with_cross_shard_ratio(0.0); // All same-shard
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let tx = workload.generate_one(&accounts, &mut rng);
        assert!(tx.is_some(), "Should generate a transaction");

        let tx = tx.unwrap();
        assert!(
            !tx.declared_writes.is_empty(),
            "Transaction should have declared writes"
        );
    }

    #[test]
    fn test_generate_cross_shard_transfer() {
        let accounts = AccountPool::generate(2, 10).unwrap();
        let workload =
            TransferWorkload::new(NetworkDefinition::simulator()).with_cross_shard_ratio(1.0); // All cross-shard
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let tx = workload.generate_one(&accounts, &mut rng);
        assert!(tx.is_some(), "Should generate a transaction");

        let tx = tx.unwrap();
        assert!(
            tx.is_cross_shard(2),
            "Transaction should be cross-shard for 2 shards"
        );
    }
}
