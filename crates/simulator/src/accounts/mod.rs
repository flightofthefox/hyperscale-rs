//! Account management for simulations.
//!
//! Provides an `AccountPool` that manages funded accounts distributed across shards.
//! Accounts are funded at genesis time for zero consensus overhead.

use crate::config::AccountDistribution;
use hyperscale_types::{shard_for_node, KeyPair, KeyType, NodeId, ShardGroupId};
use radix_common::crypto::Ed25519PublicKey;
use radix_common::math::Decimal;
use radix_common::types::ComponentAddress;
use std::collections::HashMap;
use tracing::info;

/// A funded account that can sign transactions.
#[derive(Clone)]
pub struct FundedAccount {
    /// The keypair for signing transactions.
    pub keypair: KeyPair,

    /// The Radix component address for this account.
    pub address: ComponentAddress,

    /// The shard this account belongs to.
    pub shard: ShardGroupId,

    /// Nonce counter for transaction signing.
    nonce: u64,
}

impl FundedAccount {
    /// Create a new funded account from a seed.
    pub fn from_seed(seed: u64, num_shards: u64) -> Self {
        // Create varied seed bytes from the u64 seed
        let mut seed_bytes = [0u8; 32];
        let seed_le = seed.to_le_bytes();
        for (i, chunk) in seed_bytes.chunks_mut(8).enumerate() {
            // XOR with index to ensure different chunks even for small seeds
            let varied = seed.wrapping_add(i as u64);
            chunk.copy_from_slice(&varied.to_le_bytes());
        }
        // Also incorporate the original seed directly for uniqueness
        seed_bytes[0..8].copy_from_slice(&seed_le);
        let keypair = KeyPair::from_seed(KeyType::Ed25519, &seed_bytes);
        let address = Self::address_from_keypair(&keypair);
        let shard = Self::shard_for_address(&address, num_shards);

        Self {
            keypair,
            address,
            shard,
            nonce: 0,
        }
    }

    /// Get the next nonce and increment.
    pub fn next_nonce(&mut self) -> u64 {
        let nonce = self.nonce;
        self.nonce += 1;
        nonce
    }

    /// Get current nonce without incrementing.
    pub fn current_nonce(&self) -> u64 {
        self.nonce
    }

    /// Derive account address from keypair.
    fn address_from_keypair(keypair: &KeyPair) -> ComponentAddress {
        match keypair.public_key() {
            hyperscale_types::PublicKey::Ed25519(bytes) => {
                let radix_pk = Ed25519PublicKey(bytes);
                ComponentAddress::preallocated_account_from_public_key(&radix_pk)
            }
            _ => panic!("Only Ed25519 keypairs are supported for accounts"),
        }
    }

    /// Determine which shard an address belongs to.
    fn shard_for_address(address: &ComponentAddress, num_shards: u64) -> ShardGroupId {
        let node_id = address.into_node_id();
        let det_node_id = NodeId(node_id.0[..30].try_into().unwrap());
        shard_for_node(&det_node_id, num_shards)
    }
}

/// Pool of funded accounts distributed across shards.
pub struct AccountPool {
    /// Accounts grouped by shard.
    by_shard: HashMap<ShardGroupId, Vec<FundedAccount>>,

    /// Number of shards.
    num_shards: u64,

    /// Round-robin counters per shard (for RoundRobin mode).
    round_robin_counters: HashMap<ShardGroupId, usize>,

    /// Global transaction counter for NoContention mode.
    /// Each transaction reserves 2 account slots to ensure zero conflicts.
    no_contention_counter: usize,

    /// Usage tracking: total selections per account index per shard.
    usage_counts: HashMap<ShardGroupId, Vec<u64>>,
}

impl AccountPool {
    /// Create an empty account pool.
    pub fn new(num_shards: u64) -> Self {
        let mut by_shard = HashMap::new();
        let mut round_robin_counters = HashMap::new();
        let mut usage_counts = HashMap::new();

        for shard in 0..num_shards {
            let shard_id = ShardGroupId(shard);
            by_shard.insert(shard_id, Vec::new());
            round_robin_counters.insert(shard_id, 0);
            usage_counts.insert(shard_id, Vec::new());
        }

        Self {
            by_shard,
            num_shards,
            round_robin_counters,
            no_contention_counter: 0,
            usage_counts,
        }
    }

    /// Generate accounts targeting specific shards.
    ///
    /// This searches for keypair seeds whose derived accounts land on each shard.
    /// The accounts are NOT funded yet - use `genesis_balances()` to get the
    /// balances to pass to genesis configuration.
    pub fn generate(num_shards: u64, accounts_per_shard: usize) -> Result<Self, AccountPoolError> {
        info!(num_shards, accounts_per_shard, "Generating account pool");

        let mut pool = Self::new(num_shards);

        // Find accounts for each shard
        let mut seed = 100u64; // Start after reserved seeds
        let mut found_per_shard = vec![0usize; num_shards as usize];
        let max_iterations = accounts_per_shard * num_shards as usize * 10;
        let mut iterations = 0;

        while found_per_shard
            .iter()
            .any(|&count| count < accounts_per_shard)
        {
            let account = FundedAccount::from_seed(seed, num_shards);
            let shard_idx = account.shard.0 as usize;

            if found_per_shard[shard_idx] < accounts_per_shard {
                pool.by_shard.get_mut(&account.shard).unwrap().push(account);
                found_per_shard[shard_idx] += 1;
            }

            seed = seed.wrapping_add(1);
            iterations += 1;
            if iterations > max_iterations {
                return Err(AccountPoolError::GenerationFailed {
                    shards: num_shards,
                    accounts_per_shard,
                });
            }
        }

        // Initialize usage counts for each shard
        for shard in 0..num_shards {
            let shard_id = ShardGroupId(shard);
            let count = pool.by_shard.get(&shard_id).map(|v| v.len()).unwrap_or(0);
            let counters: Vec<u64> = vec![0; count];
            pool.usage_counts.insert(shard_id, counters);
        }

        info!(
            total_accounts = pool.total_accounts(),
            "Generated accounts for all shards"
        );

        Ok(pool)
    }

    /// Get the XRD balances for a specific shard to configure in genesis.
    ///
    /// Returns a list of (address, balance) pairs suitable for passing to
    /// genesis configuration. Only includes accounts that belong to the
    /// specified shard.
    pub fn genesis_balances_for_shard(
        &self,
        shard: ShardGroupId,
        balance: Decimal,
    ) -> Vec<(ComponentAddress, Decimal)> {
        self.by_shard
            .get(&shard)
            .map(|accounts| {
                accounts
                    .iter()
                    .map(|account| (account.address, balance))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get a mutable reference to a random account on a specific shard.
    pub fn random_on_shard(
        &mut self,
        shard: ShardGroupId,
        rng: &mut impl rand::Rng,
    ) -> Option<&mut FundedAccount> {
        let accounts = self.by_shard.get_mut(&shard)?;
        if accounts.is_empty() {
            return None;
        }
        let idx = rng.gen_range(0..accounts.len());
        Some(&mut accounts[idx])
    }

    /// Get a pair of accounts on the same shard.
    /// Uses pure random selection - can cause high contention.
    pub fn same_shard_pair(
        &mut self,
        rng: &mut impl rand::Rng,
    ) -> Option<(&mut FundedAccount, &mut FundedAccount)> {
        self.same_shard_pair_with_distribution(rng, AccountDistribution::Random)
    }

    /// Get a pair of accounts on the same shard with specified distribution mode.
    pub fn same_shard_pair_with_distribution(
        &mut self,
        rng: &mut impl rand::Rng,
        distribution: AccountDistribution,
    ) -> Option<(&mut FundedAccount, &mut FundedAccount)> {
        let shard = ShardGroupId(rng.gen_range(0..self.num_shards));
        let num_accounts = self.by_shard.get(&shard)?.len();

        if num_accounts < 2 {
            return None;
        }

        let (idx1, idx2) = self.select_pair_indices(shard, num_accounts, rng, distribution);

        // Track usage
        self.record_usage(shard, idx1);
        self.record_usage(shard, idx2);

        // Use split_at_mut to get two mutable references
        let accounts = self.by_shard.get_mut(&shard)?;
        let (min_idx, max_idx) = if idx1 < idx2 {
            (idx1, idx2)
        } else {
            (idx2, idx1)
        };
        let (left, right) = accounts.split_at_mut(max_idx);
        Some((&mut left[min_idx], &mut right[0]))
    }

    /// Select a pair of distinct account indices based on distribution mode.
    fn select_pair_indices(
        &mut self,
        shard: ShardGroupId,
        num_accounts: usize,
        rng: &mut impl rand::Rng,
        distribution: AccountDistribution,
    ) -> (usize, usize) {
        match distribution {
            AccountDistribution::Random => {
                let idx1 = rng.gen_range(0..num_accounts);
                let mut idx2 = rng.gen_range(0..num_accounts);
                while idx2 == idx1 {
                    idx2 = rng.gen_range(0..num_accounts);
                }
                (idx1, idx2)
            }
            AccountDistribution::RoundRobin => {
                // Cycles through accounts sequentially: (0,1), (2,3), (4,5)...
                let counter = self.round_robin_counters.get_mut(&shard).unwrap();
                let idx1 = (*counter * 2) % num_accounts;
                let idx2 = (*counter * 2 + 1) % num_accounts;
                *counter += 1;
                (idx1, idx2)
            }
            AccountDistribution::Zipf { exponent } => {
                // Zipf distribution: P(k) ~ 1/k^s where s is the exponent
                // Higher exponent = more skewed toward lower indices (hotspots)
                let idx1 = self.zipf_index(num_accounts, exponent, rng);
                let mut idx2 = self.zipf_index(num_accounts, exponent, rng);
                while idx2 == idx1 {
                    idx2 = self.zipf_index(num_accounts, exponent, rng);
                }
                (idx1, idx2)
            }
            AccountDistribution::NoContention => {
                // Each transaction gets a disjoint pair of accounts using a GLOBAL counter.
                // This ensures no conflicts between same-shard and cross-shard transactions.
                // With N accounts per shard, can support N/2 concurrent non-conflicting transactions.
                let counter = self.no_contention_counter;
                self.no_contention_counter += 1;
                // Each counter value maps to a unique pair: (2*c, 2*c+1)
                let pair_base = (counter * 2) % num_accounts;
                let idx1 = pair_base;
                let idx2 = (pair_base + 1) % num_accounts;
                (idx1, idx2)
            }
        }
    }

    /// Select a single account index based on distribution mode.
    /// Used for cross-shard transactions where we need one account per shard.
    fn select_single_index(
        &mut self,
        shard: ShardGroupId,
        num_accounts: usize,
        rng: &mut impl rand::Rng,
        distribution: AccountDistribution,
    ) -> usize {
        match distribution {
            AccountDistribution::Random => rng.gen_range(0..num_accounts),
            AccountDistribution::RoundRobin => {
                let counter = self.round_robin_counters.get_mut(&shard).unwrap();
                let idx = *counter % num_accounts;
                *counter += 1;
                idx
            }
            AccountDistribution::NoContention => {
                // Use the global counter and reserve a full pair slot (2 indices).
                let counter = self.no_contention_counter;
                self.no_contention_counter += 1;
                (counter * 2) % num_accounts
            }
            AccountDistribution::Zipf { exponent } => self.zipf_index(num_accounts, exponent, rng),
        }
    }

    /// Generate a Zipf-distributed index.
    fn zipf_index(&self, n: usize, exponent: u32, rng: &mut impl rand::Rng) -> usize {
        // Simplified Zipf: use inverse transform sampling approximation
        let exp = exponent.max(1) as f64;
        let u: f64 = rng.gen();
        // Inverse CDF approximation for Zipf
        let idx = ((n as f64).powf(1.0 - u)).powf(1.0 / exp) as usize;
        idx.min(n - 1)
    }

    /// Record that an account was selected.
    fn record_usage(&mut self, shard: ShardGroupId, idx: usize) {
        if let Some(counts) = self.usage_counts.get_mut(&shard) {
            if let Some(counter) = counts.get_mut(idx) {
                *counter += 1;
            }
        }
    }

    /// Get a pair of accounts on different shards (for cross-shard transactions).
    pub fn cross_shard_pair(
        &mut self,
        rng: &mut impl rand::Rng,
    ) -> Option<(FundedAccount, FundedAccount)> {
        self.cross_shard_pair_with_distribution(rng, AccountDistribution::Random)
    }

    /// Get a pair of accounts on different shards with specified distribution mode.
    pub fn cross_shard_pair_with_distribution(
        &mut self,
        rng: &mut impl rand::Rng,
        distribution: AccountDistribution,
    ) -> Option<(FundedAccount, FundedAccount)> {
        if self.num_shards < 2 {
            return None;
        }

        let shard1 = ShardGroupId(rng.gen_range(0..self.num_shards));
        let mut shard2 = ShardGroupId(rng.gen_range(0..self.num_shards));
        while shard2 == shard1 {
            shard2 = ShardGroupId(rng.gen_range(0..self.num_shards));
        }

        let num_accounts1 = self.by_shard.get(&shard1)?.len();
        let num_accounts2 = self.by_shard.get(&shard2)?.len();

        // Select single index per shard (not pairs)
        let idx1 = self.select_single_index(shard1, num_accounts1, rng, distribution);
        let idx2 = self.select_single_index(shard2, num_accounts2, rng, distribution);

        // Track usage
        self.record_usage(shard1, idx1);
        self.record_usage(shard2, idx2);

        // Clone accounts since they're on different shards
        let account1 = self.by_shard.get_mut(&shard1)?[idx1].clone();
        let account2 = self.by_shard.get(&shard2)?[idx2].clone();

        Some((account1, account2))
    }

    /// Update an account after mutation (for nonce tracking with cross-shard).
    pub fn update_account(&mut self, account: FundedAccount) {
        if let Some(accounts) = self.by_shard.get_mut(&account.shard) {
            for a in accounts.iter_mut() {
                if a.address == account.address {
                    *a = account;
                    return;
                }
            }
        }
    }

    /// Total number of accounts across all shards.
    pub fn total_accounts(&self) -> usize {
        self.by_shard.values().map(|v| v.len()).sum()
    }

    /// Number of accounts on a specific shard.
    pub fn accounts_on_shard(&self, shard: ShardGroupId) -> usize {
        self.by_shard.get(&shard).map(|v| v.len()).unwrap_or(0)
    }

    /// Get all shards with accounts.
    pub fn shards(&self) -> impl Iterator<Item = ShardGroupId> + '_ {
        self.by_shard.keys().copied()
    }

    /// Get the number of shards.
    pub fn num_shards(&self) -> u64 {
        self.num_shards
    }

    /// Get usage statistics for analysis.
    pub fn usage_stats(&self) -> AccountUsageStats {
        let mut total_selections = 0u64;
        let mut max_selections = 0u64;
        let mut min_selections = u64::MAX;
        let mut account_count = 0usize;

        for counts in self.usage_counts.values() {
            for &count in counts {
                total_selections += count;
                max_selections = max_selections.max(count);
                if count > 0 {
                    min_selections = min_selections.min(count);
                }
                account_count += 1;
            }
        }

        if min_selections == u64::MAX {
            min_selections = 0;
        }

        let avg_selections = if account_count > 0 {
            total_selections as f64 / account_count as f64
        } else {
            0.0
        };

        AccountUsageStats {
            total_selections,
            avg_selections,
            max_selections,
            min_selections,
            account_count,
        }
    }
}

/// Statistics about account usage distribution.
#[derive(Clone, Debug)]
pub struct AccountUsageStats {
    /// Total number of account selections.
    pub total_selections: u64,
    /// Average selections per account.
    pub avg_selections: f64,
    /// Maximum selections for any account.
    pub max_selections: u64,
    /// Minimum selections for any account (excluding unused).
    pub min_selections: u64,
    /// Total number of accounts.
    pub account_count: usize,
}

impl AccountUsageStats {
    /// Calculate the skew ratio (max / avg). Higher = more uneven.
    pub fn skew_ratio(&self) -> f64 {
        if self.avg_selections > 0.0 {
            self.max_selections as f64 / self.avg_selections
        } else {
            0.0
        }
    }
}

/// Errors that can occur during account pool operations.
#[derive(Debug, thiserror::Error)]
pub enum AccountPoolError {
    #[error("Could not generate enough accounts for {shards} shards with {accounts_per_shard} accounts each")]
    GenerationFailed {
        shards: u64,
        accounts_per_shard: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    #[test]
    fn test_account_generation() {
        let pool = AccountPool::generate(2, 10).unwrap();
        assert_eq!(pool.total_accounts(), 20);
        assert_eq!(pool.accounts_on_shard(ShardGroupId(0)), 10);
        assert_eq!(pool.accounts_on_shard(ShardGroupId(1)), 10);
    }

    #[test]
    fn test_no_contention_same_shard_disjoint() {
        let mut pool = AccountPool::generate(2, 20).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut used_indices: HashSet<(u64, usize)> = HashSet::new();

        // Generate 10 same-shard pairs - should all be disjoint
        for _ in 0..10 {
            let (from, to) = pool
                .same_shard_pair_with_distribution(&mut rng, AccountDistribution::NoContention)
                .unwrap();

            let shard = from.shard;
            let from_addr = from.address;
            let to_addr = to.address;

            let from_idx = pool.by_shard[&shard]
                .iter()
                .position(|a| a.address == from_addr)
                .unwrap();
            let to_idx = pool.by_shard[&shard]
                .iter()
                .position(|a| a.address == to_addr)
                .unwrap();

            assert!(
                used_indices.insert((shard.0, from_idx)),
                "Account index ({}, {}) was reused!",
                shard.0,
                from_idx
            );
            assert!(
                used_indices.insert((shard.0, to_idx)),
                "Account index ({}, {}) was reused!",
                shard.0,
                to_idx
            );
        }
    }
}
