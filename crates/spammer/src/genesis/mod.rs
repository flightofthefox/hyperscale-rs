//! Genesis balance generation for cluster setup.
//!
//! Generates TOML configuration for funding spammer accounts at genesis.

use crate::accounts::AccountPool;
use radix_common::math::Decimal;
use radix_common::network::NetworkDefinition;
use radix_common::prelude::AddressBech32Encoder;
use radix_common::types::ComponentAddress;
use std::fmt::Write;

/// Generate TOML-formatted genesis balances for the account pool.
///
/// Output format:
/// ```toml
/// [[genesis.xrd_balances]]
/// address = "account_sim1..."
/// balance = "1000000"
/// ```
pub fn generate_genesis_toml(
    num_shards: u64,
    accounts_per_shard: usize,
    balance: Decimal,
) -> Result<String, GenesisError> {
    let pool = AccountPool::generate(num_shards, accounts_per_shard)?;
    let balances = pool.all_genesis_balances(balance);

    Ok(format_balances_toml(&balances))
}

/// Format a list of (address, balance) pairs as TOML.
pub fn format_balances_toml(balances: &[(ComponentAddress, Decimal)]) -> String {
    let mut output = String::new();
    let encoder = AddressBech32Encoder::new(&NetworkDefinition::simulator());

    writeln!(output, "# Generated genesis balances for spammer accounts").unwrap();
    writeln!(output, "# {} accounts total", balances.len()).unwrap();
    writeln!(output).unwrap();

    for (address, balance) in balances {
        let address_str = encoder.encode(address.as_node_id().as_bytes()).unwrap();
        writeln!(output, "[[genesis.xrd_balances]]").unwrap();
        writeln!(output, "address = \"{}\"", address_str).unwrap();
        writeln!(output, "balance = \"{}\"", balance).unwrap();
        writeln!(output).unwrap();
    }

    output
}

/// Errors during genesis generation.
#[derive(Debug, thiserror::Error)]
pub enum GenesisError {
    #[error("Account generation failed: {0}")]
    AccountGeneration(#[from] crate::accounts::AccountPoolError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_genesis_toml() {
        let toml = generate_genesis_toml(2, 2, Decimal::from(1000u32)).unwrap();

        assert!(toml.contains("[[genesis.xrd_balances]]"));
        assert!(toml.contains("address = \"account_"));
        assert!(toml.contains("balance = \"1000\""));
        // Should have 4 accounts (2 shards * 2 accounts)
        assert_eq!(toml.matches("[[genesis.xrd_balances]]").count(), 4);
    }
}
