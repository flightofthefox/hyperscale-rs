//! Hyperscale Transaction Spammer CLI
//!
//! A command-line tool for generating and submitting transactions to a Hyperscale network.

use clap::{Parser, Subcommand};
use hyperscale_spammer::accounts::SelectionMode;
use hyperscale_spammer::config::SpammerConfig;
use hyperscale_spammer::genesis::generate_genesis_toml;
use hyperscale_spammer::runner::Spammer;
use radix_common::math::Decimal;
use radix_common::network::NetworkDefinition;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "hyperscale-spammer")]
#[command(about = "Transaction spammer for Hyperscale network")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate genesis configuration with funded accounts
    Genesis {
        /// Number of shards
        #[arg(long, default_value = "2")]
        num_shards: u64,

        /// Accounts per shard
        #[arg(long, default_value = "100")]
        accounts_per_shard: usize,

        /// Initial balance per account
        #[arg(long, default_value = "1000000")]
        balance: u64,
    },

    /// Run transaction spammer against network endpoints
    Run {
        /// RPC endpoints (comma-separated, one per shard minimum)
        #[arg(short, long, value_delimiter = ',', required = true)]
        endpoints: Vec<String>,

        /// Number of shards
        #[arg(long, default_value = "2")]
        num_shards: u64,

        /// Target transactions per second
        #[arg(long, default_value = "1000")]
        tps: u64,

        /// Duration to run (e.g., "30s", "5m", "1h")
        #[arg(short, long, default_value = "60s")]
        duration: humantime::Duration,

        /// Cross-shard transaction ratio (0.0 to 1.0)
        #[arg(long, default_value = "0.3")]
        cross_shard_ratio: f64,

        /// Accounts per shard
        #[arg(long, default_value = "100")]
        accounts_per_shard: usize,

        /// Account selection mode (random, round-robin, zipf)
        #[arg(long, default_value = "random")]
        selection: String,

        /// Wait for nodes to be ready before starting
        #[arg(long)]
        wait_ready: bool,

        /// Batch size for transaction generation
        #[arg(long, default_value = "100")]
        batch_size: usize,
    },
}

fn parse_selection_mode(s: &str) -> Result<SelectionMode, String> {
    match s.to_lowercase().as_str() {
        "random" => Ok(SelectionMode::Random),
        "round-robin" | "roundrobin" => Ok(SelectionMode::RoundRobin),
        "zipf" => Ok(SelectionMode::Zipf { exponent: 1.5 }),
        s if s.starts_with("zipf:") => {
            let exp: f64 = s[5..]
                .parse()
                .map_err(|_| format!("Invalid zipf exponent: {}", &s[5..]))?;
            Ok(SelectionMode::Zipf { exponent: exp })
        }
        _ => Err(format!("Unknown selection mode: {}", s)),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Genesis {
            num_shards,
            accounts_per_shard,
            balance,
        } => {
            // Don't initialize tracing for genesis - output goes to stdout
            let toml =
                generate_genesis_toml(num_shards, accounts_per_shard, Decimal::from(balance))?;
            print!("{}", toml);
        }

        Commands::Run {
            endpoints,
            num_shards,
            tps,
            duration,
            cross_shard_ratio,
            accounts_per_shard,
            selection,
            wait_ready,
            batch_size,
        } => {
            // Initialize tracing for the run command
            tracing_subscriber::fmt::init();

            let selection_mode = parse_selection_mode(&selection)?;

            let config = SpammerConfig::new(endpoints)
                .with_num_shards(num_shards)
                .with_target_tps(tps)
                .with_cross_shard_ratio(cross_shard_ratio)
                .with_accounts_per_shard(accounts_per_shard)
                .with_selection_mode(selection_mode)
                .with_batch_size(batch_size)
                .with_network(NetworkDefinition::simulator());

            let mut spammer = Spammer::new(config)?;

            if wait_ready {
                println!("Waiting for nodes to be ready...");
                spammer.wait_for_ready(Duration::from_secs(60)).await?;
                println!("All nodes ready.");
            }

            println!("Starting spammer for {:?}...", *duration);
            let report = spammer.run_for(*duration).await;
            report.print();
        }
    }

    Ok(())
}
