//! Sync configuration.

use std::time::Duration;

/// Configuration for the sync protocol.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Timeout for individual block fetch requests.
    pub fetch_timeout: Duration,

    /// Maximum number of parallel fetch requests.
    pub max_parallel_fetches: usize,

    /// Maximum number of retries per block before giving up.
    pub max_retries_per_block: usize,

    /// How far ahead we allow fetching (to bound memory usage).
    /// Blocks beyond committed_height + fetch_window won't be requested yet.
    pub fetch_window: u64,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            fetch_timeout: Duration::from_secs(5),
            max_parallel_fetches: 4,
            max_retries_per_block: 3,
            fetch_window: 10,
        }
    }
}
