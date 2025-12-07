//! BFT configuration.

use std::time::Duration;

/// BFT consensus configuration.
#[derive(Debug, Clone)]
pub struct BftConfig {
    /// Interval between proposal attempts.
    pub proposal_interval: Duration,

    /// Timeout for view change.
    pub view_change_timeout: Duration,

    /// Maximum transactions per block.
    pub max_transactions_per_block: usize,

    /// Maximum certificates per block.
    pub max_certificates_per_block: usize,

    /// Maximum acceptable delay for proposer timestamp behind our clock (ms).
    pub max_timestamp_delay_ms: u64,

    /// Maximum acceptable rush for proposer timestamp ahead of our clock (ms).
    pub max_timestamp_rush_ms: u64,
}

impl Default for BftConfig {
    fn default() -> Self {
        Self {
            proposal_interval: Duration::from_millis(300),
            view_change_timeout: Duration::from_secs(3),
            max_transactions_per_block: 4096,
            max_certificates_per_block: 4096,
            max_timestamp_delay_ms: 30_000,
            max_timestamp_rush_ms: 2_000,
        }
    }
}

impl BftConfig {
    /// Create a new BFT configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the proposal interval.
    pub fn with_proposal_interval(mut self, interval: Duration) -> Self {
        self.proposal_interval = interval;
        self
    }

    /// Set the view change timeout.
    pub fn with_view_change_timeout(mut self, timeout: Duration) -> Self {
        self.view_change_timeout = timeout;
        self
    }

    /// Set the maximum transactions per block.
    pub fn with_max_transactions(mut self, max: usize) -> Self {
        self.max_transactions_per_block = max;
        self
    }
}
