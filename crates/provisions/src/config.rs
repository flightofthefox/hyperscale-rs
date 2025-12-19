//! Configuration for provision coordination.

/// Configuration for provision coordination and backpressure.
#[derive(Debug, Clone)]
pub struct ProvisionConfig {
    /// Maximum cross-shard transactions in flight.
    ///
    /// New cross-shard transactions are rejected when at this limit,
    /// unless they have provisions (indicating another shard has committed).
    pub max_cross_shard_pending: usize,

    /// Whether backpressure is enabled.
    ///
    /// When disabled, all transactions are accepted regardless of limit.
    pub backpressure_enabled: bool,
}

impl Default for ProvisionConfig {
    fn default() -> Self {
        Self {
            max_cross_shard_pending: 1024,
            backpressure_enabled: true,
        }
    }
}

impl ProvisionConfig {
    /// Create a new config with custom max pending limit.
    pub fn with_max_pending(max_cross_shard_pending: usize) -> Self {
        Self {
            max_cross_shard_pending,
            ..Default::default()
        }
    }

    /// Create a config with backpressure disabled.
    pub fn disabled() -> Self {
        Self {
            backpressure_enabled: false,
            ..Default::default()
        }
    }
}
