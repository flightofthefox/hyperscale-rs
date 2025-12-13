//! Per-peer rate limiting for inbound sync requests.
//!
//! Uses a token bucket algorithm to limit the rate of requests per peer.
//! Known validators (in topology) get higher limits than unknown peers.

use libp2p::PeerId;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for rate limiting.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second for known validators.
    pub validator_requests_per_sec: u32,
    /// Maximum requests per second for unknown peers.
    pub unknown_peer_requests_per_sec: u32,
    /// Maximum burst size (bucket capacity) for validators.
    pub validator_burst: u32,
    /// Maximum burst size (bucket capacity) for unknown peers.
    pub unknown_peer_burst: u32,
    /// How long to track a peer after their last request (cleanup threshold).
    pub peer_ttl: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            // Validators can request up to 1000 blocks/sec with burst of 200
            // High limits allow fast sync catch-up without being a bottleneck
            validator_requests_per_sec: 1000,
            validator_burst: 200,
            // Unknown peers limited to 10 blocks/sec with burst of 20
            unknown_peer_requests_per_sec: 10,
            unknown_peer_burst: 20,
            // Clean up peer state after 5 minutes of inactivity
            peer_ttl: Duration::from_secs(300),
        }
    }
}

/// Token bucket state for a single peer.
#[derive(Debug)]
struct TokenBucket {
    /// Current number of tokens available.
    tokens: f64,
    /// Maximum tokens (bucket capacity).
    capacity: f64,
    /// Tokens added per second.
    refill_rate: f64,
    /// Last time we updated the bucket.
    last_update: Instant,
    /// Last time this peer made a request (for cleanup).
    last_request: Instant,
}

impl TokenBucket {
    fn new(capacity: u32, refill_rate: u32) -> Self {
        let now = Instant::now();
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            refill_rate: refill_rate as f64,
            last_update: now,
            last_request: now,
        }
    }

    /// Try to consume one token. Returns true if allowed, false if rate limited.
    fn try_consume(&mut self) -> bool {
        let now = Instant::now();

        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_update = now;
        self.last_request = now;

        // Try to consume a token
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Check if this bucket has been inactive for longer than the given duration.
    fn is_stale(&self, ttl: Duration) -> bool {
        self.last_request.elapsed() > ttl
    }
}

/// Per-peer rate limiter using token buckets.
pub struct SyncRateLimiter {
    config: RateLimitConfig,
    /// Token buckets per peer.
    buckets: HashMap<PeerId, TokenBucket>,
    /// Last cleanup time.
    last_cleanup: Instant,
}

impl SyncRateLimiter {
    /// Create a new rate limiter with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: HashMap::new(),
            last_cleanup: Instant::now(),
        }
    }

    /// Check if a request from the given peer should be allowed.
    ///
    /// # Arguments
    /// * `peer` - The peer making the request
    /// * `is_validator` - Whether this peer is a known validator
    ///
    /// # Returns
    /// `true` if the request is allowed, `false` if it should be rate limited.
    pub fn check_request(&mut self, peer: &PeerId, is_validator: bool) -> bool {
        // Periodic cleanup of stale entries
        if self.last_cleanup.elapsed() > Duration::from_secs(60) {
            self.cleanup();
        }

        let bucket = self.buckets.entry(*peer).or_insert_with(|| {
            if is_validator {
                TokenBucket::new(
                    self.config.validator_burst,
                    self.config.validator_requests_per_sec,
                )
            } else {
                TokenBucket::new(
                    self.config.unknown_peer_burst,
                    self.config.unknown_peer_requests_per_sec,
                )
            }
        });

        bucket.try_consume()
    }

    /// Remove stale peer entries to prevent unbounded memory growth.
    fn cleanup(&mut self) {
        let ttl = self.config.peer_ttl;
        self.buckets.retain(|_, bucket| !bucket.is_stale(ttl));
        self.last_cleanup = Instant::now();
    }

    /// Get the number of tracked peers (for metrics/debugging).
    #[allow(dead_code)]
    pub fn tracked_peer_count(&self) -> usize {
        self.buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_peer() -> PeerId {
        PeerId::random()
    }

    #[test]
    fn test_validator_rate_limit() {
        let config = RateLimitConfig {
            validator_requests_per_sec: 10,
            validator_burst: 5,
            unknown_peer_requests_per_sec: 2,
            unknown_peer_burst: 2,
            peer_ttl: Duration::from_secs(60),
        };
        let mut limiter = SyncRateLimiter::new(config);
        let peer = test_peer();

        // Should allow burst of 5 requests
        for _ in 0..5 {
            assert!(limiter.check_request(&peer, true), "Should allow burst");
        }

        // 6th request should be rate limited
        assert!(
            !limiter.check_request(&peer, true),
            "Should rate limit after burst"
        );
    }

    #[test]
    fn test_unknown_peer_lower_limit() {
        let config = RateLimitConfig {
            validator_requests_per_sec: 100,
            validator_burst: 50,
            unknown_peer_requests_per_sec: 2,
            unknown_peer_burst: 2,
            peer_ttl: Duration::from_secs(60),
        };
        let mut limiter = SyncRateLimiter::new(config);
        let peer = test_peer();

        // Should allow burst of 2 requests for unknown peer
        assert!(limiter.check_request(&peer, false));
        assert!(limiter.check_request(&peer, false));

        // 3rd request should be rate limited
        assert!(!limiter.check_request(&peer, false));
    }

    #[test]
    fn test_separate_buckets_per_peer() {
        let config = RateLimitConfig {
            validator_requests_per_sec: 10,
            validator_burst: 2,
            unknown_peer_requests_per_sec: 2,
            unknown_peer_burst: 2,
            peer_ttl: Duration::from_secs(60),
        };
        let mut limiter = SyncRateLimiter::new(config);
        let peer1 = test_peer();
        let peer2 = test_peer();

        // Exhaust peer1's bucket
        assert!(limiter.check_request(&peer1, true));
        assert!(limiter.check_request(&peer1, true));
        assert!(!limiter.check_request(&peer1, true));

        // peer2 should still have their own bucket
        assert!(limiter.check_request(&peer2, true));
        assert!(limiter.check_request(&peer2, true));
    }

    #[test]
    fn test_token_refill() {
        let config = RateLimitConfig {
            validator_requests_per_sec: 1000, // High refill rate for test
            validator_burst: 1,
            unknown_peer_requests_per_sec: 1,
            unknown_peer_burst: 1,
            peer_ttl: Duration::from_secs(60),
        };
        let mut limiter = SyncRateLimiter::new(config);
        let peer = test_peer();

        // Use the one token
        assert!(limiter.check_request(&peer, true));
        assert!(!limiter.check_request(&peer, true));

        // Wait for refill (1000/sec = 1ms per token)
        std::thread::sleep(Duration::from_millis(5));

        // Should have tokens again
        assert!(limiter.check_request(&peer, true));
    }
}
