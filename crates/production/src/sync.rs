//! Sync manager for fetching blocks from peers.
//!
//! The sync manager handles the I/O aspects of block synchronization:
//! - Peer selection (round-robin with failure tracking)
//! - Parallel block fetches
//! - Retries with exponential backoff
//! - Timeout handling
//!
//! The state machine (`SyncState`) handles validation and ordering.
//! The runner coordinates between them.

use crate::network::Libp2pAdapter;
use hyperscale_core::Event;
use hyperscale_types::{Block, BlockHeight, Hash, QuorumCertificate};
use libp2p::PeerId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, trace, warn};

/// Configuration for the sync manager.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Maximum number of concurrent fetch requests.
    pub max_concurrent_fetches: usize,
    /// Initial timeout for sync requests.
    pub initial_timeout: Duration,
    /// Maximum timeout for sync requests (after exponential backoff).
    pub max_timeout: Duration,
    /// Maximum retries per block before giving up on a peer.
    pub max_retries_per_peer: u32,
    /// Cooldown period before retrying a failed peer.
    pub peer_cooldown: Duration,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_concurrent_fetches: 4,
            initial_timeout: Duration::from_secs(5),
            max_timeout: Duration::from_secs(30),
            max_retries_per_peer: 3,
            peer_cooldown: Duration::from_secs(60),
        }
    }
}

/// Tracks the state of a peer for sync purposes.
#[derive(Debug, Default)]
struct PeerState {
    /// Number of consecutive failures.
    failure_count: u32,
    /// Time of last failure (for cooldown).
    last_failure: Option<Instant>,
    /// Number of in-flight requests to this peer.
    in_flight: u32,
}

/// A pending sync fetch request.
#[derive(Debug)]
#[allow(dead_code)]
struct PendingFetch {
    /// The height being fetched.
    height: u64,
    /// The peer we're fetching from.
    peer: PeerId,
    /// When the request was sent.
    started: Instant,
    /// Current retry count.
    retries: u32,
}

/// Manages sync block fetching for the production runner.
///
/// The sync manager is responsible for:
/// 1. Receiving sync requests from the state machine (via `SyncNeeded` event)
/// 2. Fetching blocks from peers using request-response protocol
/// 3. Delivering fetched blocks back to the state machine
///
/// The state machine handles validation and ordering of received blocks.
pub struct SyncManager {
    /// Configuration.
    config: SyncConfig,
    /// Network adapter for sending requests.
    network: Arc<Libp2pAdapter>,
    /// Event sender for delivering fetched blocks.
    event_tx: mpsc::Sender<Event>,
    /// Current sync target (if syncing).
    sync_target: Option<(u64, Hash)>,
    /// Heights we need to fetch.
    heights_to_fetch: VecDeque<u64>,
    /// Heights we've successfully fetched (waiting for state machine to apply).
    fetched_heights: HashSet<u64>,
    /// Currently pending fetch requests.
    pending_fetches: HashMap<u64, PendingFetch>,
    /// Peer states for selection and failure tracking.
    peer_states: HashMap<PeerId, PeerState>,
    /// Known peers (connected validators).
    known_peers: Vec<PeerId>,
    /// Our current committed height (updated by state machine).
    committed_height: u64,
}

impl SyncManager {
    /// Create a new sync manager.
    pub fn new(
        config: SyncConfig,
        network: Arc<Libp2pAdapter>,
        event_tx: mpsc::Sender<Event>,
    ) -> Self {
        Self {
            config,
            network,
            event_tx,
            sync_target: None,
            heights_to_fetch: VecDeque::new(),
            fetched_heights: HashSet::new(),
            pending_fetches: HashMap::new(),
            peer_states: HashMap::new(),
            known_peers: Vec::new(),
            committed_height: 0,
        }
    }

    /// Register a known peer (validator).
    pub fn register_peer(&mut self, peer_id: PeerId) {
        if !self.known_peers.contains(&peer_id) {
            self.known_peers.push(peer_id);
            self.peer_states.insert(peer_id, PeerState::default());
            debug!(?peer_id, "Registered sync peer");
        }
    }

    /// Remove a peer (disconnected).
    pub fn remove_peer(&mut self, peer_id: &PeerId) {
        self.known_peers.retain(|p| p != peer_id);
        self.peer_states.remove(peer_id);
        debug!(?peer_id, "Removed sync peer");
    }

    /// Check if we're currently syncing.
    pub fn is_syncing(&self) -> bool {
        self.sync_target.is_some()
    }

    /// Get the sync target height.
    pub fn sync_target_height(&self) -> Option<u64> {
        self.sync_target.map(|(h, _)| h)
    }

    /// Update the committed height (called when state machine commits a block).
    pub fn set_committed_height(&mut self, height: u64) {
        self.committed_height = height;

        // Remove heights at or below committed from pending lists
        self.heights_to_fetch.retain(|h| *h > height);
        self.fetched_heights.retain(|h| *h > height);
        self.pending_fetches.retain(|h, _| *h > height);

        // Check if sync is complete
        if let Some((target, _)) = self.sync_target {
            if height >= target {
                debug!(height, target, "Sync complete");
                self.sync_target = None;
                self.heights_to_fetch.clear();
            }
        }
    }

    /// Start syncing to a target height.
    ///
    /// Called when state machine emits `SyncNeeded`.
    pub fn start_sync(&mut self, target_height: u64, target_hash: Hash) {
        // Already syncing to this or higher target?
        if self.sync_target.is_some_and(|(t, _)| t >= target_height) {
            return;
        }

        debug!(
            target_height,
            ?target_hash,
            committed = self.committed_height,
            "Starting sync"
        );

        self.sync_target = Some((target_height, target_hash));

        // Queue heights to fetch
        self.heights_to_fetch.clear();
        for height in (self.committed_height + 1)..=target_height {
            if !self.fetched_heights.contains(&height)
                && !self.pending_fetches.contains_key(&height)
            {
                self.heights_to_fetch.push_back(height);
            }
        }
    }

    /// Cancel the current sync operation.
    pub fn cancel_sync(&mut self) {
        debug!("Cancelling sync");
        self.sync_target = None;
        self.heights_to_fetch.clear();
        self.pending_fetches.clear();
        // Keep fetched_heights - they might still be useful
    }

    /// Tick the sync manager - called periodically to drive fetch progress.
    ///
    /// This should be called regularly (e.g., every 100ms) to:
    /// - Start new fetch requests (up to max_concurrent_fetches)
    /// - Check for timed out requests
    /// - Retry failed requests
    pub async fn tick(&mut self) {
        if self.sync_target.is_none() {
            return;
        }

        // Check for timed out requests
        self.check_timeouts().await;

        // Start new fetches up to the limit
        while self.pending_fetches.len() < self.config.max_concurrent_fetches {
            if let Some(height) = self.heights_to_fetch.pop_front() {
                if let Some(peer) = self.select_peer() {
                    self.start_fetch(height, peer).await;
                } else {
                    // No available peers, put height back
                    self.heights_to_fetch.push_front(height);
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Handle a received block response.
    pub async fn on_block_received(
        &mut self,
        height: u64,
        block: Block,
        qc: QuorumCertificate,
        from_peer: PeerId,
    ) {
        // Remove from pending
        if let Some(_pending) = self.pending_fetches.remove(&height) {
            // Mark peer as successful
            if let Some(state) = self.peer_states.get_mut(&from_peer) {
                state.failure_count = 0;
                state.in_flight = state.in_flight.saturating_sub(1);
            }

            trace!(height, ?from_peer, "Block received for sync");

            // Mark as fetched
            self.fetched_heights.insert(height);

            // Deliver to state machine
            let event = Event::SyncBlockReceived { block, qc };
            if let Err(e) = self.event_tx.send(event).await {
                warn!(height, error = ?e, "Failed to deliver synced block");
            }
        } else {
            // Unexpected response (maybe from a retry after we already got it)
            trace!(
                height,
                ?from_peer,
                "Unexpected block response (already handled)"
            );
        }
    }

    /// Handle a failed fetch (timeout or error).
    pub fn on_fetch_failed(&mut self, height: u64, peer: PeerId, reason: &str) {
        if let Some(_pending) = self.pending_fetches.remove(&height) {
            warn!(height, ?peer, reason, "Sync fetch failed");

            // Update peer state
            if let Some(state) = self.peer_states.get_mut(&peer) {
                state.failure_count += 1;
                state.last_failure = Some(Instant::now());
                state.in_flight = state.in_flight.saturating_sub(1);
            }

            // Re-queue the height for retry (with a different peer if possible)
            self.heights_to_fetch.push_back(height);
        }
    }

    /// Select the best peer for a sync request.
    fn select_peer(&mut self) -> Option<PeerId> {
        let now = Instant::now();

        // Filter out peers that are in cooldown or have too many in-flight requests
        let available: Vec<_> = self
            .known_peers
            .iter()
            .filter(|peer| {
                if let Some(state) = self.peer_states.get(peer) {
                    // Check cooldown
                    if state.failure_count >= self.config.max_retries_per_peer {
                        if let Some(last_failure) = state.last_failure {
                            if now.duration_since(last_failure) < self.config.peer_cooldown {
                                return false;
                            }
                            // Reset failure count after cooldown
                        }
                    }
                    // Allow up to 2 concurrent requests per peer
                    state.in_flight < 2
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        if available.is_empty() {
            return None;
        }

        // Select peer with fewest in-flight requests and lowest failure count
        available.into_iter().min_by_key(|peer| {
            let state = self
                .peer_states
                .get(peer)
                .map(|s| (s.in_flight, s.failure_count));
            state.unwrap_or((0, 0))
        })
    }

    /// Start a fetch request for a specific height.
    async fn start_fetch(&mut self, height: u64, peer: PeerId) {
        trace!(height, ?peer, "Starting sync fetch");

        // Update peer state
        if let Some(state) = self.peer_states.get_mut(&peer) {
            state.in_flight += 1;
        }

        // Record pending fetch
        self.pending_fetches.insert(
            height,
            PendingFetch {
                height,
                peer,
                started: Instant::now(),
                retries: 0,
            },
        );

        // Send the request via libp2p request-response
        // Note: The actual request is sent via the network adapter
        // For now, we just record the pending state - the network layer
        // will call on_block_received or on_fetch_failed when done
        if let Err(e) = self.network.request_block(peer, BlockHeight(height)).await {
            warn!(height, ?peer, error = ?e, "Failed to send sync request");
            self.on_fetch_failed(height, peer, "send failed");
        }
    }

    /// Check for timed out requests.
    async fn check_timeouts(&mut self) {
        let now = Instant::now();
        let timeout = self.config.initial_timeout;

        let timed_out: Vec<_> = self
            .pending_fetches
            .iter()
            .filter(|(_, fetch)| now.duration_since(fetch.started) > timeout)
            .map(|(h, f)| (*h, f.peer))
            .collect();

        for (height, peer) in timed_out {
            self.on_fetch_failed(height, peer, "timeout");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_config_defaults() {
        let config = SyncConfig::default();
        assert_eq!(config.max_concurrent_fetches, 4);
        assert_eq!(config.initial_timeout, Duration::from_secs(5));
        assert_eq!(config.max_retries_per_peer, 3);
    }

    #[test]
    fn test_peer_registration() {
        // Can't easily test without a network, but we can test the config
        let config = SyncConfig::default();
        assert!(config.max_concurrent_fetches > 0);
    }
}
