//! LivelockState sub-state machine for cycle detection and deferral management.
//!
//! This module implements the provision-based cycle detection system that
//! prevents bidirectional livelock in cross-shard transactions.

use crate::tracker::{CommittedCrossShardTracker, ProvisionTracker};
use hyperscale_core::{Action, Event, SubStateMachine};
use hyperscale_types::{
    BlockHeight, DeferReason, Hash, RoutableTransaction, ShardGroupId, StateProvision, Topology,
    TransactionDefer,
};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, trace};

/// Configuration for livelock prevention.
#[derive(Debug, Clone)]
pub struct LivelockConfig {
    /// How long to keep tombstones for deferred transactions.
    pub tombstone_ttl: Duration,
    /// Number of blocks before a stuck transaction times out.
    pub execution_timeout_blocks: u64,
    /// Maximum retry attempts before permanent abort.
    pub max_retries: u32,
}

impl Default for LivelockConfig {
    fn default() -> Self {
        Self {
            tombstone_ttl: Duration::from_secs(30),
            execution_timeout_blocks: 30,
            max_retries: 3,
        }
    }
}

/// Livelock prevention state machine.
///
/// Handles:
/// - Cycle detection (from provision signals)
/// - Deferral queuing for block inclusion
/// - Tombstones for late provision filtering
///
/// Does NOT handle:
/// - Retry creation (handled by MempoolState)
/// - Status updates (handled by MempoolState)
/// - Provision quorum tracking (handled by ExecutionState)
pub struct LivelockState {
    /// This node's shard group.
    local_shard: ShardGroupId,

    /// Network topology for shard lookups.
    topology: Arc<dyn Topology>,

    /// Tracks committed cross-shard TXs and which shards they need provisions from.
    committed_tracker: CommittedCrossShardTracker,

    /// Tracks provisions received: (tx_hash, source_shard) pairs.
    /// Used for both early detection and deduplication.
    provision_tracker: ProvisionTracker,

    /// Tombstones for deferred transactions to discard late-arriving provisions.
    /// Maps tx_hash -> tombstone expiry time.
    /// Added when deferral COMMITS (not when cycle detected).
    deferred_tombstones: HashMap<Hash, Duration>,

    /// Deferrals ready to be included in next block proposal.
    /// Kept until they appear in a committed block.
    pending_deferrals: Vec<TransactionDefer>,

    /// Current time.
    now: Duration,

    /// Configuration.
    config: LivelockConfig,
}

impl std::fmt::Debug for LivelockState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LivelockState")
            .field("local_shard", &self.local_shard)
            .field("committed_tracker_len", &self.committed_tracker.len())
            .field("provision_tracker_len", &self.provision_tracker.len())
            .field("deferred_tombstones_len", &self.deferred_tombstones.len())
            .field("pending_deferrals_len", &self.pending_deferrals.len())
            .finish()
    }
}

impl LivelockState {
    /// Create a new LivelockState.
    pub fn new(local_shard: ShardGroupId, topology: Arc<dyn Topology>) -> Self {
        Self::with_config(local_shard, topology, LivelockConfig::default())
    }

    /// Create a new LivelockState with custom configuration.
    pub fn with_config(
        local_shard: ShardGroupId,
        topology: Arc<dyn Topology>,
        config: LivelockConfig,
    ) -> Self {
        Self {
            local_shard,
            topology,
            committed_tracker: CommittedCrossShardTracker::new(),
            provision_tracker: ProvisionTracker::new(),
            deferred_tombstones: HashMap::new(),
            pending_deferrals: Vec::new(),
            now: Duration::ZERO,
            config,
        }
    }

    /// Called when a cross-shard transaction is committed.
    ///
    /// Registers the transaction for cycle detection by tracking which
    /// shards it needs provisions from.
    pub fn on_cross_shard_committed(&mut self, tx: &RoutableTransaction, height: BlockHeight) {
        let tx_hash = tx.hash();

        // Determine which shards we need provisions from
        let provisioning_shards = self.provisioning_shards_for_tx(tx);

        if provisioning_shards.is_empty() {
            // Not actually cross-shard, nothing to track
            return;
        }

        debug!(
            tx_hash = %tx_hash,
            height = height.0,
            shards = ?provisioning_shards,
            "Tracking committed cross-shard TX for cycle detection"
        );

        self.committed_tracker.add(tx_hash, provisioning_shards);
    }

    /// Called when we receive a provision from another shard.
    ///
    /// Performs cycle detection and queues a deferral if a bidirectional
    /// cycle is detected.
    ///
    /// Returns empty vec - no actions emitted. Just updates internal state.
    pub fn on_provision_received(&mut self, provision: &StateProvision) {
        let remote_tx_hash = provision.transaction_hash;
        let source_shard = provision.source_shard;

        trace!(
            remote_tx = %remote_tx_hash,
            source_shard = source_shard.0,
            "Processing provision for cycle detection"
        );

        // Check tombstone - discard late provisions for deferred TXs
        if self.deferred_tombstones.contains_key(&remote_tx_hash) {
            trace!(
                remote_tx = %remote_tx_hash,
                "Discarding provision - TX has tombstone (was deferred)"
            );
            return;
        }

        // Check if we've already processed this (tx, shard) for cycle detection
        // Only perform cycle detection once per (tx, shard) pair
        if !self.provision_tracker.add(remote_tx_hash, source_shard) {
            // Already seen this provision, skip cycle detection
            return;
        }

        // Cycle detection: Do we have any committed TXs that need provisions
        // from the source_shard? If so, and the remote TX needs provisions
        // from us, we have a bidirectional cycle.
        self.check_for_cycle(remote_tx_hash, source_shard);
    }

    /// Check for a bidirectional cycle with a remote transaction.
    ///
    /// A cycle exists when:
    /// 1. We have a local TX that needs provisions from source_shard
    /// 2. The remote TX (from source_shard) needs provisions from us
    ///
    /// When a cycle is detected, the transaction with the higher hash loses
    /// and is deferred. Both shards independently reach the same conclusion.
    fn check_for_cycle(&mut self, remote_tx_hash: Hash, source_shard: ShardGroupId) {
        // Get all our committed TXs that need provisions from the remote shard
        // Clone to avoid borrow issues when we need to call queue_deferral
        let local_txs_needing_source: Vec<Hash> = self
            .committed_tracker
            .txs_needing_shard(source_shard)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();

        if local_txs_needing_source.is_empty() {
            return;
        }

        // For each local TX that needs the source shard...
        for local_tx_hash in local_txs_needing_source {
            // Check if the local TX needs provisions from the source shard AND
            // the source shard has a TX that needs provisions from us.
            // The provision we just received proves the latter!

            // This is a bidirectional cycle: local_tx -> source_shard -> us
            // and remote_tx -> us (proven by receiving provision)

            // Determine winner by hash comparison (lower hash wins)
            let (winner, loser) = if local_tx_hash < remote_tx_hash {
                (local_tx_hash, remote_tx_hash)
            } else if remote_tx_hash < local_tx_hash {
                (remote_tx_hash, local_tx_hash)
            } else {
                // Same hash - impossible in practice, skip
                continue;
            };

            // Only defer if we own the loser
            if loser == local_tx_hash {
                debug!(
                    local_tx = %local_tx_hash,
                    remote_tx = %remote_tx_hash,
                    winner = %winner,
                    "Cycle detected - our TX loses, queuing deferral"
                );

                self.queue_deferral(local_tx_hash, winner);
            } else {
                debug!(
                    local_tx = %local_tx_hash,
                    remote_tx = %remote_tx_hash,
                    winner = %winner,
                    "Cycle detected - our TX wins, remote TX should defer"
                );
            }
        }
    }

    /// Queue a deferral for inclusion in the next block.
    fn queue_deferral(&mut self, loser_tx: Hash, winner_tx: Hash) {
        // Check if already queued
        if self.pending_deferrals.iter().any(|d| d.tx_hash == loser_tx) {
            trace!(tx = %loser_tx, "Deferral already queued");
            return;
        }

        let deferral = TransactionDefer {
            tx_hash: loser_tx,
            reason: DeferReason::LivelockCycle {
                winner_tx_hash: winner_tx,
            },
            block_height: BlockHeight(0), // Will be filled in when included in block
        };

        self.pending_deferrals.push(deferral);
    }

    /// Get pending deferrals for block inclusion.
    ///
    /// Returns a clone of the pending deferrals. Deferrals are only removed
    /// when they appear in a committed block.
    pub fn get_pending_deferrals(&self) -> Vec<TransactionDefer> {
        self.pending_deferrals.clone()
    }

    /// Called when a block is committed.
    ///
    /// Processes deferrals, aborts, and certificates to clean up tracking state.
    pub fn on_block_committed(&mut self, block: &hyperscale_types::Block) {
        let height = block.header.height;

        // Process committed deferrals
        for deferral in &block.deferred {
            self.on_deferral_committed(&deferral.tx_hash);
        }

        // Process committed aborts
        for abort in &block.aborted {
            self.on_abort_committed(&abort.tx_hash);
        }

        // Process committed certificates (transactions completed)
        for cert in &block.committed_certificates {
            self.on_certificate_committed(&cert.transaction_hash);
        }

        // Remove deferrals that were included in this block
        let deferred_hashes: std::collections::HashSet<_> =
            block.deferred.iter().map(|d| d.tx_hash).collect();
        self.pending_deferrals
            .retain(|d| !deferred_hashes.contains(&d.tx_hash));

        trace!(
            height = height.0,
            deferred = block.deferred.len(),
            aborted = block.aborted.len(),
            certificates = block.committed_certificates.len(),
            "Processed block commit for livelock state"
        );
    }

    /// Called when a deferral commits.
    fn on_deferral_committed(&mut self, tx_hash: &Hash) {
        // Add tombstone with TTL
        let expiry = self.now + self.config.tombstone_ttl;
        self.deferred_tombstones.insert(*tx_hash, expiry);

        // Remove from tracking
        self.committed_tracker.remove(tx_hash);
        self.provision_tracker.remove_tx(tx_hash);

        debug!(
            tx = %tx_hash,
            tombstone_expiry = ?expiry,
            "Deferral committed - added tombstone"
        );
    }

    /// Called when an abort commits.
    fn on_abort_committed(&mut self, tx_hash: &Hash) {
        // Remove from tracking (no tombstone needed - abort is terminal)
        self.committed_tracker.remove(tx_hash);
        self.provision_tracker.remove_tx(tx_hash);

        debug!(tx = %tx_hash, "Abort committed - removed from tracking");
    }

    /// Called when a certificate commits.
    fn on_certificate_committed(&mut self, tx_hash: &Hash) {
        // Remove from tracking
        self.committed_tracker.remove(tx_hash);
        self.provision_tracker.remove_tx(tx_hash);

        trace!(tx = %tx_hash, "Certificate committed - removed from tracking");
    }

    /// Cleanup expired tombstones.
    ///
    /// Called periodically by the cleanup timer.
    pub fn cleanup(&mut self) {
        let now = self.now;
        let before = self.deferred_tombstones.len();

        self.deferred_tombstones.retain(|_, expiry| *expiry > now);

        let removed = before - self.deferred_tombstones.len();
        if removed > 0 {
            debug!(
                removed,
                remaining = self.deferred_tombstones.len(),
                "Cleaned up expired tombstones"
            );
        }
    }

    /// Determine which shards we need to provision from for a transaction.
    ///
    /// Returns the set of shards that own state this transaction reads/writes,
    /// excluding our own shard.
    fn provisioning_shards_for_tx(&self, tx: &RoutableTransaction) -> BTreeSet<ShardGroupId> {
        let mut shards = BTreeSet::new();

        for node_id in tx.all_declared_nodes() {
            let shard = self.topology.shard_for_node_id(node_id);
            if shard != self.local_shard {
                shards.insert(shard);
            }
        }

        shards
    }

    /// Check if a transaction is cross-shard (needs provisions from other shards).
    pub fn is_cross_shard(&self, tx: &RoutableTransaction) -> bool {
        !self.provisioning_shards_for_tx(tx).is_empty()
    }
}

impl SubStateMachine for LivelockState {
    fn try_handle(&mut self, event: &Event) -> Option<Vec<Action>> {
        match event {
            Event::StateProvisionReceived { provision, .. } => {
                self.on_provision_received(provision);
                Some(vec![])
            }
            Event::BlockCommitted { block, .. } => {
                self.on_block_committed(block);
                Some(vec![])
            }
            Event::CleanupTimer => {
                self.cleanup();
                Some(vec![])
            }
            _ => None,
        }
    }

    fn set_time(&mut self, now: Duration) {
        self.now = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_types::{
        KeyPair, Signature, StaticTopology, ValidatorId, ValidatorInfo, ValidatorSet,
    };

    fn make_test_topology(local_shard: ShardGroupId) -> Arc<dyn Topology> {
        // Create a simple topology with 3 validators per shard
        let validators: Vec<_> = (0..6)
            .map(|i| ValidatorInfo {
                validator_id: ValidatorId(i),
                public_key: KeyPair::generate_ed25519().public_key(),
                voting_power: 1,
            })
            .collect();

        Arc::new(StaticTopology::with_local_shard(
            ValidatorId(local_shard.0 * 3), // First validator in shard
            local_shard,
            2,
            ValidatorSet::new(validators),
        ))
    }

    fn make_provision(tx_hash: Hash, source_shard: ShardGroupId) -> StateProvision {
        StateProvision {
            transaction_hash: tx_hash,
            target_shard: ShardGroupId(0),
            source_shard,
            block_height: BlockHeight(1),
            entries: vec![],
            validator_id: ValidatorId(0),
            signature: Signature::zero(),
        }
    }

    // Helper to create a hash with predictable ordering from raw bytes
    fn hash_with_prefix(prefix: u8) -> Hash {
        // Create a hash directly from raw bytes (not blake3 hashed)
        // This gives us predictable ordering: lower prefix = lower hash
        let mut bytes = [0u8; 32];
        bytes[0] = prefix;
        Hash::from_hash_bytes(&bytes)
    }

    #[test]
    fn test_cycle_detection_basic() {
        let topology = make_test_topology(ShardGroupId(0));
        let mut state = LivelockState::new(ShardGroupId(0), topology);

        // Create hashes with predictable ordering
        // local_tx has higher first byte (0xFF) so it loses
        // remote_tx has lower first byte (0x00) so it wins
        let local_tx = hash_with_prefix(0xFF); // Higher hash (will lose)
        let remote_tx = hash_with_prefix(0x00); // Lower hash (will win)

        // Register local TX as committed needing shard 1
        state
            .committed_tracker
            .add(local_tx, [ShardGroupId(1)].into_iter().collect());

        // Receive provision from shard 1 for remote_tx
        // This simulates shard 1 having committed a TX that needs our state
        let provision = make_provision(remote_tx, ShardGroupId(1));
        state.on_provision_received(&provision);

        // Should have queued a deferral (local_tx loses to remote_tx)
        let deferrals = state.get_pending_deferrals();
        assert_eq!(deferrals.len(), 1);
        assert_eq!(deferrals[0].tx_hash, local_tx);

        let DeferReason::LivelockCycle { winner_tx_hash } = &deferrals[0].reason;
        assert_eq!(*winner_tx_hash, remote_tx);
    }

    #[test]
    fn test_no_cycle_when_we_win() {
        let topology = make_test_topology(ShardGroupId(0));
        let mut state = LivelockState::new(ShardGroupId(0), topology);

        // Create hashes with predictable ordering
        // local_tx has lower first byte (0x00) so it wins
        // remote_tx has higher first byte (0xFF) so it loses
        let local_tx = hash_with_prefix(0x00); // Lower hash (will win)
        let remote_tx = hash_with_prefix(0xFF); // Higher hash (will lose)

        // Register local TX as committed needing shard 1
        state
            .committed_tracker
            .add(local_tx, [ShardGroupId(1)].into_iter().collect());

        // Receive provision from shard 1 for remote_tx
        let provision = make_provision(remote_tx, ShardGroupId(1));
        state.on_provision_received(&provision);

        // Should NOT have queued a deferral (we win, remote should defer)
        assert!(state.get_pending_deferrals().is_empty());
    }

    #[test]
    fn test_tombstone_filters_late_provisions() {
        let topology = make_test_topology(ShardGroupId(0));
        let mut state = LivelockState::new(ShardGroupId(0), topology);

        let tx = Hash::from_bytes(b"deferred_tx");

        // Add tombstone for a deferred TX
        state
            .deferred_tombstones
            .insert(tx, Duration::from_secs(100));

        // Receive provision for the deferred TX
        let provision = make_provision(tx, ShardGroupId(1));
        state.on_provision_received(&provision);

        // Should not have added to provision tracker (tombstone filtered)
        assert!(!state.provision_tracker.has_provision(tx, ShardGroupId(1)));
    }

    #[test]
    fn test_cleanup_expired_tombstones() {
        let topology = make_test_topology(ShardGroupId(0));
        let mut state = LivelockState::new(ShardGroupId(0), topology);

        let tx1 = Hash::from_bytes(b"tx1");
        let tx2 = Hash::from_bytes(b"tx2");

        // Add tombstones with different expiry times
        state
            .deferred_tombstones
            .insert(tx1, Duration::from_secs(10));
        state
            .deferred_tombstones
            .insert(tx2, Duration::from_secs(100));

        // Set current time past first expiry
        state.now = Duration::from_secs(50);
        state.cleanup();

        // tx1 should be cleaned up, tx2 should remain
        assert!(!state.deferred_tombstones.contains_key(&tx1));
        assert!(state.deferred_tombstones.contains_key(&tx2));
    }

    #[test]
    fn test_pending_deferral_deduplication() {
        let topology = make_test_topology(ShardGroupId(0));
        let mut state = LivelockState::new(ShardGroupId(0), topology);

        // Create hashes with predictable ordering
        let local_tx = hash_with_prefix(0xFF); // Higher hash (will lose)
        let remote_tx = hash_with_prefix(0x00); // Lower hash (will win)

        // Register local TX as committed needing shard 1
        state
            .committed_tracker
            .add(local_tx, [ShardGroupId(1)].into_iter().collect());

        // Receive provision - should queue deferral
        let provision = make_provision(remote_tx, ShardGroupId(1));
        state.on_provision_received(&provision);

        assert_eq!(state.get_pending_deferrals().len(), 1);

        // Receive same provision again - should NOT queue duplicate deferral
        state.on_provision_received(&provision);

        assert_eq!(
            state.get_pending_deferrals().len(),
            1,
            "Should not queue duplicate deferral"
        );

        // Receive provision from different shard for same cycle - still no duplicate
        let provision2 = make_provision(remote_tx, ShardGroupId(2));
        state.on_provision_received(&provision2);

        assert_eq!(
            state.get_pending_deferrals().len(),
            1,
            "Should still have only one deferral for same tx"
        );
    }

    #[test]
    fn test_committed_tracker_cleanup_on_deferral() {
        let topology = make_test_topology(ShardGroupId(0));
        let mut state = LivelockState::new(ShardGroupId(0), topology);

        let tx = hash_with_prefix(0xFF);

        // Register TX as committed
        state
            .committed_tracker
            .add(tx, [ShardGroupId(1)].into_iter().collect());

        assert!(state.committed_tracker.contains(&tx));

        // Simulate deferral being committed in a block
        let deferral = TransactionDefer {
            tx_hash: tx,
            reason: DeferReason::LivelockCycle {
                winner_tx_hash: hash_with_prefix(0x00),
            },
            block_height: BlockHeight(5),
        };

        let block = hyperscale_types::Block {
            header: hyperscale_types::BlockHeader {
                height: BlockHeight(5),
                parent_hash: Hash::from_bytes(b"parent"),
                parent_qc: hyperscale_types::QuorumCertificate::genesis(),
                proposer: ValidatorId(0),
                timestamp: 1234567890,
                round: 0,
                is_fallback: false,
            },
            transactions: vec![],
            committed_certificates: vec![],
            deferred: vec![deferral],
            aborted: vec![],
        };

        state.on_block_committed(&block);

        // TX should be removed from committed tracker
        assert!(
            !state.committed_tracker.contains(&tx),
            "Deferred TX should be removed from committed tracker"
        );

        // Tombstone should be added
        assert!(
            state.deferred_tombstones.contains_key(&tx),
            "Tombstone should be added for deferred TX"
        );
    }

    #[test]
    fn test_committed_tracker_cleanup_on_certificate() {
        let topology = make_test_topology(ShardGroupId(0));
        let mut state = LivelockState::new(ShardGroupId(0), topology);

        let tx = hash_with_prefix(0xAA);

        // Register TX as committed
        state
            .committed_tracker
            .add(tx, [ShardGroupId(1)].into_iter().collect());

        assert!(state.committed_tracker.contains(&tx));

        // Simulate certificate being committed (TX completed successfully)
        let cert = hyperscale_types::TransactionCertificate {
            transaction_hash: tx,
            decision: hyperscale_types::TransactionDecision::Accept,
            shard_proofs: std::collections::BTreeMap::new(),
        };

        let block = hyperscale_types::Block {
            header: hyperscale_types::BlockHeader {
                height: BlockHeight(5),
                parent_hash: Hash::from_bytes(b"parent"),
                parent_qc: hyperscale_types::QuorumCertificate::genesis(),
                proposer: ValidatorId(0),
                timestamp: 1234567890,
                round: 0,
                is_fallback: false,
            },
            transactions: vec![],
            committed_certificates: vec![cert],
            deferred: vec![],
            aborted: vec![],
        };

        state.on_block_committed(&block);

        // TX should be removed from committed tracker (completed successfully)
        assert!(
            !state.committed_tracker.contains(&tx),
            "Completed TX should be removed from committed tracker"
        );
    }

    #[test]
    fn test_no_false_positive_unidirectional() {
        // Test that unidirectional dependencies don't trigger cycle detection
        let topology = make_test_topology(ShardGroupId(0));
        let mut state = LivelockState::new(ShardGroupId(0), topology);

        let local_tx = hash_with_prefix(0xFF);
        let remote_tx = hash_with_prefix(0x00);

        // Local TX needs shard 1, but shard 1's TX does NOT need us
        state
            .committed_tracker
            .add(local_tx, [ShardGroupId(1)].into_iter().collect());

        // Receive provision from shard 2 (not shard 1) for remote_tx
        // This means remote_tx needs our state, but we don't need shard 2's state
        // so there's no cycle with our local_tx
        let provision = make_provision(remote_tx, ShardGroupId(2));
        state.on_provision_received(&provision);

        // Should NOT queue a deferral - no cycle exists
        // Our local_tx needs shard 1, provision is from shard 2
        assert!(
            state.get_pending_deferrals().is_empty(),
            "Unidirectional dependency should not cause deferral"
        );
    }
}
