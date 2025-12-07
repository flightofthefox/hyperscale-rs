//! Sync state machine implementation.
//!
//! This is a simplified sync state that only handles validation and ordering.
//! All I/O (peer selection, retries, timeouts) is handled by the runner.

use crate::SyncConfig;
use hyperscale_core::{Action, Event};
use hyperscale_types::{Block, Hash, QuorumCertificate};
use std::collections::BTreeMap;
use std::time::Duration;

/// A fetched block waiting to be applied.
#[derive(Debug, Clone)]
struct FetchedBlock {
    block: Block,
    qc: QuorumCertificate,
}

/// The sync protocol state machine.
///
/// This is a simplified state machine that:
/// - Tracks sync target (height and hash)
/// - Validates received blocks
/// - Orders blocks for sequential application
///
/// All I/O (peer selection, parallel fetches, retries, timeouts) is handled
/// by the runner, not by this state machine.
#[derive(Debug)]
pub struct SyncState {
    /// Configuration.
    #[allow(dead_code)]
    config: SyncConfig,

    /// Current simulation/wall time.
    now: Duration,

    /// Our current committed height (updated by BftState, mirrored here).
    committed_height: u64,

    /// Target height we're syncing to (None if not syncing).
    sync_target: Option<u64>,

    /// Target block hash (for the final block, used for verification).
    sync_target_hash: Option<Hash>,

    /// Fetched blocks waiting to be applied (keyed by height).
    /// Used to order out-of-order arrivals.
    fetched_blocks: BTreeMap<u64, FetchedBlock>,
}

impl SyncState {
    /// Create a new sync state.
    ///
    /// Note: `total_validators` parameter is kept for API compatibility but is no longer used.
    /// Peer selection is now handled by the runner.
    pub fn new(config: SyncConfig, _total_validators: u32) -> Self {
        Self {
            config,
            now: Duration::ZERO,
            committed_height: 0,
            sync_target: None,
            sync_target_hash: None,
            fetched_blocks: BTreeMap::new(),
        }
    }

    /// Set the current time.
    pub fn set_time(&mut self, now: Duration) {
        self.now = now;
    }

    /// Update the committed height (called when BftState commits a block).
    ///
    /// Returns actions to send the next block for sync, if any.
    pub fn set_committed_height(&mut self, height: u64) -> Vec<Action> {
        self.committed_height = height;

        // Clear any fetched blocks at or below this height
        self.fetched_blocks.retain(|h, _| *h > height);

        // Check if sync is complete
        if let Some(target) = self.sync_target {
            if height >= target {
                tracing::info!(height, target, "Sync complete");
                self.sync_target = None;
                self.sync_target_hash = None;
                return vec![Action::EnqueueInternal {
                    event: Event::SyncComplete { height },
                }];
            }
        }

        // If we're still syncing, try to send the next block
        if self.sync_target.is_some() {
            return self.try_apply_blocks();
        }

        vec![]
    }

    /// Check if we're currently syncing.
    pub fn is_syncing(&self) -> bool {
        self.sync_target.is_some()
    }

    /// Get the sync target height if syncing.
    pub fn sync_target(&self) -> Option<u64> {
        self.sync_target
    }

    /// Get the sync state's committed height.
    pub fn committed_height(&self) -> u64 {
        self.committed_height
    }

    /// Get the number of pending fetches.
    ///
    /// Note: This now returns the number of fetched-but-not-applied blocks,
    /// not pending network requests (which are handled by the runner).
    pub fn pending_fetch_count(&self) -> usize {
        self.fetched_blocks.len()
    }

    /// Check if a block at the given height has been fetched.
    pub fn has_fetched_block(&self, height: u64) -> bool {
        self.fetched_blocks.contains_key(&height)
    }

    /// Handle a SyncNeeded event.
    ///
    /// Called when BftState detects we're behind.
    /// Returns an empty Vec - the runner will handle fetching.
    pub fn on_sync_needed(&mut self, target_height: u64, target_hash: Hash) -> Vec<Action> {
        // Already syncing to this or higher?
        if self.sync_target.is_some_and(|t| t >= target_height) {
            return vec![];
        }

        tracing::info!(
            target_height,
            ?target_hash,
            committed = self.committed_height,
            "Starting sync"
        );

        self.sync_target = Some(target_height);
        self.sync_target_hash = Some(target_hash);

        // Return empty - runner handles fetching
        vec![]
    }

    /// Handle a SyncBlockReceived event.
    ///
    /// Called when the runner delivers a block from a peer.
    pub fn on_block_received(&mut self, block: Block, qc: QuorumCertificate) -> Vec<Action> {
        let height = block.header.height.0;

        tracing::debug!(height, "Block received");

        // Verify the block
        if !self.verify_block(&block, &qc) {
            tracing::warn!(height, "Block verification failed");
            // Runner will handle retry
            return vec![];
        }

        tracing::debug!(height, "Block verified successfully");

        // Store the fetched block
        self.fetched_blocks
            .insert(height, FetchedBlock { block, qc });

        // Try to apply consecutive blocks
        self.try_apply_blocks()
    }

    /// Verify a block against its QC.
    fn verify_block(&self, block: &Block, qc: &QuorumCertificate) -> bool {
        // Verify the QC certifies this block
        if qc.block_hash != block.hash() {
            tracing::warn!(
                block_hash = ?block.hash(),
                qc_hash = ?qc.block_hash,
                height = block.header.height.0,
                "QC block hash mismatch"
            );
            return false;
        }

        // Verify height matches
        if qc.height != block.header.height {
            tracing::warn!(
                block_height = block.header.height.0,
                qc_height = qc.height.0,
                "QC height mismatch"
            );
            return false;
        }

        // Note: QC signature verification is done by BftState when processing
        // SyncBlockReadyToApply. The sync state just validates basic properties.

        true
    }

    /// Try to apply consecutive fetched blocks.
    fn try_apply_blocks(&mut self) -> Vec<Action> {
        let mut actions = Vec::new();

        // Apply blocks in order starting from committed_height + 1
        // Note: We don't update committed_height here anymore.
        // BftState will verify QC signatures asynchronously, then commit and call
        // set_committed_height(). We track sent-but-not-confirmed heights to avoid
        // re-sending the same block.
        let next_to_send = self.committed_height + 1;
        if let Some(fetched) = self.fetched_blocks.remove(&next_to_send) {
            let height = fetched.block.header.height.0;

            tracing::debug!(height, "Sending synced block to BFT for verification");

            actions.push(Action::EnqueueInternal {
                event: Event::SyncBlockReadyToApply {
                    block: fetched.block,
                    qc: fetched.qc,
                },
            });
        }

        // Note: We intentionally don't check for sync completion here.
        // That will be checked when set_committed_height() is called after BFT commits.

        actions
    }

    /// Cancel sync (e.g., when we catch up through normal consensus).
    pub fn cancel_sync(&mut self) -> Vec<Action> {
        if self.sync_target.is_none() {
            return vec![];
        }

        tracing::info!(
            target = ?self.sync_target,
            committed = self.committed_height,
            "Cancelling sync"
        );

        self.sync_target = None;
        self.sync_target_hash = None;
        self.fetched_blocks.clear();

        // No timers to cancel - runner handles that
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_types::{
        BlockHeader, BlockHeight, Signature, SignerBitfield, ValidatorId, VotePower,
    };

    fn make_block(height: u64) -> Block {
        Block {
            header: BlockHeader {
                height: BlockHeight(height),
                parent_hash: Hash::from_bytes(&[0u8; 32]),
                parent_qc: QuorumCertificate::genesis(),
                proposer: ValidatorId(0),
                timestamp: 0,
                round: 0,
                is_fallback: false,
            },
            transactions: vec![],
            committed_certificates: vec![],
            deferred: vec![],
            aborted: vec![],
        }
    }

    fn make_qc_for_block(block: &Block) -> QuorumCertificate {
        QuorumCertificate {
            block_hash: block.hash(),
            height: block.header.height,
            parent_block_hash: block.header.parent_hash,
            round: block.header.round,
            aggregated_signature: Signature::zero(),
            signers: SignerBitfield::new(0),
            voting_power: VotePower(u64::MAX),
            weighted_timestamp_ms: 0,
        }
    }

    #[test]
    fn test_sync_needed_sets_target() {
        let config = SyncConfig::default();
        let mut sync = SyncState::new(config, 4);

        let actions = sync.on_sync_needed(10, Hash::from_bytes(&[1u8; 32]));

        // Should return empty actions (runner handles fetching)
        assert!(actions.is_empty());

        // Should be syncing
        assert!(sync.is_syncing());
        assert_eq!(sync.sync_target(), Some(10));
    }

    #[test]
    fn test_block_received_triggers_apply() {
        let config = SyncConfig::default();
        let mut sync = SyncState::new(config, 4);

        // Start sync
        sync.on_sync_needed(3, Hash::from_bytes(&[1u8; 32]));

        // Create blocks
        let block1 = make_block(1);
        let qc1 = make_qc_for_block(&block1);

        // Receive block 1
        let actions = sync.on_block_received(block1, qc1);

        // Should emit SyncBlockReadyToApply
        let apply_actions: Vec<_> = actions
            .iter()
            .filter(|a| {
                matches!(
                    a,
                    Action::EnqueueInternal {
                        event: Event::SyncBlockReadyToApply { .. }
                    }
                )
            })
            .collect();
        assert_eq!(apply_actions.len(), 1);
    }

    #[test]
    fn test_sync_complete() {
        let config = SyncConfig::default();
        let mut sync = SyncState::new(config, 4);

        // Start sync to height 2
        sync.on_sync_needed(2, Hash::from_bytes(&[1u8; 32]));

        // Receive block 1
        let block1 = make_block(1);
        let qc1 = make_qc_for_block(&block1);
        let actions = sync.on_block_received(block1, qc1);

        // Should emit SyncBlockReadyToApply for block 1
        assert!(actions.iter().any(|a| matches!(
            a,
            Action::EnqueueInternal {
                event: Event::SyncBlockReadyToApply { .. }
            }
        )));

        // Simulate BFT committing block 1 - this triggers sending block 2
        let actions = sync.set_committed_height(1);

        // Should try to send block 2 (but we haven't received it yet, so no action)
        // Actually block 2 isn't received yet, so no SyncBlockReadyToApply
        assert!(
            actions.is_empty()
                || actions.iter().all(|a| !matches!(
                    a,
                    Action::EnqueueInternal {
                        event: Event::SyncComplete { .. }
                    }
                ))
        );

        // Receive block 2
        let block2 = make_block(2);
        let qc2 = make_qc_for_block(&block2);
        let actions = sync.on_block_received(block2, qc2);

        // Should emit SyncBlockReadyToApply for block 2
        assert!(actions.iter().any(|a| matches!(
            a,
            Action::EnqueueInternal {
                event: Event::SyncBlockReadyToApply { .. }
            }
        )));

        // Simulate BFT committing block 2 - this should trigger SyncComplete
        let actions = sync.set_committed_height(2);

        // Should emit SyncComplete
        let complete_actions: Vec<_> = actions
            .iter()
            .filter(|a| {
                matches!(
                    a,
                    Action::EnqueueInternal {
                        event: Event::SyncComplete { .. }
                    }
                )
            })
            .collect();
        assert_eq!(complete_actions.len(), 1);

        // Should no longer be syncing
        assert!(!sync.is_syncing());
    }

    #[test]
    fn test_cancel_sync() {
        let config = SyncConfig::default();
        let mut sync = SyncState::new(config, 4);

        // Start sync
        sync.on_sync_needed(10, Hash::from_bytes(&[1u8; 32]));
        assert!(sync.is_syncing());

        // Cancel
        let actions = sync.cancel_sync();

        // Should return empty (no timers to cancel anymore)
        assert!(actions.is_empty());

        // Should no longer be syncing
        assert!(!sync.is_syncing());
    }
}
