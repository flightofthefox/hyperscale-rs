//! Vote batching for efficient state vote signatures.
//!
//! This module implements Merkle-batched state vote signing, where validators
//! sign a single Merkle root covering all their votes instead of signing each
//! vote individually. This reduces BLS signature operations from O(n) to O(1)
//! per block, providing ~100x speedup for signature generation.
//!
//! # Design
//!
//! The VoteBatcher collects pending votes and, on flush, builds a Merkle tree
//! over the sorted vote hashes. The validator signs the Merkle root once, and
//! each resulting StateVoteBlock carries the shared signature plus its own
//! Merkle inclusion proof.
//!
//! # Usage
//!
//! Two batching modes are supported:
//!
//! 1. **Block batches**: Single-shard transactions within a block are batched
//!    together. Flushed when block execution completes.
//!
//! 2. **Latent batches**: Cross-shard transactions that complete asynchronously
//!    are batched on a timer or size threshold.

use hyperscale_types::{
    batched_vote_message, build_merkle_tree_with_proofs, vote_leaf_hash, Hash, KeyPair,
    ShardGroupId, StateVoteBlock, ValidatorId,
};
use std::time::Duration;

/// Pending vote data before merkle batching.
#[derive(Debug, Clone)]
pub struct PendingVote {
    /// Transaction hash.
    pub tx_hash: Hash,
    /// Merkle root of execution outputs.
    pub state_root: Hash,
    /// Whether execution succeeded.
    pub success: bool,
}

/// Block-level batch of votes (flushed with block).
#[derive(Debug)]
struct BlockBatch {
    /// Block height being executed.
    block_height: u64,
    /// Accumulated votes for this block.
    votes: Vec<PendingVote>,
}

/// Batches state votes for efficient merkle-based signing.
///
/// Instead of signing each vote individually (O(n) BLS signatures per block),
/// builds a Merkle tree over all votes and signs the root once (O(1)).
/// Each vote carries a Merkle proof for verification.
///
/// This struct is deterministic - timing for latent vote flushing is controlled
/// by the caller passing the current time to relevant methods.
pub struct VoteBatcher {
    /// Block-level batch for single-shard transactions.
    /// Flushed when block execution completes.
    block_batch: Option<BlockBatch>,

    /// Pending latent votes (cross-shard, async completion).
    /// Flushed on timer or size threshold.
    latent_pending: Vec<PendingVote>,

    /// Time when latent votes were last flushed (deterministic, caller-provided).
    latent_last_flush: Duration,

    /// Signing key for creating signatures.
    signing_key: KeyPair,

    /// Local shard group.
    shard: ShardGroupId,

    /// Validator ID (derived from signing key).
    validator_id: ValidatorId,

    /// Flush latent votes after this duration.
    latent_flush_interval: Duration,

    /// Flush latent votes when this many accumulate.
    latent_flush_threshold: usize,
}

impl VoteBatcher {
    /// Create a new vote batcher.
    pub fn new(signing_key: KeyPair, shard: ShardGroupId, validator_id: ValidatorId) -> Self {
        Self {
            block_batch: None,
            latent_pending: Vec::new(),
            latent_last_flush: Duration::ZERO,
            signing_key,
            shard,
            validator_id,
            // Latent votes (cross-shard) flush immediately by default.
            // This ensures livelock resolution isn't delayed by batching.
            // Block votes still benefit from batching since they're flushed together.
            latent_flush_interval: Duration::from_millis(1),
            latent_flush_threshold: 1,
        }
    }

    /// Create a new vote batcher with custom latent flush parameters.
    pub fn with_latent_config(
        signing_key: KeyPair,
        shard: ShardGroupId,
        validator_id: ValidatorId,
        latent_flush_interval: Duration,
        latent_flush_threshold: usize,
    ) -> Self {
        Self {
            block_batch: None,
            latent_pending: Vec::new(),
            latent_last_flush: Duration::ZERO,
            signing_key,
            shard,
            validator_id,
            latent_flush_interval,
            latent_flush_threshold,
        }
    }

    /// Add a single-shard vote to the current block batch.
    ///
    /// Call this during block execution for each single-shard transaction.
    /// The batch is flushed when `flush_block()` is called.
    pub fn add_block_vote(&mut self, block_height: u64, vote: PendingVote) {
        let batch = self.block_batch.get_or_insert_with(|| BlockBatch {
            block_height,
            votes: Vec::new(),
        });

        // Sanity check: all votes in a batch should be for the same block
        debug_assert_eq!(
            batch.block_height, block_height,
            "Block height mismatch in vote batch"
        );

        batch.votes.push(vote);
    }

    /// Add a cross-shard (latent) vote.
    ///
    /// Latent votes are batched separately and flushed on a timer or size threshold.
    /// The `now` parameter should be the current deterministic time.
    /// Returns any votes ready to broadcast (if flush threshold reached).
    pub fn add_latent_vote(&mut self, vote: PendingVote, now: Duration) -> Vec<StateVoteBlock> {
        self.latent_pending.push(vote);

        // Check if we should flush latent votes
        if now.saturating_sub(self.latent_last_flush) > self.latent_flush_interval
            || self.latent_pending.len() >= self.latent_flush_threshold
        {
            self.flush_latent(now)
        } else {
            vec![]
        }
    }

    /// Flush the block batch and return signed StateVoteBlocks.
    ///
    /// Call this after block execution completes to get all votes for broadcast.
    pub fn flush_block(&mut self) -> Vec<StateVoteBlock> {
        let Some(batch) = self.block_batch.take() else {
            return vec![];
        };

        if batch.votes.is_empty() {
            return vec![];
        }

        self.create_batched_votes(batch.votes, Some(batch.block_height))
    }

    /// Flush latent votes and return signed StateVoteBlocks.
    ///
    /// The `now` parameter should be the current deterministic time.
    /// Call this periodically (e.g., on timer tick) to flush pending latent votes.
    pub fn flush_latent(&mut self, now: Duration) -> Vec<StateVoteBlock> {
        if self.latent_pending.is_empty() {
            return vec![];
        }

        let votes = std::mem::take(&mut self.latent_pending);
        self.latent_last_flush = now;

        // Latent votes use None for block height (encoded as 0 in signing message)
        self.create_batched_votes(votes, None)
    }

    /// Check if there are pending latent votes that should be flushed.
    ///
    /// The `now` parameter should be the current deterministic time.
    pub fn should_flush_latent(&self, now: Duration) -> bool {
        !self.latent_pending.is_empty()
            && (now.saturating_sub(self.latent_last_flush) > self.latent_flush_interval
                || self.latent_pending.len() >= self.latent_flush_threshold)
    }

    /// Get the number of pending block votes.
    pub fn pending_block_votes(&self) -> usize {
        self.block_batch.as_ref().map_or(0, |b| b.votes.len())
    }

    /// Get the number of pending latent votes.
    pub fn pending_latent_votes(&self) -> usize {
        self.latent_pending.len()
    }

    /// Common path: build merkle tree, sign once, create vote blocks.
    fn create_batched_votes(
        &self,
        mut votes: Vec<PendingVote>,
        block_height: Option<u64>,
    ) -> Vec<StateVoteBlock> {
        if votes.is_empty() {
            return vec![];
        }

        // 1. Sort votes deterministically by tx_hash
        votes.sort_by(|a, b| a.tx_hash.cmp(&b.tx_hash));

        // 2. Compute leaf hashes for merkle tree
        let leaf_hashes: Vec<Hash> = votes
            .iter()
            .map(|v| vote_leaf_hash(&v.tx_hash, &v.state_root, self.shard.0, v.success))
            .collect();

        // 3. Build merkle tree and get proofs
        let (merkle_root, proofs) = build_merkle_tree_with_proofs(&leaf_hashes);

        // 4. Sign merkle root ONCE
        let message = batched_vote_message(self.shard, block_height, &merkle_root);
        let signature = self.signing_key.sign(&message);

        // 5. Create vote blocks with proofs
        votes
            .into_iter()
            .zip(proofs)
            .map(|(vote, proof)| StateVoteBlock {
                transaction_hash: vote.tx_hash,
                shard_group_id: self.shard,
                state_root: vote.state_root,
                success: vote.success,
                validator: self.validator_id,
                signature: signature.clone(),
                vote_merkle_root: merkle_root,
                vote_merkle_proof: proof,
                batch_block_height: block_height,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> KeyPair {
        KeyPair::generate_bls()
    }

    #[test]
    fn test_empty_flush() {
        let keypair = test_keypair();
        let mut batcher = VoteBatcher::new(keypair, ShardGroupId(0), ValidatorId(0));

        // Empty flushes should return empty vecs
        assert!(batcher.flush_block().is_empty());
        assert!(batcher.flush_latent(Duration::ZERO).is_empty());
    }

    #[test]
    fn test_single_block_vote() {
        let keypair = test_keypair();
        let mut batcher = VoteBatcher::new(keypair, ShardGroupId(0), ValidatorId(0));

        let vote = PendingVote {
            tx_hash: Hash::from_bytes(b"tx1"),
            state_root: Hash::from_bytes(b"root1"),
            success: true,
        };

        batcher.add_block_vote(100, vote);
        let votes = batcher.flush_block();

        assert_eq!(votes.len(), 1);
        assert!(votes[0].verify_merkle_proof());
        assert_eq!(votes[0].batch_block_height, Some(100));
    }

    #[test]
    fn test_multiple_block_votes() {
        let keypair = test_keypair();
        let mut batcher = VoteBatcher::new(keypair, ShardGroupId(0), ValidatorId(0));

        // Add 10 votes
        for i in 0..10 {
            let vote = PendingVote {
                tx_hash: Hash::from_bytes(&[i; 32]),
                state_root: Hash::from_bytes(&[i + 100; 32]),
                success: i % 2 == 0,
            };
            batcher.add_block_vote(200, vote);
        }

        let votes = batcher.flush_block();

        assert_eq!(votes.len(), 10);

        // All votes should share the same merkle root and signature
        let first_root = &votes[0].vote_merkle_root;
        let first_sig = &votes[0].signature;

        for vote in &votes {
            assert!(vote.verify_merkle_proof());
            assert_eq!(&vote.vote_merkle_root, first_root);
            assert_eq!(&vote.signature, first_sig);
            assert_eq!(vote.batch_block_height, Some(200));
        }
    }

    #[test]
    fn test_latent_votes() {
        let keypair = test_keypair();
        // Use custom config with higher threshold to test batching behavior
        let mut batcher = VoteBatcher::with_latent_config(
            keypair,
            ShardGroupId(1),
            ValidatorId(1),
            Duration::from_millis(50),
            50, // threshold
        );

        // Start at time=0, add votes within the flush interval (50ms)
        // Each vote is added at the same time, so no time-based flush occurs
        let start_time = Duration::ZERO;

        // Add votes below threshold (threshold is 50, we add 10)
        for i in 0..10 {
            let vote = PendingVote {
                tx_hash: Hash::from_bytes(&[i; 32]),
                state_root: Hash::from_bytes(&[i + 50; 32]),
                success: true,
            };
            // All votes added at same time, within flush interval
            let result = batcher.add_latent_vote(vote, start_time);
            // Should not flush yet (threshold is 50 and time hasn't elapsed)
            assert!(result.is_empty(), "Vote {} unexpectedly triggered flush", i);
        }

        assert_eq!(batcher.pending_latent_votes(), 10);

        // Manual flush at a later time
        let flush_time = Duration::from_millis(100); // After flush interval
        let votes = batcher.flush_latent(flush_time);
        assert_eq!(votes.len(), 10);

        for vote in &votes {
            assert!(vote.verify_merkle_proof());
            // Latent votes have no block height
            assert_eq!(vote.batch_block_height, None);
        }

        assert_eq!(batcher.pending_latent_votes(), 0);
    }

    #[test]
    fn test_latent_votes_immediate_flush() {
        let keypair = test_keypair();
        // Default config flushes immediately (threshold=1)
        let mut batcher = VoteBatcher::new(keypair, ShardGroupId(1), ValidatorId(1));

        let vote = PendingVote {
            tx_hash: Hash::from_bytes(&[1; 32]),
            state_root: Hash::from_bytes(&[2; 32]),
            success: true,
        };

        // With default config, first vote triggers immediate flush
        let result = batcher.add_latent_vote(vote, Duration::from_millis(10));
        assert_eq!(result.len(), 1);
        assert!(result[0].verify_merkle_proof());
        assert_eq!(result[0].batch_block_height, None);
    }

    #[test]
    fn test_votes_sorted_deterministically() {
        let keypair = test_keypair();
        let mut batcher = VoteBatcher::new(keypair, ShardGroupId(0), ValidatorId(0));

        // Add votes in reverse order
        for i in (0..5).rev() {
            let vote = PendingVote {
                tx_hash: Hash::from_bytes(&[i; 32]),
                state_root: Hash::from_bytes(&[i; 32]),
                success: true,
            };
            batcher.add_block_vote(1, vote);
        }

        let votes = batcher.flush_block();

        // Votes should be sorted by tx_hash
        for i in 0..votes.len() - 1 {
            assert!(votes[i].transaction_hash < votes[i + 1].transaction_hash);
        }
    }

    #[test]
    fn test_separate_block_batches() {
        let keypair = test_keypair();
        let mut batcher = VoteBatcher::new(keypair, ShardGroupId(0), ValidatorId(0));

        // First block
        batcher.add_block_vote(
            100,
            PendingVote {
                tx_hash: Hash::from_bytes(b"tx1"),
                state_root: Hash::from_bytes(b"root1"),
                success: true,
            },
        );
        let votes1 = batcher.flush_block();

        // Second block
        batcher.add_block_vote(
            101,
            PendingVote {
                tx_hash: Hash::from_bytes(b"tx2"),
                state_root: Hash::from_bytes(b"root2"),
                success: true,
            },
        );
        let votes2 = batcher.flush_block();

        // Different blocks should have different merkle roots
        assert_ne!(votes1[0].vote_merkle_root, votes2[0].vote_merkle_root);

        // Different block heights
        assert_eq!(votes1[0].batch_block_height, Some(100));
        assert_eq!(votes2[0].batch_block_height, Some(101));
    }

    #[test]
    fn test_signature_verification() {
        let keypair = test_keypair();
        let public_key = keypair.public_key();
        let mut batcher = VoteBatcher::new(keypair, ShardGroupId(0), ValidatorId(0));

        batcher.add_block_vote(
            50,
            PendingVote {
                tx_hash: Hash::from_bytes(b"tx"),
                state_root: Hash::from_bytes(b"state"),
                success: true,
            },
        );

        let votes = batcher.flush_block();
        let vote = &votes[0];

        // Verify the signature using the signing message
        let message = vote.signing_message();
        assert!(public_key.verify(&message, &vote.signature));
    }
}
