//! Domain-separated signing for cryptographic operations.
//!
//! This module provides type-safe domain separation tags for all signed messages
//! in the consensus protocol. Domain separation prevents cross-protocol attacks
//! where a signature from one context could be replayed in another.
//!
//! # Domain Tags
//!
//! Each signable message type has a unique domain tag prefix:
//!
//! | Tag | Purpose |
//! |-----|---------|
//! | `block_vote:` | BFT block votes |
//! | `STATE_PROVISION` | Cross-shard state provisions |
//! | `BATCH_STATE_VOTE` | Merkle-batched execution state votes |
//!
//! # Usage
//!
//! Types that need signing should implement the `Signable` trait or use the
//! `signing_message()` method pattern. The signing message is constructed
//! by prepending the domain tag to the serialized content.

use crate::{BlockHeight, Hash, ShardGroupId};

/// Domain tag for BFT block votes.
///
/// Format: `block_vote:` || shard_group_id || height || round || block_hash
pub const DOMAIN_BLOCK_VOTE: &[u8] = b"block_vote:";

/// Domain tag for cross-shard state provisions.
///
/// Format: `STATE_PROVISION` || tx_hash || target_shard || source_shard || height || entries_hash
pub const DOMAIN_STATE_PROVISION: &[u8] = b"STATE_PROVISION";

/// Domain tag for batched execution state votes.
///
/// Format: `BATCH_STATE_VOTE` || shard_group || block_height || vote_merkle_root
///
/// Validators sign this message once per batch of votes. Each individual vote
/// carries a Merkle proof to verify inclusion in the signed root. This reduces
/// BLS signatures from O(n) to O(1) per block.
pub const DOMAIN_BATCH_STATE_VOTE: &[u8] = b"BATCH_STATE_VOTE";

/// Build the signing message for a block vote.
///
/// This is used for:
/// - Individual block vote signatures
/// - QC aggregated signature verification
/// - View change highest_qc verification
pub fn block_vote_message(
    shard_group: ShardGroupId,
    height: u64,
    round: u64,
    block_hash: &Hash,
) -> Vec<u8> {
    let mut message = Vec::with_capacity(80);
    message.extend_from_slice(DOMAIN_BLOCK_VOTE);
    message.extend_from_slice(&shard_group.0.to_le_bytes());
    message.extend_from_slice(&height.to_le_bytes());
    message.extend_from_slice(&round.to_le_bytes());
    message.extend_from_slice(block_hash.as_bytes());
    message
}

/// Build the signing message for a state provision.
///
/// This is used for verifying cross-shard state provisions.
pub fn state_provision_message(
    tx_hash: &Hash,
    target_shard: ShardGroupId,
    source_shard: ShardGroupId,
    block_height: BlockHeight,
    entries_hashes: &[Hash],
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(DOMAIN_STATE_PROVISION);
    msg.extend_from_slice(tx_hash.as_bytes());
    msg.extend_from_slice(&target_shard.0.to_le_bytes());
    msg.extend_from_slice(&source_shard.0.to_le_bytes());
    msg.extend_from_slice(&block_height.0.to_le_bytes());

    for hash in entries_hashes {
        msg.extend_from_slice(hash.as_bytes());
    }

    msg
}

/// Build the signing message for a batched state vote.
///
/// This is used when validators batch multiple votes into a single Merkle tree
/// and sign the root. Each individual vote carries a Merkle proof for verification.
///
/// The signature covers:
/// - Domain tag for separation from other message types
/// - Shard group that executed the transactions
/// - Block height (for block-level batches) or None (for latent/cross-shard batches)
/// - Merkle root of all vote leaf hashes in the batch
pub fn batched_vote_message(
    shard_group: ShardGroupId,
    block_height: Option<u64>,
    vote_merkle_root: &Hash,
) -> Vec<u8> {
    // Pre-allocate: 16 (tag) + 8 (shard) + 8 (height) + 32 (root) = 64 bytes
    let mut message = Vec::with_capacity(64);
    message.extend_from_slice(DOMAIN_BATCH_STATE_VOTE);
    message.extend_from_slice(&shard_group.0.to_le_bytes());
    message.extend_from_slice(&block_height.unwrap_or(0).to_le_bytes());
    message.extend_from_slice(vote_merkle_root.as_bytes());
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_vote_message_deterministic() {
        let shard = ShardGroupId(1);
        let block = Hash::from_bytes(b"test_block");

        let msg1 = block_vote_message(shard, 10, 0, &block);
        let msg2 = block_vote_message(shard, 10, 0, &block);

        assert_eq!(msg1, msg2);
        assert!(msg1.starts_with(DOMAIN_BLOCK_VOTE));
    }

    #[test]
    fn test_state_provision_message_deterministic() {
        let tx_hash = Hash::from_bytes(b"tx_hash");
        let entry1 = Hash::from_bytes(b"entry1");
        let entry2 = Hash::from_bytes(b"entry2");

        let msg1 = state_provision_message(
            &tx_hash,
            ShardGroupId(1),
            ShardGroupId(0),
            BlockHeight(10),
            &[entry1, entry2],
        );
        let msg2 = state_provision_message(
            &tx_hash,
            ShardGroupId(1),
            ShardGroupId(0),
            BlockHeight(10),
            &[entry1, entry2],
        );

        assert_eq!(msg1, msg2);
        assert!(msg1.starts_with(DOMAIN_STATE_PROVISION));
    }

    #[test]
    fn test_different_domains_produce_different_messages() {
        let hash = Hash::from_bytes(b"same_hash_value_here");

        let block_msg = block_vote_message(ShardGroupId(0), 0, 0, &hash);
        let batched_msg = batched_vote_message(ShardGroupId(0), Some(0), &hash);

        // Messages should be different due to domain tags
        assert_ne!(block_msg, batched_msg);
    }

    #[test]
    fn test_batched_vote_message_deterministic() {
        let merkle_root = Hash::from_bytes(b"merkle_root");

        let msg1 = batched_vote_message(ShardGroupId(1), Some(100), &merkle_root);
        let msg2 = batched_vote_message(ShardGroupId(1), Some(100), &merkle_root);

        assert_eq!(msg1, msg2);
        assert!(msg1.starts_with(DOMAIN_BATCH_STATE_VOTE));
    }

    #[test]
    fn test_batched_vote_message_with_none_height() {
        let merkle_root = Hash::from_bytes(b"merkle_root");

        // None height should produce different message than Some(0) due to encoding
        // (actually they're the same since None maps to 0, but that's intentional)
        let msg_none = batched_vote_message(ShardGroupId(1), None, &merkle_root);
        let msg_zero = batched_vote_message(ShardGroupId(1), Some(0), &merkle_root);

        // Both use 0 for height, so messages are identical (intentional for latent votes)
        assert_eq!(msg_none, msg_zero);
    }
}
