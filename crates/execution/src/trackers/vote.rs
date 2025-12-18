//! Vote tracker for cross-shard execution voting.
//!
//! Tracks the collection of execution votes during Phase 4 of the
//! cross-shard 2PC protocol.

use hyperscale_types::{Hash, NodeId, ShardGroupId, StateVoteBlock};
use std::collections::BTreeMap;

/// Tracks votes for a cross-shard transaction.
///
/// After executing a transaction with provisioned state, validators create
/// votes on the execution result (merkle root). This tracker collects votes
/// and determines when quorum is reached.
#[derive(Debug)]
pub struct VoteTracker {
    /// Transaction hash.
    tx_hash: Hash,
    /// Participating shards (for broadcasting certificate).
    participating_shards: Vec<ShardGroupId>,
    /// Read nodes from transaction.
    read_nodes: Vec<NodeId>,
    /// Votes grouped by merkle root.
    votes_by_root: BTreeMap<Hash, Vec<StateVoteBlock>>,
    /// Voting power per merkle root.
    power_by_root: BTreeMap<Hash, u64>,
    /// Quorum threshold (2f+1).
    quorum: u64,
}

impl VoteTracker {
    /// Create a new vote tracker.
    ///
    /// # Arguments
    ///
    /// * `tx_hash` - The transaction being tracked
    /// * `participating_shards` - All shards involved in this transaction
    /// * `read_nodes` - Nodes read by this transaction
    /// * `quorum` - Voting power required for quorum
    pub fn new(
        tx_hash: Hash,
        participating_shards: Vec<ShardGroupId>,
        read_nodes: Vec<NodeId>,
        quorum: u64,
    ) -> Self {
        Self {
            tx_hash,
            participating_shards,
            read_nodes,
            votes_by_root: BTreeMap::new(),
            power_by_root: BTreeMap::new(),
            quorum,
        }
    }

    /// Get the transaction hash this tracker is for.
    pub fn tx_hash(&self) -> Hash {
        self.tx_hash
    }

    /// Get the participating shards.
    pub fn participating_shards(&self) -> &[ShardGroupId] {
        &self.participating_shards
    }

    /// Get the read nodes.
    pub fn read_nodes(&self) -> &[NodeId] {
        &self.read_nodes
    }

    /// Add a vote and its voting power.
    pub fn add_vote(&mut self, vote: StateVoteBlock, power: u64) {
        let state_root = vote.state_root;
        self.votes_by_root.entry(state_root).or_default().push(vote);
        *self.power_by_root.entry(state_root).or_insert(0) += power;
    }

    /// Check if quorum is reached for any merkle root.
    ///
    /// Returns `Some((merkle_root, matching_votes, total_power))` if quorum
    /// is reached, `None` otherwise.
    pub fn check_quorum(&self) -> Option<(Hash, Vec<StateVoteBlock>, u64)> {
        for (merkle_root, power) in &self.power_by_root {
            if *power >= self.quorum {
                let votes = self.votes_by_root[merkle_root].clone();
                return Some((*merkle_root, votes, *power));
            }
        }
        None
    }

    /// Get the quorum needed for this tracker.
    pub fn quorum_needed(&self) -> u64 {
        self.quorum
    }

    /// Get total voting power accumulated so far.
    pub fn total_power(&self) -> u64 {
        self.power_by_root.values().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_types::{build_merkle_tree_with_proofs, vote_leaf_hash, Signature, ValidatorId};

    fn test_vote(tx_hash: Hash, state_root: Hash, validator: ValidatorId) -> StateVoteBlock {
        let shard_group_id = ShardGroupId(0);
        let success = true;

        // Build merkle tree with single leaf
        let leaf_hash = vote_leaf_hash(&tx_hash, &state_root, shard_group_id.0, success);
        let (merkle_root, proofs) = build_merkle_tree_with_proofs(&[leaf_hash]);

        StateVoteBlock {
            transaction_hash: tx_hash,
            shard_group_id,
            state_root,
            success,
            validator,
            signature: Signature::zero(),
            vote_merkle_root: merkle_root,
            vote_merkle_proof: proofs.into_iter().next().unwrap(),
            batch_block_height: None,
        }
    }

    #[test]
    fn test_vote_tracker_quorum() {
        let tx_hash = Hash::from_bytes(b"test_tx");
        let merkle_root = Hash::from_bytes(b"merkle_root");

        let mut tracker = VoteTracker::new(
            tx_hash,
            vec![ShardGroupId(0)],
            vec![],
            3, // quorum = 3
        );

        let vote = test_vote(tx_hash, merkle_root, ValidatorId(0));

        // Not quorum yet
        tracker.add_vote(vote.clone(), 1);
        assert!(tracker.check_quorum().is_none());

        tracker.add_vote(vote.clone(), 1);
        assert!(tracker.check_quorum().is_none());

        tracker.add_vote(vote.clone(), 1);

        // Now quorum
        let result = tracker.check_quorum();
        assert!(result.is_some());
        let (root, votes, power) = result.unwrap();
        assert_eq!(root, merkle_root);
        assert_eq!(votes.len(), 3);
        assert_eq!(power, 3);
    }

    #[test]
    fn test_vote_tracker_multiple_roots() {
        let tx_hash = Hash::from_bytes(b"test_tx");
        let root_a = Hash::from_bytes(b"root_a");
        let root_b = Hash::from_bytes(b"root_b");

        let mut tracker = VoteTracker::new(tx_hash, vec![ShardGroupId(0)], vec![], 3);

        let vote_a = test_vote(tx_hash, root_a, ValidatorId(0));
        let vote_b = test_vote(tx_hash, root_b, ValidatorId(1));

        // Split votes - no quorum
        tracker.add_vote(vote_a.clone(), 1);
        tracker.add_vote(vote_b.clone(), 1);
        tracker.add_vote(vote_a.clone(), 1);
        assert!(tracker.check_quorum().is_none());

        // Third vote for root_a reaches quorum
        tracker.add_vote(vote_a.clone(), 1);
        let result = tracker.check_quorum();
        assert!(result.is_some());
        let (root, _, _) = result.unwrap();
        assert_eq!(root, root_a);
    }
}
