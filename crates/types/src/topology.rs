//! Topology trait and static implementation.

use crate::{
    NodeId, PublicKey, RoutableTransaction, ShardGroupId, ValidatorId, ValidatorSet, VotePower,
};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

/// Compute which shard owns a NodeId.
pub fn shard_for_node(node_id: &NodeId, num_shards: u64) -> ShardGroupId {
    let hash = blake3::hash(&node_id.0);
    let bytes = hash.as_bytes();
    let hash_value = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    ShardGroupId(hash_value % num_shards)
}

/// Unified topology trait for consensus and execution.
pub trait Topology: Send + Sync {
    /// Get the local validator's ID.
    fn local_validator_id(&self) -> ValidatorId;

    /// Get the local shard group.
    fn local_shard(&self) -> ShardGroupId;

    /// Get the total number of shards.
    fn num_shards(&self) -> u64;

    /// Get the ordered committee members for a shard.
    fn committee_for_shard(&self, shard: ShardGroupId) -> &[ValidatorId];

    /// Get total voting power for a shard's committee.
    fn voting_power_for_shard(&self, shard: ShardGroupId) -> u64;

    /// Get voting power for a specific validator.
    fn voting_power(&self, validator_id: ValidatorId) -> Option<u64>;

    /// Get the public key for a validator.
    fn public_key(&self, validator_id: ValidatorId) -> Option<PublicKey>;

    /// Get the global validator set.
    fn global_validator_set(&self) -> &ValidatorSet;

    /// Get the validator ID at a specific index in the local committee.
    ///
    /// Returns None if the index is out of bounds.
    fn local_validator_at_index(&self, index: usize) -> Option<ValidatorId> {
        self.local_committee().get(index).copied()
    }

    // Derived methods

    /// Get the number of committee members for a shard.
    fn committee_size_for_shard(&self, shard: ShardGroupId) -> usize {
        self.committee_for_shard(shard).len()
    }

    /// Get the index of a validator in a shard's committee.
    fn committee_index_for_shard(
        &self,
        shard: ShardGroupId,
        validator_id: ValidatorId,
    ) -> Option<usize> {
        self.committee_for_shard(shard)
            .iter()
            .position(|v| *v == validator_id)
    }

    /// Check if the given voting power meets quorum for a shard (> 2/3).
    fn has_quorum_for_shard(&self, shard: ShardGroupId, voting_power: u64) -> bool {
        VotePower::has_quorum(voting_power, self.voting_power_for_shard(shard))
    }

    /// Get the minimum voting power required for quorum in a shard.
    fn quorum_threshold_for_shard(&self, shard: ShardGroupId) -> u64 {
        (self.voting_power_for_shard(shard) * 2 / 3) + 1
    }

    /// Get the ordered committee members for the local shard.
    fn local_committee(&self) -> &[ValidatorId] {
        self.committee_for_shard(self.local_shard())
    }

    /// Get total voting power for the local shard's committee.
    fn local_voting_power(&self) -> u64 {
        self.voting_power_for_shard(self.local_shard())
    }

    /// Get the number of committee members for the local shard.
    fn local_committee_size(&self) -> usize {
        self.committee_size_for_shard(self.local_shard())
    }

    /// Get the index of a validator in the local shard's committee.
    fn local_committee_index(&self, validator_id: ValidatorId) -> Option<usize> {
        self.committee_index_for_shard(self.local_shard(), validator_id)
    }

    /// Check if the given voting power meets quorum for the local shard.
    fn local_has_quorum(&self, voting_power: u64) -> bool {
        self.has_quorum_for_shard(self.local_shard(), voting_power)
    }

    /// Get the minimum voting power required for quorum in the local shard.
    fn local_quorum_threshold(&self) -> u64 {
        self.quorum_threshold_for_shard(self.local_shard())
    }

    /// Check if a validator is a member of the local shard's committee.
    fn is_committee_member(&self, validator_id: ValidatorId) -> bool {
        self.local_committee_index(validator_id).is_some()
    }

    /// Get the proposer for a given height and round.
    fn proposer_for(&self, height: u64, round: u64) -> ValidatorId {
        let index = (height + round) as usize % self.local_committee_size();
        self.local_committee()[index]
    }

    /// Check if the local validator should propose at this height and round.
    fn should_propose(&self, height: u64, round: u64) -> bool {
        self.proposer_for(height, round) == self.local_validator_id()
    }

    /// Determine which shard a NodeId belongs to.
    fn shard_for_node_id(&self, node_id: &NodeId) -> ShardGroupId {
        shard_for_node(node_id, self.num_shards())
    }

    /// Compute write shards for a transaction.
    fn consensus_shards(&self, tx: &RoutableTransaction) -> Vec<ShardGroupId> {
        tx.declared_writes
            .iter()
            .map(|node_id| self.shard_for_node_id(node_id))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Compute read-only shards for a transaction.
    fn provisioning_shards(&self, tx: &RoutableTransaction) -> Vec<ShardGroupId> {
        let write_shards: BTreeSet<_> = tx
            .declared_writes
            .iter()
            .map(|node_id| self.shard_for_node_id(node_id))
            .collect();

        tx.declared_reads
            .iter()
            .map(|node_id| self.shard_for_node_id(node_id))
            .filter(|shard| !write_shards.contains(shard))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Check if a transaction is cross-shard.
    fn is_cross_shard_transaction(&self, tx: &RoutableTransaction) -> bool {
        self.consensus_shards(tx).len() > 1
    }

    /// Check if a transaction is single-shard.
    fn is_single_shard_transaction(&self, tx: &RoutableTransaction) -> bool {
        self.consensus_shards(tx).len() <= 1
    }

    /// Get all shards involved in a transaction (both consensus and provisioning).
    fn all_shards_for_transaction(&self, tx: &RoutableTransaction) -> Vec<ShardGroupId> {
        let consensus = self.consensus_shards(tx);
        let provisioning = self.provisioning_shards(tx);
        let all: BTreeSet<_> = consensus.into_iter().chain(provisioning).collect();
        all.into_iter().collect()
    }

    /// Check if a transaction involves the local shard for consensus.
    fn involves_local_shard_for_consensus(&self, tx: &RoutableTransaction) -> bool {
        tx.declared_writes
            .iter()
            .any(|node_id| self.shard_for_node_id(node_id) == self.local_shard())
    }

    /// Check if this shard is involved in a transaction at all.
    fn involves_local_shard(&self, tx: &RoutableTransaction) -> bool {
        let local = self.local_shard();
        tx.declared_writes
            .iter()
            .chain(tx.declared_reads.iter())
            .any(|node_id| self.shard_for_node_id(node_id) == local)
    }
}

/// Errors that can occur when validating topology information.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TopologyError {
    /// Validator is not a member of the committee.
    #[error("validator {0:?} is not in the committee")]
    NotInCommittee(ValidatorId),
}

/// Per-shard committee information.
#[derive(Debug, Clone)]
struct ShardCommittee {
    committee: Vec<ValidatorId>,
    total_voting_power: u64,
}

/// Internal validator info storage.
#[derive(Debug, Clone)]
struct ValidatorInfoInternal {
    voting_power: u64,
    public_key: PublicKey,
}

/// A static topology implementation.
#[derive(Debug, Clone)]
pub struct StaticTopology {
    local_validator_id: ValidatorId,
    local_shard: ShardGroupId,
    num_shards: u64,
    shard_committees: HashMap<ShardGroupId, ShardCommittee>,
    validator_info: HashMap<ValidatorId, ValidatorInfoInternal>,
    global_validator_set: ValidatorSet,
}

impl StaticTopology {
    /// Create a new static topology from a global validator set.
    pub fn new(
        local_validator_id: ValidatorId,
        num_shards: u64,
        validator_set: ValidatorSet,
    ) -> Self {
        let local_shard = ShardGroupId(local_validator_id.0 % num_shards);

        let validator_info: HashMap<_, _> = validator_set
            .validators
            .iter()
            .map(|v| {
                (
                    v.validator_id,
                    ValidatorInfoInternal {
                        voting_power: v.voting_power,
                        public_key: v.public_key.clone(),
                    },
                )
            })
            .collect();

        let mut shard_committees: HashMap<ShardGroupId, ShardCommittee> = HashMap::new();

        for shard_id in 0..num_shards {
            shard_committees.insert(
                ShardGroupId(shard_id),
                ShardCommittee {
                    committee: Vec::new(),
                    total_voting_power: 0,
                },
            );
        }

        for v in &validator_set.validators {
            let shard = ShardGroupId(v.validator_id.0 % num_shards);
            if let Some(committee) = shard_committees.get_mut(&shard) {
                committee.committee.push(v.validator_id);
                committee.total_voting_power += v.voting_power;
            }
        }

        Self {
            local_validator_id,
            local_shard,
            num_shards,
            shard_committees,
            validator_info,
            global_validator_set: validator_set,
        }
    }

    /// Create a topology as an Arc.
    pub fn into_arc(self) -> Arc<dyn Topology> {
        Arc::new(self)
    }

    /// Create a topology with an explicit local shard assignment.
    pub fn with_local_shard(
        local_validator_id: ValidatorId,
        local_shard: ShardGroupId,
        num_shards: u64,
        validator_set: ValidatorSet,
    ) -> Self {
        let validator_info: HashMap<_, _> = validator_set
            .validators
            .iter()
            .map(|v| {
                (
                    v.validator_id,
                    ValidatorInfoInternal {
                        voting_power: v.voting_power,
                        public_key: v.public_key.clone(),
                    },
                )
            })
            .collect();

        let mut shard_committees: HashMap<ShardGroupId, ShardCommittee> = HashMap::new();

        for shard_id in 0..num_shards {
            shard_committees.insert(
                ShardGroupId(shard_id),
                ShardCommittee {
                    committee: Vec::new(),
                    total_voting_power: 0,
                },
            );
        }

        let committee = shard_committees
            .get_mut(&local_shard)
            .expect("local_shard should exist");
        for v in &validator_set.validators {
            committee.committee.push(v.validator_id);
            committee.total_voting_power += v.voting_power;
        }

        Self {
            local_validator_id,
            local_shard,
            num_shards,
            shard_committees,
            validator_info,
            global_validator_set: validator_set,
        }
    }

    /// Create a topology with explicit shard committees.
    pub fn with_shard_committees(
        local_validator_id: ValidatorId,
        local_shard: ShardGroupId,
        num_shards: u64,
        global_validator_set: &ValidatorSet,
        shard_committees: HashMap<ShardGroupId, Vec<ValidatorId>>,
    ) -> Self {
        let validator_info: HashMap<_, _> = global_validator_set
            .validators
            .iter()
            .map(|v| {
                (
                    v.validator_id,
                    ValidatorInfoInternal {
                        voting_power: v.voting_power,
                        public_key: v.public_key.clone(),
                    },
                )
            })
            .collect();

        let mut committees: HashMap<ShardGroupId, ShardCommittee> = HashMap::new();

        for shard_id in 0..num_shards {
            committees.insert(
                ShardGroupId(shard_id),
                ShardCommittee {
                    committee: Vec::new(),
                    total_voting_power: 0,
                },
            );
        }

        for (shard, validators) in shard_committees {
            if let Some(committee) = committees.get_mut(&shard) {
                for validator_id in validators {
                    let voting_power = validator_info
                        .get(&validator_id)
                        .map(|v| v.voting_power)
                        .unwrap_or(1);
                    committee.committee.push(validator_id);
                    committee.total_voting_power += voting_power;
                }
            }
        }

        Self {
            local_validator_id,
            local_shard,
            num_shards,
            shard_committees: committees,
            validator_info,
            global_validator_set: global_validator_set.clone(),
        }
    }
}

impl Topology for StaticTopology {
    fn local_validator_id(&self) -> ValidatorId {
        self.local_validator_id
    }

    fn local_shard(&self) -> ShardGroupId {
        self.local_shard
    }

    fn num_shards(&self) -> u64 {
        self.num_shards
    }

    fn committee_for_shard(&self, shard: ShardGroupId) -> &[ValidatorId] {
        self.shard_committees
            .get(&shard)
            .map(|c| c.committee.as_slice())
            .unwrap_or(&[])
    }

    fn voting_power_for_shard(&self, shard: ShardGroupId) -> u64 {
        self.shard_committees
            .get(&shard)
            .map(|c| c.total_voting_power)
            .unwrap_or(0)
    }

    fn voting_power(&self, validator_id: ValidatorId) -> Option<u64> {
        self.validator_info
            .get(&validator_id)
            .map(|v| v.voting_power)
    }

    fn public_key(&self, validator_id: ValidatorId) -> Option<PublicKey> {
        self.validator_info
            .get(&validator_id)
            .map(|v| v.public_key.clone())
    }

    fn global_validator_set(&self) -> &ValidatorSet {
        &self.global_validator_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KeyPair, ValidatorInfo};

    fn make_test_validator(id: u64, power: u64) -> ValidatorInfo {
        ValidatorInfo {
            validator_id: ValidatorId(id),
            public_key: KeyPair::generate_ed25519().public_key(),
            voting_power: power,
        }
    }

    fn make_test_topology(num_validators: u64, local_id: u64) -> StaticTopology {
        let validators: Vec<_> = (0..num_validators)
            .map(|i| make_test_validator(i, 1))
            .collect();
        StaticTopology::new(ValidatorId(local_id), 1, ValidatorSet::new(validators))
    }

    #[test]
    fn test_committee_basics() {
        let topology = make_test_topology(4, 0);

        assert_eq!(topology.local_committee_size(), 4);
        assert_eq!(topology.local_validator_id(), ValidatorId(0));
        assert_eq!(topology.local_shard(), ShardGroupId(0));
    }

    #[test]
    fn test_quorum() {
        let topology = make_test_topology(4, 0);

        assert_eq!(topology.local_voting_power(), 4);
        assert_eq!(topology.local_quorum_threshold(), 3);

        assert!(!topology.local_has_quorum(2));
        assert!(topology.local_has_quorum(3));
        assert!(topology.local_has_quorum(4));
    }

    #[test]
    fn test_proposer_rotation() {
        let topology = make_test_topology(4, 0);

        assert_eq!(topology.proposer_for(0, 0), ValidatorId(0));
        assert_eq!(topology.proposer_for(1, 0), ValidatorId(1));
        assert_eq!(topology.proposer_for(4, 0), ValidatorId(0));
        assert_eq!(topology.proposer_for(0, 1), ValidatorId(1));
    }
}
