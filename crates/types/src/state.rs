//! State-related types for cross-shard execution.

use crate::{
    exec_vote_message, state_provision_message, BlockHeight, Hash, NodeId, PartitionNumber,
    ShardGroupId, Signature, SignerBitfield, ValidatorId,
};
use sbor::prelude::*;

/// A state entry (key-value pair from the state tree).
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct StateEntry {
    /// The node (address) this entry belongs to.
    pub node_id: NodeId,

    /// Partition within the node.
    pub partition: PartitionNumber,

    /// Raw DbSortKey bytes for the substate.
    pub sort_key: Vec<u8>,

    /// SBOR-encoded substate value (None if doesn't exist).
    pub value: Option<Vec<u8>>,
}

impl StateEntry {
    /// Create a new state entry.
    pub fn new(
        node_id: NodeId,
        partition: PartitionNumber,
        sort_key: Vec<u8>,
        value: Option<Vec<u8>>,
    ) -> Self {
        Self {
            node_id,
            partition,
            sort_key,
            value,
        }
    }

    /// Compute hash of this state entry.
    pub fn hash(&self) -> Hash {
        let mut data = Vec::new();
        data.extend_from_slice(&self.node_id.0);
        data.push(self.partition.0);
        data.extend_from_slice(&self.sort_key);

        match &self.value {
            Some(value_bytes) => {
                let value_hash = Hash::from_bytes(value_bytes);
                data.extend_from_slice(value_hash.as_bytes());
            }
            None => {
                data.extend_from_slice(&[0u8; 32]); // ZERO hash
            }
        }

        Hash::from_bytes(&data)
    }
}

/// A write to a substate.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct SubstateWrite {
    /// The node being written to.
    pub node_id: NodeId,

    /// Partition within the node.
    pub partition: PartitionNumber,

    /// Key within the partition (sort key).
    pub sort_key: Vec<u8>,

    /// New value.
    pub value: Vec<u8>,
}

impl SubstateWrite {
    /// Create a new substate write.
    pub fn new(
        node_id: NodeId,
        partition: PartitionNumber,
        sort_key: Vec<u8>,
        value: Vec<u8>,
    ) -> Self {
        Self {
            node_id,
            partition,
            sort_key,
            value,
        }
    }
}

/// State provision from a source shard to a target shard.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct StateProvision {
    /// Hash of the transaction this provision is for.
    pub transaction_hash: Hash,

    /// Target shard (the shard executing the transaction).
    pub target_shard: ShardGroupId,

    /// Source shard (the shard providing the state).
    pub source_shard: ShardGroupId,

    /// Block height when this provision was created.
    pub block_height: BlockHeight,

    /// The state entries being provided.
    pub entries: Vec<StateEntry>,

    /// Validator ID in source shard who created this provision.
    pub validator_id: ValidatorId,

    /// Signature from the source shard validator.
    pub signature: Signature,
}

impl StateProvision {
    /// Create the canonical message bytes for signing.
    ///
    /// Uses the centralized `state_provision_message` function with the
    /// `DOMAIN_STATE_PROVISION` tag for domain separation.
    pub fn signing_message(&self) -> Vec<u8> {
        let entry_hashes: Vec<Hash> = self.entries.iter().map(|e| e.hash()).collect();
        state_provision_message(
            &self.transaction_hash,
            self.target_shard,
            self.source_shard,
            self.block_height,
            &entry_hashes,
        )
    }

    /// Compute a hash of all entries for comparison purposes.
    pub fn entries_hash(&self) -> Hash {
        let mut hasher = blake3::Hasher::new();
        for entry in &self.entries {
            hasher.update(entry.hash().as_bytes());
        }
        Hash::from_bytes(hasher.finalize().as_bytes())
    }
}

/// Vote on execution state from a validator.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct StateVoteBlock {
    /// Hash of the transaction.
    pub transaction_hash: Hash,

    /// Shard group that executed.
    pub shard_group_id: ShardGroupId,

    /// Merkle root of the execution outputs.
    pub state_root: Hash,

    /// Whether execution succeeded.
    pub success: bool,

    /// Validator that executed and voted.
    pub validator: ValidatorId,

    /// Signature from the voting validator.
    pub signature: Signature,
}

impl StateVoteBlock {
    /// Compute hash of this vote for aggregation.
    pub fn vote_hash(&self) -> Hash {
        let mut data = Vec::new();
        data.extend_from_slice(self.transaction_hash.as_bytes());
        data.extend_from_slice(&self.shard_group_id.0.to_le_bytes());
        data.extend_from_slice(self.state_root.as_bytes());
        data.push(if self.success { 1 } else { 0 });

        Hash::from_bytes(&data)
    }

    /// Create the canonical message bytes for signing.
    ///
    /// Uses the centralized `exec_vote_message` function with the
    /// `DOMAIN_EXEC_VOTE` tag for domain separation.
    ///
    /// Note: StateCertificates aggregate signatures from StateVoteBlocks,
    /// so StateCertificate::signing_message() returns the same format.
    pub fn signing_message(&self) -> Vec<u8> {
        exec_vote_message(
            &self.transaction_hash,
            &self.state_root,
            self.shard_group_id,
            self.success,
        )
    }
}

/// Certificate proving a shard agreed on execution state.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct StateCertificate {
    /// Hash of the transaction.
    pub transaction_hash: Hash,

    /// Shard that produced this certificate.
    pub shard_group_id: ShardGroupId,

    /// Node IDs that were READ during execution.
    pub read_nodes: Vec<NodeId>,

    /// Substate data that was WRITTEN during execution.
    pub state_writes: Vec<SubstateWrite>,

    /// Merkle root of the outputs.
    pub outputs_merkle_root: Hash,

    /// Whether execution succeeded.
    pub success: bool,

    /// Aggregated signature.
    pub aggregated_signature: Signature,

    /// Which validators signed.
    pub signers: SignerBitfield,

    /// Total voting power of all signers.
    pub voting_power: u64,
}

impl StateCertificate {
    /// Get number of signers.
    pub fn signer_count(&self) -> usize {
        self.signers.count()
    }

    /// Get list of validator indices that signed.
    pub fn signer_indices(&self) -> Vec<usize> {
        self.signers.set_indices().collect()
    }

    /// Check if execution succeeded.
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Check if execution failed.
    pub fn is_failure(&self) -> bool {
        !self.success
    }

    /// Get number of state reads.
    pub fn read_count(&self) -> usize {
        self.read_nodes.len()
    }

    /// Get number of state writes.
    pub fn write_count(&self) -> usize {
        self.state_writes.len()
    }

    /// Check if this certificate can be applied (has state writes).
    pub fn has_writes(&self) -> bool {
        !self.state_writes.is_empty()
    }

    /// Create a certificate for a single-shard transaction.
    pub fn single_shard(
        transaction_hash: Hash,
        outputs_merkle_root: Hash,
        shard_group_id: ShardGroupId,
        success: bool,
    ) -> Self {
        Self {
            transaction_hash,
            shard_group_id,
            read_nodes: vec![],
            state_writes: vec![],
            outputs_merkle_root,
            success,
            aggregated_signature: Signature::zero(),
            signers: SignerBitfield::empty(),
            voting_power: 0,
        }
    }

    /// Create the canonical message bytes for signature verification.
    ///
    /// Uses the centralized `exec_vote_message` function with the
    /// `DOMAIN_EXEC_VOTE` tag for domain separation.
    ///
    /// Note: This returns the same message format as StateVoteBlock::signing_message()
    /// because StateCertificates aggregate signatures from StateVoteBlocks. The
    /// aggregated signature is verified against this same message.
    pub fn signing_message(&self) -> Vec<u8> {
        exec_vote_message(
            &self.transaction_hash,
            &self.outputs_merkle_root,
            self.shard_group_id,
            self.success,
        )
    }
}

/// Result of executing a transaction.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct ExecutionResult {
    /// Hash of the transaction.
    pub transaction_hash: Hash,

    /// Whether execution succeeded.
    pub success: bool,

    /// Merkle root of the state changes.
    pub state_root: Hash,

    /// Writes produced by the transaction.
    pub writes: Vec<SubstateWrite>,

    /// Error message if execution failed.
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_entry_hash() {
        let entry = StateEntry {
            node_id: NodeId([1u8; 30]),
            partition: PartitionNumber(0),
            sort_key: b"key".to_vec(),
            value: Some(b"value".to_vec()),
        };

        let hash1 = entry.hash();
        let hash2 = entry.hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_single_shard_certificate() {
        let cert = StateCertificate::single_shard(
            Hash::from_bytes(b"tx"),
            Hash::from_bytes(b"root"),
            ShardGroupId(0),
            true,
        );

        assert!(cert.success);
        assert!(cert.signers.is_empty());
    }
}
