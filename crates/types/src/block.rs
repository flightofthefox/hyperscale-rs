//! Block and BlockHeader types for consensus.

use crate::{
    BlockHeight, Hash, QuorumCertificate, RoutableTransaction, TransactionAbort,
    TransactionCertificate, TransactionDefer, ValidatorId,
};
use sbor::prelude::*;

/// Block header containing consensus metadata.
///
/// The header is what validators vote on. It contains:
/// - Chain position (height, parent hash)
/// - Proposer identity
/// - Proof of parent commitment (parent QC)
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct BlockHeader {
    /// Block height in the chain (genesis = 0).
    pub height: BlockHeight,

    /// Hash of parent block.
    pub parent_hash: Hash,

    /// Quorum certificate proving parent block was committed.
    pub parent_qc: QuorumCertificate,

    /// Validator that proposed this block.
    pub proposer: ValidatorId,

    /// Unix timestamp (milliseconds) when block was proposed.
    pub timestamp: u64,

    /// View/round number for view change protocol.
    pub round: u64,

    /// Whether this block was created as a fallback when leader timed out.
    pub is_fallback: bool,
}

impl BlockHeader {
    /// Compute hash of this block header.
    pub fn hash(&self) -> Hash {
        let bytes = basic_encode(self).expect("BlockHeader serialization should never fail");
        Hash::from_bytes(&bytes)
    }

    /// Check if this is the genesis block header.
    pub fn is_genesis(&self) -> bool {
        self.height.0 == 0
    }

    /// Get the expected proposer for this height (round-robin).
    pub fn expected_proposer(&self, num_validators: u64) -> ValidatorId {
        ValidatorId((self.height.0 + self.round) % num_validators)
    }
}

/// Complete block with header and transaction data.
///
/// Blocks can contain four types of transaction-related items:
/// 1. **transactions**: New transactions being committed for the first time
/// 2. **committed_certificates**: Finalized transaction certificates (Accept/Reject decisions)
/// 3. **deferred**: Transactions deferred due to cross-shard cycles (livelock prevention)
/// 4. **aborted**: Transactions aborted due to timeout or rejection
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct Block {
    /// Block header with consensus metadata.
    pub header: BlockHeader,

    /// Transactions included in this block.
    pub transactions: Vec<RoutableTransaction>,

    /// Transaction certificates for finalized transactions.
    pub committed_certificates: Vec<TransactionCertificate>,

    /// Transactions deferred due to cross-shard livelock cycles.
    ///
    /// When cycle detection identifies a bidirectional cycle, the losing
    /// transaction (higher hash) is deferred. This releases its locks and
    /// queues it for retry after the winner completes.
    pub deferred: Vec<TransactionDefer>,

    /// Transactions aborted due to timeout or explicit rejection.
    ///
    /// Aborts are terminal - the transaction will not be retried. This is
    /// used for N-way cycles that cannot be resolved via simple deferral,
    /// or for transactions that explicitly failed during execution.
    pub aborted: Vec<TransactionAbort>,
}

impl Block {
    /// Compute hash of this block (hashes the header).
    pub fn hash(&self) -> Hash {
        self.header.hash()
    }

    /// Get block height.
    pub fn height(&self) -> BlockHeight {
        self.header.height
    }

    /// Get number of transactions in this block.
    pub fn transaction_count(&self) -> usize {
        self.transactions.len()
    }

    /// Check if this block contains a specific transaction by hash.
    pub fn contains_transaction(&self, tx_hash: &Hash) -> bool {
        self.transactions.iter().any(|tx| tx.hash() == *tx_hash)
    }

    /// Get transaction hashes for gossip messages.
    pub fn transaction_hashes(&self) -> Vec<Hash> {
        self.transactions.iter().map(|tx| tx.hash()).collect()
    }

    /// Check if this is the genesis block.
    pub fn is_genesis(&self) -> bool {
        self.header.is_genesis()
    }

    /// Create a genesis block.
    pub fn genesis(genesis_qc: QuorumCertificate) -> Self {
        Self {
            header: BlockHeader {
                height: BlockHeight(0),
                parent_hash: Hash::ZERO,
                parent_qc: genesis_qc,
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

    /// Get number of committed certificates in this block.
    pub fn certificate_count(&self) -> usize {
        self.committed_certificates.len()
    }

    /// Get transaction hashes from committed certificates.
    pub fn committed_transaction_hashes(&self) -> Vec<Hash> {
        self.committed_certificates
            .iter()
            .map(|cert| cert.transaction_hash)
            .collect()
    }

    /// Check if this block contains a certificate for a specific transaction.
    pub fn contains_certificate(&self, tx_hash: &Hash) -> bool {
        self.committed_certificates
            .iter()
            .any(|cert| &cert.transaction_hash == tx_hash)
    }

    /// Get number of deferred transactions in this block.
    pub fn deferred_count(&self) -> usize {
        self.deferred.len()
    }

    /// Get transaction hashes of deferred transactions.
    pub fn deferred_transaction_hashes(&self) -> Vec<Hash> {
        self.deferred.iter().map(|d| d.tx_hash).collect()
    }

    /// Check if this block contains a deferral for a specific transaction.
    pub fn contains_deferral(&self, tx_hash: &Hash) -> bool {
        self.deferred.iter().any(|d| &d.tx_hash == tx_hash)
    }

    /// Get number of aborted transactions in this block.
    pub fn aborted_count(&self) -> usize {
        self.aborted.len()
    }

    /// Get transaction hashes of aborted transactions.
    pub fn aborted_transaction_hashes(&self) -> Vec<Hash> {
        self.aborted.iter().map(|a| a.tx_hash).collect()
    }

    /// Check if this block contains an abort for a specific transaction.
    pub fn contains_abort(&self, tx_hash: &Hash) -> bool {
        self.aborted.iter().any(|a| &a.tx_hash == tx_hash)
    }

    /// Check if this block has any livelock-related content.
    pub fn has_livelock_content(&self) -> bool {
        !self.deferred.is_empty() || !self.aborted.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_header_hash_deterministic() {
        let header = BlockHeader {
            height: BlockHeight(1),
            parent_hash: Hash::from_bytes(b"parent"),
            parent_qc: QuorumCertificate::genesis(),
            proposer: ValidatorId(0),
            timestamp: 1234567890,
            round: 0,
            is_fallback: false,
        };

        let hash1 = header.hash();
        let hash2 = header.hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_genesis_block() {
        let genesis_qc = QuorumCertificate::genesis();
        let genesis = Block::genesis(genesis_qc);

        assert!(genesis.is_genesis());
        assert_eq!(genesis.height(), BlockHeight(0));
        assert_eq!(genesis.transaction_count(), 0);
    }
}
