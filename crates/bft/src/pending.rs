//! Pending block assembly.
//!
//! Tracks blocks being assembled from headers + gossiped transactions.

use hyperscale_types::{
    Block, BlockHeader, Hash, RoutableTransaction, TransactionAbort, TransactionCertificate,
    TransactionDefer,
};
use indexmap::IndexMap;
use std::collections::HashSet;
use std::sync::Arc;

/// Tracks a block being assembled from header + gossiped transactions + certificates.
///
/// # Lifecycle
///
/// 1. Created from BlockHeader (all transactions/certificates marked as absent by hash)
/// 2. Full Transaction objects arrive via gossip (stored in received_transactions map)
/// 3. Certificates are fetched from local CertificateProvider
/// 4. When all transactions and certificates received, block can be constructed
/// 5. Block stored to storage
/// 6. Block ready for voting
#[derive(Debug, Clone)]
pub struct PendingBlock {
    /// Block header (received first).
    header: BlockHeader,

    /// Map of transaction hash -> full RoutableTransaction object (for received transactions).
    /// Uses IndexMap for deterministic iteration order (insertion order) when constructing blocks.
    received_transactions: IndexMap<Hash, RoutableTransaction>,

    /// Set of transaction hashes we're still waiting for (HashSet for O(1) lookup).
    missing_transaction_hashes: HashSet<Hash>,

    /// Map of transaction hash -> TransactionCertificate (for received certificates).
    /// Uses IndexMap for deterministic iteration order (insertion order) when constructing blocks.
    received_certificates: IndexMap<Hash, TransactionCertificate>,

    /// Set of certificate hashes we're still waiting for (HashSet for O(1) lookup).
    missing_certificate_hashes: HashSet<Hash>,

    /// Deferred transactions (from block header gossip).
    /// These don't need to be fetched - they're included directly in the gossip message.
    deferred: Vec<TransactionDefer>,

    /// Aborted transactions (from block header gossip).
    /// These don't need to be fetched - they're included directly in the gossip message.
    aborted: Vec<TransactionAbort>,

    /// The fully constructed block (None until all transactions/certs received).
    /// Wrapped in Arc to allow cheap cloning when returning from tracker methods.
    constructed_block: Option<Arc<Block>>,
}

impl PendingBlock {
    /// Create a new pending block from a header, transaction hashes, and certificate hashes.
    ///
    /// Initializes the missing sets with all hashes.
    pub fn new(
        header: BlockHeader,
        transaction_hashes: Vec<Hash>,
        certificate_hashes: Vec<Hash>,
    ) -> Self {
        Self::full(
            header,
            transaction_hashes,
            certificate_hashes,
            vec![],
            vec![],
        )
    }

    /// Create a new pending block with all fields including deferrals and aborts.
    pub fn full(
        header: BlockHeader,
        transaction_hashes: Vec<Hash>,
        certificate_hashes: Vec<Hash>,
        deferred: Vec<TransactionDefer>,
        aborted: Vec<TransactionAbort>,
    ) -> Self {
        Self {
            header,
            received_transactions: IndexMap::with_capacity(transaction_hashes.len()),
            missing_transaction_hashes: transaction_hashes.into_iter().collect(),
            received_certificates: IndexMap::with_capacity(certificate_hashes.len()),
            missing_certificate_hashes: certificate_hashes.into_iter().collect(),
            deferred,
            aborted,
            constructed_block: None,
        }
    }

    /// Add a received transaction.
    ///
    /// Returns true if this transaction was needed, false if duplicate or not in this block.
    pub fn add_transaction(&mut self, tx: RoutableTransaction) -> bool {
        let hash = tx.hash();
        // O(1) lookup and removal with HashSet
        if self.missing_transaction_hashes.remove(&hash) {
            self.received_transactions.insert(hash, tx);
            true
        } else {
            false
        }
    }

    /// Add a received certificate.
    ///
    /// Returns true if this certificate was needed, false if duplicate or not in this block.
    pub fn add_certificate(&mut self, cert: TransactionCertificate) -> bool {
        let cert_hash = cert.transaction_hash;
        // O(1) lookup and removal with HashSet
        if self.missing_certificate_hashes.remove(&cert_hash) {
            self.received_certificates.insert(cert_hash, cert);
            true
        } else {
            false
        }
    }

    /// Check if all transactions and certificates have been received.
    pub fn is_complete(&self) -> bool {
        self.missing_transaction_hashes.is_empty() && self.missing_certificate_hashes.is_empty()
    }

    /// Check if all transactions have been received (certificates may still be pending).
    pub fn has_all_transactions(&self) -> bool {
        self.missing_transaction_hashes.is_empty()
    }

    /// Get the number of missing transaction hashes.
    pub fn missing_transaction_count(&self) -> usize {
        self.missing_transaction_hashes.len()
    }

    /// Get the missing transaction hashes as a Vec (for iteration/display).
    pub fn missing_transactions(&self) -> Vec<Hash> {
        self.missing_transaction_hashes.iter().copied().collect()
    }

    /// Check if this pending block needs a specific transaction.
    pub fn needs_transaction(&self, tx_hash: &Hash) -> bool {
        self.missing_transaction_hashes.contains(tx_hash)
    }

    /// Get the number of missing certificate hashes.
    pub fn missing_certificate_count(&self) -> usize {
        self.missing_certificate_hashes.len()
    }

    /// Get the missing certificate hashes as a Vec (for iteration/display).
    pub fn missing_certificates(&self) -> Vec<Hash> {
        self.missing_certificate_hashes.iter().copied().collect()
    }

    /// Construct the block from header + received transactions + received certificates.
    ///
    /// Should only be called when is_complete() returns true.
    ///
    /// # Errors
    ///
    /// Returns error if block is not yet complete.
    pub fn construct_block(&mut self) -> Result<Arc<Block>, String> {
        if !self.is_complete() {
            return Err(format!(
                "Cannot construct block: {} transactions and {} certificates still missing",
                self.missing_transaction_hashes.len(),
                self.missing_certificate_hashes.len()
            ));
        }

        if let Some(ref block) = self.constructed_block {
            return Ok(Arc::clone(block));
        }

        // Build the block from header + full transaction objects + certificates
        // Use drain(..) to take ownership of data instead of cloning
        let transactions: Vec<RoutableTransaction> = self
            .received_transactions
            .drain(..)
            .map(|(_, v)| v)
            .collect();
        let certificates: Vec<TransactionCertificate> = self
            .received_certificates
            .drain(..)
            .map(|(_, v)| v)
            .collect();
        // Take deferred and aborted (replace with empty)
        let deferred = std::mem::take(&mut self.deferred);
        let aborted = std::mem::take(&mut self.aborted);

        let block = Arc::new(Block {
            header: self.header.clone(),
            transactions,
            committed_certificates: certificates,
            deferred,
            aborted,
        });

        self.constructed_block = Some(Arc::clone(&block));
        Ok(block)
    }

    /// Get the constructed block, if available.
    pub fn block(&self) -> Option<Arc<Block>> {
        self.constructed_block.as_ref().map(Arc::clone)
    }

    /// Get the block header.
    pub fn header(&self) -> &BlockHeader {
        &self.header
    }

    /// Get the block hash.
    pub fn hash(&self) -> Hash {
        self.header.hash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_types::{BlockHeight, QuorumCertificate, ValidatorId};

    fn make_header(height: u64) -> BlockHeader {
        BlockHeader {
            height: BlockHeight(height),
            parent_hash: Hash::from_bytes(b"parent"),
            parent_qc: QuorumCertificate::genesis(),
            proposer: ValidatorId(0),
            timestamp: 1234567890,
            round: 0,
            is_fallback: false,
        }
    }

    #[test]
    fn test_pending_block_creation() {
        let tx1 = Hash::from_bytes(b"tx1");
        let tx2 = Hash::from_bytes(b"tx2");
        let transactions = vec![tx1, tx2];
        let header = make_header(1);

        let pb = PendingBlock::new(header.clone(), transactions, vec![]);

        assert_eq!(pb.missing_transactions().len(), 2);
        assert!(pb.missing_transactions().contains(&tx1));
        assert!(pb.missing_transactions().contains(&tx2));
        assert!(!pb.is_complete());
        assert!(pb.block().is_none());
    }

    #[test]
    fn test_empty_block_is_complete() {
        let header = make_header(1);
        let pb = PendingBlock::new(header, vec![], vec![]);

        assert!(pb.is_complete());
    }
}
