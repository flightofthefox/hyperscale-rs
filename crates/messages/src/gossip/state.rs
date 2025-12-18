//! State-related gossip messages for cross-shard transactions.

use crate::trace_context::TraceContext;
use hyperscale_types::{
    NetworkMessage, ShardMessage, StateCertificate, StateProvision, StateVoteBlock,
};
use sbor::prelude::BasicSbor;

/// Broadcasts state from owning shard to executing shard for cross-shard transactions.
/// Target shard waits for 2f+1 matching provisions before marking as "provisioned".
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct StateProvisionGossip {
    /// The state provision being broadcast
    pub provision: StateProvision,
    /// Trace context for distributed tracing (empty when feature disabled).
    pub trace_context: TraceContext,
}

impl StateProvisionGossip {
    /// Create a new state provision gossip message.
    ///
    /// Does not capture trace context. Use `with_trace_context()` to include
    /// distributed tracing information.
    pub fn new(provision: StateProvision) -> Self {
        Self {
            provision,
            trace_context: TraceContext::default(),
        }
    }

    /// Create a new state provision gossip message with trace context from current span.
    ///
    /// When `trace-propagation` feature is enabled, captures the current OpenTelemetry
    /// span context for distributed tracing across nodes.
    pub fn with_trace_context(provision: StateProvision) -> Self {
        Self {
            provision,
            trace_context: TraceContext::from_current(),
        }
    }

    /// Get the inner state provision.
    pub fn provision(&self) -> &StateProvision {
        &self.provision
    }

    /// Consume and return the inner state provision.
    pub fn into_provision(self) -> StateProvision {
        self.provision
    }

    /// Get the trace context.
    pub fn trace_context(&self) -> &TraceContext {
        &self.trace_context
    }
}

// Network message implementation
impl NetworkMessage for StateProvisionGossip {
    fn message_type_id() -> &'static str {
        "state.provision"
    }
}

impl ShardMessage for StateProvisionGossip {}

/// Vote on transaction execution results within a shard (local shard only, not cross-shard).
/// 2f+1 matching votes create a StateCertificate with aggregated BLS signature.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct StateVoteBlockGossip {
    /// The state vote being gossiped
    pub vote: StateVoteBlock,
}

impl StateVoteBlockGossip {
    /// Create a new state vote block gossip message.
    pub fn new(vote: StateVoteBlock) -> Self {
        Self { vote }
    }

    /// Get the inner state vote.
    pub fn vote(&self) -> &StateVoteBlock {
        &self.vote
    }

    /// Consume and return the inner state vote.
    pub fn into_vote(self) -> StateVoteBlock {
        self.vote
    }
}

// Network message implementation
impl NetworkMessage for StateVoteBlockGossip {
    fn message_type_id() -> &'static str {
        "state.vote"
    }
}

impl ShardMessage for StateVoteBlockGossip {}

/// Proves a shard executed a transaction (2f+1 quorum). Broadcast to all participating shards.
/// Contains full state data. Once all shards' certificates collected, transaction is finalized.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct StateCertificateGossip {
    /// The state certificate being gossiped
    pub certificate: StateCertificate,
    /// Trace context for distributed tracing (empty when feature disabled).
    pub trace_context: TraceContext,
}

impl StateCertificateGossip {
    /// Create a new state certificate gossip message.
    ///
    /// Does not capture trace context. Use `with_trace_context()` to include
    /// distributed tracing information.
    pub fn new(certificate: StateCertificate) -> Self {
        Self {
            certificate,
            trace_context: TraceContext::default(),
        }
    }

    /// Create a new state certificate gossip message with trace context from current span.
    ///
    /// When `trace-propagation` feature is enabled, captures the current OpenTelemetry
    /// span context for distributed tracing across nodes.
    pub fn with_trace_context(certificate: StateCertificate) -> Self {
        Self {
            certificate,
            trace_context: TraceContext::from_current(),
        }
    }

    /// Get the inner state certificate.
    pub fn certificate(&self) -> &StateCertificate {
        &self.certificate
    }

    /// Consume and return the inner state certificate.
    pub fn into_certificate(self) -> StateCertificate {
        self.certificate
    }

    /// Get the trace context.
    pub fn trace_context(&self) -> &TraceContext {
        &self.trace_context
    }
}

// Network message implementation
impl NetworkMessage for StateCertificateGossip {
    fn message_type_id() -> &'static str {
        "state.certificate"
    }
}

impl ShardMessage for StateCertificateGossip {}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_types::{
        BlockHeight, Hash, ShardGroupId, Signature, SignerBitfield, ValidatorId,
    };

    #[test]
    fn test_state_provision_gossip() {
        let provision = StateProvision {
            transaction_hash: Hash::from_bytes(b"tx"),
            target_shard: ShardGroupId(1),
            source_shard: ShardGroupId(0),
            block_height: BlockHeight(10),
            entries: vec![],
            validator_id: ValidatorId(0),
            signature: Signature::zero(),
        };

        let msg = StateProvisionGossip::new(provision.clone());
        assert_eq!(msg.provision(), &provision);

        let extracted = msg.into_provision();
        assert_eq!(extracted, provision);
    }

    fn test_vote() -> StateVoteBlock {
        use hyperscale_types::{build_merkle_tree_with_proofs, vote_leaf_hash};

        let tx_hash = Hash::from_bytes(b"tx");
        let state_root = Hash::from_bytes(b"root");
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
            validator: ValidatorId(0),
            signature: Signature::zero(),
            vote_merkle_root: merkle_root,
            vote_merkle_proof: proofs.into_iter().next().unwrap(),
            batch_block_height: None,
        }
    }

    fn test_certificate() -> StateCertificate {
        use hyperscale_types::{build_merkle_tree_with_proofs, vote_leaf_hash};

        let tx_hash = Hash::from_bytes(b"tx");
        let outputs_merkle_root = Hash::from_bytes(b"root");
        let shard_group_id = ShardGroupId(0);
        let success = true;

        // Build merkle tree with single leaf
        let leaf_hash = vote_leaf_hash(&tx_hash, &outputs_merkle_root, shard_group_id.0, success);
        let (merkle_root, proofs) = build_merkle_tree_with_proofs(&[leaf_hash]);

        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);

        StateCertificate {
            transaction_hash: tx_hash,
            shard_group_id,
            read_nodes: vec![],
            state_writes: vec![],
            outputs_merkle_root,
            success,
            aggregated_signature: Signature::zero(),
            signers,
            voting_power: 3,
            vote_merkle_root: merkle_root,
            vote_merkle_proof: proofs.into_iter().next().unwrap(),
            batch_block_height: None,
        }
    }

    #[test]
    fn test_state_vote_block_gossip() {
        let vote = test_vote();

        let msg = StateVoteBlockGossip::new(vote.clone());
        assert_eq!(msg.vote(), &vote);

        let extracted = msg.into_vote();
        assert_eq!(extracted, vote);
    }

    #[test]
    fn test_state_certificate_gossip() {
        let cert = test_certificate();

        let msg = StateCertificateGossip::new(cert.clone());
        assert_eq!(msg.certificate(), &cert);

        let extracted = msg.into_certificate();
        assert_eq!(extracted, cert);
    }

    #[test]
    fn test_message_type_ids() {
        assert_eq!(StateProvisionGossip::message_type_id(), "state.provision");
        assert_eq!(StateVoteBlockGossip::message_type_id(), "state.vote");
        assert_eq!(
            StateCertificateGossip::message_type_id(),
            "state.certificate"
        );
    }

    #[test]
    fn test_state_provision_trace_context() {
        let provision = StateProvision {
            transaction_hash: Hash::from_bytes(b"tx"),
            target_shard: ShardGroupId(1),
            source_shard: ShardGroupId(0),
            block_height: BlockHeight(10),
            entries: vec![],
            validator_id: ValidatorId(0),
            signature: Signature::zero(),
        };

        // new() should have empty trace context
        let msg = StateProvisionGossip::new(provision.clone());
        assert!(!msg.trace_context().has_trace());

        // with_trace_context() without active span should also be empty
        let msg_with_ctx = StateProvisionGossip::with_trace_context(provision);
        // When no span is active, trace context will be empty
        assert!(!msg_with_ctx.trace_context().has_trace() || TraceContext::is_enabled());
    }

    #[test]
    fn test_state_certificate_trace_context() {
        let cert = test_certificate();

        // new() should have empty trace context
        let msg = StateCertificateGossip::new(cert.clone());
        assert!(!msg.trace_context().has_trace());

        // with_trace_context() without active span should also be empty
        let msg_with_ctx = StateCertificateGossip::with_trace_context(cert);
        assert!(!msg_with_ctx.trace_context().has_trace() || TraceContext::is_enabled());
    }
}
