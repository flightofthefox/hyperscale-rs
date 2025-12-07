//! Pending verification types for async signature verification.
//!
//! These structs track in-flight signature verifications that have been
//! delegated to the runner. When verification completes, the runner sends
//! an event back and we look up the pending state to continue processing.

use hyperscale_types::{
    BlockHeight, NodeId, RoutableTransaction, ShardGroupId, StateCertificate, StateProvision,
    StateVoteBlock,
};

/// Tracks a pending provision broadcast waiting for state fetch.
///
/// When starting cross-shard execution, we need to fetch state entries from
/// storage before we can broadcast provisions to target shards.
#[derive(Debug)]
#[allow(dead_code)]
pub struct PendingProvisionBroadcast {
    /// The transaction we're provisioning for.
    pub transaction: RoutableTransaction,
    /// Block height when the transaction was committed.
    pub block_height: BlockHeight,
    /// Target shards to broadcast to.
    pub target_shards: Vec<ShardGroupId>,
    /// Nodes we own that need to be provisioned.
    pub owned_nodes: Vec<NodeId>,
}

/// Tracks a pending provision signature verification.
///
/// When we receive a provision from another shard, we delegate signature
/// verification to the runner before processing.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PendingProvisionVerification {
    /// The provision awaiting verification.
    pub provision: StateProvision,
}

/// Tracks a pending state vote signature verification.
///
/// When we receive a state vote, we delegate signature verification to
/// the runner before counting it toward quorum.
#[derive(Debug, Clone)]
pub struct PendingStateVoteVerification {
    /// The vote awaiting verification.
    pub vote: StateVoteBlock,
    /// Voting power of the voter.
    pub voting_power: u64,
}

/// Tracks a pending state certificate signature verification.
///
/// When we receive a state certificate from another shard, we delegate
/// aggregated signature verification to the runner before accepting it.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PendingCertificateVerification {
    /// The certificate awaiting verification.
    pub certificate: StateCertificate,
}
