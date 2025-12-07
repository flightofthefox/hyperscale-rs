//! Outbound message types for network communication.

use hyperscale_messages::{
    BlockHeaderGossip, BlockVoteGossip, StateCertificateGossip, StateProvisionGossip,
    StateVoteBlockGossip, TransactionGossip, ViewChangeCertificateGossip, ViewChangeVoteGossip,
};

/// Outbound network messages.
///
/// These are the messages that a node can send to other nodes.
/// The runner handles the actual network I/O.
#[derive(Debug, Clone)]
pub enum OutboundMessage {
    // ═══════════════════════════════════════════════════════════════════════
    // BFT Messages
    // ═══════════════════════════════════════════════════════════════════════
    /// Block header announcement.
    BlockHeader(BlockHeaderGossip),

    /// Vote on a block header.
    BlockVote(BlockVoteGossip),

    /// Vote to change view (leader timeout).
    ViewChangeVote(ViewChangeVoteGossip),

    /// Certificate proving view change quorum.
    ViewChangeCertificate(ViewChangeCertificateGossip),

    // ═══════════════════════════════════════════════════════════════════════
    // Execution Messages
    // ═══════════════════════════════════════════════════════════════════════
    /// State provision for cross-shard execution.
    StateProvision(StateProvisionGossip),

    /// Vote on execution result.
    StateVoteBlock(StateVoteBlockGossip),

    /// Certificate proving execution quorum.
    StateCertificate(StateCertificateGossip),

    // ═══════════════════════════════════════════════════════════════════════
    // Mempool Messages
    // ═══════════════════════════════════════════════════════════════════════
    /// Transaction gossip.
    TransactionGossip(Box<TransactionGossip>),
}

impl OutboundMessage {
    /// Get a human-readable name for this message type.
    pub fn type_name(&self) -> &'static str {
        match self {
            OutboundMessage::BlockHeader(_) => "BlockHeader",
            OutboundMessage::BlockVote(_) => "BlockVote",
            OutboundMessage::ViewChangeVote(_) => "ViewChangeVote",
            OutboundMessage::ViewChangeCertificate(_) => "ViewChangeCertificate",
            OutboundMessage::StateProvision(_) => "StateProvision",
            OutboundMessage::StateVoteBlock(_) => "StateVoteBlock",
            OutboundMessage::StateCertificate(_) => "StateCertificate",
            OutboundMessage::TransactionGossip(_) => "TransactionGossip",
        }
    }

    /// Check if this is a BFT consensus message.
    pub fn is_bft(&self) -> bool {
        matches!(
            self,
            OutboundMessage::BlockHeader(_)
                | OutboundMessage::BlockVote(_)
                | OutboundMessage::ViewChangeVote(_)
                | OutboundMessage::ViewChangeCertificate(_)
        )
    }

    /// Check if this is an execution message.
    pub fn is_execution(&self) -> bool {
        matches!(
            self,
            OutboundMessage::StateProvision(_)
                | OutboundMessage::StateVoteBlock(_)
                | OutboundMessage::StateCertificate(_)
        )
    }

    /// Check if this is a mempool message.
    pub fn is_mempool(&self) -> bool {
        matches!(self, OutboundMessage::TransactionGossip(_))
    }
}
