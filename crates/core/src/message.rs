//! Outbound message types for network communication.

use crate::Event;
use hyperscale_messages::{
    BlockHeaderGossip, BlockVoteGossip, StateCertificateGossip, StateProvisionGossip,
    StateVoteBlockGossip, TraceContext, TransactionGossip, ViewChangeCertificateGossip,
    ViewChangeVoteGossip,
};
use sbor::prelude::*;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

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

    /// Inject trace context into cross-shard messages for distributed tracing.
    ///
    /// Only affects messages that carry trace context:
    /// - `StateProvision` (cross-shard state)
    /// - `StateCertificate` (cross-shard 2PC completion)
    /// - `TransactionGossip` (transaction propagation)
    ///
    /// Other message types (BFT consensus, state votes) are unaffected.
    ///
    /// When `trace-propagation` feature is disabled in the messages crate,
    /// this sets an empty trace context (no-op).
    pub fn inject_trace_context(&mut self) {
        let ctx = TraceContext::from_current();
        match self {
            OutboundMessage::StateProvision(gossip) => {
                gossip.trace_context = ctx;
            }
            OutboundMessage::StateCertificate(gossip) => {
                gossip.trace_context = ctx;
            }
            OutboundMessage::TransactionGossip(gossip) => {
                gossip.trace_context = ctx;
            }
            // BFT consensus and state vote messages don't carry trace context
            OutboundMessage::BlockHeader(_)
            | OutboundMessage::BlockVote(_)
            | OutboundMessage::ViewChangeVote(_)
            | OutboundMessage::ViewChangeCertificate(_)
            | OutboundMessage::StateVoteBlock(_) => {}
        }
    }

    /// Estimate the encoded size of this message in bytes.
    ///
    /// This uses SBOR encoding to get an accurate size estimate for bandwidth analysis.
    /// Returns (payload_size, wire_size) where wire_size includes framing overhead.
    pub fn encoded_size(&self) -> (usize, usize) {
        let payload_size = match self {
            OutboundMessage::BlockHeader(gossip) => {
                basic_encode(gossip).map(|v| v.len()).unwrap_or(0)
            }
            OutboundMessage::BlockVote(gossip) => {
                basic_encode(gossip).map(|v| v.len()).unwrap_or(0)
            }
            OutboundMessage::ViewChangeVote(gossip) => {
                basic_encode(gossip).map(|v| v.len()).unwrap_or(0)
            }
            OutboundMessage::ViewChangeCertificate(gossip) => {
                basic_encode(gossip).map(|v| v.len()).unwrap_or(0)
            }
            OutboundMessage::StateProvision(gossip) => {
                basic_encode(gossip).map(|v| v.len()).unwrap_or(0)
            }
            OutboundMessage::StateVoteBlock(gossip) => {
                basic_encode(gossip).map(|v| v.len()).unwrap_or(0)
            }
            OutboundMessage::StateCertificate(gossip) => {
                basic_encode(gossip).map(|v| v.len()).unwrap_or(0)
            }
            OutboundMessage::TransactionGossip(gossip) => {
                basic_encode(gossip.as_ref()).map(|v| v.len()).unwrap_or(0)
            }
        };

        // Add framing overhead estimate:
        // - 4 bytes length prefix
        // - 1 byte message type tag
        // - ~10 bytes protocol overhead
        let wire_size = payload_size + 15;

        (payload_size, wire_size)
    }

    /// Compute a hash of the message content for deduplication.
    ///
    /// This matches the libp2p gossipsub `message_id_fn` approach: hash the
    /// encoded message data using `DefaultHasher`. Two identical messages
    /// will produce the same hash, allowing deduplication.
    pub fn message_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match self {
            OutboundMessage::BlockHeader(g) => {
                if let Ok(encoded) = basic_encode(g) {
                    encoded.hash(&mut hasher);
                }
            }
            OutboundMessage::BlockVote(g) => {
                if let Ok(encoded) = basic_encode(g) {
                    encoded.hash(&mut hasher);
                }
            }
            OutboundMessage::ViewChangeVote(g) => {
                if let Ok(encoded) = basic_encode(g) {
                    encoded.hash(&mut hasher);
                }
            }
            OutboundMessage::ViewChangeCertificate(g) => {
                if let Ok(encoded) = basic_encode(g) {
                    encoded.hash(&mut hasher);
                }
            }
            OutboundMessage::StateProvision(g) => {
                if let Ok(encoded) = basic_encode(g) {
                    encoded.hash(&mut hasher);
                }
            }
            OutboundMessage::StateVoteBlock(g) => {
                if let Ok(encoded) = basic_encode(g) {
                    encoded.hash(&mut hasher);
                }
            }
            OutboundMessage::StateCertificate(g) => {
                if let Ok(encoded) = basic_encode(g) {
                    encoded.hash(&mut hasher);
                }
            }
            OutboundMessage::TransactionGossip(g) => {
                if let Ok(encoded) = basic_encode(g.as_ref()) {
                    encoded.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Convert an outbound message to the corresponding inbound event.
    ///
    /// This is used by both the deterministic and parallel simulators
    /// to handle received messages uniformly.
    pub fn to_received_event(&self) -> Event {
        match self {
            OutboundMessage::BlockHeader(gossip) => Event::BlockHeaderReceived {
                header: gossip.header.clone(),
                tx_hashes: gossip.transaction_hashes.clone(),
                cert_hashes: gossip.certificate_hashes.clone(),
                deferred: gossip.deferred.clone(),
                aborted: gossip.aborted.clone(),
            },
            OutboundMessage::BlockVote(gossip) => Event::BlockVoteReceived {
                vote: gossip.vote.clone(),
            },
            OutboundMessage::ViewChangeVote(gossip) => Event::ViewChangeVoteReceived {
                vote: gossip.vote.clone(),
            },
            OutboundMessage::ViewChangeCertificate(gossip) => {
                Event::ViewChangeCertificateReceived {
                    cert: gossip.certificate.clone(),
                }
            }
            OutboundMessage::StateProvision(gossip) => Event::StateProvisionReceived {
                provision: gossip.provision.clone(),
            },
            OutboundMessage::StateVoteBlock(gossip) => Event::StateVoteReceived {
                vote: gossip.vote.clone(),
            },
            OutboundMessage::StateCertificate(gossip) => Event::StateCertificateReceived {
                cert: gossip.certificate.clone(),
            },
            OutboundMessage::TransactionGossip(gossip) => Event::TransactionGossipReceived {
                tx: Arc::clone(&gossip.transaction),
            },
        }
    }
}
