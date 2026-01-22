//! Message encoding and decoding for network transport.
//!
//! # Wire Format
//!
//! All gossip messages are SBOR-encoded then LZ4-compressed:
//!
//! ```text
//! [LZ4 compressed SBOR payload]
//! ```
//!
//! # Topic-Based Type Dispatch
//!
//! Message type is determined by the gossipsub topic, not by a field in the
//! message. This simplifies the wire format and allows efficient routing.

use super::wire;
use hyperscale_core::{Event, OutboundMessage};
use hyperscale_messages::gossip::{
    BlockHeaderGossip, BlockVoteGossip, StateCertificateBatch, StateProvisionBatch, StateVoteBatch,
    TransactionCertificateGossip, TransactionGossip,
};
use hyperscale_messages::TraceContext;
use hyperscale_types::ShardGroupId;
use thiserror::Error;

/// Errors that can occur during message encoding/decoding.
#[derive(Debug, Error)]
pub enum CodecError {
    #[error("Message too short")]
    MessageTooShort,

    #[error("SBOR decode error: {0}")]
    SborDecode(String),

    #[error("SBOR encode error: {0}")]
    SborEncode(String),

    #[error("Decompression error: {0}")]
    Decompress(String),

    #[error("Unknown topic: {0}")]
    UnknownTopic(String),
}

/// Encode an outbound message to wire format.
///
/// SBOR-encodes the message then LZ4-compresses it.
pub fn encode_message(message: &OutboundMessage) -> Result<Vec<u8>, CodecError> {
    let sbor_bytes =
        match message {
            OutboundMessage::BlockHeader(gossip) => sbor::basic_encode(gossip)
                .map_err(|e| CodecError::SborEncode(format!("{:?}", e)))?,
            OutboundMessage::BlockVote(gossip) => sbor::basic_encode(gossip)
                .map_err(|e| CodecError::SborEncode(format!("{:?}", e)))?,
            OutboundMessage::StateProvisionBatch(batch) => {
                sbor::basic_encode(batch).map_err(|e| CodecError::SborEncode(format!("{:?}", e)))?
            }
            OutboundMessage::StateVoteBatch(batch) => {
                sbor::basic_encode(batch).map_err(|e| CodecError::SborEncode(format!("{:?}", e)))?
            }
            OutboundMessage::StateCertificateBatch(batch) => {
                sbor::basic_encode(batch).map_err(|e| CodecError::SborEncode(format!("{:?}", e)))?
            }
            OutboundMessage::TransactionCertificateGossip(gossip) => sbor::basic_encode(gossip)
                .map_err(|e| CodecError::SborEncode(format!("{:?}", e)))?,
            OutboundMessage::TransactionGossip(gossip) => sbor::basic_encode(gossip.as_ref())
                .map_err(|e| CodecError::SborEncode(format!("{:?}", e)))?,
        };

    Ok(wire::compress(&sbor_bytes))
}

/// Result of decoding a message, including optional trace context.
pub struct DecodedMessage {
    /// The decoded events (batched messages produce multiple events).
    pub events: Vec<Event>,
    /// Trace context from the message, if present.
    /// Only cross-shard messages (StateProvisionBatch, StateCertificateBatch, TransactionGossip)
    /// carry trace context.
    ///
    /// Only read when `trace-propagation` feature is enabled.
    #[allow(dead_code)]
    pub trace_context: Option<TraceContext>,
}

/// Decode a message from wire format based on topic.
///
/// LZ4-decompresses then SBOR-decodes the message.
/// The topic determines the message type (topic-based dispatch).
/// Returns the decoded events along with any trace context for distributed tracing.
pub fn decode_message(
    parsed_topic: &crate::network::Topic,
    data: &[u8],
) -> Result<DecodedMessage, CodecError> {
    if data.is_empty() {
        return Err(CodecError::MessageTooShort);
    }

    // Decompress
    let payload = wire::decompress(data).map_err(|e| CodecError::Decompress(e.to_string()))?;

    let msg_type = parsed_topic.message_type();

    // Dispatch based on message type from topic
    match msg_type {
        "block.header" => {
            let gossip: BlockHeaderGossip = sbor::basic_decode(&payload)
                .map_err(|e| CodecError::SborDecode(format!("{:?}", e)))?;
            Ok(DecodedMessage {
                events: vec![Event::BlockHeaderReceived {
                    header: gossip.header,
                    retry_hashes: gossip.retry_hashes,
                    priority_hashes: gossip.priority_hashes,
                    tx_hashes: gossip.transaction_hashes,
                    cert_hashes: gossip.certificate_hashes,
                    deferred: gossip.deferred,
                    aborted: gossip.aborted,
                    commitment_proofs: gossip.commitment_proofs,
                }],
                trace_context: None,
            })
        }
        "block.vote" => {
            let gossip: BlockVoteGossip = sbor::basic_decode(&payload)
                .map_err(|e| CodecError::SborDecode(format!("{:?}", e)))?;
            Ok(DecodedMessage {
                events: vec![Event::BlockVoteReceived { vote: gossip.vote }],
                trace_context: None,
            })
        }
        "state.provision.batch" => {
            let batch: StateProvisionBatch = sbor::basic_decode(&payload)
                .map_err(|e| CodecError::SborDecode(format!("{:?}", e)))?;
            let trace_ctx = if batch.trace_context.has_trace() {
                Some(batch.trace_context.clone())
            } else {
                None
            };
            let events = batch
                .into_provisions()
                .into_iter()
                .map(|provision| Event::StateProvisionReceived { provision })
                .collect();
            Ok(DecodedMessage {
                events,
                trace_context: trace_ctx,
            })
        }
        "state.vote.batch" => {
            let batch: StateVoteBatch = sbor::basic_decode(&payload)
                .map_err(|e| CodecError::SborDecode(format!("{:?}", e)))?;
            let events = batch
                .into_votes()
                .into_iter()
                .map(|vote| Event::StateVoteReceived { vote })
                .collect();
            Ok(DecodedMessage {
                events,
                trace_context: None,
            })
        }
        "state.certificate.batch" => {
            let batch: StateCertificateBatch = sbor::basic_decode(&payload)
                .map_err(|e| CodecError::SborDecode(format!("{:?}", e)))?;
            let trace_ctx = if batch.trace_context.has_trace() {
                Some(batch.trace_context.clone())
            } else {
                None
            };
            let events = batch
                .into_certificates()
                .into_iter()
                .map(|cert| Event::StateCertificateReceived { cert })
                .collect();
            Ok(DecodedMessage {
                events,
                trace_context: trace_ctx,
            })
        }
        "transaction.gossip" => {
            let gossip: TransactionGossip = sbor::basic_decode(&payload)
                .map_err(|e| CodecError::SborDecode(format!("{:?}", e)))?;
            let trace_ctx = if gossip.trace_context.has_trace() {
                Some(gossip.trace_context)
            } else {
                None
            };
            Ok(DecodedMessage {
                events: vec![Event::TransactionGossipReceived {
                    tx: gossip.transaction,
                }],
                trace_context: trace_ctx,
            })
        }
        "transaction.certificate" => {
            let gossip: TransactionCertificateGossip = sbor::basic_decode(&payload)
                .map_err(|e| CodecError::SborDecode(format!("{:?}", e)))?;
            Ok(DecodedMessage {
                events: vec![Event::TransactionCertificateReceived {
                    certificate: gossip.into_certificate(),
                }],
                trace_context: None,
            })
        }
        _ => Err(CodecError::UnknownTopic(parsed_topic.to_string())),
    }
}

/// Get the topic for an outbound message.
pub fn topic_for_message(message: &OutboundMessage, shard: ShardGroupId) -> crate::network::Topic {
    use crate::network::Topic;

    match message {
        OutboundMessage::BlockHeader(_) => Topic::block_header(shard),
        OutboundMessage::BlockVote(_) => Topic::block_vote(shard),
        OutboundMessage::StateProvisionBatch(_) => Topic::state_provision_batch(shard),
        OutboundMessage::StateVoteBatch(_) => Topic::state_vote_batch(shard),
        OutboundMessage::StateCertificateBatch(_) => Topic::state_certificate_batch(shard),
        OutboundMessage::TransactionCertificateGossip(_) => Topic::transaction_certificate(shard),
        OutboundMessage::TransactionGossip(_) => Topic::transaction_gossip(shard),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscale_types::{
        BlockHeader, BlockHeight, BlockVote, Hash, QuorumCertificate, ValidatorId,
    };
    use std::collections::HashMap;

    fn make_block_header() -> BlockHeader {
        BlockHeader {
            height: BlockHeight(1),
            parent_hash: Hash::from_bytes(&[0u8; 32]),
            parent_qc: QuorumCertificate::genesis(),
            proposer: ValidatorId(0),
            timestamp: 0,
            round: 0,
            is_fallback: false,
            state_root: Hash::ZERO,
            state_version: 0,
        }
    }

    #[test]
    fn test_encode_decode_block_header() {
        let header = make_block_header();
        let gossip = BlockHeaderGossip {
            header: header.clone(),
            retry_hashes: vec![],
            priority_hashes: vec![],
            transaction_hashes: vec![],
            certificate_hashes: vec![],
            deferred: vec![],
            aborted: vec![],
            commitment_proofs: HashMap::new(),
        };
        let message = OutboundMessage::BlockHeader(Box::new(gossip));

        // Encode (now returns compressed data)
        let bytes = encode_message(&message).unwrap();
        assert!(!bytes.is_empty());

        // Decode with topic
        let topic_str = "hyperscale/block.header/shard-0/1.0.0";
        let topic = crate::network::Topic::parse(topic_str).unwrap();
        let decoded = decode_message(&topic, &bytes).unwrap();

        // Block headers don't carry trace context
        assert!(decoded.trace_context.is_none());
        assert_eq!(decoded.events.len(), 1);

        match &decoded.events[0] {
            Event::BlockHeaderReceived {
                header: decoded_header,
                ..
            } => {
                assert_eq!(decoded_header.height, header.height);
                assert_eq!(decoded_header.proposer, header.proposer);
            }
            _ => panic!("Expected BlockHeaderReceived"),
        }
    }

    #[test]
    fn test_encode_decode_block_vote() {
        let vote = BlockVote {
            block_hash: Hash::from_bytes(&[1u8; 32]),
            height: BlockHeight(1),
            voter: ValidatorId(0),
            round: 0,
            signature: hyperscale_types::zero_bls_signature(),
            timestamp: 0,
        };
        let gossip = BlockVoteGossip { vote: vote.clone() };
        let message = OutboundMessage::BlockVote(gossip);

        let bytes = encode_message(&message).unwrap();
        let topic_str = "hyperscale/block.vote/shard-0/1.0.0";
        let topic = crate::network::Topic::parse(topic_str).unwrap();
        let decoded = decode_message(&topic, &bytes).unwrap();

        // Block votes don't carry trace context
        assert!(decoded.trace_context.is_none());
        assert_eq!(decoded.events.len(), 1);

        match &decoded.events[0] {
            Event::BlockVoteReceived { vote: decoded_vote } => {
                assert_eq!(decoded_vote.block_hash, vote.block_hash);
                assert_eq!(decoded_vote.voter, vote.voter);
            }
            _ => panic!("Expected BlockVoteReceived"),
        }
    }

    #[test]
    fn test_invalid_compressed_data() {
        let bytes = vec![99, 1, 2, 3]; // invalid LZ4 data
        let topic = crate::network::Topic::parse("hyperscale/block.header/shard-0/1.0.0").unwrap();
        let result = decode_message(&topic, &bytes);
        assert!(matches!(result, Err(CodecError::Decompress(_))));
    }

    #[test]
    fn test_unknown_topic() {
        // Create valid compressed data for testing unknown topic
        let sbor_bytes = sbor::basic_encode(&()).unwrap();
        let bytes = wire::compress(&sbor_bytes);
        // Topic with unknown message type
        let topic = crate::network::Topic::parse("hyperscale/unknown.type/shard-0/1.0.0").unwrap();
        let result = decode_message(&topic, &bytes);
        assert!(matches!(result, Err(CodecError::UnknownTopic(_))));
    }

    #[test]
    fn test_topic_for_message() {
        let header = make_block_header();
        let gossip = BlockHeaderGossip {
            header,
            retry_hashes: vec![],
            priority_hashes: vec![],
            transaction_hashes: vec![],
            certificate_hashes: vec![],
            deferred: vec![],
            aborted: vec![],
            commitment_proofs: HashMap::new(),
        };
        let message = OutboundMessage::BlockHeader(Box::new(gossip));

        let topic = topic_for_message(&message, ShardGroupId(5));
        assert_eq!(topic.to_string(), "hyperscale/block.header/shard-5/1.0.0");
    }
}
