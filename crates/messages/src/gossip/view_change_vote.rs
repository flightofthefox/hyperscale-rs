//! ViewChangeVote gossip message.

use hyperscale_types::{NetworkMessage, ShardMessage};
use sbor::prelude::BasicSbor;

// Re-export ViewChangeVote from types for convenience
pub use hyperscale_types::ViewChangeVote;

/// Vote to trigger a view change. 2f+1 votes for same (height, round) advance all validators.
#[derive(Debug, Clone, PartialEq, Eq, BasicSbor)]
pub struct ViewChangeVoteGossip {
    /// The view change vote being gossiped
    pub vote: ViewChangeVote,
}

impl ViewChangeVoteGossip {
    /// Create a new view change vote gossip message.
    pub fn new(vote: ViewChangeVote) -> Self {
        Self { vote }
    }

    /// Get the inner view change vote.
    pub fn vote(&self) -> &ViewChangeVote {
        &self.vote
    }

    /// Consume and return the inner view change vote.
    pub fn into_vote(self) -> ViewChangeVote {
        self.vote
    }
}

impl NetworkMessage for ViewChangeVoteGossip {
    fn message_type_id() -> &'static str {
        "view_change.vote"
    }
}

impl ShardMessage for ViewChangeVoteGossip {}
