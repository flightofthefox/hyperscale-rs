//! Network messages for the consensus protocol.

pub mod gossip;
pub mod request;
pub mod response;

// Re-export commonly used types
pub use gossip::{
    BlockHeaderGossip, BlockVoteGossip, StateCertificateGossip, StateProvisionGossip,
    StateVoteBlockGossip, TransactionGossip, ViewChangeCertificateGossip, ViewChangeVote,
    ViewChangeVoteGossip,
};
pub use request::{GetBlockInventoryRequest, GetBlockRequest, SyncCompleteAnnouncement};
pub use response::{GetBlockInventoryResponse, GetBlockResponse};
