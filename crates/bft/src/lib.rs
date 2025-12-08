//! BFT consensus state machine.
//!
//! This crate provides a synchronous BFT consensus implementation
//! that can be used for both simulation and production.
//!
//! # Architecture
//!
//! The BFT state machine processes events synchronously:
//!
//! - `Event::ProposalTimer` → Build and broadcast block if we're the proposer
//! - `Event::BlockHeaderReceived` → Validate header, assemble block, vote
//! - `Event::BlockVoteReceived` → Collect votes, form QC when quorum reached
//! - `Event::QuorumCertificateFormed` → Update chain state, commit if ready
//! - `Event::ViewChangeTimer` → Initiate view change if no progress
//!
//! All I/O is performed by the runner via returned `Action`s.

mod config;
mod pending;
mod state;
mod view_change;
mod vote_set;

pub use config::BftConfig;
pub use pending::PendingBlock;
pub use state::{BftState, RecoveredState};
pub use view_change::ViewChangeState;
pub use vote_set::VoteSet;
