//! Core types for Hyperscale consensus.
//!
//! This crate provides the foundational types for the consensus architecture:
//!
//! - [`Event`]: All possible inputs to the state machine
//! - [`Action`]: All possible outputs from the state machine
//! - [`EventPriority`]: Ordering priority for events at the same timestamp
//! - [`StateMachine`]: The trait that all state machines implement
//!
//! # Architecture
//!
//! The core is built on a simple event-driven model:
//!
//! ```text
//! Events → StateMachine::handle() → Actions
//! ```
//!
//! The state machine is:
//! - **Synchronous**: No async, no .await
//! - **Deterministic**: Same state + event = same actions
//! - **Pure-ish**: Mutates self, but performs no I/O
//!
//! All I/O is handled by the runner (simulation or production) which:
//! 1. Delivers events to the state machine
//! 2. Executes the returned actions
//! 3. Converts action results back into events

mod action;
mod event;
mod message;
mod request;
mod traits;

pub use action::{Action, TransactionStatus};
pub use event::{Event, EventPriority};
pub use message::OutboundMessage;
pub use request::RequestId;
pub use traits::{StateMachine, SubStateMachine};

/// Type alias for timer identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimerId {
    /// Block proposal timer
    Proposal,
    /// View change timeout timer
    ViewChange,
    /// Periodic cleanup timer
    Cleanup,
}
