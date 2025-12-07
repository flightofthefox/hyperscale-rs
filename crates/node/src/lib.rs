//! Combined node state machine.
//!
//! This crate composes the BFT, execution, and mempool state machines
//! into a complete consensus node.

mod state;

pub use state::NodeIndex;
pub use state::NodeStateMachine;
