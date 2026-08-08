//! Deterministic execution state machine.
//!
//! Implements the transaction execution layer as a pure, synchronous state
//! machine. The [`ExecutionCoordinator`] consumes `ProtocolEvent`s from the
//! shard consensus layer and the network, drives the wave/EC lifecycle, and emits
//! `Action`s for asynchronous work (signature verification, state provisioning,
//! transaction execution against substate).
//!
//! # Wave lifecycle
//!
//! Cross-shard transactions are grouped into deterministic *waves*. Each
//! wave is provisioned by the source shards (state entries with JMT
//! proofs), executed once provisions are complete, and certified by an
//! `ExecutionCertificate` aggregating execution votes from the committee.
//! Resolved waves are finalized into a `Finalization` receipt that lives
//! in the corresponding block.
//!
pub mod action_handlers;
pub mod wave_state;

mod coordinator;
mod early_arrivals;
mod exec_cert_store;
mod expected_certs;
mod finalizations;
mod lookups;
mod outbound_certs;
mod provisional;
mod provisioning;
mod unresolved;
mod vote_tracker;
mod waves;

pub use coordinator::{CompletionData, ExecutionCoordinator, ExecutionMemoryStats};
pub use exec_cert_store::ExecCertStore;
pub use finalizations::FinalizationStore;
pub use lookups::provision_request;
pub use vote_tracker::VoteTracker;
pub use wave_state::WaveState;
