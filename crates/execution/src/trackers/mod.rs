//! Tracker types for cross-shard execution coordination.
//!
//! These trackers manage the state of in-flight cross-shard transactions
//! as they progress through the 2PC protocol phases.
//!
//! Note: Provision tracking has been moved to the `hyperscale-provisions` crate.
//! See `ProvisionCoordinator` for centralized provision management.

mod certificate;
mod vote;

pub use certificate::CertificateTracker;
pub use vote::VoteTracker;
