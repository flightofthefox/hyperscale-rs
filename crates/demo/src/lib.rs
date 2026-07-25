//! A browser-drivable session over the deterministic simulation.
//!
//! The page steps simulated time and renders the events each step returns.
//! Events are *derived* from what the runner already observes — committed
//! chain content — rather than reported by the protocol crates, so nothing
//! here can perturb the consensus it displays.

pub mod event;
mod session;

pub use event::{ShardPath, TraceEvent, TraceKind};
pub use session::Session;

#[cfg(target_arch = "wasm32")]
mod wasm;
