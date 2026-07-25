//! Time-domain types for consensus.
//!
//! - [`epoch_windows`]: the [`EpochWindows`](epoch_windows::EpochWindows) grid
//!   mapping weighted timestamps to epochs and detecting boundary crossings.
//! - [`limits`]: hard caps applied at admission time on peer-supplied
//!   timestamps.
//! - [`range`]: half-open [`TimestampRange`] used as a transaction validity window.
//! - [`stopwatch`]: the [`Stopwatch`](stopwatch::Stopwatch) elapsed-time probe
//!   for metrics, portable to targets without a monotonic clock.
//! - [`timeouts`]: protocol `Duration` constants — retention windows and
//!   liveness timers that every validator must enforce identically.
//! - [`timestamp`]: typed wall-clocks ([`WeightedTimestamp`], [`ProposerTimestamp`],
//!   [`LocalTimestamp`]) with distinct trust guarantees.

pub mod epoch_windows;
pub mod limits;
pub mod range;
pub mod stopwatch;
pub mod timeouts;
pub mod timestamp;
