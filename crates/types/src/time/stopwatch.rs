//! Monotonic elapsed-time probe for metrics, portable to wasm32.
//!
//! `std::time::Instant::now()` traps on `wasm32-unknown-unknown`, which has no
//! monotonic clock. Timing sites that only feed histograms and gauges use
//! [`Stopwatch`] so the same code links and runs there.

use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

/// Measures elapsed wall time for metrics only — never consensus-visible, so
/// its reading may differ across nodes. Reads zero on wasm32, which has no
/// monotonic clock.
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch {
    #[cfg(not(target_arch = "wasm32"))]
    started_at: Instant,
}

impl Stopwatch {
    /// Samples the platform monotonic clock; a no-op on wasm32.
    #[must_use]
    pub fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            started_at: Instant::now(),
        }
    }

    /// Time since [`Stopwatch::start`], or [`Duration::ZERO`] on wasm32.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.started_at.elapsed()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Duration::ZERO
        }
    }
}
