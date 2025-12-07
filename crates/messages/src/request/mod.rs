//! Request messages (point-to-point).
//!
//! These messages are sent directly to a specific peer and expect a response.

mod block;
mod sync;

pub use block::GetBlockRequest;
pub use sync::{GetBlockInventoryRequest, SyncCompleteAnnouncement};
