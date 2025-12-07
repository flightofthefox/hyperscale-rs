//! Response messages (point-to-point).
//!
//! These messages are sent in reply to request messages.

mod block;
mod sync;

pub use block::GetBlockResponse;
pub use sync::GetBlockInventoryResponse;
