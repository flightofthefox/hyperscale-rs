//! The engine's event record, re-exported from the shared vocabulary.
//!
//! The type, its caps, and its docs live in `hyperscale-vm-types`: the
//! same constants bound the kernel's emission and this workspace's
//! decode, so the two cannot drift. What binds here is the leaf hash —
//! the protocol hash is this workspace's to name.

pub use hyperscale_vm_types::{
    Event, MAX_ERROR_CODES, MAX_EVENT_PAYLOAD_BYTES, MAX_EVENT_TYPES, MAX_EVENTS_PER_TX,
};

use crate::Hash;

// The re-exports above are this module's surface; the caps ride along so
// every consumer names them through one path.

/// The event's leaf hash in a transaction's event root.
///
/// Unambiguous without framing: the emitter and type are fixed width, and
/// the payload is the only variable part, at the end.
pub trait EventExt {
    /// This event's leaf hash.
    fn hash(&self) -> Hash;
}

impl EventExt for Event {
    fn hash(&self) -> Hash {
        Hash::from_parts(&[
            &self.emitter.to_bytes(),
            &self.event_type.to_le_bytes(),
            &self.payload,
        ])
    }
}
