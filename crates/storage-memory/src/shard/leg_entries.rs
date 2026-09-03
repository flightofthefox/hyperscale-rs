//! Durable leg entries — `LegEntryStore` for [`SimShardStorage`].
//!
//! Rows live exactly as long as the store handle, which is what a
//! simulated restart preserves: dropping a coordinator and rebuilding it
//! over the same `SimShardStorage` models a crash that loses process
//! memory but keeps disk.

use hyperscale_storage::LegEntryStore;
use hyperscale_storage::lock_recover::{read_or_recover, write_or_recover};
use hyperscale_types::{LegEntry, TxHash};

use super::core::SimShardStorage;

impl LegEntryStore for SimShardStorage {
    fn persist_leg_entries(&self, entries: &[LegEntry], released: &[TxHash]) {
        let mut c = write_or_recover(&self.consensus);
        for tx_hash in released {
            c.leg_entries.remove(tx_hash);
        }
        for entry in entries {
            c.leg_entries.insert(entry.tx_hash, entry.clone());
        }
    }

    fn leg_entries(&self) -> Vec<LegEntry> {
        read_or_recover(&self.consensus)
            .leg_entries
            .values()
            .cloned()
            .collect()
    }
}
