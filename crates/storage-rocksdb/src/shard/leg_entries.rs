//! Durable leg entries — `LegEntryStore` for [`RocksDbShardStorage`].

use hyperscale_storage::LegEntryStore;
use hyperscale_types::{Hash, LegEntry, TxHash};
use rocksdb::WriteBatch;

use super::column_families::{CfHandles, LegEntriesCf};
use super::core::RocksDbShardStorage;
use crate::typed_cf::{TypedCf, batch_delete, batch_put, iter_all};

impl LegEntryStore for RocksDbShardStorage {
    fn persist_leg_entries(&self, entries: &[LegEntry], released: &[TxHash]) {
        if entries.is_empty() && released.is_empty() {
            return;
        }
        let cf = CfHandles::resolve(&self.db);
        let leg_entries_cf = LegEntriesCf::handle(&cf);
        let mut batch = WriteBatch::default();
        for tx_hash in released {
            batch_delete::<LegEntriesCf>(&mut batch, leg_entries_cf, &Hash::from(*tx_hash));
        }
        for entry in entries {
            batch_put::<LegEntriesCf>(
                &mut batch,
                leg_entries_cf,
                &Hash::from(entry.tx_hash),
                entry,
            );
        }
        // Unsynced, and durable on the next block's commit: the WAL is
        // one log, so a block flush's fsync covers every write that
        // reached it first. A row lost before that fsync is lost with
        // the block that caused it, and a restart's fold rebuilds both
        // — which is the same reason this needs no atomicity with the
        // block. A vote register cannot say that, which is why it syncs
        // and this does not.
        self.db
            .write(batch)
            .expect("persist_leg_entries: write failed");
    }

    fn leg_entries(&self) -> Vec<LegEntry> {
        let cf = CfHandles::resolve(&self.db);
        let leg_entries_cf = LegEntriesCf::handle(&cf);
        iter_all::<LegEntriesCf>(&self.db, leg_entries_cf)
            .map(|(_, entry)| entry)
            .collect()
    }
}
