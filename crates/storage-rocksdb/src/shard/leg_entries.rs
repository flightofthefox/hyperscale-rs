//! Durable leg entries — `LegEntryStore` for [`RocksDbShardStorage`].

use hyperscale_storage::LegEntryStore;
use hyperscale_types::{Hash, LegEntry, TxHash};
use rocksdb::{WriteBatch, WriteOptions};

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
        // Synced, as the vote registers are: the store is what carries an
        // entry past the window the replay reaches, so a row that reached
        // only the page cache is a row a crash can lose after the block
        // that would rebuild it has aged out.
        let mut opts = WriteOptions::default();
        opts.set_sync(true);
        self.db
            .write_opt(batch, &opts)
            .expect("persist_leg_entries: synced write failed");
    }

    fn leg_entries(&self) -> Vec<LegEntry> {
        let cf = CfHandles::resolve(&self.db);
        let leg_entries_cf = LegEntriesCf::handle(&cf);
        iter_all::<LegEntriesCf>(&self.db, leg_entries_cf)
            .map(|(_, entry)| entry)
            .collect()
    }
}
