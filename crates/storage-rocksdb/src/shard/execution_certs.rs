//! Execution certificate persistence helpers.
//!
//! Writes ECs to a column family keyed by [`hyperscale_types::TickId`],
//! plus an index from each attested transaction to the certificate
//! carrying its outcome — the key a counterpart shard actually asks by.

use hyperscale_storage::{covers_strictly_more, widest_tick_copies};
use hyperscale_types::{Block, ExecutionCertificate, Hash};
use rocksdb::{ColumnFamily, WriteBatch};

use super::column_families::{ExecutionCertsCf, TxCertIndexCf};
use super::core::RocksDbShardStorage;
use crate::typed_cf::{TypedCf, batch_put, batch_put_raw, get};

/// Append execution certificate writes for a block to an existing `WriteBatch`.
///
/// Extracts ECs from the block's finalizations and folds them into the
/// same atomic batch as JMT + block data (one fsync per block). The
/// transaction index rides the same batch, so a crash can't leave an index
/// entry pointing at a certificate that was never written.
///
/// Reads the stored copy of each tick, so the caller must hold
/// `commit_lock` — otherwise a concurrent commit could land between the
/// read and the write and take the slot back to a narrower copy.
pub fn append_block_certs_to_batch(
    storage: &RocksDbShardStorage,
    batch: &mut WriteBatch,
    block: &Block,
) {
    // Resolve the CF handles once for the whole append loop. Per-call
    // `cf_put_raw` would each invoke `storage.cf()`, re-walking the
    // name → handle map per certificate.
    let cf = storage.cf();
    let primary_cf = ExecutionCertsCf::handle(&cf);
    let index_cf = TxCertIndexCf::handle(&cf);
    // The block's own copies resolve against each other, then against
    // what is stored. A remote tick reaches this shard through whichever
    // of its own finalizations needed it, so two blocks can each carry a
    // copy of one tick — and the second is not always the wider. Writing
    // it unconditionally would replace a complete copy with a projection
    // and leave the transaction index pointing at outcomes the stored
    // copy no longer carries.
    for cert in widest_tick_copies(block).into_values() {
        let stored = get::<ExecutionCertsCf>(&*storage.db, primary_cf, cert.tick_id());
        if stored.is_some_and(|held| !covers_strictly_more(cert, &held)) {
            continue;
        }
        append_ec_to_batch(batch, primary_cf, index_cf, cert);
    }
}

fn append_ec_to_batch(
    batch: &mut WriteBatch,
    primary_cf: &ColumnFamily,
    index_cf: &ColumnFamily,
    cert: &ExecutionCertificate,
) {
    batch_put_raw::<ExecutionCertsCf>(
        batch,
        primary_cf,
        cert.tick_id(),
        cert,
        cert.cached_wire_bytes(),
    );
    for outcome in cert.tx_outcomes() {
        batch_put::<TxCertIndexCf>(
            batch,
            index_cf,
            &Hash::from(outcome.tx_hash()),
            cert.tick_id(),
        );
    }
}
