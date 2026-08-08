//! Execution certificate persistence helpers.
//!
//! Writes ECs to a column family keyed by [`hyperscale_types::WaveId`],
//! plus an index from each attested transaction to the certificate
//! carrying its outcome — the key a counterpart shard actually asks by.

use hyperscale_types::{Block, ExecutionCertificate, Hash};
use rocksdb::{ColumnFamily, WriteBatch};

use super::column_families::{ExecutionCertsCf, TxCertIndexCf};
use super::core::RocksDbShardStorage;
use crate::typed_cf::{TypedCf, batch_put, batch_put_raw};

/// Append execution certificate writes for a block to an existing `WriteBatch`.
///
/// Extracts ECs from the block's wave certificates and folds them into the
/// same atomic batch as JMT + block data (one fsync per block). The
/// transaction index rides the same batch, so a crash can't leave an index
/// entry pointing at a certificate that was never written.
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
    for fw in block.certificates().iter() {
        for ec in fw.certificate().execution_certificates() {
            append_ec_to_batch(batch, primary_cf, index_cf, ec.as_unverified());
        }
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
        cert.wave_id(),
        cert,
        cert.cached_wire_bytes(),
    );
    for outcome in cert.tx_outcomes() {
        batch_put::<TxCertIndexCf>(
            batch,
            index_cf,
            &Hash::from(outcome.tx_hash()),
            cert.wave_id(),
        );
    }
}
