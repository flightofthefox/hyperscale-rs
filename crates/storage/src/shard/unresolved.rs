//! Rebuilding the unresolved-transaction ledger from the committed chain.
//!
//! What a shard has committed and not yet resolved is a fold over its own
//! blocks, so a replica that lost its execution state can recover it by
//! replaying them. That is the whole point of keeping the account on the
//! chain rather than in wave state: a shard whose replicas all restarted
//! can still name what it committed and never finished, and therefore
//! still finish it or abort it.

use std::collections::BTreeMap;
use std::time::Duration;

use hyperscale_types::{
    BlockHeight, MAX_VALIDITY_RANGE, RETENTION_HORIZON, TxHash, WeightedTimestamp,
};

use super::chain_reader::ShardChainReader;

/// How far back a rebuild has to read.
///
/// A transaction committed at time `T` states a validity end at most
/// `MAX_VALIDITY_RANGE` beyond it, and stops being anyone's business a
/// `RETENTION_HORIZON` past that. Blocks older than this contribute
/// nothing a rebuild would keep.
const FOLD_WINDOW: Duration = MAX_VALIDITY_RANGE.saturating_add(RETENTION_HORIZON);

/// Replay the committed chain into the transactions still owed an
/// outcome, each with the validity end its deadline derives from and the
/// work its committing block reserved.
///
/// Reads forward from the oldest block in the window, so a transaction
/// and the finalization resolving it are seen in the order they
/// committed — the reverse would drop a release whose registration had
/// not happened yet.
///
/// Bounded by what the reader holds: a snap-synced replica has no blocks
/// below its anchor and recovers only what committed above it, which is
/// the same limit that applies to everything else it cannot see.
#[must_use]
pub fn fold_unresolved_txs<R: ShardChainReader + ?Sized>(
    reader: &R,
    committed_height: BlockHeight,
    committed_ts: WeightedTimestamp,
) -> Vec<(TxHash, WeightedTimestamp, u64)> {
    let cutoff = committed_ts.minus(FOLD_WINDOW);

    // Walk back to the window's edge, then fold forward from there.
    let mut oldest = committed_height;
    while let Some(previous) = oldest.prev() {
        match reader.get_block(previous) {
            Some(block) if block.block().header().parent_qc().weighted_timestamp() >= cutoff => {
                oldest = previous;
            }
            _ => break,
        }
    }

    let mut unresolved: BTreeMap<TxHash, (WeightedTimestamp, u64)> = BTreeMap::new();
    let mut height = oldest;
    loop {
        if let Some(certified) = reader.get_block(height) {
            let block = certified.block();
            for tx in block.transactions().iter() {
                unresolved.insert(
                    tx.hash(),
                    (tx.validity_range().end_timestamp_exclusive, tx.work()),
                );
            }
            for finalization in block.certificates().iter() {
                for tx_hash in finalization.tx_hashes() {
                    unresolved.remove(&tx_hash);
                }
            }
        }
        if height >= committed_height {
            break;
        }
        height = height.next();
    }
    unresolved
        .into_iter()
        .map(|(tx_hash, (validity_end, work))| (tx_hash, validity_end, work))
        .collect()
}
