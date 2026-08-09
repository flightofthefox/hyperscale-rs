//! Where a restarted replica's execution has to resume from.
//!
//! What a shard has committed and not yet resolved is a fold over its own
//! blocks, so a replica that lost its execution state can recover it by
//! replaying them. That is the whole point of keeping the account on the
//! chain rather than in tick state: a shard whose replicas all restarted
//! can still name what it committed and never finished, and therefore
//! still finish it or abort it.
//!
//! The walk belongs here, because this is what holds the blocks and knows
//! how far back the retention window reaches. What the walk *means* does
//! not: composition needs a topology and a provisioning tracker, so the
//! replay itself is the coordinator's, and this hands back where it
//! starts.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use hyperscale_types::{
    BlockHeight, CertifiedBlock, MAX_VALIDITY_RANGE, Provisions, RETENTION_HORIZON, TxHash,
    Verifiable, Verified, WeightedTimestamp,
};

use super::chain_reader::ShardChainReader;

/// How far back a rebuild has to read.
///
/// A transaction committed at time `T` states a validity end at most
/// `MAX_VALIDITY_RANGE` beyond it, and stops being anyone's business a
/// `RETENTION_HORIZON` past that. Blocks older than this contribute
/// nothing a rebuild would keep.
const FOLD_WINDOW: Duration = MAX_VALIDITY_RANGE.saturating_add(RETENTION_HORIZON);

/// The lowest height committing a transaction the chain still owes an
/// outcome for — where a replay has to start to rebuild everything
/// execution was tracking.
///
/// Every tick with an unresolved member sits at or above it: a tick
/// below would need a member committed below it, which would have been
/// this floor instead. So replaying from here reaches every tick whose
/// output has not settled, and no earlier one.
///
/// `None` when nothing is owed, where there is nothing to replay.
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
pub fn unresolved_replay_floor<R: ShardChainReader + ?Sized>(
    reader: &R,
    committed_height: BlockHeight,
    committed_ts: WeightedTimestamp,
) -> Option<BlockHeight> {
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

    let mut unresolved: BTreeMap<TxHash, BlockHeight> = BTreeMap::new();
    let mut height = oldest;
    loop {
        if let Some(certified) = reader.get_block(height) {
            let block = certified.block();
            for tx in block.transactions().iter() {
                unresolved.insert(tx.hash(), height);
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

    unresolved.into_values().min()
}

/// Where a restart resumes execution: the blocks to replay, and the
/// clock the first of them carries forward.
#[derive(Debug, Clone, Default)]
pub struct ReplayWindow {
    /// Every block from [`unresolved_replay_floor`] through the committed
    /// tip, each with the provision bundles it carried reattached. Empty
    /// when nothing is owed an outcome.
    pub blocks: Vec<Verified<CertifiedBlock>>,
    /// The parent-QC weighted timestamp of the block *below* the first
    /// one replayed — the clock execution resumes at, so the block above
    /// it stays on the exact carry path and classifies its ticks under
    /// the window they committed in.
    ///
    /// `None` when that block is not held: the floor is the chain's
    /// first block, or its predecessor has aged out. The first block
    /// replayed then classifies under its own anchor, the same fallback
    /// a chain with no history behind it takes.
    pub anchor_wt: Option<WeightedTimestamp>,
}

/// Build the window a restart replays.
///
/// The bodies sealing dropped come back through
/// [`ShardChainReader::provisions_at`], lifted under the same trust as
/// the commit path's: they reached storage inside a block this shard
/// committed, and our own disk is not a weaker source than the peer's
/// block that put them there.
///
/// A hole anywhere in the range yields an empty window rather than a
/// partial one: the folds above a missing block would sit on a baseline
/// that block was supposed to have contributed to.
#[must_use]
pub fn replay_window<R: ShardChainReader + ?Sized>(
    reader: &R,
    committed_height: BlockHeight,
    committed_ts: WeightedTimestamp,
) -> ReplayWindow {
    let Some(floor) = unresolved_replay_floor(reader, committed_height, committed_ts) else {
        return ReplayWindow::default();
    };
    let anchor_wt = floor
        .prev()
        .and_then(|below| reader.get_block(below))
        .map(|certified| certified.block().header().parent_qc().weighted_timestamp());

    let mut blocks = Vec::new();
    let mut height = floor;
    loop {
        match reader.get_block(height) {
            Some(certified) => blocks.push(rehydrate(certified, reader.provisions_at(height))),
            None => return ReplayWindow::default(),
        }
        if height >= committed_height {
            break;
        }
        height = height.next();
    }
    ReplayWindow { blocks, anchor_wt }
}

/// Put a stored block's provision bundles back on it.
fn rehydrate(
    certified: Verified<CertifiedBlock>,
    provisions: Vec<Arc<Verifiable<Provisions>>>,
) -> Verified<CertifiedBlock> {
    if provisions.is_empty() {
        return certified;
    }
    let (block, qc) = certified.into_inner().into_parts();
    // Sealing keeps the header, and the header is what the hash and the
    // QC pairing are over, so reattaching the bodies cannot break it.
    let live = block.into_live(Arc::new(provisions));
    Verified::<CertifiedBlock>::from_persisted(CertifiedBlock::new_unchecked(live, qc))
}
