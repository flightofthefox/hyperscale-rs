//! Rebuilding execution's account of what is in flight from the
//! committed chain.
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
//! not: participants need a topology and deadlines need the rule the live
//! path applies, so both are the coordinator's, and this hands back the
//! committed content they read.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use hyperscale_types::{
    BlockHeight, MAX_VALIDITY_RANGE, RETENTION_HORIZON, RevealChain, Transaction, TxHash,
    Verifiable, WeightedTimestamp,
};

use super::chain_reader::ShardChainReader;

/// How far back a rebuild has to read.
///
/// A transaction committed at time `T` states a validity end at most
/// `MAX_VALIDITY_RANGE` beyond it, and stops being anyone's business a
/// `RETENTION_HORIZON` past that. Blocks older than this contribute
/// nothing a rebuild would keep.
const FOLD_WINDOW: Duration = MAX_VALIDITY_RANGE.saturating_add(RETENTION_HORIZON);

/// One committed transaction still owed an outcome, as the block that
/// committed it recorded it.
///
/// The anchors travel with the transaction because they are its
/// committing block's and not the tip's: a member executes under the
/// clock and the draw it was admitted against, however many blocks later
/// it runs.
#[derive(Debug, Clone)]
pub struct RecoveredTx {
    /// The transaction itself, from the block that committed it.
    pub transaction: Arc<Verifiable<Transaction>>,
    /// That block's weighted timestamp — its parent QC's, which is what
    /// the live path carries as the commit clock.
    pub committed_ts: WeightedTimestamp,
    /// That block's reveal chain, the randomness anchor under the same
    /// rule.
    pub committed_reveal: RevealChain,
    /// The anchor its committee resolves from: the *previous* block's
    /// weighted timestamp, which is what classified it when it committed.
    /// A reshape inside the recovery window moves the shard set, so
    /// resolving participants against the tip instead would assign a
    /// transaction shards its own block never named.
    pub committee_anchor_ts: WeightedTimestamp,
}

/// Replay the committed chain into the transactions still owed an
/// outcome.
///
/// Transactions only. A block is stored sealed, and sealing keeps its
/// provisions' hashes rather than their contents, so the bundles a
/// cross-shard leg waits on are not on the chain to replay — a recovered
/// leg waits for the fetch path to supply them again, and reaches its
/// deadline if it never does.
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
) -> Vec<RecoveredTx> {
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

    let mut unresolved: BTreeMap<TxHash, RecoveredTx> = BTreeMap::new();
    // The anchor a block's committee resolves from is the previous
    // block's timestamp, so the walk carries it forward. The first block
    // read has no predecessor in the window and stands in with its own,
    // which is exact except across the one epoch cut it might straddle.
    let mut previous_ts: Option<WeightedTimestamp> = None;
    let mut height = oldest;
    loop {
        if let Some(certified) = reader.get_block(height) {
            let block = certified.block();
            let block_ts = block.header().parent_qc().weighted_timestamp();
            let committee_anchor_ts = previous_ts.unwrap_or(block_ts);
            for tx in block.transactions().iter() {
                unresolved.insert(
                    tx.hash(),
                    RecoveredTx {
                        transaction: Arc::clone(tx),
                        committed_ts: block_ts,
                        committed_reveal: block.header().reveal_chain(),
                        committee_anchor_ts,
                    },
                );
            }
            for finalization in block.certificates().iter() {
                for tx_hash in finalization.tx_hashes() {
                    unresolved.remove(&tx_hash);
                }
            }
            previous_ts = Some(block_ts);
        }
        if height >= committed_height {
            break;
        }
        height = height.next();
    }

    unresolved.into_values().collect()
}
