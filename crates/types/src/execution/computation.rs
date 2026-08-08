//! Block-derived helpers: the block's cross-shard transactions, per-target
//! provision merkle roots, and tick-leader selection.

use std::sync::Arc;

use hyperscale_hbor::to_vec as hbor_to_vec;

use crate::{
    Attempt, Hash, ShardId, TickId, TopologySnapshot, Transaction, TxHash, ValidatorId, Verifiable,
};

/// The block's transactions that reach beyond this shard, in block order.
///
/// This is what a remote shard reads a committed header for: which of the
/// block's transactions it might be party to, and so which outcomes it
/// should expect. Whether it *is* party is a question about its own state,
/// which it answers more precisely than any shard set carried here could.
///
/// Used in both block proposal (to populate `BlockHeader::cross_shard_txs`)
/// and validation (to verify the header's field).
pub fn compute_cross_shard_txs(
    local_shard: ShardId,
    topology_snapshot: &TopologySnapshot,
    transactions: &[Arc<Verifiable<Transaction>>],
) -> Vec<TxHash> {
    transactions
        .iter()
        .filter(|tx| reaches_beyond(local_shard, topology_snapshot, tx))
        .map(|tx| tx.hash())
        .collect()
}

/// Whether `tx` touches any shard other than `local_shard`.
fn reaches_beyond(
    local_shard: ShardId,
    topology_snapshot: &TopologySnapshot,
    tx: &Arc<Verifiable<Transaction>>,
) -> bool {
    !topology_snapshot.is_single_shard_transaction(tx)
        && topology_snapshot
            .all_shards_for_transaction(tx)
            .into_iter()
            .any(|s| s != local_shard)
}

/// Deterministically select the wave leader for a wave (attempt 0).
///
/// The wave leader collects execution votes, aggregates the EC, and
/// broadcasts it to local peers and remote shards. Convenience wrapper
/// for `tick_leader_at(tick_id, 0, committee)`.
#[must_use]
pub fn tick_leader(tick_id: &TickId, committee: &[ValidatorId]) -> ValidatorId {
    tick_leader_at(tick_id, Attempt::INITIAL, committee)
}

/// Deterministically select the wave leader with rotation for fallback.
///
/// Each `attempt` selects a different validator from the committee, enabling
/// leader rotation when the primary leader (attempt=0) fails. Validators
/// re-send their vote to `tick_leader_at(tick_id, attempt+1, committee)`
/// after a timeout.
///
/// Uses `Hash(encode(tick_id) ++ attempt.to_le_bytes()) % committee_size`
/// for deterministic selection. All validators compute the same result.
///
/// # Panics
///
/// Panics if `committee` is empty.
#[must_use]
pub fn tick_leader_at(
    tick_id: &TickId,
    attempt: Attempt,
    committee: &[ValidatorId],
) -> ValidatorId {
    assert!(!committee.is_empty(), "committee must not be empty");
    let mut buf = hbor_to_vec(tick_id).expect("TickId serialization should never fail");
    buf.extend_from_slice(&attempt.to_le_bytes());
    let selection_hash = Hash::from_bytes(&buf);
    let bytes = selection_hash.as_bytes();
    let index_val = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let index = usize::try_from(index_val % committee.len() as u64)
        .expect("modulo of usize len fits in usize");
    committee[index]
}
