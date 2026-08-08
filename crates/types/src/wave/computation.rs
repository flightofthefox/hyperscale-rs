//! Block-derived helpers: wave assignment, per-target provision merkle roots,
//! and wave-leader selection.

use std::collections::BTreeSet;
use std::sync::Arc;

use hyperscale_hbor::to_vec as hbor_to_vec;

use crate::{
    Attempt, BlockHeight, Hash, ShardId, TopologySnapshot, Transaction, TxHash, ValidatorId,
    Verifiable, WaveId,
};

/// The block's transactions that reach beyond this shard, in block order.
///
/// This is what a remote shard reads a committed header for: which of the
/// block's transactions it might be party to, and so which outcomes it
/// should expect. Whether it *is* party is a question about its own state,
/// which it answers more precisely than any shard set carried here could.
///
/// Used in both block proposal (to populate `BlockHeader::cross_shard_txs`)
/// and validation (to verify the header's field). Membership is the same
/// predicate [`compute_waves`] groups on, and has to stay that way: the
/// source shard creates waves from one and a remote shard arms its
/// expectations from the other, so a transaction listed here but absent
/// from every wave would leave an expectation nothing can ever fulfil.
pub fn compute_cross_shard_txs(
    local_shard: ShardId,
    topology_snapshot: &TopologySnapshot,
    transactions: &[Arc<Verifiable<Transaction>>],
) -> Vec<TxHash> {
    transactions
        .iter()
        .filter(|tx| remote_shards_for(local_shard, topology_snapshot, tx).is_some())
        .map(|tx| tx.hash())
        .collect()
}

/// The shards other than `local_shard` that `tx` touches, or `None` when it
/// touches none of them and is this shard's business alone.
fn remote_shards_for(
    local_shard: ShardId,
    topology_snapshot: &TopologySnapshot,
    tx: &Arc<Verifiable<Transaction>>,
) -> Option<BTreeSet<ShardId>> {
    if topology_snapshot.is_single_shard_transaction(tx) {
        return None;
    }
    let remote_shards: BTreeSet<ShardId> = topology_snapshot
        .all_shards_for_transaction(tx)
        .into_iter()
        .filter(|&s| s != local_shard)
        .collect();
    (!remote_shards.is_empty()).then_some(remote_shards)
}

/// Compute the set of cross-shard waves for a block's transactions.
///
/// Each transaction's remote shard set (shards it touches minus local shard)
/// defines its wave. Transactions with identical remote shard sets belong to
/// the same wave. Wave-zero (single-shard txs) is excluded.
///
/// Returns a sorted `Vec<WaveId>` with fully populated shard + height fields.
/// (Deterministic via `BTreeSet` ordering.)
/// Used in both block proposal (to populate `BlockHeader::waves`) and
/// validation (to verify the header's waves field).
pub fn compute_waves(
    local_shard: ShardId,
    topology_snapshot: &TopologySnapshot,
    block_height: BlockHeight,
    transactions: &[Arc<Verifiable<Transaction>>],
) -> Vec<WaveId> {
    let remote_shard_sets: BTreeSet<BTreeSet<ShardId>> = transactions
        .iter()
        .filter_map(|tx| remote_shards_for(local_shard, topology_snapshot, tx))
        .collect();

    remote_shard_sets
        .into_iter()
        .map(|remote_shards| WaveId::new(local_shard, block_height, remote_shards))
        .collect()
}

/// Deterministically select the wave leader for a wave (attempt 0).
///
/// The wave leader collects execution votes, aggregates the EC, and
/// broadcasts it to local peers and remote shards. Convenience wrapper
/// for `wave_leader_at(wave_id, 0, committee)`.
#[must_use]
pub fn wave_leader(wave_id: &WaveId, committee: &[ValidatorId]) -> ValidatorId {
    wave_leader_at(wave_id, Attempt::INITIAL, committee)
}

/// Deterministically select the wave leader with rotation for fallback.
///
/// Each `attempt` selects a different validator from the committee, enabling
/// leader rotation when the primary leader (attempt=0) fails. Validators
/// re-send their vote to `wave_leader_at(wave_id, attempt+1, committee)`
/// after a timeout.
///
/// Uses `Hash(encode(wave_id) ++ attempt.to_le_bytes()) % committee_size`
/// for deterministic selection. All validators compute the same result.
///
/// # Panics
///
/// Panics if `committee` is empty.
#[must_use]
pub fn wave_leader_at(
    wave_id: &WaveId,
    attempt: Attempt,
    committee: &[ValidatorId],
) -> ValidatorId {
    assert!(!committee.is_empty(), "committee must not be empty");
    let mut buf = hbor_to_vec(wave_id).expect("WaveId serialization should never fail");
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
