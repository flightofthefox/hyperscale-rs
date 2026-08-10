//! Tick-leader selection.

use hyperscale_hbor::to_vec as hbor_to_vec;

use crate::{Attempt, Hash, TickId, ValidatorId};

/// Deterministically select the tick leader for a tick (attempt 0).
///
/// The tick leader collects execution votes, aggregates the EC, and
/// broadcasts it to local peers and remote shards. Convenience wrapper
/// for `tick_leader_at(tick_id, 0, committee)`.
#[must_use]
pub fn tick_leader(tick_id: &TickId, committee: &[ValidatorId]) -> ValidatorId {
    tick_leader_at(tick_id, Attempt::INITIAL, committee)
}

/// Deterministically select the tick leader with rotation for fallback.
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
