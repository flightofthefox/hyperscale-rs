//! Which copy of a tick's execution certificate a store keeps.
//!
//! A certificate carries the outcomes naming its holder, so one tick can
//! reach a shard as more than one copy — a broadcast and a narrower fetch
//! answer for the same batch. Stores key certificates by tick and index
//! the transactions those certificates attest, so the choice of copy
//! decides which transactions the store can answer for. Both backends
//! resolve it here.

use std::collections::BTreeMap;

use hyperscale_types::{Block, ExecutionCertificate, TickId};

/// The widest copy of each tick the block's finalizations carry.
///
/// Resolves the copies within one block. A store folds the result against
/// what it already holds with [`covers_strictly_more`].
#[must_use]
pub fn widest_tick_copies(block: &Block) -> BTreeMap<TickId, &ExecutionCertificate> {
    let mut widest: BTreeMap<TickId, &ExecutionCertificate> = BTreeMap::new();
    for finalization in block.certificates().iter() {
        for cert in finalization.execution_certificates() {
            let cert = cert.as_unverified();
            widest
                .entry(*cert.tick_id())
                .and_modify(|held| {
                    if covers_strictly_more(cert, held) {
                        *held = cert;
                    }
                })
                .or_insert(cert);
        }
    }
    widest
}

/// Whether `candidate` carries everything `held` does and at least one
/// outcome more.
///
/// Copies of one tick can be disjoint rather than nested, so a count
/// comparison would sometimes replace coverage with different coverage
/// and leave the transaction index pointing at outcomes the kept copy no
/// longer carries. One tick has one slot, so the rule is that the slot
/// never loses ground: a copy that is not a strict superset is dropped,
/// and the transactions only it covered are served from their own shard
/// instead.
#[must_use]
pub fn covers_strictly_more(candidate: &ExecutionCertificate, held: &ExecutionCertificate) -> bool {
    if candidate.leaf_indices().len() <= held.leaf_indices().len() {
        return false;
    }
    // Leaf indices are ascending and distinct on both sides, so one
    // forward walk of the candidate decides containment.
    let mut carried = candidate.leaf_indices().iter();
    held.leaf_indices()
        .iter()
        .all(|index| carried.any(|leaf| leaf == index))
}
