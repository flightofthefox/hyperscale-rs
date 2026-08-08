//! The split-boundary settled-set predicate.
//!
//! When a shard `P` splits, its chain terminates at a terminal block `B`.
//! A cross-shard transaction's `P`-half settles only if `P` committed the
//! certificate covering it by `B` — otherwise one side of the transaction
//! would apply without the other. `S_P` is the set of transactions `P`
//! settled by `B`; a surviving counterpart reconstructs it from `P`'s tail
//! chain.
//!
//! This module holds the shared predicate both sides apply: the shard
//! coordinator's pre-vote fence (a block carrying such a transaction votes
//! only if every past-terminal outcome is settled) and the execution
//! coordinator's finalize-hygiene gate (don't even produce a finalization
//! the fence would reject). Keeping one predicate keeps the two verdicts
//! from drifting — a disagreement would let a gate produce what the fence
//! rejects.

use std::collections::{BTreeSet, HashMap};
use std::hash::BuildHasher;

use crate::{RETENTION_HORIZON, ShardId, TopologySchedule, TxHash, WeightedTimestamp};

/// A terminated shard's settled-transaction set.
///
/// `txs` are the **cross-shard** transactions whose certificate committed
/// in its chain at or before its terminal block — the only ones a
/// counterpart fence ever queries. `terminal_wt` is the weighted timestamp
/// at which the shard terminated, bounding how long the set stays relevant
/// — [`RETENTION_HORIZON`] past it, any outcome naming the shard is
/// categorically unreachable everywhere.
#[derive(Clone, Debug)]
pub struct SettledTxSet {
    /// Cross-shard transactions the terminated shard settled by its
    /// terminal block.
    pub txs: BTreeSet<TxHash>,
    /// The terminal block's weighted timestamp.
    pub terminal_wt: WeightedTimestamp,
}

/// The verdict on a set of cross-shard execution certificates against the
/// known settled sets, at an anchored weighted timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettledSetVerdict {
    /// No outcome names a past-terminal shard, or every such outcome's
    /// transaction is in that shard's settled set.
    Pass,
    /// An outcome names a transaction a past-terminal shard did not
    /// settle, names a shard evicted from every retained window, or sits
    /// past the terminated shard's retention horizon — categorically
    /// unreachable.
    Reject,
    /// An outcome names a past-terminal shard whose settled set isn't
    /// known yet, or a shard scheduled to terminate whose settled set can
    /// only exist once it does — hold until the set is reconstructed.
    Defer,
}

/// Resolve cross-shard outcomes against the known settled sets at
/// `anchored_wt`.
///
/// `outcomes` yields `(shard, tx_hash)` for each transaction a constituent
/// execution certificate attests — the question the fence actually asks,
/// which is whether that shard settled that transaction. Past-terminal-ness
/// is read off the **anchored** snapshot at `anchored_wt`, so callers that
/// must agree across replicas (the vote fence) pass the voted block's
/// `parent_qc` weighted timestamp; node-local callers (the finalize gate)
/// pass their committed timestamp.
///
/// A shard that is not yet past-terminal but is scheduled to terminate (an
/// admitted split/merge or a coast toward its terminal block) is fenced the
/// same way: `Defer` until it terminates and its settled set resolves the
/// transaction. This closes the pre-boundary window in which a survivor
/// could finalize a straddler the terminating side never settled.
pub fn settled_set_verdict<S, I>(
    settled_sets: &HashMap<ShardId, SettledTxSet, S>,
    topology_schedule: &TopologySchedule,
    local_shard: ShardId,
    anchored_wt: WeightedTimestamp,
    outcomes: I,
) -> SettledSetVerdict
where
    S: BuildHasher,
    I: IntoIterator<Item = (ShardId, TxHash)>,
{
    let mut defer = false;
    for (shard, tx_hash) in outcomes {
        if shard == local_shard {
            continue;
        }
        // Evicted from every retained window — terminated so long ago its
        // transactions can never resolve.
        let Some((_, past_terminal)) = topology_schedule.at_for_shard(shard, anchored_wt) else {
            return SettledSetVerdict::Reject;
        };
        if !past_terminal {
            // `shard` is live now, but if it is scheduled to terminate it may
            // leave the trie before it settles this transaction — and once it
            // does, only its settled set is authoritative. Finalizing on
            // coverage alone would then risk applying a transaction the
            // terminating side never settled (it can produce its outcome yet
            // still fail to receive ours before its terminal block). Defer to
            // its settled set, exactly as for an already-terminated shard.
            if topology_schedule.termination_scheduled(shard, anchored_wt) {
                defer = true;
            }
            continue;
        }
        match settled_sets.get(&shard) {
            Some(settled) if anchored_wt > settled.terminal_wt.plus(RETENTION_HORIZON) => {
                return SettledSetVerdict::Reject;
            }
            Some(settled) if !settled.txs.contains(&tx_hash) => {
                return SettledSetVerdict::Reject;
            }
            Some(_) => {}
            None => defer = true,
        }
    }
    if defer {
        SettledSetVerdict::Defer
    } else {
        SettledSetVerdict::Pass
    }
}
