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

use crate::{
    BlockHash, BlockHeight, SettledTxsRoot, ShardId, TopologySchedule, TxHash, WeightedTimestamp,
};

/// What a survivor needs to acquire a departed shard's settled set and to
/// know how long the answer is good for.
///
/// Every field is read off the node's own beacon fold — the terminal
/// anchor's height, hash and attested root from the boundary record, the
/// cut and its expiry from the schedule's window grid — so nothing here is
/// fetched and nothing here is trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalEvidence {
    /// Height of the departed shard's terminal block, naming the window
    /// end the serve reconstructs from.
    pub height: BlockHeight,
    /// Hash of the terminal block — which terminal this targets, so a
    /// revised one replaces rather than duplicates.
    pub block_hash: BlockHash,
    /// The terminal cut's weighted timestamp.
    pub terminal_wt: WeightedTimestamp,
    /// The cut's
    /// [terminal-evidence expiry](crate::EpochWindows::terminal_evidence_expiry).
    pub readable_until: WeightedTimestamp,
    /// The beacon-attested `settled_txs_root` a fetched list must
    /// recompute to.
    pub attested_root: SettledTxsRoot,
}

/// A terminated shard's settled-transaction set.
///
/// `txs` are the **cross-shard** transactions whose certificate committed
/// in its chain at or before its terminal block — the only ones a
/// counterpart fence ever queries. `terminal_wt` is the weighted timestamp
/// at which the shard terminated; `readable_until` is how long the set
/// stays relevant, past which any outcome naming the shard is
/// categorically unreachable everywhere.
#[derive(Clone, Debug)]
pub struct SettledTxSet {
    /// Cross-shard transactions the terminated shard settled by its
    /// terminal block.
    pub txs: BTreeSet<TxHash>,
    /// The terminal block's weighted timestamp.
    pub terminal_wt: WeightedTimestamp,
    /// The shard's
    /// [terminal-evidence expiry](crate::EpochWindows::terminal_evidence_expiry),
    /// derived from the schedule's window grid when the set is recorded.
    /// Carried on the set so every consumer — the fence, and the caches
    /// holding it — reads one value rather than deriving its own.
    pub readable_until: WeightedTimestamp,
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

/// What a finalization claims about one transaction, which decides which
/// way the terminated partner's settled set has to read.
///
/// The two are opposite questions about the same evidence. A settlement
/// applies this shard's half, so it needs the partner to have settled its
/// own half too. An abandonment states the transaction reaches no outcome
/// anywhere, so it needs the partner *not* to have settled — a partner
/// that did settle is the one case where aborting would tear a
/// cross-shard transaction in half.
///
/// Both read the same set and both need it readable. Opposite questions
/// about the partner's set are still questions about the partner's set,
/// so a set nobody can read leaves either claim unproven and rejects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxClaim {
    /// This shard settles the transaction, on coverage the finalization
    /// carries.
    Settled,
    /// This shard abandons the transaction: it committed it, no
    /// certificate resolved it, and its deadline has passed.
    Abandoned,
}

/// Resolve a finalization's per-transaction claims against the known
/// settled sets at `anchored_wt`.
///
/// `claims` yields `(shard, tx_hash, claim)` for each participating shard
/// of each transaction the finalization reaches a verdict for. A
/// [`Settled`](TxClaim::Settled) claim's shards are the ones whose
/// certificates the finalization carries; an
/// [`Abandoned`](TxClaim::Abandoned) claim has no counterpart certificate
/// to read them off, so its caller supplies the participants the
/// committing block assigned.
///
/// Past-terminal-ness is read off the **anchored** snapshot at
/// `anchored_wt`, so callers that must agree across replicas (the vote
/// fence) pass the voted block's `parent_qc` weighted timestamp;
/// node-local callers (the finalize gate) pass their committed timestamp.
///
/// A shard that is not yet past-terminal but is scheduled to terminate (an
/// admitted split/merge or a coast toward its terminal block) is fenced the
/// same way whichever claim names it: `Defer` until it terminates and its
/// settled set answers. This closes the pre-boundary window in which a
/// survivor could finalize a straddler the terminating side never settled,
/// and the matching one in which it could abandon one the terminating side
/// did settle.
pub fn settled_set_verdict<S, I>(
    settled_sets: &HashMap<ShardId, SettledTxSet, S>,
    topology_schedule: &TopologySchedule,
    local_shard: ShardId,
    anchored_wt: WeightedTimestamp,
    claims: I,
) -> SettledSetVerdict
where
    S: BuildHasher,
    I: IntoIterator<Item = (ShardId, TxHash, TxClaim)>,
{
    let mut defer = false;
    for (shard, tx_hash, claim) in claims {
        if shard == local_shard {
            continue;
        }
        let settlement = matches!(claim, TxClaim::Settled);

        // Evicted from every retained window — terminated so long ago its
        // settled set can never be acquired. Both claims need the set. A
        // settlement is categorically unreachable without the partner's
        // coverage. And an abandonment turns on the partner *not* having
        // settled, which a set nobody can read cannot establish: a
        // certificate of this shard's covering the transaction is enough
        // for the partner to have settled against, and whether one exists
        // is not something a replica can answer about itself — a restart
        // loses the tick that produced it while the certificate itself
        // outlives the restart.
        let Some((_, past_terminal)) = topology_schedule.at_for_shard(shard, anchored_wt) else {
            return SettledSetVerdict::Reject;
        };
        if !past_terminal {
            // `shard` is live now, but if it is scheduled to terminate it may
            // leave the trie before it resolves this transaction — and once it
            // does, only its settled set is authoritative. Deciding on
            // coverage alone would then risk applying a transaction the
            // terminating side never settled (it can produce its outcome yet
            // still fail to receive ours before its terminal block), or
            // abandoning one it did. Defer to its settled set, exactly as for
            // an already-terminated shard.
            if topology_schedule.termination_scheduled(shard, anchored_wt) {
                defer = true;
            }
            continue;
        }
        match settled_sets.get(&shard) {
            // Past the evidence window the set stops being readable at all,
            // which rejects for the same reason eviction does.
            Some(settled) if anchored_wt > settled.readable_until => {
                return SettledSetVerdict::Reject;
            }
            // The partner's verdict, read the way the claim needs it: a
            // settlement needs the partner to have settled, an
            // abandonment needs it not to have.
            Some(settled) => {
                if settled.txs.contains(&tx_hash) != settlement {
                    return SettledSetVerdict::Reject;
                }
            }
            None => defer = true,
        }
    }
    if defer {
        SettledSetVerdict::Defer
    } else {
        SettledSetVerdict::Pass
    }
}
