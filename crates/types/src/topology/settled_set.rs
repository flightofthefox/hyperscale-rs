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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::sync::Arc;

    use super::*;
    use crate::{
        Epoch, EpochWindows, Hash, NetworkDefinition, TERMINAL_EVIDENCE_EPOCHS, TopologySnapshot,
        ValidatorSet,
    };

    const LOCAL: ShardId = ShardId::leaf(1, 0);
    const PEER: ShardId = ShardId::leaf(1, 1);

    const EPOCH_MS: u64 = 300_000;
    /// The cut PEER terminates at: the close of the window it last lives in.
    const CUT_MS: u64 = EPOCH_MS;

    fn wt(ms: u64) -> WeightedTimestamp {
        WeightedTimestamp::from_millis(ms)
    }

    fn tx(seed: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[seed]))
    }

    fn snap(leaves: &[ShardId], cut: &[(ShardId, u64)]) -> Arc<TopologySnapshot> {
        Arc::new(
            TopologySnapshot::from_explicit_committees(
                NetworkDefinition::simulator(),
                &ValidatorSet::new(Vec::new()),
                leaves.iter().map(|s| (*s, Vec::new())).collect(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeSet::new(),
            )
            .with_scheduled_terminals(cut.iter().map(|(s, e)| (*s, Epoch::new(*e))).collect()),
        )
    }

    /// PEER lives in window 0 and terminates at its close; its children hold
    /// the keyspace from window 1 on.
    fn peer_terminated() -> TopologySchedule {
        let (left, right) = PEER.children();
        let mut sched =
            TopologySchedule::new(EPOCH_MS, Epoch::new(0), snap(&[LOCAL, PEER], &[(PEER, 0)]));
        // Retained through the whole evidence window, so an anchor inside it
        // resolves a window rather than falling off the head.
        for epoch in 1..=TERMINAL_EVIDENCE_EPOCHS + 1 {
            sched.insert(Epoch::new(epoch), snap(&[LOCAL, left, right], &[]));
        }
        sched
    }

    /// A set for PEER carrying `txs`, readable for the terminal-evidence
    /// window past the cut — what the acquisition stamps on it.
    fn set_of(txs: &[TxHash]) -> HashMap<ShardId, SettledTxSet> {
        let windows = EpochWindows::new(EPOCH_MS);
        let mut sets = HashMap::new();
        sets.insert(
            PEER,
            SettledTxSet {
                txs: txs.iter().copied().collect(),
                terminal_wt: wt(CUT_MS),
                readable_until: windows.terminal_evidence_expiry(wt(CUT_MS)),
            },
        );
        sets
    }

    fn verdict(
        sets: &HashMap<ShardId, SettledTxSet>,
        anchored_ms: u64,
        claims: &[(ShardId, TxHash, TxClaim)],
    ) -> SettledSetVerdict {
        settled_set_verdict(
            sets,
            &peer_terminated(),
            LOCAL,
            wt(anchored_ms),
            claims.iter().copied(),
        )
    }

    /// The set is the authority, and the two claims read it opposite ways: a
    /// settlement needs the departed shard to have settled, an abandonment
    /// needs it not to have.
    #[test]
    fn the_two_claims_read_the_same_set_in_opposite_directions() {
        let inside = CUT_MS + EPOCH_MS;
        let settled = set_of(&[tx(1)]);

        assert_eq!(
            verdict(&settled, inside, &[(PEER, tx(1), TxClaim::Settled)]),
            SettledSetVerdict::Pass,
        );
        assert_eq!(
            verdict(&settled, inside, &[(PEER, tx(1), TxClaim::Abandoned)]),
            SettledSetVerdict::Reject,
            "the partner settled it, so abandoning would tear it in half",
        );
        assert_eq!(
            verdict(&settled, inside, &[(PEER, tx(2), TxClaim::Abandoned)]),
            SettledSetVerdict::Pass,
            "one the set does not name is one the partner never settled",
        );
        assert_eq!(
            verdict(&settled, inside, &[(PEER, tx(2), TxClaim::Settled)]),
            SettledSetVerdict::Reject,
            "and settling it would need coverage that never existed",
        );
    }

    /// A past-terminal shard whose set is not held yet defers rather than
    /// deciding — the evidence exists, this replica has not read it.
    #[test]
    fn an_unheld_set_defers_both_claims() {
        let none = HashMap::new();
        let inside = CUT_MS + EPOCH_MS;
        for claim in [TxClaim::Settled, TxClaim::Abandoned] {
            assert_eq!(
                verdict(&none, inside, &[(PEER, tx(1), claim)]),
                SettledSetVerdict::Defer,
            );
        }
    }

    /// The window that decides is the set's own `readable_until`, which the
    /// acquisition derives from the schedule's epoch grid. It spans the two
    /// folds a terminal contribution takes to be attested at all, so a block
    /// anchored two epochs past the cut — the first that can carry the
    /// evidence — still reads the set rather than rejecting on age.
    #[test]
    fn the_evidence_window_outlives_the_fold_that_delivers_it() {
        let settled = set_of(&[tx(1)]);
        let delivered = CUT_MS + 2 * EPOCH_MS;

        assert_eq!(
            verdict(&settled, delivered, &[(PEER, tx(1), TxClaim::Settled)]),
            SettledSetVerdict::Pass,
            "the roots reach a boundary record two folds past the cut",
        );

        let expiry = EpochWindows::new(EPOCH_MS).terminal_evidence_expiry(wt(CUT_MS));
        assert_eq!(
            verdict(
                &settled,
                expiry.as_millis(),
                &[(PEER, tx(1), TxClaim::Settled)]
            ),
            SettledSetVerdict::Pass,
            "readable to the last instant of the window",
        );
        assert_eq!(
            verdict(
                &settled,
                expiry.as_millis() + 1,
                &[(PEER, tx(1), TxClaim::Settled)],
            ),
            SettledSetVerdict::Reject,
            "and unreadable past it, which refuses both claims alike",
        );
    }

    /// Claims about this shard are not the fence's business — it asks only
    /// what a *counterpart* did.
    #[test]
    fn a_claim_about_the_local_shard_is_not_asked() {
        let none = HashMap::new();
        assert_eq!(
            verdict(
                &none,
                CUT_MS + EPOCH_MS,
                &[(LOCAL, tx(1), TxClaim::Abandoned)],
            ),
            SettledSetVerdict::Pass,
        );
    }

    /// A shard still live in the anchored window is decided by its own
    /// certificates, not by a set that cannot exist yet — unless it is
    /// scheduled to terminate, which defers instead. The pre-boundary hold
    /// is what keeps a survivor from applying a straddler the terminating
    /// side never settles.
    #[test]
    fn a_shard_scheduled_to_terminate_defers_before_its_cut() {
        let none = HashMap::new();
        assert_eq!(
            verdict(&none, 0, &[(PEER, tx(1), TxClaim::Settled)]),
            SettledSetVerdict::Defer,
            "PEER is live in window 0 and carries an admitted terminal",
        );
    }

    /// One deferral does not soften a rejection elsewhere: a torn claim is
    /// refused whatever else the same finalization asks about.
    #[test]
    fn a_rejection_dominates_a_deferral() {
        let settled = set_of(&[tx(1)]);
        assert_eq!(
            verdict(
                &settled,
                CUT_MS + EPOCH_MS,
                &[
                    (PEER, tx(1), TxClaim::Abandoned),
                    (ShardId::leaf(2, 2), tx(3), TxClaim::Settled),
                ],
            ),
            SettledSetVerdict::Reject,
        );
    }
}
