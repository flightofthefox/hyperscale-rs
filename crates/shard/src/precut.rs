//! What a successor knows about the chains that ran before it.
//!
//! A reshape successor holds no record of what its predecessor
//! committed, so it refuses every transaction whose validity window
//! opened before its own origin. That refusal is a superset of the real
//! hazard — a transaction submitted before the cut and never committed is
//! harmless there, and landing it here is its first commit — and this is
//! what narrows it: per-predecessor answers, each proven against a
//! `committed_txs_root` the successor commit-proved.
//!
//! Nothing here applies to a chain older than `MAX_VALIDITY_RANGE`. That
//! is the widest a validity window gets, so no transaction a chain of
//! that age can be offered opens before its origin.

use std::collections::HashMap;

use hyperscale_types::{PredecessorTerminal, ShardId, TxHash};

/// What a successor knows about one transaction across every chain it
/// succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecutStatus {
    /// Proven absent from every predecessor's committed set — the
    /// transaction is admissible, and landing it here is its first
    /// commit.
    Absent,
    /// Some predecessor committed it. Admitting it would be the second
    /// commit of one transaction across the boundary.
    Committed,
    /// At least one predecessor has not answered yet. Indistinguishable
    /// from `Committed` for admission purposes, but not for votes: an
    /// answer is still coming, so the vote waits rather than refusing.
    Unresolved,
}

/// Per-predecessor answers about transactions that predate this chain.
///
/// Keyed by predecessor as well as transaction because a merged parent
/// succeeds two children and a transaction is only admissible when it is
/// absent from **both**: an absence proof against one child's root says
/// nothing about what the other committed.
#[derive(Debug, Clone, Default)]
pub struct PrecutResolutions {
    /// `true` where the named predecessor proved the transaction absent
    /// from its committed set, `false` where it reported committing it.
    answers: HashMap<(ShardId, TxHash), bool>,
}

impl PrecutResolutions {
    /// Record one predecessor's answer.
    ///
    /// The caller has already verified an `absent` answer against that
    /// predecessor's attested root; a `committed` answer needs no proof,
    /// because it leaves the standing refusal in place.
    pub fn record(&mut self, predecessor: ShardId, tx_hash: TxHash, absent: bool) {
        self.answers.insert((predecessor, tx_hash), absent);
    }

    /// What is known about `tx_hash` across `predecessors`.
    ///
    /// A chain with no predecessors resolves nothing: it either has none
    /// (born at network genesis, where no transaction predates it) or has
    /// not been handed them yet, and both cases keep the strict rule.
    #[must_use]
    pub fn status(&self, predecessors: &[PredecessorTerminal], tx_hash: &TxHash) -> PrecutStatus {
        if predecessors.is_empty() {
            return PrecutStatus::Unresolved;
        }
        let mut all_answered = true;
        for predecessor in predecessors {
            match self.answers.get(&(predecessor.shard, *tx_hash)) {
                // One predecessor committing it settles the question; no
                // other answer can make it admissible.
                Some(false) => return PrecutStatus::Committed,
                Some(true) => {}
                None => all_answered = false,
            }
        }
        if all_answered {
            PrecutStatus::Absent
        } else {
            PrecutStatus::Unresolved
        }
    }

    /// The `(predecessor, transaction)` pairs still owed an answer — what
    /// a driver turns into queries.
    ///
    /// The predecessor rides out whole rather than as its shard id: a
    /// query names the terminal block it resolves against, and the
    /// absence proof that comes back is checked against that terminal's
    /// root.
    #[must_use]
    pub fn outstanding(
        &self,
        predecessors: &[PredecessorTerminal],
        tx_hashes: impl IntoIterator<Item = TxHash>,
    ) -> Vec<(PredecessorTerminal, TxHash)> {
        let mut out = Vec::new();
        for tx_hash in tx_hashes {
            for predecessor in predecessors {
                if !self.answers.contains_key(&(predecessor.shard, tx_hash)) {
                    out.push((*predecessor, tx_hash));
                }
            }
        }
        out
    }
}

/// What a block's pre-cut content means for the vote on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecutVerdict {
    /// Nothing in the block predates this chain, or everything that does
    /// is proven absent from every predecessor's committed set.
    Pass,
    /// The block carries content this chain must never commit. Naming
    /// what, for the log.
    Reject(String),
    /// The block carries a transaction that predates this chain and whose
    /// status is still outstanding. The vote waits: refusing would make a
    /// slow answer look like a bad block, and every honest validator
    /// reaches the same verdict once the answer lands.
    Defer(TxHash),
}

#[cfg(test)]
mod tests {
    use hyperscale_types::{BlockHash, BlockHeight, CommittedTxsRoot, Hash};

    use super::*;

    fn tx(seed: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[seed]))
    }

    fn predecessor(shard: ShardId) -> PredecessorTerminal {
        PredecessorTerminal {
            shard,
            height: BlockHeight::new(9),
            block_hash: BlockHash::ZERO,
            committed_txs_root: CommittedTxsRoot::ZERO,
        }
    }

    fn one() -> Vec<PredecessorTerminal> {
        vec![predecessor(ShardId::leaf(1, 0))]
    }

    fn two() -> Vec<PredecessorTerminal> {
        vec![
            predecessor(ShardId::leaf(2, 0)),
            predecessor(ShardId::leaf(2, 1)),
        ]
    }

    #[test]
    fn an_unanswered_transaction_is_unresolved() {
        let resolutions = PrecutResolutions::default();
        assert_eq!(resolutions.status(&one(), &tx(1)), PrecutStatus::Unresolved);
    }

    #[test]
    fn a_single_predecessors_answers_settle_it() {
        let mut resolutions = PrecutResolutions::default();
        let predecessors = one();
        let shard = predecessors[0].shard;

        resolutions.record(shard, tx(1), true);
        resolutions.record(shard, tx(2), false);
        assert_eq!(
            resolutions.status(&predecessors, &tx(1)),
            PrecutStatus::Absent
        );
        assert_eq!(
            resolutions.status(&predecessors, &tx(2)),
            PrecutStatus::Committed
        );
    }

    /// A merged parent succeeds both children, so one absence proof is
    /// not enough — the transaction stays unresolved until the second
    /// child answers, and a single `committed` settles it against
    /// admission however the other answers.
    #[test]
    fn a_merged_parent_needs_both_children() {
        let predecessors = two();
        let (left, right) = (predecessors[0].shard, predecessors[1].shard);

        let mut resolutions = PrecutResolutions::default();
        resolutions.record(left, tx(1), true);
        assert_eq!(
            resolutions.status(&predecessors, &tx(1)),
            PrecutStatus::Unresolved,
            "one child's absence proof settles nothing on its own"
        );
        resolutions.record(right, tx(1), true);
        assert_eq!(
            resolutions.status(&predecessors, &tx(1)),
            PrecutStatus::Absent
        );

        // The other child committed it: absent from one, committed by the
        // other, and inadmissible.
        let mut mixed = PrecutResolutions::default();
        mixed.record(left, tx(2), true);
        mixed.record(right, tx(2), false);
        assert_eq!(mixed.status(&predecessors, &tx(2)), PrecutStatus::Committed);
    }

    /// A `committed` answer settles the question before every predecessor
    /// has spoken — nothing a later answer says can make it admissible.
    #[test]
    fn one_committed_answer_settles_it_early() {
        let predecessors = two();
        let mut resolutions = PrecutResolutions::default();
        resolutions.record(predecessors[0].shard, tx(1), false);
        assert_eq!(
            resolutions.status(&predecessors, &tx(1)),
            PrecutStatus::Committed
        );
    }

    /// With no predecessors on hand nothing resolves, so the strict rule
    /// stands. This is the seat that missed the flip, not a chain born at
    /// genesis — that one never asks, because nothing predates it.
    #[test]
    fn no_predecessors_resolves_nothing() {
        let mut resolutions = PrecutResolutions::default();
        resolutions.record(ShardId::leaf(1, 0), tx(1), true);
        assert_eq!(resolutions.status(&[], &tx(1)), PrecutStatus::Unresolved);
    }

    /// Outstanding pairs are exactly what has not been answered, per
    /// predecessor — a merged parent owes two answers per transaction.
    #[test]
    fn outstanding_names_every_unanswered_pair() {
        let predecessors = two();
        let (left, right) = (predecessors[0].shard, predecessors[1].shard);
        let mut resolutions = PrecutResolutions::default();
        resolutions.record(left, tx(1), true);

        let outstanding: Vec<(ShardId, TxHash)> = resolutions
            .outstanding(&predecessors, [tx(1), tx(2)])
            .into_iter()
            .map(|(predecessor, tx_hash)| (predecessor.shard, tx_hash))
            .collect();
        assert_eq!(
            outstanding,
            vec![(right, tx(1)), (left, tx(2)), (right, tx(2))]
        );

        assert!(
            resolutions
                .outstanding(&predecessors, std::iter::empty())
                .is_empty()
        );
    }
}
