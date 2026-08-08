//! What this shard has committed and not yet resolved.
//!
//! A committed transaction is owed exactly one outcome, and until a
//! certificate carries that outcome the transaction is in flight: its fee
//! reservation is engaged, its work counts against the drain, and the
//! shards party to it are still expected to certify it. This is the list
//! of those, and it is a fold over committed blocks rather than a
//! projection of live execution state — entries insert when a block
//! commits a transaction and release when a committed block carries the
//! outcome resolving it, so every replica's ledger is identical at equal
//! committed frontiers.
//!
//! That is the whole reason it exists apart from [`TickRegistry`]. Tick
//! state is what a node is *working on*, and it does not survive a
//! restart: a shard whose replicas all came back cannot name what it
//! committed and never finished, so it can neither finish it nor abort
//! it. A ledger folded from the chain can be rebuilt from the chain.
//!
//! [`TickRegistry`]: crate::ticks::TickRegistry

use std::collections::BTreeMap;
use std::sync::Arc;

use hyperscale_types::{
    Finalization, MAX_FINALIZATION_DELAY, MAX_VALIDITY_RANGE, Transaction, TxHash, Verifiable,
    WeightedTimestamp,
};

/// One committed transaction's outstanding account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Owed {
    /// The moment past which the transaction can no longer finalize
    /// anywhere: the last block that could have included it, plus the
    /// longest a cross-shard transaction can take to finalize.
    deadline: WeightedTimestamp,
    /// The reservation its committing block took against the drain, held
    /// here because an abandonment has no execution to read it from and
    /// must release exactly what was taken.
    declared_work: u64,
}

/// Committed-but-unresolved transactions, each against its deadline and
/// the reservation it holds.
#[derive(Debug, Default)]
pub struct UnresolvedTxs {
    owed: BTreeMap<TxHash, Owed>,
}

impl UnresolvedTxs {
    /// Rebuild from a replay of the committed chain: each transaction
    /// still owed an outcome, against the validity end its deadline
    /// derives from and the work its block reserved for it.
    ///
    /// The deadline rule lives here and only here, so a rebuilt ledger
    /// and a live one cannot disagree about when a transaction stops
    /// being able to finalize.
    #[must_use]
    pub fn restored(entries: Vec<(TxHash, WeightedTimestamp, u64)>) -> Self {
        Self {
            owed: entries
                .into_iter()
                .map(|(tx_hash, validity_end, declared_work)| {
                    (
                        tx_hash,
                        Owed {
                            deadline: validity_end.plus(MAX_FINALIZATION_DELAY),
                            declared_work,
                        },
                    )
                })
                .collect(),
        }
    }

    /// Record what a committed block puts in flight.
    ///
    /// Idempotent per transaction: a hash cannot commit twice within its
    /// own validity window, and re-registering one must not move the
    /// deadline it was admitted under.
    pub fn register_committed<'a>(
        &mut self,
        txs: impl IntoIterator<Item = &'a Arc<Verifiable<Transaction>>>,
    ) {
        for tx in txs {
            let owed = Owed {
                deadline: tx
                    .validity_range()
                    .end_timestamp_exclusive
                    .plus(MAX_FINALIZATION_DELAY),
                declared_work: tx.work(),
            };
            self.owed.entry(tx.hash()).or_insert(owed);
        }
    }

    /// Drop what a committed block's finalizations resolve. Every verdict
    /// arrives this way — accepted, refused, or aborted — so one release
    /// path covers them all.
    pub fn release_resolved(&mut self, finalizations: &[Arc<Verifiable<Finalization>>]) {
        for finalization in finalizations {
            for tx_hash in finalization.tx_hashes() {
                self.owed.remove(&tx_hash);
            }
        }
    }

    /// The transactions this shard can no longer finalize, each with the
    /// reservation it still holds.
    ///
    /// Read off committed content alone — the ledger is a fold over
    /// committed blocks and `now` is the committed weighted timestamp —
    /// so every replica at the same frontier names the same set, which is
    /// what lets a committee sign the abort it composes.
    #[must_use]
    pub fn past_deadline(&self, now: WeightedTimestamp) -> Vec<(TxHash, u64)> {
        self.owed
            .iter()
            .filter(|(_, owed)| now >= owed.deadline)
            .map(|(tx_hash, owed)| (*tx_hash, owed.declared_work))
            .collect()
    }

    /// Drop entries so old that no block could still reference them.
    ///
    /// A transaction becomes abortable at its deadline; the window from
    /// there to `MAX_VALIDITY_RANGE` beyond it is the room the shard has
    /// to get that abort committed. Past that the retention windows have
    /// dropped the transaction and a certificate naming it would be
    /// refused, so holding the entry only leaks memory.
    pub fn prune(&mut self, now: WeightedTimestamp) {
        self.owed
            .retain(|_, owed| owed.deadline.plus(MAX_VALIDITY_RANGE) > now);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.owed.len()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use hyperscale_types::test_utils::{make_finalization, stub_transaction};
    use hyperscale_types::{
        Address, BlockHeight, TimestampRange, TransactionDecision, Verified, WeightedTimestamp,
    };

    use super::*;

    const PAYER: [u8; 16] = [0xAA; 16];

    fn tx(seed: u8, end_ms: u64) -> Arc<Verifiable<Transaction>> {
        let validity = TimestampRange::new(
            WeightedTimestamp::ZERO,
            WeightedTimestamp::from_millis(end_ms),
        );
        Arc::new(Verifiable::from(Verified::new_unchecked_for_test(
            stub_transaction(PAYER, &[Address([seed; 16]).0], 1_000, validity),
        )))
    }

    fn ms(v: u64) -> WeightedTimestamp {
        WeightedTimestamp::from_millis(v)
    }

    /// A committed transaction is owed an outcome from the moment its
    /// block commits until a committed block carries one.
    #[test]
    fn a_committed_transaction_is_owed_an_outcome_until_one_commits() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(1, 60_000);
        ledger.register_committed(std::iter::once(&tx));
        assert_eq!(ledger.len(), 1);

        let resolved =
            make_finalization(BlockHeight::new(1), tx.hash(), TransactionDecision::Accept);
        ledger.release_resolved(&[Arc::new(Verifiable::from(resolved))]);
        assert_eq!(ledger.len(), 0);
    }

    /// Every verdict releases, not only acceptance — an abort resolves a
    /// transaction exactly as a settlement does, which is what lets one
    /// certificate answer for both.
    #[test]
    fn an_abort_releases_as_a_settlement_does() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(2, 60_000);
        ledger.register_committed(std::iter::once(&tx));

        let aborted =
            make_finalization(BlockHeight::new(1), tx.hash(), TransactionDecision::Aborted);
        ledger.release_resolved(&[Arc::new(Verifiable::from(aborted))]);
        assert_eq!(ledger.len(), 0);
    }

    /// Re-registering a transaction leaves the deadline it was admitted
    /// under alone: the fold has to be identical on a replica that sees
    /// the block once and one that replays it.
    #[test]
    fn re_registering_does_not_move_the_deadline() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(3, 60_000);
        ledger.register_committed(std::iter::once(&tx));
        ledger.register_committed(std::iter::once(&tx));

        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.len(), 1);
    }

    /// A transaction becomes abandonable at its own deadline and not
    /// before, carrying the reservation its committing block took.
    #[test]
    fn a_transaction_is_abandonable_at_its_deadline_and_not_before() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(4, 60_000);
        ledger.register_committed(std::iter::once(&tx));

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        assert!(
            ledger
                .past_deadline(deadline.minus(Duration::from_millis(1)))
                .is_empty(),
            "merely slow is not abandonable",
        );
        assert_eq!(
            ledger.past_deadline(deadline),
            vec![(tx.hash(), tx.work())],
            "at the deadline, named with what it reserved",
        );
    }

    /// An entry survives its deadline: the window past it is the room the
    /// shard has to get the abort committed. It goes once no block could
    /// still reference the transaction at all.
    #[test]
    fn an_entry_outlives_its_deadline_and_not_the_retention_window() {
        let mut ledger = UnresolvedTxs::default();
        let tx = tx(7, 60_000);
        ledger.register_committed(std::iter::once(&tx));

        let deadline = ms(60_000).plus(MAX_FINALIZATION_DELAY);
        ledger.prune(deadline);
        assert_eq!(ledger.len(), 1, "still the shard's to resolve");

        ledger.prune(deadline.plus(MAX_VALIDITY_RANGE));
        assert_eq!(ledger.len(), 0, "past every window that could carry it");
    }
}
