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
//! That is the whole reason it exists apart from [`WaveRegistry`]. Wave
//! state is what a node is *working on*, and it does not survive a
//! restart: a shard whose replicas all came back cannot name what it
//! committed and never finished, so it can neither finish it nor abort
//! it. A ledger folded from the chain can be rebuilt from the chain.
//!
//! [`WaveRegistry`]: crate::waves::WaveRegistry

use std::collections::BTreeMap;
use std::sync::Arc;

use hyperscale_types::{
    Finalization, MAX_VALIDITY_RANGE, Transaction, TxHash, Verifiable, WAVE_TIMEOUT,
    WeightedTimestamp,
};

/// Committed-but-unresolved transactions, each against the moment past
/// which it can no longer finalize anywhere: the last block that could
/// have included it, plus the longest a cross-shard transaction can take
/// to finalize.
#[derive(Debug, Default)]
pub struct UnresolvedTxs {
    deadlines: BTreeMap<TxHash, WeightedTimestamp>,
}

impl UnresolvedTxs {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
            let deadline = tx
                .validity_range()
                .end_timestamp_exclusive
                .plus(WAVE_TIMEOUT);
            self.deadlines.entry(tx.hash()).or_insert(deadline);
        }
    }

    /// Drop what a committed block's finalizations resolve. Every verdict
    /// arrives this way — accepted, refused, or aborted — so one release
    /// path covers them all.
    pub fn release_resolved(&mut self, finalizations: &[Arc<Verifiable<Finalization>>]) {
        for finalization in finalizations {
            for tx_hash in finalization.tx_hashes() {
                self.deadlines.remove(&tx_hash);
            }
        }
    }

    /// Drop entries so old that no block could still reference them.
    ///
    /// A transaction becomes abortable at its deadline; the window from
    /// there to `MAX_VALIDITY_RANGE` beyond it is the room the shard has
    /// to get that abort committed. Past that the retention windows have
    /// dropped the transaction and a certificate naming it would be
    /// refused, so holding the entry only leaks memory.
    pub fn prune(&mut self, now: WeightedTimestamp) {
        self.deadlines
            .retain(|_, deadline| deadline.plus(MAX_VALIDITY_RANGE) > now);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.deadlines.len()
    }
}

#[cfg(test)]
mod tests {
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
        let mut ledger = UnresolvedTxs::new();
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
        let mut ledger = UnresolvedTxs::new();
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
        let mut ledger = UnresolvedTxs::new();
        let tx = tx(3, 60_000);
        ledger.register_committed(std::iter::once(&tx));
        ledger.register_committed(std::iter::once(&tx));

        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.len(), 1);
    }

    /// An entry survives its deadline: the window past it is the room the
    /// shard has to get the abort committed. It goes once no block could
    /// still reference the transaction at all.
    #[test]
    fn an_entry_outlives_its_deadline_and_not_the_retention_window() {
        let mut ledger = UnresolvedTxs::new();
        let tx = tx(7, 60_000);
        ledger.register_committed(std::iter::once(&tx));

        let deadline = ms(60_000).plus(WAVE_TIMEOUT);
        ledger.prune(deadline);
        assert_eq!(ledger.len(), 1, "still the shard's to resolve");

        ledger.prune(deadline.plus(MAX_VALIDITY_RANGE));
        assert_eq!(ledger.len(), 0, "past every window that could carry it");
    }
}
