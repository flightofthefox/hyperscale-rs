//! In-flight fee reservations at the payer shard.
//!
//! A committed transaction whose fee payer routes to this shard holds
//! `max_fee` against the payer's vault until its tick finalizes — the
//! window in which the reservation is engaged but not yet settled. The
//! ledger tracks exactly that window from chain content: entries insert
//! when a block commits the transaction and release when a committed
//! block carries the finalization resolving it, so every replica's
//! ledger is identical at equal committed frontiers.
//!
//! Entries are deadline-bounded like [`crate::commit_dedup`]'s tiers: a
//! transaction resolved outside the certificate path — a reshape
//! terminal's abort-by-omission, where no finalization ever commits —
//! prunes at its validity end plus the retention horizon rather than
//! encumbering the payer forever.

use std::collections::HashMap;
use std::sync::Arc;

use hyperscale_types::{
    Address, Finalization, PrincipalAddr, RETENTION_HORIZON, Transaction, TxHash, Verifiable,
    WeightedTimestamp,
};

/// One engaged reservation: the payer's owner prefix, the held ceiling,
/// and the prune deadline.
struct Hold {
    payer: PrincipalAddr,
    max_fee: u128,
    deadline: WeightedTimestamp,
}

pub struct FeeReservationLedger {
    holds: HashMap<TxHash, Hold>,
}

impl FeeReservationLedger {
    pub fn new() -> Self {
        Self {
            holds: HashMap::new(),
        }
    }

    /// Record the reservations a committed block engages: every VM
    /// transaction whose fee payer `payer_local` claims for this shard.
    pub fn register_committed(
        &mut self,
        transactions: &[Arc<Verifiable<Transaction>>],
        payer_local: impl Fn(Address) -> bool,
    ) {
        for tx in transactions {
            let vm = tx.body();
            if !payer_local(vm.fee_payer.address()) {
                continue;
            }
            let deadline = tx
                .validity_range()
                .end_timestamp_exclusive
                .plus(RETENTION_HORIZON);
            self.holds.entry(tx.hash()).or_insert(Hold {
                payer: vm.fee_payer,
                max_fee: vm.max_fee,
                deadline,
            });
        }
    }

    /// Release the reservations a committed block's finalizations
    /// resolve — settlement and abort both arrive as finalizations.
    pub fn release_finalized(&mut self, finalizations: &[Arc<Verifiable<Finalization>>]) {
        for tick in finalizations {
            for tx_hash in tick.tx_hashes() {
                self.holds.remove(&tx_hash);
            }
        }
    }

    /// Drop holds past their deadline. `now` is the latest committed
    /// block's weighted timestamp.
    pub fn prune(&mut self, now: WeightedTimestamp) {
        self.holds.retain(|_, hold| hold.deadline > now);
    }

    /// The total engaged reservation against `payer`, saturating.
    #[must_use]
    pub fn held_for(&self, payer: Address) -> u128 {
        self.holds
            .values()
            .filter(|hold| hold.payer == payer)
            .fold(0u128, |sum, hold| sum.saturating_add(hold.max_fee))
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::test_utils::{make_finalization, stub_transaction};
    use hyperscale_types::{
        AddressClass, BlockHeight, PrincipalAddr, TimestampRange, TransactionDecision, Verified,
    };

    use super::*;

    const PAYER: PrincipalAddr = PrincipalAddr::new([0xAA; 31]);
    const PAYER_ADDR: Address = PAYER.address();

    fn transaction(max_fee: u128, end_ms: u64) -> Arc<Verifiable<Transaction>> {
        let validity = TimestampRange::new(
            WeightedTimestamp::ZERO,
            WeightedTimestamp::from_millis(end_ms),
        );
        Arc::new(Verifiable::from(Verified::new_unchecked_for_test(
            stub_transaction(PAYER, &[PAYER.address()], max_fee, validity),
        )))
    }

    #[test]
    fn holds_accumulate_and_release_on_finalizations() {
        let mut ledger = FeeReservationLedger::new();
        let tx = transaction(1_000, 60_000);
        ledger.register_committed(std::slice::from_ref(&tx), |_| true);
        assert_eq!(ledger.held_for(PAYER_ADDR), 1_000);
        assert_eq!(
            ledger.held_for(Address::new([0xBB; 31], AddressClass::Component)),
            0
        );

        let tick = Arc::new(Verifiable::from(make_finalization(
            BlockHeight::new(1),
            tx.hash(),
            TransactionDecision::Accept,
        )));
        ledger.release_finalized(std::slice::from_ref(&tick));
        assert_eq!(ledger.held_for(PAYER_ADDR), 0);
    }

    #[test]
    fn a_transaction_this_shard_does_not_pay_for_holds_nothing() {
        let mut ledger = FeeReservationLedger::new();
        let tx = transaction(1_000, 60_000);
        ledger.register_committed(std::slice::from_ref(&tx), |_| false);
        assert_eq!(ledger.held_for(PAYER_ADDR), 0);
    }

    #[test]
    fn prune_drops_holds_past_the_retention_deadline() {
        let mut ledger = FeeReservationLedger::new();
        let tx = transaction(1_000, 100);
        ledger.register_committed(std::slice::from_ref(&tx), |_| true);

        ledger.prune(WeightedTimestamp::from_millis(100));
        assert_eq!(ledger.held_for(PAYER_ADDR), 1_000);

        let past = WeightedTimestamp::from_millis(100)
            .plus(RETENTION_HORIZON)
            .plus(std::time::Duration::from_millis(1));
        ledger.prune(past);
        assert_eq!(ledger.held_for(PAYER_ADDR), 0);
    }
}
