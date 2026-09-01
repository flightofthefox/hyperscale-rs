//! The figures an abandonment record must restate, held from commit so
//! a vote can check them.
//!
//! A record restates each named transaction's deadline, reservation and
//! charge so that a replica whose replay window fell short composes the
//! same abort as one that held the transaction. Restating without
//! checking would let a proposer naming any genuinely unsettleable
//! transaction choose the vault and the amount the abort burns. So a
//! validator that holds the transaction checks the restatement and
//! refuses the block on a mismatch, and one that does not defers its vote
//! rather than accepting.
//!
//! This is the checking side of that: every committed transaction's
//! figures, read once off its body at commit and kept for as long as a
//! record may still name it. Not persisted and not rebuilt from a seeded
//! window — a validator that restarted or synced past a transaction's
//! block defers on records naming it until they age out, which costs
//! liveness on the abandonment path and never safety.

use std::collections::HashMap;
use std::sync::Arc;

use hyperscale_types::{
    MAX_VALIDITY_RANGE, Transaction, TxHash, UnsettledTx, Verifiable, WeightedTimestamp,
};

/// The figures of every committed transaction a record may still name.
pub struct AbandonmentFigures {
    held: HashMap<TxHash, UnsettledTx>,
}

/// How a record's entry stands against what this validator holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restatement {
    /// Every figure is the one the transaction fixes.
    Exact,
    /// A figure differs from the one the transaction fixes: the record is
    /// refused.
    Wrong,
    /// This validator does not hold the transaction, so it cannot say:
    /// the vote is deferred.
    Unknown,
}

impl AbandonmentFigures {
    pub fn new() -> Self {
        Self {
            held: HashMap::new(),
        }
    }

    /// Hold the figures of a committed block's transactions.
    ///
    /// Idempotent per transaction, like the execution ledger it mirrors:
    /// every figure is a function of the body, so re-registering one
    /// cannot move it.
    pub fn register_committed(&mut self, transactions: &[Arc<Verifiable<Transaction>>]) {
        for tx in transactions {
            self.held
                .entry(tx.hash())
                .or_insert_with(|| UnsettledTx::for_transaction(tx));
        }
    }

    /// Drop the figures no record can still name.
    ///
    /// The execution ledger lets an owed transaction go at its deadline
    /// plus [`MAX_VALIDITY_RANGE`] and offers a record only for what it
    /// still holds, so figures kept to that horizon outlive every record
    /// a peer can propose.
    pub fn prune(&mut self, now: WeightedTimestamp) {
        self.held
            .retain(|_, figures| figures.deadline.plus(MAX_VALIDITY_RANGE) > now);
    }

    /// How `entry` stands against the transaction it names.
    #[must_use]
    pub fn check(&self, entry: &UnsettledTx) -> Restatement {
        match self.held.get(&entry.tx_hash) {
            None => Restatement::Unknown,
            Some(held) if held == entry => Restatement::Exact,
            Some(_) => Restatement::Wrong,
        }
    }

    /// Hold `entry` as the transaction's own figures.
    #[cfg(test)]
    pub fn remember(&mut self, entry: UnsettledTx) {
        self.held.insert(entry.tx_hash, entry);
    }
}

impl Default for AbandonmentFigures {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use hyperscale_types::{
        AbortCharge, Address, AddressClass, Hash, LocalKey, MAX_VALIDITY_RANGE, SubstateKey,
        TxHash, UnsettledTx, WeightedTimestamp,
    };

    use super::{AbandonmentFigures, Restatement};

    fn figures(seed: u8) -> UnsettledTx {
        UnsettledTx {
            tx_hash: TxHash::from(Hash::from_bytes(&[seed; 32])),
            deadline: WeightedTimestamp::from_millis(10_000),
            declared_work: 5,
            charge: AbortCharge {
                vault: SubstateKey {
                    owner: Address::new([seed; 31], AddressClass::Component),
                    local: LocalKey([seed; 16]),
                },
                floor: 3,
            },
        }
    }

    /// A holder checks every figure: the same entry is exact, and one
    /// naming another vault, another floor, another reservation or
    /// another deadline is wrong.
    #[test]
    fn a_holder_checks_every_figure() {
        let mut held = AbandonmentFigures::new();
        held.remember(figures(1));

        assert_eq!(held.check(&figures(1)), Restatement::Exact);
        assert_eq!(
            held.check(&UnsettledTx {
                charge: AbortCharge {
                    vault: figures(2).charge.vault,
                    ..figures(1).charge
                },
                ..figures(1)
            }),
            Restatement::Wrong,
        );
        assert_eq!(
            held.check(&UnsettledTx {
                charge: AbortCharge {
                    floor: 4,
                    ..figures(1).charge
                },
                ..figures(1)
            }),
            Restatement::Wrong,
        );
        assert_eq!(
            held.check(&UnsettledTx {
                declared_work: 6,
                ..figures(1)
            }),
            Restatement::Wrong,
        );
        assert_eq!(
            held.check(&UnsettledTx {
                deadline: WeightedTimestamp::from_millis(10_001),
                ..figures(1)
            }),
            Restatement::Wrong,
        );
    }

    /// A validator that does not hold the transaction cannot say either
    /// way, which is a third answer and not a pass.
    #[test]
    fn a_non_holder_cannot_say() {
        let held = AbandonmentFigures::new();
        assert_eq!(held.check(&figures(1)), Restatement::Unknown);
    }

    /// Figures outlive the deadline by the ledger's own margin, and no
    /// longer: past it no record can name the transaction, so nothing
    /// is left to check.
    #[test]
    fn figures_are_kept_as_long_as_a_record_can_name_them() {
        let mut held = AbandonmentFigures::new();
        held.remember(figures(1));
        let deadline = figures(1).deadline;

        held.prune(
            deadline
                .plus(MAX_VALIDITY_RANGE)
                .minus(Duration::from_millis(1)),
        );
        assert_eq!(held.check(&figures(1)), Restatement::Exact);

        held.prune(deadline.plus(MAX_VALIDITY_RANGE));
        assert_eq!(held.check(&figures(1)), Restatement::Unknown);
    }
}
