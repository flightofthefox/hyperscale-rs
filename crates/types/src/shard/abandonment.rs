//! What this shard owes an outcome for that no counterpart can ever
//! settle, and the evidence that says so.
//!
//! A cross-shard transaction needs every certificate its settlement
//! waits on, so one whose counterpart can never certify it can never
//! settle anywhere. That is the fact this shard needs in order to abandon
//! it, and three things establish it. The counterpart left without
//! settling: its settled set is complete and beacon-attested, so absence
//! from it is proof, but the set can only be fetched while the terminal
//! it belongs to is still served. The core refused: its certificate says
//! so, and a refusal ends the transaction outright. Or the core never
//! committed it, as of one of its blocks past the deadline: a
//! non-inclusion proof of the committed-transaction cell against that
//! block's state root says so, and before the deadline it says nothing,
//! since the core may still legitimately commit.
//!
//! So the answer is written down while it can still be read. A record
//! names the transactions this chain still owes an outcome for that its
//! counterpart can never settle, with the kind of evidence and the
//! moment it was taken at, and once committed it is ordinary history:
//! every replica reads the same verdicts off its own chain at any
//! distance, including one that was switched off when the counterpart
//! left.
//!
//! Only the negative is recorded. That a counterpart *did* settle a
//! transaction changes nothing this shard can act on — the transaction
//! stays owed and unabandonable either way — while the impossibility of
//! a settlement is what licenses a verdict.
//!
//! Each name carries the figures composing the abort takes: the deadline
//! it opens at, the reservation it returns, and the charge it settles.
//! All are functions of the transaction body, so a proposer restates them
//! and a voter holding the transaction checks the restatement — and a
//! replica whose rebuild never reached the transaction's own block still
//! holds enough to compose the same verdict as its peers.

use hyperscale_hbor::Hbor;

use crate::{
    MAX_FINALIZATION_DELAY, MAX_UNSETTLED_PER_BLOCK, ShardId, SubstateKey, Transaction, TxHash,
    WeightedTimestamp,
};

/// What an abort of one transaction burns, and out of whose vault.
///
/// Both are functions of signed content — the vault the fee payer's
/// address derives, the amount its declaration prices to — so the
/// receipt settling an abort is the same receipt on every replica
/// whether or not the transaction ever reached an engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hbor)]
pub struct AbortCharge {
    /// The fee payer's vault, which the burn debits.
    pub vault: SubstateKey,
    /// The declared price: what every attempt owes, whatever refused it.
    pub amount: u128,
}

/// One transaction a counterpart can never settle, with what abandoning
/// it takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hbor)]
pub struct UnsettledTx {
    /// The transaction.
    pub tx_hash: TxHash,
    /// The moment past which it can no longer finalize anywhere:
    /// `validity_range.end_timestamp_exclusive + MAX_FINALIZATION_DELAY`.
    ///
    /// Also the anchor an [`Unsettleable::Unclaimed`] probe has to sit at
    /// or past, so a voter checks the proof's block against this figure
    /// rather than against a clock.
    pub deadline: WeightedTimestamp,
    /// The reservation its committing block took against the drain, which
    /// the abandonment returns exactly.
    pub declared_work: u64,
    /// What the abandonment burns, settled by the shard holding the
    /// vault and by no other.
    pub charge: AbortCharge,
}

impl UnsettledTx {
    /// What abandoning `tx` states, read off the transaction itself.
    ///
    /// The one place every figure is derived, so a proposer restating
    /// them and a voter checking the restatement compute one value: the
    /// deadline is the validity end plus [`MAX_FINALIZATION_DELAY`], the
    /// reservation is the declared work, and the charge is the fee vault
    /// at the declared price.
    ///
    /// # Panics
    ///
    /// As [`Transaction::work`], on a transaction that was never derived.
    #[must_use]
    pub fn for_transaction(tx: &Transaction) -> Self {
        Self {
            tx_hash: tx.hash(),
            deadline: tx
                .validity_range()
                .end_timestamp_exclusive
                .plus(MAX_FINALIZATION_DELAY),
            declared_work: tx.work(),
            charge: AbortCharge {
                vault: tx.fee_vault(),
                amount: tx.price(),
            },
        }
    }
}

/// How a block's resolutions stand against the transactions they name:
/// the figures its records restate, and the deliveries its finalizations
/// carry.
///
/// The voter's answer, read off committed bodies. A validator whose
/// store holds a transaction answers for it — a figure exactly or
/// wrongly, a delivery inside its window or past the lapse — and one
/// whose store never held it, having synced past its block, cannot say,
/// which is a third answer and not a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolutions {
    /// Every figure of every name is the one its transaction fixes, and
    /// no delivery has lapsed.
    Exact,
    /// A figure of this name differs from the one its transaction fixes:
    /// the block is refused.
    Wrong(TxHash),
    /// A finalization delivers a crossing of this transaction at an
    /// anchor at or past its lapse, where its issuer may already have
    /// proved the claim absent and taken the crossing back: the block is
    /// refused.
    Lapsed(TxHash),
    /// This validator does not hold this name's transaction, so it cannot
    /// say: the vote is deferred.
    Unknown(TxHash),
}

impl Resolutions {
    /// How `entries` stand against the figures `held` reads off each
    /// named transaction, `None` for one it does not hold.
    ///
    /// A misstatement answers over an unknown name: a proposer who
    /// restates one figure wrongly is refused whatever else the record
    /// names, and only a record every name of which checks out is exact.
    pub fn of(
        entries: impl IntoIterator<Item = UnsettledTx>,
        held: impl Fn(TxHash) -> Option<UnsettledTx>,
    ) -> Self {
        let mut unknown = None;
        for entry in entries {
            match held(entry.tx_hash) {
                Some(figures) if figures == entry => {}
                Some(_) => return Self::Wrong(entry.tx_hash),
                None => {
                    unknown.get_or_insert(entry.tx_hash);
                }
            }
        }
        unknown.map_or(Self::Exact, Self::Unknown)
    }

    /// This answer folded with the deliveries the block's finalizations
    /// carry, `lapsed` saying whether each has lapsed at the block's
    /// anchor, `None` for one this validator does not hold.
    ///
    /// A refusal answers over a deferral: a block carrying a lapsed
    /// delivery is refused whatever else this validator cannot say.
    #[must_use]
    pub fn and_deliveries(
        self,
        deliveries: impl IntoIterator<Item = TxHash>,
        lapsed: impl Fn(TxHash) -> Option<bool>,
    ) -> Self {
        let mut unknown = match self {
            Self::Wrong(_) | Self::Lapsed(_) => return self,
            Self::Unknown(tx_hash) => Some(tx_hash),
            Self::Exact => None,
        };
        for tx_hash in deliveries {
            match lapsed(tx_hash) {
                Some(true) => return Self::Lapsed(tx_hash),
                Some(false) => {}
                None => {
                    unknown.get_or_insert(tx_hash);
                }
            }
        }
        unknown.map_or(Self::Exact, Self::Unknown)
    }
}

/// Why a counterpart can never settle the transactions a record names,
/// and when that became so.
///
/// Every arm carries a moment and none carries its proof. The proof is
/// fetched by the voter — a settled set, a certificate, or a state proof
/// against a commit-proven header — and a voter that cannot verify
/// defers. An absence proof is a JMT non-inclusion path; carrying one
/// per entry would blow the record's size budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hbor)]
pub enum Unsettleable {
    /// The shard left without settling. Absence from its complete,
    /// beacon-attested settled set is the proof.
    Departed {
        /// Its terminal block's weighted timestamp — what a validator
        /// resolves its settled set against, and what dates the record
        /// against the transactions it speaks for.
        terminal_wt: WeightedTimestamp,
    },
    /// The core refused. Its certificate is the proof, and a refusal ends
    /// the transaction outright.
    Refused {
        /// The weighted timestamp of the refusing certificate's anchor.
        refused_wt: WeightedTimestamp,
    },
    /// The core did not commit it, as of one of its blocks inside the
    /// absence window. A non-inclusion proof of the transaction's
    /// committed cell against that block's state root is the proof, and
    /// the fact it proves is the same at every anchor in the window: the
    /// core's admission rule fences it at the validity end, so absent
    /// at one block past the deadline is absent at every later one —
    /// until the cell's own sweep, past which the cell is gone whether
    /// or not the core committed and the proof says nothing.
    Unclaimed {
        /// The weighted timestamp of the block the absence was proved
        /// against. At or past every named transaction's deadline and
        /// short of its committed cell's sweep, or the proof says
        /// nothing.
        probed_wt: WeightedTimestamp,
    },
    /// The delivery never claimed it, as of one of the delivering
    /// shard's blocks inside the lapse window — from the delivery
    /// window's close plus the finalization delay to the claim cell's
    /// sweep. A non-inclusion proof of the crossing's claim cell against
    /// that block's state root is the proof, and the fact is the same at
    /// every anchor in the window: the delivery's admission is fenced at
    /// the close, so a claim absent at one block past the lapse is
    /// absent at every later one the cell still exists at.
    Lapsed {
        /// The weighted timestamp of the block the absence was proved
        /// against. At or past every named transaction's lapse — its
        /// deadline plus one validity range — and short of the claim
        /// cell's sweep, or the proof says nothing.
        probed_wt: WeightedTimestamp,
    },
}

impl Unsettleable {
    /// The moment the evidence was taken at.
    #[must_use]
    pub const fn moment(&self) -> WeightedTimestamp {
        match self {
            Self::Departed { terminal_wt } => *terminal_wt,
            Self::Refused { refused_wt } => *refused_wt,
            Self::Unclaimed { probed_wt } | Self::Lapsed { probed_wt } => *probed_wt,
        }
    }

    /// The arm's byte in a record's leaf.
    ///
    /// The arms license different aborts, and a leaf naming only the
    /// moment would let one pass as the other.
    #[must_use]
    pub const fn discriminant(&self) -> u8 {
        match self {
            Self::Departed { .. } => 0,
            Self::Refused { .. } => 1,
            Self::Unclaimed { .. } => 2,
            Self::Lapsed { .. } => 3,
        }
    }
}

/// A core shard's refusal of a transaction a leg here issued for, as
/// mirrored off its signature-verified certificate.
///
/// What a `Refused` record restates and what a voter checks it against:
/// the anchor the refusing certificate carried, and the transaction's
/// own deadline, which is the clock the mirror lives on — a refusal is
/// held exactly as long as the leg entry it licenses a reclaim of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refusal {
    /// The weighted timestamp of the refusing certificate's anchor.
    pub refused_wt: WeightedTimestamp,
    /// The refused transaction's deadline, as the leg's ledger holds it.
    pub deadline: WeightedTimestamp,
}

/// A counterpart's failure to take a transaction a leg here issued for,
/// as proved off its commit-proven state.
///
/// Proved at a block past the anchor the question is meaningful at: a
/// core's committed cell past the transaction's deadline, or a
/// delivery's claim cell past its lapse.
///
/// What an `Unclaimed` or a `Lapsed` record restates and what a voter
/// checks it against. The anchor is the voter's own probe, which need
/// not be the proposer's: absence past the floor is the same fact at
/// every anchor, so a voter holding a proof at any block past it holds
/// the evidence the record claims. The floor is the clock the mirror
/// lives on, as a refusal's deadline is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Absence {
    /// The weighted timestamp of the block the absence was proved
    /// against — at or past `floor`, and short of the probed cell's
    /// sweep, one validity range on.
    pub probed_wt: WeightedTimestamp,
    /// The anchor the proof has to sit at or past: the transaction's
    /// deadline for a core's committed cell, its lapse — the deadline
    /// plus one validity range — for a delivery's claim cell.
    pub floor: WeightedTimestamp,
}

/// One counterpart's unsettleable remainder, as this chain sees it.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct AbandonmentRecord {
    /// The counterpart shard that can never settle these.
    shard: ShardId,
    /// Why not, and as of when.
    evidence: Unsettleable,
    /// Transactions this chain still owes an outcome for that `shard`
    /// can never settle.
    ///
    /// Sorted by hash and duplicate-free on it, so the record has one form
    /// and a validator checking it walks the same order it would build.
    #[hbor(max = MAX_UNSETTLED_PER_BLOCK)]
    unsettled: Vec<UnsettledTx>,
}

impl AbandonmentRecord {
    /// Build a record over `unsettled`, in the canonical order.
    #[must_use]
    pub fn new(
        shard: ShardId,
        evidence: Unsettleable,
        unsettled: impl IntoIterator<Item = UnsettledTx>,
    ) -> Self {
        let mut unsettled: Vec<UnsettledTx> = unsettled.into_iter().collect();
        unsettled.sort_unstable_by_key(|entry| entry.tx_hash);
        unsettled.dedup_by_key(|entry| entry.tx_hash);
        Self {
            shard,
            evidence,
            unsettled,
        }
    }

    /// A record over what a shard that left at `terminal_wt` did not
    /// settle.
    #[must_use]
    pub fn departed(
        shard: ShardId,
        terminal_wt: WeightedTimestamp,
        unsettled: impl IntoIterator<Item = UnsettledTx>,
    ) -> Self {
        Self::new(shard, Unsettleable::Departed { terminal_wt }, unsettled)
    }

    /// A record over what `shard`, a core, refused at `refused_wt`.
    #[must_use]
    pub fn refused(
        shard: ShardId,
        refused_wt: WeightedTimestamp,
        unsettled: impl IntoIterator<Item = UnsettledTx>,
    ) -> Self {
        Self::new(shard, Unsettleable::Refused { refused_wt }, unsettled)
    }

    /// A record over what `shard`, a core, had not committed as of its
    /// block at `probed_wt`.
    #[must_use]
    pub fn unclaimed(
        shard: ShardId,
        probed_wt: WeightedTimestamp,
        unsettled: impl IntoIterator<Item = UnsettledTx>,
    ) -> Self {
        Self::new(shard, Unsettleable::Unclaimed { probed_wt }, unsettled)
    }

    /// A record over what `shard`, delivering, had not claimed as of its
    /// block at `probed_wt`, past the lapse.
    #[must_use]
    pub fn lapsed(
        shard: ShardId,
        probed_wt: WeightedTimestamp,
        unsettled: impl IntoIterator<Item = UnsettledTx>,
    ) -> Self {
        Self::new(shard, Unsettleable::Lapsed { probed_wt }, unsettled)
    }

    /// The counterpart shard.
    #[must_use]
    pub const fn shard(&self) -> ShardId {
        self.shard
    }

    /// Why it can never settle these, and as of when.
    #[must_use]
    pub const fn evidence(&self) -> Unsettleable {
        self.evidence
    }

    /// The transactions it can never settle, each with what abandoning
    /// it takes.
    #[must_use]
    pub fn unsettled(&self) -> &[UnsettledTx] {
        &self.unsettled
    }

    /// Just the transactions named.
    pub fn tx_hashes(&self) -> impl Iterator<Item = TxHash> + '_ {
        self.unsettled.iter().map(|entry| entry.tx_hash)
    }

    /// Whether the record is in the one form it may take: sorted, without
    /// repeats, and naming something.
    ///
    /// An empty record asserts nothing and would cost a block a leaf for
    /// it, so it is not well-formed rather than merely pointless. The
    /// upper bound is the block's, which a single record may spend the
    /// whole of; what stops several records spending it each is the sum
    /// the block's own check applies.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        !self.unsettled.is_empty()
            && self.unsettled.len() <= MAX_UNSETTLED_PER_BLOCK
            && self
                .unsettled
                .windows(2)
                .all(|pair| pair[0].tx_hash < pair[1].tx_hash)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{Address, AddressClass, Hash, LocalKey};

    fn tx(seed: u8) -> UnsettledTx {
        UnsettledTx {
            tx_hash: TxHash::from(Hash::from_bytes(&[seed; 32])),
            deadline: WeightedTimestamp::from_millis(u64::from(seed) * 100),
            declared_work: u64::from(seed) * 7,
            charge: AbortCharge {
                vault: SubstateKey {
                    owner: Address::new([seed; 31], AddressClass::Component),
                    local: LocalKey([seed; 16]),
                },
                amount: u128::from(seed) * 3,
            },
        }
    }

    fn wt() -> WeightedTimestamp {
        WeightedTimestamp::from_millis(1_000)
    }

    /// One form: whatever order a caller offers, the record it builds is
    /// the record every other builder would have produced.
    /// A holder checks every figure: the same entry is exact, and one
    /// naming another vault, another amount, another reservation or
    /// another deadline is wrong.
    #[test]
    fn a_holder_checks_every_figure() {
        let held = |hash: TxHash| (hash == tx(1).tx_hash).then(|| tx(1));
        let restated = |entry: UnsettledTx| Resolutions::of([entry], held);

        assert_eq!(restated(tx(1)), Resolutions::Exact);
        let wrong = Resolutions::Wrong(tx(1).tx_hash);
        assert_eq!(
            restated(UnsettledTx {
                charge: AbortCharge {
                    vault: tx(2).charge.vault,
                    ..tx(1).charge
                },
                ..tx(1)
            }),
            wrong,
        );
        assert_eq!(
            restated(UnsettledTx {
                charge: AbortCharge {
                    amount: tx(1).charge.amount + 1,
                    ..tx(1).charge
                },
                ..tx(1)
            }),
            wrong,
        );
        assert_eq!(
            restated(UnsettledTx {
                declared_work: tx(1).declared_work + 1,
                ..tx(1)
            }),
            wrong,
        );
        assert_eq!(
            restated(UnsettledTx {
                deadline: tx(1).deadline.plus(Duration::from_millis(1)),
                ..tx(1)
            }),
            wrong,
        );
    }

    /// A validator that does not hold a transaction cannot say either
    /// way, which is a third answer and not a pass — and a misstatement
    /// elsewhere in the record answers over it.
    #[test]
    fn a_non_holder_cannot_say_unless_a_figure_is_wrong() {
        let held = |hash: TxHash| (hash == tx(1).tx_hash).then(|| tx(1));
        assert_eq!(
            Resolutions::of([tx(2)], held),
            Resolutions::Unknown(tx(2).tx_hash)
        );
        assert_eq!(
            Resolutions::of([tx(2), tx(1)], held),
            Resolutions::Unknown(tx(2).tx_hash)
        );
        assert_eq!(
            Resolutions::of(
                [
                    tx(2),
                    UnsettledTx {
                        declared_work: tx(1).declared_work + 1,
                        ..tx(1)
                    }
                ],
                held
            ),
            Resolutions::Wrong(tx(1).tx_hash)
        );
        assert_eq!(Resolutions::of([], held), Resolutions::Exact);

        // Deliveries fold after the figures: a lapsed one refuses over an
        // unknown name, an unknown delivery defers, and a wrong figure
        // stands whatever the deliveries say.
        let lapsed = |tx_hash: TxHash| {
            if tx_hash == tx(3).tx_hash {
                Some(true)
            } else if tx_hash == tx(1).tx_hash {
                Some(false)
            } else {
                None
            }
        };
        assert_eq!(
            Resolutions::Exact.and_deliveries([tx(1).tx_hash], lapsed),
            Resolutions::Exact
        );
        assert_eq!(
            Resolutions::Exact.and_deliveries([tx(2).tx_hash], lapsed),
            Resolutions::Unknown(tx(2).tx_hash)
        );
        assert_eq!(
            Resolutions::Unknown(tx(2).tx_hash).and_deliveries([tx(3).tx_hash], lapsed),
            Resolutions::Lapsed(tx(3).tx_hash)
        );
        assert_eq!(
            Resolutions::Wrong(tx(1).tx_hash).and_deliveries([tx(3).tx_hash], lapsed),
            Resolutions::Wrong(tx(1).tx_hash)
        );
    }

    #[test]
    fn a_record_is_built_in_its_canonical_order() {
        let jumbled =
            AbandonmentRecord::departed(ShardId::ROOT, wt(), [tx(3), tx(1), tx(3), tx(2)]);
        let ordered = AbandonmentRecord::departed(ShardId::ROOT, wt(), [tx(1), tx(2), tx(3)]);

        assert_eq!(jumbled, ordered, "sorted and deduplicated on the way in");
        assert!(jumbled.is_well_formed());
    }

    /// A record naming nothing is not a record. It would commit a block to
    /// a claim it does not make.
    #[test]
    fn an_empty_record_is_not_well_formed() {
        let empty = AbandonmentRecord::departed(ShardId::ROOT, wt(), []);
        assert!(!empty.is_well_formed());
    }

    /// Out of order or repeating is a second form of the same claim, and
    /// the root would differ from the one the canonical form produces.
    #[test]
    fn a_record_out_of_its_canonical_order_is_refused() {
        let reversed = AbandonmentRecord {
            shard: ShardId::ROOT,
            evidence: Unsettleable::Departed { terminal_wt: wt() },
            unsettled: vec![tx(2), tx(1)],
        };
        assert!(!reversed.is_well_formed());

        let repeating = AbandonmentRecord {
            shard: ShardId::ROOT,
            evidence: Unsettleable::Departed { terminal_wt: wt() },
            unsettled: vec![tx(1), tx(1)],
        };
        assert!(!repeating.is_well_formed());
    }

    /// The figures ride each name, so a record that reaches a replica
    /// holding none of the transactions still says what abandoning them
    /// takes.
    #[test]
    fn a_name_carries_what_abandoning_it_takes() {
        let record = AbandonmentRecord::departed(ShardId::ROOT, wt(), [tx(2), tx(1)]);
        assert_eq!(
            record.unsettled(),
            &[tx(1), tx(2)],
            "each name keeps its own deadline and reservation through the sort",
        );
    }

    /// Each arm is its own byte, and every arm reads its moment back.
    #[test]
    fn every_arm_has_its_own_discriminant_and_reads_its_moment() {
        let arms = [
            Unsettleable::Departed { terminal_wt: wt() },
            Unsettleable::Refused { refused_wt: wt() },
            Unsettleable::Unclaimed { probed_wt: wt() },
            Unsettleable::Lapsed { probed_wt: wt() },
        ];
        let mut bytes: Vec<u8> = arms.iter().map(Unsettleable::discriminant).collect();
        bytes.dedup();
        assert_eq!(bytes.len(), arms.len());
        for arm in arms {
            assert_eq!(arm.moment(), wt());
        }
    }
}
