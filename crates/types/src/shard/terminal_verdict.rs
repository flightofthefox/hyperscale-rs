//! What a departed shard left unresolved of this shard's business.
//!
//! A cross-shard transaction needs every participant's certificate to
//! settle, so one whose counterpart leaves without settling it can never
//! settle anywhere. That is the fact this shard needs in order to abandon
//! it — and it is readable for exactly one window: the departed shard's
//! settled set is complete and beacon-attested, so absence from it is
//! proof, but the set can only be fetched while the terminal it belongs to
//! is still served.
//!
//! So the answer is written down while it can still be read. A record
//! names the transactions this chain still owes an outcome for that the
//! departed shard did not settle, and once committed it is ordinary
//! history: every replica reads the same verdicts off its own chain at any
//! distance, including one that was switched off when the counterpart
//! left.
//!
//! Only the negative is recorded. That a departed shard *did* settle a
//! transaction changes nothing this shard can act on — the transaction
//! stays owed and unabandonable either way — while the absence of a
//! settlement is what licenses a verdict.

use hyperscale_hbor::Hbor;

use crate::{MAX_UNSETTLED_PER_BLOCK, ShardId, TxHash, WeightedTimestamp};

/// One departed shard's unsettled remainder, as this chain sees it.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct TerminalVerdict {
    /// The shard that left.
    shard: ShardId,
    /// Its terminal block's weighted timestamp — what a validator
    /// resolves its settled set against, and what dates the record
    /// against the transactions it speaks for.
    terminal_wt: WeightedTimestamp,
    /// Transactions this chain still owes an outcome for that `shard` did
    /// not settle before it went.
    ///
    /// Sorted and duplicate-free, so the record has one form and a
    /// validator checking it walks the same order it would build.
    #[hbor(max = MAX_UNSETTLED_PER_BLOCK)]
    unsettled: Vec<TxHash>,
}

impl TerminalVerdict {
    /// Build a record over `unsettled`, in the canonical order.
    #[must_use]
    pub fn new(
        shard: ShardId,
        terminal_wt: WeightedTimestamp,
        unsettled: impl IntoIterator<Item = TxHash>,
    ) -> Self {
        let mut unsettled: Vec<TxHash> = unsettled.into_iter().collect();
        unsettled.sort_unstable();
        unsettled.dedup();
        Self {
            shard,
            terminal_wt,
            unsettled,
        }
    }

    /// The shard that left.
    #[must_use]
    pub const fn shard(&self) -> ShardId {
        self.shard
    }

    /// Its terminal block's weighted timestamp.
    #[must_use]
    pub const fn terminal_wt(&self) -> WeightedTimestamp {
        self.terminal_wt
    }

    /// The transactions it left unsettled.
    #[must_use]
    pub fn unsettled(&self) -> &[TxHash] {
        &self.unsettled
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
            && self.unsettled.windows(2).all(|pair| pair[0] < pair[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Hash;

    fn tx(seed: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[seed; 32]))
    }

    fn wt() -> WeightedTimestamp {
        WeightedTimestamp::from_millis(1_000)
    }

    /// One form: whatever order a caller offers, the record it builds is
    /// the record every other builder would have produced.
    #[test]
    fn a_record_is_built_in_its_canonical_order() {
        let jumbled = TerminalVerdict::new(ShardId::ROOT, wt(), [tx(3), tx(1), tx(3), tx(2)]);
        let ordered = TerminalVerdict::new(ShardId::ROOT, wt(), [tx(1), tx(2), tx(3)]);

        assert_eq!(jumbled, ordered, "sorted and deduplicated on the way in");
        assert!(jumbled.is_well_formed());
    }

    /// A record naming nothing is not a record. It would commit a block to
    /// a claim it does not make.
    #[test]
    fn an_empty_record_is_not_well_formed() {
        let empty = TerminalVerdict::new(ShardId::ROOT, wt(), []);
        assert!(!empty.is_well_formed());
    }

    /// Out of order or repeating is a second form of the same claim, and
    /// the root would differ from the one the canonical form produces.
    #[test]
    fn a_record_out_of_its_canonical_order_is_refused() {
        let reversed = TerminalVerdict {
            shard: ShardId::ROOT,
            terminal_wt: wt(),
            unsettled: vec![tx(2), tx(1)],
        };
        assert!(!reversed.is_well_formed());

        let repeating = TerminalVerdict {
            shard: ShardId::ROOT,
            terminal_wt: wt(),
            unsettled: vec![tx(1), tx(1)],
        };
        assert!(!repeating.is_well_formed());
    }
}
