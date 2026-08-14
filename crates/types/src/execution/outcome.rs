//! Per-tx execution outcome ([`TxOutcome`]) and the [`ExecutionOutcome`] enum
//! carried inside execution certificates.

use hyperscale_hbor::Hbor;

use crate::{GlobalReceiptHash, MAX_PROVISION_TARGET_SHARDS, ShardId, TxHash};

/// Per-transaction execution outcome within a tick.
///
/// Carried inside execution certificates so remote shards can extract
/// individual transaction results for cross-shard finalization.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct TxOutcome {
    tx_hash: TxHash,
    outcome: ExecutionOutcome,
    /// Set when this shard charges a transaction it pays for without
    /// applying its effects: the hash of the receipt carrying that charge.
    ///
    /// An aborted transaction's own effects never apply — that is what
    /// makes a cross-shard abort atomic — and a failed one produced none
    /// to apply. Either way the payer still owes for the work the attempt
    /// consumed, and state moves only through receipts. The fee receipt is
    /// the reconciliation: it carries that debit and nothing else, and
    /// naming its hash here puts it under the signed receipt root like any
    /// other receipt's content.
    ///
    /// A failure that settles one takes the place of the `Failed` receipt
    /// it would otherwise store, so the one-receipt-per-outcome pairing is
    /// unchanged.
    fee_receipt: Option<GlobalReceiptHash>,
    /// What this shard attests it did for the transaction, under the
    /// engine's schedule.
    ///
    /// Beside the outcome rather than inside the receipt, and the
    /// distinction is load-bearing. A receipt is the effect record every
    /// participant of a cross-shard transaction derives identically —
    /// locality decides what is *applied* from it, never what it *says*.
    /// Work is the opposite kind of quantity: this shard's own share,
    /// which the participants are meant to differ on. It also covers every
    /// verdict, where a receipt covers only the outcomes that produced
    /// one, so an attempt that failed or aborted still reports the
    /// declaration work it really did.
    attested_work: u64,
    /// What carrying this transaction cost a block, in work units — the
    /// quantity admission reserved against the drain budget.
    ///
    /// The mirror image of [`TxOutcome::attested_work`]. That is this
    /// shard's own share of what the transaction *cost*, which
    /// participants are meant to differ on; this is what it *reserved*,
    /// derived from the whole declaration and therefore identical on
    /// every participant.
    ///
    /// Attested rather than re-derived because release has to work
    /// without the transaction. A block settling a tick releases the
    /// reservation its committing block took, and a validator holding
    /// the certificate but not the transactions — a node that snap-synced
    /// past them — still has to reach the same total. Reserving one
    /// number and releasing another leaves the running total drifting
    /// upward and never returning to zero.
    declared_work: u64,
    /// The other shards party to the transaction — the ones whose
    /// certificates its settlement waits on. Ascending and distinct;
    /// empty for a transaction reaching no further than this shard.
    ///
    /// A function of the declaration and the topology that committed it,
    /// so every participant derives the same set. The certifying shard is
    /// left out: the certificate carrying this outcome is its report.
    ///
    /// Attested rather than re-derived for the reason
    /// [`TxOutcome::declared_work`] is — a validator holding the
    /// certificate but not the transaction still has to reach the same
    /// answer — and it is what lets a set of certificates state how
    /// complete it needs to be. Without it the rule discarding a refused
    /// transaction's effects is read over whichever certificates the set
    /// happens to carry, and a set with the refusal dropped reads as
    /// unanimous.
    #[hbor(max = MAX_PROVISION_TARGET_SHARDS)]
    counterparts: Vec<ShardId>,
}

impl TxOutcome {
    /// Create a new `TxOutcome` settling no fee receipt.
    #[must_use]
    pub const fn new(tx_hash: TxHash, outcome: ExecutionOutcome) -> Self {
        Self::attesting(tx_hash, outcome, 0)
    }

    /// A `TxOutcome` attesting `work`, settling no fee receipt.
    #[must_use]
    pub const fn attesting(tx_hash: TxHash, outcome: ExecutionOutcome, work: u64) -> Self {
        Self {
            attested_work: work,
            declared_work: 0,
            tx_hash,
            outcome,
            fee_receipt: None,
            counterparts: Vec::new(),
        }
    }

    /// Bind what this transaction reserved against the drain budget.
    #[must_use]
    pub const fn reserving(mut self, declared_work: u64) -> Self {
        self.declared_work = declared_work;
        self
    }

    /// Bind the shards this transaction's settlement waits on, in the one
    /// form the set may take: ascending, distinct, and without the shard
    /// whose certificate carries the outcome.
    #[must_use]
    pub fn awaiting(mut self, counterparts: impl IntoIterator<Item = ShardId>) -> Self {
        let mut counterparts: Vec<ShardId> = counterparts.into_iter().collect();
        counterparts.sort_unstable();
        counterparts.dedup();
        self.counterparts = counterparts;
        self
    }

    /// Create a `TxOutcome` that settles the payer's charge through the
    /// named fee receipt.
    ///
    /// Both outcomes that owe a charge without applying the transaction's
    /// own effects use this: an abort, whose effects are discarded to keep
    /// the cross-shard settlement atomic, and a failure, whose effects the
    /// engine never produced. In either case the transaction did work its
    /// payer owes for, and the receipt named here is the only thing that
    /// moves that charge.
    #[must_use]
    pub const fn with_fee(
        tx_hash: TxHash,
        outcome: ExecutionOutcome,
        fee_receipt: GlobalReceiptHash,
        work: u64,
    ) -> Self {
        Self {
            attested_work: work,
            declared_work: 0,
            tx_hash,
            outcome,
            fee_receipt: Some(fee_receipt),
            counterparts: Vec::new(),
        }
    }

    /// What this shard attests it did for the transaction.
    #[must_use]
    pub const fn attested_work(&self) -> u64 {
        self.attested_work
    }

    /// What carrying this transaction reserved against the drain budget.
    #[must_use]
    pub const fn declared_work(&self) -> u64 {
        self.declared_work
    }

    /// The fee receipt this outcome settles, if any.
    #[must_use]
    pub const fn fee_receipt(&self) -> Option<GlobalReceiptHash> {
        self.fee_receipt
    }

    /// The shards this transaction's settlement waits on, besides the one
    /// certifying this outcome.
    #[must_use]
    pub fn counterparts(&self) -> &[ShardId] {
        &self.counterparts
    }

    /// Transaction hash.
    #[must_use]
    pub const fn tx_hash(&self) -> TxHash {
        self.tx_hash
    }

    /// The execution outcome for this transaction.
    #[must_use]
    pub const fn outcome(&self) -> &ExecutionOutcome {
        &self.outcome
    }

    /// Consume the outcome and return its parts.
    #[must_use]
    pub fn into_parts(self) -> (TxHash, ExecutionOutcome) {
        (self.tx_hash, self.outcome)
    }

    /// Whether this outcome is an abort.
    #[must_use]
    pub const fn is_aborted(&self) -> bool {
        matches!(self.outcome, ExecutionOutcome::Aborted)
    }
}

/// The outcome of executing a transaction on a single shard.
///
/// The variant tag IS the outcome — there is no separate `success: bool`
/// flag. Failed transactions carry no `receipt_hash` on the wire (the
/// canonical [`FAILED_RECEIPT_HASH`](crate::FAILED_RECEIPT_HASH) is
/// derivable at hash time).
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub enum ExecutionOutcome {
    /// Engine committed the transaction; state changes applied.
    Succeeded {
        /// Hash of the global receipt produced by this execution.
        receipt_hash: GlobalReceiptHash,
    },
    /// Engine rejected the transaction; no state changes applied.
    /// Carries no payload — every failure is consensus-equivalent.
    Failed,
    /// Transaction aborted before execution could complete.
    Aborted,
}

impl ExecutionOutcome {
    /// Whether the transaction was aborted.
    #[must_use]
    pub const fn is_aborted(&self) -> bool {
        matches!(self, Self::Aborted)
    }
}
