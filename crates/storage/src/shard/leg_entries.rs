//! Durable leg entries.

use hyperscale_types::{LegEntry, TxHash};

/// The leg entries a shard holds beside its chain.
///
/// The execution ledger is a fold over the committed chain from the
/// replay window's floor, so an entry whose block sits below that floor
/// is not rebuilt. A leg entry outlives the floor: it stands until the
/// record cell it would take back is retired, and a record is retired on
/// a counterpart's evidence rather than on a clock. This is what carries
/// it across the restart.
///
/// The store is not the authority inside the replay window — the fold
/// is. A row seeds the ledger and the replay folds on top, so a write or
/// a delete lost to a crash is one the fold redoes: the block that
/// caused it is, by definition, one the window still reaches. That is
/// why writes need not be atomic with the block that prompts them.
///
/// All methods take `&self`; implementations use interior mutability.
pub trait LegEntryStore: Send + Sync {
    /// Write `entries`, replacing any row for the same transaction, and
    /// drop the rows for `released`.
    ///
    /// One call per commit so a backend can batch the block's whole
    /// change to the set.
    fn persist_leg_entries(&self, entries: &[LegEntry], released: &[TxHash]);

    /// Every row held, in transaction order.
    fn leg_entries(&self) -> Vec<LegEntry>;
}
