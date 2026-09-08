//! The rows a store derives from one committed leaf.

use hyperscale_types::{EntryKey, Hash, SubstateKey};

use crate::{SweepRows, entry_from_leaf, package_of_cell, sweepable_expiry};

/// What a store indexes beside a leaf, judged from the leaf alone.
///
/// Every index over the cells is derived state — the leaves are the
/// authority — and each family is one judgement of the bytes, so an
/// index built at commit and one rebuilt from imported leaves hold the
/// same rows. A site that lands leaves destructures this whole: a
/// family added here is a family every such site has to place.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LeafRows {
    /// The ordered-collection entry the leaf commits, keyed for range
    /// scans.
    pub entry: Option<(EntryKey, Vec<u8>)>,
    /// The content address the leaf publishes; the artifact is the
    /// leaf's own bytes.
    pub package: Option<Hash>,
    /// The expiry the leaf carries, which is the sweep bucket it counts
    /// under.
    pub sweep: Option<u64>,
}

impl LeafRows {
    /// The rows `value` at `key` yields.
    #[must_use]
    pub fn of(key: SubstateKey, value: &[u8]) -> Self {
        Self {
            entry: entry_from_leaf(key, value),
            package: package_of_cell(key, value),
            sweep: sweepable_expiry(key, value),
        }
    }
}

/// Fold one committed leaf's derived rows: retract the sweep row its
/// prior held, take the one its new value holds, and hand back the
/// package it publishes.
///
/// The judgement both backends share, so a family added to [`LeafRows`]
/// reaches memory and `RocksDB` at once rather than in two edits that have
/// to agree. What differs between them is where the package artifact
/// lands — a map or a write batch — which is why that half is returned
/// rather than applied.
///
/// Of the prior's rows only the sweep row moves: the package index is
/// content-addressed and never retracts, and an entry's index row is
/// written from the settled entries.
#[must_use]
pub fn index_leaf(
    key: SubstateKey,
    prior: Option<&[u8]>,
    change: Option<&[u8]>,
    sweep_rows: &mut SweepRows,
) -> Option<Hash> {
    let was = prior.and_then(|bytes| sweepable_expiry(key, bytes));
    let LeafRows {
        entry: _,
        package,
        sweep: now,
    } = change.map_or_else(LeafRows::default, |bytes| LeafRows::of(key, bytes));
    sweep_rows.delta(key.owner, was, now);
    package
}
