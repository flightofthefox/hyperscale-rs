//! What a sweep retires, and how a leaf says so.
//!
//! A sweepable cell answers "am I still needed" from itself: its value
//! carries the expiry it stops being needed at, and its key derives from
//! that expiry. Nothing weaker travels. A rule keyed off the transaction
//! needs the body retention drops; a rule keyed off a side index needs
//! an index a split child does not inherit. The prefix is the only thing
//! guaranteed to arrive, so the cell has to carry the answer.

use hyperscale_types::{SubstateKey, SweepFrontier, protocol_statics, protocol_statics_installed};

/// When a committed cell stops being needed, or `None` for every cell a
/// sweep does not reach.
///
/// The judgement is the VM's, for the reason it is the VM's for a
/// package cell: the value re-derives the key under its family's own
/// domain, so no tag and no trust in the writer enters into it. Every
/// backend indexes through here — the commit batch and the import that
/// rebuilds a store from leaves alike — because an index built one way
/// at commit and another at import is an index two replicas would sweep
/// different sets from.
///
/// Without the protocol answers installed (bare storage tests) nothing
/// is sweepable, matching the package index's seam.
#[must_use]
pub fn sweepable_expiry(key: SubstateKey, value: &[u8]) -> Option<u64> {
    if !protocol_statics_installed() {
        return None;
    }
    protocol_statics().sweepable_cell(key.owner.to_bytes(), key.local.0, value)
}

/// Read access to the sweepable cells this store's committed state
/// holds, in sweep order.
///
/// Derived state — the leaves are the authority — kept because the
/// keyspace is owner-major and always will be, so nothing about a leaf
/// key lets a walk find the next cell to expire. What the index answers
/// is which owners hold cells in which bucket; which cells is a question
/// the leaves answer for themselves, since the bucket leads a sweepable
/// cell's local half and one owner's bucket is a contiguous range.
///
/// The default is the empty index, for stores that never commit one
/// (test doubles, ephemeral views).
pub trait SweepIndex {
    /// The sweepable cells strictly after `frontier` and strictly below
    /// `ceiling`, in sweep order, at most `limit` of them, each with the
    /// expiry its value carries.
    ///
    /// Ascending in [`SweepFrontier`] order, which is the order the
    /// index and the leaves are already stored in, so the walk sorts
    /// nothing and stopping at `limit` is stopping rather than
    /// discarding. That is what makes the cap a bound on work and not
    /// only on output.
    ///
    /// `ceiling` is exclusive and is a bucket boundary, so every cell
    /// this returns is in a bucket wholly in the past.
    fn sweep_candidates(
        &self,
        frontier: SweepFrontier,
        ceiling: SweepFrontier,
        limit: usize,
    ) -> Vec<(SubstateKey, u64)> {
        let _ = (frontier, ceiling, limit);
        Vec::new()
    }
}
