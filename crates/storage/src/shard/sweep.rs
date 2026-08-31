//! What a sweep retires, and how a leaf says so.
//!
//! A sweepable cell answers "am I still needed" from itself: its value
//! carries the expiry it stops being needed at, and its key derives from
//! that expiry. Nothing weaker travels. A rule keyed off the transaction
//! needs the body retention drops; a rule keyed off a side index needs
//! an index a split child does not inherit. The prefix is the only thing
//! guaranteed to arrive, so the cell has to carry the answer.

use std::collections::BTreeMap;
use std::sync::Arc;

use hyperscale_types::{
    MAX_SWEEP_PER_BLOCK, SettledWrites, SubstateKey, SweepFrontier, WeightedTimestamp,
    protocol_statics, protocol_statics_installed,
};

use crate::tree::JmtSnapshot;

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

/// The cells a block anchored at `clock` removes, and where its frontier
/// lands.
///
/// The pair is the whole of a block's sweep. Removals are the sweepable
/// cells strictly above the parent's frontier and strictly below the
/// ceiling `clock` allows, in sweep order, at most
/// [`MAX_SWEEP_PER_BLOCK`]; the frontier is where the walk stopped.
///
/// It stops in one of two places, and which one is the block's claim
/// about whether it finished:
///
/// - **At the ceiling**, when fewer than the cap remained. Nothing
///   sweepable is left below the clock's own bucket, so the frontier
///   takes the ceiling itself and the next block starts from there.
/// - **On the last cell removed**, when the cap is what stopped it. The
///   frontier names that cell's position, so the next block resumes at
///   exactly the next one — a same-instant pile drains across blocks
///   rather than landing whole on whichever block straddles it.
///
/// A block whose ceiling has not moved past its parent's frontier
/// removes nothing and repeats the frontier it inherited. That is the
/// ordinary case at sub-second block times against a bucket spanning a
/// minute, which is why the frontier's own rule is monotone rather than
/// strictly advancing — the obligation that bites is reaching the
/// ceiling, not moving at all.
#[must_use]
pub fn sweep_for_block(
    store: &(impl SweepIndex + ?Sized),
    parent_frontier: SweepFrontier,
    clock: WeightedTimestamp,
) -> (Vec<SubstateKey>, SweepFrontier) {
    let ceiling = SweepFrontier::ceiling_at(clock);
    if parent_frontier >= ceiling {
        return (Vec::new(), parent_frontier);
    }
    let found = store.sweep_candidates(parent_frontier, ceiling, MAX_SWEEP_PER_BLOCK);
    let capped = found.len() >= MAX_SWEEP_PER_BLOCK;
    let frontier = match found.last() {
        Some((key, _)) if capped => SweepFrontier::of_leaf(*key),
        _ => ceiling,
    };
    (found.into_iter().map(|(key, _)| key).collect(), frontier)
}

/// The layered sweep walk every overlay reader shares: the base's
/// candidates at a limit widened by what the overlay retires, then the
/// overlay's own sweepable cells folded in and its removals taken out.
///
/// The widening is what lets an interval the overlay has mostly retired
/// still fill `limit` from the cells behind it — the shape
/// [`merge_entry_overlay_with`](crate::merge_entry_overlay_with) already
/// has, and for the same reason.
///
/// Both directions matter and both break state if missed. A cell an
/// unpersisted ancestor created reads as absent from the base, so a
/// removal the chain owes would go unmade; a cell one retired reads as
/// live, so a removal would be made twice — and the second lands on a
/// key the tree no longer holds.
#[must_use]
pub fn merge_sweep_overlay(
    base: impl FnOnce(usize) -> Vec<(SubstateKey, u64)>,
    overlay: &[Arc<JmtSnapshot>],
    frontier: SweepFrontier,
    ceiling: SweepFrontier,
    limit: usize,
) -> Vec<(SubstateKey, u64)> {
    if limit == 0 || frontier >= ceiling {
        return Vec::new();
    }
    // What the unpersisted chain says about cells in the interval, latest
    // write per key, `None` where it retired one.
    let mut touched: BTreeMap<SubstateKey, Option<u64>> = BTreeMap::new();
    for snapshot in overlay {
        for (key, change) in snapshot.settled.cells() {
            let position = SweepFrontier::of_leaf(*key);
            if position <= frontier || position >= ceiling {
                continue;
            }
            touched.insert(
                *key,
                change
                    .as_deref()
                    .and_then(|bytes| sweepable_expiry(*key, bytes)),
            );
        }
    }
    let retired = touched.values().filter(|expiry| expiry.is_none()).count();
    let mut merged: BTreeMap<SweepFrontier, (SubstateKey, u64)> =
        base(limit.saturating_add(retired))
            .into_iter()
            .map(|(key, expiry)| (SweepFrontier::of_leaf(key), (key, expiry)))
            .collect();
    for (key, expiry) in touched {
        let position = SweepFrontier::of_leaf(key);
        match expiry {
            Some(expiry) => {
                merged.insert(position, (key, expiry));
            }
            None => {
                merged.remove(&position);
            }
        }
    }
    merged.into_values().take(limit).collect()
}

/// Fold a block's removals into the settled set its receipts produced.
///
/// A removal is an ordinary write — `None` at the key — so a sweep needs
/// no commit path of its own: the tombstones fold with everything else
/// and land under `state_root` like any other change. That is the fact
/// that makes this tractable rather than novel.
///
/// # Panics
///
/// If a removal names a cell the block's own receipts also write. The
/// two would be one entry in the settled map, so one of them would
/// silently not happen — and which one is decided by insertion order,
/// which is not something a validator can check. Validation refuses such
/// a block before anything reaches here; this is the assertion that it
/// did.
#[must_use]
pub fn with_removals(settled: SettledWrites, removals: &[SubstateKey]) -> SettledWrites {
    if removals.is_empty() {
        return settled;
    }
    let (mut cells, entries) = settled.into_parts();
    for key in removals {
        assert!(
            cells.insert(*key, None).is_none(),
            "a sweep removed {key:?}, which this block's receipts also write",
        );
    }
    SettledWrites::from_parts(cells, entries)
}

#[cfg(test)]
mod tests {
    use hyperscale_types::test_utils::{install_stub_protocol_statics, stub_sweepable_cell};
    use hyperscale_types::{
        Address, AddressClass, BlockHeight, SWEEP_BUCKET_MS, StateRoot, SweepBucket,
    };

    use super::*;
    use crate::CollectedWrites;

    fn owner(tag: u8) -> Address {
        Address::new([tag; 31], AddressClass::Principal)
    }

    /// A sweepable cell of `owner`, expiring in `bucket`.
    fn cell(tag: u8, bucket: u32, body: u8) -> (SubstateKey, u64, Vec<u8>) {
        let expiry = u64::from(bucket) * SWEEP_BUCKET_MS;
        let (local, value) = stub_sweepable_cell(expiry, body);
        (
            SubstateKey {
                owner: owner(tag),
                local,
            },
            expiry,
            value,
        )
    }

    /// A pending ancestor whose settled writes are `cells`.
    fn ancestor(cells: Vec<(SubstateKey, Option<Vec<u8>>)>) -> Arc<JmtSnapshot> {
        Arc::new(JmtSnapshot::from_collected_writes(
            CollectedWrites::default(),
            SettledWrites::from_absolutes(cells.into_iter().collect()),
            StateRoot::ZERO,
            BlockHeight::GENESIS,
            StateRoot::ZERO,
            BlockHeight::new(1),
        ))
    }

    fn all() -> SweepFrontier {
        SweepFrontier::start_of(SweepBucket(u32::MAX))
    }

    /// A cell an unpersisted ancestor created is one this block owes a
    /// removal for, and the persisted store does not know it exists.
    #[test]
    fn the_overlay_adds_what_an_ancestor_created() {
        install_stub_protocol_statics();
        let (key, expiry, value) = cell(1, 3, 0x11);
        let merged = merge_sweep_overlay(
            |_| Vec::new(),
            &[ancestor(vec![(key, Some(value))])],
            SweepFrontier::ZERO,
            all(),
            10,
        );
        assert_eq!(merged, vec![(key, expiry)]);
    }

    /// And one an ancestor retired is not swept a second time — the
    /// second removal would land on a key the tree no longer holds.
    #[test]
    fn the_overlay_drops_what_an_ancestor_retired() {
        install_stub_protocol_statics();
        let (key, expiry, _) = cell(1, 3, 0x11);
        let merged = merge_sweep_overlay(
            |_| vec![(key, expiry)],
            &[ancestor(vec![(key, None)])],
            SweepFrontier::ZERO,
            all(),
            10,
        );
        assert!(merged.is_empty());
    }

    /// The base fetch widens by what the overlay retires, so an interval
    /// the overlay has mostly emptied still fills the limit from the
    /// cells behind it.
    #[test]
    fn the_base_fetch_widens_by_what_the_overlay_retired() {
        install_stub_protocol_statics();
        let (gone, gone_expiry, _) = cell(1, 3, 0x11);
        let (kept, kept_expiry, _) = cell(1, 3, 0x22);
        let asked = std::cell::Cell::new(0usize);
        let merged = merge_sweep_overlay(
            |widened| {
                asked.set(widened);
                vec![(gone, gone_expiry), (kept, kept_expiry)]
            },
            &[ancestor(vec![(gone, None)])],
            SweepFrontier::ZERO,
            all(),
            1,
        );
        assert_eq!(asked.get(), 2, "one retired, so one more was asked for");
        assert_eq!(merged, vec![(kept, kept_expiry)]);
    }

    /// A write outside the interval changes nothing, whichever side of
    /// it the write falls on.
    #[test]
    fn the_overlay_ignores_writes_outside_the_interval() {
        install_stub_protocol_statics();
        let (below, _, below_value) = cell(1, 2, 0x11);
        let (above, _, above_value) = cell(1, 9, 0x22);
        let merged = merge_sweep_overlay(
            |_| Vec::new(),
            &[ancestor(vec![
                (below, Some(below_value)),
                (above, Some(above_value)),
            ])],
            SweepFrontier::start_of(SweepBucket(3)),
            SweepFrontier::start_of(SweepBucket(9)),
            10,
        );
        assert!(merged.is_empty());
    }
}
