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

use hyperscale_jmt::NibblePath;
use hyperscale_types::{
    Block, MAX_SWEEP_PER_BLOCK, SettledWrites, ShardId, ShardTrie, StoredReceipt, SubstateKey,
    SweepBucket, SweepFrontier, Transaction, TxHash, WeightedTimestamp, protocol_statics,
    protocol_statics_installed,
};
use hyperscale_vm_effects::{CommittedTxCell, ProtocolHasher, committed_tx_key};
use hyperscale_vm_types::NULLIFIER_GRACE_MS;

use crate::tree::JmtSnapshot;
use crate::{
    Anchored, filter_state_writes_to_prefix, filter_writes_to_prefix, merge_receipts,
    merge_writes_from_receipts, settle_writes,
};

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

/// Whether a committed cell is an escrow record — value the shard holds
/// for a crossing it issued.
///
/// The same seam as [`sweepable_expiry`] and its complement: a record is
/// outside every sweep's reach by construction, so a store that inherits
/// a prefix whole has nothing else to tell it that the leaf it just
/// imported is an obligation. Lives here because both questions are the
/// same question about a leaf — which family wrote it — asked of the one
/// authority that can answer from the bytes.
#[must_use]
pub fn is_record_cell(key: SubstateKey, value: &[u8]) -> bool {
    protocol_statics_installed()
        && protocol_statics().record_cell(key.owner.to_bytes(), key.local.0, value)
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

/// Fold what the block itself writes — the committed-transaction cells
/// it creates and the sweep's removals — into the settled set its
/// receipts produced.
///
/// Both are ordinary writes, a value at the key or `None` at it, so
/// neither needs a commit path of its own: they fold with everything
/// else and land under `state_root` like any other change. That is the
/// fact that makes this tractable rather than novel.
///
/// # Panics
///
/// If a creation or a removal names a cell the block's own receipts
/// also write. The two would be one entry in the settled map, so one of
/// them would silently not happen — and which one is decided by
/// insertion order, which is not something a validator can check. A
/// creation sits under the shard's own owner, which no receipt writes,
/// and validation refuses a block sweeping what it writes before
/// anything reaches here; this is the assertion that both hold.
#[must_use]
pub fn with_sweep(
    settled: SettledWrites,
    creations: &[(SubstateKey, Vec<u8>)],
    removals: &[SubstateKey],
) -> SettledWrites {
    if creations.is_empty() && removals.is_empty() {
        return settled;
    }
    let (mut cells, entries) = settled.into_parts();
    for (key, value) in creations {
        assert!(
            cells.insert(*key, Some(value.clone())).is_none(),
            "the chain created {key:?}, which this block's receipts also write",
        );
    }
    for key in removals {
        assert!(
            cells.insert(*key, None).is_none(),
            "a sweep removed {key:?}, which this block's receipts also write",
        );
    }
    SettledWrites::from_parts(cells, entries)
}

/// The cells a followed block removed: everything sweepable strictly
/// after `parent_frontier` and at or below `frontier`, the position the
/// block's header claims its sweep landed on.
///
/// A follower holds a prefix of the chain's tree and no cap of its own:
/// it removes exactly what the header's interval names, whether that
/// interval ended at a ceiling or on the last cell a capped walk took.
#[must_use]
pub fn sweep_through(
    store: &(impl SweepIndex + ?Sized),
    parent_frontier: SweepFrontier,
    frontier: SweepFrontier,
) -> Vec<SubstateKey> {
    if parent_frontier >= frontier {
        return Vec::new();
    }
    let past = SweepFrontier::start_of(SweepBucket(frontier.bucket().0.saturating_add(1)));
    store
        .sweep_candidates(parent_frontier, past, usize::MAX)
        .into_iter()
        .map(|(key, _)| key)
        .filter(|key| SweepFrontier::of_leaf(*key) <= frontier)
        .collect()
}

/// What a block writes, composed once for every path that commits one.
///
/// The receipts its ticks settled, resolved against `prior`; the
/// committed cell of every transaction it carries; and the `removals`
/// its sweep names.
#[must_use]
pub fn block_settled_writes(
    block: &Block,
    prior: &dyn Anchored,
    removals: &[SubstateKey],
) -> SettledWrites {
    let settling: Vec<StoredReceipt> = block
        .certificates()
        .iter()
        .flat_map(|fw| fw.settling_receipts())
        .collect();
    let creations = committed_tx_cells(
        block.header().shard_id(),
        block.transactions().iter().map(|tx| tx.as_unverified()),
    );
    with_sweep(
        merge_writes_from_receipts(&settling, prior),
        &creations,
        removals,
    )
}

/// What a followed block writes under `prefix`, composed as the chain
/// composed it: the receipts its ticks settled, the committed cell of
/// every transaction it carries, and the sweep its header names.
///
/// The removals read `store` as it stands before the block, from the
/// bottom of the sweep order: a follower mirrors the chain's state, so
/// every live cell at or below the header's frontier is one the block
/// removed, and one the block created sits far above it.
#[must_use]
pub fn followed_block_writes(
    store: &(impl SweepIndex + ?Sized),
    prior: &dyn Anchored,
    block: &Block,
    prefix: &NibblePath,
) -> SettledWrites {
    let settling: Vec<StoredReceipt> = block
        .certificates()
        .iter()
        .flat_map(|fw| fw.settling_receipts())
        .collect();
    // Restricted before resolving: a follower holds its prefix of the
    // tree and nothing else, so a movement on any other cell reads an
    // empty prior here and is the owning store's to judge, not this one's.
    let merged = settle_writes(
        &filter_state_writes_to_prefix(&merge_receipts(&settling), prefix),
        prior,
    );
    let creations = committed_tx_cells(
        block.header().shard_id(),
        block.transactions().iter().map(|tx| tx.as_unverified()),
    );
    let removals = sweep_through(store, SweepFrontier::ZERO, block.header().sweep_frontier());
    filter_writes_to_prefix(&with_sweep(merged, &creations, &removals), prefix)
}

/// The committed-transaction cells a block of `local_shard` creates: one
/// per transaction it carries, under the shard's own owner, expiring at
/// the transaction's validity end plus the grace.
///
/// What makes a shard's committed set provable and refutable against the
/// state root every header carries. Derived from the block's own
/// transactions and the shard's identity, so the proposer and every
/// verifier fold the same cells, and a prober holding the transaction
/// derives the same key from nothing else.
#[must_use]
pub fn committed_tx_cells<'a>(
    local_shard: ShardId,
    transactions: impl IntoIterator<Item = &'a Transaction>,
) -> Vec<(SubstateKey, Vec<u8>)> {
    transactions
        .into_iter()
        .map(|tx| {
            let validity_end = tx.validity_range().end_timestamp_exclusive;
            let cell = CommittedTxCell {
                tx: tx.hash(),
                expiry_ms: committed_tx_expiry_ms(validity_end),
            };
            (
                committed_tx_cell_key(local_shard, cell.tx, validity_end),
                cell.to_bytes(),
            )
        })
        .collect()
}

/// The key `shard` writes its committed cell for `tx` under.
///
/// Derived from signed content and the shard alone, so a prober asking
/// whether a shard committed a transaction names the same cell the
/// shard's own commit wrote, from nothing but the transaction and the
/// shard.
#[must_use]
pub fn committed_tx_cell_key(
    shard: ShardId,
    tx: TxHash,
    validity_end: WeightedTimestamp,
) -> SubstateKey {
    committed_tx_key(
        &ProtocolHasher,
        ShardTrie::shard_owner(shard),
        tx,
        committed_tx_expiry_ms(validity_end),
    )
}

/// When a committed cell stops being needed: the transaction's validity
/// end plus the grace, on the nullifier's clock.
const fn committed_tx_expiry_ms(validity_end: WeightedTimestamp) -> u64 {
    validity_end.as_millis().saturating_add(NULLIFIER_GRACE_MS)
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

    /// The chain's own creations fold in beside the sweep's removals, as
    /// ordinary writes under the same root.
    #[test]
    fn creations_fold_in_beside_removals() {
        let (created, _, value) = cell(1, 3, 0xC1);
        let (removed, _, _) = cell(2, 1, 0xD1);
        let settled = with_sweep(
            SettledWrites::default(),
            &[(created, value.clone())],
            &[removed],
        );
        assert_eq!(settled.cells().get(&created), Some(&Some(value)));
        assert_eq!(settled.cells().get(&removed), Some(&None));
    }

    /// One committed cell per transaction, under the shard's own owner,
    /// self-describing as sweepable at the transaction's validity end
    /// plus the grace — the same cells whether a proposer or a verifier
    /// derives them.
    #[test]
    fn a_block_creates_one_committed_cell_per_transaction() {
        use hyperscale_hbor::from_slice;
        use hyperscale_types::test_utils::{install_stub_protocol_statics, test_transaction};
        use hyperscale_vm_effects::CommittedTxCell;
        use hyperscale_vm_types::NULLIFIER_GRACE_MS;

        install_stub_protocol_statics();
        let shard = ShardId::leaf(1, 1);
        let txs = [test_transaction(1), test_transaction(2)];
        let cells = committed_tx_cells(shard, txs.iter());
        assert_eq!(cells.len(), 2);
        for ((key, value), tx) in cells.iter().zip(&txs) {
            assert!(ShardTrie::shard_owns_prefix(shard, key.owner));
            let decoded: CommittedTxCell = from_slice(value).expect("decodes");
            assert_eq!(decoded.tx, tx.hash());
            assert_eq!(
                decoded.expiry_ms,
                tx.validity_range().end_timestamp_exclusive.as_millis() + NULLIFIER_GRACE_MS
            );
        }
        assert_eq!(
            cells,
            committed_tx_cells(shard, txs.iter()),
            "derived the same way on every replica"
        );
    }
}
