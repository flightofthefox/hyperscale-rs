//! Execution-tick output chain.
//!
//! One tick per committed block that carries executable work: the block's
//! single-shard transactions plus every cross-shard wave whose provisions
//! completed at that commit, executed as one batch. Each tick's output is
//! the next tick's baseline, overlaid on the persisted base — execution
//! dispatch reads through [`TickChain::view_at`] instead of the
//! settlement-derived [`crate::PendingChain`] overlay.
//!
//! A tick is clocked by commits, never by time. It is one execution
//! batch, and a block carrying no work produces none — so the ticks a
//! chain holds are neither periodic nor one per height, and the timer the
//! word suggests elsewhere in this workspace has nothing to do with it.
//!
//! A tick's output is a pure function of the committed chain prefix, so
//! the chain is a deterministic cache: any replica rebuilds it by
//! replaying committed blocks forward from the persisted tip.
//!
//! **Provisional entries are never readable.** A cross-shard
//! transaction's local writes sit beside the determined fold as per-wave
//! provisional entries until the wave's certificate commits. Resolution
//! either promotes them into the readable fold (settled) or drops them
//! (aborted) without recomputing any tick output — no chained output ever
//! depends on an entry that resolution changes.
//!
//! **A contribution is readable until the base carries it, and never
//! both.** A receipt says what an exclusive write left and what a
//! commutative access moved, and the second of those does not survive
//! being applied twice. So each contribution records the height its
//! wave's settlement reaches the base at, and a read folds it only while
//! its anchor sits below that height. Overlap is not idempotent and is
//! therefore not permitted.
//!
//! **This overlay never feeds state-root computation.** State roots stay
//! settlement-derived through `PendingChain`; `TickChain` deliberately
//! implements none of the commit-pipeline surfaces (`ShardChainWriter`,
//! `TreeReader`), so the two overlays cannot be cross-fed.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, RwLock};

use hyperscale_types::{
    BlockHeight, ProvisionalHolds, StateWrites, SubstateKey, TxHash, WaveId, amount_cell,
    read_amount,
};

use crate::lock_recover::{read_or_recover, write_or_recover};
use crate::shard::writes::fold_state_writes;
use crate::{SubstateDatabase, VersionedStore};

/// One cross-shard transaction's provisional contribution to a tick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvisionalTx {
    /// Transaction whose wave verdict decides which side settles.
    pub tx_hash: TxHash,
    /// Local execution writes; promoted when the wave settles this
    /// transaction as accepted. `None` for a failed attempt, which
    /// produced no effects.
    pub writes: Option<StateWrites>,
    /// The payer-vault charge held beside the effects — the abort floor a
    /// completed leg owes if its wave aborts, or the class charge of a
    /// failed attempt. Promoted when the wave settles this transaction as
    /// aborted (the substitute receipt), dropped with everything else if
    /// the wave never finalizes.
    pub reserve: Option<StateWrites>,
    /// What this leg holds against each amount cell while its wave is
    /// unresolved — its declared reservations.
    ///
    /// Recorded here rather than tracked beside the chain so the hold and
    /// the writes it stands for share one lifetime: both are promoted or
    /// dropped by the same resolution, and a reader can never see a
    /// balance with the debit already in it while the hold still says the
    /// value is spoken for.
    pub reserved: BTreeMap<SubstateKey, u128>,
}

/// The execution output of one tick.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickOutput {
    /// What each single-shard member left, by the wave it belongs to and
    /// in the canonical order the batch fold produced them in —
    /// absolutes where an exclusive write stated the value, movements
    /// where a commutative access said what it moved, unconditional fee
    /// burns included. Readable by every subsequent tick immediately.
    ///
    /// Keyed by wave for the same reason the provisional side is: a tick
    /// can carry more than one single-shard wave — its block's own, plus
    /// any whose members an earlier tick deferred — and each reaches the
    /// persisted base when its *own* settling block persists.
    pub determined: BTreeMap<WaveId, Vec<(TxHash, StateWrites)>>,
    /// Per-wave provisional contributions, unreadable until the wave
    /// resolves.
    pub provisional: BTreeMap<WaveId, Vec<ProvisionalTx>>,
}

/// How a wave's fate became known from chain content.
#[derive(Clone, Debug)]
pub enum TickResolution {
    /// The wave's certificate committed. Transactions in `aborted` settle
    /// their reserve charge; every other member settles its execution
    /// writes.
    Settled {
        /// Height of the block that committed the certificate.
        height: BlockHeight,
        /// Members whose verdict discards their execution effects.
        aborted: BTreeSet<TxHash>,
    },
    /// The wave will never finalize (counterpart terminated, gate
    /// rejection). Nothing settles; its provisional entries drop.
    Aborted {
        /// Height at which the abort became known.
        height: BlockHeight,
    },
}

/// One transaction's readable contribution to a tick.
struct Contribution {
    /// The wave whose verdict governs it, and so the wave whose
    /// settlement puts these writes into the base.
    wave: WaveId,
    /// What the transaction left, in the form its receipt states it.
    writes: StateWrites,
    /// The height the settled base gains these writes at, known once the
    /// wave's fate commits.
    ///
    /// A read anchored below it needs the fold; at or above it the base
    /// already carries the same change. Folding both would leave an
    /// exclusive write unchanged and a movement applied twice, which is
    /// why this is a height rather than a flag.
    in_base_from: Option<BlockHeight>,
}

/// One tick's retained state.
struct TickEntry {
    /// Readable contributions in canonical transaction order — the order
    /// the batch fold that produced them ran in. The single-shard wave's
    /// members are here from the append; a cross-shard wave's arrive when
    /// its verdict promotes them.
    readable: BTreeMap<TxHash, Contribution>,
    /// Waves whose fate is still unknown. Includes the single-shard wave
    /// (with no provisional entries) so eviction waits for its
    /// settlement.
    pending: HashMap<WaveId, Vec<ProvisionalTx>>,
}

impl TickEntry {
    /// The entry one tick's output produces.
    fn from_output(output: TickOutput) -> Self {
        let mut entry = Self {
            readable: BTreeMap::new(),
            pending: HashMap::new(),
        };
        for (wave, txs) in output.determined {
            for (tx_hash, writes) in txs {
                entry.readable.insert(
                    tx_hash,
                    Contribution {
                        wave: wave.clone(),
                        writes,
                        in_base_from: None,
                    },
                );
            }
            entry.pending.insert(wave, Vec::new());
        }
        for (wave, txs) in output.provisional {
            entry.pending.insert(wave, txs);
        }
        entry
    }

    /// Apply one wave's verdict: promote each member's surviving side
    /// into the readable fold, or drop the wave's entries outright, and
    /// record the height the base gains whatever survives at.
    /// A no-op for a wave already resolved.
    fn resolve(&mut self, wave_id: &WaveId, resolution: &TickResolution) {
        let Some(txs) = self.pending.remove(wave_id) else {
            return;
        };
        match resolution {
            TickResolution::Settled { height, aborted } => {
                for tx in txs {
                    let promoted = if aborted.contains(&tx.tx_hash) {
                        tx.reserve
                    } else {
                        tx.writes
                    };
                    if let Some(writes) = promoted {
                        self.readable.insert(
                            tx.tx_hash,
                            Contribution {
                                wave: wave_id.clone(),
                                writes,
                                in_base_from: None,
                            },
                        );
                    }
                }
                // The base gains everything this wave settled, at the
                // height that settled it.
                for contribution in self.readable.values_mut() {
                    if contribution.wave == *wave_id {
                        contribution.in_base_from = Some(*height);
                    }
                }
            }
            // Nothing settles, so nothing enters the base and no height
            // is recorded. Only a wave whose whole contribution is
            // provisional can take this path: its entries were never
            // readable, so dropping them changes no baseline any tick has
            // already read.
            //
            // A readable fold here would have no correct handling. Later
            // ticks have read those writes and no base will ever carry
            // them, so stamping a height would drop them from a later
            // baseline while an earlier one kept them, and leaving them
            // unstamped would pin the tick forever. The rule is upstream:
            // abandonment is a verdict about a cross-shard wave, and a
            // wave with determined output is not one.
            TickResolution::Aborted { .. } => {
                if self.readable.values().any(|c| c.wave == *wave_id) {
                    // Left folded rather than dropped or stamped: every
                    // read keeps agreeing with the ones before it, and
                    // the cost is a tick this entry pins.
                    tracing::error!(
                        wave = %wave_id,
                        "abandoned a wave whose fold later ticks have read"
                    );
                    debug_assert!(false, "a wave with a readable fold cannot be abandoned");
                }
            }
        }
    }

    /// Whether the settled base at `floor` already carries everything
    /// this entry holds, so no future read can still need it.
    fn covered_by(&self, floor: BlockHeight) -> bool {
        self.pending.is_empty()
            && self
                .readable
                .values()
                .all(|c| c.in_base_from.is_some_and(|height| height <= floor))
    }
}

/// Deterministic chain of tick outputs, shared between the shard loop and
/// dispatch closures via `Arc`.
///
/// Ticks are keyed by the committing block's height. Blocks that carry no
/// executable work produce no entry; [`Self::view_at`] folds every
/// retained tick at or below its anchor.
pub struct TickChain<S> {
    base: Arc<S>,
    entries: RwLock<BTreeMap<BlockHeight, TickEntry>>,
    /// The highest tick whose output has been appended. Ticks execute
    /// serially in height order, so the next tick to run anchors at or
    /// above this — which is what makes it the floor eviction may not
    /// outrun.
    executed: RwLock<BlockHeight>,
    /// The highest height the settled base is known to carry. Read
    /// beside the entries so a contribution is folded exactly while the
    /// base is missing it, and evicted once the base is not.
    persisted: RwLock<BlockHeight>,
}

impl<S> TickChain<S>
where
    S: VersionedStore,
{
    /// Create an empty tick chain over the given base storage.
    ///
    /// Seeded from the store's committed tip, not from genesis: a
    /// restarted node's base already carries everything up to it, and a
    /// read anchored below the store's retention floor is a panic.
    pub fn new(base: Arc<S>) -> Self {
        let persisted = base.jmt_height();
        Self {
            base,
            entries: RwLock::new(BTreeMap::new()),
            executed: RwLock::new(BlockHeight::GENESIS),
            persisted: RwLock::new(persisted),
        }
    }

    /// Append the output of the tick at `height`.
    ///
    /// Same-shard vnodes share one chain and each executes the tick under
    /// its own validator identity, so a height can be appended more than
    /// once. A tick's output is a pure function of the committed chain
    /// prefix, so every repeat carries byte-identical content and the
    /// first append is the whole of it — which is why a repeat is
    /// dropped rather than folded. Folding it would leave an exclusive
    /// write unchanged and apply every movement it carries a second
    /// time, and it would re-add waves the chain has since resolved.
    pub fn append(&self, height: BlockHeight, output: TickOutput) {
        write_or_recover(&self.entries)
            .entry(height)
            .or_insert_with(|| TickEntry::from_output(output));
        let mut executed = write_or_recover(&self.executed);
        *executed = (*executed).max(height);
    }

    /// Resolve a wave's fate: a settled verdict promotes each member's
    /// surviving side into the readable fold, an abort drops the wave's
    /// entries.
    ///
    /// Every retained tick is searched rather than the one at the wave's
    /// own height. A wave joins whichever tick it became executable at —
    /// later than its origin block when its provisions arrived late — and
    /// contributes to more than one when a member waits on a cell another
    /// wave holds provisionally. Resolving the wrong entry would leave the
    /// promotion unapplied and the tick unevictable.
    ///
    /// Idempotent and tolerant of unknown waves: every entry may already
    /// be evicted (every wave resolved and persisted) or torn down at a
    /// reshape boundary.
    pub fn resolve(&self, wave_id: &WaveId, resolution: &TickResolution) {
        for entry in write_or_recover(&self.entries).values_mut() {
            entry.resolve(wave_id, resolution);
        }
    }

    /// Record what the base now carries and drop every tick a future read
    /// can no longer need. Called on `BlockPersisted`.
    ///
    /// The eviction floor is the lower of what has persisted and what has
    /// executed, and the second half is load-bearing. A contribution
    /// enters the base at the height its wave *settled*, which can be
    /// above the anchor a still-queued tick will read from — so dropping
    /// on persistence alone lets eviction outrun a lagging tick queue,
    /// and that tick then finds neither the fold nor a base old enough to
    /// hold it. Two replicas at different execution positions would
    /// derive different receipts from one committed chain.
    pub fn prune_persisted(&self, persisted: BlockHeight) {
        {
            let mut recorded = write_or_recover(&self.persisted);
            *recorded = (*recorded).max(persisted);
        }
        let floor = persisted.min(*read_or_recover(&self.executed));
        write_or_recover(&self.entries).retain(|_, entry| !entry.covered_by(floor));
    }

    /// Tear the chain down. A reshape boundary terminates the shard's
    /// chain; successors seed from settled state, never from tick
    /// outputs.
    pub fn clear(&self) {
        write_or_recover(&self.entries).clear();
    }

    /// Number of retained ticks (diagnostics).
    #[must_use]
    pub fn len(&self) -> usize {
        read_or_recover(&self.entries).len()
    }

    /// Whether any tick is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        read_or_recover(&self.entries).is_empty()
    }

    /// Build the baseline view for the tick after `tick`: every retained
    /// contribution at or below `tick` the base does not already carry,
    /// over the base as of the height it is read at.
    ///
    /// The base is read at `min(tick, persisted)` rather than at the live
    /// tip. Execution runs behind consensus, so a replica whose tick
    /// queue lags has already persisted settlements belonging to *later*
    /// ticks, which 2f+1 other replicas certified without waiting for it;
    /// reading the live base would fold those into this tick's baseline
    /// and split its receipts from the committee's.
    ///
    /// Anchoring alone is not enough, because a read at or above the
    /// persisted tip sees everything the base holds while the overlay may
    /// still hold the same contribution — harmless for an absolute and
    /// wrong for a movement. So the two are made disjoint instead: the
    /// base covers every contribution settled at or below the read
    /// height, the overlay covers the rest, and the answer is the same
    /// whichever side of that line a replica's persistence happens to
    /// sit.
    pub fn view_at(&self, tick: BlockHeight) -> TickView<S> {
        let mut overlay = StateWrites::default();
        let mut holds = ProvisionalHolds::new();
        // One acquisition for the base height, the overlay and the holds.
        // A resolution landing between two reads would drop a leg's hold
        // without the debit it stood for reaching the overlay, and the
        // reader would spend a balance twice.
        let entries = read_or_recover(&self.entries);
        let base_at = tick.min(*read_or_recover(&self.persisted));
        // Ascending, because an exclusive write supersedes what stood
        // before it and a movement composes onto it — an order the fold
        // has to respect even though movements commute among themselves.
        for (_, entry) in entries.range(..=tick) {
            for contribution in entry.readable.values() {
                if contribution
                    .in_base_from
                    .is_some_and(|height| height <= base_at)
                {
                    continue;
                }
                fold_state_writes(&mut overlay, &contribution.writes);
            }
            for leg in entry.pending.values().flatten() {
                for (cell, amount) in &leg.reserved {
                    *holds
                        .entry(*cell)
                        .or_default()
                        .entry(leg.tx_hash)
                        .or_default() += *amount;
                }
            }
        }
        drop(entries);
        TickView {
            base: Arc::clone(&self.base),
            anchor: base_at,
            overlay: Arc::new(overlay),
            holds: Arc::new(holds),
        }
    }
}

/// Read view over the tick chain's folds at one anchor.
///
/// Falls through to the base as of that anchor. The engine's eager
/// pre-read consumes this via [`Self::snapshot`], which is the view's
/// only read path — a baseline that could be read unanchored would not
/// be deterministic.
pub struct TickView<S> {
    base: Arc<S>,
    anchor: BlockHeight,
    overlay: Arc<StateWrites>,
    holds: Arc<ProvisionalHolds>,
}

impl<S: VersionedStore> TickView<S> {
    /// Snapshot for batch execution: the folds over the base as of this
    /// view's anchor height.
    #[must_use]
    pub fn snapshot(&self) -> TickViewSnapshot<S::Snapshot<'_>> {
        TickViewSnapshot {
            base_snapshot: self.base.snapshot_at(self.anchor),
            overlay: Arc::clone(&self.overlay),
        }
    }

    /// What unresolved legs hold against the cells this view reads.
    ///
    /// The companion to the overlay: the overlay is what a reader may
    /// see, and this is what it may not spend of what it sees.
    #[must_use]
    pub fn holds(&self) -> Arc<ProvisionalHolds> {
        Arc::clone(&self.holds)
    }
}

/// Snapshot from a [`TickView`] — the same overlay on the base storage's
/// snapshot.
pub struct TickViewSnapshot<Snap> {
    base_snapshot: Snap,
    overlay: Arc<StateWrites>,
}

impl<Snap: SubstateDatabase> SubstateDatabase for TickViewSnapshot<Snap> {
    fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
        let written = self.overlay.cells.get(&key);
        let Some(movement) = self.overlay.movements.get(&key) else {
            return written.map_or_else(|| self.base_snapshot.substate(key), Clone::clone);
        };
        // A cell the folds only moved resolves here, where the base it
        // moved from is finally in reach; one an exclusive write also
        // reached moves from that write instead.
        let before = written
            .cloned()
            .unwrap_or_else(|| self.base_snapshot.substate(key))
            .as_deref()
            .and_then(read_amount)
            .unwrap_or(0);
        amount_cell(movement.apply(before).unwrap_or(0)).map(|cell| cell.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use hyperscale_types::{
        Address, Hash, LocalKey, MerkleInclusionProof, Movement, ShardId, StateRoot, encode_amount,
    };

    use super::*;
    use crate::SubstateStore;
    use crate::lock_recover::lock_or_recover;

    /// A base built from settled write sets at heights, readable as of
    /// any anchor — the store's history without the JMT.
    struct StubStore {
        history: BTreeMap<BlockHeight, HashMap<SubstateKey, Vec<u8>>>,
        tip: BlockHeight,
        /// Anchor heights `snapshot_at` was asked for, so a test can pin
        /// which version a tick read the base at.
        anchors: Mutex<Vec<BlockHeight>>,
    }

    impl StubStore {
        /// A base holding `value` at `key` from genesis, with nothing
        /// settled since.
        fn with_cell(key: SubstateKey, value: &[u8]) -> Self {
            Self::settling(key, value, BlockHeight::GENESIS, value)
        }

        /// A base holding `genesis` at `key`, which a settlement replaces
        /// with `settled` at `height` — the store's tip.
        fn settling(key: SubstateKey, genesis: &[u8], height: BlockHeight, settled: &[u8]) -> Self {
            Self {
                history: BTreeMap::from([
                    (
                        BlockHeight::GENESIS,
                        HashMap::from([(key, genesis.to_vec())]),
                    ),
                    (height, HashMap::from([(key, settled.to_vec())])),
                ]),
                tip: height,
                anchors: Mutex::new(Vec::new()),
            }
        }

        fn cells_at(&self, height: BlockHeight) -> HashMap<SubstateKey, Vec<u8>> {
            let mut cells = HashMap::new();
            for (_, settled) in self.history.range(..=height) {
                cells.extend(settled.iter().map(|(k, v)| (*k, v.clone())));
            }
            cells
        }
    }

    impl SubstateDatabase for StubStore {
        fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
            self.cells_at(self.tip).get(&key).cloned()
        }
    }

    struct StubSnapshot(HashMap<SubstateKey, Vec<u8>>);
    impl SubstateDatabase for StubSnapshot {
        fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
            self.0.get(&key).cloned()
        }
    }

    impl VersionedStore for StubStore {
        fn snapshot_at(&self, height: BlockHeight) -> Self::Snapshot<'_> {
            lock_or_recover(&self.anchors).push(height);
            StubSnapshot(self.cells_at(height))
        }
        fn substate_bytes_at(&self, _height: BlockHeight) -> Option<u64> {
            None
        }
    }

    impl SubstateStore for StubStore {
        type Snapshot<'a> = StubSnapshot;
        fn snapshot(&self) -> Self::Snapshot<'_> {
            StubSnapshot(self.cells_at(self.tip))
        }
        fn jmt_height(&self) -> BlockHeight {
            self.tip
        }
        fn state_root(&self) -> StateRoot {
            StateRoot::ZERO
        }
        fn get_substate_at_height(
            &self,
            _key: SubstateKey,
            _block_height: BlockHeight,
        ) -> Option<Option<Vec<u8>>> {
            None
        }
        fn generate_merkle_proofs(
            &self,
            _keys: &[SubstateKey],
            _block_height: BlockHeight,
        ) -> Option<MerkleInclusionProof> {
            None
        }
    }

    fn key(byte: u8) -> SubstateKey {
        SubstateKey {
            owner: Address([byte; 16]),
            local: LocalKey([byte; 16]),
        }
    }

    fn writes(entries: &[(SubstateKey, Option<&[u8]>)]) -> StateWrites {
        StateWrites {
            cells: entries
                .iter()
                .map(|(k, v)| (*k, v.map(<[u8]>::to_vec)))
                .collect(),
            movements: BTreeMap::new(),
        }
    }

    /// A receipt that debits `amount` from `cell` — what every fee burn
    /// and every commutative balance change actually carries.
    fn debit(cell: SubstateKey, amount: u128) -> StateWrites {
        let mut moved = StateWrites::default();
        moved.movements.insert(
            cell,
            Movement {
                credit: 0,
                debit: amount,
            },
        );
        moved
    }

    fn amount(cell: &[u8]) -> u128 {
        read_amount(cell).expect("an amount cell")
    }

    fn wave(height: u64, remote: &[u64]) -> WaveId {
        WaveId::new(
            ShardId::leaf(0, 0),
            BlockHeight::new(height),
            remote.iter().map(|s| ShardId::leaf(2, *s)).collect(),
        )
    }

    fn tx(byte: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[byte]))
    }

    #[test]
    fn determined_writes_chain_and_shadow_base() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: BTreeMap::from([(
                    wave(1, &[]),
                    vec![(
                        tx(1),
                        writes(&[(key(1), Some(b"one")), (key(2), Some(b"two"))]),
                    )],
                )]),
                provisional: BTreeMap::new(),
            },
        );
        chain.append(
            BlockHeight::new(2),
            TickOutput {
                determined: BTreeMap::from([(
                    wave(2, &[]),
                    vec![(tx(2), writes(&[(key(1), None)]))],
                )]),
                provisional: BTreeMap::new(),
            },
        );

        // Anchored below tick 2: tick 1's write shadows base.
        let view = chain.view_at(BlockHeight::new(1));
        assert_eq!(view.snapshot().substate(key(1)), Some(b"one".to_vec()));
        assert_eq!(view.snapshot().substate(key(2)), Some(b"two".to_vec()));

        // Anchored at tick 2: the removal wins; untouched key falls through.
        let view = chain.view_at(BlockHeight::new(2));
        assert_eq!(view.snapshot().substate(key(1)), None);
        assert_eq!(view.snapshot().substate(key(2)), Some(b"two".to_vec()));
    }

    /// A determined fold carries what its receipts carry, movements
    /// included. A fee burn is a debit and every payment moves a balance,
    /// so a fold that kept only the absolutes would hold nothing at all
    /// for ordinary traffic — and the next tick would read a balance its
    /// predecessor had already spent.
    #[test]
    fn a_determined_movement_reaches_the_next_tick() {
        let store = Arc::new(StubStore::with_cell(key(1), encode_amount(1_000).as_ref()));
        let chain = TickChain::new(store);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: BTreeMap::from([(wave(1, &[]), vec![(tx(1), debit(key(1), 300))])]),
                provisional: BTreeMap::new(),
            },
        );

        let view = chain.view_at(BlockHeight::new(1));
        assert_eq!(amount(&view.snapshot().substate(key(1)).unwrap()), 700);
    }

    /// A tick reads the base no later than its own height, and no later
    /// than what has persisted. The first keeps a lagging replica — which
    /// has already persisted settlements belonging to ticks it has not
    /// run — from folding them into an earlier tick's baseline. The
    /// second is what makes the base and the overlay disjoint: a read
    /// above the persisted tip sees everything the base holds, so the
    /// overlay has to know exactly where that line is.
    #[test]
    fn a_tick_reads_the_base_no_later_than_its_own_height() {
        let store = Arc::new(StubStore::settling(
            key(1),
            b"base",
            BlockHeight::new(9),
            b"settled",
        ));
        let chain = TickChain::new(Arc::clone(&store));
        chain.prune_persisted(BlockHeight::new(9));

        let _ = chain.view_at(BlockHeight::new(4)).snapshot();
        let _ = chain.view_at(BlockHeight::new(12)).snapshot();

        assert_eq!(
            *lock_or_recover(&store.anchors),
            vec![BlockHeight::new(4), BlockHeight::new(9)],
            "the anchor is the tick, clamped to what has persisted"
        );
    }

    #[test]
    fn provisional_entries_unreadable_until_settled() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let w = wave(1, &[2]);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: BTreeMap::new(),
                provisional: BTreeMap::from([(
                    w.clone(),
                    vec![ProvisionalTx {
                        tx_hash: tx(7),
                        writes: Some(writes(&[(key(1), Some(b"provisional"))])),
                        reserve: None,
                        reserved: BTreeMap::new(),
                    }],
                )]),
            },
        );

        let view = chain.view_at(BlockHeight::new(1));
        assert_eq!(view.snapshot().substate(key(1)), Some(b"base".to_vec()));

        chain.resolve(
            &w,
            &TickResolution::Settled {
                height: BlockHeight::new(3),
                aborted: BTreeSet::new(),
            },
        );
        let view = chain.view_at(BlockHeight::new(1));
        assert_eq!(
            view.snapshot().substate(key(1)),
            Some(b"provisional".to_vec())
        );
    }

    /// A wave joins whichever tick it became executable at, which is not
    /// its origin block when provisions arrived late, and it contributes
    /// to two when a member waited on a provisional cell. Resolution has
    /// to reach every entry holding it, or the promotion is dropped and
    /// the tick never evicts.
    #[test]
    fn resolution_reaches_every_tick_the_wave_contributed_to() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let w = wave(1, &[2]);
        for (height, cell, member) in [(4u64, key(1), 7u8), (6, key(2), 8)] {
            chain.append(
                BlockHeight::new(height),
                TickOutput {
                    determined: BTreeMap::new(),
                    provisional: BTreeMap::from([(
                        w.clone(),
                        vec![ProvisionalTx {
                            tx_hash: tx(member),
                            writes: Some(writes(&[(cell, Some(b"promoted"))])),
                            reserve: None,
                            reserved: BTreeMap::new(),
                        }],
                    )]),
                },
            );
        }

        chain.resolve(
            &w,
            &TickResolution::Settled {
                height: BlockHeight::new(6),
                aborted: BTreeSet::new(),
            },
        );

        let view = chain.view_at(BlockHeight::new(6));
        assert_eq!(view.snapshot().substate(key(1)), Some(b"promoted".to_vec()));
        assert_eq!(view.snapshot().substate(key(2)), Some(b"promoted".to_vec()));
        chain.prune_persisted(BlockHeight::new(6));
        assert!(chain.is_empty(), "both entries must become evictable");
    }

    #[test]
    fn aborted_member_settles_reserve_not_writes() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let w = wave(1, &[2]);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: BTreeMap::new(),
                provisional: BTreeMap::from([(
                    w.clone(),
                    vec![ProvisionalTx {
                        tx_hash: tx(7),
                        writes: Some(writes(&[(key(1), Some(b"effects"))])),
                        reserve: Some(writes(&[(key(9), Some(b"floor"))])),
                        reserved: BTreeMap::new(),
                    }],
                )]),
            },
        );

        chain.resolve(
            &w,
            &TickResolution::Settled {
                height: BlockHeight::new(3),
                aborted: BTreeSet::from([tx(7)]),
            },
        );
        let view = chain.view_at(BlockHeight::new(1));
        assert_eq!(view.snapshot().substate(key(1)), Some(b"base".to_vec()));
        assert_eq!(view.snapshot().substate(key(9)), Some(b"floor".to_vec()));
    }

    #[test]
    fn wholesale_abort_drops_everything() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let w = wave(1, &[2]);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: BTreeMap::new(),
                provisional: BTreeMap::from([(
                    w.clone(),
                    vec![ProvisionalTx {
                        tx_hash: tx(7),
                        writes: Some(writes(&[(key(1), Some(b"effects"))])),
                        reserve: Some(writes(&[(key(9), Some(b"floor"))])),
                        reserved: BTreeMap::new(),
                    }],
                )]),
            },
        );

        chain.resolve(
            &w,
            &TickResolution::Aborted {
                height: BlockHeight::new(3),
            },
        );
        let view = chain.view_at(BlockHeight::new(1));
        assert_eq!(view.snapshot().substate(key(1)), Some(b"base".to_vec()));
        assert_eq!(view.snapshot().substate(key(9)), None);
    }

    /// Same-shard vnodes share one chain and each appends the tick it
    /// executed. The outputs are byte-identical, so the second is
    /// dropped: refolding it would leave an absolute unchanged and apply
    /// every movement a second time.
    #[test]
    fn a_repeated_append_is_dropped_rather_than_refolded() {
        let store = Arc::new(StubStore::with_cell(key(1), encode_amount(1_000).as_ref()));
        let chain = TickChain::new(store);
        let output = || TickOutput {
            determined: BTreeMap::from([(wave(1, &[]), vec![(tx(1), debit(key(1), 100))])]),
            provisional: BTreeMap::new(),
        };
        chain.append(BlockHeight::new(1), output());
        chain.append(BlockHeight::new(1), output());

        let view = chain.view_at(BlockHeight::new(1));
        assert_eq!(amount(&view.snapshot().substate(key(1)).unwrap()), 900);
    }

    /// A fold and the base must never both carry one contribution. The
    /// eviction floor holds a settled fold while execution lags, and the
    /// base gains the same writes as soon as the settling block persists
    /// — so the two windows overlap, and the reader has to pick exactly
    /// one of them.
    #[test]
    fn a_retained_fold_stops_applying_once_the_base_carries_it() {
        let store = Arc::new(StubStore::settling(
            key(1),
            encode_amount(1_000).as_ref(),
            BlockHeight::new(8),
            encode_amount(900).as_ref(),
        ));
        let chain = TickChain::new(Arc::clone(&store));

        let w = wave(3, &[2]);
        chain.append(
            BlockHeight::new(3),
            TickOutput {
                determined: BTreeMap::new(),
                provisional: BTreeMap::from([(
                    w.clone(),
                    vec![ProvisionalTx {
                        tx_hash: tx(7),
                        writes: Some(debit(key(1), 100)),
                        reserve: None,
                        reserved: BTreeMap::new(),
                    }],
                )]),
            },
        );
        chain.resolve(
            &w,
            &TickResolution::Settled {
                height: BlockHeight::new(8),
                aborted: BTreeSet::new(),
            },
        );
        chain.prune_persisted(BlockHeight::new(10));
        assert_eq!(
            chain.len(),
            1,
            "execution has only reached tick 3, so the fold is retained"
        );

        // A queued tick anchored below the settlement needs the fold.
        let view = chain.view_at(BlockHeight::new(5));
        assert_eq!(amount(&view.snapshot().substate(key(1)).unwrap()), 900);

        // A later tick reads a base that already holds it, and must not
        // apply the debit again.
        let view = chain.view_at(BlockHeight::new(11));
        assert_eq!(amount(&view.snapshot().substate(key(1)).unwrap()), 900);
    }

    /// A fold survives until nothing can still need it: every wave in it
    /// resolved, the settlement persisted, and execution advanced past the
    /// settling height. The last is what keeps eviction from outrunning a
    /// queued tick, whose anchor can sit below the height the base gained
    /// the write at.
    #[test]
    fn eviction_waits_for_resolution_persistence_and_execution() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let w = wave(1, &[]);
        let append_at = |height: u64, wave: WaveId| {
            chain.append(
                BlockHeight::new(height),
                TickOutput {
                    determined: BTreeMap::from([(
                        wave,
                        vec![(tx(1), writes(&[(key(1), Some(b"one"))]))],
                    )]),
                    provisional: BTreeMap::new(),
                },
            );
        };
        append_at(1, w.clone());

        // Unresolved: survives any persistence progress.
        chain.prune_persisted(BlockHeight::new(10));
        assert_eq!(chain.len(), 1);

        chain.resolve(
            &w,
            &TickResolution::Settled {
                height: BlockHeight::new(3),
                aborted: BTreeSet::new(),
            },
        );

        // Resolved and persisted, but execution has only reached tick 1 —
        // the next tick anchors there, and the base gained the write at 3.
        chain.prune_persisted(BlockHeight::new(3));
        assert_eq!(
            chain.len(),
            1,
            "a fold a queued tick still anchors below must not evict"
        );

        // Execution reaches the settling height: nothing anchors below it
        // any more.
        append_at(3, wave(3, &[]));
        chain.prune_persisted(BlockHeight::new(2));
        assert_eq!(chain.len(), 2, "persistence has not caught up");
        chain.prune_persisted(BlockHeight::new(3));
        assert_eq!(chain.len(), 1, "only the settled tick evicts");
    }

    #[test]
    fn resolution_is_idempotent_and_tolerates_unknown_waves() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let settled = TickResolution::Settled {
            height: BlockHeight::new(1),
            aborted: BTreeSet::new(),
        };
        // Unknown wave: no entry at its origin height.
        chain.resolve(&wave(5, &[2]), &settled);

        let w = wave(1, &[2]);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: BTreeMap::new(),
                provisional: BTreeMap::from([(w.clone(), Vec::new())]),
            },
        );
        chain.resolve(&w, &settled);
        chain.resolve(&w, &settled);
        chain.prune_persisted(BlockHeight::new(1));
        assert!(chain.is_empty());
    }
}
