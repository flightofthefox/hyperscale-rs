//! Execution-tick output chain.
//!
//! One tick per committed block that carries executable work: the block's
//! single-shard transactions plus every cross-shard wave whose provisions
//! completed at that commit, executed as one batch. Each tick's output is
//! the next tick's baseline, overlaid on the persisted base — execution
//! dispatch reads through [`TickChain::view_at`] instead of the
//! settlement-derived [`crate::PendingChain`] overlay.
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
//! **This overlay never feeds state-root computation.** State roots stay
//! settlement-derived through `PendingChain`; `TickChain` deliberately
//! implements none of the commit-pipeline surfaces (`ShardChainWriter`,
//! `TreeReader`), so the two overlays cannot be cross-fed.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, RwLock};

use hyperscale_types::{BlockHeight, StateWrites, SubstateKey, TxHash, WaveId};

use crate::lock_recover::{read_or_recover, write_or_recover};
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
}

/// The execution output of one tick.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickOutput {
    /// Folded absolute cells determined at commit: the single-shard
    /// wave's writes, including its unconditional fee burns. Readable by
    /// every subsequent tick immediately.
    pub determined: StateWrites,
    /// The single-shard wave the determined fold came from, if the tick
    /// had one. Tracked for eviction only: the fold reaches the persisted
    /// base when this wave's settling block persists.
    pub determined_wave: Option<WaveId>,
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

/// One tick's retained state.
struct TickEntry {
    /// Folded readable cells: determined at append, plus entries promoted
    /// by resolution.
    readable: HashMap<SubstateKey, Option<Vec<u8>>>,
    /// Waves whose fate is still unknown. Includes the single-shard wave
    /// (with no provisional entries) so eviction waits for its
    /// settlement.
    pending: HashMap<WaveId, Vec<ProvisionalTx>>,
    /// Highest settling height seen among resolved waves, initially the
    /// tick's own height. The tick's readable fold is fully covered by
    /// the persisted base only once this height persists.
    max_resolution: BlockHeight,
}

impl TickEntry {
    /// An entry for the tick at `height`, before any output folds in.
    fn empty(height: BlockHeight) -> Self {
        Self {
            readable: HashMap::new(),
            pending: HashMap::new(),
            max_resolution: height,
        }
    }

    /// Fold a tick output in. Waves already recorded keep the entries
    /// they have, so a repeat append never duplicates one. A repeat that
    /// re-adds a wave the chain has already resolved is transient: the
    /// appending coordinator holds its own verdict for that wave until
    /// its tick lands, and emits it immediately after.
    fn absorb(&mut self, output: TickOutput) {
        for (key, change) in output.determined.cells {
            self.readable.insert(key, change);
        }
        for (wave, txs) in output.provisional {
            self.pending.entry(wave).or_insert(txs);
        }
        if let Some(wave) = output.determined_wave {
            self.pending.entry(wave).or_default();
        }
    }

    /// Apply one wave's verdict: promote each member's surviving side
    /// into the readable fold, or drop the wave's entries outright.
    /// A no-op for a wave already resolved.
    fn resolve(&mut self, wave_id: &WaveId, resolution: &TickResolution) {
        let Some(txs) = self.pending.remove(wave_id) else {
            return;
        };
        let height = match resolution {
            TickResolution::Settled { height, aborted } => {
                for tx in txs {
                    let promoted = if aborted.contains(&tx.tx_hash) {
                        tx.reserve
                    } else {
                        tx.writes
                    };
                    if let Some(writes) = promoted {
                        for (key, change) in writes.cells {
                            self.readable.insert(key, change);
                        }
                    }
                }
                *height
            }
            TickResolution::Aborted { height } => *height,
        };
        self.max_resolution = self.max_resolution.max(height);
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
}

impl<S> TickChain<S>
where
    S: VersionedStore,
{
    /// Create an empty tick chain over the given base storage.
    pub const fn new(base: Arc<S>) -> Self {
        Self {
            base,
            entries: RwLock::new(BTreeMap::new()),
        }
    }

    /// Append the output of the tick at `height`.
    ///
    /// Same-shard vnodes share one chain and each executes the tick under
    /// its own validator identity, so a height can be appended more than
    /// once. A tick's output is a pure function of the committed chain
    /// prefix, so the repeats carry the same content: folding them in is
    /// value-preserving, and a wave already resolved out of the entry
    /// stays resolved.
    pub fn append(&self, height: BlockHeight, output: TickOutput) {
        write_or_recover(&self.entries)
            .entry(height)
            .or_insert_with(|| TickEntry::empty(height))
            .absorb(output);
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

    /// Drop every tick fully covered by the persisted base: all waves
    /// resolved, and every settling height (and the tick itself) at or
    /// below `persisted`. Called on `BlockPersisted`.
    pub fn prune_persisted(&self, persisted: BlockHeight) {
        write_or_recover(&self.entries).retain(|height, entry| {
            !(entry.pending.is_empty() && *height <= persisted && entry.max_resolution <= persisted)
        });
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
    /// tick's readable fold at or below `tick`, in height order, over the
    /// base as of `tick`.
    ///
    /// The base read is anchored at `tick`, not at whatever this replica
    /// has currently persisted. Execution runs behind consensus, so a
    /// replica whose tick queue lags still commits and persists blocks —
    /// including the settlement of a wave belonging to a *later* tick,
    /// which 2f+1 other replicas certified without waiting for this one.
    /// Reading the live base would fold that later tick's writes into
    /// this tick's baseline, and the receipts would disagree with every
    /// replica that was not lagging. Anchoring makes the baseline a
    /// function of the tick alone.
    ///
    /// Folds cover the window the anchor cannot: settlements between the
    /// persisted tip and `tick` belong to ticks at or below `tick`, and a
    /// tick is only evicted once the base has absorbed it.
    pub fn view_at(&self, tick: BlockHeight) -> TickView<S> {
        let mut overlay: HashMap<SubstateKey, Option<Vec<u8>>> = HashMap::new();
        {
            let entries = read_or_recover(&self.entries);
            for (_, entry) in entries.range(..=tick) {
                for (key, change) in &entry.readable {
                    overlay.insert(*key, change.clone());
                }
            }
        }
        TickView {
            base: Arc::clone(&self.base),
            anchor: tick,
            overlay: Arc::new(overlay),
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
    overlay: Arc<HashMap<SubstateKey, Option<Vec<u8>>>>,
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
}

/// Snapshot from a [`TickView`] — the same overlay on the base storage's
/// snapshot.
pub struct TickViewSnapshot<Snap> {
    base_snapshot: Snap,
    overlay: Arc<HashMap<SubstateKey, Option<Vec<u8>>>>,
}

impl<Snap: SubstateDatabase> SubstateDatabase for TickViewSnapshot<Snap> {
    fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
        if let Some(change) = self.overlay.get(&key) {
            return change.clone();
        }
        self.base_snapshot.substate(key)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use hyperscale_types::{Address, Hash, LocalKey, MerkleInclusionProof, ShardId, StateRoot};

    use super::*;
    use crate::SubstateStore;
    use crate::lock_recover::lock_or_recover;

    /// A base whose every version reads the same — the anchor is
    /// exercised by [`StubStore::anchors`], not by versioned values.
    struct StubStore {
        cells: HashMap<SubstateKey, Vec<u8>>,
        /// Anchor heights `snapshot_at` was asked for, so a test can
        /// pin that a tick reads the base as of its own height.
        anchors: Mutex<Vec<BlockHeight>>,
    }

    impl StubStore {
        fn with_cell(key: SubstateKey, value: &[u8]) -> Self {
            Self {
                cells: HashMap::from([(key, value.to_vec())]),
                anchors: Mutex::new(Vec::new()),
            }
        }
    }

    impl SubstateDatabase for StubStore {
        fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
            self.cells.get(&key).cloned()
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
            StubSnapshot(self.cells.clone())
        }
        fn substate_bytes_at(&self, _height: BlockHeight) -> Option<u64> {
            None
        }
    }

    impl SubstateStore for StubStore {
        type Snapshot<'a> = StubSnapshot;
        fn snapshot(&self) -> Self::Snapshot<'_> {
            StubSnapshot(self.cells.clone())
        }
        fn jmt_height(&self) -> BlockHeight {
            BlockHeight::GENESIS
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
        }
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
                determined: writes(&[(key(1), Some(b"one")), (key(2), Some(b"two"))]),
                determined_wave: Some(wave(1, &[])),
                provisional: BTreeMap::new(),
            },
        );
        chain.append(
            BlockHeight::new(2),
            TickOutput {
                determined: writes(&[(key(1), None)]),
                determined_wave: Some(wave(2, &[])),
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

    /// A tick reads the base as of its own height, not as of whatever
    /// this replica has persisted. Execution runs behind consensus, so a
    /// lagging replica has already persisted settlements belonging to
    /// ticks it has not run; an unanchored read would fold those into an
    /// earlier tick's baseline and split its receipts from the
    /// committee's.
    #[test]
    fn a_tick_reads_the_base_as_of_its_own_height() {
        let store = Arc::new(StubStore::with_cell(key(1), b"base"));
        let chain = TickChain::new(Arc::clone(&store));

        let _ = chain.view_at(BlockHeight::new(4)).snapshot();
        let _ = chain.view_at(BlockHeight::new(9)).snapshot();

        assert_eq!(
            *lock_or_recover(&store.anchors),
            vec![BlockHeight::new(4), BlockHeight::new(9)]
        );
    }

    #[test]
    fn provisional_entries_unreadable_until_settled() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let w = wave(1, &[2]);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: StateWrites::default(),
                determined_wave: None,
                provisional: BTreeMap::from([(
                    w.clone(),
                    vec![ProvisionalTx {
                        tx_hash: tx(7),
                        writes: Some(writes(&[(key(1), Some(b"provisional"))])),
                        reserve: None,
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
                    determined: StateWrites::default(),
                    determined_wave: None,
                    provisional: BTreeMap::from([(
                        w.clone(),
                        vec![ProvisionalTx {
                            tx_hash: tx(member),
                            writes: Some(writes(&[(cell, Some(b"promoted"))])),
                            reserve: None,
                        }],
                    )]),
                },
            );
        }

        chain.resolve(
            &w,
            &TickResolution::Settled {
                height: BlockHeight::new(7),
                aborted: BTreeSet::new(),
            },
        );

        let view = chain.view_at(BlockHeight::new(6));
        assert_eq!(view.snapshot().substate(key(1)), Some(b"promoted".to_vec()));
        assert_eq!(view.snapshot().substate(key(2)), Some(b"promoted".to_vec()));
        chain.prune_persisted(BlockHeight::new(7));
        assert!(chain.is_empty(), "both entries must become evictable");
    }

    #[test]
    fn aborted_member_settles_reserve_not_writes() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let w = wave(1, &[2]);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: StateWrites::default(),
                determined_wave: None,
                provisional: BTreeMap::from([(
                    w.clone(),
                    vec![ProvisionalTx {
                        tx_hash: tx(7),
                        writes: Some(writes(&[(key(1), Some(b"effects"))])),
                        reserve: Some(writes(&[(key(9), Some(b"floor"))])),
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
                determined: StateWrites::default(),
                determined_wave: None,
                provisional: BTreeMap::from([(
                    w.clone(),
                    vec![ProvisionalTx {
                        tx_hash: tx(7),
                        writes: Some(writes(&[(key(1), Some(b"effects"))])),
                        reserve: Some(writes(&[(key(9), Some(b"floor"))])),
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

    #[test]
    fn eviction_waits_for_resolution_and_persistence() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let w = wave(1, &[]);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: writes(&[(key(1), Some(b"one"))]),
                determined_wave: Some(w.clone()),
                provisional: BTreeMap::new(),
            },
        );

        // Unresolved: survives any persistence progress.
        chain.prune_persisted(BlockHeight::new(10));
        assert_eq!(chain.len(), 1);

        // Resolved at height 3: survives persistence to 2, evicts at 3.
        chain.resolve(
            &w,
            &TickResolution::Settled {
                height: BlockHeight::new(3),
                aborted: BTreeSet::new(),
            },
        );
        chain.prune_persisted(BlockHeight::new(2));
        assert_eq!(chain.len(), 1);
        chain.prune_persisted(BlockHeight::new(3));
        assert!(chain.is_empty());
    }

    #[test]
    fn resolution_is_idempotent_and_tolerates_unknown_waves() {
        let chain = TickChain::new(Arc::new(StubStore::with_cell(key(1), b"base")));
        let settled = TickResolution::Settled {
            height: BlockHeight::new(3),
            aborted: BTreeSet::new(),
        };
        // Unknown wave: no entry at its origin height.
        chain.resolve(&wave(5, &[2]), &settled);

        let w = wave(1, &[2]);
        chain.append(
            BlockHeight::new(1),
            TickOutput {
                determined: StateWrites::default(),
                determined_wave: None,
                provisional: BTreeMap::from([(w.clone(), Vec::new())]),
            },
        );
        chain.resolve(&w, &settled);
        chain.resolve(&w, &settled);
        chain.prune_persisted(BlockHeight::new(3));
        assert!(chain.is_empty());
    }
}
