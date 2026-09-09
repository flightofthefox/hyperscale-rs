//! The counterpart cells this node has proven for itself, as one mirror.
//!
//! A block states what a counterpart's commit-proven header says about a
//! cell; it does not carry the proof. So a voter holds the statement to a
//! reading it took itself, and this is where its own readings live: the
//! answers a fetched multiproof reconstructed under the anchor's root,
//! and the proof each was read from.
//!
//! Nothing enters unverified. A proof is walked against the anchor's root
//! on the way in, so a reading held here is one the counterpart's own
//! committed state attests, whoever served the bytes. That is what lets
//! the bytes come from anywhere: a peer relaying a proof it fetched
//! cannot make this node believe a cell it cannot prove.
//!
//! # Kept apart from the mirror
//!
//! What is held here licenses a vote and nothing else. Records are
//! composed from committed content alone, so two validators at one
//! committed height compose the same records however their own fetches
//! landed — which is why a reading here never reaches
//! [`CounterpartMirror`](crate::CounterpartMirror).
//!
//! # Node-local, and shared
//!
//! Which cells a node has proven is that node's own view — it is why the
//! fence defers rather than refusing — so nothing here is consensus
//! content. One instance is shared by handle: the execution coordinator
//! writes it, the vote fence reads it, and the state-proof server relays
//! from it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    Anchor, BlockHeight, Inclusion, MerkleInclusionProof, RETENTION_HORIZON, ShardId, SubstateKey,
    WeightedTimestamp,
};

/// What one anchor has been proven to say.
#[derive(Debug, Default)]
struct Proven {
    /// Each cell asked of the anchor, and what its root said.
    cells: BTreeMap<SubstateKey, Inclusion>,
    /// The proofs those readings came from, kept whole to answer a peer
    /// that could not reach the counterpart. One fetch, one proof: a
    /// later fetch at the same anchor covers its own keys, so which
    /// proof answers a query is a question of coverage.
    proofs: Vec<(BTreeSet<SubstateKey>, MerkleInclusionProof)>,
}

/// Every counterpart cell this node has proven, by the anchor it was
/// proven against.
#[derive(Debug, Default)]
pub struct ProvenCells {
    by_anchor: RwLock<BTreeMap<Anchor, Proven>>,
    /// Advanced by every reading recorded, so a vote deferred for want
    /// of one is re-driven when the count has moved.
    generation: AtomicU64,
}

impl ProvenCells {
    /// An empty mirror.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Walk `proof` against `anchor` and keep what it attests, returning
    /// the reading it gives each key.
    ///
    /// `None` when the proof does not answer for `keys` under the
    /// anchor's root, and then nothing is kept: the one gate every
    /// reading passes through.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned, which means a consumer panicked holding
    /// it — the node is already unsound at that point.
    pub fn record(
        &self,
        anchor: Anchor,
        keys: Vec<SubstateKey>,
        proof: MerkleInclusionProof,
    ) -> Option<Vec<(SubstateKey, Inclusion)>> {
        let inclusions = proof
            .inclusions(anchor.state_root, anchor.shard, &keys)
            .ok()?;
        let mut by_anchor = self.write();
        let proven = by_anchor.entry(anchor).or_default();
        proven.cells.extend(inclusions.iter().copied());
        proven.proofs.push((keys.into_iter().collect(), proof));
        drop(by_anchor);
        self.generation.fetch_add(1, Ordering::AcqRel);
        Some(inclusions)
    }

    /// What this node has proven `key` to be at `anchor`, if it has
    /// proven anything.
    ///
    /// The whole anchor is the key, not the shard and the height: a
    /// reading is only about the root it was taken under, and a block
    /// naming another root at that height is naming another chain.
    ///
    /// # Panics
    ///
    /// As [`Self::record`].
    #[must_use]
    pub fn reading(&self, anchor: Anchor, key: SubstateKey) -> Option<Inclusion> {
        self.read()
            .get(&anchor)
            .and_then(|proven| proven.cells.get(&key))
            .copied()
    }

    /// A proof covering every one of `keys` at `shard`'s `height`, for a
    /// committee peer that could not obtain one from the counterpart
    /// itself.
    ///
    /// Coverage rather than equality, because a multiproof authenticates
    /// any subset of the claims it carries, so a fetch made for a wider
    /// set answers a narrower question. Nothing is composed across
    /// proofs: two proofs each covering half the keys leave the peer to
    /// ask for the halves.
    ///
    /// # Panics
    ///
    /// As [`Self::record`].
    #[must_use]
    pub fn relay(
        &self,
        shard: ShardId,
        height: BlockHeight,
        keys: &[SubstateKey],
    ) -> Option<MerkleInclusionProof> {
        self.read()
            .iter()
            .filter(|(anchor, _)| anchor.shard == shard && anchor.height == height)
            .flat_map(|(_, proven)| &proven.proofs)
            .find(|(covered, _)| keys.iter().all(|key| covered.contains(key)))
            .map(|(_, proof)| proof.clone())
    }

    /// Hold `cells` as proven at `anchor`, for a test that has no tree
    /// to build a proof against. Nothing to relay comes of it.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn proven(
        &self,
        anchor: Anchor,
        cells: impl IntoIterator<Item = (SubstateKey, Inclusion)>,
    ) {
        self.write().entry(anchor).or_default().cells.extend(cells);
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// How many facts have been recorded, ever. A reader that remembers
    /// the value it last read at knows whether anything is new.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Retire every reading whose anchor no window can still be read at,
    /// on the clock the committed block carries.
    ///
    /// The same floor [`ProvenAnchors::retire_below`] applies, since a
    /// reading is only ever checked against an anchor still held there.
    ///
    /// [`ProvenAnchors::retire_below`]: crate::ProvenAnchors::retire_below
    ///
    /// # Panics
    ///
    /// As [`Self::record`].
    pub fn retire_below(&self, now: WeightedTimestamp) {
        let floor = now.minus(RETENTION_HORIZON);
        self.write().retain(|anchor, _| anchor.ts >= floor);
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<Anchor, Proven>> {
        self.by_anchor.read().expect("proven cells lock poisoned")
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<Anchor, Proven>> {
        self.by_anchor.write().expect("proven cells lock poisoned")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap as StdBTreeMap;

    use hyperscale_jmt::{Blake3Hasher, Key as JmtKey, LeafValue, MemoryStore, NodeKey, Tree};

    use super::*;
    use crate::state_key::jmt_value_hash;
    use crate::test_utils::test_key;
    use crate::{Hash, StateRoot};

    type Jmt = Tree<Blake3Hasher, 1>;

    const SHARD: ShardId = ShardId::ROOT;

    /// A one-version tree holding `present`, and a proof over `asked`
    /// against it.
    fn tree_and_proof(
        present: &[SubstateKey],
        asked: &[SubstateKey],
    ) -> (StateRoot, MerkleInclusionProof) {
        let mut store = MemoryStore::new();
        let updates: StdBTreeMap<JmtKey, Option<LeafValue>> = present
            .iter()
            .map(|key| {
                let value = key.to_bytes().to_vec();
                let leaf = LeafValue::new(jmt_value_hash(&value), value.len() as u64);
                (key.to_bytes(), Some(leaf))
            })
            .collect();
        let result = Jmt::apply_updates(&store, None, 1, &updates).unwrap();
        let root = StateRoot::from_raw(Hash::from_hash_bytes(&result.root_hash));
        store.apply(&result);
        let jmt_keys: Vec<JmtKey> = asked.iter().map(SubstateKey::to_bytes).collect();
        let proof = Jmt::prove(&store, &NodeKey::root(1), &jmt_keys).unwrap();
        (root, MerkleInclusionProof::new(proof.encode()))
    }

    fn anchor_at(root: StateRoot, height: u64, ts_ms: u64) -> Anchor {
        Anchor {
            shard: SHARD,
            height: BlockHeight::new(height),
            state_root: root,
            ts: WeightedTimestamp::from_millis(ts_ms),
        }
    }

    /// A proof against another root attests nothing, and a mirror that
    /// refused one is a mirror that never saw it: no reading, no proof
    /// to relay, and no generation for a deferred vote to wake on.
    #[test]
    fn a_proof_that_does_not_reconstruct_the_anchor_leaves_nothing_behind() {
        let (held, missing) = (test_key(1), test_key(2));
        let (root, proof) = tree_and_proof(&[held], &[held, missing]);
        let cells = ProvenCells::new();
        let elsewhere = anchor_at(
            StateRoot::from_raw(Hash::from_bytes(b"elsewhere")),
            4,
            4_000,
        );

        assert!(
            cells
                .record(elsewhere, vec![held, missing], proof.clone())
                .is_none()
        );
        assert_eq!(cells.reading(elsewhere, held), None);
        assert_eq!(cells.relay(SHARD, BlockHeight::new(4), &[held]), None);
        assert_eq!(cells.generation(), 0);

        let anchor = anchor_at(root, 4, 4_000);
        let readings = cells
            .record(anchor, vec![held, missing], proof)
            .expect("the proof reconstructs its own anchor's root");
        assert_eq!(readings.len(), 2);
        assert_eq!(cells.generation(), 1);
    }

    /// A reading answers for the anchor it was taken at and no other.
    /// Two heights of one shard are two questions, and one height under
    /// two roots is two chains.
    #[test]
    fn a_reading_answers_only_for_the_anchor_it_was_taken_at() {
        let (held, missing) = (test_key(1), test_key(2));
        let (root, proof) = tree_and_proof(&[held], &[held, missing]);
        let anchor = anchor_at(root, 4, 4_000);
        let cells = ProvenCells::new();
        cells.record(anchor, vec![held, missing], proof).unwrap();

        assert!(cells.reading(anchor, held).unwrap().is_present());
        assert_eq!(cells.reading(anchor, missing), Some(Inclusion::Absent));
        assert_eq!(cells.reading(anchor, test_key(3)), None);
        assert_eq!(cells.reading(anchor_at(root, 5, 5_000), held), None);
        assert_eq!(
            cells.reading(
                anchor_at(StateRoot::from_raw(Hash::from_bytes(b"fork")), 4, 4_000),
                held
            ),
            None,
            "one height under another root is another chain",
        );
    }

    /// A peer asking for part of what a proof carries is served that
    /// proof, since a multiproof authenticates any subset of its claims.
    /// One asking for a key no single proof covers is served nothing
    /// rather than a proof that would not answer it.
    #[test]
    fn a_peer_is_relayed_a_proof_that_covers_everything_it_asks() {
        let (first, second, third) = (test_key(1), test_key(2), test_key(3));
        let (root, proof) = tree_and_proof(&[first, second], &[first, second]);
        let anchor = anchor_at(root, 4, 4_000);
        let cells = ProvenCells::new();
        cells.record(anchor, vec![first, second], proof).unwrap();

        let height = BlockHeight::new(4);
        assert!(cells.relay(SHARD, height, &[first]).is_some());
        assert!(cells.relay(SHARD, height, &[first, second]).is_some());
        assert_eq!(cells.relay(SHARD, height, &[first, third]), None);
        assert_eq!(cells.relay(SHARD, BlockHeight::new(5), &[first]), None);
        assert_eq!(cells.relay(ShardId::leaf(1, 1), height, &[first]), None);
    }

    /// Readings retire with the anchors they were taken at: past the
    /// horizon no window reads them and no block may still claim them.
    #[test]
    fn readings_retire_with_the_anchors_they_were_taken_at() {
        let key = test_key(1);
        let (root, proof) = tree_and_proof(&[key], &[key]);
        let cells = ProvenCells::new();
        let old = anchor_at(root, 4, 4_000);
        let recent = anchor_at(root, 9, 9_000);
        cells.record(old, vec![key], proof.clone()).unwrap();
        cells.record(recent, vec![key], proof).unwrap();

        let horizon = u64::try_from(RETENTION_HORIZON.as_millis()).expect("fits");
        cells.retire_below(WeightedTimestamp::from_millis(9_000 + horizon));
        assert_eq!(cells.reading(old, key), None);
        assert!(cells.reading(recent, key).is_some());
    }
}
