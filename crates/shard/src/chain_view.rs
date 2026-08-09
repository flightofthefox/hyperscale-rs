//! Read-only view over the node's knowledge of the chain.
//!
//! `ChainView<'a>` bundles the committed-tip scalars, the latest QC, and a
//! borrowed reference to the pending block map. It unifies reads that would
//! otherwise have to thread half a dozen coordinator fields through every
//! helper — proposal building, header validation, commit decisions all
//! consult the same chain state.
//!
//! The view is **strictly a borrow**: no state is owned here, no lifecycle,
//! no mutations. It's a lens, not a sub-machine. The underlying fields live
//! on `ShardCoordinator` / `PendingBlock` just as before.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hyperscale_types::{
    BlockHash, BlockHeader, BlockHeight, CertifiedBlock, ChainOrigin, ProvisionHash,
    QuorumCertificate, RevealChain, ShardId, ShardLoad, StateRoot, TxHash, Verified, WorkInFlight,
};
use tracing::warn;

use crate::pending::{PendingBlock, PendingBlocks};

pub struct ChainView<'a> {
    local_shard: ShardId,
    chain_origin: ChainOrigin,
    committed_height: BlockHeight,
    committed_hash: BlockHash,
    committed_state_root: StateRoot,
    committed_in_flight: Option<WorkInFlight>,
    committed_settled_frontier: Option<BlockHeight>,
    committed_load: Option<ShardLoad>,
    committed_reveal_chain: Option<RevealChain>,
    latest_qc: Option<&'a Verified<QuorumCertificate>>,
    pending: &'a PendingBlocks,
    certified: &'a HashMap<BlockHash, Arc<Verified<CertifiedBlock>>>,
}

impl<'a> ChainView<'a> {
    #[allow(clippy::too_many_arguments)] // a borrow-bundle over the coordinator's chain fields
    pub const fn new(
        local_shard: ShardId,
        chain_origin: ChainOrigin,
        committed_height: BlockHeight,
        committed_hash: BlockHash,
        committed_state_root: StateRoot,
        committed_in_flight: Option<WorkInFlight>,
        committed_settled_frontier: Option<BlockHeight>,
        committed_load: Option<ShardLoad>,
        committed_reveal_chain: Option<RevealChain>,
        latest_qc: Option<&'a Verified<QuorumCertificate>>,
        pending: &'a PendingBlocks,
        certified: &'a HashMap<BlockHash, Arc<Verified<CertifiedBlock>>>,
    ) -> Self {
        Self {
            local_shard,
            chain_origin,
            committed_height,
            committed_hash,
            committed_state_root,
            committed_in_flight,
            committed_settled_frontier,
            committed_load,
            committed_reveal_chain,
            latest_qc,
            pending,
            certified,
        }
    }

    /// Borrow a pending block by hash. Used by callers that need to inspect
    /// per-block state (received transactions, finalizations) beyond what
    /// the dedicated header / state-root accessors expose.
    pub fn get_pending(&self, block_hash: BlockHash) -> Option<&PendingBlock> {
        self.pending.get(block_hash)
    }

    /// Header-only lookup. Pending blocks always carry their header even
    /// before full assembly, so this succeeds even when the body hasn't been
    /// constructed yet. A block absent from `pending` resolves through the
    /// verified-certified cache — the home of a sync-admitted block whose
    /// round-contiguous commit is still pending, which a halt recovery's
    /// fresh committee extends as its proposal parent.
    pub fn get_header(&self, block_hash: BlockHash) -> Option<&BlockHeader> {
        self.pending
            .get(block_hash)
            .map(PendingBlock::header)
            .or_else(|| {
                self.certified
                    .get(&block_hash)
                    .map(|certified| certified.block().header())
            })
    }

    /// State root of the parent block. Returns the committed-tip state root
    /// when `parent_block_hash` IS the committed tip (may have been pruned
    /// from `pending` by cleanup) or when lookup otherwise fails.
    pub fn parent_state_root(&self, parent_block_hash: BlockHash) -> StateRoot {
        if parent_block_hash == self.committed_hash {
            return self.committed_state_root;
        }
        self.get_header(parent_block_hash).map_or_else(
            || {
                warn!(
                    ?parent_block_hash,
                    committed_hash = ?self.committed_hash,
                    "Parent header not found for state root lookup"
                );
                self.committed_state_root
            },
            BlockHeader::state_root,
        )
    }

    /// Drain total on the parent header, or `None` when the parent is
    /// unresolvable: pruned from `pending` and — when the parent is the
    /// committed tip itself — the committed-tip scalar wasn't recovered.
    /// A snap-synced joiner extending its boundary anchor resolves through
    /// the scalar (the anchor header never enters `pending`); a `None`
    /// skips the vote, since the claimed in-flight count can't be checked.
    pub fn parent_in_flight_checked(&self, parent_block_hash: BlockHash) -> Option<WorkInFlight> {
        if let Some(header) = self.get_header(parent_block_hash) {
            return Some(header.work_in_flight());
        }
        (parent_block_hash == self.committed_hash)
            .then_some(self.committed_in_flight)
            .flatten()
    }

    /// Settlement frontier on the parent header — the highest tick whose
    /// determined half has settled at or below it. `None` when the parent
    /// is unresolvable, under the same conditions as
    /// [`Self::parent_in_flight_checked`]; a `None` skips the vote, since
    /// the claimed frontier can't be checked.
    #[must_use]
    pub fn parent_settled_frontier_checked(
        &self,
        parent_block_hash: BlockHash,
    ) -> Option<BlockHeight> {
        if let Some(header) = self.get_header(parent_block_hash) {
            return Some(header.settled_tick_frontier());
        }
        (parent_block_hash == self.committed_hash)
            .then_some(self.committed_settled_frontier)
            .flatten()
    }

    /// The parent's settlement frontier, or genesis when unresolvable —
    /// the proposer-side read, where an unresolvable parent means the
    /// block being built settles from the bottom rather than skipping.
    #[must_use]
    pub fn parent_settled_frontier(&self, parent_block_hash: BlockHash) -> BlockHeight {
        self.parent_settled_frontier_checked(parent_block_hash)
            .unwrap_or(BlockHeight::GENESIS)
    }

    /// Attested load on the parent header — the running gas total the next
    /// block advances. `None` when the parent is unresolvable, under the
    /// same conditions as [`Self::parent_in_flight_checked`].
    #[must_use]
    pub fn parent_load_checked(&self, parent_block_hash: BlockHash) -> Option<ShardLoad> {
        if let Some(header) = self.get_header(parent_block_hash) {
            return Some(header.load());
        }
        (parent_block_hash == self.committed_hash)
            .then_some(self.committed_load)
            .flatten()
    }

    /// Reveal chain on the parent header — the value the next block extends,
    /// or reseeds past when it anchors in a later epoch. `None` when the
    /// parent is unresolvable, under the same conditions as
    /// [`Self::parent_in_flight_checked`]: pruned from `pending` and, when
    /// the parent is the committed tip itself, the committed-tip scalar
    /// wasn't recovered. There is no safe default — a guessed chain produces
    /// a header every other replica rejects — so a `None` skips the vote and
    /// defers the build until the first commit reseats the scalar.
    pub fn parent_reveal_chain(&self, parent_block_hash: BlockHash) -> Option<RevealChain> {
        if let Some(header) = self.get_header(parent_block_hash) {
            return Some(header.reveal_chain());
        }
        (parent_block_hash == self.committed_hash)
            .then_some(self.committed_reveal_chain)
            .flatten()
    }

    /// Drain total on the parent header. Returns zero if the parent is
    /// unresolvable (see [`Self::parent_in_flight_checked`]).
    pub fn parent_in_flight(&self, parent_block_hash: BlockHash) -> WorkInFlight {
        self.parent_in_flight_checked(parent_block_hash)
            .unwrap_or(WorkInFlight::ZERO)
    }

    /// Parent to use when building the next proposal: the latest QC's block
    /// if any, otherwise the committed tip under a genesis QC tagged with
    /// the local shard and the chain's start-time anchor.
    pub fn proposal_parent(&self) -> (BlockHash, Verified<QuorumCertificate>) {
        self.latest_qc.map_or_else(
            || {
                (
                    self.committed_hash,
                    Verified::<QuorumCertificate>::genesis(self.local_shard, self.chain_origin),
                )
            },
            |qc| (qc.block_hash(), qc.clone()),
        )
    }

    /// Walk the QC chain from `parent_block_hash` back to committed height,
    /// collecting certificate, transaction, and provision hashes from
    /// ancestor blocks. Used by the proposer (to filter duplicates) and
    /// validators (to reject blocks containing already-included items).
    ///
    /// The manifest carries the full tx / cert / provision hash lists for
    /// every pending ancestor whether or not its body has assembled, so a
    /// single walk reads from it uniformly. Reading the block body instead
    /// would stop the walk at the first not-yet-assembled ancestor and drop
    /// the dedup contributions of every assembled block below it. The
    /// just-committed block (at or below `committed_height`) is covered
    /// separately by
    /// [`CommitDedupIndex`](crate::commit_dedup::CommitDedupIndex)'s
    /// `contains_*` queries, populated synchronously inside
    /// [`crate::coordinator::ShardCoordinator::record_block_committed`].
    pub fn collect_ancestor_hashes(
        &self,
        parent_block_hash: BlockHash,
    ) -> (HashSet<TxHash>, HashSet<ProvisionHash>) {
        let mut tx_hashes: HashSet<TxHash> = HashSet::new();
        let mut provision_hashes: HashSet<ProvisionHash> = HashSet::new();

        let mut current_hash = parent_block_hash;
        while let Some(pending) = self.pending.get(current_hash) {
            if pending.header().height() <= self.committed_height {
                break;
            }
            let manifest = pending.manifest();
            for tx_hash in manifest.tx_hashes() {
                tx_hashes.insert(*tx_hash);
            }
            for batch_hash in manifest.provision_hashes() {
                provision_hashes.insert(*batch_hash);
            }
            current_hash = pending.header().parent_block_hash();
        }

        (tx_hashes, provision_hashes)
    }

    /// The transactions the QC chain's uncommitted ancestors have already
    /// reached a verdict for, from `parent_block_hash` back to committed
    /// height.
    ///
    /// Read from the finalizations themselves rather than the manifest,
    /// which names ticks and not the transactions under them. An ancestor
    /// whose finalizations this node is still fetching contributes
    /// nothing, so the answer is what this node can see — the same
    /// direction every content rule here takes, since a node that under-
    /// reports can only fail to reject, and the rule needs a quorum of
    /// enforcers rather than every node.
    pub fn ancestor_resolved_txs(&self, parent_block_hash: BlockHash) -> HashSet<TxHash> {
        let mut resolved: HashSet<TxHash> = HashSet::new();
        let mut current_hash = parent_block_hash;
        while let Some(pending) = self.pending.get(current_hash) {
            if pending.header().height() <= self.committed_height {
                break;
            }
            for fw in pending.finalizations() {
                resolved.extend(fw.tx_hashes());
            }
            current_hash = pending.header().parent_block_hash();
        }
        resolved
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hyperscale_types::{
        AggregateSignature, Block, BlockHeaderParts, BlockManifest, Hash, LocalTimestamp,
        ProposerTimestamp, QuorumCertificate, Round, ShardId, SignerBitfield, Transaction,
        Verifiable, WeightedTimestamp, WitnessSources, test_utils,
    };

    use super::*;

    fn make_header(height: u8, parent_block_hash: BlockHash) -> BlockHeader {
        BlockHeader::new(BlockHeaderParts {
            height: BlockHeight::new(u64::from(height)),
            parent_block_hash,
            parent_qc: QuorumCertificate::genesis(ShardId::ROOT, ChainOrigin::ROOT).into(),
            timestamp: ProposerTimestamp::from_millis(1000),
            state_root: StateRoot::from_raw(Hash::from_bytes(&[height; 32])),
            provision_tx_roots: std::collections::BTreeMap::new(),
            work_in_flight: WorkInFlight::new(u64::from(height)),
            ..Default::default()
        })
    }

    fn make_block(height: u8, parent_block_hash: BlockHash) -> Block {
        Block::Live {
            header: make_header(height, parent_block_hash),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        }
    }

    /// Build a `ChainView` referencing scoped dummy state. Ownership stays
    /// with the closure so the view's borrows are safe.
    fn run_view<R>(
        committed_height: u64,
        committed_hash: BlockHash,
        committed_state_root: StateRoot,
        pending: &PendingBlocks,
        latest_qc: Option<&Verified<QuorumCertificate>>,
        f: impl FnOnce(&ChainView<'_>) -> R,
    ) -> R {
        let certified = HashMap::new();
        let view = ChainView {
            local_shard: ShardId::ROOT,
            chain_origin: ChainOrigin::ROOT,
            committed_height: BlockHeight::new(committed_height),
            committed_hash,
            committed_state_root,
            committed_in_flight: None,
            committed_settled_frontier: None,
            committed_load: None,
            committed_reveal_chain: None,
            latest_qc,
            pending,
            certified: &certified,
        };
        f(&view)
    }

    fn bh(tag: &[u8]) -> BlockHash {
        BlockHash::from_raw(Hash::from_bytes(tag))
    }

    fn pending_from_block(block: &Block) -> PendingBlock {
        let mut pb = PendingBlock::from_complete_block(block, vec![], vec![], LocalTimestamp::ZERO);
        pb.construct_block().expect("construct block");
        pb
    }

    #[test]
    fn get_header_returns_header_even_when_block_not_assembled() {
        let parent = bh(b"parent");
        let header = make_header(3, parent);
        let block_hash = header.hash();

        // Pending block without a constructed inner block — should still
        // yield a header.
        let pending_block =
            PendingBlock::from_manifest(header, BlockManifest::default(), LocalTimestamp::ZERO);
        let mut pending = PendingBlocks::new();
        pending.insert(pending_block);

        run_view(
            0,
            BlockHash::ZERO,
            StateRoot::ZERO,
            &pending,
            None,
            |view| {
                assert!(
                    view.get_pending(block_hash)
                        .is_some_and(|p| p.block().is_none())
                );
                let h = view.get_header(block_hash).expect("header available");
                assert_eq!(h.height(), BlockHeight::new(3));
            },
        );
    }

    #[test]
    fn collect_ancestor_hashes_covers_assembled_block_below_unassembled() {
        // Chain above the committed tip: walk start `middle` (manifest-only) ->
        // `low` (assembled, height 1) -> committed. `low`'s transaction must
        // still land in the dedup set even though an unassembled ancestor sits
        // between it and the walk start — otherwise a descendant could
        // re-include a transaction already present above the committed tip.
        let tx: Arc<Verifiable<Transaction>> =
            Arc::new(Verifiable::from(test_utils::test_transaction(7)));
        let tx_hash = tx.hash();
        let low = Block::Live {
            header: make_header(1, BlockHash::ZERO),
            transactions: Arc::new(vec![tx]),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        };
        let low_pending = pending_from_block(&low);
        let low_hash = low_pending.header().hash();

        let middle = PendingBlock::from_manifest(
            make_header(2, low_hash),
            BlockManifest::default(),
            LocalTimestamp::ZERO,
        );
        let middle_hash = middle.header().hash();

        let mut pending = PendingBlocks::new();
        pending.insert(low_pending);
        pending.insert(middle);

        run_view(
            0,
            BlockHash::ZERO,
            StateRoot::ZERO,
            &pending,
            None,
            |view| {
                // Precondition: `low` is assembled, `middle` is not.
                assert!(
                    view.get_pending(low_hash)
                        .is_some_and(|p| p.block().is_some())
                );
                assert!(
                    view.get_pending(middle_hash)
                        .is_some_and(|p| p.block().is_none())
                );

                let (tx_hashes, _provisions) = view.collect_ancestor_hashes(middle_hash);
                assert!(
                    tx_hashes.contains(&tx_hash),
                    "assembled ancestor below an unassembled one dropped from dedup set",
                );
            },
        );
    }

    #[test]
    fn parent_state_root_uses_committed_tip_on_match() {
        let tip_hash = bh(b"tip");
        let tip_root = StateRoot::from_raw(Hash::from_bytes(b"tip_root"));

        run_view(
            10,
            tip_hash,
            tip_root,
            &PendingBlocks::new(),
            None,
            |view| {
                assert_eq!(view.parent_state_root(tip_hash), tip_root);
            },
        );
    }

    #[test]
    fn parent_state_root_reads_header_state_root_when_present() {
        let tip_hash = bh(b"tip");
        let tip_root = StateRoot::ZERO;

        let block = make_block(5, BlockHash::ZERO);
        let hash = block.hash();
        let expected_state_root = block.header().state_root();

        let mut pending = PendingBlocks::new();
        pending.insert(pending_from_block(&block));

        run_view(4, tip_hash, tip_root, &pending, None, |view| {
            assert_eq!(view.parent_state_root(hash), expected_state_root);
        });
    }

    #[test]
    fn parent_state_root_falls_back_to_tip_when_unknown() {
        let tip_hash = bh(b"tip");
        let tip_root = StateRoot::from_raw(Hash::from_bytes(b"tip_root"));
        let unknown = bh(b"unknown");

        run_view(
            10,
            tip_hash,
            tip_root,
            &PendingBlocks::new(),
            None,
            |view| {
                assert_eq!(view.parent_state_root(unknown), tip_root);
            },
        );
    }

    #[test]
    fn parent_in_flight_returns_header_value_or_zero() {
        let block = make_block(7, BlockHash::ZERO);
        let hash = block.hash();
        let expected_in_flight = block.header().work_in_flight();
        let unknown = bh(b"unknown");

        let mut pending = PendingBlocks::new();
        pending.insert(pending_from_block(&block));

        run_view(
            0,
            BlockHash::ZERO,
            StateRoot::ZERO,
            &pending,
            None,
            |view| {
                assert_eq!(view.parent_in_flight(hash), expected_in_flight);
                assert_eq!(view.parent_in_flight(unknown), WorkInFlight::ZERO);
            },
        );
    }

    #[test]
    fn proposal_parent_returns_latest_qc_when_present() {
        let qc_block = bh(b"qc_block");
        let qc = QuorumCertificate::new(
            qc_block,
            ShardId::ROOT,
            BlockHeight::new(5),
            BlockHash::ZERO,
            Round::INITIAL,
            SignerBitfield::empty(),
            AggregateSignature::ZERO,
            WeightedTimestamp::from_millis(1000),
        );
        // SAFETY: synthetic test fixture, no real signature.
        let qc = Verified::<QuorumCertificate>::new_unchecked_for_test(qc);

        run_view(
            0,
            BlockHash::ZERO,
            StateRoot::ZERO,
            &PendingBlocks::new(),
            Some(&qc),
            |view| {
                let (hash, returned_qc) = view.proposal_parent();
                assert_eq!(hash, qc_block);
                assert_eq!(returned_qc.height(), BlockHeight::new(5));
            },
        );
    }

    #[test]
    fn proposal_parent_falls_back_to_committed_tip_without_qc() {
        let tip_hash = bh(b"tip");

        run_view(
            0,
            tip_hash,
            StateRoot::ZERO,
            &PendingBlocks::new(),
            None,
            |view| {
                let (hash, qc) = view.proposal_parent();
                assert_eq!(hash, tip_hash);
                assert!(qc.is_genesis());
            },
        );
    }
}
