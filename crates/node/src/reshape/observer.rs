//! Sans-io observer bootstrap sequencer.
//!
//! A cohort observer of a pending child syncs exactly the child's key
//! span out of the splitting shard's beacon-attested boundary anchor:
//! the child span is partitioned into parallel sub-range fetches served
//! by the splitting shard's committee, every chunk is verified into the
//! parent's attested `state_root` and staged into the observer's
//! child-rooted store as it arrives, and the finalize builds the child
//! subtree from the staged leaves.
//!
//! There is no anchor to compare the imported root against — the beacon
//! holds only the parent's root, a one-way hash over the child
//! subtrees. The trust source is the chunks themselves: each one proves
//! its leaves into the attested parent root with completeness, so the
//! imported set is exactly the tree's leaves under the child prefix,
//! and prefix-rooted hashing makes the resulting store root the parent
//! tree's subtree node at that prefix by construction.
//!
//! Sans-io like [`ShardBootstrap`](crate::bootstrap::ShardBootstrap): drivers own
//! transport, peer selection, and the staging and finalize writes, and
//! pump it through the same [`BootstrapRequest`] surface (the
//! witness-history variant never appears — the pending child's
//! accumulator starts empty).

use hyperscale_types::network::request::GetBlockRequest;
use hyperscale_types::network::response::{GetBlockResponse, GetStateRangeResponse};
use hyperscale_types::{
    Block, BlockHash, BlockHeader, BlockHeight, CertifiedBlockHeader, ChainOrigin, CommitProof,
    NetworkDefinition, QuorumCertificate, ReadySignal, ResolvedCommittee, ShardAnchor, ShardId,
    SignError, Signer, StateRoot, StoredReceipt, ValidatorId, WeightedTimestamp,
    ready_signal_message, ready_signal_window, shard_prefix_path,
};

use crate::bootstrap::snap_sync::{SnapSync, StateRangeOutcome};
use crate::bootstrap::{BootstrapRequest, SPLIT_BITS, STATE_CHUNK_LIMIT};

/// The self-signed ready signal an observer broadcasts to the
/// splitting shard's committee on completing its child-span bootstrap.
///
/// Windowed from the splitting shard's attested anchor weighted
/// timestamp — the freshest committed clock the observer holds an
/// authenticated view of. The anchor refreshes every epoch boundary, so
/// the [`ready_signal_window`] span (scaled to `epoch_duration_ms`)
/// comfortably covers the chain's progress since; a signal that somehow
/// passes uncollected is re-emitted against a newer anchor. At the
/// committee, the signal classifies as a `ReshapeReady` witness leaf —
/// the sender's observer seat rides the window's topology snapshot.
/// # Errors
///
/// Propagates [`SignError`] when the signer cannot sign.
pub fn observer_ready_signal(
    network: &NetworkDefinition,
    validator: ValidatorId,
    child: ShardId,
    signer: &dyn Signer,
    anchor: ShardAnchor,
    epoch_duration_ms: u64,
) -> Result<ReadySignal, SignError> {
    let start = anchor.weighted_timestamp;
    let end = start.plus(ready_signal_window(epoch_duration_ms));
    let msg = ready_signal_message(network, validator, child, start, end);
    let sig = signer.sign(&msg)?;
    Ok(ReadySignal::new(validator, child, start, end, sig))
}

enum Phase {
    /// Assembling the child span of the parent's committed state; every
    /// verified chunk is staged by the driver as it arrives.
    State(SnapSync),
    /// Every sub-range staged, waiting for the driver to take the
    /// finalize.
    FinalizeReady,
    /// Driver took the finalize; waiting for the imported root.
    Finalizing,
    /// Imported: the child store holds the parent tree's child subtree.
    Complete(StateRoot),
}

/// Sequencing state for one observer's pending-child bootstrap.
pub struct ObserverBootstrap {
    anchor: ShardAnchor,
    child: ShardId,
    phase: Phase,
    /// Total value bytes across the chunks handed to the driver for
    /// staging — the child half's substate byte total, seeding the
    /// byte frontier the child chain starts from.
    imported_substate_bytes: u64,
}

impl ObserverBootstrap {
    /// Start a bootstrap of `child`'s span against `parent`'s attested
    /// boundary `anchor`.
    ///
    /// # Panics
    ///
    /// Panics unless `child` is a child of `parent` — an observer seat
    /// only ever names one of the splitting shard's two children.
    #[must_use]
    pub fn new(parent: ShardId, anchor: ShardAnchor, child: ShardId) -> Self {
        assert_eq!(
            child.parent(),
            Some(parent),
            "observer bootstrap target {child:?} is not a child of {parent:?}",
        );
        Self {
            anchor,
            child,
            phase: Phase::State(SnapSync::spanning(
                anchor,
                shard_prefix_path(parent),
                &shard_prefix_path(child),
                SPLIT_BITS,
                STATE_CHUNK_LIMIT,
            )),
            imported_substate_bytes: 0,
        }
    }

    /// The parent-shard anchor this bootstrap verifies against.
    #[must_use]
    pub const fn anchor(&self) -> ShardAnchor {
        self.anchor
    }

    /// The pending child whose span this bootstrap assembles.
    #[must_use]
    pub const fn child(&self) -> ShardId {
        self.child
    }

    /// Every request the current phase wants in flight. Empty while
    /// requests are outstanding, a finalize is pending, or the
    /// bootstrap is complete. Only [`BootstrapRequest::StateRange`]
    /// ever appears.
    pub fn next_requests(&mut self) -> Vec<BootstrapRequest> {
        match &mut self.phase {
            Phase::State(snap) => snap
                .next_requests()
                .into_iter()
                .map(|(id, request)| BootstrapRequest::StateRange(id, request))
                .collect(),
            Phase::FinalizeReady | Phase::Finalizing | Phase::Complete(_) => Vec::new(),
        }
    }

    /// Feed one state range response for `sub_range`. A staged outcome
    /// must be persisted by the driver before it pumps further
    /// responses; after the final chunk the finalize becomes available
    /// via [`Self::take_finalize`].
    pub fn on_state_range(
        &mut self,
        sub_range: usize,
        response: &GetStateRangeResponse,
    ) -> StateRangeOutcome {
        let Phase::State(snap) = &mut self.phase else {
            return StateRangeOutcome::Rejected("state response outside the state phase");
        };
        let outcome = snap.on_response(sub_range, response);
        if let StateRangeOutcome::Staged { .. } = &outcome {
            self.imported_substate_bytes = snap.staged_bytes();
            if snap.is_complete() {
                self.phase = Phase::FinalizeReady;
            }
        }
        outcome
    }

    /// Re-arm a state sub-range after a transport-level failure.
    pub fn on_state_range_failure(&mut self, sub_range: usize) {
        if let Phase::State(snap) = &mut self.phase {
            snap.on_failure(sub_range);
        }
    }

    /// The staged assembly's finalize height, ready for
    /// `BoundaryStore::finalize_boundary_import` on the observer's
    /// child-rooted store. `Some` exactly once; the driver answers with
    /// the imported root via [`Self::on_imported`].
    pub fn take_finalize(&mut self) -> Option<BlockHeight> {
        if !matches!(self.phase, Phase::FinalizeReady) {
            return None;
        }
        self.phase = Phase::Finalizing;
        Some(self.anchor.height)
    }

    /// Record the imported child-subtree root and complete.
    ///
    /// # Panics
    ///
    /// Panics unless the finalize was taken via [`Self::take_finalize`].
    pub fn on_imported(&mut self, root: StateRoot) {
        assert!(
            matches!(self.phase, Phase::Finalizing),
            "on_imported outside the finalize phase",
        );
        self.phase = Phase::Complete(root);
    }

    /// Whether the bootstrap is still assembling state — the only
    /// phase that depends on serving peers retaining the targeted
    /// boundary pin, and the last at which restarting against a newer
    /// anchor is sound (nothing has been imported into the store yet).
    #[must_use]
    pub const fn is_assembling_state(&self) -> bool {
        matches!(self.phase, Phase::State(_))
    }

    /// Whether the child span is imported.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self.phase, Phase::Complete(_))
    }

    /// The imported child-subtree root — the parent tree's node at the
    /// child prefix as of the anchor. `None` until complete.
    #[must_use]
    pub const fn imported_root(&self) -> Option<StateRoot> {
        match self.phase {
            Phase::Complete(root) => Some(root),
            _ => None,
        }
    }

    /// The imported substate byte total of the child half — the byte
    /// frontier the child chain starts from. Accrues per staged chunk.
    #[must_use]
    pub const fn imported_substate_bytes(&self) -> u64 {
        self.imported_substate_bytes
    }
}

/// Outcome of feeding one block-sync response to [`ObserverTail`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailOutcome {
    /// The block chains and is queued for application via
    /// [`ObserverTail::take_apply`].
    Accepted,
    /// The peer doesn't hold the requested height — the parent chain
    /// hasn't reached it yet, or the peer is behind. Re-arm and retry.
    NotYetAvailable,
    /// The response is invalid for this follow; the driver rotates peers.
    Rejected(&'static str),
}

/// The parent's terminal block, recognised by a follower as it passes,
/// with the child genesis derived from it.
///
/// A block `B` is the terminal crossing when its parent QC sits at or
/// before the cut and the QC certifying `B` sits past it. The certifying
/// QC used here is the *canonical* one — carried as the `parent_qc` of
/// `B`'s committed child, the next block the follow accepts — never the
/// QC served alongside `B`, which may be a higher-round re-certification
/// from the parent's coast and stamps a different weighted timestamp.
///
/// The genesis is derived from `B` alone: its `split_child_roots` pair
/// verified to compose to its own committed `state_root`, so a parent
/// cannot name a child subtree its terminal root doesn't contain, and
/// the canonical timestamp as the child clock's start anchor. That is
/// the same derivation the beacon fold performs an epoch later when it
/// seeds the child's anchor.
#[derive(Debug, Clone)]
pub struct TerminalSighting {
    /// The terminal block's header. A split child reads its
    /// `split_child_roots`; a merged parent composes its `state_root`
    /// with the sibling terminal's.
    pub header: BlockHeader,
    /// The canonical certificate over the terminal — the `parent_qc` of
    /// its committed successor. Its weighted timestamp is the child
    /// clock's start anchor.
    pub canonical_qc: QuorumCertificate,
    /// The derived child genesis, absent when the terminal carried no
    /// `split_child_roots` pair or one that fails to compose to its own
    /// state root.
    pub genesis: Option<DerivedGenesis>,
    /// The two-chain proving the terminal *committed* rather than merely
    /// certified, with the committee its QCs verify against. Absent when
    /// the follow never captured the parent's committee, or when the
    /// block after the terminal is not round-contiguous with it — a view
    /// change between them means no direct commit to prove here.
    pub commit_proof: Option<(CommitProof, ResolvedCommittee)>,
}

/// A child genesis derived locally from the parent's terminal block.
#[derive(Debug, Clone)]
pub struct DerivedGenesis {
    /// The genesis block the child adopts.
    pub block: Block,
    /// The chain origin it starts from.
    pub origin: ChainOrigin,
}

/// Derive `child`'s genesis from the parent's terminal header.
///
/// `None` when the terminal carries no `split_child_roots` pair, or one
/// that does not compose to the terminal's own committed `state_root` —
/// collision resistance makes the composition check enough on its own,
/// so a parent cannot name a child subtree its terminal root doesn't
/// contain.
fn derive_child_genesis(
    child: ShardId,
    terminal: &BlockHeader,
    canonical_wt: WeightedTimestamp,
) -> Option<DerivedGenesis> {
    let pair = terminal.split_child_roots()?;
    if !pair.composes_to(terminal.state_root()) {
        return None;
    }
    let child_root = if child.path() & 1 == 0 {
        pair.left
    } else {
        pair.right
    };
    Some(DerivedGenesis {
        block: Block::split_child_genesis(child, child_root, terminal, canonical_wt),
        origin: ChainOrigin {
            genesis_height: terminal.height().next(),
            anchor_wt: canonical_wt,
        },
    })
}

/// Sans-io tail-follower keeping an observer's synced child store
/// current with the splitting parent's chain.
///
/// The child-span bootstrap imports the parent's child subtree as of an
/// epoch anchor `A`, but the child genesis adopts the subtree of the
/// parent's *terminal* root `B` — every parent commit between them
/// moves the child half. The follower fetches the parent's committed
/// blocks above `A` one at a time and hands each to the driver to apply
/// through `BoundaryStore::follow_block_writes` (the store-prefix
/// subset of the block's writes; partition independence keeps the
/// store's root exactly the parent tree's child subtree node).
///
/// Trust: each accepted block must extend a hash chain seeded by the
/// beacon-attested anchor's block hash, and in the parent's final epoch
/// every header carries `split_child_roots` — the follower checks its
/// own applied root against its side after each application. Neither
/// authenticates a forged *extension* of the chain on its own (the
/// driver may additionally verify the served QCs); the flip's
/// fail-closed equality against the beacon-seeded child anchor is the
/// end-to-end check, and a corrupted follow costs the duty, never
/// safety.
pub struct ObserverTail {
    child: ShardId,
    /// Hash-chain cursor: the last accepted block's hash, seeded by the
    /// attested anchor's.
    last_hash: BlockHash,
    /// Next parent height to fetch.
    next: BlockHeight,
    /// The parent's scheduled cut, once the beacon has published one.
    /// `None` while the split is admitted but unscheduled.
    terminal_cut: Option<WeightedTimestamp>,
    /// The previously accepted block's header. The crossing test for a
    /// block needs the QC that certifies it, which only arrives with the
    /// next block — and the genesis derivation needs the terminal header
    /// itself, so the follow keeps one block of history.
    prev: Option<BlockHeader>,
    /// The parent's consensus committee, captured while it was still
    /// live so the terminal's QCs stay verifiable after the head moves on.
    parent_committee: Option<ResolvedCommittee>,
    /// The terminal block, once the follow has walked past it.
    terminal: Option<TerminalSighting>,
    /// Height of the last block the driver applied into the child store.
    /// The genesis adopts the child subtree as of the *terminal* root, so
    /// a flip before the store has applied through it would adopt the
    /// wrong subtree.
    applied: Option<BlockHeight>,
    /// Walk headers without applying anything. A parent half already holds
    /// the parent's state and seeds its child by cloning it, so it needs
    /// the walk only to find which of the parent's blocks is the terminal
    /// and to derive the genesis from it.
    recognition_only: bool,
    /// The store's root after the last application (the imported root
    /// until the first one).
    root: StateRoot,
    in_flight: bool,
    /// Accepted block waiting for the driver to apply and answer.
    pending: Option<PendingFollow>,
    /// Set while a taken application is out for the driver to apply,
    /// cleared on [`Self::on_applied`]. Guards [`Self::take_apply`] against
    /// re-emitting the same application when the driver re-polls before the
    /// apply answers — the production pump ticks on a timer, so a `step`
    /// can land between the take and its answer.
    apply_in_flight: bool,
}

struct PendingFollow {
    height: BlockHeight,
    receipts: Vec<StoredReceipt>,
    /// The block's `split_child_roots` slot for this child, when the
    /// header carried the pair — the applied root must reproduce it.
    expected_root: Option<StateRoot>,
}

impl ObserverTail {
    /// Start following the parent chain above the `anchor` a completed
    /// [`ObserverBootstrap`] imported at, from its `imported_root`.
    #[must_use]
    pub const fn new(anchor: ShardAnchor, child: ShardId, imported_root: StateRoot) -> Self {
        Self {
            child,
            last_hash: anchor.block_hash,
            next: anchor.height.next(),
            root: imported_root,
            in_flight: false,
            pending: None,
            apply_in_flight: false,
            terminal_cut: None,
            prev: None,
            parent_committee: None,
            terminal: None,
            applied: None,
            recognition_only: false,
        }
    }

    /// A follow that only recognises the terminal, applying nothing.
    ///
    /// For a parent half: it was on the parent, so its child store is a
    /// clone of state it already holds rather than something the follow
    /// builds. It walks the same headers for the same reason an observer
    /// does — to find the terminal crossing and derive the child genesis
    /// from it — and skips every write.
    #[must_use]
    pub fn recognizing(anchor: ShardAnchor, child: ShardId) -> Self {
        Self {
            recognition_only: true,
            ..Self::new(anchor, child, StateRoot::ZERO)
        }
    }

    /// Publish the parent's scheduled cut, so the follow can recognise the
    /// terminal crossing as it walks past it.
    ///
    /// The driver re-supplies it every step and the last published value
    /// is retained, for the same reason [`Self::capture_committee`]
    /// latches: a cut never moves, but it is *projected* for one window
    /// only. The fold that applies the reshape consumes the record at the
    /// top of the parent's final window, so the next promotion freezes an
    /// empty projection — while the crossing this identifies is only
    /// recognisable once the terminal's successor arrives, at the close of
    /// that same window. Clearing on `None` would drop the cut exactly as
    /// the block that needs it lands.
    pub const fn set_terminal_cut(&mut self, cut: Option<WeightedTimestamp>) {
        if cut.is_some() {
            self.terminal_cut = cut;
        }
    }

    /// The latched cut, for a driver that needs the instant itself rather
    /// than the crossing it identifies — a merged parent's clock anchors
    /// there. Reading it back here rather than from the projection is what
    /// keeps one latch instead of two.
    #[must_use]
    pub const fn terminal_cut(&self) -> Option<WeightedTimestamp> {
        self.terminal_cut
    }

    /// Capture the parent's consensus committee while it is still live, so
    /// the terminal's QCs can be verified after the head has moved on.
    ///
    /// The driver re-supplies it every step and the last non-empty capture
    /// wins: a split parent leaves the head's committee set the moment its
    /// applying fold lands, which is around when the follow reaches its
    /// terminal, so resolving on demand would come up empty exactly when
    /// the proof is needed. Committees are frozen per window, so the copy
    /// taken during the final window is the set that signed it.
    pub fn capture_committee(&mut self, committee: Option<ResolvedCommittee>) {
        if let Some(committee) = committee {
            self.parent_committee = Some(committee);
        }
    }

    /// The terminal sighting, but only once the follow has reached the
    /// block *after* the terminal — applied for a following tail, walked
    /// past for a recognizing one.
    ///
    /// Two reasons it is the successor and not the terminal itself. A
    /// followed store's root must be the child subtree as of the terminal
    /// root, which applying the terminal establishes — and the successor is
    /// a coast block past the cut, empty by rule, so it moves no state. But
    /// the adopt reads the child's substate byte total at the genesis
    /// height, which is `terminal + 1`, so that version has to exist. A
    /// recognizing tail writes nothing, and needs the successor anyway: its
    /// `parent_qc` is the only canonical source of the genesis clock.
    #[must_use]
    pub fn settled_terminal(&self) -> Option<&TerminalSighting> {
        let terminal = self.terminal.as_ref()?;
        (self.applied? >= terminal.header.height().next()).then_some(terminal)
    }

    /// The next block fetch, when none is outstanding and nothing is
    /// waiting to be applied.
    pub fn next_request(&mut self) -> Option<GetBlockRequest> {
        if self.in_flight || self.pending.is_some() {
            return None;
        }
        self.in_flight = true;
        Some(GetBlockRequest::new(self.next, self.next))
    }

    /// Feed the response for the outstanding fetch.
    pub fn on_response(&mut self, response: &GetBlockResponse) -> TailOutcome {
        if !self.in_flight {
            return TailOutcome::Rejected("unsolicited block response");
        }
        self.in_flight = false;
        let Some(elided) = &response.certified else {
            return TailOutcome::NotYetAvailable;
        };
        // The follower advertises no inventory, so every body is inline
        // and rehydration resolves nothing; this also pins the QC to the
        // header. The QC's signature is the driver's to verify.
        let Ok(certified) = elided.try_rehydrate(|_| None, |_| None, |_| None) else {
            return TailOutcome::Rejected("elided or mispaired block body");
        };
        let header = certified.block().header();
        if header.height() != self.next {
            return TailOutcome::Rejected("block height does not match the requested height");
        }
        if header.parent_block_hash() != self.last_hash {
            return TailOutcome::Rejected("block does not extend the attested anchor chain");
        }
        let receipts: Vec<StoredReceipt> = certified
            .block()
            .certificates()
            .iter()
            .flat_map(|fw| fw.receipts().iter().cloned())
            .collect();
        let expected_root = header.split_child_roots().map(|pair| {
            if self.child.path() & 1 == 0 {
                pair.left
            } else {
                pair.right
            }
        });
        // This block's parent QC is the canonical certificate over its
        // predecessor, so accepting it is what decides whether that
        // predecessor was the terminal crossing: the predecessor's own
        // parent QC sits at or before the cut, and this one past it.
        let parent_qc_wt = header.parent_qc().weighted_timestamp();
        if let Some(cut) = self.terminal_cut
            && self.terminal.is_none()
            && let Some(terminal) = &self.prev
            && terminal.parent_qc().weighted_timestamp().as_millis() <= cut.as_millis()
            && parent_qc_wt.as_millis() > cut.as_millis()
        {
            // The two-chain that commits the terminal: its own certificate
            // is this block's parent QC, and this block carries the QC over
            // itself. Only a round-contiguous pair is a direct commit; a
            // view change between them leaves nothing to prove here.
            let commit_proof = self
                .parent_committee
                .clone()
                .filter(|_| header.round() == terminal.round().next())
                .map(|committee| {
                    (
                        CommitProof::direct(
                            CertifiedBlockHeader::new(terminal.clone(), header.parent_qc().clone()),
                            CertifiedBlockHeader::new(header.clone(), certified.qc().clone()),
                        ),
                        committee,
                    )
                });
            self.terminal = Some(TerminalSighting {
                header: terminal.clone(),
                canonical_qc: header.parent_qc().clone(),
                genesis: derive_child_genesis(self.child, terminal, parent_qc_wt),
                commit_proof,
            });
        }
        self.prev = Some(header.clone());

        if self.recognition_only {
            self.applied = Some(header.height());
        } else {
            self.pending = Some(PendingFollow {
                height: header.height(),
                receipts,
                expected_root,
            });
        }
        self.last_hash = certified.block().hash();
        self.next = self.next.next();
        TailOutcome::Accepted
    }

    /// Re-arm after a transport-level failure.
    pub const fn on_failure(&mut self) {
        self.in_flight = false;
    }

    /// The accepted block's application, ready for
    /// `BoundaryStore::follow_block_writes` on the observer's store.
    /// `Some` once per accepted block; the driver answers with the
    /// resulting root via [`Self::on_applied`].
    pub fn take_apply(&mut self) -> Option<(BlockHeight, Vec<StoredReceipt>)> {
        if self.apply_in_flight {
            return None;
        }
        let pending = self.pending.as_mut()?;
        self.apply_in_flight = true;
        Some((pending.height, std::mem::take(&mut pending.receipts)))
    }

    /// Record the applied root, checking it against the header's
    /// `split_child_roots` slot when the block carried the pair.
    ///
    /// # Errors
    ///
    /// Returns a description when the applied root contradicts the
    /// followed header — the store has diverged from the parent's child
    /// subtree and the duty must fail closed (the flip falls back to a
    /// fresh snap-sync).
    ///
    /// # Panics
    ///
    /// Panics unless an application was taken via [`Self::take_apply`].
    pub fn on_applied(&mut self, root: StateRoot) -> Result<(), String> {
        self.apply_in_flight = false;
        let pending = self
            .pending
            .take()
            .expect("on_applied outside a taken application");
        if let Some(expected) = pending.expected_root
            && expected != root
        {
            return Err(format!(
                "followed store diverged at height {}: applied root {root:?} ≠ carried {expected:?}",
                pending.height,
            ));
        }
        self.applied = Some(pending.height);
        self.root = root;
        Ok(())
    }

    /// The store's root after the last applied block.
    #[must_use]
    pub const fn root(&self) -> StateRoot {
        self.root
    }

    /// The next parent height the follower wants.
    #[must_use]
    pub const fn next_height(&self) -> BlockHeight {
        self.next
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hyperscale_jmt::{Blake3Hasher, Hasher};
    use hyperscale_storage::test_helpers::pin_snap_sync_replica;
    use hyperscale_storage::{BoundaryStore, SubstateStore, WitnessSeed};
    use hyperscale_storage_memory::SimShardStorage;
    use hyperscale_types::{
        AggregateSignature, BeaconWitnessLeafCount, BeaconWitnessRoot, BoundedVec, CertificateRoot,
        ElidedCertifiedBlock, Hash, InFlightCount, Inventory, LocalReceiptRoot, ProposerTimestamp,
        ProvisionsRoot, Round, SignerBitfield, SplitChildRoots, TransactionRoot, WitnessSources,
    };

    use super::*;
    use crate::bootstrap::state_range_serve::serve_state_range_request;

    const ENTRIES: u8 = 12;

    /// A committed parent replica (whole-keyspace root shard), pinned at
    /// its boundary for serving.
    fn parent_replica() -> (Arc<SimShardStorage>, ShardAnchor) {
        let storage = SimShardStorage::default();
        let anchor = pin_snap_sync_replica(&storage, ENTRIES, &[]);
        (Arc::new(storage), anchor)
    }

    /// Drive one observer bootstrap to completion against `serving`,
    /// importing into a fresh store rooted at the child's prefix.
    /// Returns the child store and its imported root.
    fn observe(
        serving: &Arc<SimShardStorage>,
        anchor: ShardAnchor,
        child: ShardId,
    ) -> (SimShardStorage, StateRoot) {
        let store = SimShardStorage::new(shard_prefix_path(child));
        let mut bootstrap = ObserverBootstrap::new(ShardId::ROOT, anchor, child);
        for _ in 0..1_000 {
            if bootstrap.is_complete() {
                let root = bootstrap.imported_root().expect("complete");
                return (store, root);
            }
            for request in bootstrap.next_requests() {
                let BootstrapRequest::StateRange(id, request) = request else {
                    panic!("observer bootstrap emitted a non-state request");
                };
                let response = serve_state_range_request(serving, &request);
                match bootstrap.on_state_range(id, &response) {
                    StateRangeOutcome::Staged { leaves, progress } => {
                        store.stage_import_chunk(&progress, &leaves).unwrap();
                    }
                    StateRangeOutcome::Rejected(reason) => {
                        panic!("state range rejected: {reason}")
                    }
                }
            }
            if let Some(height) = bootstrap.take_finalize() {
                let root = store
                    .finalize_boundary_import(height, WitnessSeed::default())
                    .unwrap();
                bootstrap.on_imported(root);
            }
        }
        panic!("observer bootstrap did not complete");
    }

    /// The keystone identity, end to end: each child store adopts
    /// exactly the parent tree's subtree at its prefix, the two halves
    /// partition the parent's substates, and the parent's attested root
    /// recomposes from the two imported roots.
    #[test]
    fn observer_bootstraps_adopt_the_child_subtrees() {
        let (serving, anchor) = parent_replica();
        let (left, right) = ShardId::ROOT.children();

        let (left_store, left_root) = observe(&serving, anchor, left);
        let (right_store, right_root) = observe(&serving, anchor, right);

        assert_eq!(left_store.state_root(), left_root);
        assert_eq!(right_store.state_root(), right_root);
        assert_eq!(
            StateRoot::from_raw(Hash::from_hash_bytes(&Blake3Hasher::hash_internal(&[
                *left_root.as_raw().as_bytes(),
                *right_root.as_raw().as_bytes(),
            ]))),
            anchor.state_root,
            "imported child roots must recompose to the parent's attested root",
        );
    }

    /// Both halves together hold every parent substate exactly once.
    #[test]
    fn child_spans_partition_the_parent_population() {
        let (serving, anchor) = parent_replica();
        let children: [ShardId; 2] = ShardId::ROOT.children().into();

        let mut counts = Vec::new();
        for child in children {
            let mut bootstrap = ObserverBootstrap::new(ShardId::ROOT, anchor, child);
            for _ in 0..1_000 {
                if bootstrap.take_finalize().is_some() {
                    break;
                }
                for request in bootstrap.next_requests() {
                    if let BootstrapRequest::StateRange(id, request) = request {
                        let response = serve_state_range_request(&serving, &request);
                        bootstrap.on_state_range(id, &response);
                    }
                }
            }
            counts.push(bootstrap.imported_substate_bytes());
        }
        let parent_bytes = serving
            .substate_bytes_at_version(anchor.height.inner())
            .expect("parent byte total");
        assert_eq!(
            counts.iter().sum::<u64>(),
            parent_bytes,
            "the two halves' byte totals must sum to the parent's, no leaf lost or duplicated",
        );
        assert!(
            counts.iter().all(|&c| c > 0),
            "fixture population must straddle the split bit; got {counts:?}",
        );
    }

    /// A tampered leaf value fails the chunk verification and rejects.
    #[test]
    fn tampered_chunk_is_rejected() {
        let (serving, anchor) = parent_replica();
        let (left, _) = ShardId::ROOT.children();
        let mut bootstrap = ObserverBootstrap::new(ShardId::ROOT, anchor, left);

        let mut rejected = false;
        'outer: for _ in 0..1_000 {
            for request in bootstrap.next_requests() {
                let BootstrapRequest::StateRange(id, request) = request else {
                    unreachable!();
                };
                let mut response = serve_state_range_request(&serving, &request);
                if let Some(chunk) = &mut response.chunk
                    && !chunk.leaves.is_empty()
                {
                    let mut leaves: Vec<_> = chunk.leaves.iter().cloned().collect();
                    let mut value = leaves[0].value.to_vec();
                    value[0] ^= 1;
                    leaves[0].value = value.into();
                    chunk.leaves = leaves.into();
                    rejected = matches!(
                        bootstrap.on_state_range(id, &response),
                        StateRangeOutcome::Rejected(_),
                    );
                    break 'outer;
                }
                bootstrap.on_state_range(id, &response);
            }
        }
        assert!(rejected, "tampered chunk must reject");
    }

    /// An observer seat only ever names a child of the splitting shard.
    #[test]
    #[should_panic(expected = "is not a child of")]
    fn rejects_a_target_outside_the_split() {
        let (_, anchor) = parent_replica();
        let _ = ObserverBootstrap::new(ShardId::ROOT, anchor, ShardId::leaf(2, 0b11));
    }

    // ─── terminal recognition ───────────────────────────────────────────

    const CUT_MS: u64 = 6_000;

    /// A QC over `block` stamping `wt` — the value a follower reads when
    /// this QC arrives as the *next* block's `parent_qc`.
    fn qc_over(header: &BlockHeader, wt: u64) -> QuorumCertificate {
        QuorumCertificate::new(
            header.hash(),
            header.shard_id(),
            header.height(),
            header.parent_block_hash(),
            Round::new(header.round().inner()),
            SignerBitfield::new(4),
            AggregateSignature::ZERO,
            WeightedTimestamp::from_millis(wt),
        )
    }

    /// One parent-chain block. `pred_wt` is the stamp on its own
    /// `parent_qc` — the value the crossing test compares against the cut
    /// — and `pair` its `split_child_roots`, carried by every final-window
    /// header.
    fn parent_block(
        height: u64,
        round: u64,
        parent: BlockHash,
        pred_wt: u64,
        state_root: StateRoot,
        pair: Option<SplitChildRoots>,
    ) -> Block {
        let parent_qc = QuorumCertificate::new(
            parent,
            ShardId::ROOT,
            BlockHeight::new(height.saturating_sub(1)),
            BlockHash::ZERO,
            Round::new(round.saturating_sub(1)),
            SignerBitfield::new(4),
            AggregateSignature::ZERO,
            WeightedTimestamp::from_millis(pred_wt),
        );
        let header = BlockHeader::new(
            ShardId::ROOT,
            BlockHeight::new(height),
            parent,
            parent_qc,
            ValidatorId::new(0),
            ProposerTimestamp::ZERO,
            Round::new(round),
            false,
            state_root,
            TransactionRoot::ZERO,
            CertificateRoot::ZERO,
            LocalReceiptRoot::ZERO,
            ProvisionsRoot::ZERO,
            Vec::new(),
            std::collections::BTreeMap::new(),
            InFlightCount::ZERO,
            BeaconWitnessRoot::ZERO,
            BeaconWitnessLeafCount::ZERO,
            BeaconWitnessLeafCount::ZERO,
            pair,
            None,
        );
        Block::Live {
            header,
            transactions: Arc::new(BoundedVec::new()),
            certificates: Arc::new(BoundedVec::new()),
            provisions: Arc::new(BoundedVec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        }
    }

    /// The response a follower gets for `block`, certified by a QC
    /// stamping `served_wt` — deliberately *not* the stamp the derivation
    /// may use, so a test that reads the served QC's clock fails.
    fn response(block: &Block, served_wt: u64) -> GetBlockResponse {
        GetBlockResponse::found(ElidedCertifiedBlock::elide(
            block,
            qc_over(block.header(), served_wt),
            &Inventory::empty(),
        ))
    }

    /// A parent chain straddling the cut: the terminal at height 2 (its
    /// own `parent_qc` at or before the cut) and the coast block at
    /// height 3 whose `parent_qc` lands past it. The terminal carries the
    /// child-root pair composing to its own state root.
    ///
    /// Returns the anchor the follow starts from and the two blocks.
    fn straddling_chain() -> (ShardAnchor, Block, Block, SplitChildRoots) {
        let left = StateRoot::from_raw(Hash::from_bytes(b"left subtree"));
        let right = StateRoot::from_raw(Hash::from_bytes(b"right subtree"));
        let pair = SplitChildRoots { left, right };
        let terminal_root = pair.composed_root();

        let anchor_block = parent_block(1, 1, BlockHash::ZERO, 4_000, StateRoot::ZERO, None);
        let anchor = ShardAnchor {
            state_root: StateRoot::ZERO,
            block_hash: anchor_block.hash(),
            height: BlockHeight::new(1),
            weighted_timestamp: WeightedTimestamp::from_millis(4_000),
            witness_base: BeaconWitnessLeafCount::ZERO,
            settled_waves_root: None,
        };
        // The terminal's own parent QC sits at the cut exactly — the
        // boundary instant counts as not yet crossed.
        let terminal = parent_block(2, 2, anchor.block_hash, CUT_MS, terminal_root, Some(pair));
        // The coast block's parent QC is the canonical certificate over
        // the terminal, stamped past the cut.
        let coast = parent_block(
            3,
            3,
            terminal.hash(),
            CUT_MS + 500,
            terminal_root,
            Some(pair),
        );
        (anchor, terminal, coast, pair)
    }

    /// A recognizing follow walks the parent chain, spots the crossing
    /// when the coast block's `parent_qc` lands past the cut, and derives
    /// the child genesis the beacon fold composes from the same terminal.
    ///
    /// The canonical clock is the coast block's `parent_qc`, never the QC
    /// served alongside either block — a terminal re-certified at a higher
    /// round during the coast carries a divergent stamp.
    #[test]
    fn recognizing_follow_derives_the_child_genesis_from_the_terminal() {
        let (anchor, terminal, coast, pair) = straddling_chain();
        let (child, _) = ShardId::ROOT.children();

        let mut tail = ObserverTail::recognizing(anchor, child);
        tail.set_terminal_cut(Some(WeightedTimestamp::from_millis(CUT_MS)));

        assert!(
            tail.next_request().is_some(),
            "the walk starts above the anchor"
        );
        assert_eq!(
            tail.on_response(&response(&terminal, 9_999)),
            TailOutcome::Accepted
        );
        assert!(
            tail.settled_terminal().is_none(),
            "the terminal is not recognised until its successor arrives",
        );

        assert!(tail.next_request().is_some());
        assert_eq!(
            tail.on_response(&response(&coast, 9_999)),
            TailOutcome::Accepted
        );

        let sighting = tail.settled_terminal().expect("the crossing is recognised");
        assert_eq!(sighting.header.hash(), terminal.hash());
        assert_eq!(
            sighting.canonical_qc.weighted_timestamp(),
            WeightedTimestamp::from_millis(CUT_MS + 500),
            "the clock is the coast block's parent QC, not either served QC",
        );

        let derived = sighting.genesis.as_ref().expect("the pair composes");
        let expected = Block::split_child_genesis(
            child,
            pair.left,
            terminal.header(),
            WeightedTimestamp::from_millis(CUT_MS + 500),
        );
        assert_eq!(derived.block.hash(), expected.hash());
        assert_eq!(derived.origin.genesis_height, BlockHeight::new(3));
    }

    /// The cut is projected for one window, and the block that resolves
    /// the crossing arrives at its close. A `None` published after the
    /// projection drops it must not clear what the follow already holds,
    /// or the recognition is lost exactly when it becomes possible.
    #[test]
    fn a_withdrawn_projection_does_not_clear_the_published_cut() {
        let (anchor, terminal, coast, _) = straddling_chain();
        let (child, _) = ShardId::ROOT.children();

        let mut tail = ObserverTail::recognizing(anchor, child);
        tail.set_terminal_cut(Some(WeightedTimestamp::from_millis(CUT_MS)));
        let _ = tail.next_request();
        assert_eq!(
            tail.on_response(&response(&terminal, 9_999)),
            TailOutcome::Accepted
        );

        // The applying fold consumed the reshape record, so the head
        // projection no longer names a cut for the parent.
        tail.set_terminal_cut(None);

        let _ = tail.next_request();
        assert_eq!(
            tail.on_response(&response(&coast, 9_999)),
            TailOutcome::Accepted
        );
        assert!(
            tail.settled_terminal().is_some(),
            "the latched cut must still recognise the crossing",
        );
    }

    /// Without a cut the follow is a plain tail: it walks the same blocks
    /// and recognises nothing, which is what a duty discovered too late
    /// falls back from.
    #[test]
    fn a_follow_with_no_cut_recognises_nothing() {
        let (anchor, terminal, coast, _) = straddling_chain();
        let (child, _) = ShardId::ROOT.children();

        let mut tail = ObserverTail::recognizing(anchor, child);
        for block in [&terminal, &coast] {
            let _ = tail.next_request();
            assert_eq!(
                tail.on_response(&response(block, 9_999)),
                TailOutcome::Accepted
            );
        }
        assert!(tail.settled_terminal().is_none());
    }

    /// A following tail recognises the same crossing, but withholds the
    /// sighting until the store has applied through the terminal — the
    /// genesis adopts the child subtree as of *its* root.
    #[test]
    fn a_following_tail_withholds_the_sighting_until_it_has_applied() {
        let (anchor, terminal, coast, _) = straddling_chain();
        let (child, _) = ShardId::ROOT.children();

        let mut tail = ObserverTail::new(anchor, child, StateRoot::ZERO);
        tail.set_terminal_cut(Some(WeightedTimestamp::from_millis(CUT_MS)));

        let _ = tail.next_request();
        assert_eq!(
            tail.on_response(&response(&terminal, 9_999)),
            TailOutcome::Accepted
        );
        let (height, _) = tail
            .take_apply()
            .expect("the terminal is pending application");
        assert_eq!(height, BlockHeight::new(2));
        // The terminal carries the pair, so the applied root must reproduce
        // this child's half.
        tail.on_applied(StateRoot::from_raw(Hash::from_bytes(b"left subtree")))
            .expect("the applied root matches the carried half");

        let _ = tail.next_request();
        assert_eq!(
            tail.on_response(&response(&coast, 9_999)),
            TailOutcome::Accepted
        );
        assert!(
            tail.settled_terminal().is_none(),
            "the successor is accepted but not yet applied",
        );

        let (height, _) = tail
            .take_apply()
            .expect("the coast block is pending application");
        assert_eq!(height, BlockHeight::new(3));
        tail.on_applied(StateRoot::from_raw(Hash::from_bytes(b"left subtree")))
            .expect("a coast block moves no state under the prefix");
        assert!(
            tail.settled_terminal().is_some(),
            "applying through the successor releases the sighting",
        );
    }
}
