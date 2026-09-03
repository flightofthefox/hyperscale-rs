//! Sync-flow dispatch arms.
//!
//! When `BlockSyncComplete` fires we fan out across all three
//! coordinators in one pass: shard consensus exits sync mode and re-issues any
//! pending block fetches it had suppressed; remote-headers and
//! provisions flush their expected sets so we can immediately
//! participate in execution for blocks within the `MAX_FINALIZATION_DELAY` window.

use std::collections::BTreeMap;

use hyperscale_core::{Action, FetchRequest, ProtocolEvent};
use hyperscale_shard::SettledTxSet;
use hyperscale_types::{
    PredecessorTerminal, ShardId, TerminalEvidence, TopologySchedule, TxHash,
    derive_block_transactions,
};

use super::ShardParticipation;

impl ShardParticipation {
    /// Dispatch a sync-category `ProtocolEvent`.
    pub(in crate::state) fn handle_sync(
        &mut self,
        topology_schedule: &TopologySchedule,
        event: ProtocolEvent,
    ) -> Vec<Action> {
        match event {
            ProtocolEvent::BlockSyncReadyToApply { certified } => {
                derive_block_transactions(certified.block(), self.derivation.as_ref());
                self.shard_coordinator.on_sync_block_ready_to_apply(
                    topology_schedule,
                    std::sync::Arc::unwrap_or_clone(certified),
                )
            }
            // BlockSync finished fetching: exit shard consensus sync mode + flush
            // expected provisions + flush expected headers, all in one
            // pass.
            ProtocolEvent::BlockSyncComplete { .. } => {
                let mut actions = self
                    .shard_coordinator
                    .on_block_sync_complete(topology_schedule);
                actions.extend(
                    self.remote_headers_coordinator
                        .flush_expected_headers(topology_schedule),
                );
                actions.extend(self.provisions_coordinator.flush_expected_provisions());
                actions
            }
            // Both halves resume from the same recovered tip: consensus
            // from the frontier it stored, execution from the
            // transactions that frontier still owes an outcome for.
            ProtocolEvent::CommittedStateRestored { height, hash, qc } => {
                let mut actions = self
                    .shard_coordinator
                    .on_committed_state_restored(height, hash, qc);
                actions.extend(
                    self.execution_coordinator
                        .on_committed_state_restored(topology_schedule, self.derivation.as_ref()),
                );
                actions
            }
            // Remote-header catch-up finished. The coordinator keeps no
            // sync-mode state to reconcile on completion, so the event is
            // an acknowledged no-op.
            ProtocolEvent::RemoteHeaderSyncComplete { .. } => vec![],
            // A past-terminal shard's settled set is reconstructed: record
            // it for the split-boundary fence, then re-drive any votes
            // that deferred for want of it.
            ProtocolEvent::SettledTxsReconstructed {
                shard,
                txs,
                terminal_wt,
            } => self.on_settled_txs_reconstructed(
                topology_schedule,
                shard,
                SettledTxSet { txs, terminal_wt },
            ),
            // Evidence the execution coordinator mirrored about a
            // transaction a leg here issued for — a core's refusal, a
            // core's absence, a consumer's claim: the vote fence checks
            // the matching record arm against exactly this mirror, so
            // each is recorded and the votes that deferred without it
            // re-driven.
            ProtocolEvent::RefusalObserved {
                shard,
                tx_hash,
                refusal,
            } => {
                self.shard_coordinator
                    .record_refusal(shard, tx_hash, refusal);
                self.redrive_fence(topology_schedule)
            }
            ProtocolEvent::AbsenceObserved {
                shard,
                tx_hash,
                absence,
            } => {
                self.shard_coordinator
                    .record_absence(shard, tx_hash, absence);
                self.redrive_fence(topology_schedule)
            }
            ProtocolEvent::ClaimObserved {
                shard,
                tx_hash,
                presence,
            } => {
                self.shard_coordinator
                    .record_presence(shard, tx_hash, presence);
                self.redrive_fence(topology_schedule)
            }
            // A state proof against a core's commit-proven header
            // verified: the execution coordinator reads its probes off
            // it and hands each absence on.
            ProtocolEvent::StateProofVerified {
                anchor,
                proof,
                inclusions,
            } => self
                .execution_coordinator
                .on_state_proof_verified(anchor, proof, &inclusions),
            // A predecessor answered which of the queried transactions it
            // committed. Record the answers, then re-drive the votes that
            // deferred for want of them and the proposal that was
            // filtering them out.
            ProtocolEvent::PrecutResolutionsReceived {
                predecessor,
                answers,
            } => {
                for (tx_hash, absent) in answers {
                    self.shard_coordinator
                        .record_precut_resolution(predecessor, tx_hash, absent);
                }
                let actions = self
                    .shard_coordinator
                    .redrive_pending_votes(topology_schedule);
                self.shard_coordinator.queue_ready_proposal();
                actions
            }
            _ => unreachable!("non-sync event routed to handle_sync"),
        }
    }

    /// Beacon advanced an epoch — replay any cross-shard artifacts buffered
    /// because their committee epoch wasn't yet in the schedule (remote headers,
    /// ECs, finalized txs), then acquire any newly-attested settled set
    /// the fence needs. Dispatched from `handle_beacon`'s `BeaconBlockPersisted`
    /// arm via the option guard, so a vnode that only follows the beacon no-ops.
    pub(in crate::state) fn on_beacon_block_persisted(
        &mut self,
        sched: &TopologySchedule,
    ) -> Vec<Action> {
        let mut actions = self
            .remote_headers_coordinator
            .on_beacon_block_persisted(sched);
        actions.extend(self.execution_coordinator.on_beacon_block_persisted(sched));
        actions.extend(self.shard_coordinator.on_beacon_block_persisted(sched));
        // A proposer whose committee lookup stalled on the missing epoch has no
        // other retry signal: without this kick the view-change timer fires
        // first, the height is re-proposed in a later round, and the
        // round-contiguous commit rule never sees the consecutive rounds it
        // needs. The post-dispatch hook turns the latch into one `try_propose`.
        self.shard_coordinator.queue_ready_proposal();
        actions.extend(self.scan_settled_txs_acquisitions(sched));
        actions
    }

    /// Start a one-shot settled-set acquisition for every past-terminal shard
    /// whose beacon-attested `settled_txs_root` this node's own fold now
    /// carries and whose `S_P` the fence doesn't yet hold.
    ///
    /// Everything the acquisition needs comes from the node's beacon projection:
    /// the terminal block and attested root from the boundary anchor, the
    /// terminal weighted timestamp from the schedule's terminal cut, and the
    /// peers from the terminal-clamped routing committees. A shard already
    /// recorded (or live) is skipped, so the scan re-runs harmlessly each commit
    /// until the set is acquired.
    fn scan_settled_txs_acquisitions(&self, sched: &TopologySchedule) -> Vec<Action> {
        let head = sched.head();
        let mut actions = Vec::new();
        for (shard, peers) in sched.routing_committees() {
            if shard == self.local_shard {
                continue;
            }
            let Some(anchor) = head.boundary(shard) else {
                continue;
            };
            let Some(attested_root) = anchor.terminal_roots.map(|roots| roots.settled_txs) else {
                continue;
            };
            if self.shard_coordinator.settled_set(shard).is_some() {
                continue;
            }
            let Some(terminal_wt) = sched.terminal_cut_wt(shard) else {
                continue;
            };
            actions.push(Action::StartSettledTxsAcquisition {
                shard,
                evidence: TerminalEvidence {
                    height: anchor.height,
                    block_hash: anchor.block_hash,
                    terminal_wt,
                    attested_root,
                    expires: sched.handoff_evidence_expiry(shard),
                },
                peers,
            });
        }
        actions
    }

    /// Ask each predecessor about the pre-cut transactions this node is
    /// still holding a refusal over.
    ///
    /// One request per predecessor, carrying that predecessor's complete
    /// outstanding set — the io side diffs it against what the fetch
    /// already holds, so a pair that drops out of the set here is what
    /// releases its slot. That is why a predecessor with nothing
    /// outstanding still gets a request: an empty set is how the last
    /// query retires.
    ///
    /// Once the rule retires the outstanding set is empty rather than
    /// unasked, so the last release still goes out; the predecessors are
    /// dropped immediately after, which is what makes this the final pass.
    pub(in crate::state) fn scan_precut_queries(&mut self) -> Vec<Action> {
        if !self.shard_coordinator.has_precut_predecessors() {
            return Vec::new();
        }
        let live = self.shard_coordinator.precut_rule_live();
        let outstanding = if live {
            let cut = self.shard_coordinator.chain_origin().anchor_wt;
            let candidates = self.mempool_coordinator.pending_opening_before(cut);
            self.shard_coordinator
                .outstanding_precut_queries(candidates)
        } else {
            Vec::new()
        };

        let mut by_predecessor: BTreeMap<PredecessorTerminal, Vec<TxHash>> = self
            .shard_coordinator
            .predecessors()
            .iter()
            .map(|predecessor| (*predecessor, Vec::new()))
            .collect();
        for (predecessor, tx_hash) in outstanding {
            by_predecessor.entry(predecessor).or_default().push(tx_hash);
        }
        let actions: Vec<Action> = by_predecessor
            .into_iter()
            .map(|(predecessor, tx_hashes)| {
                Action::Fetch(FetchRequest::CommittedTxs {
                    predecessor,
                    tx_hashes,
                    preferred: None,
                    class: None,
                })
            })
            .collect();
        if !live {
            self.shard_coordinator.retire_precut();
        }
        actions
    }

    /// Record a past-terminal shard's settled set on both coordinators,
    /// then re-drive the votes and the gate-held finalizations that
    /// deferred for want of it.
    fn on_settled_txs_reconstructed(
        &mut self,
        topology_schedule: &TopologySchedule,
        shard: ShardId,
        set: SettledTxSet,
    ) -> Vec<Action> {
        // What this shard's ledger says the departed shard was party
        // to, mirrored beside the set for the vote fence: a departure
        // record names only the departed shard's own business here.
        let parties = self.execution_coordinator.party_to(shard, set.terminal_wt);
        let mut actions =
            self.execution_coordinator
                .record_settled_txs(topology_schedule, shard, set.clone());
        self.shard_coordinator
            .record_settled_txs(shard, set, parties);
        actions.extend(
            self.shard_coordinator
                .redrive_pending_votes(topology_schedule),
        );
        actions.extend(
            self.execution_coordinator
                .redrive_gated_finalizations(topology_schedule),
        );
        actions
    }

    /// Re-drive the votes the fence deferred for want of a mirror that
    /// has just landed.
    fn redrive_fence(&mut self, topology_schedule: &TopologySchedule) -> Vec<Action> {
        self.shard_coordinator
            .redrive_pending_votes(topology_schedule)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hyperscale_core::{Action, FetchRequest, ProtocolEvent, StateMachine};
    use hyperscale_types::test_utils::make_live_block;
    use hyperscale_types::{
        Block, BlockHash, BlockHeader, BlockHeaderParts, BlockHeight, CertifiedBlockHeader,
        ChainOrigin, Hash, LocalTimestamp, ProvisionTxRoot, QuorumCertificate, ShardId,
        ValidatorId, Verified,
    };

    use crate::assert_emits;
    use crate::state::test_support::TestNode;

    /// `BlockSyncComplete` fans out to shard consensus, remote-headers, and
    /// provisions in one pass. The provisions flush is the most
    /// directly observable: when a verified remote header has seeded
    /// `expected_provisions`, the flush surfaces an
    /// `Action::Fetch(FetchRequest::RemoteProvisions { .. })`. This
    /// test catches a regression where the provisions flush is
    /// dropped from the sync-complete arm.
    #[test]
    fn block_sync_complete_flushes_expected_provisions() {
        let TestNode { mut node, .. } = TestNode::builder().build();

        // Seed provisions.expected via a verified remote header whose
        // tick depends on local.
        let mut block = make_live_block(
            ShardId::leaf(1, 1),
            BlockHeight::new(5),
            /* timestamp_ms */ 1_000,
            ValidatorId::new(0),
            vec![],
            vec![],
        );
        if let Block::Live { ref mut header, .. } = block {
            *header = BlockHeader::new(BlockHeaderParts {
                shard_id: header.shard_id(),
                height: header.height(),
                parent_block_hash: header.parent_block_hash(),
                parent_qc: header.parent_qc().clone().into(),
                proposer: header.proposer(),
                timestamp: header.timestamp(),
                round: header.round(),
                is_fallback: header.is_fallback(),
                state_root: header.state_root(),
                transaction_root: header.transaction_root(),
                certificate_root: header.certificate_root(),
                local_receipt_root: header.local_receipt_root(),
                provision_root: header.provision_root(),
                provision_tx_roots: std::collections::BTreeMap::from([(
                    ShardId::ROOT,
                    ProvisionTxRoot::from_raw(Hash::from_bytes(b"placeholder-tx-root")),
                )]),
                work_in_flight: header.work_in_flight(),
                ..Default::default()
            });
        }
        let certified_header =
            Arc::new(Verified::new_unchecked_for_test(CertifiedBlockHeader::new(
                block.header().clone(),
                QuorumCertificate::genesis(ShardId::leaf(1, 1), ChainOrigin::ROOT),
            )));
        let _ = node.handle(
            LocalTimestamp::ZERO,
            ProtocolEvent::RemoteHeaderAdmitted { certified_header },
        );

        // Now trigger sync-complete. The provisions flush must surface
        // a fetch request for the seeded expected entry.
        let actions = node.handle(
            LocalTimestamp::ZERO,
            ProtocolEvent::BlockSyncComplete {
                height: BlockHeight::new(5),
            },
        );

        assert_emits!(
            actions,
            Action::Fetch(FetchRequest::RemoteProvisions { source_shard, .. })
                if *source_shard == ShardId::leaf(1, 1)
        );
    }

    /// `CommittedStateRestored` is the boot-time hand-off from `RocksDB`
    /// to the in-memory shard consensus state. The orchestrator routes it to
    /// `shard.on_committed_state_restored`, which restores
    /// `committed_height` so subsequent header validation and pending-
    /// block routing accept blocks at the correct tip. A regression
    /// that drops the routing leaves a freshly-booted node convinced
    /// it's still at genesis — silent until the first real header
    /// arrives and gets rejected as "below committed height".
    #[test]
    fn committed_state_restored_advances_shard_committed_height() {
        let TestNode { mut node, .. } = TestNode::new();
        assert_eq!(
            node.shard_coordinator().committed_height(),
            BlockHeight::new(0),
            "fresh node must start at genesis",
        );

        let restored_height = BlockHeight::new(42);
        let _ = node.handle(
            LocalTimestamp::ZERO,
            ProtocolEvent::CommittedStateRestored {
                height: restored_height,
                hash: Some(BlockHash::ZERO),
                qc: None,
            },
        );

        assert_eq!(
            node.shard_coordinator().committed_height(),
            restored_height,
            "committed height must reflect the restored value",
        );
    }
}
