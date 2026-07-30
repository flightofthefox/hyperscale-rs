//! Per-host driver for shard-less vnodes: the beacon-follower pool.
//!
//! A [`PoolLoop`] drives the vnodes a host runs with `shard: None` — logical
//! nodes that follow the beacon chain (adopt committed beacon blocks, track
//! topology, surface their own seat triggers) but run no shard consensus. It is
//! the lightweight sibling of [`ShardLoop`](crate::shard::ShardLoop): a
//! `Vec<Vnode>` plus a cloned `Arc<ProcessIo>` and per-step scratch — no
//! `ShardIo`, no batch accumulators, no per-payload fetches.
//!
//! A follower's entire action set is handled **inline**. The delegated-dispatch
//! path a `ShardLoop` uses is unbuildable here anyway (its `ActionContext` needs
//! a `PendingChain` a shard-less host has no storage for), and a follower's
//! actions are cheap: a beacon block's cert verifies with one signature aggregate
//! check, and adoption only touches process-shared state (the beacon commit
//! dedup, the topology `ArcSwap`). Because a pooled vnode no-ops
//! `BeaconBlockPersisted` (it has no shard coordinators to replay into), the
//! whole `received → verify → adopt → commit` cascade runs to quiescence within
//! one `PoolLoop::dispatch_event`, with no event-channel round trip.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use hyperscale_core::{Action, ParticipationChange, ProtocolEvent, StateMachine};
use hyperscale_dispatch::Dispatch;
use hyperscale_network::{Network, RequestError, ResponseVerdict};
use hyperscale_storage::ShardStorage;
use hyperscale_types::network::request::beacon::GetBeaconBlockRequest;
use hyperscale_types::network::response::beacon::GetBeaconBlockResponse;
use hyperscale_types::{
    CertifiedBeaconBlock, CertifiedBeaconBlockVerifyContext, Epoch, LocalTimestamp, ShardId,
    Verifiable,
};
use tracing::warn;

use crate::beacon::{self, BeaconBlockSync, BeaconSyncSink, beacon_block_sync_config};
use crate::event::{HostEvent, PoolScopedInput, classify_fetch_error};
use crate::process::ProcessIo;
use crate::shard::{StepOutput, TimerOp};
use crate::vnode::Vnode;

/// Cadence at which a syncing pool retries deferred beacon-block fetches.
///
/// An idle pool never fires it: the production thread's `select!` just
/// re-checks shutdown each interval, and the simulation schedules no tick.
pub const POOL_FETCH_TICK_INTERVAL: Duration = Duration::from_secs(1);

/// Active driver for a host's shard-less, beacon-following vnodes.
pub struct PoolLoop<S, N, D>
where
    S: ShardStorage,
    D: Dispatch,
{
    /// Process-scoped resources shared with every other driver on the host:
    /// beacon storage, the beacon-commit dedup gate, the topology `ArcSwap`.
    pub(crate) process: Arc<ProcessIo<S, N, D>>,

    /// The shard-less vnodes this host follows the beacon with. Driven in order
    /// each step; each independently folds the same committed beacon blocks.
    pub vnodes: Vec<Vnode>,

    /// Cached wall-clock time, set by [`NodeHost::set_time`](crate::host::NodeHost::set_time).
    now: LocalTimestamp,

    /// Per-step scratch: placement deltas emitted via
    /// `Action::ReconfigureParticipation` — the seat/drain triggers the runner
    /// acts on. Cleared at step entry, drained into the step's `StepOutput`.
    pub(crate) pending_participation_changes: Vec<ParticipationChange>,

    /// Per-step scratch: timer operations the pooled vnodes emitted,
    /// `shard: None` (a follower has no shard — the runner keys its pool
    /// timers by `TimerId` alone and routes fires back through the beacon
    /// channel). A follower is a ratify-pool voter and can be drawn onto
    /// an SPC committee, so its beacon timer chain must stay live.
    pub(crate) pending_timer_ops: Vec<TimerOp>,

    /// Per-step scratch: count of actions the pooled vnodes produced.
    pub(crate) actions_generated: usize,

    /// Beacon-block catch-up sync, scope `()`. A follower's coordinator
    /// emits `Action::StartBeaconBlockSync` when a gossiped block sits more
    /// than one epoch ahead of its tip; this FSM drives the
    /// `GetBeaconBlockRequest` fetches that close the gap, fed back through
    /// the host's beacon channel.
    beacon_block: BeaconBlockSync,
}

impl<S, N, D> PoolLoop<S, N, D>
where
    S: ShardStorage,
    N: Network,
    D: Dispatch,
{
    /// Build a pool driver over the host's shard-less vnodes. Used by
    /// `NodeHost::new` at construction and by the production supervisor when
    /// it builds a follower pool at runtime.
    ///
    /// Every follower's beacon startup timers are armed into the timer
    /// scratch here; the caller drains them to its timer table (the sim
    /// through the construction-time `drain_pending_output`, the
    /// production pool thread by re-arming at spawn).
    pub fn new(process: Arc<ProcessIo<S, N, D>>, vnodes: Vec<Vnode>) -> Self {
        let mut pool = Self {
            process,
            vnodes,
            now: LocalTimestamp::ZERO,
            pending_participation_changes: Vec::new(),
            pending_timer_ops: Vec::new(),
            actions_generated: 0,
            beacon_block: BeaconBlockSync::new(beacon_block_sync_config()),
        };
        pool.arm_startup_timers();
        pool
    }

    /// Add a follower at runtime — a validator that drained off its last
    /// shard — arming its beacon startup timers into the timer scratch for
    /// the caller to drain.
    pub fn add_vnode(&mut self, vnode: Vnode) {
        self.vnodes.push(vnode);
        self.arm_beacon_startup(self.vnodes.len() - 1);
    }

    /// Arm every follower's beacon startup timers into the timer scratch.
    /// Idempotent: re-arming replaces the pending fire at the runner's
    /// table, and the durations recompute identically from each
    /// coordinator's clock.
    fn arm_startup_timers(&mut self) {
        for vnode_idx in 0..self.vnodes.len() {
            self.arm_beacon_startup(vnode_idx);
        }
    }

    /// Re-arm every follower's beacon startup timers and drain the result —
    /// the production pool thread's spawn-time bootstrap. Replayable: the
    /// construction-time arming may already have been drained by a genesis
    /// ceremony on the host, and re-arming is idempotent.
    pub fn startup_output(&mut self) -> StepOutput {
        self.clear_scratch();
        self.arm_startup_timers();
        self.take_output()
    }

    /// Drive one follower's startup arming through the action path, so the
    /// emitted `SetTimer`s land in the timer scratch like any other step's.
    fn arm_beacon_startup(&mut self, vnode_idx: usize) {
        let actions = self.vnodes[vnode_idx].state.beacon_startup_actions();
        self.actions_generated += actions.len();
        let mut queue = VecDeque::new();
        for action in actions {
            self.process_action(vnode_idx, action, &mut queue);
        }
        debug_assert!(queue.is_empty(), "startup arming emits only timer sets");
    }

    /// Set the cached wall-clock time observed by `state.handle(now, _)`.
    pub const fn set_time(&mut self, now: LocalTimestamp) {
        self.now = now;
    }

    /// Number of pooled vnodes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.vnodes.len()
    }

    /// Whether the pool holds no vnodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.vnodes.is_empty()
    }

    /// Clear per-step scratch. Called by [`NodeHost::step`](crate::host::NodeHost::step)
    /// before dispatch so the drained output reflects only this step.
    pub(crate) fn clear_scratch(&mut self) {
        self.pending_participation_changes.clear();
        self.pending_timer_ops.clear();
        self.actions_generated = 0;
    }

    /// Drive one pool input through every pooled vnode and return the step's
    /// output: the placement deltas the followers surfaced (the seat triggers
    /// the supervisor acts on) and their timer operations. Clears per-step
    /// scratch first, mirroring
    /// [`ShardLoop::run_step`](crate::shard::ShardLoop::run_step); used by
    /// the production pool thread, which owns the `PoolLoop` directly rather
    /// than driving it through [`NodeHost::step`](crate::host::NodeHost::step).
    pub fn run_step(&mut self, input: PoolScopedInput) -> StepOutput {
        self.clear_scratch();
        self.dispatch_event(input);
        self.take_output()
    }

    /// Drain this step's scratch into a [`StepOutput`]. A follower emits no
    /// transaction statuses, so that field stays empty; the counterpart to
    /// [`Self::clear_scratch`], used by both this loop's [`Self::run_step`]
    /// and the whole-host [`NodeHost::step`](crate::host::NodeHost::step) so
    /// the scratch field set lives in one place.
    pub(crate) fn take_output(&mut self) -> StepOutput {
        StepOutput {
            actions_generated: std::mem::replace(&mut self.actions_generated, 0),
            timer_ops: std::mem::take(&mut self.pending_timer_ops),
            participation_changes: std::mem::take(&mut self.pending_participation_changes),
            ..StepOutput::default()
        }
    }

    /// Route a [`PoolScopedInput`] to the pooled vnodes or the catch-up
    /// sync FSM.
    pub(crate) fn dispatch_event(&mut self, input: PoolScopedInput) {
        match input {
            PoolScopedInput::Protocol(event) => self.dispatch_protocol(*event),
            PoolScopedInput::BeaconBlockSyncResponseReceived { epoch, block } => {
                beacon::on_response(self, epoch, block);
            }
            PoolScopedInput::BeaconBlockSyncFetchFailed { epoch, kind } => {
                beacon::on_fetch_failed(self, epoch, kind);
            }
            PoolScopedInput::FetchTick => beacon::on_tick(self),
        }
    }

    /// Fan a beacon [`ProtocolEvent`] across every pooled vnode, driving each
    /// one's follower cascade to quiescence.
    fn dispatch_protocol(&mut self, event: ProtocolEvent) {
        let count = self.vnodes.len();
        if count == 0 {
            return;
        }
        // Clone for every vnode except the last; the last takes ownership.
        for vnode_idx in 0..count - 1 {
            self.drive(vnode_idx, event.clone());
        }
        self.drive(count - 1, event);
    }

    /// Drive one pooled vnode: feed `event`, handle the emitted actions inline,
    /// and feed back any beacon continuation (a verify result) until the vnode
    /// stops producing them.
    fn drive(&mut self, vnode_idx: usize, event: ProtocolEvent) {
        let now = self.now;
        let mut queue = VecDeque::from([event]);
        while let Some(ev) = queue.pop_front() {
            let actions = self.vnodes[vnode_idx].state.handle(now, ev);
            self.actions_generated += actions.len();
            for action in actions {
                self.process_action(vnode_idx, action, &mut queue);
            }
        }
    }

    /// Handle one action from a beacon follower. The set is small: verify a
    /// gossiped block, commit an adopted one, adopt a fresh topology, surface
    /// a seat trigger, buffer a timer op. Anything else belongs to shard
    /// consensus, which a follower never runs.
    fn process_action(
        &mut self,
        vnode_idx: usize,
        action: Action,
        queue: &mut VecDeque<ProtocolEvent>,
    ) {
        match action {
            Action::VerifyBeaconBlock {
                block,
                committee,
                active_pool,
                equivocation_signers,
            } => {
                // Inline signature verify of the block's certs — no off-thread
                // dispatch. The result re-enters the same vnode as a
                // continuation so adoption runs within this cascade.
                let coordinator = self.vnodes[vnode_idx].state.beacon_coordinator();
                let network = coordinator.network_definition();
                let verifier = Arc::clone(coordinator.verifier());
                let result = Arc::unwrap_or_clone(block)
                    .upgrade(&CertifiedBeaconBlockVerifyContext {
                        verifier: verifier.as_ref(),
                        network,
                        committee: &committee,
                        active_pool: &active_pool,
                        equivocation_signers: &equivocation_signers,
                    })
                    .map(Arc::new)
                    .map_err(|(_, e)| e);
                queue.push_back(ProtocolEvent::BeaconBlockVerified { result });
            }
            Action::CommitBeaconBlock { block, state } => {
                let epoch = block.epoch();
                // Process-scoped dedup: the first vnode to reach this
                // `(epoch, hash)` writes to the host's beacon storage. A pooled
                // vnode no-ops `BeaconBlockPersisted`, so it isn't fed back.
                self.process
                    .beacon_commit
                    .commit(&self.process.beacon_storage, &block, &state);
                // Advance the sync FSM's committed watermark on every commit
                // (gossip or sync) so a serial catch-up unblocks the next
                // epoch's fetch and a later sync starts from current+1.
                beacon::on_admitted(self, epoch);
            }
            Action::TopologyChanged {
                epoch,
                topology_snapshot,
                routing_committees,
            } => {
                self.process
                    .apply_topology(epoch, &topology_snapshot, routing_committees);
            }
            Action::ReconfigureParticipation(change) => {
                self.pending_participation_changes.push(change);
            }
            // A follower's beacon timer chain is liveness-critical: it is a
            // ratify-pool voter (`BeaconRatifyTrigger` starts and paces its
            // skip rounds) and can be drawn onto an SPC committee
            // (`BeaconCommitteeStart` and the SPC timers). Buffer the ops for
            // the runner's timer table, keyed by `TimerId` alone — no shard.
            Action::SetTimer { id, duration } => {
                self.pending_timer_ops.push(TimerOp::Set {
                    shard: None,
                    id,
                    duration,
                });
            }
            Action::CancelTimer { id } => {
                self.pending_timer_ops
                    .push(TimerOp::Cancel { shard: None, id });
            }
            // A follower never amplifies blocks — committee members broadcast
            // the commits it folds.
            Action::BroadcastBeaconBlock { .. } => {}
            // Catch-up sync: a follower fell behind a gossiped block, so drive
            // the FSM to fetch the missing epochs from a live committee.
            Action::StartBeaconBlockSync { target } => {
                beacon::start(self, target);
            }
            other => {
                warn!(
                    action = other.type_name(),
                    "PoolLoop: unexpected action from a beacon follower — dropping"
                );
            }
        }
    }

    /// Whether a catch-up sync is in flight — actively fetching or holding
    /// epochs deferred behind a backoff. The driver ticks the pool while true
    /// and lets it idle otherwise.
    #[must_use]
    pub fn is_beacon_syncing(&self) -> bool {
        beacon::has_pending(&self.beacon_block)
    }

    /// Pick a live shard whose committee can serve the follower's beacon
    /// fetch. Every shard member holds the beacon chain, so any live leaf
    /// answers; spreading by the follower's own id keeps a host of followers
    /// from all hammering one committee.
    fn fetch_shard(&self) -> Option<ShardId> {
        let vnode = self.vnodes.first()?;
        let leaves: Vec<ShardId> = vnode
            .state
            .topology_snapshot()
            .shard_trie()
            .leaves()
            .collect();
        if leaves.is_empty() {
            return None;
        }
        let idx = usize::try_from(vnode.validator_id.inner() % leaves.len() as u64)
            .expect("modulo of leaves.len() fits usize");
        Some(leaves[idx])
    }
}

impl<S, N, D> BeaconSyncSink for PoolLoop<S, N, D>
where
    S: ShardStorage,
    N: Network,
    D: Dispatch,
{
    fn beacon_fsm(&mut self) -> &mut BeaconBlockSync {
        &mut self.beacon_block
    }

    fn deliver_block(&mut self, block: Arc<Verifiable<CertifiedBeaconBlock>>) {
        // Deliver to every follower inline — each runs the verify/adopt/commit
        // cascade to quiescence within this call.
        self.dispatch_protocol(ProtocolEvent::BeaconBlockSyncReadyToApply { block });
    }

    fn dispatch_fetch(&self, epoch: Epoch) {
        let Some(shard) = self.fetch_shard() else {
            warn!("PoolLoop: no live shard to fetch beacon blocks from; deferring");
            return;
        };
        let beacon_tx = self.process.beacon_event_sender.clone();
        self.process.network.request(
            shard,
            None,
            GetBeaconBlockRequest::new(epoch),
            None,
            Box::new(
                move |result: Result<GetBeaconBlockResponse, RequestError>| {
                    let input = match result {
                        Ok(resp) => PoolScopedInput::BeaconBlockSyncResponseReceived {
                            epoch,
                            block: resp.block,
                        },
                        Err(err) => PoolScopedInput::BeaconBlockSyncFetchFailed {
                            epoch,
                            kind: classify_fetch_error(&err),
                        },
                    };
                    let _ = beacon_tx.send(HostEvent::Beacon(input));
                    // "Peer doesn't have this epoch" is ambiguous (it may be
                    // behind us) — never Reject.
                    ResponseVerdict::Accept
                },
            ),
        );
    }

    fn beacon_tip(&self) -> Option<Epoch> {
        self.process.beacon_storage.latest_committed_epoch()
    }

    fn now(&self) -> LocalTimestamp {
        self.now
    }
}
