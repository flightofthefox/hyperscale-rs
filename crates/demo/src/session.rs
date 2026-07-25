//! The simulation session and its event derivation.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use hyperscale_node::shard::{HostEvent, ProcessScopedInput};
use hyperscale_simulation::{EPOCH_MS, SimConfig, SimulationRunner};
use hyperscale_storage::ShardChainReader;
use hyperscale_types::{
    BeaconChainConfig, BlockHeight, ReshapeThresholds, ShardId, TimestampRange,
    TransactionDecision, TransactionStatus, TxHash, WeightedTimestamp, build_transfer_tx,
};
use radix_common::crypto::Ed25519PrivateKey;
use radix_common::math::Decimal;
use radix_common::network::NetworkDefinition;
use radix_common::types::ComponentAddress;

use crate::event::TraceEvent;

/// The signer for demo account `seed`, and the preallocated account it
/// controls. Deterministic so a seeded session funds and spends the same
/// accounts every run.
fn signer_from_seed(seed: u8) -> Ed25519PrivateKey {
    Ed25519PrivateKey::from_bytes(&[seed; 32]).expect("32 bytes is a valid Ed25519 key")
}

fn account_from_seed(seed: u8) -> ComponentAddress {
    ComponentAddress::preallocated_account_from_public_key(&signer_from_seed(seed).public_key())
}

/// A validity window bracketing `now`, wide enough that a transaction stays
/// valid while it waits out ordering and settlement.
fn validity_around(now: Duration) -> TimestampRange {
    TimestampRange::new(
        WeightedTimestamp::ZERO.plus(now.saturating_sub(Duration::from_secs(5))),
        WeightedTimestamp::ZERO.plus(now + Duration::from_secs(150)),
    )
}

/// Genesis-funded accounts the load generator draws from. Small enough that
/// every transfer pair is visually distinguishable, large enough that
/// consecutive transfers rarely contend on the same account — contending
/// transfers are held by the ready set (INV-EXEC-3) rather than run, which
/// looks like a stall to anyone watching.
const ACCOUNTS: u8 = 8;

/// The terminal outcome, in the vocabulary the docs use.
const fn decision_label(decision: TransactionDecision) -> &'static str {
    match decision {
        TransactionDecision::Accept => "succeeded",
        TransactionDecision::Reject => "rejected",
        TransactionDecision::Aborted => "aborted",
    }
}

/// How the cluster a session opens is shaped.
#[derive(Debug, Clone, Copy)]
pub struct SessionConfig {
    /// Validators per shard committee.
    pub shard_size: u32,
    /// Leaves the topology may grow to. Must be a power of two.
    ///
    /// The session always *starts* at a single ROOT shard, because that is
    /// where every network starts. Above one, the split trigger is armed and
    /// the pool is staffed for the splits it allows, so the topology grows
    /// while the session runs and a viewer sees the reshape happen rather
    /// than arriving after it. Growth stops on its own once the pool can no
    /// longer staff a child committee: admission is gated on a deep enough
    /// free pool, so the surplus is the ceiling.
    pub max_shards: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            shard_size: 4,
            max_shards: 1,
        }
    }
}

/// A running simulation plus the watermarks of what has been reported.
pub struct Session {
    runner: SimulationRunner,
    now: Duration,
    /// Highest height already emitted per shard. A shard absent from the map
    /// has had nothing emitted yet.
    reported_through: BTreeMap<ShardId, BlockHeight>,
    /// The partition as last reported, so a step emits only changes.
    reported_shards: Vec<ShardId>,
    /// Submitted transactions and the last status reported for each, so a
    /// step emits only transitions.
    tracked: BTreeMap<TxHash, Option<TransactionStatus>>,
    /// Events raised between steps — submissions happen on the caller's
    /// clock, not the simulation's, so they wait here for the next drain.
    pending: Vec<TraceEvent>,
    nonce: u32,
}

impl Session {
    /// Build a single-shard cluster at `seed`, fund [`ACCOUNTS`] accounts, and
    /// run genesis.
    ///
    /// The topology grows from here as the session steps, up to
    /// [`SessionConfig::max_shards`].
    #[must_use]
    pub fn new(config: SessionConfig, seed: u64) -> Self {
        let splits = config.max_shards.saturating_sub(1);
        let sim_config = SimConfig {
            shard_size: config.shard_size,
            // Each split staffs its children from the free pool, so the grow
            // needs one spare cohort per split or the readiness gate never
            // passes (INV-RESHAPE-1).
            pool_surplus: splits * config.shard_size,
            beacon_chain_config: Some(BeaconChainConfig {
                shard_size: config.shard_size,
                // The default five-minute epoch makes a split span thousands
                // of blocks, which is minutes of boot before the page shows
                // anything. The simulation's own epoch is the shortest value
                // the recovery timeouts are still sized against.
                epoch_duration_ms: EPOCH_MS,
                // Arm the split trigger unconditionally: the demo grows on
                // demand rather than waiting for a shard to outgrow a byte
                // threshold in real time.
                reshape_thresholds: ReshapeThresholds { split_bytes: 0 },
                ..BeaconChainConfig::default()
            }),
            ..SimConfig::default()
        };
        let mut runner = SimulationRunner::new(&sim_config, seed);
        let balances: Vec<_> = (1..=ACCOUNTS)
            .map(|s| (account_from_seed(s), Decimal::from(100_000)))
            .collect();
        runner.initialize_genesis_with_balances(&balances);
        // Seed the partition watermark with the genesis topology, so the
        // first change reported is a real one rather than the session
        // announcing the shard it opened on.
        let opening = (0..runner.num_hosts())
            .find_map(|host| runner.host_topology(host))
            .map(|topology| topology.shard_trie().leaves().collect())
            .unwrap_or_default();
        Self {
            runner,
            now: Duration::ZERO,
            reported_shards: opening,
            reported_through: BTreeMap::new(),
            tracked: BTreeMap::new(),
            pending: Vec::new(),
            nonce: 0,
        }
    }

    /// Submit an XRD transfer between two funded accounts, returning its hash.
    ///
    /// Payer and payee rotate with the nonce, so a caller driving a steady
    /// rate spreads load across accounts instead of serializing on one.
    ///
    /// # Panics
    ///
    /// Panics if the transfer fails to build, which for genesis-funded demo
    /// accounts means a malformed manifest — a programming error, not input.
    pub fn submit_transfer(&mut self) -> TxHash {
        let from = u8::try_from(self.nonce % u32::from(ACCOUNTS)).unwrap_or(0) + 1;
        let to = (from % ACCOUNTS) + 1;
        let tx = build_transfer_tx(
            &signer_from_seed(from),
            account_from_seed(from),
            account_from_seed(to),
            Decimal::from(1),
            &NetworkDefinition::simulator(),
            self.nonce,
            validity_around(self.now),
        )
        .expect("a transfer between funded demo accounts builds");
        let hash = tx.hash();
        self.runner.schedule_initial_event(
            0,
            Duration::from_millis(1),
            HostEvent::process(ProcessScopedInput::SubmitTransaction { tx: Arc::new(tx) }),
        );
        self.nonce += 1;
        self.tracked.insert(hash, None);
        let wt = u64::try_from(self.now.as_millis()).unwrap_or(u64::MAX);
        self.pending.push(TraceEvent::tx_submitted(wt, hash));
        hash
    }

    /// Advance simulated time by `ms` and return everything observed.
    pub fn step(&mut self, ms: u64) -> Vec<TraceEvent> {
        // Reshape duties are driven by the harness, not by the event queue:
        // an orchestrator only advances on a step. Driving it here is what
        // lets a split unfold across the session instead of being completed
        // before the first frame.
        self.runner.reshape_step();
        self.now += Duration::from_millis(ms);
        self.runner.run_until(self.now);
        let mut events = std::mem::take(&mut self.pending);
        events.extend(self.drain_topology());
        events.extend(self.drain_committed());
        events.extend(self.drain_tx_status());
        // One batch, one timeline: the viewer renders in weighted-time order
        // regardless of which derivation produced an event.
        events.sort_by_key(|event| event.wt);
        events
    }

    /// Report a partition change, if the trie's leaves moved this step.
    fn drain_topology(&mut self) -> Vec<TraceEvent> {
        let current = self.live_shards();
        if current == self.reported_shards {
            return Vec::new();
        }
        let appeared = current
            .iter()
            .filter(|s| !self.reported_shards.contains(s))
            .copied()
            .collect();
        let retired = self
            .reported_shards
            .iter()
            .filter(|s| !current.contains(s))
            .copied()
            .collect();
        let wt = u64::try_from(self.now.as_millis()).unwrap_or(u64::MAX);
        let event = TraceEvent::topology_changed(wt, &current, appeared, retired);
        self.reported_shards = current;
        vec![event]
    }

    /// Report every tracked transaction whose status moved this step.
    ///
    /// Polled rather than pushed: status is a projection of committed chain
    /// content, so reading it back is an observation, not a hook into the
    /// path that produces it.
    fn drain_tx_status(&mut self) -> Vec<TraceEvent> {
        let wt = u64::try_from(self.now.as_millis()).unwrap_or(u64::MAX);
        let mut events = Vec::new();
        for (hash, last) in &mut self.tracked {
            let current = self.runner.tx_status(0, hash);
            if current.as_ref() == last.as_ref() {
                continue;
            }
            if let Some(status) = current.clone() {
                let (label, height) = match &status {
                    TransactionStatus::Pending => ("pending", None),
                    TransactionStatus::Committed(h) => ("committed", Some(h.inner())),
                    TransactionStatus::Completed(decision) => (decision_label(*decision), None),
                };
                events.push(TraceEvent::tx_status(wt, *hash, label, height));
            }
            *last = current;
        }
        events
    }

    /// The shards the topology currently partitions the keyspace into, in
    /// trie order.
    ///
    /// Read from the beacon-derived topology rather than from what hosts
    /// happen to store: a split parent's store is retained past its terminal
    /// block so late joiners and counterparties can still resolve it
    /// (INV-BEACON-8), so host storage lists shards that no longer exist.
    /// The trie's leaves are the live partition by definition.
    #[must_use]
    pub fn live_shards(&self) -> Vec<ShardId> {
        (0..self.runner.num_hosts())
            .find_map(|host| self.runner.host_topology(host))
            .map(|topology| topology.shard_trie().leaves().collect())
            .unwrap_or_default()
    }

    /// Walk each shard's newly committed blocks and emit one event apiece.
    ///
    /// A block is reported only once its committing child's header is
    /// readable, because that child's parent QC carries the block's canonical
    /// weighted timestamp (INV-SHARD-6). Reading the block's own QC instead
    /// would pick up whatever round it was last certified in — a gossip
    /// artifact, not consensus content — and the timeline would jitter under
    /// re-certification. The cost is that the chain tip is reported one block
    /// late, which is invisible at any watchable playback rate.
    fn drain_committed(&mut self) -> Vec<TraceEvent> {
        let mut events = Vec::new();
        let mut watermarks: Vec<(ShardId, BlockHeight)> = Vec::new();

        for shard in self.live_shards() {
            // Hosts of one shard agree on committed content, so the first
            // one serving it answers for all of them.
            let Some(storage) =
                (0..self.runner.num_hosts()).find_map(|host| self.runner.hosts_shard(host, shard))
            else {
                continue;
            };
            let committed = storage.committed_height().inner();

            // On first sight, start at the tip rather than replaying the
            // chain: a split child seeds at its parent's terminal height, so
            // its first height is wherever the parent stopped and heights
            // below that were never its own. Recorded straight away so the
            // next step resumes from here instead of treating the shard as
            // new again and skipping ahead.
            let reported = self.reported_through.get(&shard).map_or_else(
                || {
                    let start = committed.saturating_sub(1);
                    watermarks.push((shard, BlockHeight::new(start)));
                    start
                },
                |height| height.inner(),
            );

            // The tip has no committing child yet, so stop one short of it.
            for height in (reported + 1)..committed {
                let Some(header) = storage.get_certified_header(BlockHeight::new(height)) else {
                    break;
                };
                let Some(child) = storage.get_certified_header(BlockHeight::new(height + 1)) else {
                    break;
                };
                let header = header.as_ref().header();
                events.push(TraceEvent::block_committed(
                    child
                        .as_ref()
                        .header()
                        .parent_qc()
                        .weighted_timestamp()
                        .as_millis(),
                    shard,
                    header.height(),
                    header.round(),
                    header.is_fallback(),
                    header.proposer().inner(),
                    u32::try_from(header.waves().len()).unwrap_or(u32::MAX),
                ));
                watermarks.push((shard, BlockHeight::new(height)));
            }
        }

        for (shard, height) in watermarks {
            self.reported_through.insert(shard, height);
        }
        events.sort_by_key(|event| event.wt);
        events
    }
}
