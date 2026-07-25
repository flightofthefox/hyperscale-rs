//! The simulation session and its event derivation.

use std::collections::BTreeMap;
use std::time::Duration;

use hyperscale_simulation::{SimConfig, SimulationRunner};
use hyperscale_storage::ShardChainReader;
use hyperscale_types::{BlockHeight, ShardId};

use crate::event::TraceEvent;

/// A running simulation plus the watermark of what has been reported.
pub struct Session {
    runner: SimulationRunner,
    now: Duration,
    /// Highest height already emitted per shard. A shard absent from the map
    /// has had nothing emitted yet.
    reported_through: BTreeMap<ShardId, BlockHeight>,
}

impl Session {
    /// Build a cluster at `seed` and run genesis.
    #[must_use]
    pub fn new(config: &SimConfig, seed: u64) -> Self {
        let mut runner = SimulationRunner::new(config, seed);
        runner.initialize_genesis();
        Self {
            runner,
            now: Duration::ZERO,
            reported_through: BTreeMap::new(),
        }
    }

    /// Advance simulated time by `ms` and return everything observed.
    pub fn step(&mut self, ms: u64) -> Vec<TraceEvent> {
        self.now += Duration::from_millis(ms);
        self.runner.run_until(self.now);
        self.drain_committed()
    }

    /// Shards with at least one host serving them, in trie order.
    #[must_use]
    pub fn live_shards(&self) -> Vec<ShardId> {
        let mut shards: Vec<ShardId> = (0..self.runner.num_hosts())
            .flat_map(|host| self.runner.hosted_shards_of(host))
            .collect();
        shards.sort_unstable();
        shards.dedup();
        shards
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
            let reported = self
                .reported_through
                .get(&shard)
                .map_or(0, |h| h.inner().max(1));

            // Hosts of one shard agree on committed content, so the first
            // one serving it answers for all of them.
            let Some(storage) =
                (0..self.runner.num_hosts()).find_map(|host| self.runner.hosts_shard(host, shard))
            else {
                continue;
            };
            let committed = storage.committed_height().inner();

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
