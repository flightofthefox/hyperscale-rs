//! Instance record acquisition: the fetch that lets a node resolve a
//! component whose seal it never committed.
//!
//! A component's record is committed on the shard owning its own prefix,
//! so every other shard holds none of it — yet each has to resolve the
//! target to say what a transaction naming it declares. What closes the
//! gap is a fetch keyed by the component's address, verified by
//! re-deriving that address from the record's own contents, and seated
//! in the registry derivation reads.
//!
//! Discovery is demand-driven, unlike the artifact fetch beside it.
//! Packages reconcile against the beacon's registry, which lists every
//! one of them; components have no such registry and could not have one,
//! so what a node asks for is what a derivation just told it was
//! missing.

use std::collections::BTreeMap;
use std::sync::Arc;

use crossbeam::channel::Sender;
use hyperscale_dispatch::{Dispatch, DispatchPool};
use hyperscale_engine::instance_of_record;
use hyperscale_network::{Network, ResponseVerdict};
use hyperscale_storage::ShardStorage;
use hyperscale_types::network::request::{
    GetInstanceRecordsRequest, MAX_INSTANCE_RECORDS_PER_REQUEST,
};
use hyperscale_types::{Address, MessageClass, ShardId, ValidatorId};

use crate::config::NodeConfig;
use crate::fetch::{Fetch, FetchBinding, FetchInput, partition_solicited};
use crate::shard::{HostEvent, ShardIo, ShardLoop, ShardScopedInput, push_shard_input};

/// Per-component record fetch, keyed by the address the record derives.
pub type InstanceRecordFetch = Fetch<Address>;

/// Per-shard record-acquisition state: the record fetch instance.
pub struct InstancesState {
    pub(crate) fetch: InstanceRecordFetch,
}

impl InstancesState {
    pub fn new(config: &NodeConfig) -> Self {
        let mut fetch_config = config.instance_record_fetch.clone();
        fetch_config.max_ids_per_request = fetch_config
            .max_ids_per_request
            .min(MAX_INSTANCE_RECORDS_PER_REQUEST);
        Self {
            fetch: InstanceRecordFetch::new("instance_record", fetch_config),
        }
    }

    /// Whether the record fetch has work the tick loop should drive.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.fetch.has_pending()
    }
}

/// Marker type for the instance record fetch.
pub struct InstanceRecordBinding;

impl FetchBinding for InstanceRecordBinding {
    type Id = Address;

    const NAME: &'static str = "instance_record";

    fn fetch_mut<S: ShardStorage>(shard: &mut ShardIo<S>) -> &mut Fetch<Address> {
        &mut shard.instances.fetch
    }

    fn dispatch_chunk<N: Network>(
        ids: Vec<Address>,
        local_shard: ShardId,
        shard: ShardId,
        preferred: Option<ValidatorId>,
        class: Option<MessageClass>,
        network: &N,
        sender: &Sender<HostEvent>,
    ) {
        let es = sender.clone();
        let requested = ids.clone();
        network.request(
            shard,
            preferred,
            GetInstanceRecordsRequest::new(ids),
            class,
            Box::new(move |result| {
                if let Ok(resp) = result {
                    // Re-deriving the address from the record is the
                    // whole verification: a component's address is the
                    // hash of its record, so bytes deriving anything
                    // else are dropped here, before the registry sees
                    // them. The address travels on beside the bytes, so
                    // nothing downstream derives it again.
                    let addressed: Vec<(Address, Vec<u8>)> = resp
                        .records
                        .into_iter()
                        .filter_map(|record| {
                            instance_of_record(&record).map(|address| (address, record))
                        })
                        .collect();
                    let split =
                        partition_solicited(addressed, &requested, |(address, _)| [*address]);
                    if !split.kept.is_empty() {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::InstanceRecordsFetched {
                                records: split.kept,
                            },
                        );
                    }
                    if !split.missing.is_empty() {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::InstanceRecordsFetchFailed {
                                ids: split.missing.clone(),
                            },
                        );
                    }
                    if split.unsolicited > 0 || !split.missing.is_empty() {
                        ResponseVerdict::Reject
                    } else {
                        ResponseVerdict::Accept
                    }
                } else {
                    push_shard_input(
                        &es,
                        local_shard,
                        ShardScopedInput::InstanceRecordsFetchFailed { ids: requested },
                    );
                    ResponseVerdict::Accept
                }
            }),
        );
    }
}

impl<S, N, D> ShardLoop<S, N, D>
where
    S: ShardStorage,
    N: Network,
    D: Dispatch,
{
    /// Ask the shard owning each component's prefix for its record.
    ///
    /// Called with what a derivation could not resolve. Idempotent: a
    /// record is immutable once sealed and self-verifying on arrival, so
    /// asking twice costs a round trip and settles the same way.
    pub(crate) fn fetch_instance_records(&mut self, instances: Vec<Address>) {
        let executor = Arc::clone(&self.process.dispatch_handles.executor);
        let snapshot = self.process.topology_snapshot.load();
        let mut by_shard: BTreeMap<ShardId, Vec<Address>> = BTreeMap::new();
        for instance in instances {
            if executor.instance_known(instance) {
                continue;
            }
            by_shard
                .entry(snapshot.shard_trie().shard_for_prefix(instance))
                .or_default()
                .push(instance);
        }
        drop(snapshot);
        for (shard, ids) in by_shard {
            self.drive_fetch::<InstanceRecordBinding>(FetchInput::Request {
                ids,
                shard,
                preferred: None,
                class: None,
            });
        }
    }

    /// Seat verified fetched records in the registry derivation reads.
    ///
    /// Off the shard loop, as the artifact install is: seating decodes a
    /// record and re-derives its address, and nothing here is on the
    /// path of a verdict — the bytes were verified against the address
    /// asked for before this was posted.
    pub(crate) fn handle_instance_records_fetched(&mut self, records: Vec<(Address, Vec<u8>)>) {
        let ids: Vec<Address> = records.iter().map(|(address, _)| *address).collect();
        let handles = Arc::clone(&self.process.dispatch_handles);
        self.process
            .dispatch
            .spawn(DispatchPool::Throughput, move || {
                for (instance, record) in records {
                    handles.executor.install_instance(instance, &record);
                }
            });
        self.drive_fetch::<InstanceRecordBinding>(FetchInput::Admitted { ids });
    }
}
