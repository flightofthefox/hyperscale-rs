//! Package artifact acquisition: the fetch that makes the beacon's
//! package registry locally runnable.
//!
//! Every beacon commit reconciles the global registry against what the
//! engine holds; anything missing is fetched from the shard owning its
//! publisher's prefix, verified by hashing the returned bytes, installed
//! into the engine, and persisted beside the beacon store so a restart
//! reconciles instead of refetching the world.

use std::collections::BTreeMap;
use std::sync::Arc;

use crossbeam::channel::Sender;
use hyperscale_dispatch::Dispatch;
use hyperscale_engine::artifact_package;
use hyperscale_network::{Network, ResponseVerdict};
use hyperscale_storage::ShardStorage;
use hyperscale_types::network::request::{
    GetPackageArtifactsRequest, MAX_PACKAGE_ARTIFACTS_PER_REQUEST,
};
use hyperscale_types::{BeaconState, Hash, MessageClass, ShardId, ValidatorId};

use crate::config::NodeConfig;
use crate::fetch::{Fetch, FetchBinding, FetchInput, partition_solicited};
use crate::shard::{HostEvent, ShardIo, ShardLoop, ShardScopedInput, push_shard_input};

/// Per-package artifact fetch keyed by content address.
pub type PackageArtifactFetch = Fetch<Hash>;

/// Per-shard package-acquisition state: the artifact fetch instance.
pub struct PackagesState {
    pub(crate) fetch: PackageArtifactFetch,
}

impl PackagesState {
    pub fn new(config: &NodeConfig) -> Self {
        // The wire caps the batch far below the generic default: an
        // artifact runs to a transaction's whole byte budget.
        let mut fetch_config = config.package_artifact_fetch.clone();
        fetch_config.max_ids_per_request = fetch_config
            .max_ids_per_request
            .min(MAX_PACKAGE_ARTIFACTS_PER_REQUEST);
        Self {
            fetch: PackageArtifactFetch::new("package_artifact", fetch_config),
        }
    }

    /// Whether the artifact fetch has work the tick loop should drive.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.fetch.has_pending()
    }
}

/// Marker type for the package artifact fetch.
pub struct PackageArtifactBinding;

impl FetchBinding for PackageArtifactBinding {
    type Id = Hash;

    const NAME: &'static str = "package_artifact";

    fn fetch_mut<S: ShardStorage>(shard: &mut ShardIo<S>) -> &mut Fetch<Hash> {
        &mut shard.packages.fetch
    }

    fn dispatch_chunk<N: Network>(
        ids: Vec<Hash>,
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
            GetPackageArtifactsRequest::new(ids),
            class,
            Box::new(move |result| {
                if let Ok(resp) = result {
                    // Hashing the bytes is the whole verification: an
                    // artifact either is the one asked for or is dropped
                    // here, before anything installs it.
                    let split = partition_solicited(resp.artifacts, &requested, |artifact| {
                        [artifact_package(artifact)]
                    });
                    if !split.kept.is_empty() {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::PackageArtifactsFetched {
                                artifacts: split.kept,
                            },
                        );
                    }
                    if !split.missing.is_empty() {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::PackageArtifactsFetchFailed {
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
                        ShardScopedInput::PackageArtifactsFetchFailed { ids: requested },
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
    /// Reconcile the beacon's package registry against what the engine
    /// holds, fetching anything missing from the shard that owns its
    /// publisher's prefix.
    ///
    /// Runs on every beacon commit and tolerates arbitrary staleness:
    /// content addressing makes every enqueue idempotent, and a newly
    /// seated or restarted node catches the whole backlog on its first
    /// commit.
    pub(crate) fn reconcile_packages(&mut self, state: &BeaconState) {
        let executor = Arc::clone(&self.process.dispatch_handles.executor);
        let snapshot = self.process.topology_snapshot.load();
        let mut by_shard: BTreeMap<ShardId, Vec<Hash>> = BTreeMap::new();
        for (package, fact) in &state.packages {
            if !executor.package_known(*package) {
                by_shard
                    .entry(snapshot.shard_trie().shard_for_prefix(fact.publisher))
                    .or_default()
                    .push(*package);
            }
        }
        drop(snapshot);
        for (shard, ids) in by_shard {
            self.drive_fetch::<PackageArtifactBinding>(FetchInput::Request {
                ids,
                shard,
                preferred: None,
                class: None,
            });
        }
    }

    /// Install verified fetched artifacts: code into the engine, bytes
    /// into the node-level cache a restart reconciles from.
    pub(crate) fn handle_package_artifacts_fetched(&mut self, artifacts: &[Vec<u8>]) {
        let handles = Arc::clone(&self.process.dispatch_handles);
        let mut ids = Vec::with_capacity(artifacts.len());
        for artifact in artifacts {
            let package = artifact_package(artifact);
            handles.executor.install_artifact(artifact);
            handles
                .beacon_storage
                .store_fetched_package(package, artifact);
            ids.push(package);
        }
        self.drive_fetch::<PackageArtifactBinding>(FetchInput::Admitted { ids });
    }
}
