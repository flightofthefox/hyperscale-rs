//! Beacon timer-chain liveness at the vnode-seating seams.
//!
//! Every ratify-eligible validator runs a live beacon timer chain: a pool
//! follower's `BeaconRatifyTrigger` re-arm survives to the pool's timer
//! seam instead of being discarded, and a vnode constructed mid-life —
//! seated or pooled, no genesis ceremony — arms the beacon startup timers
//! (`BeaconCommitteeStart` + `BeaconRatifyTrigger`) the way
//! `initialize_genesis` arms them for a genesis-born vnode. Without the
//! chain, a validator adopts beacon blocks by gossip but never proposes,
//! never drives SPC rounds, and never casts ratify votes; enough of them
//! in one ratify pool starves the pool quorum and parks the beacon.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arc_swap::ArcSwap;
use crossbeam::channel::unbounded;
use hyperscale_beacon::genesis::build_genesis_beacon_state;
use hyperscale_core::{ProtocolEvent, TimerId};
use hyperscale_crypto_bls::BlsVerifier;
use hyperscale_dispatch_sync::SyncDispatch;
use hyperscale_engine::{RadixExecutor, TransactionValidation};
use hyperscale_mempool::MempoolConfig;
use hyperscale_network::HandlerRegistry;
use hyperscale_network_memory::SimNetworkAdapter;
use hyperscale_node::shard::HostEvent;
use hyperscale_node::{
    NodeConfig, NodeHost, SeatFollower, SeatVnodeGroup, TimerOp, VnodeInit, seat_follower,
    seat_vnode_group,
};
use hyperscale_provisions::ProvisionConfig;
use hyperscale_shard::ShardConsensusConfig;
use hyperscale_storage::{BeaconStorage, RecoveredState};
use hyperscale_storage_memory::{SimBeaconStorage, SimShardStorage};
use hyperscale_types::test_utils::TestCommittee;
use hyperscale_types::{
    BeaconChainConfig, BeaconGenesisConfig, CertifiedBeaconBlock, GenesisConfigHash, GenesisPool,
    GenesisValidator, LocalTimestamp, MIN_STAKE_FLOOR, NetworkDefinition, Randomness, ShardId,
    Signer, Stake, StakePoolId, TopologySnapshot, ValidatorId, ValidatorInfo, ValidatorSet,
    Verified, genesis_config_hash, shard_prefix_path,
};

const SHARD_A: ShardId = ShardId::leaf(1, 0);

/// Shared genesis: four validators in one pool, all on the initial
/// committees; shard A staffed by validators 0–1.
struct Fixture {
    committee: TestCommittee,
    config_hash: GenesisConfigHash,
    topology_snapshot: Arc<TopologySnapshot>,
    beacon_storage: Arc<dyn BeaconStorage>,
}

fn fixture() -> Fixture {
    let committee = TestCommittee::new(4, 7);
    let network = NetworkDefinition::simulator();
    let pool_id = StakePoolId::new(0);
    let initial_validators: Vec<GenesisValidator> = (0..4)
        .map(|i| GenesisValidator {
            id: committee.validator_id(i),
            pool: pool_id,
            pubkey: *committee.public_key(i),
        })
        .collect();
    let config = BeaconGenesisConfig {
        chain_config: BeaconChainConfig::default(),
        initial_validators,
        initial_pools: vec![GenesisPool {
            id: pool_id,
            total_stake: Stake::from_attos(4 * MIN_STAKE_FLOOR.attos()),
        }],
        initial_beacon_committee: (0..4).map(|i| committee.validator_id(i)).collect(),
        initial_shard_committee: (0..4).map(|i| committee.validator_id(i)).collect(),
        initial_randomness: Randomness::new([0x42; 32]),
    };
    let genesis_state = build_genesis_beacon_state(&config);
    let config_hash = genesis_config_hash(&config, &network);
    let genesis_block = Arc::new(Verified::<CertifiedBeaconBlock>::genesis(config_hash));

    let validator_set = ValidatorSet::new(
        (0..4)
            .map(|i| ValidatorInfo {
                validator_id: committee.validator_id(i),
                public_key: *committee.public_key(i),
            })
            .collect(),
    );
    let shard_committees: HashMap<ShardId, Vec<ValidatorId>> = std::iter::once((
        SHARD_A,
        vec![committee.validator_id(0), committee.validator_id(1)],
    ))
    .collect();
    let topology_snapshot = Arc::new(TopologySnapshot::with_shard_committees(
        network,
        2,
        &validator_set,
        shard_committees,
    ));
    let beacon_storage: Arc<dyn BeaconStorage> = Arc::new(SimBeaconStorage::new());
    beacon_storage.commit_beacon_block(&genesis_block, &Arc::new(genesis_state));
    Fixture {
        committee,
        config_hash,
        topology_snapshot,
        beacon_storage,
    }
}

impl Fixture {
    /// A shard-less beacon follower for `committee[idx]`, built through the
    /// real seating seam.
    fn follower_init(&self, idx: usize) -> VnodeInit {
        seat_follower(SeatFollower {
            verifier: Arc::new(BlsVerifier),
            beacon_storage: self.beacon_storage.as_ref(),
            beacon_network: NetworkDefinition::simulator(),
            beacon_config_hash: self.config_hash,
            now: LocalTimestamp::ZERO,
            validator: self.committee.validator_id(idx),
            signer: self.committee.signer(idx),
        })
    }

    /// A shard-seated vnode group for `committee[idx]` on `shard`, built
    /// through the real seating seam — the runtime-join construction, not
    /// the genesis ceremony.
    fn seated_inits(&self, idx: usize, shard: ShardId) -> Vec<VnodeInit> {
        let vnodes: Vec<(ValidatorId, Arc<dyn Signer>)> =
            vec![(self.committee.validator_id(idx), self.committee.signer(idx))];
        seat_vnode_group(SeatVnodeGroup {
            verifier: Arc::new(BlsVerifier),
            beacon_storage: self.beacon_storage.as_ref(),
            beacon_network: NetworkDefinition::simulator(),
            beacon_config_hash: self.config_hash,
            now: LocalTimestamp::ZERO,
            shard,
            recovered: &RecoveredState::default(),
            shard_config: &ShardConsensusConfig::default(),
            mempool_config: MempoolConfig::default(),
            provision_config: ProvisionConfig::default(),
            vnodes,
        })
    }

    /// A follower-only host: no hosted shards, one pooled vnode.
    fn follower_host(
        &self,
        idx: usize,
    ) -> NodeHost<SimShardStorage, SimNetworkAdapter, SyncDispatch> {
        let registry = Arc::new(HandlerRegistry::new(std::collections::BTreeSet::new()));
        let network = SimNetworkAdapter::new(registry);
        let (event_tx, _event_rx) = unbounded::<HostEvent>();
        NodeHost::new(
            vec![self.follower_init(idx)],
            HashMap::<ShardId, SimShardStorage>::new(),
            Arc::clone(&self.beacon_storage),
            NetworkDefinition::simulator(),
            RadixExecutor::new(NetworkDefinition::simulator()),
            network,
            SyncDispatch,
            BTreeMap::new(),
            event_tx,
            Arc::new(ArcSwap::from(Arc::clone(&self.topology_snapshot))),
            NodeConfig::default(),
            Arc::new(TransactionValidation::new(NetworkDefinition::simulator())),
        )
    }
}

/// Whether `ops` holds a `Set` for timer `want`.
fn has_set(ops: &[TimerOp], want: &TimerId) -> bool {
    ops.iter()
        .any(|op| matches!(op, TimerOp::Set { id, .. } if id == want))
}

/// A pool follower's `BeaconRatifyTrigger` re-arm must reach the runner's
/// timer seam. The ratify handler re-arms on every fire (early or due), so
/// the chain stays alive only if the pool honors the `SetTimer` instead of
/// discarding it — a follower is a ratify-pool voter, and the trigger is
/// what starts its skip prevotes when an epoch stalls.
#[test]
fn follower_ratify_rearm_survives_to_the_timer_seam() {
    let fix = fixture();
    let mut host = fix.follower_host(0);
    host.set_time(LocalTimestamp::from_millis(1_000));

    let out = host.step(HostEvent::beacon(ProtocolEvent::BeaconRatifyTimer));

    assert!(
        has_set(&out.timer_ops, &TimerId::BeaconRatifyTrigger),
        "the follower's BeaconRatifyTrigger re-arm must survive to the pool's \
         timer seam, not be dropped: got {:?}",
        out.timer_ops
    );
}

/// A follower built mid-life (or at boot — it never runs the genesis
/// ceremony either way) must arm the beacon startup timers when it enters
/// the pool, exactly as `initialize_genesis` arms them for a genesis-born
/// vnode: `BeaconRatifyTrigger` because it votes in the ratify pool from
/// the moment it is active, and `BeaconCommitteeStart` because any pool
/// validator can be drawn onto the next epoch's SPC committee.
#[test]
fn follower_seat_arms_the_startup_timers() {
    let fix = fixture();
    let mut host = fix.follower_host(0);

    let out = host.drain_pending_output();

    assert!(
        has_set(&out.timer_ops, &TimerId::BeaconRatifyTrigger),
        "seating a follower must arm BeaconRatifyTrigger: got {:?}",
        out.timer_ops
    );
    assert!(
        has_set(&out.timer_ops, &TimerId::BeaconCommitteeStart),
        "seating a follower must arm BeaconCommitteeStart: got {:?}",
        out.timer_ops
    );
}

/// A vnode group seated mid-life — the runtime join both harnesses drive
/// via `attach_shard` + committed-state resume, never `initialize_genesis`
/// — must arm the beacon startup timers at its resume. Without them the
/// vnode adopts beacon blocks by gossip but never bootstraps its own SPC
/// participation or ratify voting.
#[test]
fn midlife_seat_arms_the_startup_timers() {
    let fix = fixture();
    // The host begins life with only a follower (validator 1); validator 0
    // is then seated onto shard A at runtime, the grow-cohort shape.
    let mut host = fix.follower_host(1);
    let (event_tx, _event_rx) = unbounded::<HostEvent>();
    host.add_shard(
        fix.seated_inits(0, SHARD_A),
        SimShardStorage::new(shard_prefix_path(SHARD_A)),
        event_tx,
    );

    let out = host.resume_shard_committed(SHARD_A, &RecoveredState::default());

    assert!(
        has_set(&out.timer_ops, &TimerId::BeaconRatifyTrigger),
        "a mid-life seat must arm BeaconRatifyTrigger at its resume: got {:?}",
        out.timer_ops
    );
    assert!(
        has_set(&out.timer_ops, &TimerId::BeaconCommitteeStart),
        "a mid-life seat must arm BeaconCommitteeStart at its resume: got {:?}",
        out.timer_ops
    );
}
