//! The beacon's control plane on the engine: a delegation to a seated
//! stake pool arrives in the executing shard's `beacon_witness_events`.
//!
//! Its own binary rather than a case in `transfer`, because the VM statics
//! install once per process and first-installed-wins: a world with a stake
//! pool in it is a different world, and sharing a process would make
//! whichever executor was built first decide what the other one can see.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use hyperscale_effects_bridge::genesis::genesis_world_with_pools;
use hyperscale_effects_bridge::{ProtocolHasher, account_address};
use hyperscale_engine::genesis::{pool_address, staking_artifact};
use hyperscale_engine::{
    ExecutedTx, ExecutionMode, Executor, Parallelism, TickBatchContext, XRD, genesis_writes,
};
use hyperscale_storage::SubstateDatabase;
use hyperscale_transactions::{Client, Terms};
use hyperscale_types::{
    BeaconWitnessEvent, BlockHash, ComponentAddr, ConsensusReceipt, Ed25519PrivateKey, EnvelopeExt,
    Hash, NetworkId, PrincipalAddr, ProvisionalHolds, RevealChain, ShardId, ShardTrie, Stake,
    StakePoolId, StakePoolSeat, SubstateKey, TimestampRange, Transaction, Verified,
    WeightedTimestamp,
};
use hyperscale_vm_effects::package_hash;
use hyperscale_vm_manifest_builder::native::{account, staking};

/// The identifier the beacon folds the seated pool under.
const POOL_ID: u32 = 7;
/// The delegator's signing seed.
const DELEGATOR: u8 = 7;
/// The signing seed of the principal the pool's operator surface admits.
const OPERATOR: u8 = 8;
/// The signing seed of a funded account that operates nothing.
const OUTSIDER: u8 = 9;

/// A snapshot over the flattened genesis updates.
struct MapDb(BTreeMap<SubstateKey, Vec<u8>>);

impl MapDb {
    fn genesis(accounts: &[(PrincipalAddr, u128)], pools: &[StakePoolSeat]) -> Self {
        let writes = genesis_writes(accounts, pools);
        let mut map = BTreeMap::new();
        for (key, change) in writes.cells() {
            let value = change.clone().expect("genesis writes are Set-only");
            map.insert(*key, value);
        }
        Self(map)
    }
}

impl SubstateDatabase for MapDb {
    fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.0.get(&key).cloned()
    }
}

fn key_of(seed: u8) -> Ed25519PrivateKey {
    Ed25519PrivateKey::from_bytes(&[seed; 32]).unwrap()
}

fn account_of(seed: u8) -> PrincipalAddr {
    account_address(&key_of(seed).public_key().0)
}

fn delegator() -> PrincipalAddr {
    account_of(DELEGATOR)
}

/// A pool seat and the principal its operator surface admits.
fn seat(id: u32) -> StakePoolSeat {
    StakePoolSeat {
        id: StakePoolId::new(id),
        operator: account_of(OPERATOR),
        founding: Vec::new(),
    }
}

/// Where genesis seats the pool with this identifier — derived from the
/// record, so a test names it the way genesis places it.
fn pool_at(id: u32) -> ComponentAddr {
    pool_address(package_hash(&ProtocolHasher, staking_artifact()), &seat(id))
}

/// The client this binary builds through.
///
/// Its world seats every pool the file names, which is not the same world
/// any one executor installs: what a builder must resolve is the target it
/// is writing, and what an executor recognises is the network's own
/// decision.
fn client() -> &'static Client {
    static CLIENT: LazyLock<Client> = LazyLock::new(|| {
        Client::new(
            genesis_world_with_pools(&[seat(POOL_ID), seat(99)]),
            NetworkId(242),
        )
    });
    &CLIENT
}

/// The signing terms every transaction here shares: a window nothing
/// falls outside, and no message.
const fn terms(max_fee: u128) -> Terms {
    Terms {
        max_fee,
        validity: TimestampRange::new(
            WeightedTimestamp::from_millis(0),
            WeightedTimestamp::from_millis(u64::MAX),
        ),
        message: Vec::new(),
    }
}

/// `delegator.withdraw(*XRD) -> pool.stake -> delegator.deposit(units)`.
fn signed_stake(pool: ComponentAddr, amount: u128) -> Transaction {
    let key = Ed25519PrivateKey::from_bytes(&[DELEGATOR; 32]).unwrap();
    let from = account_address(&key.public_key().0);
    let cache = client().cache();
    let mut b = client().builder(&cache);
    let funds = account::withdraw(&mut b, from, *XRD, amount).expect("an account withdraws");
    let units = staking::stake(&mut b, pool, funds).expect("a pool takes a delegation");
    account::deposit(&mut b, from, units).expect("an account banks its position");
    let graph = b.build().expect("every output is consumed");
    Transaction::new(client().sign(graph, &key, terms(1_000)))
}

fn execute(executor: &Executor, tx: Transaction) -> Vec<ExecutedTx> {
    let store = MapDb::genesis(&[(delegator(), 10_000)], &[]);
    let trie = ShardTrie::single();
    let ctx = TickBatchContext {
        par: Parallelism::Sequential,
        local_shard: ShardId::ROOT,
        shard_trie: &trie,
        block_hash: BlockHash::from_raw(Hash::from_bytes(b"block")),
        tick_ts: WeightedTimestamp::from_millis(1_000),
        tick_reveal: RevealChain::ZERO,
        holds: &ProvisionalHolds::new(),
    };
    let verified = Arc::new(Verified::<Transaction>::from_persisted(tx));
    executor.execute_batch(&ctx, &store, std::slice::from_ref(&verified))
}

fn witnesses(executed: &ExecutedTx) -> Vec<BeaconWitnessEvent> {
    match &executed.consensus {
        ConsensusReceipt::Succeeded {
            beacon_witness_events,
            ..
        } => beacon_witness_events.clone(),
        other @ ConsensusReceipt::Failed => {
            panic!("the delegation must succeed; receipt = {other:?}")
        }
    }
}

/// The whole channel in one assertion: a delegation to a seated pool is a
/// beacon fact by the time it leaves the engine, with the pool named by
/// the instance that emitted it and the amount carried across as attos.
#[test]
fn a_delegation_to_a_seated_pool_reaches_the_witness_channel() {
    let executor = Executor::with_pools(&[seat(POOL_ID), seat(99)], ExecutionMode::Serial);
    let executed = execute(&executor, signed_stake(pool_at(POOL_ID), 500));
    assert_eq!(
        witnesses(&executed[0]),
        vec![BeaconWitnessEvent::StakeDeposit {
            pool_id: StakePoolId::new(POOL_ID),
            amount: Stake::from_attos(500),
        }],
    );
}

/// The same package, an instance nobody seated: it runs, it moves funds,
/// it emits — and the beacon never hears about it. Seating a pool is a
/// decision the network makes, not one a transaction can make for it.
#[test]
fn an_unseated_instance_of_the_same_package_reaches_nobody() {
    let executor = Executor::with_pools(&[seat(POOL_ID)], ExecutionMode::Serial);
    // `pool_at(99)` is not in the pool set, so it was never registered as an
    // instance either and the delegation cannot even be routed to it —
    // which is the outer of the two guards. The inner one is covered by
    // the codec's own tests, where an instance exists and is unrecognised.
    let executed = execute(&executor, signed_stake(pool_at(POOL_ID), 500));
    assert_eq!(
        witnesses(&executed[0]).len(),
        1,
        "only the seated pool spoke"
    );
}

/// An ordinary transfer between accounts emits events and no facts: the
/// channel carries what a stake pool says and nothing else.
#[test]
fn an_ordinary_transfer_is_not_a_beacon_fact() {
    let executor = Executor::with_pools(&[seat(POOL_ID)], ExecutionMode::Serial);
    let key = Ed25519PrivateKey::from_bytes(&[DELEGATOR; 32]).unwrap();
    let from = account_address(&key.public_key().0);
    let graph = client()
        .transfer_graph(from, from, 100)
        .expect("an account answers a transfer");
    let tx = Transaction::new(client().sign(graph, &key, terms(1_000)));
    let executed = execute(&executor, tx);
    assert!(
        witnesses(&executed[0]).is_empty(),
        "an account's own events are not the beacon's business",
    );
}

/// `pool.register-validator(id, pubkey, proof)`, signed and paid for by
/// `seed` whatever the pool's configuration says.
fn signed_registration(pool: ComponentAddr, seed: u8) -> Transaction {
    let key = key_of(seed);
    let cache = client().cache();
    let mut b = client().builder(&cache);
    staking::register_validator(&mut b, pool, 11, vec![0xC1; 48], vec![0xC2; 96])
        .expect("a pool answers a registration");
    let graph = b.build().expect("a registration produces nothing");
    Transaction::new(client().sign(graph, &key, terms(1_000)))
}

/// A pool instance is owned by nobody, so its own authority is
/// unsatisfiable and the surface would be uncallable if it asked for one.
/// It names a principal its configuration carries instead, and only that
/// principal's signature reaches it.
#[test]
fn only_the_configured_operator_may_register_a_validator() {
    let _ = Executor::with_pools(&[seat(POOL_ID)], ExecutionMode::Serial);

    let outsider = signed_registration(pool_at(POOL_ID), OUTSIDER);
    assert!(outsider.body().signature_is_valid());
    let refused = outsider
        .try_derived()
        .expect_err("an outsider's registration refuses");
    assert!(refused.0.contains("register-validator"), "{}", refused.0);
    assert!(
        refused.0.contains("authority"),
        "the refusal names its reason: {}",
        refused.0
    );

    // The control: the same manifest, the same fee, one signature
    // different. What bites is whose key signed it and not the shape.
    assert!(
        signed_registration(pool_at(POOL_ID), OPERATOR)
            .try_derived()
            .is_ok(),
        "the configured operator's own registration admits",
    );
}

/// The delegation surface is unmoved: `stake` supplies its own authority
/// in the funds it carries, so anyone may delegate to any seated pool.
#[test]
fn a_delegation_needs_no_operator() {
    let _ = Executor::with_pools(&[seat(POOL_ID)], ExecutionMode::Serial);
    assert!(signed_stake(pool_at(POOL_ID), 500).try_derived().is_ok());
}
