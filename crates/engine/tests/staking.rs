//! The beacon's control plane on the engine: a delegation to a seated
//! stake pool arrives in the executing shard's `beacon_witness_events`.
//!
//! Its own binary rather than a case in `transfer`, because the VM statics
//! install once per process and first-installed-wins: a world with a stake
//! pool in it is a different world, and sharing a process would make
//! whichever executor was built first decide what the other one can see.

use std::collections::BTreeMap;
use std::sync::Arc;

use hyperscale_effects_bridge::{ProtocolHasher, account_address, encode_tree};
use hyperscale_engine::genesis::{pool_address, stake_unit, staking_artifact};
use hyperscale_engine::{
    ExecutedTx, ExecutionMode, Executor, Parallelism, TickBatchContext, XRD, genesis_writes,
};
use hyperscale_storage::SubstateDatabase;
use hyperscale_types::{
    BeaconWitnessEvent, BlockHash, CallTarget, ComponentAddr, ConsensusReceipt, Ed25519PrivateKey,
    EnvelopeExt, Hash, NetworkId, PrincipalAddr, ProvisionalHolds, ResourceRef, RevealChain,
    ShardId, ShardTrie, Stake, StakePoolId, StakePoolSeat, SubstateKey, Transaction,
    TransactionBody, TransactionEnvelope, Verified, WeightedTimestamp,
};
use hyperscale_vm_effects::{
    Address, Constraint, EdgeRef, EnvelopeTree, GraphArg, GraphNode, IntentDecl, ManifestGraph,
    Value, package_hash,
};

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

fn withdraw(
    target: impl Into<CallTarget>,
    resource: impl Into<Address>,
    amount: u128,
) -> GraphNode {
    GraphNode {
        target: target.into(),
        method: "withdraw".into(),
        args: vec![
            GraphArg::Literal(Value::Address(resource.into())),
            GraphArg::Literal(Value::U128(amount)),
        ],
    }
}

fn from_edge(producer: u32, resource: impl Into<ResourceRef>) -> GraphArg {
    GraphArg::Edge {
        edge: EdgeRef {
            producer,
            output: 0,
        },
        constraints: vec![Constraint::ResourceIs(resource.into())],
    }
}

/// `delegator.withdraw(*XRD) -> pool.stake -> delegator.deposit(units)`.
fn signed_stake(pool: ComponentAddr, amount: u128) -> Transaction {
    let key = Ed25519PrivateKey::from_bytes(&[DELEGATOR; 32]).unwrap();
    let from = account_address(&key.public_key().0);
    let graph = ManifestGraph {
        nodes: vec![
            withdraw(from, *XRD, amount),
            GraphNode {
                target: pool.into(),
                method: "stake".into(),
                args: vec![from_edge(0, *XRD)],
            },
            GraphNode {
                target: from.into(),
                method: "deposit".into(),
                args: vec![from_edge(1, stake_unit(pool))],
            },
        ],
    };
    let tree = EnvelopeTree {
        root: IntentDecl {
            graph,
            params: Vec::new(),
        },
        root_bindings: Vec::new(),
        subintents: Vec::new(),
    };
    Transaction::new(
        TransactionEnvelope {
            body: TransactionBody::Call(encode_tree(&tree)),
            subintent_sigs: Vec::new(),
            fee_payer: from,
            max_fee: 1_000,
            gas_limit: 1_000_000,
            validity_start_ms: 0,
            validity_end_ms: u64::MAX,
            message: Vec::new(),
            network: NetworkId(242),
            signer: [0; 32],
            signature: [0; 64],
        }
        .sign(&key),
    )
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
    let graph = ManifestGraph {
        nodes: vec![
            withdraw(from, *XRD, 100),
            GraphNode {
                target: from.into(),
                method: "deposit".into(),
                args: vec![from_edge(0, *XRD)],
            },
        ],
    };
    let tree = EnvelopeTree {
        root: IntentDecl {
            graph,
            params: Vec::new(),
        },
        root_bindings: Vec::new(),
        subintents: Vec::new(),
    };
    let tx = Transaction::new(
        TransactionEnvelope {
            body: TransactionBody::Call(encode_tree(&tree)),
            subintent_sigs: Vec::new(),
            fee_payer: from,
            max_fee: 1_000,
            gas_limit: 1_000_000,
            validity_start_ms: 0,
            validity_end_ms: u64::MAX,
            message: Vec::new(),
            network: NetworkId(242),
            signer: [0; 32],
            signature: [0; 64],
        }
        .sign(&key),
    );
    let executed = execute(&executor, tx);
    assert!(
        witnesses(&executed[0]).is_empty(),
        "an account's own events are not the beacon's business",
    );
}

/// `pool.register-validator(id, pubkey, proof)`, signed and paid for by
/// `seed` whatever the pool's configuration says.
fn signed_registration(pool: impl Into<CallTarget>, seed: u8) -> Transaction {
    let key = key_of(seed);
    let tree = EnvelopeTree {
        root: IntentDecl {
            graph: ManifestGraph {
                nodes: vec![GraphNode {
                    target: pool.into(),
                    method: "register-validator".into(),
                    args: vec![
                        GraphArg::Literal(Value::U64(11)),
                        GraphArg::Literal(Value::Bytes(vec![0xC1; 48])),
                        GraphArg::Literal(Value::Bytes(vec![0xC2; 96])),
                    ],
                }],
            },
            params: Vec::new(),
        },
        root_bindings: Vec::new(),
        subintents: Vec::new(),
    };
    Transaction::new(
        TransactionEnvelope {
            body: TransactionBody::Call(encode_tree(&tree)),
            subintent_sigs: Vec::new(),
            fee_payer: account_of(seed),
            max_fee: 1_000,
            gas_limit: 1_000_000,
            validity_start_ms: 0,
            validity_end_ms: u64::MAX,
            message: Vec::new(),
            network: NetworkId(242),
            signer: [0; 32],
            signature: [0; 64],
        }
        .sign(&key),
    )
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
