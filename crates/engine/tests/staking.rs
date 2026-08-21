//! The beacon's control plane on the engine: a delegation to a seated
//! stake pool arrives in the executing shard's `beacon_witness_events`.
//!
//! Every case here runs against a world with a stake pool seated in it,
//! which is what makes the delegation's events beacon facts.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use hyperscale_effects_bridge::genesis::genesis_world_with_pools;
use hyperscale_effects_bridge::vm_statics::config_key;
use hyperscale_effects_bridge::{ProtocolHasher, account_address};
use hyperscale_engine::genesis::{
    GenesisPackages, OWNER_BADGE_ID, pool_address, pool_meta, pool_owner_badge, staking_artifact,
};
use hyperscale_engine::{
    ExecutedTx, ExecutionMode, Executor, TickBatchContext, XRD, genesis_writes,
};
use hyperscale_storage::Substates;
use hyperscale_transactions::{Client, Terms};
use hyperscale_types::{
    BeaconWitnessEvent, ComponentAddr, ConsensusReceipt, Ed25519PrivateKey, EntryKey, EnvelopeExt,
    NetworkId, PrincipalAddr, ProvisionalHolds, RevealChain, ShardId, ShardTrie, Stake,
    StakePoolId, StakePoolSeat, SubstateKey, TimestampRange, Transaction, Verified,
    WeightedTimestamp, absorb_committed_cells,
};
use hyperscale_vm_effects::{
    Composed, holdings_collection, instance_data_key, package_hash, resource_record_key,
};
use hyperscale_vm_manifest_builder::{EnvelopeBuilder, TypedError};
use hyperscale_vm_stdlib::{account, staking};
use hyperscale_vm_types::{Address, CollectionId};

/// The identifier the beacon folds the seated pool under.
const POOL_ID: u32 = 7;
/// The delegator's signing seed.
const DELEGATOR: u8 = 7;
/// The signing seed of the principal the pool's operator surface admits.
const OPERATOR: u8 = 8;
/// The signing seed of a funded account that operates nothing.
const OUTSIDER: u8 = 9;

/// A snapshot over the flattened genesis updates.
struct MapDb {
    cells: BTreeMap<SubstateKey, Vec<u8>>,
    entries: BTreeMap<EntryKey, Vec<u8>>,
}

impl MapDb {
    fn genesis(accounts: &[(PrincipalAddr, u128)], pools: &[StakePoolSeat]) -> Self {
        let (cells, entries) =
            genesis_writes(accounts, pools, &GenesisPackages::protocol()).into_parts();
        Self {
            cells: cells
                .into_iter()
                .map(|(key, change)| (key, change.expect("genesis writes are Set-only")))
                .collect(),
            entries: entries
                .into_iter()
                .map(|(key, change)| (key, change.expect("genesis writes are Set-only")))
                .collect(),
        }
    }
}

impl Substates for MapDb {
    fn cell(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.cells.get(&key).cloned()
    }

    fn entries_in_range(
        &self,
        owner: Address,
        collection: CollectionId,
        lo: u128,
        hi: u128,
        limit: usize,
    ) -> Vec<(u128, Vec<u8>)> {
        if lo > hi {
            return Vec::new();
        }
        let lo_key = EntryKey {
            owner,
            collection,
            order: lo,
        };
        let hi_key = EntryKey {
            owner,
            collection,
            order: hi,
        };
        self.entries
            .range(lo_key..=hi_key)
            .take(limit)
            .map(|(key, value)| (key.order, value.clone()))
            .collect()
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

/// A pool seat; genesis deposits the pool's owner badge into the named
/// account, and presenting that badge is what operates the pool.
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
            genesis_world_with_pools(&[seat(POOL_ID), seat(99)], &GenesisPackages::protocol()),
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
    let chain = client().records();
    let mut b = client().builder(&chain);
    let sender = account::authorize(&mut b, from).expect("an account signs in");
    let funds = account::withdraw(&mut b, sender, *XRD, amount).expect("an account withdraws");
    let units = staking::Staking::at(pool)
        .stake(&mut b, funds)
        .expect("a pool takes a delegation");
    account::deposit(&mut b, from, units).expect("an account banks its position");
    let graph = b.build().expect("every output is consumed");
    Transaction::new(client().sign(graph, &key, terms(1_000)))
}

fn execute(executor: &Executor, tx: Transaction) -> Vec<ExecutedTx> {
    let store = MapDb::genesis(
        &[
            (delegator(), 10_000),
            (account_of(OPERATOR), 10_000),
            (account_of(OUTSIDER), 10_000),
        ],
        &[seat(POOL_ID), seat(99)],
    );
    let trie = ShardTrie::single();
    let ctx = TickBatchContext {
        local_shard: ShardId::ROOT,
        shard_trie: &trie,
        tick_ts: WeightedTimestamp::from_millis(1_000),
        tick_reveal: RevealChain::ZERO,
        holds: &ProvisionalHolds::new(),
    };
    tx.try_derived(executor.derivation().as_ref())
        .expect("a fixture transaction derives");
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
/// An executor over a network seating `pools`, on the protocol's own
/// packages — the staking surface reaches no fixture.
fn seated(pools: &[StakePoolSeat], mode: ExecutionMode) -> Executor {
    Executor::with_genesis(pools, &GenesisPackages::protocol(), mode)
}

/// Apply what a settled transaction wrote, so the next one reads it as
/// committed state.
fn absorb(store: &mut MapDb, executed: &ExecutedTx, executor: &Executor) {
    let ConsensusReceipt::Succeeded { writes, .. } = &executed.consensus else {
        panic!("the transaction must settle: {:?}", executed.consensus);
    };
    // What the commit path does with a block's receipts: every committed
    // cell is offered to the caches, so a seal that settled is a
    // component the next transaction can name with nothing presented.
    absorb_committed_cells([&executed.consensus], executor.derivation().as_ref());
    for (key, change) in &writes.cells {
        match change {
            Some(value) => store.cells.insert(*key, value.clone()),
            None => store.cells.remove(key),
        };
    }
    for (key, change) in &writes.entries {
        match change {
            Some(value) => store.entries.insert(*key, value.clone()),
            None => store.entries.remove(key),
        };
    }
}

/// One envelope that makes an unseated pool actual: it presents the
/// pool's instance record and seals it, which is the one call a
/// derivable component answers.
fn signed_instantiate(seed: u8, seat: &StakePoolSeat) -> Transaction {
    let key = key_of(seed);
    let chain = client().records();
    let meta = pool_meta(package_hash(&ProtocolHasher, staking_artifact()), seat);
    // The composer types the call against a record the chain does not
    // answer for yet: the seal is what makes it answer.
    let composed = Composed::new(&chain, std::slice::from_ref(&meta), &ProtocolHasher);
    let pool = meta.address(&ProtocolHasher);
    let (mut env, mut root) = EnvelopeBuilder::new(&composed, &ProtocolHasher);
    staking::Staking::at(pool)
        .instantiate(&mut root)
        .expect("a derivable pool answers its seal");
    env.instance(meta);
    env.seal(root)
        .expect("the root declares nothing to discharge");
    let tree = env.build().expect("the intent declares no hole");
    Transaction::new(client().sign_tree(&tree, Vec::new(), &key, terms(1_000)))
}

/// One envelope that founds an instantiated pool: it founds it as the
/// configured founder and files the badge in that account.
///
/// Carries no record. The pool's seal has committed by the time this
/// runs, so the chain answers for the target and a record beside the
/// call would be a second source for one fact.
fn signed_found(seed: u8, seat: &StakePoolSeat) -> Transaction {
    let key = key_of(seed);
    let from = account_address(&key.public_key().0);
    let _chain = client().records();
    // The composer knows the record and types the call against it; the
    // envelope carries nothing, because the chain answers for the
    // target itself once the seal has committed.
    let meta = pool_meta(package_hash(&ProtocolHasher, staking_artifact()), seat);
    let chain = client().records();
    let composed = Composed::new(&chain, std::slice::from_ref(&meta), &ProtocolHasher);
    let pool = meta.address(&ProtocolHasher);
    let (mut env, mut root) = EnvelopeBuilder::new(&composed, &ProtocolHasher);
    let founder = account::authorize(&mut root, from).expect("an account signs in");
    let badge = staking::Staking::at(pool)
        .found(&mut root, founder)
        .expect("a pool answers a founding");
    account::deposit_nf(&mut root, from, badge).expect("an account banks the badge");
    env.seal(root)
        .expect("the root declares nothing to discharge");
    let tree = env.build().expect("the intent declares no hole");
    Transaction::new(client().sign_tree(&tree, Vec::new(), &key, terms(1_000)))
}

/// A pool nobody seated instantiates and founds itself, and the cells it
/// ends holding are the cells genesis writes for a seated one, byte for
/// byte — two writers of one object, held to each other. Two
/// transactions, because a presence condition is judged against
/// committed state: the seal has to be there before the founding's
/// fence can see it.
#[test]
fn a_founded_pool_holds_the_cells_genesis_writes_for_a_seated_one() {
    let unseated = seat(55);
    let executor = seated(&[], ExecutionMode::Serial);
    let mut store = MapDb::genesis(&[(account_of(OPERATOR), 10_000)], &[]);
    let trie = ShardTrie::single();
    let ctx = TickBatchContext {
        local_shard: ShardId::ROOT,
        shard_trie: &trie,
        tick_ts: WeightedTimestamp::from_millis(1_000),
        tick_reveal: RevealChain::ZERO,
        holds: &ProvisionalHolds::new(),
    };
    // Built one at a time rather than up front: the founding names a
    // target the chain answers for, and it only does once the seal
    // ahead of it has committed.
    for build in [signed_instantiate, signed_found] {
        let raw = build(OPERATOR, &unseated);
        raw.try_derived(executor.derivation().as_ref())
            .expect("a fixture transaction derives");
        let tx = Arc::new(Verified::<Transaction>::from_persisted(raw));
        let executed = executor.execute_batch(&ctx, &store, std::slice::from_ref(&tx));
        absorb(&mut store, &executed[0], &executor);
    }

    let (genesis_cells, genesis_entries) = genesis_writes(
        &[],
        std::slice::from_ref(&unseated),
        &GenesisPackages::protocol(),
    )
    .into_parts();
    let pool = pool_address(package_hash(&ProtocolHasher, staking_artifact()), &unseated);
    let badge = pool_owner_badge(pool);
    for key in [
        config_key(pool),
        resource_record_key(&ProtocolHasher, pool, badge),
        instance_data_key(&ProtocolHasher, pool, badge, OWNER_BADGE_ID),
    ] {
        assert!(genesis_cells.contains_key(&key), "genesis writes the cell");
        assert_eq!(store.cells.get(&key), genesis_cells[&key].as_ref());
    }
    let entry = EntryKey {
        owner: unseated.operator.address(),
        collection: holdings_collection(&ProtocolHasher, unseated.operator, badge),
        order: u128::from(OWNER_BADGE_ID),
    };
    assert!(
        genesis_entries.contains_key(&entry),
        "genesis writes the entry"
    );
    assert_eq!(store.entries.get(&entry), genesis_entries[&entry].as_ref());
}

/// A pool nobody instantiated cannot be called at all.
///
/// Its address derives — anybody can compute it — but the chain answers
/// for no such component, and a caller may not supply the answer: a
/// record stands for the seal that makes a component actual and for no
/// other call. So the composition fails where it is cheapest to fail,
/// with nothing signed and nothing priced.
#[test]
fn a_pool_nobody_instantiated_answers_nothing() {
    let unseated = seat(56);
    let pool = pool_address(package_hash(&ProtocolHasher, staking_artifact()), &unseated);
    let chain = client().records();
    let (_, mut root) = EnvelopeBuilder::new(&chain, &ProtocolHasher);
    let founder = account::authorize(&mut root, account_of(OPERATOR)).expect("an account signs in");
    let refusal = staking::Staking::at(pool)
        .found(&mut root, founder)
        .expect_err("a pool nobody sealed resolves nothing");
    assert!(
        matches!(refusal, TypedError::UnknownInstance(address) if address == pool.address()),
        "refused as an address the chain answers nothing for: {refusal:?}"
    );
}

#[test]
fn a_delegation_to_a_seated_pool_reaches_the_witness_channel() {
    let executor = Executor::with_genesis(
        &[seat(POOL_ID), seat(99)],
        &GenesisPackages::protocol(),
        ExecutionMode::Serial,
    );
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
    let executor = seated(&[seat(POOL_ID)], ExecutionMode::Serial);
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
    let executor = seated(&[seat(POOL_ID)], ExecutionMode::Serial);
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
/// `seed`, presenting the pool's owner badge from their own account —
/// whether or not they hold it.
fn signed_registration(pool: ComponentAddr, seed: u8) -> Transaction {
    let key = key_of(seed);
    let chain = client().records();
    let mut b = client().builder(&chain);
    let proof = account::present_instance(
        &mut b,
        account_address(&key.public_key().0),
        pool_owner_badge(pool),
        OWNER_BADGE_ID,
    )
    .expect("a presentation types");
    staking::Staking::at(pool)
        .register_validator(&mut b, proof, 11, vec![0xC1; 48], vec![0xC2; 96])
        .expect("a pool answers a registration");
    let graph = b.build().expect("a registration produces nothing");
    Transaction::new(client().sign(graph, &key, terms(1_000)))
}

/// A pool instance is owned by nobody, so its own authority is
/// unsatisfiable and the surface would be uncallable if it asked for
/// one. It admits whoever presents the pool's owner badge instead, and
/// genesis seats that badge in the seat's operator account.
#[test]
fn only_the_badge_holder_may_register_a_validator() {
    let executor = seated(&[seat(POOL_ID)], ExecutionMode::Serial);

    // Well-formed: the outsider presents a badge from their own
    // account, which is what admission asks of a custodial call. The
    // badge is what they do not hold, and the gate says so when the
    // call reaches it.
    let outsider = signed_registration(pool_at(POOL_ID), OUTSIDER);
    assert!(outsider.body().signature_is_valid());
    assert!(
        outsider.try_derived(executor.derivation().as_ref()).is_ok(),
        "the shape is well-formed"
    );
    let executed = execute(&executor, outsider);
    assert!(
        matches!(&executed[0].consensus, ConsensusReceipt::Failed),
        "an outsider's registration must not settle: {:?}",
        executed[0].consensus
    );

    // The control: the same manifest, the same fee, one signature
    // different. What bites is whose key signed it and not the shape.
    let executed = execute(&executor, signed_registration(pool_at(POOL_ID), OPERATOR));
    assert!(
        matches!(&executed[0].consensus, ConsensusReceipt::Succeeded { .. }),
        "the badge holder's own registration settles: {:?}",
        executed[0].consensus
    );
}

/// The delegation surface is unmoved: `stake` supplies its own authority
/// in the funds it carries, so anyone may delegate to any seated pool.
#[test]
fn a_delegation_needs_no_operator() {
    let executor = seated(&[seat(POOL_ID)], ExecutionMode::Serial);
    assert!(
        signed_stake(pool_at(POOL_ID), 500)
            .try_derived(executor.derivation().as_ref())
            .is_ok()
    );
}
