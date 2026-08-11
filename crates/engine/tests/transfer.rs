//! The VM engine end to end at the seam: signed transfer graphs through
//! derivation, the batch executor, and the movement fold, against a
//! genesis-seeded snapshot.

use std::collections::BTreeMap;
use std::sync::Arc;

use hyperscale_effects_bridge::vm_statics::package_key;
use hyperscale_effects_bridge::{
    ProtocolHasher, account_address, admit_package, attach_metadata, encode_tree,
};
use hyperscale_engine::genesis::{account_artifact, entropy_key, vault_key};
use hyperscale_engine::{
    ExecutedTx, ExecutionMode, Executor, Parallelism, PreviewGrants, PreviewInputs, PreviewOutcome,
    PreviewReport, ResourceChange, TickBatchContext, XRD, genesis_writes,
};
use hyperscale_storage::{SubstateDatabase, SubstateStore, TickChain, TickOutput, VersionedStore};
use hyperscale_types::test_utils::test_prefix;
use hyperscale_types::{
    BlockHash, BlockHeight, ConsensusReceipt, Ed25519PrivateKey, EnvelopeExt, Hash,
    MerkleInclusionProof, NetworkId, ProvisionalHolds, RevealChain, SettledWrites, ShardId,
    ShardTrie, StateRoot, StateWrites, SubstateKey, Transaction, TransactionBody,
    TransactionEnvelope, Verified, WeightedTimestamp, absorb_committed_cells,
};
use hyperscale_vm_effects::{
    AbiParam, Address, Constraint, EdgeRef, EnvelopeTree, Expr, GraphArg, GraphNode, IntentDecl,
    ManifestGraph, Value, package_hash,
};
use hyperscale_vm_kernel::{amount_cell, encode_amount};
use hyperscale_vm_stdlib::{ACCOUNT_COMPONENT, account_metadata};

/// The two accounts the transfer cases move funds between, as signing
/// seeds rather than as literal addresses: a withdrawing node admits only
/// the signature its target's address derives from, so an account that
/// spends has to be one a key here derives.
const ALICE_SEED: u8 = 41;
const BOB_SEED: u8 = 42;

/// The ceiling the plain transfer cases name, and — being under what a
/// transfer costs — the fee they are charged exactly.
///
/// Small enough to stay legible in the balance assertions, which matters
/// now that a withdrawal's own account is also the one paying: the payer
/// is the signer, and the signer is whoever the withdrawing node names.
const TRANSFER_FEE: u128 = 100;

fn alice() -> Address {
    fee_payer(ALICE_SEED)
}

fn bob() -> Address {
    fee_payer(BOB_SEED)
}

/// A snapshot over the flattened genesis updates.
struct MapDb(BTreeMap<SubstateKey, Vec<u8>>);

impl MapDb {
    fn genesis(accounts: &[(Address, u128)]) -> Self {
        let writes = genesis_writes(accounts, &[]);
        let mut map = BTreeMap::new();
        for (key, change) in writes.cells() {
            let value = change.clone().expect("genesis writes are Set-only");
            map.insert(*key, value);
        }
        Self(map)
    }
}

impl MapDb {
    /// Apply a receipt's committed writes, as the commit path would —
    /// resolving what it moved against what this map holds, which is
    /// what settlement does.
    fn apply(&mut self, writes: &StateWrites) {
        let writes = writes.resolve(&mut |key| self.0.get(&key).cloned());
        for (key, change) in writes.cells() {
            match change {
                Some(value) => {
                    self.0.insert(*key, value.clone());
                }
                None => {
                    self.0.remove(key);
                }
            }
        }
    }
}

impl SubstateDatabase for MapDb {
    fn substate(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.0.get(&key).cloned()
    }
}

/// The store surface a [`TickChain`] needs. The map is the whole world
/// at one version, so every anchor reads the same.
impl SubstateStore for MapDb {
    type Snapshot<'a> = Self;
    fn snapshot(&self) -> Self::Snapshot<'_> {
        Self(self.0.clone())
    }
    fn jmt_height(&self) -> BlockHeight {
        BlockHeight::GENESIS
    }
    fn state_root(&self) -> StateRoot {
        StateRoot::ZERO
    }
    fn get_substate_at_height(
        &self,
        _key: SubstateKey,
        _block_height: BlockHeight,
    ) -> Option<Option<Vec<u8>>> {
        None
    }
    fn generate_merkle_proofs(
        &self,
        _keys: &[SubstateKey],
        _block_height: BlockHeight,
    ) -> Option<MerkleInclusionProof> {
        None
    }
}

impl VersionedStore for MapDb {
    fn snapshot_at(&self, _height: BlockHeight) -> Self::Snapshot<'_> {
        Self(self.0.clone())
    }
    fn substate_bytes_at(&self, _height: BlockHeight) -> Option<u64> {
        None
    }
}

fn transfer_graph(from: Address, to: Address, amount: u128) -> ManifestGraph {
    ManifestGraph {
        nodes: vec![
            GraphNode {
                target: from,
                method: "withdraw".into(),
                args: vec![
                    GraphArg::Literal(Value::Address(*XRD)),
                    GraphArg::Literal(Value::U128(amount)),
                ],
            },
            GraphNode {
                target: to,
                method: "deposit".into(),
                args: vec![GraphArg::Edge {
                    edge: EdgeRef {
                        producer: 0,
                        output: 0,
                    },
                    constraints: vec![Constraint::ResourceIs(*XRD)],
                }],
            },
        ],
    }
}

fn signed_transfer(seed: u8, from: Address, to: Address, amount: u128) -> Transaction {
    signed_transfer_with_fee(seed, from, to, amount, TRANSFER_FEE)
}

/// A transfer whose recipient signs a floor the withdrawal cannot meet.
fn signed_transfer_under_bound(
    seed: u8,
    from: Address,
    to: Address,
    amount: u128,
    min: u128,
    max_fee: u128,
) -> Transaction {
    let key = Ed25519PrivateKey::from_bytes(&[seed; 32]).unwrap();
    let mut graph = transfer_graph(from, to, amount);
    let GraphArg::Edge { constraints, .. } = &mut graph.nodes[1].args[0] else {
        panic!("the deposit consumes an edge");
    };
    constraints.push(Constraint::MinAmount(min));
    let tree = EnvelopeTree {
        root: IntentDecl {
            graph,
            params: Vec::new(),
        },
        root_bindings: Vec::new(),
        subintents: Vec::new(),
    };
    let vm = TransactionEnvelope {
        body: TransactionBody::Call(encode_tree(&tree)),
        subintent_sigs: Vec::new(),
        fee_payer: account_address(&key.public_key().0),
        max_fee,
        gas_limit: 1_000_000,
        validity_start_ms: 0,
        validity_end_ms: u64::MAX,
        message: Vec::new(),
        network: NetworkId(242),
        signer: [0; 32],
        signature: [0; 64],
    }
    .sign(&key);
    Transaction::new(vm)
}

fn signed_transfer_with_fee(
    seed: u8,
    from: Address,
    to: Address,
    amount: u128,
    max_fee: u128,
) -> Transaction {
    let key = Ed25519PrivateKey::from_bytes(&[seed; 32]).unwrap();
    let tree = EnvelopeTree {
        root: IntentDecl {
            graph: transfer_graph(from, to, amount),
            params: Vec::new(),
        },
        root_bindings: Vec::new(),
        subintents: Vec::new(),
    };
    let vm = TransactionEnvelope {
        body: TransactionBody::Call(encode_tree(&tree)),
        subintent_sigs: Vec::new(),
        fee_payer: account_address(&key.public_key().0),
        max_fee,
        gas_limit: 1_000_000,
        validity_start_ms: 0,
        validity_end_ms: u64::MAX,
        message: Vec::new(),
        network: NetworkId(242),
        signer: [0; 32],
        signature: [0; 64],
    }
    .sign(&key);
    Transaction::new(vm)
}

/// The account address the fee-paying tests derive from their signing key.
fn fee_payer(seed: u8) -> Address {
    let key = Ed25519PrivateKey::from_bytes(&[seed; 32]).unwrap();
    account_address(&key.public_key().0)
}

/// Every account any test in this binary transacts with.
///
/// The VM statics are process-global and first-installed-wins, so every
/// executor here has to be built over one world: sharing a process, the
/// first `Executor::new` fixes the instance registry for every test that
/// follows, and an address missing from it fails admission with `no
/// instance` rather than anything to do with the test's own subject. Per-test
/// balances are unaffected — those come from the snapshot `execute_on`
/// builds, which is separate from the world.
fn world_accounts() -> Vec<(Address, u128)> {
    vec![
        (alice(), 1_000),
        (bob(), 50),
        (fee_payer(7), 1_000),
        (fee_payer(11), 110),
        (fee_payer(23), 1_000),
        (fee_payer(31), 1_000),
        (fee_payer(32), 1_000),
    ]
}

fn execute(executor: &Executor, transactions: &[Arc<Verified<Transaction>>]) -> Vec<ExecutedTx> {
    execute_on(&[(alice(), 1_000), (bob(), 50)], executor, transactions)
}

/// A signed single-node stamp: the account records the transaction's
/// randomness draw in its entropy leaf.
fn signed_stamp(seed: u8, owner: Address) -> Transaction {
    signed_stamp_with_fee(seed, owner, 1_000_000)
}

fn signed_stamp_with_fee(seed: u8, owner: Address, max_fee: u128) -> Transaction {
    let key = Ed25519PrivateKey::from_bytes(&[seed; 32]).unwrap();
    let tree = EnvelopeTree {
        root: IntentDecl {
            graph: ManifestGraph {
                nodes: vec![GraphNode {
                    target: owner,
                    method: "stamp-entropy".into(),
                    args: vec![],
                }],
            },
            params: Vec::new(),
        },
        root_bindings: Vec::new(),
        subintents: Vec::new(),
    };
    let vm = TransactionEnvelope {
        body: TransactionBody::Call(encode_tree(&tree)),
        subintent_sigs: Vec::new(),
        fee_payer: account_address(&key.public_key().0),
        max_fee,
        gas_limit: 1_000_000,
        validity_start_ms: 0,
        validity_end_ms: u64::MAX,
        message: Vec::new(),
        network: NetworkId(242),
        signer: [0; 32],
        signature: [0; 64],
    }
    .sign(&key);
    Transaction::new(vm)
}

/// Execute `transactions` as a single-shard batch anchored on `reveal`.
fn execute_anchored(
    executor: &Executor,
    reveal: RevealChain,
    transactions: &[Arc<Verified<Transaction>>],
) -> Vec<ExecutedTx> {
    let snapshot_store = MapDb::genesis(&[(alice(), 1_000), (bob(), 50)]);
    let trie = ShardTrie::single();
    let ctx = TickBatchContext {
        par: Parallelism::Sequential,
        local_shard: ShardId::ROOT,
        shard_trie: &trie,
        block_hash: BlockHash::from_raw(Hash::from_bytes(b"block")),
        tick_ts: WeightedTimestamp::from_millis(1_000),
        tick_reveal: reveal,
        holds: &ProvisionalHolds::new(),
    };
    executor.execute_batch(&ctx, &snapshot_store, transactions)
}

/// The entropy leaf a stamp wrote, if any.
fn entropy_cell(executed: &ExecutedTx, owner: Address) -> Option<Vec<u8>> {
    let writes = executed.consensus.writes()?;
    writes.cells.get(&entropy_key(owner)).cloned().flatten()
}

/// The stamp writes a draw fixed by the anchor: the same anchor gives the
/// same 32 bytes, a different anchor gives different ones — which is what
/// makes the payer block, and not the executing block, decide a
/// randomness-reading guest's receipt.
#[test]
fn a_stamp_writes_the_draw_its_anchor_fixes() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let tx = Arc::new(Verified::<Transaction>::from_persisted(signed_stamp(
        ALICE_SEED,
        alice(),
    )));
    let anchor = RevealChain::from_raw(Hash::from_bytes(b"payer block"));

    let executed = execute_anchored(&executor, anchor, std::slice::from_ref(&tx));
    let stamped = entropy_cell(&executed[0], alice()).expect("the stamp wrote the entropy leaf");
    assert_eq!(stamped.len(), 32);

    let again = execute_anchored(&executor, anchor, std::slice::from_ref(&tx));
    assert_eq!(
        entropy_cell(&again[0], alice()),
        Some(stamped.clone()),
        "one anchor, one draw"
    );

    let elsewhere = execute_anchored(
        &executor,
        RevealChain::from_raw(Hash::from_bytes(b"another block")),
        std::slice::from_ref(&tx),
    );
    assert_ne!(
        entropy_cell(&elsewhere[0], alice()),
        Some(stamped),
        "a different anchor is a different draw"
    );
}

/// Two independent payments into one account, each in its own batch.
///
/// The recipient's vault is a `delta` on both sides, which the mode
/// lattice calls compatible — so nothing defers the second behind the
/// first, and the two may legitimately be included in different blocks.
/// What each receipt then carries is an *absolute* value for the cell,
/// derived from whatever baseline its batch read.
///
/// Threaded, that is right: the second batch reads the first's applied
/// credit and writes the sum. Against a shared baseline it is not: both
/// batches read the same starting balance, both write the same absolute,
/// and whichever settles last silently discards the other's credit.
///
/// Both halves assert threading. The first threads by hand, applying each
/// receipt before the next batch reads; the second threads the way the
/// shard does, through the tick chain, where the payments land in
/// consecutive blocks and the earlier one is committed but not yet
/// settled when the later one executes — the case that used to read a
/// shared baseline.
#[test]
fn consecutive_payments_thread_through_the_tick_chain() {
    let payer_a = fee_payer(31);
    let payer_b = fee_payer(32);
    let hot = bob();
    let accounts = [(payer_a, 1_000), (payer_b, 1_000), (hot, 10)];
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);

    let pay = |seed: u8, from: Address| {
        Arc::new(Verified::<Transaction>::from_persisted(
            signed_transfer_with_fee(seed, from, hot, 100, 10),
        ))
    };

    // Threaded by hand: each batch reads what the previous one committed.
    let mut store = MapDb::genesis(&accounts);
    let mut threaded = Vec::new();
    for (seed, from) in [(31u8, payer_a), (32, payer_b)] {
        let executed = execute_batch_on(&store, &executor, &[pay(seed, from)]);
        let updates = executed[0]
            .consensus
            .writes()
            .expect("a completed payment commits updates");
        threaded.push(vault_cell(&settled_on(updates, &store), hot));
        store.apply(updates);
    }
    assert_eq!(
        threaded,
        vec![
            Some(encode_amount(110).to_vec()),
            Some(encode_amount(210).to_vec())
        ],
        "each payment must land on top of the last"
    );

    // Threaded by the tick chain: two blocks inside one unsettled window.
    // Nothing settles between them, so the base never moves; tick 2's
    // baseline is tick 1's output over it.
    let chain = TickChain::new(Arc::new(MapDb::genesis(&accounts)));
    let mut chained = Vec::new();
    for (height, (seed, from)) in [(31u8, payer_a), (32, payer_b)].into_iter().enumerate() {
        let tick = BlockHeight::new(height as u64 + 1);
        let baseline = chain.view_at(BlockHeight::new(height as u64));
        let executed = execute_batch_on(&baseline.snapshot(), &executor, &[pay(seed, from)]);
        let updates = executed[0]
            .consensus
            .writes()
            .expect("a completed payment commits updates");
        // Against this tick's own baseline — the state it lands on.
        chained.push(vault_cell(&settled_on(updates, &baseline.snapshot()), hot));
        chain.append(
            tick,
            TickOutput {
                determined: vec![(executed[0].tx_hash, updates.clone())],
                provisional: Vec::new(),
            },
        );
    }
    assert_eq!(
        chained, threaded,
        "a payment committed while an earlier one is unsettled must still \
         read the earlier one's credit"
    );
}

fn execute_on(
    accounts: &[(Address, u128)],
    executor: &Executor,
    transactions: &[Arc<Verified<Transaction>>],
) -> Vec<ExecutedTx> {
    execute_batch_on(&MapDb::genesis(accounts), executor, transactions)
}

/// Execute one batch against an explicit store, so a caller can thread
/// committed state between batches the way the commit path does.
fn execute_batch_on(
    snapshot_store: &(dyn SubstateDatabase + Sync),
    executor: &Executor,
    transactions: &[Arc<Verified<Transaction>>],
) -> Vec<ExecutedTx> {
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
    executor.execute_batch(&ctx, snapshot_store, transactions)
}

/// A receipt's writes as they settle onto `accounts`.
///
/// A receipt says what it moved, not what the cell ends at, so an
/// assertion about a balance has to name the state the movement lands
/// on. These tests start from `accounts` and settle one batch onto it.
/// A receipt's writes as they settle onto the state they land on.
fn settled_on(writes: &StateWrites, state: &impl SubstateDatabase) -> SettledWrites {
    writes.resolve(&mut |key| state.substate(key))
}

fn settled(writes: &StateWrites, accounts: &[(Address, u128)]) -> SettledWrites {
    eprintln!(
        "SETTLEDBG cells={:?} movements={:?} accounts={:?}",
        writes.cells.keys().collect::<Vec<_>>(),
        writes.movements,
        accounts
            .iter()
            .map(|(o, a)| (vault_key(*o, *XRD), a))
            .collect::<Vec<_>>()
    );
    writes.resolve(&mut |key| {
        accounts
            .iter()
            .find(|(owner, _)| vault_key(*owner, *XRD) == key)
            .and_then(|(_, amount)| amount_cell(*amount).map(|cell| cell.to_vec()))
    })
}

fn vault_cell(writes: &SettledWrites, owner: Address) -> Option<Vec<u8>> {
    writes
        .cells()
        .get(&vault_key(owner, *XRD))
        .cloned()
        .flatten()
}

/// Whether the batch removed the vault cell outright — a drain, never a
/// zero write.
fn vault_removed(writes: &SettledWrites, owner: Address) -> bool {
    writes.cells().get(&vault_key(owner, *XRD)) == Some(&None)
}

#[test]
fn a_transfer_folds_to_identity_keyed_absolute_updates() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let tx = Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
        ALICE_SEED,
        alice(),
        bob(),
        100,
    )));
    let executed = execute(&executor, &[tx]);
    assert_eq!(executed.len(), 1);
    let ConsensusReceipt::Succeeded {
        writes: database_updates,
        receipt_hash,
        ..
    } = &executed[0].consensus
    else {
        panic!("transfer must succeed: {:?}", executed[0].consensus);
    };
    assert_ne!(receipt_hash.as_raw(), &Hash::ZERO);
    // Withdraw settled 100 off the sender and the fee another 100 —
    // the sender signs, so the sender pays. Deposit credited the
    // recipient. Absolute values, identity-keyed.
    assert_eq!(
        vault_cell(&settled(database_updates, &world_accounts()), alice()),
        Some(encode_amount(1_000 - 100 - TRANSFER_FEE).to_vec())
    );
    assert_eq!(
        vault_cell(&settled(database_updates, &world_accounts()), bob()),
        Some(encode_amount(150).to_vec())
    );
}

#[test]
fn an_uncovered_withdrawal_aborts_and_the_batch_carries_on() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let over = Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
        BOB_SEED,
        bob(),
        alice(),
        500,
    )));
    let floor = over.body().abort_floor();
    let fine = Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
        ALICE_SEED,
        alice(),
        bob(),
        25,
    )));
    let executed = execute(&executor, &[Arc::clone(&over), Arc::clone(&fine)]);
    assert_eq!(executed.len(), 2);
    // Input order is preserved: the infeasible reservation aborts its
    // own transaction only.
    assert!(matches!(executed[0].consensus, ConsensusReceipt::Failed));
    assert!(
        matches!(executed[1].consensus, ConsensusReceipt::Succeeded { .. }),
        "the covered transfer must succeed"
    );

    // Commit the batch the way the settlement path does — every settled
    // receipt's writes merged in hash order, later receipts winning per
    // cell. Whichever side of the failure the transfer's hash lands on,
    // the committed end state carries both its movement and the
    // failure's floor debit.
    let mut store = MapDb::genesis(&[(alice(), 1_000), (bob(), 50)]);
    let mut ordered: Vec<&ExecutedTx> = executed.iter().collect();
    ordered.sort_by_key(|e| e.tx_hash);
    for e in &ordered {
        if let Some(writes) = e.consensus.writes() {
            store.apply(writes);
        }
        if let Some(writes) = e.fee_receipt.as_ref().and_then(ConsensusReceipt::writes) {
            store.apply(writes);
        }
    }
    assert_eq!(
        store.substate(vault_key(alice(), *XRD)),
        Some(encode_amount(1_000 - 25 - TRANSFER_FEE).to_vec())
    );
    assert_eq!(
        store.substate(vault_key(bob(), *XRD)),
        Some(encode_amount(75 - floor).to_vec()),
        "the credit lands and the failure's floor stays charged"
    );
}

/// A charged failure's debit survives a sibling folded after it.
///
/// Each records what it moved on the vault — the failure its floor, the
/// credit its amount — so settling both leaves the vault holding the
/// sum. Neither receipt has to know about the other, which is what a
/// movement buys: an absolute would have had to carry the sibling's
/// debit to avoid reverting it.
#[test]
fn a_failed_charge_survives_a_later_sibling_credit() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let failed = Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
        BOB_SEED,
        bob(),
        alice(),
        500,
    )));
    let floor = failed.body().abort_floor();
    // Receipts fold and commit hash-ascending, and only a credit landing
    // after the failure can revert its debit — so pick a transfer amount
    // whose hash does.
    let (amount, credit) = (100u128..200)
        .map(|amount| {
            let tx = Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
                ALICE_SEED,
                alice(),
                bob(),
                amount,
            )));
            (amount, tx)
        })
        .find(|(_, tx)| tx.hash() > failed.hash())
        .expect("some amount in range hashes after the failure");

    let executed = execute(&executor, &[Arc::clone(&failed), Arc::clone(&credit)]);
    assert!(matches!(executed[0].consensus, ConsensusReceipt::Failed));
    let ConsensusReceipt::Succeeded { writes, .. } = &executed[1].consensus else {
        panic!("the credit must succeed");
    };
    // Settle both, in commit order, and the debit is still there.
    let mut db = MapDb::genesis(&world_accounts());
    let charge = executed[0]
        .fee_receipt
        .as_ref()
        .and_then(|receipt| receipt.writes())
        .expect("a charged failure settles its floor");
    db.apply(charge);
    db.apply(writes);
    assert_eq!(
        db.substate(vault_key(bob(), *XRD)),
        Some(encode_amount(50 + amount - floor).to_vec()),
        "a later sibling's credit must compose with the charged floor, not revert it"
    );
}

#[test]
fn serial_and_parallel_scheduling_produce_identical_receipts() {
    let serial = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let parallel = Executor::new(&world_accounts(), ExecutionMode::Parallel);
    let txs: Vec<Arc<Verified<Transaction>>> = (0..4u128)
        .map(|i| {
            Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
                ALICE_SEED,
                alice(),
                bob(),
                10 + i,
            )))
        })
        .collect();
    let a = execute(&serial, &txs);
    let b = execute(&parallel, &txs);
    assert_eq!(a.len(), b.len());
    for (left, right) in a.iter().zip(&b) {
        assert_eq!(left.tx_hash, right.tx_hash);
        assert_eq!(left.consensus, right.consensus);
    }
}

/// A completed transfer burns its attested actual — fuel, capped at the
/// signed ceiling — from the payer's vault as part of the receipt's own
/// writes, so the burn rides the attested `writes_root` and the
/// sync-replayable work items.
/// An attempt that applies nothing still attests the declaration it made,
/// and a completed one attests strictly more.
///
/// This is the half of attested work that fuel alone misses. A shard that
/// executes a leg it does not own can burn almost no fuel while holding the
/// exclusivity its declaration claimed, so pricing on compute alone would
/// under-pay exactly the participants cross-shard compensation exists for.
/// Here the same shape is visible within one shard: the uncovered
/// withdrawal never applies an effect, and its work is still positive
/// because the declaration was admitted, routed, and locked regardless.
#[test]
fn an_unapplied_attempt_still_attests_its_declaration() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let over = Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
        ALICE_SEED,
        alice(),
        bob(),
        1_000_000,
    )));
    let executed = execute(&executor, &[over]);
    assert_eq!(executed.len(), 1);
    assert!(
        matches!(executed[0].consensus, ConsensusReceipt::Failed),
        "an uncovered withdrawal must not apply"
    );
    let unapplied = executed[0].attested_work;
    assert!(
        unapplied > 0,
        "an attempt that applied nothing still declared, routed, and locked"
    );

    // A completed transfer of the same shape attests the same declaration
    // plus the compute it actually consumed.
    let fine = Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
        ALICE_SEED,
        alice(),
        bob(),
        100,
    )));
    let executed = execute(&executor, &[fine]);
    assert_eq!(executed.len(), 1);
    assert!(
        matches!(executed[0].consensus, ConsensusReceipt::Succeeded { .. }),
        "a covered transfer must apply"
    );
    assert!(
        executed[0].attested_work > unapplied,
        "a completed execution attests its compute on top of its declaration: \
         completed = {}, unapplied = {unapplied}",
        executed[0].attested_work,
    );
}

#[test]
fn a_completed_transfer_burns_the_fee_ceiling_from_its_payer() {
    let payer = fee_payer(7);
    let accounts = [(payer, 1_000), (bob(), 50)];
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    // A transfer's fuel far exceeds the tiny ceiling, so the burn is
    // exactly `max_fee` — the cap working.
    let tx = Arc::new(Verified::<Transaction>::from_persisted(
        signed_transfer_with_fee(7, payer, bob(), 100, 10),
    ));
    let executed = execute_on(&accounts, &executor, &[tx]);
    assert_eq!(executed.len(), 1);
    let ConsensusReceipt::Succeeded {
        writes: database_updates,
        ..
    } = &executed[0].consensus
    else {
        panic!("transfer must succeed: {:?}", executed[0].consensus);
    };
    assert_eq!(
        vault_cell(&settled(database_updates, &accounts), payer),
        Some(encode_amount(1_000 - 100 - 10).to_vec())
    );
    assert_eq!(
        vault_cell(&settled(database_updates, &accounts), bob()),
        Some(encode_amount(150).to_vec())
    );
}

/// A fee-paying call whose manifest never touches the payer's own vault
/// still pays: the batch baseline pre-reads every local payer's vault,
/// so the burn has a cell to debit even when no declared effect loads
/// one. The stamp is the canonical shape — it writes only the entropy
/// leaf.
#[test]
fn a_call_that_never_touches_its_payers_vault_still_pays() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    // A stamp's fuel far exceeds the tiny ceiling, so the burn is
    // exactly `max_fee`.
    let tx = Arc::new(Verified::<Transaction>::from_persisted(
        signed_stamp_with_fee(ALICE_SEED, alice(), 10),
    ));
    let executed = execute(&executor, &[tx]);
    let ConsensusReceipt::Succeeded {
        writes: database_updates,
        ..
    } = &executed[0].consensus
    else {
        panic!("the stamp must succeed: {:?}", executed[0].consensus);
    };
    assert!(
        database_updates.cells.contains_key(&entropy_key(alice())),
        "the stamp wrote its entropy leaf"
    );
    assert_eq!(
        vault_cell(&settled(database_updates, &world_accounts()), alice()),
        Some(encode_amount(1_000 - 10).to_vec()),
        "the payer's vault carries the burn even though the manifest never loaded it"
    );
}

/// A batch with two distinct payers keeps each burn in its own receipt:
/// a receipt is one transaction's effect record, so a sibling payer's
/// vault never appears in it — however the batch's canonical fold
/// happens to order the two.
#[test]
fn a_receipt_carries_only_its_own_payers_burn() {
    let payer_a = fee_payer(31);
    let payer_b = fee_payer(32);
    let accounts = [(payer_a, 1_000), (payer_b, 1_000), (bob(), 50)];
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let txs = vec![
        Arc::new(Verified::<Transaction>::from_persisted(
            signed_transfer_with_fee(31, payer_a, bob(), 100, 10),
        )),
        Arc::new(Verified::<Transaction>::from_persisted(
            signed_transfer_with_fee(32, payer_b, bob(), 100, 10),
        )),
    ];
    let executed = execute_on(&accounts, &executor, &txs);

    for (own, sibling, tx) in [
        (payer_a, payer_b, &executed[0]),
        (payer_b, payer_a, &executed[1]),
    ] {
        let ConsensusReceipt::Succeeded { writes, .. } = &tx.consensus else {
            panic!("both transfers must succeed: {:?}", tx.consensus);
        };
        assert_eq!(
            vault_cell(&settled(writes, &accounts), own),
            Some(encode_amount(1_000 - 100 - 10).to_vec()),
            "a receipt carries its own payer's burn"
        );
        assert!(
            !writes.cells.contains_key(&vault_key(sibling, *XRD)),
            "a receipt never carries a sibling payer's vault"
        );
    }
}

/// Three transactions sharing one payer settle to the sum of their
/// burns: each receipt's vault write carries the cumulative debit at its
/// position in the batch's canonical order, so applying the receipts in
/// block order — which is that same order, blocks being strictly
/// hash-sorted — loses none of them.
#[test]
fn shared_payer_burns_accumulate_across_a_batch() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    // Distinct ceilings make three distinct stamps; the stamp's fuel
    // exceeds all of them, so each burns exactly its ceiling.
    let mut txs: Vec<Arc<Verified<Transaction>>> = [10u128, 11, 12]
        .into_iter()
        .map(|fee| {
            Arc::new(Verified::<Transaction>::from_persisted(
                signed_stamp_with_fee(ALICE_SEED, alice(), fee),
            ))
        })
        .collect();
    txs.sort_by_key(|tx| tx.hash());

    let mut store = MapDb::genesis(&[(alice(), 1_000), (bob(), 50)]);
    let executed = execute_batch_on(&store, &executor, &txs);
    for tx in &executed {
        let ConsensusReceipt::Succeeded { writes, .. } = &tx.consensus else {
            panic!("every stamp must succeed: {:?}", tx.consensus);
        };
        store.apply(writes);
    }
    assert_eq!(
        store.substate(vault_key(alice(), *XRD)),
        Some(encode_amount(1_000 - 10 - 11 - 12).to_vec()),
        "every burn in the batch reaches the committed balance"
    );
}

/// A missed edge bound is an infeasibility, not a defect: the sender
/// declared what it would accept and the world moved between signing and
/// execution, so nothing but the class floor leaves its vault.
#[test]
fn a_missed_edge_bound_charges_its_payer_the_floor() {
    let payer = fee_payer(23);
    let funded = 1_000;
    let accounts = [(payer, funded), (bob(), 50)];
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    // The withdrawal is covered and the guest is honest — it returns the
    // 100 it reserved. What fails is the recipient's signed floor.
    let tx = signed_transfer_under_bound(23, payer, bob(), 100, 150, 1_000);
    let floor = tx.body().abort_floor();
    let executed = execute_on(
        &accounts,
        &executor,
        &[Arc::new(Verified::<Transaction>::from_persisted(tx))],
    );
    assert_eq!(executed.len(), 1);
    assert!(
        matches!(executed[0].consensus, ConsensusReceipt::Failed),
        "a missed bound must not apply: {:?}",
        executed[0].consensus
    );

    // The charge stands in for the receipt the execution never produced,
    // which is what keeps state moving through receipts alone.
    let Some(ConsensusReceipt::Succeeded {
        writes: database_updates,
        ..
    }) = executed[0].fee_receipt.as_ref()
    else {
        panic!("a charged abort settles a fee receipt");
    };
    assert_eq!(
        vault_cell(&settled(database_updates, &accounts), payer),
        Some(encode_amount(funded - floor).to_vec()),
        "the floor and nothing else"
    );
    assert_eq!(
        vault_cell(&settled(database_updates, &accounts), bob()),
        None,
        "the transfer's own effects are discarded"
    );
}

#[test]
fn a_payer_drained_by_its_own_fee_deletes_its_vault() {
    // The burn folds outside the kernel store, so it has to apply the
    // store's delete-on-zero rule itself — otherwise the commonest way a
    // vault empties leaves sixteen zero bytes behind, and a storage bond
    // that can never be refunded.
    let payer = fee_payer(11);
    // Exactly the transfer plus the ceiling: nothing survives the burn.
    let accounts = [(payer, 110), (bob(), 50)];
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let tx = Arc::new(Verified::<Transaction>::from_persisted(
        signed_transfer_with_fee(11, payer, bob(), 100, 10),
    ));
    let executed = execute_on(&accounts, &executor, &[tx]);
    let ConsensusReceipt::Succeeded {
        writes: database_updates,
        ..
    } = &executed[0].consensus
    else {
        panic!("transfer must succeed: {:?}", executed[0].consensus);
    };

    assert!(
        vault_removed(&settled(database_updates, &accounts), payer),
        "a drained payer vault is deleted, not zeroed"
    );
    // The recipient is untouched by the rule.
    assert_eq!(
        vault_cell(&settled(database_updates, &accounts), bob()),
        Some(encode_amount(150).to_vec())
    );
}

/// An account whose prefix routes to the other half of a two-shard trie
/// from [`alice`], derived by flipping the bit that trie splits on so the
/// pair straddles it whatever address derivation produces.
fn far() -> Address {
    let base = alice();
    let mut body = base.body();
    body[0] ^= 0x80;
    Address::new(body, base.class())
}

/// Execute one batch as `local_shard` under a two-leaf trie.
fn execute_on_shard(
    executor: &Executor,
    local_shard: ShardId,
    transactions: &[Arc<Verified<Transaction>>],
) -> Vec<ExecutedTx> {
    let snapshot_store = MapDb::genesis(&[(alice(), 1_000), (far(), 50)]);
    let trie = ShardTrie::uniform(1);
    let ctx = TickBatchContext {
        par: Parallelism::Sequential,
        local_shard,
        shard_trie: &trie,
        block_hash: BlockHash::from_raw(Hash::from_bytes(b"block")),
        tick_ts: WeightedTimestamp::from_millis(1_000),
        tick_reveal: RevealChain::ZERO,
        holds: &ProvisionalHolds::new(),
    };
    executor.execute_batch(&ctx, &snapshot_store, transactions)
}

fn events_of(executed: &ExecutedTx) -> Vec<(Address, u32)> {
    let ConsensusReceipt::Succeeded { events, .. } = &executed.consensus else {
        panic!("transfer must succeed: {:?}", executed.consensus);
    };
    events
        .iter()
        .map(|event| (event.emitter, event.event_type))
        .collect()
}

fn hash_of(executed: &ExecutedTx) -> Hash {
    let ConsensusReceipt::Succeeded { receipt_hash, .. } = &executed.consensus else {
        panic!("transfer must succeed");
    };
    *receipt_hash.as_raw()
}

/// A transfer's two legs emit from accounts on different shards. Each
/// shard's receipt keeps only the events its own instances emitted, while
/// the receipt hash — which covers the union — stays identical, so the
/// committees agree on what the transaction emitted without either shard
/// storing the other's events.
#[test]
fn an_event_lands_only_on_its_emitters_home_shard() {
    let world = vec![(alice(), 1_000u128), (far(), 50), (fee_payer(7), 1_000)];
    let executor = Executor::new(&world, ExecutionMode::Serial);
    let trie = ShardTrie::uniform(1);
    let (near_shard, far_shard) = (trie.shard_for_prefix(alice()), trie.shard_for_prefix(far()));
    assert_ne!(
        near_shard, far_shard,
        "the two accounts must sit on different shards"
    );

    // A zero ceiling: the fee burn is a payer-shard write, so a nonzero
    // fee would make the union differ by exactly that cell between the
    // two sides. Events are the subject here; the fee stays out of it.
    let tx = Arc::new(Verified::<Transaction>::from_persisted(
        signed_transfer_with_fee(ALICE_SEED, alice(), far(), 100, 0),
    ));
    let sender_side = execute_on_shard(&executor, near_shard, std::slice::from_ref(&tx));
    let recipient_side = execute_on_shard(&executor, far_shard, &[tx]);

    assert_eq!(events_of(&sender_side[0]), vec![(alice(), 0)]);
    assert_eq!(events_of(&recipient_side[0]), vec![(far(), 1)]);
    assert_eq!(
        hash_of(&sender_side[0]),
        hash_of(&recipient_side[0]),
        "the receipt hash covers the union, so it cannot differ by shard",
    );
}

/// A fan-out: two withdrawals from one account funding two deposits in
/// a single manifest — the shape a multi-recipient cross-shard payment
/// takes.
#[test]
fn a_two_recipient_fan_out_executes() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let graph = ManifestGraph {
        nodes: vec![
            GraphNode {
                target: alice(),
                method: "withdraw".into(),
                args: vec![
                    GraphArg::Literal(Value::Address(*XRD)),
                    GraphArg::Literal(Value::U128(5)),
                ],
            },
            GraphNode {
                target: bob(),
                method: "deposit".into(),
                args: vec![GraphArg::Edge {
                    edge: EdgeRef {
                        producer: 0,
                        output: 0,
                    },
                    constraints: vec![Constraint::ResourceIs(*XRD)],
                }],
            },
            GraphNode {
                target: alice(),
                method: "withdraw".into(),
                args: vec![
                    GraphArg::Literal(Value::Address(*XRD)),
                    GraphArg::Literal(Value::U128(6)),
                ],
            },
            GraphNode {
                target: fee_payer(7),
                method: "deposit".into(),
                args: vec![GraphArg::Edge {
                    edge: EdgeRef {
                        producer: 2,
                        output: 0,
                    },
                    constraints: vec![Constraint::ResourceIs(*XRD)],
                }],
            },
        ],
    };
    let key = Ed25519PrivateKey::from_bytes(&[ALICE_SEED; 32]).unwrap();
    let tree = EnvelopeTree {
        root: IntentDecl {
            graph,
            params: Vec::new(),
        },
        root_bindings: Vec::new(),
        subintents: Vec::new(),
    };
    let vm = TransactionEnvelope {
        body: TransactionBody::Call(encode_tree(&tree)),
        subintent_sigs: Vec::new(),
        fee_payer: alice(),
        max_fee: 10,
        gas_limit: 1_000_000,
        validity_start_ms: 0,
        validity_end_ms: u64::MAX,
        message: Vec::new(),
        network: NetworkId(242),
        signer: [0; 32],
        signature: [0; 64],
    }
    .sign(&key);
    let tx = Arc::new(Verified::<Transaction>::from_persisted(Transaction::new(
        vm,
    )));
    let executed = execute_on(
        &[(alice(), 1_000), (bob(), 50), (fee_payer(7), 1_000)],
        &executor,
        &[tx],
    );
    let ConsensusReceipt::Succeeded { writes, .. } = &executed[0].consensus else {
        panic!("the fan-out must succeed: {:?}", executed[0].consensus);
    };
    assert_eq!(
        vault_cell(&settled(writes, &world_accounts()), alice()),
        Some(encode_amount(1_000 - 5 - 6 - 10).to_vec())
    );
    assert_eq!(
        vault_cell(&settled(writes, &world_accounts()), bob()),
        Some(encode_amount(55).to_vec())
    );
    assert_eq!(
        vault_cell(&settled(writes, &world_accounts()), fee_payer(7)),
        Some(encode_amount(1_006).to_vec())
    );
}

/// A signed publish of `artifact`, paid for by `seed`'s account.
fn signed_publish(seed: u8, artifact: Vec<u8>) -> Transaction {
    let key = Ed25519PrivateKey::from_bytes(&[seed; 32]).unwrap();
    let vm = TransactionEnvelope {
        body: TransactionBody::Publish(artifact),
        subintent_sigs: Vec::new(),
        fee_payer: account_address(&key.public_key().0),
        max_fee: 1_000_000,
        gas_limit: 1_000_000,
        validity_start_ms: 0,
        validity_end_ms: u64::MAX,
        message: Vec::new(),
        network: NetworkId(242),
        signer: [0; 32],
        signature: [0; 64],
    }
    .sign(&key);
    Transaction::new(vm)
}

/// The raw update a batch made to a package's cell under `publisher`.
fn package_cell(writes: &StateWrites, publisher: Address, artifact: &[u8]) -> Option<Vec<u8>> {
    let key = package_key(publisher, package_hash(&ProtocolHasher, artifact));
    writes.cells.get(&key).cloned().flatten()
}

#[test]
fn a_publish_writes_the_artifact_under_its_publisher() {
    let payer = fee_payer(7);
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let artifact = account_artifact().to_vec();
    let tx = Arc::new(Verified::<Transaction>::from_persisted(signed_publish(
        7,
        artifact.clone(),
    )));
    // Funded well above the burn: at the placeholder rate of one unit
    // per artifact byte, publishing the stdlib guest costs more than the
    // balances the transfer fixtures use.
    let executed = execute_on(&[(payer, 1_000_000)], &executor, &[tx]);

    let ConsensusReceipt::Succeeded {
        writes: database_updates,
        ..
    } = &executed[0].consensus
    else {
        panic!("a publish must succeed: {:?}", executed[0].consensus);
    };
    assert_eq!(
        package_cell(database_updates, payer, &artifact).as_deref(),
        Some(artifact.as_slice()),
        "the artifact lands whole in its content-addressed cell"
    );
    // The publisher paid: the vault carries the burn, and the fee is the
    // only other thing a publish writes.
    let paid = vault_cell(&settled(database_updates, &[(payer, 1_000_000)]), payer)
        .expect("the payer's vault was written");
    assert_eq!(
        paid,
        encode_amount(1_000_000 - artifact.len() as u128).to_vec(),
        "the publisher paid exactly what judging its artifact cost"
    );
    assert!(
        executed[0].attested_work > 0,
        "judging the artifact is attested work"
    );
}

/// The stdlib artifact's ABI bindings survive the round trip through the
/// metadata section, which is what makes the account guest callable
/// through them rather than through a table that knows its method names.
#[test]
fn the_stdlib_artifact_carries_resolvable_bindings() {
    let metadata = admit_package(account_artifact()).expect("the stdlib artifact admits");
    assert_eq!(
        metadata.methods["withdraw"].abi,
        vec![AbiParam::Handle(0), AbiParam::Derived(Expr::Arg(1))],
        "the binding decoded is the binding authored"
    );
    assert_eq!(
        metadata.methods["deposit"].abi,
        vec![AbiParam::Handle(0), AbiParam::Bucket(0)],
        "a bucket's amount is the one argument a signature cannot derive"
    );
}

#[test]
fn a_publish_that_is_not_a_package_never_reaches_execution() {
    // The whole publish verdict is a function of the artifact's bytes,
    // so it is reached at admission: derivation refuses, the transaction
    // is never included, and nobody pays for it or stores it.
    let junk = b"\0asm\x01\0\0\0".to_vec();
    assert!(
        admit_package(&junk).is_err(),
        "well-formed wasm framing is not a package"
    );
    assert!(
        admit_package(account_artifact()).is_ok(),
        "the stdlib artifact is one"
    );
}

#[test]
fn a_committed_publish_grows_the_cache_that_routing_reads() {
    let payer = fee_payer(7);
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);

    // A package the world has never seen: the stdlib artifact with its
    // metadata attached a second time under a different publisher would
    // be the same bytes, so vary the metadata to vary the address.
    let mut metadata = account_metadata();
    metadata.events.push("republished".into());
    let artifact = attach_metadata(ACCOUNT_COMPONENT, &metadata).expect("attaches");
    let package = package_hash(&ProtocolHasher, &artifact);

    let cache = executor.packages();
    assert!(
        cache.load().get(package).is_none(),
        "the package is unknown before its block commits"
    );

    let tx = Arc::new(Verified::<Transaction>::from_persisted(signed_publish(
        7, artifact,
    )));
    let executed = execute_on(&[(payer, 1_000_000)], &executor, &[tx]);
    let ConsensusReceipt::Succeeded { .. } = &executed[0].consensus else {
        panic!("the publish must succeed: {:?}", executed[0].consensus);
    };

    // Executing is not committing: the cache learns the package from the
    // committed receipt, which is what a synced replica also replays.
    assert!(
        cache.load().get(package).is_none(),
        "execution alone does not publish"
    );
    absorb_committed_cells([&executed[0].consensus]);
    assert_eq!(
        cache.load().get(package),
        Some(&metadata),
        "the committed cell published exactly the metadata the artifact declares"
    );
}

#[test]
fn only_a_cell_that_addresses_its_own_contents_publishes() {
    // A package cell is self-identifying: its key is the content address
    // of the value it holds. Without that check, any committed cell
    // whose bytes happened to parse as an artifact would publish a
    // package — no publish transaction, no fee, no cell of its own.
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let cache = executor.packages();

    let mut metadata = account_metadata();
    metadata.events.push("smuggled".into());
    let artifact = attach_metadata(ACCOUNT_COMPONENT, &metadata).expect("attaches");
    let package = package_hash(&ProtocolHasher, &artifact);
    let publisher = fee_payer(11);

    // The right bytes at the wrong key: a vault slot, not the content
    // address. Refused.
    let vault = vault_key(publisher, *XRD);
    cache.absorb_cell(publisher, vault.local.0, &artifact);
    assert!(
        cache.load().get(package).is_none(),
        "an artifact stored anywhere but its own address is not a package"
    );

    // The same bytes at the key their own hash builds. Published.
    let cell = package_key(publisher, package);
    cache.absorb_cell(publisher, cell.local.0, &artifact);
    assert_eq!(cache.load().get(package), Some(&metadata));
}

/// Preview `tx` against a genesis snapshot of `accounts`, committing
/// nothing.
fn preview_on(
    accounts: &[(Address, u128)],
    executor: &Executor,
    tx: &Transaction,
    grants: PreviewGrants,
) -> PreviewReport {
    let snapshot_store = MapDb::genesis(accounts);
    executor.preview(
        &snapshot_store,
        tx,
        PreviewInputs {
            clock: WeightedTimestamp::from_millis(1_000),
            randomness: RevealChain::ZERO,
            grants,
        },
    )
}

/// The reported change to `owner`'s native vault.
fn change_for(report: &PreviewReport, owner: Address) -> ResourceChange {
    let key = vault_key(owner, *XRD);
    *report
        .changes
        .iter()
        .find(|change| change.key == key)
        .unwrap_or_else(|| panic!("no reported change for {owner:?}: {:?}", report.changes))
}

/// The preview fixture: a payer who funds the transfer and its fee, and a
/// recipient. The ceiling sits far below the fuel a transfer burns, so the
/// charge is the cap rather than the actual.
const PREVIEW_CEILING: u128 = 10;

struct PreviewFixture {
    payer: Address,
    accounts: Vec<(Address, u128)>,
    tx: Transaction,
}

fn preview_fixture() -> PreviewFixture {
    let payer = fee_payer(7);
    PreviewFixture {
        payer,
        accounts: vec![(payer, 1_000), (bob(), 50)],
        tx: signed_transfer_with_fee(7, payer, bob(), 100, PREVIEW_CEILING),
    }
}

/// A preview reports the transfer's resource changes: what leaves the
/// sender's vault, what reaches the recipient's, and what the fee costs on
/// top — read off the receipt's settles and movements without committing
/// anything.
#[test]
fn a_preview_reports_the_resource_changes_a_transfer_would_make() {
    let PreviewFixture {
        payer,
        accounts,
        tx,
    } = preview_fixture();
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let report = preview_on(&accounts, &executor, &tx, PreviewGrants::default());

    assert_eq!(report.outcome, PreviewOutcome::Completed);
    assert_eq!(report.fee, PREVIEW_CEILING, "a transfer's fuel exceeds it");
    assert_eq!(report.changes.len(), 2, "two vaults move: {report:?}");

    let sender = change_for(&report, payer);
    assert_eq!((sender.before, sender.after), (1_000, 1_000 - 100 - 10));
    assert_eq!(
        (sender.settled, sender.credit, sender.debit),
        (100, 0, 0),
        "a withdrawal leaves through its reservation's settle"
    );

    let recipient = change_for(&report, bob());
    assert_eq!((recipient.before, recipient.after), (50, 150));
    assert_eq!(
        (recipient.credit, recipient.debit, recipient.settled),
        (100, 0, 0),
        "a deposit arrives as a commutative credit"
    );

    // Nothing moved: the same preview twice reports the same thing.
    assert_eq!(
        preview_on(&accounts, &executor, &tx, PreviewGrants::default()),
        report,
        "a preview commits nothing, so it is repeatable"
    );
}

/// The preview's arithmetic is the tick's arithmetic: what it says a
/// vault would hold is what the committed receipt writes there.
#[test]
fn a_preview_agrees_with_the_tick_that_would_commit_it() {
    let PreviewFixture {
        payer,
        accounts,
        tx,
    } = preview_fixture();
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let report = preview_on(&accounts, &executor, &tx, PreviewGrants::default());

    let verified = Arc::new(Verified::<Transaction>::from_persisted(tx));
    let executed = execute_on(&accounts, &executor, &[verified]);
    let ConsensusReceipt::Succeeded {
        writes: database_updates,
        ..
    } = &executed[0].consensus
    else {
        panic!("the transfer must succeed: {:?}", executed[0].consensus);
    };

    for owner in [payer, bob()] {
        assert_eq!(
            vault_cell(&settled(database_updates, &world_accounts()), owner),
            Some(encode_amount(change_for(&report, owner).after).to_vec()),
            "the preview's figure for {owner:?} is what the tick commits"
        );
    }
}

/// Free credit: the fee is still priced and reported, but it never
/// reaches the payer's vault, so a wallet can price an envelope its payer
/// could not cover.
#[test]
fn free_credit_reports_the_fee_without_charging_it() {
    let PreviewFixture {
        payer,
        accounts,
        tx,
    } = preview_fixture();
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let charged = preview_on(&accounts, &executor, &tx, PreviewGrants::default());
    let credited = preview_on(
        &accounts,
        &executor,
        &tx,
        PreviewGrants {
            free_credit: true,
            ..PreviewGrants::default()
        },
    );

    assert_eq!(credited.fee, charged.fee, "the fee is priced either way");
    assert_eq!(
        change_for(&credited, payer).after,
        change_for(&charged, payer).after + charged.fee,
        "credit keeps exactly the fee off the payer's vault"
    );
    assert_eq!(
        change_for(&credited, bob()),
        change_for(&charged, bob()),
        "a grant to the payer moves nobody else"
    );
}

/// An uncovered withdrawal previews as the abort it would be, priced at
/// the class floor: the sender lost a deterministic race rather than
/// making a mistake, so nothing but the floor leaves its vault.
#[test]
fn a_preview_prices_an_abort_at_its_class_floor() {
    let payer = fee_payer(7);
    let accounts = [(payer, 1_000), (bob(), 50)];
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let tx = signed_transfer_with_fee(7, payer, bob(), 5_000, PREVIEW_CEILING);
    let report = preview_on(&accounts, &executor, &tx, PreviewGrants::default());

    let PreviewOutcome::Aborted { reason } = &report.outcome else {
        panic!("an uncovered withdrawal must abort: {:?}", report.outcome);
    };
    assert!(reason.contains("infeasible"), "reason = {reason}");
    assert_eq!(report.fee, PREVIEW_CEILING / 10, "the abort floor");
    assert_eq!(
        report.changes,
        vec![ResourceChange {
            key: vault_key(payer, *XRD),
            before: 1_000,
            after: 1_000 - PREVIEW_CEILING / 10,
            credit: 0,
            debit: 0,
            settled: 0,
        }],
        "an abort moves nothing but the floor"
    );
}

/// An envelope admission would refuse previews as refused, and costs
/// nothing: it could never enter a block, so nobody would pay for it.
#[test]
fn a_preview_refuses_what_admission_would_refuse() {
    let stranger = test_prefix(0xAB);
    assert!(
        !world_accounts().iter().any(|(a, _)| *a == stranger),
        "the address must be outside the world"
    );
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let tx = signed_transfer_with_fee(7, stranger, bob(), 10, PREVIEW_CEILING);
    let report = preview_on(&[(bob(), 50)], &executor, &tx, PreviewGrants::default());

    assert!(
        matches!(report.outcome, PreviewOutcome::Refused { .. }),
        "outcome = {:?}",
        report.outcome
    );
    assert_eq!(report.fee, 0);
    assert!(report.changes.is_empty());
}

/// A preview holds a node to its target's authority like the chain does,
/// and the grant is what a wallet reaches for when it wants an answer
/// about an envelope its counterparties have not signed yet.
///
/// Without it, a wallet composing a two-party trade would be told
/// "refused" and have nothing to show the user. With it, the report is
/// what the composition would do once signed — which is exactly the
/// question being asked.
#[test]
fn a_preview_holds_a_node_to_its_targets_authority_unless_granted() {
    let payer = fee_payer(7);
    let accounts = [(payer, 1_000), (alice(), 1_000), (bob(), 50)];
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    // Signed by 7, withdrawing from Alice: the shape the gate refuses.
    let tx = signed_transfer_with_fee(7, alice(), bob(), 100, PREVIEW_CEILING);

    let refused = preview_on(&accounts, &executor, &tx, PreviewGrants::default());
    assert!(
        matches!(refused.outcome, PreviewOutcome::Refused { .. }),
        "outcome = {:?}",
        refused.outcome
    );
    assert_eq!(refused.fee, 0);

    let granted = preview_on(
        &accounts,
        &executor,
        &tx,
        PreviewGrants {
            assume_target_auth: true,
            ..PreviewGrants::default()
        },
    );
    assert_eq!(granted.outcome, PreviewOutcome::Completed);
    assert_eq!(change_for(&granted, alice()).debit, 0);
    assert_eq!(change_for(&granted, alice()).settled, 100);
    assert_eq!(change_for(&granted, bob()).credit, 100);
}

/// A publish previews too, and its price needs no state: judging an
/// artifact costs one unit per byte, which is the whole answer.
#[test]
fn a_preview_prices_a_publish_by_its_artifact() {
    let payer = fee_payer(7);
    let artifact = account_artifact().to_vec();
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let tx = signed_publish(7, artifact.clone());
    let report = preview_on(
        &[(payer, 1_000_000)],
        &executor,
        &tx,
        PreviewGrants::default(),
    );

    assert_eq!(report.outcome, PreviewOutcome::Completed);
    assert_eq!(report.fee, artifact.len() as u128);
    let vault = change_for(&report, payer);
    assert_eq!(
        (vault.before, vault.after),
        (1_000_000, 1_000_000 - artifact.len() as u128)
    );

    // An artifact that is not a package is refused at admission, so it
    // never enters a block and costs its publisher nothing.
    let junk = signed_publish(7, b"\0asm\x01\0\0\0".to_vec());
    let refused = preview_on(
        &[(payer, 1_000_000)],
        &executor,
        &junk,
        PreviewGrants::default(),
    );
    assert!(matches!(refused.outcome, PreviewOutcome::Refused { .. }));
    assert_eq!(refused.fee, 0);
}

#[test]
fn a_committed_cell_that_is_not_a_package_is_ignored() {
    // The other half: ordinary traffic cannot grow the cache by
    // accident, whatever it writes.
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let tx = Arc::new(Verified::<Transaction>::from_persisted(signed_transfer(
        ALICE_SEED,
        alice(),
        bob(),
        100,
    )));
    let executed = execute(&executor, &[tx]);
    let bogus = package_hash(&ProtocolHasher, encode_amount(900).as_ref());
    absorb_committed_cells([&executed[0].consensus]);
    assert!(
        executor.packages().load().get(bogus).is_none(),
        "vault writes are not packages"
    );
}
