//! The theft probe: a manifest moving an account's funds under someone
//! else's signature, run through the whole real path — derivation, the
//! verification gate, and the batch executor.
//!
//! Its own binary because the assertion is about what a signature reaches,
//! and the VM statics install once per process: the victim has to be a
//! funded, registered account in the world every executor here shares.
//!
//! The refusal is asserted at the admission gate rather than at execution.
//! A verdict reachable from signed content alone belongs where the sender
//! pays nothing for it, and nothing downstream re-derives its own opinion:
//! routing runs through the same statics, so an envelope the gate refused
//! cannot reach a block at all.

use std::collections::BTreeMap;
use std::sync::Arc;

use hyperscale_effects_bridge::{account_address, encode_tree};
use hyperscale_engine::genesis::vault_key;
use hyperscale_engine::{
    ExecutedTx, ExecutionMode, Executor, Parallelism, WaveBatchContext, XRD, genesis_writes,
};
use hyperscale_storage::SubstateDatabase;
use hyperscale_types::{
    BlockHash, ConsensusReceipt, Ed25519PrivateKey, EnvelopeExt, Hash, NetworkId, ProvisionalHolds,
    RevealChain, ShardId, ShardTrie, StateWrites, SubstateKey, Transaction, TransactionBody,
    TransactionEnvelope, Verified, WeightedTimestamp,
};
use hyperscale_vm_effects::{
    Address, Constraint, EdgeRef, EnvelopeTree, GraphArg, GraphNode, IntentDecl, ManifestGraph,
    Value,
};
use hyperscale_vm_kernel::encode_amount;

/// A funded account whose key nothing in this binary holds — the address
/// is all an attacker has, and the address is public.
const VICTIM: [u8; 16] = [0x99; 16];
/// The signing seed of the account that pays for the theft.
const THIEF: u8 = 3;
/// What both accounts hold at genesis.
const FUNDED: u128 = 10_000;

/// A snapshot over the flattened genesis updates.
struct MapDb(BTreeMap<SubstateKey, Vec<u8>>);

impl MapDb {
    fn genesis(accounts: &[([u8; 16], u128)]) -> Self {
        let writes = genesis_writes(accounts, &[]);
        let mut map = BTreeMap::new();
        for (key, change) in &writes.cells {
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

fn thief() -> [u8; 16] {
    let key = Ed25519PrivateKey::from_bytes(&[THIEF; 32]).unwrap();
    account_address(&key.public_key().0)
}

/// Every address any test in this binary transacts with.
fn world_accounts() -> Vec<([u8; 16], u128)> {
    vec![(VICTIM, FUNDED), (thief(), FUNDED)]
}

fn withdraw(target: [u8; 16], amount: u128) -> GraphNode {
    GraphNode {
        target: Address(target),
        method: "withdraw".into(),
        args: vec![
            GraphArg::Literal(Value::Address(XRD)),
            GraphArg::Literal(Value::U128(amount)),
        ],
    }
}

fn deposit(target: [u8; 16], producer: u32) -> GraphNode {
    GraphNode {
        target: Address(target),
        method: "deposit".into(),
        args: vec![GraphArg::Edge {
            edge: EdgeRef {
                producer,
                output: 0,
            },
            constraints: vec![Constraint::ResourceIs(XRD)],
        }],
    }
}

/// `from.withdraw(XRD, amount) -> to.deposit(..)`, signed and paid for by
/// the thief whatever `from` says.
fn signed_transfer(from: [u8; 16], to: [u8; 16], amount: u128) -> Transaction {
    let key = Ed25519PrivateKey::from_bytes(&[THIEF; 32]).unwrap();
    let tree = EnvelopeTree {
        root: IntentDecl {
            graph: ManifestGraph {
                nodes: vec![withdraw(from, amount), deposit(to, 0)],
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
            fee_payer: Address(thief()),
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
    let store = MapDb::genesis(&world_accounts());
    let trie = ShardTrie::single();
    let ctx = WaveBatchContext {
        par: Parallelism::Sequential,
        local_shard: ShardId::ROOT,
        shard_trie: &trie,
        block_hash: BlockHash::from_raw(Hash::from_bytes(b"block")),
        wave_start_ts: WeightedTimestamp::from_millis(1_000),
        wave_start_reveal: RevealChain::ZERO,
        holds: &ProvisionalHolds::new(),
    };
    let verified = Arc::new(Verified::<Transaction>::from_persisted(tx));
    executor.execute_wave_batch(&ctx, &store, std::slice::from_ref(&verified))
}

/// An account's native vault as the batch left it.
fn vault_cell(writes: &StateWrites, owner: [u8; 16]) -> Option<Vec<u8>> {
    writes.cells.get(&vault_key(owner, XRD)).cloned().flatten()
}

/// The defect, closed: an address is public, and knowing one buys nothing.
#[test]
fn draining_an_account_the_envelope_does_not_sign_for_is_refused() {
    let _ = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let theft = signed_transfer(VICTIM, thief(), 5_000);

    assert!(theft.body().signature_is_valid());
    let refused = theft.try_derived().expect_err("derivation refuses");
    assert!(refused.0.contains("withdraw"), "{}", refused.0);
    assert!(
        refused.0.contains("authority"),
        "the refusal names its reason: {}",
        refused.0
    );

    // The thief spending their own account is the admitted case, so what
    // bites is whose account the node names and not the shape of the
    // manifest.
    assert!(
        signed_transfer(thief(), VICTIM, 5_000)
            .try_derived()
            .is_ok()
    );
}

/// What the gate is holding back, stated as an amount.
///
/// The same two-node manifest with one address changed, settled end to
/// end: a withdrawal moves whatever the node asks for, so what a target
/// binding protects is a whole balance rather than a fee floor of it.
#[test]
fn the_gated_node_is_the_one_that_moves_the_balance() {
    let executor = Executor::new(&world_accounts(), ExecutionMode::Serial);
    let executed = execute(&executor, signed_transfer(thief(), VICTIM, 5_000));
    let ConsensusReceipt::Succeeded {
        writes: database_updates,
        ..
    } = &executed[0].consensus
    else {
        panic!(
            "the signed transfer must settle: {:?}",
            executed[0].consensus
        );
    };
    // Half the payer's balance in one node, less the fee they also pay,
    // and the recipient credited without having signed anything.
    assert_eq!(
        vault_cell(database_updates, thief()),
        Some(encode_amount(4_000).to_vec())
    );
    assert_eq!(
        vault_cell(database_updates, VICTIM),
        Some(encode_amount(15_000).to_vec())
    );
}
