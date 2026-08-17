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
use std::sync::{Arc, LazyLock};

use hyperscale_effects_bridge::account_address;
use hyperscale_engine::genesis::{GenesisPackages, vault_key};
use hyperscale_engine::{
    ExecutedTx, ExecutionMode, Executor, TickBatchContext, XRD, genesis_writes,
};
use hyperscale_storage::Substates;
use hyperscale_transactions::{Client, Terms};
use hyperscale_types::{
    ConsensusReceipt, Ed25519PrivateKey, EnvelopeExt, NetworkId, PrincipalAddr, ProvisionalHolds,
    RevealChain, SettledWrites, ShardId, ShardTrie, StateWrites, SubstateKey, TimestampRange,
    Transaction, Verified, WeightedTimestamp,
};
use hyperscale_vm_effects::{Address, CollectionId};
use hyperscale_vm_kernel::{amount_cell, encode_amount};

/// A funded account whose key nothing in this binary holds — the address
/// is all an attacker has, and the address is public.
const VICTIM: PrincipalAddr = PrincipalAddr::new([0x99; 31]);
/// The signing seed of the account that pays for the theft.
const THIEF: u8 = 3;
/// What both accounts hold at genesis.
const FUNDED: u128 = 10_000;

/// A snapshot over the flattened genesis updates.
struct MapDb(BTreeMap<SubstateKey, Vec<u8>>);

impl MapDb {
    fn genesis(accounts: &[(PrincipalAddr, u128)]) -> Self {
        let writes = genesis_writes(accounts, &[], &GenesisPackages::protocol());
        let mut map = BTreeMap::new();
        for (key, change) in writes.cells() {
            let value = change.clone().expect("genesis writes are Set-only");
            map.insert(*key, value);
        }
        Self(map)
    }
}

impl Substates for MapDb {
    fn cell(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.0.get(&key).cloned()
    }

    fn entries_in_range(
        &self,
        _owner: Address,
        _collection: CollectionId,
        _lo: u128,
        _hi: u128,
        _limit: usize,
    ) -> Vec<(u128, Vec<u8>)> {
        Vec::new()
    }
}

fn thief() -> PrincipalAddr {
    let key = Ed25519PrivateKey::from_bytes(&[THIEF; 32]).unwrap();
    account_address(&key.public_key().0)
}

/// The opening balances every test in this binary starts from.
fn world_accounts() -> Vec<(PrincipalAddr, u128)> {
    vec![(VICTIM, FUNDED), (thief(), FUNDED)]
}

/// The client this binary builds through: the stdlib world, on the one
/// network its envelopes name.
fn client() -> &'static Client {
    static CLIENT: LazyLock<Client> = LazyLock::new(|| Client::genesis(NetworkId(242)));
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

/// `from.withdraw(*XRD, amount) -> to.deposit(..)`, signed and paid for by
/// the thief whatever `from` says.
fn signed_transfer(from: PrincipalAddr, to: PrincipalAddr, amount: u128) -> Transaction {
    let key = Ed25519PrivateKey::from_bytes(&[THIEF; 32]).unwrap();
    let graph = client()
        .transfer_graph(from, to, amount)
        .expect("an account answers a transfer");
    Transaction::new(client().sign(graph, &key, terms(1_000)))
}

fn execute(executor: &Executor, tx: Transaction) -> Vec<ExecutedTx> {
    let store = MapDb::genesis(&world_accounts());
    let trie = ShardTrie::single();
    let ctx = TickBatchContext {
        local_shard: ShardId::ROOT,
        shard_trie: &trie,
        tick_ts: WeightedTimestamp::from_millis(1_000),
        tick_reveal: RevealChain::ZERO,
        holds: &ProvisionalHolds::new(),
    };
    let verified = Arc::new(Verified::<Transaction>::from_persisted(tx));
    executor.execute_batch(&ctx, &store, std::slice::from_ref(&verified))
}

/// A receipt's writes as they settle onto `accounts`.
///
/// A receipt says what it moved, not what the cell ends at, so a balance
/// assertion has to name the state the movement lands on.
fn settled(writes: &StateWrites, accounts: &[(PrincipalAddr, u128)]) -> SettledWrites {
    writes.resolve(&mut |key| {
        accounts
            .iter()
            .find(|(owner, _)| vault_key(*owner, *XRD) == key)
            .and_then(|(_, amount)| amount_cell(*amount).map(|cell| cell.to_vec()))
    })
}

/// An account's native vault as the batch left it.
fn vault_cell(writes: &SettledWrites, owner: impl Into<Address>) -> Option<Vec<u8>> {
    writes
        .cells()
        .get(&vault_key(owner, *XRD))
        .cloned()
        .flatten()
}

/// The defect, closed: an address is public, and knowing one buys nothing.
///
/// The envelope is well-formed — the thief's own badge is presented, and
/// admission asks only that a guarded call present something — so the
/// verdict is the victim's account's to give, and it aborts. The thief
/// pays the ceiling they signed for having asked.
#[test]
fn draining_an_account_the_envelope_does_not_sign_for_is_refused() {
    let executor = Executor::new(ExecutionMode::Serial);
    let theft = signed_transfer(VICTIM, thief(), 5_000);

    assert!(theft.body().signature_is_valid());
    assert!(theft.try_derived().is_ok(), "the shape is well-formed");

    let executed = execute(&executor, theft);
    let ConsensusReceipt::Failed = &executed[0].consensus else {
        panic!("the theft must not settle: {:?}", executed[0].consensus);
    };

    // The victim's balance is untouched, which is the property the whole
    // mechanism exists for.
    let executed = execute(&executor, signed_transfer(thief(), VICTIM, 5_000));
    assert!(matches!(
        &executed[0].consensus,
        ConsensusReceipt::Succeeded { .. }
    ));
}

/// What the gate is holding back, stated as an amount.
///
/// The same two-node manifest with one address changed, settled end to
/// end: a withdrawal moves whatever the node asks for, so what a target
/// binding protects is a whole balance rather than a fee floor of it.
#[test]
fn the_gated_node_is_the_one_that_moves_the_balance() {
    let executor = Executor::new(ExecutionMode::Serial);
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
    // and the recipient credited without having signed anything. The
    // remainder tracks the fee, so it moves whenever what the guests
    // execute does.
    assert_eq!(
        vault_cell(&settled(database_updates, &world_accounts()), thief()),
        Some(encode_amount(4_726).to_vec())
    );
    assert_eq!(
        vault_cell(&settled(database_updates, &world_accounts()), VICTIM),
        Some(encode_amount(15_000).to_vec())
    );
}
