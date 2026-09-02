//! Derived read-only combinators over a [`Cluster`], plus the store-level
//! queries the adaptors share.
//!
//! The combinators are projections of [`Cluster::beacon_state`]; the
//! store-level queries ([`chain_fate`], [`status_rank`]) back both adaptors'
//! trait impls. All are kept out of the trait so the two adaptors share one
//! definition and cannot drift apart.

use std::collections::BTreeSet;

use hyperscale_engine::genesis::vault_key;
use hyperscale_engine::{XRD, publish_work};
use hyperscale_storage::ShardChainReader;
use hyperscale_types::{
    Address, BlockHash, BlockHeight, ConsensusPublicKey, Epoch, PendingReshape, ResourceAddr,
    ShardId, ShardTrie, Stake, StakePool, StakePoolId, StateRoot, SubstateKey, Transaction,
    TransactionDecision, TransactionStatus, TxHash, ValidatorId, ValidatorStatus,
};

use super::Cluster;

/// The deepest leaf [`served_shards`] looks for. A scenario that grows or
/// splits reaches depth two; nothing here goes deeper.
const MAX_SEARCHED_DEPTH: u32 = 3;

/// What a scenario transaction is charged, whatever refuses it.
///
/// The one figure the fee schedule has: a success burns it, and so does
/// every refusal a sender can reach, so a scenario asserting what a
/// vault moved reads it once and compares against it either way.
/// Borrowed from the cluster's own derivation, because a transaction a
/// scenario assembled has been derived by nobody. A publish is priced by
/// its artifact's length and held to its ceiling rather than by the call
/// rule.
///
/// # Panics
///
/// Panics if the transaction does not derive, which for a scenario
/// fixture means it was built wrong.
pub fn declared_price<C: Cluster + ?Sized>(c: &C, tx: &Transaction) -> u128 {
    let body = tx.try_body().expect("a scenario fixture decodes");
    if let Some(artifact) = body.artifact() {
        return u128::from(publish_work(artifact)).min(body.max_fee);
    }
    tx.price_under(c.derivation().as_ref())
        .expect("a scenario fixture derives")
}

/// Every shard this cluster currently serves, ascending.
///
/// There is no enumeration on [`Cluster`] — a scenario names the shards
/// it stood up — so this asks the only question the trait answers, over
/// every leaf a scenario could have reached.
#[must_use]
pub fn served_shards<C: Cluster + ?Sized>(c: &C) -> Vec<ShardId> {
    (0..=MAX_SEARCHED_DEPTH)
        .flat_map(|depth| (0..1u64 << depth).map(move |path| ShardId::leaf(depth, path)))
        .filter(|shard| c.serves_shard(*shard))
        .collect()
}

/// What `owner` holds of `resource`, wherever its prefix currently sits.
///
/// Routed rather than searched, because a scenario that reshapes moves
/// the answer: the cell keeps its key across a split — the key is the
/// owner's prefix and a local half, neither of which a reshape rewrites
/// — but the shard serving it changes, and a shard that has handed its
/// prefix on still answers for it, at the state it froze at.
#[must_use]
pub fn held<C: Cluster + ?Sized>(c: &C, owner: Address, resource: ResourceAddr) -> u128 {
    held_at(c, vault_key(owner, resource))
}

/// The committed amount in one cell, wherever its prefix currently sits.
///
/// The general form of [`held`], for value a component keeps in its own
/// state rather than in the protocol's vault slot: an account's balance
/// is reachable from its address, and a pool's reserves are reachable
/// only from the package's own slot, which the scenario declaring the
/// package is what knows.
#[must_use]
pub fn held_at<C: Cluster + ?Sized>(c: &C, cell: SubstateKey) -> u128 {
    let shard = owning_shard(c, cell.owner);
    c.substate(shard, cell.owner, cell.local.0)
        .map_or(0, |bytes| {
            <[u8; 16]>::try_from(bytes.as_slice()).map_or(0, u128::from_le_bytes)
        })
}

/// The live shard whose prefix `owner` falls under.
///
/// The beacon's live leaf partition is the authority: a split's parent
/// and a merge's children keep serving their frozen stores after the
/// cut, and a search over everything served would read whichever of
/// them answered first. Before the beacon has folded anything the root
/// is the one shard there is.
#[must_use]
pub fn owning_shard<C: Cluster + ?Sized>(c: &C, owner: Address) -> ShardId {
    let live = live_shards(c);
    if live.is_empty() {
        return ShardId::ROOT;
    }
    ShardTrie::from_leaves(live).shard_for_prefix(owner)
}

/// Walk `store`'s committed chain from height 1 for `tx`'s fate.
///
/// Returns the height at which `tx` was committed (rides a block's
/// `transactions`) and the height plus decision at which it was finalized
/// (rides a `Finalization` certificate). The decision matters at a reshape
/// boundary: a counterpart abort finalizes the straddler with `Aborted`,
/// which a presence-only check would misread as a one-sided apply.
#[must_use]
pub fn chain_fate(
    store: &impl ShardChainReader,
    tx: TxHash,
) -> (
    Option<BlockHeight>,
    Option<(BlockHeight, TransactionDecision)>,
) {
    let mut committed = None;
    let mut finalized = None;
    let tip = store.committed_height();
    let mut height = BlockHeight::new(1);
    while height <= tip {
        if let Some(certified) = store.get_block(height) {
            let block = certified.block();
            if block.transactions().iter().any(|t| t.hash() == tx) {
                committed = Some(height);
            }
            for fw in block.certificates().iter() {
                if let Some((_, decision)) = fw.tx_decisions().into_iter().find(|(h, _)| *h == tx) {
                    finalized = Some((height, decision));
                }
            }
        }
        height = height.next();
    }
    (committed, finalized)
}

/// The committed balance of `owner`'s native vault on `shard`, read through
/// the harness's client-proven snapshot seam.
///
/// # Panics
///
/// Panics if the cell holds anything but an amount.
#[must_use]
pub fn vault_balance<C: Cluster>(c: &C, shard: ShardId, owner: impl Into<Address>) -> u128 {
    let vault = vault_key(owner, *XRD);
    c.substate(shard, vault.owner, vault.local.0)
        .map_or(0, |bytes| {
            let cell: [u8; 16] = bytes.as_slice().try_into().expect("an amount cell");
            u128::from_le_bytes(cell)
        })
}

/// Rank a transaction status so a cluster-wide view takes the most advanced
/// observation.
#[must_use]
pub const fn status_rank(status: &TransactionStatus) -> u8 {
    match status {
        TransactionStatus::Pending => 0,
        TransactionStatus::Committed(_) => 1,
        TransactionStatus::Completed(_) => 2,
    }
}

/// The latest committed beacon epoch, if the cluster has folded one.
#[must_use]
pub fn beacon_epoch<C: Cluster>(c: &C) -> Option<Epoch> {
    c.beacon_state().map(|state| state.current_epoch)
}

/// Whether the beacon has admitted a split for `parent` — a pending `Split`
/// record carrying the drawn observer cohort.
#[must_use]
pub fn split_admitted<C: Cluster>(c: &C, parent: ShardId) -> bool {
    c.beacon_state().is_some_and(|state| {
        matches!(
            state.pending_reshapes.get(&parent),
            Some(PendingReshape::Split { .. })
        )
    })
}

/// The final epoch window the beacon has scheduled `parent`'s reshape to
/// terminate on — the cut its chain ends at and its successors take over
/// at.
///
/// `None` while the reshape is admitted but its readiness gate has yet to
/// stamp a cut. A `Some` answer is irrevocable, so a scenario may build
/// against it.
#[must_use]
pub fn scheduled_terminal_epoch<C: Cluster>(c: &C, parent: ShardId) -> Option<Epoch> {
    c.beacon_state()
        .and_then(|state| state.pending_reshapes.get(&parent)?.scheduled_terminal())
}

/// Milliseconds of weighted time per epoch, off the beacon's own chain
/// config — what turns an epoch number into the boundary it starts at.
#[must_use]
pub fn epoch_duration_ms<C: Cluster>(c: &C) -> Option<u64> {
    c.beacon_state()
        .map(|state| state.chain_config.epoch_duration_ms)
}

/// The beacon-composed anchor root for `shard` — the `boundaries` `state_root`
/// a flip must reproduce.
#[must_use]
pub fn anchor_root<C: Cluster>(c: &C, shard: ShardId) -> Option<StateRoot> {
    c.beacon_state()
        .and_then(|state| state.boundaries.get(&shard).map(|b| b.state_root))
}

/// The genesis height the beacon composed onto `shard`'s boundary, once the
/// fold that publishes the anchor has run.
///
/// `None` while the record is still the placeholder a reshape cut installs,
/// which carries a zero block hash. A split child outruns that fold — it
/// flips from its own follow of the parent and serves from the cut — so
/// "served" no longer implies "anchored".
#[must_use]
pub fn anchored_genesis_height<C: Cluster>(c: &C, shard: ShardId) -> Option<BlockHeight> {
    c.beacon_state().and_then(|state| {
        state
            .boundaries
            .get(&shard)
            .filter(|boundary| boundary.block_hash != BlockHash::ZERO)
            .map(|boundary| boundary.height)
    })
}

/// The number of keepers drawn for a merge into `parent`, once paired (both
/// children hold a live half). `None` before pairing.
#[must_use]
pub fn merge_keeper_count<C: Cluster>(c: &C, parent: ShardId) -> Option<usize> {
    c.beacon_state()
        .and_then(|state| match state.pending_reshapes.get(&parent) {
            Some(PendingReshape::Merge {
                keepers,
                admitted_at: Some(_),
                ..
            }) => Some(keepers.len()),
            _ => None,
        })
}

/// The number of validators seated on `shard`'s current committee, or `None` if
/// the beacon seats no committee there (the shard is unborn or terminated).
#[must_use]
pub fn committee_size<C: Cluster>(c: &C, shard: ShardId) -> Option<usize> {
    c.beacon_state().and_then(|state| {
        state
            .shard_committees
            .get(&shard)
            .map(|cm| cm.members.len())
    })
}

/// The set of shards the beacon currently seats a committee for — the live leaf
/// partition.
#[must_use]
pub fn live_shards<C: Cluster + ?Sized>(c: &C) -> BTreeSet<ShardId> {
    c.beacon_state()
        .map(|state| state.shard_committees.keys().copied().collect())
        .unwrap_or_default()
}

/// The total stake folded into `pool`, or `None` if the beacon holds no record
/// of it — counting deposits whether or not they have unbonded.
#[must_use]
pub fn pool_total_stake<C: Cluster>(c: &C, pool: StakePoolId) -> Option<Stake> {
    c.beacon_state()
        .and_then(|state| state.pools.get(&pool).map(|p| p.total_stake))
}

/// The effective (bonded) stake of `pool` — total less any stake still inside
/// its unbonding window. A withdrawal drops this immediately while
/// [`pool_total_stake`] holds until the unbond matures.
#[must_use]
pub fn pool_effective_stake<C: Cluster>(c: &C, pool: StakePoolId) -> Option<Stake> {
    c.beacon_state()
        .and_then(|state| state.pools.get(&pool).map(StakePool::effective_stake))
}

/// The folded status of validator `id`, or `None` if the beacon holds no record
/// of it.
#[must_use]
pub fn validator_status<C: Cluster>(c: &C, id: ValidatorId) -> Option<ValidatorStatus> {
    c.beacon_state()
        .and_then(|state| state.validators.get(&id).map(|r| r.status))
}

/// The registered consensus public key of validator `id`, or `None` if unregistered.
#[must_use]
pub fn validator_pubkey<C: Cluster>(c: &C, id: ValidatorId) -> Option<ConsensusPublicKey> {
    c.beacon_state()
        .and_then(|state| state.validators.get(&id).map(|r| r.pubkey))
}
