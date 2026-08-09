//! Reshape lifecycle scenarios.

use std::collections::BTreeSet;
use std::sync::Arc;

use hyperscale_types::{BlockHeight, ShardId, Transaction, WeightedTimestamp};

use crate::support::query::{committee_size, live_shards};
use crate::support::tx::{build_probe_transfer_tx, validity_around};
use crate::support::wait::{
    assert_height_frozen, await_beacon_epoch, await_height, await_merge_keeper_count,
    await_root_matches_anchor, await_serves, await_serves_ahead_of_anchor, await_split_admitted,
};
use crate::support::{Cluster, epochs, grow_to, vote_reshape_threshold};

/// Reshape `split_bytes` the vote installs after the grow. Its derived
/// `merge_bytes = split_bytes / 8` sits far above each cold child's byte total,
/// so both assert the merge once the change activates, while staying far above
/// them so neither re-splits.
const MERGE_VOTE_SPLIT_BYTES: u64 = 80_000_000;

/// Grow the single-shard root to a `target`-leaf partition and assert the
/// reached topology against the beacon's committed committees.
///
/// Drives the portable [`grow_to`] step, then checks that the live committees are
/// exactly the `target`-leaf partition with every leaf at the genesis committee's
/// full strength. Committee seating is the property: a split that seats an
/// under-strength child still commits past genesis (quorum can hold below full
/// strength), so [`grow_to`]'s own commit check alone would miss it. Requires a
/// config with `split_bytes = 0` and `(target - 1)` cohorts of pool surplus.
///
/// # Panics
///
/// Panics if the beacon does not fold before the grow, the grow misses its
/// budget, or the reached topology is not the full-strength `target` partition.
fn grow_reaches_topology(c: &mut impl Cluster, target: u32) {
    assert!(
        await_beacon_epoch(c, 1, epochs(6)),
        "the beacon must fold before the grow so the genesis committee strength is known",
    );
    let strength = committee_size(c, ShardId::ROOT).expect("genesis seats the root committee");

    grow_to(c, target);

    let depth = target.trailing_zeros();
    let leaves: Vec<ShardId> = (0..u64::from(target))
        .map(|path| ShardId::leaf(depth, path))
        .collect();
    let expected: BTreeSet<ShardId> = leaves.iter().copied().collect();
    assert!(
        c.run_until(epochs(8), |c| {
            live_shards(c) == expected
                && leaves
                    .iter()
                    .all(|&leaf| committee_size(c, leaf) == Some(strength))
        }),
        "the grow must seat exactly the {target}-leaf partition, each at full committee strength ({strength})",
    );
}

/// Grow the root into a two-leaf partition, both leaves at full committee
/// strength.
///
/// # Panics
///
/// Panics if the grow misses its budget or the reached topology is not the
/// full-strength two-leaf partition.
pub fn grow_reaches_two_shard_topology(c: &mut impl Cluster) {
    grow_reaches_topology(c, 2);
}

/// Grow the root into a four-leaf partition through two split generations, every
/// leaf at full committee strength.
///
/// # Panics
///
/// Panics if the grow misses its budget or the reached topology is not the
/// full-strength four-leaf partition.
pub fn grow_reaches_four_shard_topology(c: &mut impl Cluster) {
    grow_reaches_topology(c, 4);
}

/// Arm an organic split of the root shard and drive it to completion.
///
/// The beacon admits the split from the armed trigger, both children are served
/// and commit past genesis at the beacon-composed anchor root, and the parent
/// terminates. Requires a config with `split_bytes = 0` and one cohort of pool
/// surplus.
///
/// # Panics
///
/// Panics if any lifecycle stage misses its budget.
pub fn split_lifecycle(c: &mut impl Cluster) {
    let root = ShardId::ROOT;
    let (left, right) = root.children();

    assert!(
        await_split_admitted(c, root, epochs(8)),
        "beacon did not admit the root split within budget"
    );

    // Keep the parent committing through its final window: a real transfer
    // gives it activity so it coasts to its crossing and the fold seeds both
    // children from the terminal contribution.
    c.submit(Arc::new(build_probe_transfer_tx(validity_around(c.now()))));

    // Each child seats from the terminal crossing its members followed,
    // ahead of the fold that publishes its anchor. Serving alone would
    // also pass on the anchor fallback an epoch later, so the assertion
    // is on *when*: a child that seats only once its anchor exists has
    // taken the fallback, and the cut-over is not being exercised.
    for child in [left, right] {
        match await_serves_ahead_of_anchor(c, child, epochs(28)) {
            Some(true) => {}
            Some(false) => {
                panic!("split child {child} seated off its beacon anchor, not its parent's cut")
            }
            None => panic!("split child {child} was not served within budget"),
        }
    }
    assert!(
        await_height(c, left, 1, epochs(8)) && await_height(c, right, 1, epochs(8)),
        "split children did not commit past genesis within budget"
    );
    assert!(
        await_root_matches_anchor(c, left, epochs(8))
            && await_root_matches_anchor(c, right, epochs(8)),
        "split child roots did not match the beacon anchor within budget"
    );
    assert_height_frozen(c, root, epochs(2));
}

/// How many probes the replay train submits per epoch of the parent's
/// remaining life.
///
/// A probe is only a replay candidate while its own validity window still
/// contains the moment it is resubmitted, and `MAX_VALIDITY_RANGE` is
/// under an epoch, so one probe per epoch would leave gaps a cut can land
/// in. Four spaces them at a quarter of a window's reach even when an
/// epoch is the production five minutes.
const PROBES_PER_EPOCH: u64 = 4;

/// Upper bound on the train, and so on the funding
/// [`probe_train_genesis_accounts`] has to cover.
pub const MAX_REPLAY_PROBES: u32 = 48;

/// Whether both split children are seated and committing.
fn children_live<C: Cluster>(c: &C, left: ShardId, right: ShardId) -> bool {
    [left, right].iter().all(|&child| {
        c.serves_shard(child) && c.committed_height(child).is_some_and(|h| h.inner() >= 1)
    })
}

/// A transaction the terminating parent committed cannot commit again on
/// either child.
///
/// The parent's `CommitDedupIndex` dies with its chain and both children
/// construct their own empty, so nothing a child holds refuses the
/// resubmission: its QC chain is empty, its mempool holds no tombstone
/// from the parent's sweep, and the transaction's own validity window
/// still contains its anchor. The replay is refused only if a child
/// inherits what its parent committed.
///
/// Requires [`probe_train_genesis_accounts`] funding and a config with
/// `split_bytes = 0` and one cohort of pool surplus.
///
/// # Panics
///
/// Panics if any lifecycle stage misses its budget, if no probe survives
/// to be a replay candidate (which would make the assertion vacuous), or
/// if either child commits the replay.
pub fn split_boundary_refuses_a_replay(c: &mut impl Cluster) {
    let root = ShardId::ROOT;
    let (left, right) = root.children();

    // Block cadence is activity-driven and scales with neither the epoch
    // nor the harness, so the train's spacing is measured rather than
    // assumed. The probe supplies the activity; the measurement epoch is
    // absorbed into the wait for admission that follows it.
    c.submit(Arc::new(build_probe_transfer_tx(validity_around(c.now()))));
    let before = committed_height(c, root);
    c.run_until(epochs(1), |_| false);
    let blocks_per_epoch = committed_height(c, root).saturating_sub(before);
    let spacing = (blocks_per_epoch / PROBES_PER_EPOCH).max(1);

    assert!(
        await_split_admitted(c, root, epochs(8)),
        "beacon did not admit the root split within budget"
    );

    // A train rather than one probe. The replay candidate has to be
    // committed on the parent *and* still inside its validity window once
    // the children accept blocks, while admission to cut runs one to two
    // epochs — longer than any window a transaction is allowed to sign.
    // A single probe submitted at admission is expired by the cut, and the
    // resubmission would then be refused for expiry rather than for the
    // duplicate this is about.
    let mut probes: Vec<Arc<Transaction>> = Vec::new();
    while probes.len() < MAX_REPLAY_PROBES as usize && !children_live(c, left, right) {
        let probe = Arc::new(build_probe_transfer_tx(validity_around(c.now())));
        probes.push(Arc::clone(&probe));
        c.submit(probe);
        let from = committed_height(c, root);
        c.run_until(epochs(4), |c| {
            children_live(c, left, right) || committed_height(c, root) >= from + spacing
        });
    }

    assert!(
        children_live(c, left, right),
        "split children were not live within the train's budget",
    );

    // The candidate: committed on the parent, and still signed for the
    // instant it is about to be resubmitted at. Read off the chain rather
    // than inferred from when it was submitted, so the scenario does not
    // depend on where the cut fell.
    let anchor = WeightedTimestamp::ZERO.plus(c.now());
    let replay = probes
        .iter()
        .rev()
        .find(|probe| {
            probe.validity_range().contains(anchor) && c.chain_fate(root, probe.hash()).0.is_some()
        })
        .cloned();
    let replay = replay.expect(
        "no probe both committed on the parent and outlived the cut — \
         the train is mis-spaced and the replay assertion would be vacuous",
    );
    let replayed = replay.hash();

    c.submit(replay);
    c.run_until(epochs(4), |_| false);

    for child in [left, right] {
        assert!(
            c.chain_fate(child, replayed).0.is_none(),
            "child {child} committed {replayed}, which its parent had already committed",
        );
    }
}

/// `shard`'s committed height as a plain number, zero before it commits.
fn committed_height<C: Cluster>(c: &C, shard: ShardId) -> u64 {
    c.committed_height(shard).map_or(0, BlockHeight::inner)
}

/// Grow the root into two shards, then merge the two cold children back into it.
///
/// Composes [`split_lifecycle`] for the grow, then votes the reshape threshold
/// up so the children fall under the derived merge threshold — a grown topology
/// can't merge under the frozen threshold that split it, so the vote is the
/// honest trigger. The beacon pairs the merge and draws the keeper committee,
/// the keepers seat the reformed parent, and its committed root reproduces the
/// beacon-composed anchor. Requires a config with `split_bytes = 0`, one cohort
/// of pool surplus, and a funded straddler account (`31`) to pay the vote.
///
/// # Panics
///
/// Panics if any lifecycle stage misses its budget.
pub fn merge_lifecycle(c: &mut impl Cluster) {
    let root = ShardId::ROOT;

    split_lifecycle(c);

    // Vote the reshape threshold up so the cold grown children fall under the
    // derived merge threshold. The straddler account `31`, funded at genesis and
    // seated on a child by the grow, pays the system-action fee.
    vote_reshape_threshold(c, MERGE_VOTE_SPLIT_BYTES);

    // The vote activates, both children assert the merge, and the beacon pairs
    // it — drawing a quorum (2f+1 of the four-validator merged committee).
    assert!(
        await_merge_keeper_count(c, root, 3, epochs(20)),
        "the merge did not pair a keeper quorum within budget"
    );
    // The keepers seat the reformed parent, which commits past its merged
    // genesis at the beacon-composed anchor root.
    assert!(
        await_serves(c, root, epochs(28)),
        "the merged parent was not served within budget"
    );
    assert!(
        await_height(c, root, 1, epochs(8)),
        "the merged parent did not commit past genesis within budget"
    );
    assert!(
        await_root_matches_anchor(c, root, epochs(8)),
        "the merged root did not match the beacon anchor within budget"
    );
}

/// Merge two cold children back into the root and assert the reformed parent
/// seats a full keeper committee that keeps committing past its merged genesis.
///
/// Composes [`merge_lifecycle`] (grow → vote → keepers paired → reformed parent
/// served → anchor matched), then layers the seating outcome: the keeper set —
/// half of each child's committee — seats a full committee on the parent, which
/// then commits a real block past its merged genesis (the lifecycle's anchor
/// match only requires the merged genesis block itself). Requires the
/// [`merge_lifecycle`] preconditions.
///
/// # Panics
///
/// Panics if the merge misses its budget, the reformed parent is under committee
/// strength, or it stalls at its merged genesis.
pub fn merge_seats_full_keeper_committee(c: &mut impl Cluster) {
    assert!(
        await_beacon_epoch(c, 1, epochs(6)),
        "the beacon must fold before the grow so the genesis committee strength is known",
    );
    let strength = committee_size(c, ShardId::ROOT).expect("genesis seats the root committee");

    merge_lifecycle(c);

    let root = ShardId::ROOT;
    assert!(
        c.run_until(epochs(6), |c| committee_size(c, root) == Some(strength)),
        "the reformed parent must seat a full keeper committee of {strength}",
    );

    // A committed-height probe can transiently read `None` while a vnode's
    // serving surface hands over, so wait the height into view before
    // taking the base.
    assert!(
        c.run_until(epochs(2), |c| c.committed_height(root).is_some()),
        "the reformed parent must report a committed height",
    );
    let base = c
        .committed_height(root)
        .expect("the reformed parent commits");
    assert!(
        c.run_until(epochs(6), |c| c
            .committed_height(root)
            .is_some_and(|h| h > base)),
        "the reformed parent must keep committing past its merged genesis",
    );
}
