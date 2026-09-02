//! Straddler atomicity scenarios.
//!
//! A *straddler* is a cross-shard tick whose source side commits on a shard that
//! terminates at a reshape boundary. The surviving counterpart must read the
//! terminating shard's beacon-attested settled set and settle the straddler only
//! when the terminating shard settled it by its terminal block — never one-sided,
//! and never holding a permanent lock on the ones it didn't.

use std::fmt::Write;
use std::sync::Arc;

use hyperscale_engine::XRD;
use hyperscale_types::{
    BlockHeight, Ed25519PrivateKey, Epoch, PrincipalAddr, ShardId, TransactionDecision,
    TransactionStatus, TxHash,
};

use crate::reshape::split_lifecycle;
use crate::support::conservation::{Charges, World};
use crate::support::query::{
    anchored_genesis_height, beacon_epoch, committee_size, split_admitted, vault_balance,
};
use crate::support::tx::{
    MERGE_STRADDLER_LEFT, MERGE_STRADDLER_RIGHT, MERGE_STRADDLER_SURVIVOR, STRADDLER_SPLITTER,
    STRADDLER_SUCCESSOR, STRADDLER_SURVIVOR, SplitStraddlerSetup, TERMINATING_PAYER_FUNDING,
    build_reshape_threshold_vote_tx, build_transfer_tx, merge_straddler_setup, pool_operator,
    split_straddler_setup, stdlib_flash_bytes, validity_around,
};
use crate::support::wait::{
    await_anchor_seeded, await_beacon_epoch, await_merge_keeper_count, await_root_matches_anchor,
    await_serves, await_split_admitted, await_tx_terminal,
};
use crate::support::{Cluster, FaultHandle, FaultableCluster, epochs};

/// Cut every path by which `shard`'s committee obtains `peer_shard`'s execution
/// certificate, so a cross-shard tick the two share cannot finalize on `shard`'s
/// side.
///
/// Fault rules gate pushes (gossip) and request legs, never response legs, so EC
/// intake is cut on the push and on the pulls, with the correct direction per
/// leg: `peer_shard` pushes its EC by gossip (`execution.cert.batch`), and
/// `shard` pulls the EC and the finalization that bundles it
/// (`execution_cert.request`, `finalization.request`). Provisions and headers
/// still flow, so `shard` still executes the tick and produces its own EC; it
/// just never receives `peer_shard`'s.
///
/// Faithful only with disjoint committees — if the two shards share a host, its
/// co-hosted vnodes hand the EC across in-process, which no network rule
/// intercepts.
#[must_use]
pub fn isolate_ec_intake(
    c: &mut impl FaultableCluster,
    shard: ShardId,
    peer_shard: ShardId,
) -> FaultHandle {
    let shard_hosts = c.committee_hosts(shard);
    let peer_hosts = c.committee_hosts(peer_shard);
    let handles = [
        c.drop_type_between(&peer_hosts, &shard_hosts, "execution.cert.batch"),
        c.drop_type_between(&shard_hosts, &peer_hosts, "execution_cert.request"),
        c.drop_type_between(&shard_hosts, &peer_hosts, "finalization.request"),
    ];
    FaultHandle::new(move || handles.iter().map(FaultHandle::fired).sum())
}

/// Epochs of lead the threshold vote carries beyond the fold budget, covering
/// the tally read at `activate_at - 1` plus scheduling slack.
const VOTE_ACTIVATE_LEAD_BASE: u64 = 3;

/// Epochs of lead before the threshold vote activates: the harness's vote
/// fold budget expressed in epochs, plus the base. Unlike
/// `vote_reshape_threshold`'s retry loop, this vote is cast once — the
/// activation epoch anchors the straddler boundary choreography — so the lead
/// must cover the harness's cast-to-fold latency up front.
const fn vote_activate_lead(fold_budget_ms: u64, epoch_ms: u64) -> u64 {
    VOTE_ACTIVATE_LEAD_BASE + fold_budget_ms.div_ceil(epoch_ms)
}

/// Reshape `split_bytes` the vote installs after the grow: between the
/// survivor's byte total and the splitter's, so only the heavier splitter
/// crosses and terminates while the survivor stays a live leaf.
///
/// Offset by the genesis package flash, which the survivor's half holds
/// beside its own ballast, while the splitter's ballast carries a fixed
/// lead over the flash — so the threshold sits between the two with
/// margins that hold as the stdlib grows, and the derived merge floor (an
/// eighth of it) stays below every live leaf, including the splitter's
/// own children once it splits.
pub fn straddler_split_bytes() -> u64 {
    split_bytes_over(stdlib_flash_bytes())
}

/// The voted-down threshold for a pair whose survivor holds a genesis
/// flash of `flash` bytes beside its ballast, and whose splitter's
/// ballast leads the flash by a fixed margin.
pub const fn split_bytes_over(flash: u64) -> u64 {
    flash + 12_000
}

/// Verify a split straddler settles atomically across the reshape boundary.
///
/// Grows the root into two shards (the heavier `leaf(1,0)` splitter and the
/// lighter `leaf(1,1)` survivor), votes `split_bytes` down so only the splitter
/// crosses, then submits cross-shard transfers from the survivor into the
/// splitter spread across the splitter's grow — the earliest settle before its
/// terminal block, the latest name a splitter that has already terminated. After
/// the split the survivor must reach a terminal verdict on every straddler,
/// consistent with what the splitter settled: never applying one the splitter
/// never settled, never contradicting one it did, and aborting (not hanging) the
/// rest. Requires the [`split_straddler_setup`] genesis funding.
///
/// # Panics
///
/// Panics if the grow or split misses its budget, or the settled-transaction fence is
/// breached (a one-sided application, a mismatch, or a hung straddler).
pub fn split_straddler_atomic(c: &mut impl Cluster) {
    let run = split_straddler_run(c, |_| {});
    assert_fence_held(c, run.splitter, run.terminal_b, &run.probes);
    run.assert_conserved(c);
}

/// Verify a split straddler settles atomically when the terminating splitter is
/// isolated from the survivor's execution certificate.
///
/// The same choreography as [`split_straddler_atomic`], but with the splitter's
/// EC intake cut ([`isolate_ec_intake`]) once committees stabilize: provisions
/// still flow, so the splitter executes each straddler and produces its own EC,
/// but never receives the survivor's and so settles none. The pre-boundary
/// settlement fence must hold atomicity anyway — the survivor cannot finalize a
/// straddler naming the splitter while the splitter has an admitted terminating
/// reshape, so no straddler resolves one-sided.
///
/// Requires disjoint splitter/survivor committees (no shared host), or a
/// co-hosted vnode bridges the EC across in-process, which no network rule
/// intercepts. The simulation seats these via dedicated pool hosts.
///
/// # Panics
///
/// Panics if the choreography misses its budget or any straddler resolves
/// one-sided (the survivor applies one the splitter never settled).
pub fn split_straddler_ec_partition_atomic(c: &mut impl FaultableCluster) {
    let run = split_straddler_run(c, |c| {
        let _ = isolate_ec_intake(c, STRADDLER_SPLITTER, STRADDLER_SURVIVOR);
    });
    let one_sided = straddler_one_sided_count(c, run.splitter, run.terminal_b, &run.probes);
    assert_eq!(
        one_sided, 0,
        "the survivor applied {one_sided} straddler(s) one-sided the splitter never settled",
    );
    run.assert_conserved(c);
}

/// Grow the root into the splitter/survivor pair and vote the reshape
/// threshold down so only the heavier splitter crosses it and terminates.
///
/// The shared front half of every split-straddler choreography: on return the
/// splitter has an admitted split, the survivor has none, and both committees
/// are stable — the seam a fault probe keys a rule on.
///
/// # Panics
///
/// Panics if the grow misses its budget, the splitter fails to admit a split,
/// or the survivor admits one.
pub fn arm_splitter_termination<C: Cluster>(c: &mut C) {
    split_lifecycle(c);
    vote_splitter_down(c);
}

/// Vote the reshape threshold down on an already grown pair so only the
/// heavier splitter crosses it, and await its admission.
///
/// The second half of [`arm_splitter_termination`], for a cluster that
/// reached the pair some other way — a grow the harness drove itself, as
/// one born running a package does.
///
/// # Panics
///
/// Panics if either shard is unserved, the splitter fails to admit a
/// split, or the survivor admits one.
pub fn vote_splitter_down<C: Cluster>(c: &mut C) {
    vote_splitter_down_to(c, straddler_split_bytes());
}

/// [`vote_splitter_down`] to an explicit threshold, for a pair whose byte
/// bands a scenario sized itself.
///
/// # Panics
///
/// As [`vote_splitter_down`].
pub fn vote_splitter_down_to<C: Cluster>(c: &mut C, split_bytes: u64) {
    let splitter = STRADDLER_SPLITTER;
    let survivor = STRADDLER_SURVIVOR;
    cast_splitter_vote(c, split_bytes);
    assert!(
        await_split_admitted(c, splitter, epochs(20)),
        "only the over-threshold splitter must admit a split",
    );
    assert!(
        !split_admitted(c, survivor),
        "the under-threshold survivor must not split",
    );
}

/// Cast the vote that takes the threshold to `split_bytes`, below the
/// splitter's bytes, without waiting for the admission it leads to.
///
/// # Panics
///
/// Panics if either shard of the pair is unserved.
pub fn cast_splitter_vote<C: Cluster>(c: &mut C, split_bytes: u64) {
    assert!(
        c.serves_shard(STRADDLER_SPLITTER) && c.serves_shard(STRADDLER_SURVIVOR),
        "the grow must seat both the splitter and the survivor",
    );

    let current = beacon_epoch(c).expect("post-grow beacon epoch");
    let epoch_ms = c
        .beacon_state()
        .expect("post-grow beacon state")
        .chain_config
        .epoch_duration_ms;
    let vote = build_reshape_threshold_vote_tx(
        &pool_operator().0,
        split_bytes,
        Epoch::new(current.inner() + vote_activate_lead(c.vote_fold_budget_ms(), epoch_ms)),
        validity_around(c.now()),
    );
    c.submit(Arc::new(vote));
}

/// What a straddler choreography leaves for its caller to judge: the
/// probes, the shard that terminated and its terminal block, and the
/// conservation ledger the run was opened over.
pub struct StraddlerRun {
    /// The probe hashes, in submission order.
    pub probes: Vec<TxHash>,
    /// The shard that terminated.
    pub splitter: ShardId,
    /// Its terminal block height.
    pub terminal_b: BlockHeight,
    world: World,
    charges: Charges,
}

impl StraddlerRun {
    /// Assert every straddler's payer and recipient hold between them what
    /// they started with, less the prices: a straddler the fence abandoned
    /// moved nothing and still paid, and one it settled moved its payment
    /// exactly once.
    pub fn assert_conserved<C: Cluster>(&self, c: &mut C) {
        self.world.assert_settles_within(
            c,
            &self.charges,
            epochs(8),
            "straddlers across a terminal",
        );
    }
}

/// Everything a setup's straddlers can reach: each leg's payer and
/// recipient. The ballast holds the byte skew and never spends.
fn straddler_world<C: Cluster>(
    c: &C,
    straddlers: &[(Ed25519PrivateKey, PrincipalAddr, PrincipalAddr)],
) -> World {
    World::open(
        c,
        *XRD,
        straddlers
            .iter()
            .flat_map(|(_, from, to)| [from.address(), to.address()]),
        [],
    )
}

/// The split-straddler choreography, minus the terminal assertion.
///
/// Grows, votes the threshold down, submits settling then straddling ticks,
/// drives the split, and waits for every straddler to reach a terminal verdict.
/// Returns the probe hashes, the splitter shard, and its terminal block height
/// for the caller to judge. `before_settling` runs once after the split is
/// admitted (committees stable, splitter still live) and before the settling
/// ticks are submitted — the seam a fault probe uses to install a rule keyed on
/// the live committees.
///
/// # Panics
///
/// Panics if the grow or split misses its budget, or a straddler never reaches
/// a terminal verdict.
pub fn split_straddler_run<C: Cluster>(
    c: &mut C,
    mut before_settling: impl FnMut(&mut C),
) -> StraddlerRun {
    let splitter = STRADDLER_SPLITTER;
    let (child_left, child_right) = splitter.children();
    let setup = split_straddler_setup();

    arm_splitter_termination(c);

    let world = straddler_world(c, &setup.straddlers);
    let mut charges = Charges::default();
    let mut probes: Vec<TxHash> = Vec::new();

    // Fault-injection seam: committees are stable and the splitter is still
    // live, so a probe can key a rule on the live splitter/survivor committees
    // before any straddler EC crosses.
    before_settling(c);

    // Settling ticks: submitted while the splitter still commits real blocks, so
    // it finalizes them before its terminal cut — they settle atomically.
    let half = setup.straddlers.len() / 2;
    for (key, from, to) in setup.straddlers.iter().take(half) {
        probes.push(submit_straddler(c, &mut charges, key, *from, *to));
    }

    // Advance until the gate drains the splitter from `pending_reshapes`: the
    // settling ticks finalize on it in this window, and it then coasts to its
    // terminal crossing committing only empty blocks.
    assert!(
        c.run_until(epochs(14), |c| !split_admitted(c, splitter)),
        "the splitter's split must gate within budget",
    );

    // Straddling ticks: submitted all at once during the coast — the splitter is
    // still the active leaf, so the survivor provisions to it, but its empty
    // coast blocks settle nothing, leaving them in flight when it terminates.
    for (key, from, to) in setup.straddlers.iter().skip(half) {
        probes.push(submit_straddler(c, &mut charges, key, *from, *to));
    }

    // The split executes: both children seat and commit past genesis.
    assert!(
        await_serves(c, child_left, epochs(28)) && await_serves(c, child_right, epochs(28)),
        "both splitter children must be served within budget",
    );

    // The splitter's terminal block sits one below the children's genesis. The
    // children serve from the cut, ahead of the fold that publishes their
    // anchor, so the height only reads off the boundary once that fold lands.
    assert!(
        await_anchor_seeded(c, child_left, epochs(6)),
        "the beacon must compose the split children's anchor",
    );
    let terminal_b = anchored_genesis_height(c, child_left)
        .and_then(BlockHeight::prev)
        .expect("the children's seeded genesis pins the splitter's terminal block");

    // Every straddler must reach a terminal verdict on the survivor.
    for hash in &probes {
        let status = await_tx_terminal(c, *hash, epochs(10));
        assert!(
            matches!(status, Some(TransactionStatus::Completed(_))),
            "a straddler hung on the settled-transaction fence; status = {status:?}",
        );
    }

    StraddlerRun {
        probes,
        splitter,
        terminal_b,
        world,
        charges,
    }
}

/// Submit the encumbered payer's second transaction beside an unencumbered
/// payer's, and assert the splitter takes one and refuses the other.
///
/// The control is what makes the refusal mean something: submitted into the
/// same shard at the same instant, it separates "this shard is refusing
/// everything" — which it does once its split gates and it coasts on empty
/// blocks — from "this shard is refusing this payer".
///
/// # Panics
///
/// Panics if the control never commits or the encumbered probe does.
fn assert_payer_is_blocked<C: FaultableCluster>(
    c: &mut C,
    charges: &mut Charges,
    splitter: ShardId,
    setup: &SplitStraddlerSetup,
) {
    let (payer_key, payer, _) = &setup.terminating;
    let blocked = build_transfer_tx(
        payer_key,
        *payer,
        setup.successor_recipient,
        STRADDLER_PAYMENT,
        validity_around(c.now()),
    );
    let blocked_hash = charges.submit(c, blocked);

    let (control_key, control_payer) = &setup.control;
    let control = build_transfer_tx(
        control_key,
        *control_payer,
        setup.successor_recipient,
        STRADDLER_PAYMENT,
        validity_around(c.now()),
    );
    let control_hash = charges.submit(c, control);

    assert!(
        c.run_until(epochs(6), |c| c
            .chain_fate(splitter, control_hash)
            .0
            .is_some()),
        "an unencumbered payer's transaction must still commit on the splitter, \
         or the refusal below proves nothing about the payer",
    );
    assert!(
        c.chain_fate(splitter, blocked_hash).0.is_none(),
        "the splitter must refuse the payer while its first transaction still \
         holds that payer's vault",
    );
}

/// The terminating payer, its recipient, the control and the recipient
/// the successor is asked to pay: everything a transfer in
/// [`split_terminating_payer_releases_its_reservation`] reaches.
fn terminating_payer_world<C: Cluster>(c: &C, setup: &SplitStraddlerSetup) -> World {
    let (_, payer, recipient) = &setup.terminating;
    World::open(
        c,
        *XRD,
        [
            payer.address(),
            recipient.address(),
            setup.control.1.address(),
            setup.successor_recipient.address(),
        ],
        [],
    )
}

/// Verify a payer whose shard terminates mid-flight is not stranded by it.
///
/// The one straddler shape the other scenarios never reach: their payers all
/// live on the survivor, so the shard that dies is never the one holding the
/// in-flight state. Here the payer is ground into the splitter's left child,
/// so the splitter commits its transaction, engages `max_fee` against the
/// payer's vault and takes the conflict lock on it, and then terminates.
///
/// [`isolate_ec_intake`] cuts every path by which the splitter obtains the
/// survivor's execution certificate, which is what keeps that state standing.
/// The payer's deadline still arrives and it still speaks its abort, so the
/// transaction goes terminal — but a tick with an engaged counterpart needs
/// that counterpart's certificate to finalize, and both the reservation's
/// release and the settlement that would clear the lock key on a *finalized
/// tick*, not on a verdict.
///
/// What the middle of this scenario measures is that the payer really is
/// blocked, and that being blocked is about the payer rather than about the
/// moment: a second transaction from it is refused while an unencumbered
/// payer's, submitted into the same shard at the same instant, commits. It
/// does not separate the reservation from the conflict lock, and cannot — a
/// call transaction's withdraw names the very vault its fee burns from, so one
/// payer's two transactions contend on one cell either way. Both are per-shard
/// state, and the claim here is about all of it at once.
///
/// Then the shard terminates and the same shape is admitted by the successor.
/// Neither the ledger nor the lock set is carried across: both are projections
/// of a shard's own committed chain, and the successor's chain begins at its
/// seeded genesis. The release is by construction rather than by an explicit
/// sweep, which is what this pins.
///
/// The payer is funded above one fee ceiling and below two, so an inherited
/// reservation would refuse the successor's probe on its own.
///
/// Requires disjoint splitter/survivor committees (no shared host), as
/// [`split_straddler_ec_partition_atomic`] does, and the
/// [`split_straddler_setup`] genesis funding.
///
/// # Panics
///
/// Panics if the splitter never commits the transaction, the control is
/// refused, the encumbered probe is admitted anyway, the transaction never goes
/// terminal or finalizes despite the cut, the split misses its budget, the
/// transaction applies on either chain, the payer's vault moves, or the
/// successor refuses the payer's next transaction.
pub fn split_terminating_payer_releases_its_reservation(c: &mut impl FaultableCluster) {
    let splitter = STRADDLER_SPLITTER;
    let survivor = STRADDLER_SURVIVOR;
    let successor = STRADDLER_SUCCESSOR;
    let setup = split_straddler_setup();
    let (payer_key, payer, recipient) = &setup.terminating;

    arm_splitter_termination(c);

    // Cut the splitter's execution-certificate intake before anything crosses:
    // it will execute its own leg and speak its own verdict, and never hold the
    // survivor's half, so no tick of its can finalize.
    let _ = isolate_ec_intake(c, splitter, survivor);

    let world = terminating_payer_world(c, &setup);
    let mut charges = Charges::default();

    let held = build_transfer_tx(
        payer_key,
        *payer,
        *recipient,
        STRADDLER_PAYMENT,
        validity_around(c.now()),
    );
    let held_hash = charges.submit(c, held);

    // The reservation engages when the splitter commits the transaction —
    // before the split executes, so the shard that dies is the one holding it.
    assert!(
        c.run_until(epochs(10), |c| c
            .chain_fate(splitter, held_hash)
            .0
            .is_some()),
        "the splitter must commit the transaction and engage its reservation \
         before it terminates",
    );
    assert_eq!(
        vault_balance(c, splitter, *payer),
        TERMINATING_PAYER_FUNDING,
        "the reservation must not move the payer's vault: it is an engagement, \
         not an on-chain hold",
    );

    assert_payer_is_blocked(c, &mut charges, splitter, &setup);

    // The payer's deadline arrives and it speaks its abort. No certificate
    // follows it: the counterpart engaged, so finalization needs a certificate
    // the cut will never deliver — and release keys on a finalization, not on
    // a verdict, so the hold the probes just measured stands through the
    // transaction's own resolution and on to the terminal.
    let verdict = await_tx_terminal(c, held_hash, epochs(12));
    assert!(
        matches!(
            verdict,
            Some(TransactionStatus::Completed(TransactionDecision::Aborted))
        ),
        "the payer must abort at its deadline once no engagement can settle; \
         status = {verdict:?}",
    );
    assert!(
        c.chain_fate(splitter, held_hash).1.is_none(),
        "nothing may finalize while the counterpart's certificate is cut off — \
         without that, the reservation would release and the probe below would \
         prove nothing",
    );

    // The split executes and the successor seats with the payer's cells.
    let (child_left, child_right) = splitter.children();
    assert!(
        await_serves(c, child_left, epochs(28)) && await_serves(c, child_right, epochs(28)),
        "both splitter children must be served within budget",
    );
    assert!(
        await_anchor_seeded(c, successor, epochs(6)),
        "the beacon must compose the successor's anchor",
    );

    // Nothing applied on either side, and nothing was charged: an abort with no
    // certificate commits no fee receipt, and a receipt is the only thing that
    // moves state.
    for (shard, label) in [(splitter, "splitter"), (survivor, "survivor")] {
        let fate = c.chain_fate(shard, held_hash).1.map(|(_, d)| d);
        assert!(
            fate != Some(TransactionDecision::Accept),
            "the {label} applied a transaction its counterpart never settled; fate = {fate:?}",
        );
    }
    // Nothing moved. The transfer aborted with no certificate behind it, so it
    // settled no fee receipt, and a receipt is the only thing that moves state;
    // the refused probe was never included anywhere at all.
    assert_eq!(
        vault_balance(c, successor, *payer),
        TERMINATING_PAYER_FUNDING,
        "the payer's vault must carry across the split untouched",
    );

    // The release: the same shape the predecessor refused is admitted by the
    // successor. Its ledger is a projection of its own committed chain, which
    // begins at the seeded genesis, so there is no hold left to count against
    // the payer — the reservation died with the shard that held it, without
    // anything having to sweep it.
    let released = build_transfer_tx(
        payer_key,
        *payer,
        setup.successor_recipient,
        STRADDLER_PAYMENT,
        validity_around(c.now()),
    );
    let released_hash = charges.submit(c, released);
    let status = await_tx_terminal(c, released_hash, epochs(10));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "the successor must admit what its predecessor refused — no shard's \
         in-flight state outlives the shard; status = {status:?}",
    );
    // A terminal status is reported when this shard decides the outcome,
    // which is before the finalization carrying its writes commits — so
    // the spend has to be waited for rather than read at the instant the
    // status flips. Waiting on any movement keeps the amount unasserted
    // until the check below.
    assert!(
        c.run_until(epochs(8), |c| vault_balance(c, successor, *payer)
            < TERMINATING_PAYER_FUNDING),
        "the released transaction must actually have spent from the payer's vault",
    );

    // The abort moved nothing and charged nothing, the blocked probe was
    // never included, and the control and the release each moved their
    // payment once and paid once.
    world.assert_settles_within(
        c,
        &charges,
        epochs(8),
        "a terminating payer's abort and the successor's release",
    );
}

/// Verify a survivor whose counterpart terminates mid-flight releases what
/// it reserved.
///
/// The mirror of [`split_terminating_payer_releases_its_reservation`], which
/// asserts only about the shard that dies. There the release is by
/// construction — a terminated shard's ledger dies with its chain. Here the
/// shard lives, so nothing tears the reservation down for it: the survivor
/// holds a tick that executed its own leg, speaks its own verdict, and waits
/// on a certificate the counterpart will never send.
///
/// [`isolate_ec_intake`] runs in *both* directions, which is what makes the
/// straddler stranded rather than merely fenced: neither side holds the
/// other's certificate, the splitter reaches its terminal having settled
/// nothing, and abandoning is the only outcome the straddler can reach.
/// Cutting one direction only is
/// [`split_survivor_recovers_a_settlement_it_never_received`].
///
/// The drain is what this measures. A verdict alone would pass without it:
/// the reservation releases on a *committed finalization* carrying the work
/// figure, so a shard that reported an outcome and never certified one still
/// holds the level. Baseline is read before submission and compared after,
/// rather than against zero, because the choreography's own traffic — the
/// threshold vote and the grow — is in flight on the same shard.
///
/// # Panics
///
/// Panics if the choreography misses its budget, either shard fails to commit
/// the straddler while both are live, the reservation never engages, the
/// survivor applies the straddler one-sided, it reaches no terminal outcome,
/// or the survivor's drain never returns to its baseline.
pub fn split_surviving_counterpart_releases_its_reservation(c: &mut impl FaultableCluster) {
    let splitter = STRADDLER_SPLITTER;
    let survivor = STRADDLER_SURVIVOR;
    let setup = split_straddler_setup();

    arm_splitter_termination(c);

    // Neither side may hold the other's certificate. Provisions and headers
    // still flow, so each shard commits the straddler and executes its own
    // leg — which is the state under test, and the reason this cannot be
    // read off a shard that never composed a tick for it.
    let _ = isolate_ec_intake(c, survivor, splitter);
    let _ = isolate_ec_intake(c, splitter, survivor);

    let baseline = c
        .committed_work_in_flight(survivor)
        .expect("the survivor must serve a committed tip before the straddler");

    let world = straddler_world(c, &setup.straddlers[..1]);
    let mut charges = Charges::default();
    let (key, from, to) = &setup.straddlers[0];
    let (hash, reserved) = submit_straddler_reserving(c, &mut charges, key, *from, *to);

    // Both while both are live: the splitter's coast blocks commit nothing,
    // so a straddler landing later would leave the survivor with no tick at
    // all — released by the deadline path without the counterpart's terminal
    // mattering, which is the vacuous version of this scenario.
    assert!(
        c.run_until(epochs(12), |c| c.chain_fate(survivor, hash).0.is_some()
            && c.chain_fate(splitter, hash).0.is_some()),
        "both shards must commit the straddler while both are live",
    );
    let engaged = c
        .committed_work_in_flight(survivor)
        .expect("the survivor must serve a committed tip once it holds the straddler");
    assert!(
        engaged > baseline,
        "the straddler must engage a reservation against the survivor's drain, \
         or its release below proves nothing; baseline = {baseline}, engaged = {engaged}",
    );

    // The splitter terminates and its children seat.
    let (child_left, child_right) = splitter.children();
    assert!(
        await_serves(c, child_left, epochs(28)) && await_serves(c, child_right, epochs(28)),
        "both splitter children must be served within budget",
    );
    assert!(
        await_anchor_seeded(c, child_left, epochs(6)),
        "the beacon must compose the split children's anchor",
    );

    // Nothing applied on the survivor: the splitter settled no straddler, so
    // applying this one would be the one-sided settlement the fence exists to
    // prevent, and it is what the release below must not have cost.
    assert!(
        !chain_settled(c, survivor, hash),
        "the survivor applied a straddler the splitter never settled",
    );

    let status = await_tx_terminal(c, hash, epochs(12));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Aborted))
        ),
        "a straddler no shard can settle must abort rather than hang; status = {status:?}",
    );

    // The release. It rides a committed finalization, which lands a block or
    // more after the outcome is reported, so it is waited for rather than
    // read at the instant the status flips.
    //
    // Two conditions, because either alone can be satisfied by something
    // other than the release. Returning to the baseline is the property
    // `MAX_DRAIN_WORK` documents, but the choreography's own traffic is
    // in flight at the moment the baseline is read, and its later
    // settlement lowers the level whatever the straddler does. So the
    // straddler's own reservation is named as well: the level has to fall
    // by at least what this transaction declared, which no other
    // transaction settling can supply.
    assert!(
        c.run_until(epochs(12), |c| c
            .committed_work_in_flight(survivor)
            .is_some_and(
                |level| level <= baseline && level.saturating_add(reserved) <= engaged
            )),
        "the survivor's drain must return to its baseline once the straddler is \
         abandoned, and must fall by the {reserved} the straddler reserved; \
         baseline = {baseline}, engaged = {engaged}, still owing = {:?}",
        c.committed_work_in_flight(survivor),
    );

    // Abandoned, so nothing moved — and the price was still paid.
    world.assert_settles_within(c, &charges, epochs(8), "an abandoned straddler");
}

/// Verify a straddler the departing splitter settled applies on both sides,
/// when the survivor never receives the certificate that settled it.
///
/// The mirror of [`split_surviving_counterpart_releases_its_reservation`]:
/// [`isolate_ec_intake`] cuts only the survivor's intake, so the splitter
/// holds both certificates and settles, applying its half, while the
/// survivor holds only its own and cannot apply until the splitter's
/// reaches it.
///
/// What that leaves the survivor is a transaction its counterpart's settled
/// set names as settled — so no record covers it, the fence refuses any
/// abandonment of it, and the only resolution left is the certificate
/// itself. The certificate is committed on the splitter's tail chain, which
/// is the same chain the settled set is reconstructed from; whether the
/// survivor recovers it from there is what this measures.
///
/// The cut lifts once the splitter's children seat, so what is measured is
/// a retention limit and not a partition: the survivor asks for the
/// certificate on a whole network, having missed it while the cut was up.
///
/// Requires disjoint splitter/survivor committees (no shared host), or a
/// co-hosted vnode bridges the certificate across in-process, which no
/// network rule intercepts.
///
/// # Panics
///
/// Panics if the choreography misses its budget, either shard fails to
/// commit the straddler while both are live, the splitter fails to settle
/// it, the survivor never applies what the splitter settled, or the
/// survivor's drain never returns to its baseline.
pub fn split_survivor_recovers_a_settlement_it_never_received(c: &mut impl FaultableCluster) {
    let splitter = STRADDLER_SPLITTER;
    let survivor = STRADDLER_SURVIVOR;
    let setup = split_straddler_setup();

    arm_splitter_termination(c);

    // One direction only. The splitter still receives the survivor's
    // certificate and so can settle; the survivor never receives the
    // splitter's and so cannot.
    let _ = isolate_ec_intake(c, survivor, splitter);

    let baseline = c
        .committed_work_in_flight(survivor)
        .expect("the survivor must serve a committed tip before the straddler");

    let world = straddler_world(c, &setup.straddlers[..1]);
    let mut charges = Charges::default();
    let (key, from, to) = &setup.straddlers[0];
    let hash = submit_straddler(c, &mut charges, key, *from, *to);

    assert!(
        c.run_until(epochs(12), |c| c.chain_fate(survivor, hash).0.is_some()
            && c.chain_fate(splitter, hash).0.is_some()),
        "both shards must commit the straddler while both are live",
    );

    // The premise. Without it this is the stranded scenario wearing a
    // different name, and what follows would prove nothing.
    assert!(
        c.run_until(epochs(10), |c| chain_settled(c, splitter, hash)),
        "the splitter holds both certificates and must settle the straddler,          or nothing here is one-sided to begin with",
    );

    // The splitter terminates and its children seat.
    let (child_left, child_right) = splitter.children();
    assert!(
        await_serves(c, child_left, epochs(28)) && await_serves(c, child_right, epochs(28)),
        "both splitter children must be served within budget",
    );
    assert!(
        await_anchor_seeded(c, child_left, epochs(6)),
        "the beacon must compose the split children's anchor",
    );

    // Whole network from here. Anything the survivor still cannot obtain is
    // something no longer being kept, rather than something it cannot reach.
    c.clear_drops();

    // The measurement. The splitter applied its half before it went; a
    // transaction applied on one side and not the other is torn, whatever
    // the survivor's reason for never holding the certificate.
    assert!(
        c.run_until(epochs(16), |c| chain_settled(c, survivor, hash)),
        "the splitter settled the straddler and the survivor never did — applied \
         on one side only; survivor fate = {:?}",
        c.chain_fate(survivor, hash).1,
    );

    // And the reservation goes with the resolution, on the same mechanism
    // every other verdict returns it by.
    assert!(
        c.run_until(epochs(12), |c| c
            .committed_work_in_flight(survivor)
            .is_some_and(|level| level <= baseline)),
        "the survivor's drain must return to its baseline once the straddler \
         resolves; baseline = {baseline}, still owing = {:?}",
        c.committed_work_in_flight(survivor),
    );

    // Settled on both sides: the payment landed once and the price once.
    world.assert_settles_within(
        c,
        &charges,
        epochs(8),
        "a straddler settled off the tail chain",
    );
}

/// Verify a surviving sibling's second-generation split seats correctly.
///
/// Composes [`split_straddler_atomic`] (grow → vote the threshold down so only
/// the splitter crosses → settled-transaction fence), then layers the seating outcome:
/// the splitter retires into two full-strength child committees while the survivor
/// keeps its own, each child's committed root reproduces the beacon-composed
/// anchor, and both children commit a real block past their seeded genesis.
/// Requires the [`split_straddler_setup`] genesis funding on a config grown from a
/// single root.
///
/// # Panics
///
/// Panics if the lifecycle misses its budget, a committee is under strength, the
/// splitter fails to retire, a child root diverges from the anchor, or a child
/// stalls at its seeded genesis.
pub fn surviving_sibling_split_seats_full_committees(c: &mut impl Cluster) {
    assert!(
        await_beacon_epoch(c, 1, epochs(6)),
        "the beacon must fold before the grow so the genesis committee strength is known",
    );
    let strength = committee_size(c, ShardId::ROOT).expect("genesis seats the root committee");

    split_straddler_atomic(c);

    let splitter = STRADDLER_SPLITTER;
    let survivor = STRADDLER_SURVIVOR;
    let (child_left, child_right) = splitter.children();
    assert!(
        c.run_until(epochs(6), |c| committee_size(c, survivor) == Some(strength)
            && committee_size(c, child_left) == Some(strength)
            && committee_size(c, child_right) == Some(strength)
            && committee_size(c, splitter).is_none()),
        "the survivor and both splitter children must seat full committees of {strength}, and the splitter must retire",
    );

    assert!(
        await_root_matches_anchor(c, child_left, epochs(8))
            && await_root_matches_anchor(c, child_right, epochs(8)),
        "both splitter children's roots must reproduce the beacon anchor",
    );

    // A committed-height probe can transiently read `None` while a vnode's
    // serving surface hands over (the anchor probe above may have sampled a
    // host that has since rotated), so wait the heights back into view
    // before taking the bases.
    assert!(
        c.run_until(epochs(2), |c| c.committed_height(child_left).is_some()
            && c.committed_height(child_right).is_some()),
        "both splitter children must report a committed height",
    );
    let left_base = c
        .committed_height(child_left)
        .expect("the left child commits");
    let right_base = c
        .committed_height(child_right)
        .expect("the right child commits");
    assert!(
        c.run_until(epochs(6), |c| c
            .committed_height(child_left)
            .is_some_and(|h| h > left_base)
            && c.committed_height(child_right)
                .is_some_and(|h| h > right_base)),
        "both splitter children must keep committing past their seeded genesis",
    );
}

/// Verify a merge straddler settles atomically across the reshape boundary.
///
/// The cluster grows into four shards (the caller's `with_grown_balances`), then
/// the lighter `leaf(2, 2)`/`leaf(2, 3)` pair — funded below the derived merge
/// threshold — collapses into `leaf(1, 1)`, while the bulk-funded survivors
/// `leaf(2, 0)`/`leaf(2, 1)` stay above it and keep the left half alive: once the
/// topology is grown, the merge fires from the byte skew alone. Cross-shard
/// transfers run from the survivor `leaf(2, 0)` into the merging `leaf(2, 2)`, so
/// each tick names a shard that terminates at the merge. The first tick settles
/// before `leaf(2, 2)`'s terminal block; the second straddles it, in flight when
/// it terminates. After the merge the survivor must reach a terminal verdict on
/// every straddler, consistent with what `leaf(2, 2)` settled by its terminal
/// block — never one-sided, never contradicting a settlement, never hanging.
/// Exercises the merge-child terminal's settled-transaction attestation, the path a
/// split child's terminal cannot cover. Requires the [`merge_straddler_setup`]
/// funding on a config grown to four shards.
///
/// # Panics
///
/// Panics if the merge misses its budget, the merged parent never seats, or the
/// settled-transaction fence is breached (a one-sided application, a mismatch, or a
/// hung straddler).
pub fn merge_straddler_atomic(c: &mut impl Cluster) {
    let survivor = MERGE_STRADDLER_SURVIVOR;
    let merge_left = MERGE_STRADDLER_LEFT;
    let merge_right = MERGE_STRADDLER_RIGHT;
    let merge_parent = merge_left.parent().expect("a depth-2 leaf has a parent");
    let setup = merge_straddler_setup();

    // The cluster reaches this body grown to four shards; confirm every quarter
    // is seated and serving before driving the merge.
    assert!(
        await_serves(c, survivor, epochs(4))
            && await_serves(c, merge_left, epochs(4))
            && await_serves(c, merge_right, epochs(4))
            && await_serves(c, ShardId::leaf(2, 3), epochs(4)),
        "the grown four-shard topology must seat every quarter",
    );

    let world = straddler_world(c, &setup.straddlers);
    let mut charges = Charges::default();
    let mut probes: Vec<TxHash> = Vec::new();
    let half = setup.straddlers.len() / 2;

    // Settling ticks: submitted while `leaf(2, 0)` still commits real blocks, so
    // their cross-shard settlement can finalize at or below its terminal block
    // and land in the attested settled set. Submitted before the keeper
    // pairing arms the gate, then awaited to finalize on `leaf(2, 0)` so
    // settlement can't lose the race to the cut — a settler only needs to
    // finalize before the terminal.
    let settling: Vec<TxHash> = setup
        .straddlers
        .iter()
        .take(half)
        .map(|(key, from, to)| submit_straddler(c, &mut charges, key, *from, *to))
        .collect();
    probes.extend_from_slice(&settling);
    assert!(
        c.run_until(epochs(12), |c| settling
            .iter()
            .all(|hash| chain_settled(c, merge_left, *hash))),
        "the settling ticks must finalize on the merging child before its terminal",
    );

    // The light merging pair asserts the merge from its genesis byte skew; the
    // beacon pairs it and draws a keeper quorum (2f+1 of the four-validator
    // reformed committee). The heavy survivor pair never pairs.
    assert!(
        await_merge_keeper_count(c, merge_parent, 3, epochs(24)),
        "the light merging pair must pair a keeper quorum within budget",
    );

    // Straddling ticks: submitted once the merge has paired and `leaf(2, 0)` is
    // coasting to its terminal — the survivor still provisions to it, but its
    // coast blocks settle nothing, leaving them in flight when it terminates.
    for (key, from, to) in setup.straddlers.iter().skip(half) {
        probes.push(submit_straddler(c, &mut charges, key, *from, *to));
    }

    // Drive the merge to fire: the keepers' ready signals collapse the children
    // into `leaf(1, 1)`, seating it in the lookahead. Gate on the reformed parent
    // appearing in the lookahead — not merely on the pending record clearing — so
    // a pairing that lapses and re-pairs under the seeded schedule isn't read as
    // the gate.
    assert!(
        c.run_until(epochs(16), |c| merge_executed(c, merge_parent)),
        "the merge must gate within budget",
    );

    // The merge executes: the reformed parent seats and commits past genesis.
    assert!(
        await_serves(c, merge_parent, epochs(28)),
        "the merged parent must be served within budget",
    );

    // The merged parent's composed boundary records its seeded genesis height,
    // folded from both children's terminals after the gate seats the placeholder
    // at `GENESIS`. Wait for the composed height (a real genesis above `GENESIS`,
    // so its predecessor — the merging child's terminal — exists).
    assert!(
        c.run_until(epochs(12), |c| merged_genesis_height(c, merge_parent)
            .and_then(BlockHeight::prev)
            .is_some()),
        "the merged parent's composed boundary must fold within budget",
    );

    // The merging child's terminal block sits one below the merged genesis.
    let terminal_b = merged_genesis_height(c, merge_parent)
        .and_then(BlockHeight::prev)
        .expect("the merged seeded genesis pins the merging child's terminal block");

    // Every straddler must reach a terminal verdict on the survivor.
    for hash in &probes {
        let status = await_tx_terminal(c, *hash, epochs(12));
        assert!(
            matches!(status, Some(TransactionStatus::Completed(_))),
            "a straddler hung on the settled-transaction fence; status = {status:?}",
        );
    }

    assert_fence_held(c, merge_left, terminal_b, &probes);
    world.assert_settles_within(c, &charges, epochs(8), "straddlers across a merge");
}

/// Whether the merge into `parent` has executed: the reformed parent is seated
/// in the lookahead committee set and no longer pending.
fn merge_executed<C: Cluster>(c: &C, parent: ShardId) -> bool {
    c.beacon_state().is_some_and(|state| {
        !state.pending_reshapes.contains_key(&parent)
            && state.next_shard_committees.contains_key(&parent)
    })
}

/// The merged parent's seeded genesis height, from its composed boundary.
fn merged_genesis_height<C: Cluster>(c: &C, parent: ShardId) -> Option<BlockHeight> {
    c.beacon_state()
        .and_then(|state| state.boundaries.get(&parent).map(|b| b.height))
}

/// Whether `hash` finalized a non-abort decision on `shard`'s committed chain —
/// the source side of a cross-shard tick settling before a reshape terminal.
pub fn chain_settled<C: Cluster>(c: &C, shard: ShardId, hash: TxHash) -> bool {
    matches!(
        c.chain_fate(shard, hash).1,
        Some((_, decision)) if decision != TransactionDecision::Aborted
    )
}

/// The payment every straddler leg carries.
pub const STRADDLER_PAYMENT: u128 = 100;

/// Build a straddler transfer (payer → counterpart-shard recipient) bracketing
/// the current clock, submit it, and return its hash.
///
/// Each leg draws its own payer and recipient, so no two straddlers share
/// signed content — hash dedup would otherwise read them as one transaction.
pub fn submit_straddler<C: Cluster>(
    c: &mut C,
    charges: &mut Charges,
    key: &Ed25519PrivateKey,
    from: PrincipalAddr,
    to: PrincipalAddr,
) -> TxHash {
    submit_straddler_reserving(c, charges, key, from, to).0
}

/// [`submit_straddler`], also reporting what the transaction reserves
/// against its shards' drains.
///
/// The figure a committed abandonment returns exactly, so a scenario
/// measuring the release can name the straddler's own contribution rather
/// than inferring it from a level other traffic also moves.
fn submit_straddler_reserving<C: Cluster>(
    c: &mut C,
    charges: &mut Charges,
    key: &Ed25519PrivateKey,
    from: PrincipalAddr,
    to: PrincipalAddr,
) -> (TxHash, u64) {
    let tx = build_transfer_tx(key, from, to, STRADDLER_PAYMENT, validity_around(c.now()));
    tx.try_derived(c.derivation().as_ref())
        .expect("a scenario transfer derives");
    let work = tx.work();
    let hash = charges.submit(c, tx);
    (hash, work)
}

/// Assert the settled-transaction fence held for `probes`: every straddler the
/// survivor reached agrees with what the splitter settled by `terminal_b`, none
/// applied one-sided or contradicted a settlement, and at least one settled
/// atomically.
fn assert_fence_held<C: Cluster>(
    c: &C,
    splitter: ShardId,
    terminal_b: BlockHeight,
    probes: &[TxHash],
) {
    let tally = straddler_tally(c, splitter, terminal_b, probes);

    assert_eq!(
        tally.one_sided, 0,
        "the survivor applied a straddler the splitter never settled — one-sided:{}",
        tally.report,
    );
    assert_eq!(
        tally.mismatch, 0,
        "the survivor's verdict contradicted the splitter's settlement:{}",
        tally.report,
    );
    assert!(
        tally.consistent > 0,
        "no straddler settled atomically — submission timing needs retuning:{}",
        tally.report,
    );
}

/// How each straddler resolved on the survivor versus what the splitter settled
/// by its terminal block.
struct StraddlerTally {
    /// Survivor verdict matched the splitter's settlement.
    consistent: u32,
    /// Splitter never settled it; survivor correctly aborted.
    doomed: u32,
    /// Survivor applied a decision the splitter never settled — a broken fence.
    one_sided: u32,
    /// Survivor's verdict contradicted the splitter's settlement.
    mismatch: u32,
    /// Per-probe detail for assertion messages.
    report: String,
}

fn straddler_tally<C: Cluster>(
    c: &C,
    splitter: ShardId,
    terminal_b: BlockHeight,
    probes: &[TxHash],
) -> StraddlerTally {
    let mut tally = StraddlerTally {
        consistent: 0,
        doomed: 0,
        one_sided: 0,
        mismatch: 0,
        report: String::new(),
    };

    for (idx, hash) in probes.iter().enumerate() {
        let (_, splitter_final) = c.chain_fate(splitter, *hash);
        // The splitter settled it iff it finalized a non-abort decision at or
        // before its terminal block.
        let settled = splitter_final
            .and_then(|(h, d)| (h <= terminal_b && d != TransactionDecision::Aborted).then_some(d));
        let verdict = match c.tx_status(*hash) {
            Some(TransactionStatus::Completed(d)) => Some(d),
            _ => None,
        };
        let _ = write!(
            tally.report,
            "\n  #{idx}: splitter settled={settled:?}; survivor verdict={verdict:?}",
        );
        match (settled, verdict) {
            (Some(t), Some(v)) if t == v => tally.consistent += 1,
            (Some(_), Some(_)) => tally.mismatch += 1,
            (None, Some(TransactionDecision::Aborted)) => tally.doomed += 1, // correctly aborted
            (None, Some(_)) => tally.one_sided += 1,
            (_, None) => {} // unresolved — the abandonment gate caught it
        }
    }
    let _ = write!(
        tally.report,
        "\n  consistent={} doomed={}",
        tally.consistent, tally.doomed,
    );
    tally
}

/// The number of straddlers the survivor applied one-sided.
///
/// A one-sided straddler is one the survivor finalized on a decision the
/// splitter never settled by its terminal block. Zero when the fence holds; a
/// probe that cuts the survivor→splitter EC channel across the boundary watches
/// whether it goes positive.
#[must_use]
pub fn straddler_one_sided_count<C: Cluster>(
    c: &C,
    splitter: ShardId,
    terminal_b: BlockHeight,
    probes: &[TxHash],
) -> u32 {
    straddler_tally(c, splitter, terminal_b, probes).one_sided
}
