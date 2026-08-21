//! The single-shard catalogue on the engine.
//!
//! Every scenario drives signed manifest graphs — the account guest's
//! withdraw+deposit — through the live pipeline: gossip, derived-key
//! admission, proposal, tick execution on the batch executor,
//! receipts, commit. The bodies are portable over [`Cluster`]; the
//! kernel-level invariants (handle capabilities, snapshot semantics,
//! schedule invariance) are pinned in the vm repo's differential suite —
//! here the assertions are consensus-shaped: acceptance, deterministic
//! aborts, ordering, and committed state roots.

use std::collections::BTreeSet;
use std::sync::Arc;

use hyperscale_effects_bridge::ProtocolHasher;
use hyperscale_effects_bridge::vm_statics::package_key;
use hyperscale_engine::genesis::{draw_key, vault_key};
use hyperscale_engine::{
    PreviewGrants, PreviewOutcome, PreviewReport, ResourceChange, XRD, account_address,
};
use hyperscale_types::{
    AccountSigner, Address, BlockHeight, Hash, SchemeId, ShardId, TransactionDecision,
    TransactionStatus, TxHash,
};
use hyperscale_vm_effects::{InstanceMeta, package_hash};

use crate::contention::{ContentionReport, Lcg, settle_and_report, zipf_cdf};
use crate::support::faultable::FaultableCluster;
use crate::support::tx::{
    OVERDRAW_AMOUNT, build_composed_tx, build_draw_tx, build_instance_deposit_tx,
    build_instantiate_tx, build_publish_tx, build_securify_tx, build_transfer_paid_by,
    build_transfer_tx, build_unbound_payer_tx, cross_shard_cast, cross_shard_keys, lottery_on,
    native_pq_cast, nullifier_race_cast, overdraw_cast, payment_request, recipient, securify_cast,
    sender, shared_recipient_cast, storm_artifact, storm_publishers, unbound_payer_cast,
    unbound_remote_payer_cast, validity_around,
};
use crate::support::wait::{await_height, await_tx_terminal};
use crate::support::{Cluster, epochs};

/// Per-payment amount of the contention scenarios.
const PAYMENT: u128 = 5;

/// The payment a nullifier race contends over.
const REQUEST: u128 = 100;

/// Two compositions carrying one signed subintent: exactly one commits.
///
/// The request is a declaration and nothing else — its hash covers its
/// own graph and parameters, no envelope — so both composers bind the
/// identical one and both derive the same nullifier key under its
/// signer's prefix. That shared declared write is what puts them in one
/// conflict group, where the spent check sees the winner's cell.
///
/// The loser is charged as a lost race rather than as a defect: canonical
/// order picked the winner, and nothing a composer could read at signing
/// time would have told it which way.
///
/// # Panics
///
/// Panics if both compositions settle the same way, if the request is
/// filled more than once, or if either payer is charged the wrong class.
pub fn nullifier_race_admits_exactly_one(c: &mut impl Cluster) {
    let shard = ShardId::ROOT;
    let (first_key, second_key, requester_key) = nullifier_race_cast();
    let first = account_address(&first_key.public_key().0);
    let second = account_address(&second_key.public_key().0);
    let requester = account_address(&requester_key.public_key().0);

    let before = [
        vault_balance(c, shard, first),
        vault_balance(c, shard, second),
        vault_balance(c, shard, requester),
    ];

    let request = payment_request(requester, REQUEST);
    let window = validity_around(c.now());
    let mut hashes = Vec::new();
    let mut floors = Vec::new();
    for (composer, from) in [(&first_key, first), (&second_key, second)] {
        let tx = build_composed_tx(composer, from, &requester_key, &request, REQUEST, window);
        floors.push(tx.body().abort_floor());
        hashes.push(tx.hash());
        c.submit(Arc::new(tx));
    }

    let verdicts: Vec<Option<TransactionStatus>> = hashes
        .iter()
        .map(|hash| await_tx_terminal(c, *hash, epochs(8)))
        .collect();
    let accepted = verdicts
        .iter()
        .filter(|status| {
            matches!(
                status,
                Some(TransactionStatus::Completed(TransactionDecision::Accept))
            )
        })
        .count();
    assert_eq!(
        accepted, 1,
        "exactly one composition may fill a once-only request; verdicts = {verdicts:?}"
    );

    // The request was filled once, so the requester banked one payment.
    let after_requester = vault_balance(c, shard, requester);
    assert_eq!(
        after_requester - before[2],
        REQUEST,
        "the request must be filled exactly once"
    );

    // The winner paid the payment and its fee; the loser paid the class
    // floor and nothing more.
    let won = matches!(
        verdicts[0],
        Some(TransactionStatus::Completed(TransactionDecision::Accept))
    );
    let (winner_spent, loser_spent) = if won {
        (
            before[0] - vault_balance(c, shard, first),
            before[1] - vault_balance(c, shard, second),
        )
    } else {
        (
            before[1] - vault_balance(c, shard, second),
            before[0] - vault_balance(c, shard, first),
        )
    };
    assert!(
        winner_spent > REQUEST,
        "the winner paid the request plus a fee; spent = {winner_spent}"
    );
    assert_eq!(
        loser_spent, floors[0],
        "a lost race settles the class floor, not the ceiling"
    );
}

/// Submit one transfer between genesis-funded accounts and assert
/// it accepts and lands state.
///
/// The committed state root must move off its pre-submission value: the
/// transfer's identity-keyed vault cells entered the shard's JMT on
/// every replica, or the commit could not have certified.
///
/// # Panics
///
/// Panics if the transfer does not accept within budget, the root shard
/// does not advance, or the state root does not move.
pub fn single_transfer(c: &mut impl Cluster) {
    let (payer, from) = sender(0);
    let to = recipient(0);
    let before = c.committed_state_root(ShardId::ROOT);
    let transfer = build_transfer_tx(&payer, from, to, 100, validity_around(c.now()));
    let hash = transfer.hash();
    c.submit(Arc::new(transfer));

    let status = await_tx_terminal(c, hash, epochs(8));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "VM transfer did not accept within budget; status = {status:?}"
    );
    assert!(
        await_height(c, ShardId::ROOT, 1, epochs(2)),
        "root shard did not advance past genesis"
    );
    let after = c.committed_state_root(ShardId::ROOT);
    assert!(
        after.is_some() && after != before,
        "the committed state root must reflect the transfer's vault cells; \
         before = {before:?}, after = {after:?}"
    );
}

/// An uncovered withdrawal aborts deterministically on every replica
/// and the chain carries on.
///
/// Every replica reaches the same verdict from the same committed
/// state, so a failure is consensus content like any success.
///
/// The over-withdrawal's reservation is infeasible against committed
/// state, so every replica derives the identical `Failed` receipt and
/// the block certifies; a covered transfer from the same payer then
/// accepts, showing the abort wedged nothing. (The kernel half — an
/// undeclared substate has no handle, a forged handle traps — is pinned
/// by the vm repo's differential corpus.)
///
/// # Panics
///
/// Panics if the over-withdrawal does not reject, or the follow-up does
/// not accept.
pub fn abort_converges(c: &mut impl Cluster) {
    let (payer, from) = sender(0);
    let to = recipient(0);

    let over = build_transfer_tx(&payer, from, to, 1_000_000, validity_around(c.now()));
    let over_hash = over.hash();
    c.submit(Arc::new(over));
    let status = await_tx_terminal(c, over_hash, epochs(8));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Reject))
        ),
        "an uncovered VM withdrawal must reject deterministically; status = {status:?}"
    );

    let fine = build_transfer_tx(&payer, from, to, 50, validity_around(c.now()));
    let fine_hash = fine.hash();
    c.submit(Arc::new(fine));
    let status = await_tx_terminal(c, fine_hash, epochs(8));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "a covered VM transfer must accept after an abort; status = {status:?}"
    );
}

/// A dependent transfer reads its own block's attested baseline.
///
/// The second transfer spends more than its payer's genesis balance and
/// is covered only by the first transfer's committed deposit. It accepts
/// only if its tick's reads pin to the state its block attests — which
/// includes the funding commit — never a stale baseline; and it cannot
/// read further forward either, since nothing beyond its baseline
/// exists. (Submitted after the funding settles: concurrent submission
/// would leave the pair's serialization order to admission, making the
/// dependent leg's verdict scheduling-dependent by design.)
///
/// # Panics
///
/// Panics if either transfer misses its budget, the dependent transfer
/// does not accept, or the commit order is not strictly increasing.
pub fn reads_the_committed_baseline(c: &mut impl Cluster) {
    let (alice_key, alice) = sender(0);
    let (bob_key, bob) = sender(1);
    let carol = recipient(0);

    // Bob holds 10_000 at genesis; after Alice's 5_000 deposit he can
    // cover 12_000.
    let first = build_transfer_tx(&alice_key, alice, bob, 5_000, validity_around(c.now()));
    let first_hash = first.hash();
    c.submit(Arc::new(first));
    let status = await_tx_terminal(c, first_hash, epochs(10));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "funding transfer must accept; status = {status:?}"
    );

    let second = build_transfer_tx(&bob_key, bob, carol, 12_000, validity_around(c.now()));
    let second_hash = second.hash();
    c.submit(Arc::new(second));
    let status = await_tx_terminal(c, second_hash, epochs(10));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "dependent transfer must accept against the committed baseline; status = {status:?}"
    );

    let (first_committed, _) = c.chain_fate(ShardId::ROOT, first_hash);
    let (second_committed, _) = c.chain_fate(ShardId::ROOT, second_hash);
    let (first_committed, second_committed) = (
        first_committed.expect("accepted transfer has a commit height"),
        second_committed.expect("accepted transfer has a commit height"),
    );
    assert!(
        second_committed > first_committed,
        "the dependent transfer must commit after its funding \
         ({second_committed:?} vs {first_committed:?})"
    );
}

/// Zipf-skewed VM payments: `senders` transfers into `recipients` payees
/// drawn from a Zipf(`skew`) distribution — the catalogue's contention
/// shape on the engine.
///
/// # Panics
///
/// Panics if any payment misses its budget or does not accept.
pub fn zipf_payments(
    c: &mut impl Cluster,
    senders: u8,
    recipients: u8,
    skew: f64,
) -> ContentionReport {
    let cdf = zipf_cdf(recipients as usize, skew);
    let mut rng = Lcg(0x5eed_c0de ^ u64::from(senders) << 8 ^ u64::from(recipients));
    let mut submissions = Vec::with_capacity(senders as usize);
    for index in 0..senders {
        let (payer, from) = sender(index);
        let draw = rng.unit();
        let rank = cdf.iter().position(|&c| draw < c).unwrap_or(cdf.len() - 1);
        let to = recipient(u8::try_from(rank).expect("recipient rank fits"));
        let tx = build_transfer_tx(&payer, from, to, PAYMENT, validity_around(c.now()));
        submissions.push((tx.hash(), c.now()));
        c.submit(Arc::new(tx));
    }
    settle_and_report(c, &submissions, epochs(10))
}

/// One hot VM recipient, composed rather than serialized.
///
/// Every payer deposits to the same vault cell. Admission no longer
/// arbitrates that overlap, so the payments ride whatever blocks they
/// land in and the batch's conflict groups sequence them — deposits are
/// `delta`, which the mode lattice calls commutative, so they run in one
/// group against one overlay.
///
/// The balance is the assertion that matters: it is the only thing that
/// distinguishes payments threaded through a batch from payments each
/// computing an absolute against a baseline that excludes its siblings.
///
/// # Panics
///
/// Panics if any payment misses its budget, does not accept, or the hot
/// vault does not hold every accepted payment.
pub fn hot_recipient(c: &mut impl Cluster, senders: u8) -> (ContentionReport, u64) {
    let hot = recipient(0);
    let before = vault_balance(c, ShardId::ROOT, hot);
    let mut submissions = Vec::with_capacity(senders as usize);
    for index in 0..senders {
        let (payer, from) = sender(index);
        let tx = build_transfer_tx(&payer, from, hot, PAYMENT, validity_around(c.now()));
        submissions.push((tx.hash(), c.now()));
        c.submit(Arc::new(tx));
    }
    let report = settle_and_report(c, &submissions, epochs(16));

    let mut heights = Vec::with_capacity(submissions.len());
    for (hash, _) in &submissions {
        let (committed, _) = c.chain_fate(ShardId::ROOT, *hash);
        heights.push(committed.expect("accepted payment has a commit height"));
    }
    heights.sort_unstable();
    let span = heights
        .last()
        .map_or(0, |last| last.inner() - heights[0].inner() + 1);

    // Every payment has to be *in* the hot vault: only the balance says
    // none was overwritten by another executing against the same
    // baseline — `settle_and_report` has already asserted all of them
    // accepted.
    let settled = u128::try_from(report.submitted).expect("bounded");
    assert_eq!(
        vault_balance(c, ShardId::ROOT, hot) - before,
        settled * PAYMENT,
        "the hot vault must hold every accepted payment: {settled} settled",
    );
    (report, span)
}

/// A cross-shard transfer settles through the payer-first holdback.
///
/// The reserve leg lives on the payer's shard and the delta leg on the
/// recipient's; neither leg provisions state (both are commutative), so
/// the payer's tick records an empty dependency set and
/// dispatches immediately. The recipient engages only on the transaction
/// commit proof — the payer's empty-entry bundle, consumable once the
/// payer's block commit-proves — so its commit trails the payer's by one
/// cross-shard hop, and its tick's requirement is satisfied by the
/// bundle committing beside the transaction. Settlement is then the EC
/// exchange.
///
/// # Panics
///
/// Panics if the transfer misses its budget, does not accept, or either
/// shard's chain never commits it.
pub fn cross_shard_transfer(c: &mut impl Cluster) {
    let (payer, from, to) = cross_shard_cast();
    let tx = build_transfer_tx(&payer, from, to, 100, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));

    let status = await_tx_terminal(c, hash, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "cross-shard VM transfer did not accept; status = {status:?}"
    );
    // Atomic settlement: both legs' chains carry the transaction.
    let (left, _) = c.chain_fate(ShardId::leaf(1, 0), hash);
    let (right, _) = c.chain_fate(ShardId::leaf(1, 1), hash);
    assert!(left.is_some(), "payer shard never committed the transfer");
    assert!(
        right.is_some(),
        "recipient shard never committed the transfer"
    );
}

/// A payer cannot spend one balance twice across two ticks.
///
/// The sibling of [`cross_shard_credit_survives_a_later_local_credit`] on
/// the paying side. A cross-shard leg's reservation is judged against its
/// tick's baseline, and the debit it settles is provisional until the
/// tick resolves — so a second withdrawal one tick later would be judged
/// against a balance the first has not come out of, and two withdrawals
/// the vault cannot jointly cover would each be individually feasible.
///
/// One of them has to be refused. Which one is not the claim: the
/// baseline the second is judged against is, and a vault that funded both
/// is the only way both could pass. Conservation is the second half —
/// the refused withdrawal must not credit its recipient either, which is
/// the counterpart's side of the same verdict.
///
/// The recipients are distinct so the two deposits land on different
/// cells — what this measures is the payer's reservation, not the
/// recipients' deposits.
///
/// # Panics
///
/// Panics if the payer's shard never commits the first withdrawal, if
/// either withdrawal misses its budget, if the vault covers both, or if
/// the recipients bank more than the payer gave up.
pub fn a_payer_cannot_spend_one_balance_twice(c: &mut impl Cluster) {
    let payer_shard = ShardId::leaf(1, 0);
    let recipient_shard = ShardId::leaf(1, 1);
    let (payer, from, first_to, second_to) = overdraw_cast();
    let payer_before = vault_balance(c, payer_shard, from);
    let banked_before =
        vault_balance(c, recipient_shard, first_to) + vault_balance(c, recipient_shard, second_to);
    assert!(
        payer_before < 2 * OVERDRAW_AMOUNT,
        "the pair has to be jointly uncoverable to measure anything: \
         holding {payer_before}",
    );

    let first = build_transfer_tx(
        &payer,
        from,
        first_to,
        OVERDRAW_AMOUNT,
        validity_around(c.now()),
    );
    let first_hash = first.hash();
    c.submit(Arc::new(first));
    assert!(
        c.run_until(epochs(16), |c| c
            .chain_fate(payer_shard, first_hash)
            .0
            .is_some()),
        "the payer's shard never committed the first withdrawal"
    );

    let second = build_transfer_tx(
        &payer,
        from,
        second_to,
        OVERDRAW_AMOUNT,
        validity_around(c.now()),
    );
    let second_hash = second.hash();
    c.submit(Arc::new(second));

    let verdicts: Vec<Option<TransactionStatus>> = [first_hash, second_hash]
        .iter()
        .map(|hash| await_tx_terminal(c, *hash, epochs(16)))
        .collect();
    let accepted = verdicts
        .iter()
        .filter(|status| {
            matches!(
                status,
                Some(TransactionStatus::Completed(TransactionDecision::Accept))
            )
        })
        .count();
    assert_eq!(
        accepted, 1,
        "only one withdrawal is covered, so only one may pass; verdicts = {verdicts:?}"
    );

    // Both settlements have to persist before either vault is read.
    c.run_until(epochs(4), |_| false);
    let paid = payer_before.saturating_sub(vault_balance(c, payer_shard, from));
    let banked = (vault_balance(c, recipient_shard, first_to)
        + vault_balance(c, recipient_shard, second_to))
    .saturating_sub(banked_before);
    assert!(
        paid >= banked,
        "the payer gave up {paid} out of {payer_before} and the recipients \
         banked {banked}: a refused withdrawal still credited its recipient"
    );
}

/// A cross-shard credit and a later local one over the same vault both
/// survive.
///
/// The cross-shard leg's local writes are provisional until its tick
/// settles — nothing may read them, or an abort would retroactively
/// change an answer already given. So a transaction the shard commits
/// afterwards sees the vault as it was before the leg ran, and both
/// receipts carry an absolute the other's effect is missing from:
/// whichever settles second overwrites the first, and one credit is
/// gone.
///
/// The local payment is submitted only once the recipient's shard has
/// committed the crossing, so the pair is genuinely ordered across two
/// ticks with the tick still open between them. The balance is the whole
/// assertion — both credits or neither.
///
/// # Panics
///
/// Panics if either payment misses its budget or does not accept, if the
/// recipient's shard never commits the crossing, or if the vault does not
/// hold both credits.
pub fn cross_shard_credit_survives_a_later_local_credit(c: &mut impl Cluster) {
    const CROSSING: u128 = 100;
    const LOCAL: u128 = 50;

    let recipient_shard = ShardId::leaf(1, 1);
    let (remote_payer, remote_from, local_payer, local_from, to) = shared_recipient_cast();
    let before = vault_balance(c, recipient_shard, to);

    let crossing = build_transfer_tx(
        &remote_payer,
        remote_from,
        to,
        CROSSING,
        validity_around(c.now()),
    );
    let crossing_hash = crossing.hash();
    c.submit(Arc::new(crossing));

    // The recipient's shard has the leg in a tick from the moment it
    // commits it; the tick cannot settle for several blocks yet, so the
    // local payment below lands in a later tick with the leg's writes
    // still provisional.
    assert!(
        c.run_until(epochs(16), |c| c
            .chain_fate(recipient_shard, crossing_hash)
            .0
            .is_some()),
        "the recipient's shard never committed the crossing"
    );

    let local = build_transfer_tx(
        &local_payer,
        local_from,
        to,
        LOCAL,
        validity_around(c.now()),
    );
    let local_hash = local.hash();
    c.submit(Arc::new(local));

    for (hash, label) in [(crossing_hash, "crossing"), (local_hash, "local payment")] {
        let status = await_tx_terminal(c, hash, epochs(16));
        assert!(
            matches!(
                status,
                Some(TransactionStatus::Completed(TransactionDecision::Accept))
            ),
            "the {label} did not accept; status = {status:?}"
        );
    }

    // Settlement trails the decision by the persistence step, and the two
    // ticks settle at different heights — wait for the total rather than
    // reading whichever landed first.
    let expected = before + CROSSING + LOCAL;
    let held = c.run_until(epochs(8), |c| {
        vault_balance(c, recipient_shard, to) == expected
    });
    assert!(
        held,
        "the shared vault must hold both credits: expected {expected}, holding {}",
        vault_balance(c, recipient_shard, to)
    );
}

/// A multi-shard transaction's events land only on their emitters' home
/// receipts.
///
/// The withdrawal emits from the payer's account and the deposit from the
/// recipient's, and the two accounts sit on different shards. Each shard
/// stores its own event and not the other's, while the receipt hash both
/// committees agree on covers the union — so attribution splits the
/// storage without splitting the agreement.
///
/// # Panics
///
/// Panics if the transfer does not accept, if either shard never holds a
/// receipt for it, or if either shard stores an event whose emitter lives
/// on the other.
pub fn events_land_on_their_emitters_home_shard(c: &mut impl Cluster) {
    let (payer, from, to) = cross_shard_cast();
    let tx = build_transfer_tx(&payer, from, to, 100, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));

    let status = await_tx_terminal(c, hash, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "cross-shard VM transfer did not accept; status = {status:?}"
    );

    let (sender_shard, recipient_shard) = (ShardId::leaf(1, 0), ShardId::leaf(1, 1));
    // Receipts persist a beat behind the decision, so wait for both.
    let stored = c.run_until(epochs(8), |c| {
        c.events(sender_shard, hash).is_some() && c.events(recipient_shard, hash).is_some()
    });
    assert!(stored, "both shards must hold a receipt for the transfer");

    let sender_events = c.events(sender_shard, hash).expect("payer receipt");
    let recipient_events = c.events(recipient_shard, hash).expect("recipient receipt");
    assert_eq!(
        sender_events
            .iter()
            .map(|event| event.emitter)
            .collect::<Vec<_>>(),
        vec![from],
        "the payer shard stores its own emission and nothing else"
    );
    assert_eq!(
        recipient_events
            .iter()
            .map(|event| event.emitter)
            .collect::<Vec<_>>(),
        vec![to],
        "the recipient shard stores its own emission and nothing else"
    );
}

/// Both shards' attested load reaches the beacon, including the
/// counterpart's — the shard the fee never paid.
///
/// Fees never move cross-shard, and this exercises the whole of what
/// replaces them. A cross-shard transfer
/// burns its fee at the payer's shard alone, so the counterpart executes
/// its leg for nothing; the work it did is instead attested as gas on its
/// own receipts, carried on its own headers, and folded onto its own
/// boundary record, where the emission weighting reads it. The assertion
/// that carries the rule is the counterpart's mark moving at all.
///
/// The byte level is checked for stability rather than for conservation
/// across a reshape: without storage bonds there is no conserved
/// quantity to balance. What is checkable is that the channel neither
/// invents nor loses state — a quiesced network's recorded levels do
/// not drift.
///
/// # Panics
///
/// Panics if the transfer does not accept, if either shard's mark or byte
/// level never reaches the beacon within budget, if the counterpart
/// attests no work for the leg it executed, or if a recorded byte level
/// moves while nothing is executing.
pub fn attested_load_reaches_the_beacon(c: &mut impl Cluster) {
    let left = ShardId::leaf(1, 0);
    let right = ShardId::leaf(1, 1);

    let (payer, from, to) = cross_shard_cast();
    let tx = build_transfer_tx(&payer, from, to, 100, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));
    let status = await_tx_terminal(c, hash, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "cross-shard VM transfer did not accept; status = {status:?}"
    );

    // Wait for both shards to fold a crossing carrying a non-zero mark.
    let both_attested = |c: &_| {
        recorded_gas(c, left).is_some_and(|g| g > 0)
            && recorded_gas(c, right).is_some_and(|g| g > 0)
    };
    assert!(
        c.run_until(epochs(24), both_attested),
        "attested work never reached the beacon: left = {:?}, right = {:?}",
        recorded_gas(c, left),
        recorded_gas(c, right),
    );

    // The counterpart burned no fee and still attested its work — without
    // this the emission weighting would pay it only the participation
    // floor, and cross-shard execution would be unfunded.
    let counterpart_gas = recorded_gas(c, right).expect("counterpart record present");
    assert!(
        counterpart_gas > 0,
        "the counterpart shard attested no work for a leg it executed"
    );

    // The byte levels are recorded, and a quiesced network does not drift:
    // nothing executes, so no state appears or vanishes on either record.
    let settled = |c: &_| recorded_bytes(c, left).is_some() && recorded_bytes(c, right).is_some();
    assert!(
        c.run_until(epochs(8), settled),
        "byte levels never reached the beacon"
    );
    let before = (recorded_bytes(c, left), recorded_bytes(c, right));
    // Burn the budget with nothing to wait for: the condition never holds,
    // so this runs the cluster on for the whole span and returns false.
    c.run_until(epochs(8), |_| false);
    let after = (recorded_bytes(c, left), recorded_bytes(c, right));
    assert_eq!(
        before, after,
        "recorded byte levels drifted with nothing executing"
    );
}

/// A transaction that never applied an effect still attests the work its
/// shard did for it.
///
/// This is what moving the quantity off the receipt buys. A receipt records
/// effects, so an attempt that produced none has nothing to carry — and a
/// failure or an abort is exactly when a shard has already paid for
/// admission, routing, and locking. The outcome exists for every verdict,
/// so the work rides there and the shard is credited for the attempt.
///
/// # Panics
///
/// Panics if the uncovered withdrawal does not reject, or if the shard's
/// attested mark fails to move across a block whose only transaction
/// applied nothing.
pub fn a_failed_attempt_still_attests_work(c: &mut impl Cluster) {
    let shard = ShardId::ROOT;
    let (payer, from) = sender(0);
    let to = recipient(0);

    // Settle any earlier traffic so the mark below moves only for the
    // failure this scenario submits.
    assert!(
        c.run_until(epochs(4), |c| recorded_gas(c, shard).is_some()),
        "the beacon never folded a crossing for the shard"
    );
    let before = recorded_gas(c, shard).expect("a folded crossing");

    let over = build_transfer_tx(&payer, from, to, 1_000_000, validity_around(c.now()));
    let over_hash = over.hash();
    c.submit(Arc::new(over));
    let status = await_tx_terminal(c, over_hash, epochs(8));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Reject))
        ),
        "an uncovered VM withdrawal must reject deterministically; status = {status:?}"
    );

    // The attempt applied nothing and still attests work: a failed
    // verdict produces no receipt, so the attestation must ride the
    // outcome itself for the mark to move.
    assert!(
        c.run_until(epochs(24), |c| recorded_gas(c, shard)
            .is_some_and(|now| now > before)),
        "a failed attempt attested no work: mark stuck at {before}"
    );
}

/// The gas mark on `shard`'s boundary record, if the beacon has folded a
/// crossing for it.
fn recorded_gas<C: Cluster>(c: &C, shard: ShardId) -> Option<u64> {
    c.beacon_state()
        .and_then(|state| state.boundaries.get(&shard).map(|b| b.attested_work))
}

/// The stored-byte level on `shard`'s boundary record.
fn recorded_bytes<C: Cluster>(c: &C, shard: ShardId) -> Option<u64> {
    c.beacon_state()
        .and_then(|state| state.boundaries.get(&shard).map(|b| b.substate_bytes))
}

/// A randomness-reading transaction derives one draw on both shards.
///
/// Two lottery instances, one per shard, settle their rounds in the same
/// transaction, so the two recorded draws agree exactly when the two
/// committees executed under one value. The draw is anchored on the
/// payer block — the block that committed the transaction on the payer's
/// shard, whose reveal chain rides the engagement bundle — so agreement
/// holds by construction rather than by the two shards happening to
/// commit in step. Each result is an exclusive write, so this is also the
/// read-set-provisioned shape in both directions: each shard executes on
/// the other's shipped prior.
///
/// # Panics
///
/// Panics if the draw misses its budget, does not accept, either shard's
/// chain never commits it, either round is unsettled, or the two rounds
/// settled on different draws.
pub fn randomness_draw_agrees_across_shards<C: Cluster>(c: &mut C) {
    let (payer, ..) = cross_shard_keys();
    let left = lottery_on(ShardId::leaf(1, 0));
    let right = lottery_on(ShardId::leaf(1, 1));
    // Both components are sealed first, in their own transaction: a
    // draw's fence reads a committed leaf, and nothing is committed
    // inside the transaction that writes it.
    let seal = build_instantiate_tx(
        &payer,
        &[left.clone(), right.clone()],
        validity_around(c.now()),
    );
    let sealed = seal.hash();
    c.submit(Arc::new(seal));
    let status = await_tx_terminal(c, sealed, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "the lotteries were never sealed; status = {status:?}"
    );

    let tx = build_draw_tx(
        &payer,
        &[left.clone(), right.clone()],
        validity_around(c.now()),
    );
    let hash = tx.hash();
    c.submit(Arc::new(tx));

    let status = await_tx_terminal(c, hash, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "cross-shard VM draw did not accept; status = {status:?}"
    );
    let (left_fate, _) = c.chain_fate(ShardId::leaf(1, 0), hash);
    let (right_fate, _) = c.chain_fate(ShardId::leaf(1, 1), hash);
    assert!(left_fate.is_some(), "payer shard never committed the draw");
    assert!(
        right_fate.is_some(),
        "counterpart shard never committed the draw"
    );

    // The rounds are read off each shard's own committed state, which
    // trails the settling block by the persistence step.
    let read = |c: &C, shard: ShardId, lottery: &InstanceMeta| -> Option<Vec<u8>> {
        let key = draw_key(lottery.address(&ProtocolHasher));
        c.substate(shard, key.owner, key.local.0)
    };
    assert!(
        c.run_until(epochs(4), |c| read(c, ShardId::leaf(1, 0), &left).is_some()
            && read(c, ShardId::leaf(1, 1), &right).is_some()),
        "both rounds must have settled"
    );
    let left = read(c, ShardId::leaf(1, 0), &left).expect("settled");
    let right = read(c, ShardId::leaf(1, 1), &right).expect("settled");
    // Nobody entered either round, so each cell holds the draw and no
    // winner: thirty-two bytes at the width the record states, and one
    // byte saying there is no winner.
    assert_eq!(
        left.len(),
        33,
        "an unentered round is the draw and no winner"
    );
    assert_eq!(
        left, right,
        "the two shards executed the transaction under different draws"
    );
}

/// An insolvent payer's transaction engages nothing anywhere.
///
/// The payer's balance cannot cover the signed fee ceiling, so the
/// reservation is uncoverable: no payer-shard proposer selects the
/// transaction, and the reservation verification refuses any block that
/// carries it — it never commits at the payer shard, so no bundle ever
/// flows and no counterpart engages a lock. The transaction expires in
/// the mempool while both chains carry on.
///
/// # Panics
///
/// Panics if either chain stalls, the transaction completes, or either
/// shard's chain ever includes it.
pub fn insolvent_payer_engages_nothing(c: &mut impl Cluster) {
    let (payer, from, to) = cross_shard_cast();
    let tx = build_transfer_tx(&payer, from, to, 5, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));

    // Both chains keep advancing while the transaction goes nowhere.
    assert!(
        await_height(c, ShardId::leaf(1, 0), 3, epochs(6)),
        "payer shard chain must keep advancing"
    );
    assert!(
        await_height(c, ShardId::leaf(1, 1), 3, epochs(6)),
        "counterpart shard chain must keep advancing"
    );
    let status = c.tx_status(hash);
    assert!(
        !matches!(status, Some(TransactionStatus::Completed(_))),
        "an uncoverable reservation must never complete; status = {status:?}"
    );
    let (payer_inclusion, _) = c.chain_fate(ShardId::leaf(1, 0), hash);
    assert!(
        payer_inclusion.is_none(),
        "the insolvent payer's transaction must never commit at the payer shard"
    );
    let (counterpart_inclusion, _) = c.chain_fate(ShardId::leaf(1, 1), hash);
    assert!(
        counterpart_inclusion.is_none(),
        "the counterpart must not engage an insolvent payer's transaction"
    );
}

/// A payer whose rule does not admit the signer engages nothing
/// anywhere.
///
/// The manifest, the signature, and the fee ceiling are all honest —
/// only the payer field names an account the signer's key does not
/// open, and that account is funded, so nothing about solvency can
/// refuse it. Derivation admits the envelope: the binding is the payer
/// shard's own verdict, judged where the payer's rule lives. No
/// payer-shard proposer selects the transaction, the reservation
/// verification refuses any block that carries it, and it never
/// commits at the payer shard — so no bundle flows, no counterpart
/// engages, and the named account is debited nothing.
///
/// # Panics
///
/// Panics if either chain stalls, the transaction completes, or either
/// shard's chain ever includes it.
pub fn unbound_payer_engages_nothing(c: &mut impl Cluster) {
    let (signer, from, to, victim) = unbound_payer_cast();
    let tx = build_unbound_payer_tx(&signer, from, to, victim, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));

    // Both chains keep advancing while the transaction goes nowhere.
    assert!(
        await_height(c, ShardId::leaf(1, 0), 3, epochs(6)),
        "payer shard chain must keep advancing"
    );
    assert!(
        await_height(c, ShardId::leaf(1, 1), 3, epochs(6)),
        "counterpart shard chain must keep advancing"
    );
    let status = c.tx_status(hash);
    assert!(
        !matches!(status, Some(TransactionStatus::Completed(_))),
        "an unbound payer's transaction must never complete; status = {status:?}"
    );
    let (payer_inclusion, _) = c.chain_fate(ShardId::leaf(1, 0), hash);
    assert!(
        payer_inclusion.is_none(),
        "the unbound payer's transaction must never commit at the payer shard"
    );
    let (counterpart_inclusion, _) = c.chain_fate(ShardId::leaf(1, 1), hash);
    assert!(
        counterpart_inclusion.is_none(),
        "the counterpart must not engage an unbound payer's transaction"
    );
}

/// The same refusal when the payer's shard holds nothing else of the
/// transaction.
///
/// The manifest — signer, sender, recipient — lives whole on the
/// counterpart shard, and the payer shard's only stake is the fee vault
/// and the stored-authority cell beside it. The binding verdict cannot
/// lean on a manifest leg it would have judged anyway; it stands alone,
/// and it alone must keep the reservation from engaging — so the
/// transaction never commits on either shard and the remote victim is
/// debited nothing.
///
/// # Panics
///
/// Panics if either chain stalls, the transaction completes, or either
/// shard's chain ever includes it.
pub fn unbound_remote_payer_engages_nothing(c: &mut impl Cluster) {
    let (signer, from, to, victim) = unbound_remote_payer_cast();
    let tx = build_unbound_payer_tx(&signer, from, to, victim, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));

    // Both chains keep advancing while the transaction goes nowhere.
    assert!(
        await_height(c, ShardId::leaf(1, 0), 3, epochs(6)),
        "payer shard chain must keep advancing"
    );
    assert!(
        await_height(c, ShardId::leaf(1, 1), 3, epochs(6)),
        "manifest shard chain must keep advancing"
    );
    let status = c.tx_status(hash);
    assert!(
        !matches!(status, Some(TransactionStatus::Completed(_))),
        "an unbound remote payer's transaction must never complete; status = {status:?}"
    );
    let (payer_inclusion, _) = c.chain_fate(ShardId::leaf(1, 0), hash);
    assert!(
        payer_inclusion.is_none(),
        "the unbound payer's transaction must never commit at the payer shard"
    );
    let (manifest_inclusion, _) = c.chain_fate(ShardId::leaf(1, 1), hash);
    assert!(
        manifest_inclusion.is_none(),
        "the manifest shard must not engage without the payer's reservation"
    );
}

/// The whole securify transition, through consensus.
///
/// The founding key pays and settles, signs its account over to another
/// identity's rule, and from that commit on is dead at its own payer
/// shard — while the installed rule's key acts and pays from the
/// account it governs.
///
/// The installed identity is ML-DSA-65, so this is also the account
/// migration to post-quantum authority end to end: the classical key
/// retires, the account keeps its address, balance and placement, and a
/// kilobyte-scale signature carries a transaction through mempool
/// admission, proposal, and the vote-time verification. Nothing between
/// the rule and the reservation is told which scheme happened — a rule
/// names an address, and the scheme is already folded into it.
///
/// The retired key's refusal is the payer shard's binding verdict over
/// the stored cell, judged at the anchored read height like the balance
/// beside it; the corpus pins the same flip inside one process, and
/// this drives it across the reservation machinery — mempool advisory,
/// proposal build, and the vote-time verification — with the recipient
/// on the counterpart shard so the engagement rails carry the verdict.
///
/// # Panics
///
/// Panics if the baseline or securify transactions fail, the retired
/// key's transfer is ever included, the holder's transfer fails, or the
/// recipient's balance shows anything but the two settled transfers.
pub fn securify_retires_the_key_at_the_payer_shard(c: &mut impl Cluster) {
    let payer_shard = ShardId::leaf(1, 0);
    let counterpart = ShardId::leaf(1, 1);
    let (owner_key, owner, holder_key, holder, to) = securify_cast();
    assert_eq!(
        holder_key.scheme(),
        SchemeId::ML_DSA_65,
        "the identity this account migrates to must be post-quantum"
    );

    // Baseline: the founding key pays for its own account and settles
    // cross-shard.
    let tx = build_transfer_tx(&owner_key, owner, to, 100, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));
    let status = await_tx_terminal(c, hash, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "the baseline transfer must settle; status = {status:?}"
    );

    // The one-way transition: the account's stored rule becomes the
    // holder's identity.
    let tx = build_securify_tx(&owner_key, owner, holder, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));
    let status = await_tx_terminal(c, hash, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "securify must settle; status = {status:?}"
    );

    // The retired key still derives the account's address, and that
    // identity is exactly what the stored rule no longer admits: no
    // proposer selects its transfer, and it never commits anywhere.
    let tx = build_transfer_tx(&owner_key, owner, to, 7, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));
    let payer_height = c
        .committed_height(payer_shard)
        .map_or(0, BlockHeight::inner);
    assert!(
        await_height(c, payer_shard, payer_height + 3, epochs(6)),
        "payer shard chain must keep advancing past the refused transfer"
    );
    let status = c.tx_status(hash);
    assert!(
        !matches!(status, Some(TransactionStatus::Completed(_))),
        "the retired key's transfer must never complete; status = {status:?}"
    );
    let (payer_inclusion, _) = c.chain_fate(payer_shard, hash);
    assert!(
        payer_inclusion.is_none(),
        "the retired key's transfer must never commit at the payer shard"
    );
    let (counterpart_inclusion, _) = c.chain_fate(counterpart, hash);
    assert!(
        counterpart_inclusion.is_none(),
        "the counterpart must not engage the retired key's transfer"
    );

    // The installed rule's key signs in as the account it governs and
    // pays from it, from another shard's identity entirely.
    let tx = build_transfer_paid_by(&holder_key, owner, to, 9, owner, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));
    let status = await_tx_terminal(c, hash, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "the installed rule's key must act and pay; status = {status:?}"
    );

    // The recipient holds exactly the two settled transfers: the
    // refused one credited nothing. Awaited, because the terminal
    // status precedes the credit's settlement round on the recipient's
    // shard.
    assert!(
        c.run_until(epochs(8), |c| vault_balance(c, counterpart, to)
            == 10 + 100 + 9),
        "the recipient's balance must carry the settled transfers alone; balance = {}",
        vault_balance(c, counterpart, to)
    );
}

/// An account founded on a post-quantum key pays its own way.
///
/// The other half of the post-quantum story from
/// [`securify_retires_the_key_at_the_payer_shard`], and the half that
/// needs no transition: this account never held a classical key. Its
/// address is what an ML-DSA-65 key derives, so the identity that opens
/// it by signature is the one its own address names, and it is a virtual
/// account with nothing stored.
///
/// What that costs the protocol is the whole point of driving it here.
/// The account signs, is admitted, reserves against its own vault, and
/// settles a transfer to a recipient on another shard — so a
/// kilobyte-scale signature travels the full cross-shard path, and the
/// fee binding names a principal that no curve derived.
///
/// # Panics
///
/// Panics if the payer's key is not post-quantum, the transfer does not
/// accept, the recipient is not credited, or the payer's vault does not
/// show the transfer and a fee on top of it.
pub fn a_native_post_quantum_account_pays_its_own_way(c: &mut impl Cluster) {
    let payer_shard = ShardId::leaf(1, 0);
    let counterpart = ShardId::leaf(1, 1);
    let (payer_key, payer, to) = native_pq_cast();
    assert_eq!(
        payer_key.scheme(),
        SchemeId::ML_DSA_65,
        "this account's founding key must be post-quantum"
    );
    assert_eq!(
        vault_balance(c, payer_shard, payer),
        10_000,
        "genesis must seed a post-quantum address like any other"
    );

    let tx = build_transfer_tx(&payer_key, payer, to, 100, validity_around(c.now()));
    let hash = tx.hash();
    c.submit(Arc::new(tx));
    let status = await_tx_terminal(c, hash, epochs(16));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "the post-quantum account's own transfer must settle; status = {status:?}"
    );

    // Awaited, because the terminal status precedes the credit's
    // settlement round on the recipient's shard.
    assert!(
        c.run_until(epochs(8), |c| vault_balance(c, counterpart, to) == 10 + 100),
        "the recipient must carry the settled transfer; balance = {}",
        vault_balance(c, counterpart, to)
    );

    // The account paid the transfer *and* a fee out of its own vault:
    // the reservation engaged against a principal an ML-DSA key derived,
    // which is what distinguishes paying its own way from merely being
    // a signature that verified.
    let remaining = vault_balance(c, payer_shard, payer);
    assert!(
        remaining < 10_000 - 100,
        "the payer must have settled a fee beyond the transfer; balance = {remaining}"
    );
}

/// A payer whose counterpart never engages settles the abort floor and
/// nothing else.
///
/// Cutting both channels the payer's bundle travels — the gossip
/// broadcast and the fetch that backs it up — makes the counterpart's
/// absence structural: engagement demands that evidence, so the
/// transaction can never enter a block there. The payer's own leg is
/// dependency-free, so it commits, reserves, and executes at once, then
/// waits. It speaks once: with no engagement echo and the window closed,
/// its single statement is the abort. The transaction's effects are
/// discarded, so the recipient is never credited and the transferred
/// amount never leaves; what leaves is the class floor, settled through
/// the fee receipt the abort names.
///
/// # Panics
///
/// Panics if the harness cannot read the payer's vault, the payer never
/// commits, the bundle is never suppressed, the transaction fails to
/// reach a terminal abort, the counterpart engages, or the payer's
/// balance moves by anything other than the floor.
pub fn abort_floor_settles_on_deadline(c: &mut impl FaultableCluster) {
    let payer_shard = ShardId::leaf(1, 0);
    let counterpart = ShardId::leaf(1, 1);
    let (payer, from, to) = cross_shard_cast();
    let before = vault_balance(c, payer_shard, from);

    // Both channels the bundle travels. The fetch rule names the
    // *request* type: the fault engine tags a request and its response
    // alike, so dropping the response id would never match.
    let broadcast_dropped = c.drop_type("provisions.broadcast");
    let fetch_dropped = c.drop_type("provision.request");

    let tx = build_transfer_tx(&payer, from, to, 100, validity_around(c.now()));
    let hash = tx.hash();
    let floor = tx.body().abort_floor();
    c.submit(Arc::new(tx));

    assert!(
        c.run_until(epochs(8), |c| c.chain_fate(payer_shard, hash).0.is_some()),
        "the payer shard must commit and reserve for the transaction"
    );

    let verdict = await_tx_terminal(c, hash, epochs(90));
    assert!(
        matches!(
            verdict,
            Some(TransactionStatus::Completed(TransactionDecision::Aborted))
        ),
        "an unechoed cross-shard VM transaction must abort at the deadline; \
         verdict = {verdict:?}",
    );
    assert!(
        broadcast_dropped.fired() > 0 && fetch_dropped.fired() > 0,
        "both bundle channels must actually have been exercised and cut"
    );
    let (counterpart_inclusion, _) = c.chain_fate(counterpart, hash);
    assert!(
        counterpart_inclusion.is_none(),
        "the counterpart must never have engaged the transaction",
    );

    // The floor left the payer's vault; the transfer did not.
    //
    // A terminal status is reported the moment this shard decides the
    // abort, which is a block or more before the finalization carrying its
    // fee receipt commits — so the charge has to be waited for rather than
    // read at the instant the status flips. Waiting on the balance moving
    // at all keeps the amount itself unasserted until below.
    assert!(
        c.run_until(epochs(8), |c| vault_balance(c, payer_shard, from) != before),
        "the abort's charge must reach committed state"
    );
    let after = vault_balance(c, payer_shard, from);
    assert_eq!(
        before.saturating_sub(after),
        floor,
        "the abort must burn exactly the class floor: before = {before}, after = {after}",
    );
}

/// A transaction that fails still pays, and what it pays depends on whose
/// fault the failure was.
///
/// Failing must never be the cheaper way to buy execution. An uncovered
/// withdrawal loses a deterministic race it could not have foreseen — the
/// sender declared honestly and another transaction got there first — so
/// it settles the class floor, not the ceiling. What matters is that it
/// settles something: an attempt that cost nothing would make trapping
/// strictly cheaper than succeeding for identical work.
///
/// # Panics
///
/// Panics if the uncovered withdrawal does not reject, if the covered
/// transfer that follows does not accept, or if the rejected attempt
/// moves the payer's vault by anything other than the floor.
pub fn failure_charges_its_payer(c: &mut impl Cluster) {
    let shard = ShardId::ROOT;
    let (payer, from) = sender(0);
    let to = recipient(0);

    let before = vault_balance(c, shard, from);
    let over = build_transfer_tx(&payer, from, to, 1_000_000, validity_around(c.now()));
    let floor = over.body().abort_floor();
    let over_hash = over.hash();
    c.submit(Arc::new(over));
    let status = await_tx_terminal(c, over_hash, epochs(8));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Reject))
        ),
        "an uncovered VM withdrawal must reject deterministically; status = {status:?}"
    );

    let after = vault_balance(c, shard, from);
    assert_eq!(
        before.saturating_sub(after),
        floor,
        "a rejected attempt must settle exactly the class floor: \
         before = {before}, after = {after}, floor = {floor}"
    );

    // The charge is the only thing that moved: the payer can still spend.
    let fine = build_transfer_tx(&payer, from, to, 50, validity_around(c.now()));
    let fine_hash = fine.hash();
    c.submit(Arc::new(fine));
    let status = await_tx_terminal(c, fine_hash, epochs(8));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "a covered VM transfer must accept after a charged failure; status = {status:?}"
    );
}

/// Many withdrawals in one validity window, all drawing on one vault.
///
/// Every withdrawal reserves the same cell, so admission once handed
/// them out one per commit cycle and the set took as many blocks as it
/// had members. Composed into ticks instead, they land in one conflict
/// group and run sequentially against one overlay: each sees what its
/// predecessors took, and all of them land.
///
/// The amounts ascend so that no two envelopes are identical; the sum
/// stays inside the vault, because an uncoverable envelope is refused at
/// admission rather than aborted at execution — that path is
/// [`failure_charges_its_payer`]'s.
///
/// Returns the number of blocks the accepted set occupied, which is the
/// figure the serialization ceiling used to pin at one per member.
///
/// # Panics
///
/// Panics if any withdrawal misses its budget or does not accept, if the
/// recipient's balance disagrees with the total withdrawn, or if the
/// payer settled less than it moved.
pub fn withdrawals_compose_over_one_vault(c: &mut impl Cluster, count: u8) -> u64 {
    let shard = ShardId::ROOT;
    let (payer, from) = sender(0);
    let to = recipient(0);
    let payer_before = vault_balance(c, shard, from);
    let recipient_before = vault_balance(c, shard, to);

    let amount_for = |index: u8| -> u128 { 1 + u128::from(index) };
    let mut submissions = Vec::with_capacity(count as usize);
    for index in 0..count {
        let tx = build_transfer_tx(
            &payer,
            from,
            to,
            amount_for(index),
            validity_around(c.now()),
        );
        submissions.push((tx.hash(), c.now()));
        c.submit(Arc::new(tx));
    }
    settle_and_report(c, &submissions, epochs(16));

    let moved: u128 = (0..count).map(amount_for).sum();
    assert_eq!(
        vault_balance(c, shard, to) - recipient_before,
        moved,
        "the recipient must hold every withdrawal — a shared baseline would \
         leave only the last one",
    );
    let spent = payer_before - vault_balance(c, shard, from);
    assert!(
        spent >= moved,
        "the payer settled every withdrawal it moved: spent = {spent}, moved = {moved}",
    );

    // How tightly the set packed. One block per withdrawal is the
    // serialization ceiling; fewer means they composed.
    let mut heights: Vec<BlockHeight> = submissions
        .iter()
        .map(|(hash, _)| {
            c.chain_fate(shard, *hash)
                .0
                .expect("an accepted withdrawal has a commit height")
        })
        .collect();
    heights.sort_unstable();
    heights.dedup();
    assert!(
        heights.len() < count as usize,
        "{count} withdrawals over one vault occupied {} blocks — they are still \
         serialized one per block",
        heights.len(),
    );
    heights.len() as u64
}

/// The committed balance of a account's native vault, read through the
/// harness's client-proven snapshot seam.
fn vault_balance(c: &impl Cluster, shard: ShardId, owner: impl Into<Address>) -> u128 {
    let vault = vault_key(owner, *XRD);
    c.substate(shard, vault.owner, vault.local.0)
        .map_or(0, |bytes| {
            let cell: [u8; 16] = bytes.as_slice().try_into().expect("an amount cell");
            u128::from_le_bytes(cell)
        })
}

/// The reported change to `owner`'s native vault.
fn preview_change(report: &PreviewReport, owner: impl Into<Address>) -> ResourceChange {
    let owner = owner.into();
    let vault = vault_key(owner, *XRD);
    *report
        .changes
        .iter()
        .find(|change| change.key == vault)
        .unwrap_or_else(|| panic!("no reported change for {owner:?}: {:?}", report.changes))
}

/// A wallet's question before it signs, answered off the tip: what would
/// this transfer move, and what would it cost?
///
/// Preview is engine-side and consensus-free, and the scenario holds it
/// to both halves of that. The candidate is never submitted while it is
/// being previewed — the chain advances past it and has never heard of
/// it, and the payer's committed balance is exactly where it was — and
/// then the same envelope is committed for real, where the balances it
/// lands on are the figures the report named. A preview that reported
/// plausible numbers nobody ever checked against a commit would be
/// decoration.
///
/// Free credit is the one grant a preview carries: it prices the fee
/// without charging it, which is what lets a wallet cost an envelope
/// its payer could not cover.
///
/// # Panics
///
/// Panics if the root shard serves no preview, if the report disagrees
/// with the committed baseline or with what the transfer commits, if the
/// preview leaks the transaction into the chain, or if the transfer does
/// not accept.
pub fn preview_reports_resource_changes(c: &mut impl Cluster) {
    const AMOUNT: u128 = 100;
    let (payer, from) = sender(0);
    let to = recipient(0);

    // A preview reads the chain's own attested clock and reveal, so it
    // wants a chain that has spoken at least once.
    assert!(
        await_height(c, ShardId::ROOT, 1, epochs(2)),
        "root shard did not advance past genesis"
    );
    let sender_before = vault_balance(c, ShardId::ROOT, from);
    let recipient_before = vault_balance(c, ShardId::ROOT, to);

    let candidate = build_transfer_tx(&payer, from, to, AMOUNT, validity_around(c.now()));
    let hash = candidate.hash();
    let report = c
        .preview(ShardId::ROOT, &candidate, PreviewGrants::default())
        .expect("the root shard serves a preview");

    assert_eq!(
        report.outcome,
        PreviewOutcome::Completed,
        "a covered transfer previews as completed"
    );
    assert!(report.fee > 0, "a transfer costs its payer something");

    let sender = preview_change(&report, from);
    assert_eq!(
        (sender.before, sender.settled, sender.after),
        (sender_before, AMOUNT, sender_before - AMOUNT - report.fee),
        "the sender pays the transfer through its reservation's settle, plus the fee"
    );
    let recipient = preview_change(&report, to);
    assert_eq!(
        (recipient.before, recipient.credit, recipient.after),
        (recipient_before, AMOUNT, recipient_before + AMOUNT),
        "the recipient is credited the transfer and charged nothing"
    );

    let credited = c
        .preview(
            ShardId::ROOT,
            &candidate,
            PreviewGrants {
                free_credit: true,
                ..PreviewGrants::default()
            },
        )
        .expect("the root shard serves a preview");
    assert_eq!(credited.fee, report.fee, "the fee is priced either way");
    assert_eq!(
        preview_change(&credited, from).after,
        sender.after + report.fee,
        "free credit keeps exactly the fee off the payer's vault"
    );

    // Nothing was submitted, gossiped, or committed: the chain advances
    // past the preview without ever holding the transaction.
    let ahead = c
        .committed_height(ShardId::ROOT)
        .map_or(2, |h| h.inner() + 2);
    assert!(
        await_height(c, ShardId::ROOT, ahead, epochs(4)),
        "the root shard did not advance past the preview"
    );
    assert!(
        c.tx_status(hash).is_none(),
        "a preview must not submit the transaction it previewed"
    );
    assert_eq!(
        c.chain_fate(ShardId::ROOT, hash),
        (None, None),
        "a preview must reach no chain"
    );
    assert_eq!(
        vault_balance(c, ShardId::ROOT, from),
        sender_before,
        "a preview writes nothing"
    );

    // The same envelope for real: the report was the truth about it.
    c.submit(Arc::new(candidate));
    let status = await_tx_terminal(c, hash, epochs(8));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "the previewed transfer did not accept; status = {status:?}"
    );
    assert_eq!(
        vault_balance(c, ShardId::ROOT, from),
        sender.after,
        "the commit landed on the figure the preview named for the sender"
    );
    assert_eq!(
        vault_balance(c, ShardId::ROOT, to),
        recipient.after,
        "the commit landed on the figure the preview named for the recipient"
    );
}

/// A published package runs only once the chain says it may — and when
/// it runs, it runs on nodes that never committed its publish.
///
/// A publish commits on its publisher's shard, where the code then lives
/// alone; every other node has to fetch it, and it learns to on the
/// beacon fact the publish raises. The maturity window is the time that
/// fetch is given. Holding a transaction out of a block until the window
/// closes is what turns "does this node hold the code" from a race into
/// a fact about the chain.
///
/// The probe is a deposit into an instance of the freshly published
/// package, offered at each of the three moments the rule distinguishes:
/// before the beacon has registered the package, after it has but inside
/// the window, and after the window. Only the last is committed, and its
/// settling is what says the code reached the nodes that never held it —
/// every shard the transaction touches runs the whole of it.
///
/// What a simulated cluster puts under test is the compiled half. The
/// process has one metadata cache however many hosts it stands up, so
/// every host here can route a call the moment the publish commits;
/// only the code is per host, and only the fetch supplies it. So the
/// two earlier moments are held by the registry rule alone, which is
/// the guarantee being probed — a node's own holdings never enter it.
///
/// # Panics
///
/// Panics if the publish does not settle, if a call before the window
/// closes reaches a decision, or if a call after it does not.
pub fn a_published_package_matures_before_it_runs(c: &mut impl Cluster) {
    const SALT: u8 = 9;
    const DEPOSIT: u128 = 40;

    let publishers = storm_publishers();
    let (key, publisher) = &publishers[0];
    let artifact = storm_artifact(4_242);
    let package = package_hash(&ProtocolHasher, &artifact);
    let registered = Hash::from(package.0);
    let cell = package_key(*publisher, package);

    let publish = build_publish_tx(key, artifact.clone(), validity_around(c.now()));
    let publish_hash = publish.hash();
    c.submit(Arc::new(publish));
    let status = await_tx_terminal(c, publish_hash, epochs(24));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "the publish did not accept; status = {status:?}"
    );
    let in_state = c.run_until(epochs(2), |c| {
        c.substate(ShardId::leaf(1, 0), cell.owner, cell.local.0)
            .is_some()
    });
    assert!(in_state, "the package cell never reached persisted state");

    // Before the beacon has heard of it at all. A rule that refused only
    // the registered and immature would have to let this through on an
    // argument about who holds what metadata — true of a real network,
    // and unavailable to any node as a local check. The rule refuses it
    // for the one reason every node can check alike: the registry does
    // not list it.
    let unregistered =
        build_instance_deposit_tx(key, &artifact, SALT, DEPOSIT, validity_around(c.now()));
    let unregistered_hash = unregistered.hash();
    c.submit(Arc::new(unregistered));
    c.run_until(epochs(8), |c| {
        c.tx_status(unregistered_hash).is_some_and(|s| s.is_final())
            || c.beacon_state()
                .is_some_and(|state| state.packages.contains_key(&registered))
    });
    let before_registry = c.tx_status(unregistered_hash);
    assert!(
        !before_registry
            .as_ref()
            .is_some_and(TransactionStatus::is_final),
        "a call was decided before the beacon registered its package: {before_registry:?}"
    );
    assert!(
        c.beacon_state()
            .is_some_and(|state| state.packages.contains_key(&registered)),
        "the beacon never registered the publish"
    );

    // Registered, still inside the window. Every honest proposer filters
    // it and every honest voter would refuse it, so it waits — which is
    // the whole of the guarantee, since a transaction committed here
    // could be handed to a node whose fetch had not landed.
    let early = build_instance_deposit_tx(key, &artifact, SALT, DEPOSIT, validity_around(c.now()));
    let early_hash = early.hash();
    c.submit(Arc::new(early));
    c.run_until(epochs(8), |c| {
        c.tx_status(early_hash).is_some_and(|s| s.is_final())
            || c.beacon_state().is_some_and(|state| {
                state
                    .packages
                    .get(&registered)
                    .is_some_and(|fact| fact.usable_in(state.current_epoch))
            })
    });
    let held = c.tx_status(early_hash);
    assert!(
        !held.as_ref().is_some_and(TransactionStatus::is_final),
        "a call was decided while its package was still maturing: {held:?}"
    );

    // Past the window. The same call settles now, and settling it means
    // the code reached the nodes that never committed the publish —
    // every shard the transaction touches runs the whole of it.
    let late = build_instance_deposit_tx(key, &artifact, SALT, DEPOSIT, validity_around(c.now()));
    let late_hash = late.hash();
    c.submit(Arc::new(late));
    let status = await_tx_terminal(c, late_hash, epochs(24));
    assert!(
        matches!(
            status,
            Some(TransactionStatus::Completed(TransactionDecision::Accept))
        ),
        "a matured package's call did not settle; status = {status:?}"
    );
}

/// An adversarial deploy storm rides out: throughput degrades, no shard
/// stalls.
///
/// Every publisher spams distinct packages at its own shard at once, so
/// both committees are simultaneously carrying the heaviest transaction
/// the protocol admits — a full artifact each, validated at admission and
/// written whole into state. This is the probe the commit-fed cache has
/// to survive: if publishing could wedge a committee, or if the cache
/// feed could make a shard's commit path fall behind its consensus, this
/// is where it would show.
///
/// The assertion is deliberately about liveness rather than latency.
/// Every publish reaching a terminal decision is itself the anti-stall
/// proof — a wedged committee settles nothing — and both shards' heights
/// advancing past the storm says the chains never stopped.
///
/// # Panics
///
/// Panics if any publish fails to settle, if a publish did not commit on
/// its publisher's shard, or if either shard's chain failed to advance.
pub fn deploy_storm_rides_out(c: &mut impl Cluster) {
    const PER_PUBLISHER: u16 = 6;

    let publishers = storm_publishers();
    let shards = [ShardId::leaf(1, 0), ShardId::leaf(1, 1)];
    let before: Vec<Option<BlockHeight>> = shards
        .iter()
        .map(|shard| c.committed_height(*shard))
        .collect();

    let validity = validity_around(c.now());
    let mut submitted: Vec<(TxHash, ShardId)> = Vec::new();
    let mut cells: Vec<(ShardId, Address, [u8; 16])> = Vec::new();
    for (index, (key, publisher)) in (0u16..).zip(publishers.iter()) {
        for nonce in 0..PER_PUBLISHER {
            // Distinct per publisher as well as per nonce, so the two
            // shards never race to publish one content address.
            let artifact = storm_artifact(nonce + index * 1_000);
            let cell = package_key(*publisher, package_hash(&ProtocolHasher, &artifact));
            let tx = build_publish_tx(key, artifact, validity);
            let shard = shards[usize::from(index)];
            submitted.push((tx.hash(), shard));
            cells.push((shard, cell.owner, cell.local.0));
            c.submit(Arc::new(tx));
        }
    }
    assert_eq!(
        cells
            .iter()
            .map(|(_, owner, local)| (*owner, *local))
            .collect::<BTreeSet<_>>()
            .len(),
        cells.len(),
        "the storm must deploy distinct packages, or it is one publish repeated"
    );

    for (hash, shard) in &submitted {
        let status = await_tx_terminal(c, *hash, epochs(24));
        assert!(
            matches!(
                status,
                Some(TransactionStatus::Completed(TransactionDecision::Accept))
            ),
            "a publish in the storm did not accept; status = {status:?}"
        );
        let (fate, _) = c.chain_fate(*shard, *hash);
        assert!(
            fate.is_some(),
            "the publisher's shard never committed its own publish"
        );
    }

    // Every package the storm deployed is in state, which is what makes
    // the distinctness above load-bearing: idempotent duplicates would
    // collapse into one cell and the storm would be a single publish.
    // The read is served from the persisted frontier, which trails the
    // commit that settled the transaction, so presence is awaited like
    // every other observation here rather than asserted at an instant.
    let all_present = c.run_until(epochs(2), |c| {
        cells
            .iter()
            .all(|(shard, owner, local)| c.substate(*shard, *owner, *local).is_some())
    });
    assert!(
        all_present,
        "a package cell the storm published never reached persisted state"
    );

    for (shard, height) in shards.iter().zip(before) {
        let after = c.committed_height(*shard);
        assert!(
            after > height,
            "{shard:?} did not advance through the storm: {height:?} -> {after:?}"
        );
    }
}
