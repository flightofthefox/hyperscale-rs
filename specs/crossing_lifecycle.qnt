// Model L (register: specs/README.md) — the crossing lifecycle: a value
// crossing is issued by one shard, claimed by another, or reclaimed by its
// issuer, and it is disposed at most once (INV-EXEC-11) and never left
// with no admissible claimant (INV-EXEC-12).
//
// Two instances, split on the invariant, two shards each. `exclusivity`
// is the producer and the core: disposal is claim-or-reclaim and the
// delivery leg never appears. `delivery` is the core and the recipient:
// what is modelled is whether an issued outbound crossing is always
// disposed of — claimed inside its window, or reclaimed by its issuer
// once the delivery lapsed. Shard consensus is an oracle (Model B):
// each shard is a monotone attested clock plus ordered commits, and a
// committee is honest (INV-SEC-1). Model A justifies treating the
// attested chain clock as monotone.
//
// Code anchors:
//   - The five evidence forms a reclaim is licensed by, and nothing else:
//     a refusal certificate of the core's, a departure record, an absence
//     proof of the core's committed cell anchored no earlier than the
//     probe anchor, an absence proof of a one-shard core consumer's claim
//     cell anchored no earlier than the same, or an absence proof of the
//     delivery's claim cell anchored no earlier than the lapse —
//     `reclaimable` / `probe_silent_counterparts` in crates/execution/src/coordinator.rs,
//     `Unsettleable` in crates/types/src/shard/abandonment.rs. The
//     departure arm is Model D's and is not modelled here.
//   - The probe anchor, D = validity_end + MAX_FINALIZATION_DELAY —
//     `reclaim_probe_anchor` / `absence_licenses_reclaim` in
//     crates/types/src/time/timeouts.rs; the probe is gated on the
//     producer's own clock passing D (`Unresolved::probeable`) and reads a
//     commit-proven core header whose clock is past D.
//   - What the probe reads: the core's committed cell for the transaction,
//     written when the block carrying it commits — `committed_tx_cells` in
//     crates/storage/src/shard/sweep.rs, proved by
//     `MerkleInclusionProof::inclusions` in crates/types/src/provisioning/proof.rs.
//     A block is admitted at an anchor inside the validity window and
//     commits within MAX_FINALIZATION_DELAY of it, which is what puts every
//     commit before D. Or, for a core of one shard, the consumer's claim
//     cell, written when the block carrying the core's success commits —
//     a block that would do so at or past D is refused
//     (`Resolutions::Overdue`, crates/shard/src/action_handlers.rs), which
//     is `FENCED` here, and what lets the claim's absence past D be read.
//     A core of more shards settles on its siblings' clock and is not
//     fenced, so its answer stays the committed cell (`LICENCE_CLAIM`).
//   - Once-only disposal on one shard — the claim-cell checks in
//     vm/crates/kernel/src/executor.rs (`crossing_committed`). Across two
//     shards nothing reads the other's cell, so exclusivity rests on the
//     evidence rule alone, which is what this model checks.
//   - The outbound claim's lifetime — `delivery_admissible` /
//     `delivery_window_close` in crates/types/src/time/timeouts.rs: the
//     deadline opens the late-delivery window and MAX_VALIDITY_RANGE
//     closes it; `late_deliveries` in crates/shard/src/proposal.rs admits
//     a delivery past its validity end to that close.
//   - The lapse anchor, L = close + MAX_FINALIZATION_DELAY —
//     `lapse_probe_anchor` / `lapse_licenses_reclaim` in
//     crates/types/src/time/timeouts.rs: a claim admitted under the close
//     has committed by L or never will, so the claim cell's absence at a
//     recipient block past L is the lapse. The record the reclaim takes
//     back outlives L outright: no sweep reaches a record cell.
//   - Both licenses are half-open windows, not floors: the cells a probe
//     asks about are swept at their own expiry — `committed_tx_expiry_ms`
//     in crates/storage/src/shard/sweep.rs for the committed cell,
//     `CROSSING_GRACE_MS` for the claim cell — and a non-inclusion proof of
//     a swept cell is honest and says nothing. `absence_licenses_reclaim`
//     and `lapse_licenses_reclaim` carry the ceiling; `sweep` and
//     `sweepClaim` here are what make it load-bearing.
//
// Modeling assumptions:
//   - One crossing. Clocks are per-shard monotone integers advancing by
//     1..MAX_TICK per step and stopping at MAX_CLOCK, so a state with both
//     clocks at MAX_CLOCK and no enabled transition is terminal.
//   - Certificates and bundles travel asynchronously in `exclusivity`,
//     within MAX_SKEW between the two shards' clocks. The skew bound is
//     what the absence window's ceiling costs: a producer whose core's
//     chain has already run past the window never gets to ask, and the
//     crossing strands — the outcome the design names and the claim
//     cell's life bounds, not a disposal the model can promise. The window's own
//     width is the figure, since any drift inside it leaves the producer a
//     licensed header to ask at.
//   - A probe is taken while it can be: the clock does not leave either
//     window with an unproved absence pending, which is the prober asking
//     at every licensed header it holds rather than a fairness assumption. In `delivery` the two
//     shards share one clock, a bundle not lost is delivered within
//     DELIVERY_LAG of its issue, an honest recipient admits a delivered
//     claim within CLAIM_LAG, and an admitted claim commits within the
//     finalization delay — the outbound window is a bound on the first
//     two, and a bundle may be lost outright, which is what the lapse
//     reclaim disposes of.
//   - The core's verdict is claim or refuse, nondeterministic, once it has
//     committed the transaction; a refusal reaches the producer
//     asynchronously.
//
// Run (quint 0.32.0, Apalache 0.56.1, OpenJDK 17; constants as in the
// instances below: DEADLINE = 2, FINALIZATION_DELAY = 1, MAX_TICK = 2,
// MAX_CLOCK = 5 in `exclusivity`, 8 in `delivery`):
//   quint typecheck specs/crossing_lifecycle.qnt
//   # twin floors, each confirmed clean at floor-1 and violating at floor:
//   quint verify --main=clock_fenced_reclaim --invariant=invExclusive  --max-steps=7  specs/crossing_lifecycle.qnt
//   quint verify --main=early_absence_probe  --invariant=invExclusive  --max-steps=11 specs/crossing_lifecycle.qnt
//   quint verify --main=late_absence_probe   --invariant=invExclusive  --max-steps=13 specs/crossing_lifecycle.qnt
//   quint verify --main=no_reclaim           --invariant=invDisposable --max-steps=9  specs/crossing_lifecycle.qnt
//   quint verify --main=unfenced_claim_commit --invariant=invExclusive --max-steps=11 specs/crossing_lifecycle.qnt
//   quint verify --main=deadline_on_delivery --invariant=invDelivered  --max-steps=11 specs/crossing_lifecycle.qnt
//   quint verify --main=early_lapse_probe    --invariant=invExclusive  --max-steps=12 specs/crossing_lifecycle.qnt
//   quint verify --main=late_lapse_probe     --invariant=invExclusive  --max-steps=12 specs/crossing_lifecycle.qnt
//   quint verify --main=no_lapse_reclaim     --invariant=invDisposed   --max-steps=9  specs/crossing_lifecycle.qnt
//   # healthy instances, as deep as an affordable run reaches (wall-clock
//   # on an Apple M-series laptop: exclusivity 6s at 10, 12s at 12, 26s at
//   # 14, 109s at 16; delivery about 5s through 14, 5s at 16):
//   quint verify --main=exclusivity_ok --invariant=invAll --max-steps=16 specs/crossing_lifecycle.qnt
//   quint verify --main=claim_licence_ok --invariant=invAll --max-steps=16 specs/crossing_lifecycle.qnt
//   quint verify --main=delivery_ok    --invariant=invAll --max-steps=16 specs/crossing_lifecycle.qnt
//   # random search walks a twin up to its floor sub-second per point:
//   quint run --main=<twin> --invariant=<inv> --max-steps=N --max-samples=20000 --seed=7 specs/crossing_lifecycle.qnt
// Give every concurrent `verify` its own `--server-endpoint=localhost:<port>`.

module exclusivity_core {
  // The transaction's validity end, T: the producer issues before it and
  // the core admits the block carrying the transaction before it.
  const DEADLINE: int
  // MAX_FINALIZATION_DELAY: a block admitted at anchor a commits by
  // a + FINALIZATION_DELAY, so D = DEADLINE + FINALIZATION_DELAY is past
  // every commit.
  const FINALIZATION_DELAY: int
  // The earliest core clock an absence proof may be anchored at. Healthy:
  // DEADLINE + FINALIZATION_DELAY. Twin `early_absence_probe`: DEADLINE.
  const PROBE_ANCHOR: int
  // The core clock the absence license stops at. An absence proof is
  // evidence only while the cell it asks about still exists, so the
  // healthy ceiling is the sweep itself. Twin `late_absence_probe`: past
  // MAX_CLOCK, which is the floor with no ceiling.
  const PROBE_CEILING: int
  // The core clock at which the committed cell is retired. The cell
  // carries an expiry and the frontier walk removes it there, so a probe
  // anchored past this reads absent whatever the core committed.
  const SWEEP_AT: int
  // Twin `clock_fenced_reclaim`: the producer reclaims on its own clock
  // passing the deadline, with no evidence.
  const RECLAIM_ON_CLOCK: bool
  // Twin `no_reclaim`: the producer never reclaims.
  const RECLAIM_ENABLED: bool
  // The licence the producer reads. False: the core's committed cell,
  // absent past the anchor — what a core of more than one shard answers
  // with, since its claim commits on its siblings' clock. True: the core
  // consumer's claim cell, absent past the anchor — what a core of one
  // shard answers with, whose one execution wrote the claim by the
  // deadline or never will.
  const LICENCE_CLAIM: bool
  // Whether a block committing the core's claim at or past PROBE_ANCHOR
  // is refused. What makes the claim licence sound. Twin
  // `unfenced_claim_commit`: the claim licence without the fence.
  const FENCED: bool
  // How far the two chains' attested clocks may drift apart. The absence
  // license has a ceiling, so a producer whose core's chain has already
  // run past the window never gets to ask: the crossing strands, which is
  // the outcome the design names and the claim cell's life bounds. What
  // this bounds is the drift, and the window's own width is the figure —
  // any skew inside it leaves the producer a licensed header to ask at.
  const MAX_SKEW: int
  // Max clock advance per step, and where clocks stop.
  const MAX_TICK: int
  const MAX_CLOCK: int

  pure val PRODUCER = 1
  pure val CORE = 2
  pure val SHARDS = Set(PRODUCER, CORE)
  pure val NONE = -1

  // Per-shard attested chain clock (monotone).
  var clock: int -> int
  // The producer's leg ran and wrote the record cell.
  var issued: bool
  // The anchor the core's block carrying the transaction was admitted at
  // (NONE before): the block exists but its committed cell does not yet.
  var proposedAt: int
  // The core clock at which that block committed (NONE before): the
  // committed cell the probe reads exists from here.
  var committedAt: int
  // The core's verdict on the crossing: claimed or refused (once). A
  // claim is the execution's; its cell is in the core's committed state
  // only from claimCommittedAt.
  var claimed: bool
  var claimCommittedAt: int
  var refused: bool
  // The refusal certificate reached the producer.
  var refusalDelivered: bool
  // The committed cell has been swept out of the core's state.
  var swept: bool
  // The producer holds an absence proof: a commit-proven core header
  // inside the absence window whose committed set lacks the transaction.
  var absenceProved: bool
  // The producer took the crossing back.
  var reclaimed: bool

  action init = all {
    clock' = SHARDS.mapBy(s => 0),
    issued' = false,
    proposedAt' = NONE,
    committedAt' = NONE,
    claimed' = false,
    refused' = false,
    refusalDelivered' = false,
    swept' = false,
    absenceProved' = false,
    reclaimed' = false,
    claimCommittedAt' = NONE,
  }

  action unchangedButClock = all {
    issued' = issued, proposedAt' = proposedAt, committedAt' = committedAt,
    claimed' = claimed, refused' = refused, refusalDelivered' = refusalDelivered,
    swept' = swept, absenceProved' = absenceProved, reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }

  action tick = {
    nondet s = SHARDS.oneOf()
    nondet d = 1.to(MAX_TICK).oneOf()
    all {
      clock.get(s) + d <= MAX_CLOCK,
      // The two chains stay within MAX_SKEW of each other.
      SHARDS.forall(t => clock.get(s) + d - clock.get(t) <= MAX_SKEW),
      // The producer asks while it can. A crossing issued whose core has
      // neither claimed nor refused is probed at every licensed header the
      // producer holds, and the core produces one every block — so the
      // core's clock does not leave the absence window with the proof
      // untaken.
      not(issued) or absenceProved or refused or s != CORE
        or (if (LICENCE_CLAIM) claimCellPresent else (cellPresent or claimed))
        or clock.get(CORE) + d < PROBE_CEILING,
      clock' = clock.setBy(s, c => c + d),
      unchangedButClock,
    }
  }

  // The producer's leg runs inside the validity window and issues the
  // crossing: the record cell is written by its own verdict.
  val canIssue = not(issued) and clock.get(PRODUCER) < DEADLINE
  action issue = all {
    canIssue,
    issued' = true,
    clock' = clock, proposedAt' = proposedAt, committedAt' = committedAt,
    claimed' = claimed, refused' = refused, refusalDelivered' = refusalDelivered,
    swept' = swept, absenceProved' = absenceProved, reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }

  // The core's chain admits the block carrying the transaction, at an
  // anchor inside the validity window.
  val canPropose = proposedAt == NONE and clock.get(CORE) < DEADLINE
  action propose = all {
    canPropose,
    proposedAt' = clock.get(CORE),
    clock' = clock, issued' = issued, committedAt' = committedAt,
    claimed' = claimed, refused' = refused, refusalDelivered' = refusalDelivered,
    swept' = swept, absenceProved' = absenceProved, reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }

  // That block commits, within the finalization delay of its anchor: the
  // committed cell exists from here. A block that misses the delay never
  // commits.
  val canCommit = proposedAt != NONE and committedAt == NONE
    and clock.get(CORE) <= proposedAt + FINALIZATION_DELAY
  action commit = all {
    canCommit,
    committedAt' = clock.get(CORE),
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    claimed' = claimed, refused' = refused, refusalDelivered' = refusalDelivered,
    swept' = swept, absenceProved' = absenceProved, reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }

  // The core's verdict, once it holds the crossing: it claims, or it
  // refuses. A core that committed and never received the crossing
  // refuses at its own deadline; both are one refusal here.
  val canClaim = committedAt != NONE and issued and not(claimed) and not(refused)
  action claim = all {
    canClaim,
    claimed' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, refused' = refused,
    refusalDelivered' = refusalDelivered, swept' = swept,
    absenceProved' = absenceProved, reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }
  // The block carrying the core's success commits, and the claim cell
  // is in its state from here. Fenced at the probe anchor: past it the
  // block is refused, the core's tick aborts, and the claim never
  // commits — which is what lets its absence past the anchor be read.
  val canCommitClaim = claimed and claimCommittedAt == NONE
    and (not(FENCED) or clock.get(CORE) < PROBE_ANCHOR)
  action commitClaim = all {
    canCommitClaim,
    claimCommittedAt' = clock.get(CORE),
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed, refused' = refused,
    refusalDelivered' = refusalDelivered, swept' = swept,
    absenceProved' = absenceProved, reclaimed' = reclaimed,
  }
  val canRefuse = committedAt != NONE and not(claimed) and not(refused)
  action refuse = all {
    canRefuse,
    refused' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed,
    refusalDelivered' = refusalDelivered, swept' = swept,
    absenceProved' = absenceProved, reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }

  // The refusal certificate reaches the producer, whenever the network
  // delivers it.
  val canDeliverRefusal = refused and not(refusalDelivered)
  action deliverRefusal = all {
    canDeliverRefusal,
    refusalDelivered' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed, refused' = refused,
    swept' = swept, absenceProved' = absenceProved, reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }

  // The core's frontier retires the committed cell at its expiry. Past
  // this a non-inclusion proof of the cell is honest and says nothing
  // about what the core committed.
  val canSweep = not(swept) and clock.get(CORE) >= SWEEP_AT
  action sweep = all {
    canSweep,
    swept' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed, refused' = refused,
    refusalDelivered' = refusalDelivered, absenceProved' = absenceProved,
    reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }

  // The producer, past D on its own clock, reads the core's committed set
  // off a commit-proven header inside the absence window. Absence at that
  // header is the proof; presence answers the probe with nothing. The cell
  // is absent when the core never committed the transaction — and, once
  // the sweep has run, when it did, which is what the ceiling is for.
  val cellPresent = committedAt != NONE and not(swept)
  val claimCellPresent = claimCommittedAt != NONE
  // What the probe reads present, under the licence in force.
  val answerPresent = if (LICENCE_CLAIM) claimCellPresent else cellPresent
  val canProbe = issued and not(absenceProved)
    and clock.get(PRODUCER) >= DEADLINE + FINALIZATION_DELAY
    and clock.get(CORE) >= PROBE_ANCHOR
    and clock.get(CORE) < PROBE_CEILING
    and not(answerPresent)
  action probe = all {
    canProbe,
    absenceProved' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed, refused' = refused,
    refusalDelivered' = refusalDelivered, swept' = swept, reclaimed' = reclaimed,
    claimCommittedAt' = claimCommittedAt,
  }

  // The reclaim, on the evidence the rule licenses and nothing else. The
  // twin adds the producer's own clock as a license.
  val licensed = refusalDelivered or absenceProved
    or (RECLAIM_ON_CLOCK and clock.get(PRODUCER) >= DEADLINE)
  val canReclaim = RECLAIM_ENABLED and issued and not(reclaimed) and licensed
  action reclaim = all {
    canReclaim,
    reclaimed' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed, refused' = refused,
    refusalDelivered' = refusalDelivered, swept' = swept,
    absenceProved' = absenceProved,
    claimCommittedAt' = claimCommittedAt,
  }

  // Nothing but the clocks can move, and the clocks have stopped.
  val quiescent = SHARDS.forall(s => clock.get(s) == MAX_CLOCK)
    and not(canIssue) and not(canPropose) and not(canCommit)
    and not(canClaim) and not(canCommitClaim) and not(canRefuse) and not(canDeliverRefusal)
    and not(canSweep) and not(canProbe) and not(canReclaim)

  // A terminal state stutters: the model ends there, not in a deadlock.
  action idle = all {
    quiescent,
    clock' = clock,
    unchangedButClock,
  }

  action step = any {
    tick, issue, propose, commit, claim, commitClaim, refuse, deliverRefusal,
    sweep, probe, reclaim, idle
  }

  // INV-EXEC-11: a crossing is disposed at most once. A claim the core
  // executed but never committed disposed of nothing: its block was
  // refused, and the value stays in the record for the reclaim.
  val invExclusive = not(claimCellPresent and reclaimed)

  // INV-EXEC-12: at a terminal state an issued crossing is disposed.
  val invDisposable = quiescent implies (not(issued) or claimCellPresent or reclaimed)

  val invAll = invExclusive and invDisposable
}

module delivery_core {
  // The transaction's validity end, T. The core admits the block carrying
  // the transaction before it and its verdict, the outbound crossing,
  // commits within the finalization delay.
  const DEADLINE: int
  const FINALIZATION_DELAY: int
  // The recipient may admit a claim of the outbound crossing while its
  // clock is below this. Healthy: DEADLINE + MAX_VALIDITY_RANGE. Twin
  // `deadline_on_delivery`: DEADLINE.
  const DELIVERY_CLOSE: int
  // The earliest recipient anchor at which the claim cell's absence
  // proves the crossing lapsed. Healthy: DELIVERY_CLOSE +
  // FINALIZATION_DELAY, past every commit a claim admitted under the
  // close can reach. Twin `early_lapse_probe`: DELIVERY_CLOSE.
  const LAPSE_ANCHOR: int
  // The anchor the lapse license stops at: the claim cell's own sweep,
  // past which its absence says nothing. Twin `late_lapse_probe`: past
  // MAX_CLOCK, which is the floor with no ceiling.
  const LAPSE_CEILING: int
  // The recipient clock at which the claim cell is retired.
  const CLAIM_SWEEP_AT: int
  // Whether the issuer reclaims on a proved lapse. Twin `no_lapse_reclaim`:
  // false, and a lost bundle strands the crossing.
  const RECLAIM_ENABLED: bool
  // A bundle that is not lost reaches the recipient within this many
  // clock units of its issue, and the recipient's chain admits a
  // delivered claim within this many of its delivery: the honest-committee
  // liveness the window is sized against.
  const DELIVERY_LAG: int
  const CLAIM_LAG: int
  const MAX_TICK: int
  const MAX_CLOCK: int

  pure val NONE = -1

  // One clock: the two shards' attested clocks read alike here, since
  // what is checked is a window against a bounded delivery, not skew.
  var clock: int
  // The core's block carrying the transaction was admitted (anchor).
  var proposedAt: int
  // The core's verdict committed, issuing the outbound crossing.
  var issuedAt: int
  // The bundle carrying the crossing reached the recipient.
  var deliveredAt: int
  // The bundle was dropped for good: a recipient cut off, or frozen.
  var lost: bool
  // The recipient admitted its claim (anchor).
  var claimedAt: int
  // The claim committed: the claim cell is in the recipient's state.
  var claimCommitted: bool
  // The claim cell has been swept out of the recipient's state.
  var claimSwept: bool
  // The issuer holds a proof of the claim cell's absence past the lapse.
  var lapseProved: bool
  // The issuer took the crossing back.
  var reclaimed: bool

  action init = all {
    clock' = 0, proposedAt' = NONE, issuedAt' = NONE, deliveredAt' = NONE,
    lost' = false, claimedAt' = NONE, claimCommitted' = false,
    claimSwept' = false, lapseProved' = false, reclaimed' = false,
  }

  action unchangedButClock = all {
    proposedAt' = proposedAt, issuedAt' = issuedAt, deliveredAt' = deliveredAt,
    lost' = lost, claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    claimSwept' = claimSwept, lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  action tick = {
    nondet d = 1.to(MAX_TICK).oneOf()
    all {
      clock + d <= MAX_CLOCK,
      // A bundle in flight and not lost lands within its lag; a delivered
      // claim still admissible is admitted within its own; and an admitted
      // claim commits within the finalization delay. The clock does not
      // run past any of these while the step it bounds is pending. A
      // claim the window has already closed on binds nothing.
      issuedAt == NONE or deliveredAt != NONE or lost or clock + d <= issuedAt + DELIVERY_LAG,
      deliveredAt == NONE or claimedAt != NONE or clock >= DELIVERY_CLOSE
        or clock + d <= deliveredAt + CLAIM_LAG,
      claimedAt == NONE or claimCommitted or clock + d <= claimedAt + FINALIZATION_DELAY,
      // The issuer asks while it can: a crossing whose claim cell is
      // absent and unproved is probed at every licensed anchor, so the
      // clock does not leave the lapse window with the proof untaken.
      issuedAt == NONE or lapseProved or claimCellPresent
        or clock + d < LAPSE_CEILING,
      clock' = clock + d,
      unchangedButClock,
    }
  }

  val canPropose = proposedAt == NONE and clock < DEADLINE
  action propose = all {
    canPropose,
    proposedAt' = clock,
    clock' = clock, issuedAt' = issuedAt, deliveredAt' = deliveredAt, lost' = lost,
    claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    claimSwept' = claimSwept, lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  val canIssue = proposedAt != NONE and issuedAt == NONE
    and clock <= proposedAt + FINALIZATION_DELAY
  action issue = all {
    canIssue,
    issuedAt' = clock,
    clock' = clock, proposedAt' = proposedAt, deliveredAt' = deliveredAt, lost' = lost,
    claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    claimSwept' = claimSwept, lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  val canDeliver = issuedAt != NONE and deliveredAt == NONE and not(lost)
  action deliver = all {
    canDeliver,
    deliveredAt' = clock,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt, lost' = lost,
    claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    claimSwept' = claimSwept, lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  // The bundle is dropped for good: nothing delivers it after this.
  val canLose = issuedAt != NONE and deliveredAt == NONE and not(lost)
  action lose = all {
    canLose,
    lost' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    claimSwept' = claimSwept, lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  // The recipient admits a claim of a delivered crossing while the window
  // is open.
  val canClaim = deliveredAt != NONE and claimedAt == NONE and clock < DELIVERY_CLOSE
  action claim = all {
    canClaim,
    claimedAt' = clock,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimCommitted' = claimCommitted,
    claimSwept' = claimSwept, lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  // An admitted claim commits within the finalization delay.
  val canCommitClaim = claimedAt != NONE and not(claimCommitted)
    and clock <= claimedAt + FINALIZATION_DELAY
  action commitClaim = all {
    canCommitClaim,
    claimCommitted' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimedAt' = claimedAt,
    claimSwept' = claimSwept, lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  // The recipient's frontier retires the claim cell at its grace. Past
  // this a non-inclusion proof of the cell is honest and says nothing
  // about whether the delivery claimed.
  val canSweepClaim = not(claimSwept) and clock >= CLAIM_SWEEP_AT
  action sweepClaim = all {
    canSweepClaim,
    claimSwept' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimedAt' = claimedAt,
    claimCommitted' = claimCommitted, lapseProved' = lapseProved,
    reclaimed' = reclaimed,
  }

  // The issuer reads the recipient's committed state at a block inside the
  // lapse window: the claim cell is absent when no claim has committed —
  // and, once the sweep has run, when one has, which is what the ceiling
  // is for. A claim admitted but not yet committed reads as absent, which
  // is what the anchor's finalization delay exists to put behind it.
  val claimCellPresent = claimCommitted and not(claimSwept)
  val canProbe = issuedAt != NONE and not(lapseProved) and not(claimCellPresent)
    and clock >= LAPSE_ANCHOR and clock < LAPSE_CEILING
  action probeLapse = all {
    canProbe,
    lapseProved' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimedAt' = claimedAt,
    claimCommitted' = claimCommitted, claimSwept' = claimSwept,
    reclaimed' = reclaimed,
  }

  val canReclaim = RECLAIM_ENABLED and issuedAt != NONE and lapseProved and not(reclaimed)
  action reclaim = all {
    canReclaim,
    reclaimed' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimedAt' = claimedAt,
    claimCommitted' = claimCommitted, claimSwept' = claimSwept,
    lapseProved' = lapseProved,
  }

  val quiescent = clock == MAX_CLOCK
    and not(canPropose) and not(canIssue) and not(canDeliver) and not(canLose)
    and not(canClaim) and not(canCommitClaim) and not(canSweepClaim)
    and not(canProbe) and not(canReclaim)

  // A terminal state stutters: the model ends there, not in a deadlock.
  action idle = all {
    quiescent,
    clock' = clock,
    unchangedButClock,
  }

  action step = any {
    tick, propose, issue, deliver, lose, claim, commitClaim, sweepClaim,
    probeLapse, reclaim, idle
  }

  // INV-EXEC-11 across the delivery: the crossing is claimed by its
  // recipient or reclaimed by its issuer, never both.
  val invExclusive = not(claimCommitted and reclaimed)
  // INV-EXEC-12, the outbound half: at a terminal state an issued crossing
  // has been disposed of — claimed, or reclaimed once its delivery lapsed.
  val invDisposed = quiescent implies (issuedAt == NONE or claimCommitted or reclaimed)
  // And a bundle that was delivered is claimed: the window is wide enough
  // for an honest recipient to claim what reaches it.
  val invDelivered = quiescent implies (issuedAt == NONE or lost or claimCommitted)

  val invAll = invExclusive and invDisposed and invDelivered
}

// Healthy exclusivity: the reclaim is licensed by a refusal or an absence
// proof anchored past D, and by nothing else.
module exclusivity_ok {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3, PROBE_CEILING = 5,
    SWEEP_AT = 5, RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = true,
    LICENCE_CLAIM = false, FENCED = false,
    MAX_SKEW = 2, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Healthy claim licence: a core of one shard, whose claim commits under
// the fence or never. The producer reads the claim cell absent past D.
module claim_licence_ok {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3, PROBE_CEILING = 5,
    SWEEP_AT = 5, RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = true,
    LICENCE_CLAIM = true, FENCED = true,
    MAX_SKEW = 2, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Twin: the claim licence without the fence. The core claims under the
// deadline, its block commits past the anchor, and the producer — having
// read the claim absent in between — has reclaimed what the core took.
module unfenced_claim_commit {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3, PROBE_CEILING = 5,
    SWEEP_AT = 5, RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = true,
    LICENCE_CLAIM = true, FENCED = false,
    MAX_SKEW = 2, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Twin: the producer reclaims on its own clock passing the deadline. The
// core, on a clock behind the producer's, commits and claims after it.
module clock_fenced_reclaim {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3, PROBE_CEILING = 5,
    SWEEP_AT = 5, RECLAIM_ON_CLOCK = true, RECLAIM_ENABLED = true,
    LICENCE_CLAIM = false, FENCED = false,
    MAX_SKEW = 2, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Twin: the absence proof may be anchored at the deadline. A block
// admitted just under it commits after the anchor, and the core claims
// what the producer has reclaimed.
module early_absence_probe {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 2, PROBE_CEILING = 5,
    SWEEP_AT = 5, RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = true,
    LICENCE_CLAIM = false, FENCED = false,
    MAX_SKEW = 2, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Twin: the absence license is a floor with no ceiling. The core commits
// and claims, its committed cell is swept at its expiry, and the honest
// non-inclusion proof of a cell that is simply gone licenses the reclaim
// of a crossing the core took.
module late_absence_probe {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3, PROBE_CEILING = 6,
    SWEEP_AT = 5, RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = true,
    LICENCE_CLAIM = false, FENCED = false,
    MAX_SKEW = 2, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Twin: the producer never reclaims. A refused crossing is left issued
// with nobody able to claim it.
module no_reclaim {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3, PROBE_CEILING = 5,
    SWEEP_AT = 5, RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = false,
    LICENCE_CLAIM = false, FENCED = false,
    MAX_SKEW = 2, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Healthy delivery: the outbound claim is admissible to the delivery
// window's close, past the deadline by more than the finalization delay
// and the delivery lag together; a lost bundle lapses, proved absent
// past the close plus the finalization delay, and the issuer reclaims.
module delivery_ok {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 5, LAPSE_ANCHOR = 6,
    LAPSE_CEILING = 8, CLAIM_SWEEP_AT = 8, RECLAIM_ENABLED = true,
    DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}

// Twin: the outbound claim expires at the deadline. A crossing issued
// just under it, or delivered just after, reaches a recipient that can
// no longer claim it.
module deadline_on_delivery {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 2, LAPSE_ANCHOR = 3,
    LAPSE_CEILING = 8, CLAIM_SWEEP_AT = 8, RECLAIM_ENABLED = true,
    DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}

// Twin: the lapse is proved at the window's close itself. A claim
// admitted just under the close commits after the anchor, and the issuer
// reclaims what the recipient has claimed.
module early_lapse_probe {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 5, LAPSE_ANCHOR = 5,
    LAPSE_CEILING = 8, CLAIM_SWEEP_AT = 8, RECLAIM_ENABLED = true,
    DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}

// Twin: the lapse license is a floor with no ceiling. The recipient
// claims inside its window, the claim cell is swept at its grace, and the
// honest non-inclusion proof of a cell that is simply gone licenses the
// issuer to take back what the recipient holds.
module late_lapse_probe {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 5, LAPSE_ANCHOR = 6,
    LAPSE_CEILING = 9, CLAIM_SWEEP_AT = 8, RECLAIM_ENABLED = true,
    DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}

// Twin: the issuer never reclaims. A lost bundle leaves the crossing
// issued with nobody able to claim it.
module no_lapse_reclaim {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 5, LAPSE_ANCHOR = 6,
    LAPSE_CEILING = 8, CLAIM_SWEEP_AT = 8, RECLAIM_ENABLED = false,
    DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}
