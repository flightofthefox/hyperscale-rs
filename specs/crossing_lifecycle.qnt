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
//   - The four evidence forms a reclaim is licensed by, and nothing else:
//     a refusal certificate of the core's, a departure record, an absence
//     proof of the core's committed cell anchored no earlier than the
//     probe anchor, or an absence proof of the delivery's claim cell
//     anchored no earlier than the lapse —
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
//     commit before D.
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
//     back outlives L by a validity range (`ESCROW_GRACE_MS`).
//
// Modeling assumptions:
//   - One crossing. Clocks are per-shard monotone integers advancing by
//     1..MAX_TICK per step and stopping at MAX_CLOCK, so a state with both
//     clocks at MAX_CLOCK and no enabled transition is terminal.
//   - Certificates and bundles travel asynchronously: any interleaving
//     between the shards' clocks in `exclusivity`. In `delivery` the two
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
//   quint verify --main=clock_fenced_reclaim --invariant=invExclusive  --max-steps=6  specs/crossing_lifecycle.qnt
//   quint verify --main=early_absence_probe  --invariant=invExclusive  --max-steps=10 specs/crossing_lifecycle.qnt
//   quint verify --main=no_reclaim           --invariant=invDisposable --max-steps=8  specs/crossing_lifecycle.qnt
//   quint verify --main=deadline_on_delivery --invariant=invDelivered  --max-steps=10 specs/crossing_lifecycle.qnt
//   quint verify --main=early_lapse_probe    --invariant=invExclusive  --max-steps=12 specs/crossing_lifecycle.qnt
//   quint verify --main=no_lapse_reclaim     --invariant=invDisposed   --max-steps=8  specs/crossing_lifecycle.qnt
//   # healthy instances, as deep as an affordable run reaches (wall-clock
//   # on an Apple M-series laptop: exclusivity 6s at 10, 12s at 12, 26s at
//   # 14, 109s at 16; delivery about 5s through 14, 5s at 16):
//   quint verify --main=exclusivity_ok --invariant=invAll --max-steps=16 specs/crossing_lifecycle.qnt
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
  // Twin `clock_fenced_reclaim`: the producer reclaims on its own clock
  // passing the deadline, with no evidence.
  const RECLAIM_ON_CLOCK: bool
  // Twin `no_reclaim`: the producer never reclaims.
  const RECLAIM_ENABLED: bool
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
  // The core's verdict on the crossing: claimed or refused (once).
  var claimed: bool
  var refused: bool
  // The refusal certificate reached the producer.
  var refusalDelivered: bool
  // The producer holds an absence proof: a commit-proven core header at
  // or past PROBE_ANCHOR whose committed set lacks the transaction.
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
    absenceProved' = false,
    reclaimed' = false,
  }

  action unchangedButClock = all {
    issued' = issued, proposedAt' = proposedAt, committedAt' = committedAt,
    claimed' = claimed, refused' = refused, refusalDelivered' = refusalDelivered,
    absenceProved' = absenceProved, reclaimed' = reclaimed,
  }

  action tick = {
    nondet s = SHARDS.oneOf()
    nondet d = 1.to(MAX_TICK).oneOf()
    all {
      clock.get(s) + d <= MAX_CLOCK,
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
    absenceProved' = absenceProved, reclaimed' = reclaimed,
  }

  // The core's chain admits the block carrying the transaction, at an
  // anchor inside the validity window.
  val canPropose = proposedAt == NONE and clock.get(CORE) < DEADLINE
  action propose = all {
    canPropose,
    proposedAt' = clock.get(CORE),
    clock' = clock, issued' = issued, committedAt' = committedAt,
    claimed' = claimed, refused' = refused, refusalDelivered' = refusalDelivered,
    absenceProved' = absenceProved, reclaimed' = reclaimed,
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
    absenceProved' = absenceProved, reclaimed' = reclaimed,
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
    refusalDelivered' = refusalDelivered, absenceProved' = absenceProved,
    reclaimed' = reclaimed,
  }
  val canRefuse = committedAt != NONE and not(claimed) and not(refused)
  action refuse = all {
    canRefuse,
    refused' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed,
    refusalDelivered' = refusalDelivered, absenceProved' = absenceProved,
    reclaimed' = reclaimed,
  }

  // The refusal certificate reaches the producer, whenever the network
  // delivers it.
  val canDeliverRefusal = refused and not(refusalDelivered)
  action deliverRefusal = all {
    canDeliverRefusal,
    refusalDelivered' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed, refused' = refused,
    absenceProved' = absenceProved, reclaimed' = reclaimed,
  }

  // The producer, past D on its own clock, reads the core's committed set
  // off a commit-proven header at or past the probe anchor. Absence at
  // that header is the proof; presence answers the probe with nothing.
  val canProbe = issued and not(absenceProved)
    and clock.get(PRODUCER) >= DEADLINE + FINALIZATION_DELAY
    and clock.get(CORE) >= PROBE_ANCHOR
    and committedAt == NONE
  action probe = all {
    canProbe,
    absenceProved' = true,
    clock' = clock, issued' = issued, proposedAt' = proposedAt,
    committedAt' = committedAt, claimed' = claimed, refused' = refused,
    refusalDelivered' = refusalDelivered, reclaimed' = reclaimed,
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
    refusalDelivered' = refusalDelivered, absenceProved' = absenceProved,
  }

  // Nothing but the clocks can move, and the clocks have stopped.
  val quiescent = SHARDS.forall(s => clock.get(s) == MAX_CLOCK)
    and not(canIssue) and not(canPropose) and not(canCommit)
    and not(canClaim) and not(canRefuse) and not(canDeliverRefusal)
    and not(canProbe) and not(canReclaim)

  // A terminal state stutters: the model ends there, not in a deadlock.
  action idle = all {
    quiescent,
    clock' = clock,
    unchangedButClock,
  }

  action step = any {
    tick, issue, propose, commit, claim, refuse, deliverRefusal, probe, reclaim, idle
  }

  // INV-EXEC-11: a crossing is disposed at most once.
  val invExclusive = not(claimed and reclaimed)

  // INV-EXEC-12: at a terminal state an issued crossing is disposed.
  val invDisposable = quiescent implies (not(issued) or claimed or reclaimed)

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
  // The issuer holds a proof of the claim cell's absence past the lapse.
  var lapseProved: bool
  // The issuer took the crossing back.
  var reclaimed: bool

  action init = all {
    clock' = 0, proposedAt' = NONE, issuedAt' = NONE, deliveredAt' = NONE,
    lost' = false, claimedAt' = NONE, claimCommitted' = false,
    lapseProved' = false, reclaimed' = false,
  }

  action unchangedButClock = all {
    proposedAt' = proposedAt, issuedAt' = issuedAt, deliveredAt' = deliveredAt,
    lost' = lost, claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    lapseProved' = lapseProved, reclaimed' = reclaimed,
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
    lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  val canIssue = proposedAt != NONE and issuedAt == NONE
    and clock <= proposedAt + FINALIZATION_DELAY
  action issue = all {
    canIssue,
    issuedAt' = clock,
    clock' = clock, proposedAt' = proposedAt, deliveredAt' = deliveredAt, lost' = lost,
    claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  val canDeliver = issuedAt != NONE and deliveredAt == NONE and not(lost)
  action deliver = all {
    canDeliver,
    deliveredAt' = clock,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt, lost' = lost,
    claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  // The bundle is dropped for good: nothing delivers it after this.
  val canLose = issuedAt != NONE and deliveredAt == NONE and not(lost)
  action lose = all {
    canLose,
    lost' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, claimedAt' = claimedAt, claimCommitted' = claimCommitted,
    lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  // The recipient admits a claim of a delivered crossing while the window
  // is open.
  val canClaim = deliveredAt != NONE and claimedAt == NONE and clock < DELIVERY_CLOSE
  action claim = all {
    canClaim,
    claimedAt' = clock,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimCommitted' = claimCommitted,
    lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  // An admitted claim commits within the finalization delay.
  val canCommitClaim = claimedAt != NONE and not(claimCommitted)
    and clock <= claimedAt + FINALIZATION_DELAY
  action commitClaim = all {
    canCommitClaim,
    claimCommitted' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimedAt' = claimedAt,
    lapseProved' = lapseProved, reclaimed' = reclaimed,
  }

  // The issuer reads the recipient's committed state at a block at or past
  // the lapse anchor: the claim cell is absent exactly when no claim has
  // committed. A claim admitted but not yet committed reads as absent —
  // which is what the anchor's finalization delay exists to put behind it.
  val canProbe = issuedAt != NONE and not(lapseProved) and not(claimCommitted)
    and clock >= LAPSE_ANCHOR
  action probeLapse = all {
    canProbe,
    lapseProved' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimedAt' = claimedAt,
    claimCommitted' = claimCommitted, reclaimed' = reclaimed,
  }

  val canReclaim = RECLAIM_ENABLED and issuedAt != NONE and lapseProved and not(reclaimed)
  action reclaim = all {
    canReclaim,
    reclaimed' = true,
    clock' = clock, proposedAt' = proposedAt, issuedAt' = issuedAt,
    deliveredAt' = deliveredAt, lost' = lost, claimedAt' = claimedAt,
    claimCommitted' = claimCommitted, lapseProved' = lapseProved,
  }

  val quiescent = clock == MAX_CLOCK
    and not(canPropose) and not(canIssue) and not(canDeliver) and not(canLose)
    and not(canClaim) and not(canCommitClaim) and not(canProbe) and not(canReclaim)

  // A terminal state stutters: the model ends there, not in a deadlock.
  action idle = all {
    quiescent,
    clock' = clock,
    unchangedButClock,
  }

  action step = any { tick, propose, issue, deliver, lose, claim, commitClaim, probeLapse, reclaim, idle }

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
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3,
    RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = true, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Twin: the producer reclaims on its own clock passing the deadline. The
// core, on a clock behind the producer's, commits and claims after it.
module clock_fenced_reclaim {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3,
    RECLAIM_ON_CLOCK = true, RECLAIM_ENABLED = true, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Twin: the absence proof may be anchored at the deadline. A block
// admitted just under it commits after the anchor, and the core claims
// what the producer has reclaimed.
module early_absence_probe {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 2,
    RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = true, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Twin: the producer never reclaims. A refused crossing is left issued
// with nobody able to claim it.
module no_reclaim {
  import exclusivity_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, PROBE_ANCHOR = 3,
    RECLAIM_ON_CLOCK = false, RECLAIM_ENABLED = false, MAX_TICK = 2, MAX_CLOCK = 5
  ).*
}

// Healthy delivery: the outbound claim is admissible to the delivery
// window's close, past the deadline by more than the finalization delay
// and the delivery lag together; a lost bundle lapses, proved absent
// past the close plus the finalization delay, and the issuer reclaims.
module delivery_ok {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 5, LAPSE_ANCHOR = 6,
    RECLAIM_ENABLED = true, DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}

// Twin: the outbound claim expires at the deadline. A crossing issued
// just under it, or delivered just after, reaches a recipient that can
// no longer claim it.
module deadline_on_delivery {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 2, LAPSE_ANCHOR = 3,
    RECLAIM_ENABLED = true, DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}

// Twin: the lapse is proved at the window's close itself. A claim
// admitted just under the close commits after the anchor, and the issuer
// reclaims what the recipient has claimed.
module early_lapse_probe {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 5, LAPSE_ANCHOR = 5,
    RECLAIM_ENABLED = true, DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}

// Twin: the issuer never reclaims. A lost bundle leaves the crossing
// issued with nobody able to claim it.
module no_lapse_reclaim {
  import delivery_core(
    DEADLINE = 2, FINALIZATION_DELAY = 1, DELIVERY_CLOSE = 5, LAPSE_ANCHOR = 6,
    RECLAIM_ENABLED = false, DELIVERY_LAG = 1, CLAIM_LAG = 1, MAX_TICK = 2, MAX_CLOCK = 8
  ).*
}
