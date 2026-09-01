// Model K (register: specs/README.md) — the deterministic substate
// sweep: the consensus operation that retires committed cells. Two
// modules. `substate_sweep` — one chain's frontier walk: what a block
// removes, where its cursor lands, and the three things that make the
// removal set a function of committed state rather than of a proposer's
// choice or a replica's own clock (INV-SWEEP-1, 3, 4, 6, 7, 9).
// `sweep_inheritance` — what a successor's cursor starts at, which is a
// different claim from what a block may create below it (INV-SWEEP-8,
// and INV-SWEEP-2 by abstraction).
//
// One shard's chain per module. The environment plays the committed
// clock forward, admits transactions that write sweepable cells, and
// proposes blocks that sweep. Everything else — consensus, execution,
// the index that finds the candidates — is an oracle: Model B commits
// the blocks, and the index is derived state whose agreement with the
// leaves is a storage obligation rather than a protocol one.
//
// Code anchors:
//   - The cursor: crates/types/src/shard/sweep.rs `SweepFrontier` — the
//     leaf key rotated to put the expiry bucket first
//     (`bucket ++ owner ++ local[4..]`), so lexicographic order on it is
//     the order a walk visits cells in. Modeled as an integer
//     `bucket * SEQ_WIDTH + seq`: what matters is that a cell's position
//     orders by *bucket* first and by key within it, and that exact
//     expiry orders nothing.
//   - The ceiling: `SweepFrontier::ceiling_at` — the start of the bucket
//     the clock falls in, so every position below it is in a bucket
//     wholly in the past and no expiry has to be read to decide
//     ordering.
//   - The walk: crates/storage/src/shard/sweep.rs `sweep_for_block` —
//     candidates strictly above the parent's frontier and strictly below
//     the ceiling, at most `MAX_SWEEP_PER_BLOCK`; the frontier takes the
//     ceiling when fewer than the cap remain and the last removed cell's
//     position when the cap is what stopped it.
//   - The check: crates/shard/src/action_handlers.rs, the
//     `VerifyStateRoot` arm — the header's `sweep_frontier` must equal
//     the one the walk produces, refused as
//     `StateRootVerifyError::SweepFrontierMismatch` otherwise. One
//     equality, in the shape `state_root` already has.
//   - The clock: the block's `parent_qc.weighted_timestamp()`, which
//     `QuorumCertificate::from_verified_votes` clamps monotone against
//     the parent's — never a proposer's own `ProposerTimestamp` and
//     never a validator's wall clock.
//   - Co-location: crates/types/src/transaction/vm.rs `Routing` — a
//     nullifier's creation is an exclusive *write* key, so
//     `TopologySnapshot::involves_shard_for_consensus` puts the shard
//     holding the cell in the consensus path of every transaction that
//     could spend it, and that shard enforces the effective window
//     against the same clock its frontier is capped at.
//   - The start: crates/types/src/shard/header.rs — `genesis`,
//     `split_child_genesis` and `merge_parent_genesis` all set
//     `sweep_frontier: SweepFrontier::ZERO`.
//
// Properties:
//   invNothingBelow   — no live cell ever sits at or below the chain's
//                       frontier. Two distinct ways to break it, which
//                       is why one property carries both: a block
//                       creating a cell below the cursor it may advance
//                       to (INV-SWEEP-7), and a successor starting its
//                       cursor above a predecessor's (INV-SWEEP-8). The
//                       `sweep_inheritance_adopt_max` twin is the
//                       second — a merged chain taking the higher of its
//                       predecessors' cursors strands every cell the
//                       other one held between them: unswept,
//                       unreachable, and invisible to INV-SWEEP-7,
//                       because nothing was *created* there.
//   invNoLiveRemoval  — no cell is removed while a transaction that
//                       reads it could still be admitted (INV-SWEEP-3).
//                       The twin is `substate_sweep_split_clock`: it
//                       keeps the ceiling honest but lets admission
//                       answer to a *different* chain's clock, which is
//                       the escrow-pair shape and the halt-recovery
//                       hazard this plan was gated on.
//
//                       Beside it sits a **control that holds, and whose
//                       holding is the result**:
//                       `substate_sweep_grace_early` inflates the
//                       ceiling by a whole grace — the plausible "the
//                       grace already covers it, sweep a grace early"
//                       error — and still cannot take a live cell. It
//                       sweeps (the frontier advances), so it is not
//                       vacuous; it simply cannot reach one, because the
//                       clock it inflates is the same clock admission is
//                       judged against. That isolates co-location as the
//                       load-bearing property rather than any margin:
//                       INV-SWEEP-9 mechanised. Break co-location and
//                       the very same ceiling arithmetic takes live
//                       cells, which is what the twin shows.
//   invSameSet        — two replicas of one chain compute the same
//                       removal set and the same frontier (INV-SWEEP-1).
//                       The `substate_sweep_local_clock` twin reads each
//                       replica's own clock instead of the committed
//                       one, and they diverge on the first drift — a
//                       state-root fork with no Byzantine action.
//   invMandatory      — a committed block's frontier is the one the walk
//                       produces (INV-SWEEP-6), and the backlog of cells
//                       already past their bucket stays bounded, which
//                       is what the obligation buys. The
//                       `substate_sweep_optional` twin lets a proposer
//                       stop short — the economically rational choice,
//                       since a removal earns no fee and costs block
//                       space — and the backlog grows without bound.
//   invCapped         — a block removes at most the cap, and stops
//                       rather than discarding: the frontier it lands on
//                       is exactly where the next block resumes, so a
//                       same-instant pile drains across blocks with
//                       nothing skipped and nothing repeated
//                       (INV-SWEEP-4).
//
// Modeling assumptions:
//   - `GRACE > BUCKET` is the model's one arithmetic premise and it is
//     the code's (`NULLIFIER_GRACE_MS` 144_000 against a bucket of
//     65_536). It is what makes INV-SWEEP-7 structural rather than
//     checked: a cell born at clock `t` expires at least a grace later,
//     so its bucket is strictly above the bucket a block anchored at `t`
//     may advance its frontier to.
//   - A cell's position is an integer rather than 48 bytes. Only the
//     order is load-bearing, and the model keeps the property that
//     decides the design: within a bucket, position is the key's hash
//     body and bears no relation to exact expiry. A model that ordered
//     by expiry would verify a cursor the store cannot walk.
//   - Transaction admission is collapsed to "the clock is inside the
//     window". Whether a transaction's window is well formed, whether it
//     narrows under a bound subintent, and whether its signatures verify
//     are `subintent-format`'s subject; the model takes the effective
//     window as given.
//   - The index that answers which owners hold cells in which bucket is
//     not modeled. Its agreement with the leaves is a storage
//     obligation, discharged by rebuilding it from them at every path
//     that writes cells; a divergent index is a defect in that layer
//     rather than a reachable protocol behavior.
//   - Two replicas suffice for `invSameSet`: the removal set is a pure
//     function of `(frontier, clock, cells)`, so if two readings of the
//     one varying input agree everywhere then all readings do.
//   - `invSameSet` is *vacuous* in the healthy instance, and that is the
//     property rather than a weakness: the shipped rule never consults a
//     replica's own reading, so the two cannot differ. What gives it
//     content is the twin, where the reading decides and the replicas
//     fork on the first drift. Stated here rather than dressed up,
//     because a reader who checks will find it holds trivially.
//   - The cap is a count, not a work term, because a removal is not
//     optional and so is never priced. `MAX_SWEEPABLE_CREATED_PER_BLOCK`
//     — the other half of the bound — is a block-content limit checked
//     off the derivations and needs no temporal model.
//
// Run:
//   quint typecheck specs/substate_sweep.qnt
//   quint verify --main=substate_sweep_ok --invariant=invAll --max-steps=8 specs/substate_sweep.qnt
//     (holds; depth 8 covers admit → tick past the grace → sweep under
//      the cap → sweep at the cap → resume on the next block)
//   quint verify --main=substate_sweep_grace_early --invariant=invNoLiveRemoval --max-steps=8 specs/substate_sweep.qnt
//     (control: holds, and the holding is the result — a ceiling
//      inflated by a whole grace still cannot take a live cell while one
//      clock orders removal and admission)
//   quint verify --main=substate_sweep_split_clock --invariant=invNoLiveRemoval --max-steps=10 specs/substate_sweep.qnt
//     (the twin lets admission answer to another chain's clock; expect a
//      counterexample — the holding chain retires a cell whose window is
//      still open where the claims are admitted)
//   quint verify --main=substate_sweep_local_clock --invariant=invSameSet --max-steps=6 specs/substate_sweep.qnt
//     (the twin sweeps on each replica's own reading; expect a
//      counterexample)
//   quint verify --main=substate_sweep_optional --invariant=invMandatory --max-steps=6 specs/substate_sweep.qnt
//     (the twin lets a proposer stop short; expect a counterexample)
//   quint verify --main=sweep_inheritance_ok --invariant=invNothingBelow --max-steps=6 specs/substate_sweep.qnt
//     (holds)
//   quint verify --main=sweep_inheritance_adopt_max --invariant=invNothingBelow --max-steps=6 specs/substate_sweep.qnt
//     (the twin adopts the higher predecessor cursor; expect a
//      counterexample)

module substate_sweep {
  // Bucket width, in the model's own clock units.
  const BUCKET: int
  // How long a cell outlives the window it derives from
  // (`NULLIFIER_GRACE_MS`). Must exceed BUCKET — see the assumptions.
  const GRACE: int
  // The longest window a transaction may be admitted inside.
  const WINDOW: int
  // Positions per bucket: the room a bucket's keys sort within.
  const SEQ_WIDTH: int
  // `MAX_SWEEP_PER_BLOCK`.
  const CAP: int
  // How far the committed clock may run.
  const MAX_CLOCK: int
  // How far a replica's own reading may drift from the committed clock,
  // and how far the admitting chain's clock may lag under SPLIT_CLOCK.
  const MAX_DRIFT: int

  // Control (not a twin): the ceiling is inflated by a whole grace, the
  // plausible "the grace already covers it, sweep a grace early" error.
  // The instance *holds*, which is the point — see `invNoLiveRemoval`.
  // False is the shipped `ceiling_at`.
  const GRACE_EARLY: bool
  // Broken twin: each replica sweeps on its own clock reading rather
  // than the committed one. False is the shipped rule.
  const LOCAL_CLOCK: bool
  // Broken twin: a proposer may stop short of the frontier the walk
  // produces. False is the shipped obligation.
  const OPTIONAL: bool
  // Broken twin: the chain holding the cell is not the chain admitting
  // the transactions that read it, so the two answer to clocks that
  // drift. False is the shipped co-location (INV-SWEEP-9).
  const SPLIT_CLOCK: bool

  // A committed sweepable cell: where it sorts, and when it stops being
  // owed. `pos` is `bucket(expiry) * SEQ_WIDTH + seq` — bucket first,
  // then key order, with exact expiry ordering nothing inside a bucket.
  type Cell = { pos: int, expiry: int }

  // The chain's committed clock — the parent QC's weighted timestamp,
  // monotone by the per-vote clamp.
  var clock: int
  // Two replicas' own readings of it. Only LOCAL_CLOCK lets them decide
  // anything; under the shipped rule they are observers.
  var readA: int
  var readB: int
  // The clock admission answers to. The same one under the shipped
  // co-location; a drifting other chain's under SPLIT_CLOCK.
  var admitClock: int
  // Live sweepable cells.
  var cells: Set[Cell]
  // The chain's cursor.
  var frontier: int
  // Next key ordinal, so cells take distinct positions. Starts at 1: a
  // bucket's start position zeroes the owner bytes, tag included, and a
  // zero tag names no address class — so the ceiling sits strictly below
  // every cell in its own bucket and no cell can occupy it.
  var nextSeq: int
  // Latched: some removal took a cell a transaction could still have
  // been admitted to read.
  var tookLive: bool
  // Latched: some committed block's frontier was not the one the walk
  // produces.
  var stoppedShort: bool

  pure def bucketOf(t: int): int = t / BUCKET
  // The start of the bucket `t` falls in — every position below it is in
  // a bucket wholly in the past.
  pure def ceilingAt(t: int): int = bucketOf(t) * SEQ_WIDTH
  pure def posOf(expiry: int, seq: int): int = bucketOf(expiry) * SEQ_WIDTH + seq

  // The cells a block anchored at `t` may reach: strictly above the
  // cursor, strictly below the ceiling.
  pure def candidatesAt(cs: Set[Cell], f: int, t: int): Set[Cell] =
    cs.filter(c => c.pos > f and c.pos < ceilingAt(t))

  // Where the walk stops: the ceiling when fewer than the cap remain,
  // and the cap-th position when the cap is what stopped it. The second
  // arm is what makes a capped sweep resumable — the next block picks up
  // at exactly the next cell.
  pure def frontierAfter(cs: Set[Cell], f: int, t: int): int =
    val cands = candidatesAt(cs, f, t)
    if (cands.size() <= CAP) ceilingAt(t)
    else cands
      .filter(c => cands.filter(d => d.pos <= c.pos).size() == CAP)
      .fold(f, (acc, c) => c.pos)

  pure def removalsAt(cs: Set[Cell], f: int, t: int): Set[Cell] =
    candidatesAt(cs, f, t).filter(c => c.pos <= frontierAfter(cs, f, t))

  // The clock a replica sweeps on. The shipped rule ignores the reading
  // entirely; the twin lets it decide.
  pure def sweepClockOf(committed: int, reading: int): int =
    if (LOCAL_CLOCK) reading else committed

  // The ceiling a block uses. The control reaches a whole grace
  // further — and still cannot take a live cell, because the clock it
  // inflates is the clock admission is judged against.
  pure def sweepCeilingClock(t: int): int = if (GRACE_EARLY) t + GRACE else t

  action init = all {
    clock' = 0,
    readA' = 0,
    readB' = 0,
    admitClock' = 0,
    cells' = Set(),
    frontier' = 0,
    nextSeq' = 1,
    tookLive' = false,
    stoppedShort' = false,
  }

  // The committed clock advances. Monotone, and the replicas' own
  // readings follow it within a drift; under the shipped rule that drift
  // decides nothing.
  action tick = {
    nondet d = 1.to(BUCKET).oneOf()
    nondet driftA = 0.to(MAX_DRIFT).oneOf()
    nondet driftB = 0.to(MAX_DRIFT).oneOf()
    // The admitting chain's own clock. Co-located it is this chain's;
    // split, it is another's and may lag arbitrarily inside the drift.
    nondet lag = 0.to(MAX_DRIFT).oneOf()
    all {
      clock + d <= MAX_CLOCK,
      clock' = clock + d,
      // Pinned to the committed clock unless the twin lets a reading
      // decide something. Under the shipped rule a replica's own clock
      // is an observer, so letting it range would multiply the state
      // space by nothing.
      readA' = if (LOCAL_CLOCK) clock + d - driftA else clock + d,
      readB' = if (LOCAL_CLOCK) clock + d - driftB else clock + d,
      // Clamped at zero: a chain's committed clock is never negative,
      // and a counterexample resting on one would be an artifact rather
      // than the hazard.
      admitClock' =
        if (SPLIT_CLOCK) (if (clock + d - lag > 0) clock + d - lag else 0)
        else clock + d,
      cells' = cells, frontier' = frontier, nextSeq' = nextSeq,
      tookLive' = tookLive, stoppedShort' = stoppedShort,
    }
  }

  // A transaction is admitted and writes the cell that records it. Its
  // window ends inside WINDOW of the clock that admitted it, and the
  // cell is owed a grace past that.
  action admit = {
    nondet w = 1.to(WINDOW).oneOf()
    val windowEnd = admitClock + w
    val expiry = windowEnd + GRACE
    all {
      nextSeq < SEQ_WIDTH,
      cells' = cells.union(Set({ pos: posOf(expiry, nextSeq), expiry: expiry })),
      nextSeq' = nextSeq + 1,
      clock' = clock, readA' = readA, readB' = readB,
      admitClock' = admitClock, frontier' = frontier,
      tookLive' = tookLive, stoppedShort' = stoppedShort,
    }
  }

  // A block sweeps. The frontier it lands on is the walk's, and a
  // validator refuses any other — except under OPTIONAL, where a
  // proposer may keep the block space instead.
  action sweep = {
    nondet declined = Bool.oneOf()
    val t = sweepCeilingClock(sweepClockOf(clock, readA))
    val mandated = frontierAfter(cells, frontier, t)
    val landed = if (OPTIONAL and declined) frontier else mandated
    val removed = candidatesAt(cells, frontier, t).filter(c => c.pos <= landed)
    all {
      landed >= frontier,
      cells' = cells.exclude(removed),
      frontier' = landed,
      // A removal is live if a transaction reading the cell could still
      // be admitted: the chain admitting them has not passed the window
      // the cell's expiry derives from.
      tookLive' = (tookLive or removed.exists(c => admitClock < c.expiry - GRACE)),
      stoppedShort' = (stoppedShort or landed != mandated),
      clock' = clock, readA' = readA, readB' = readB,
      admitClock' = admitClock, nextSeq' = nextSeq,
    }
  }

  action step = any { tick, admit, sweep }

  // Nothing live sits at or below the cursor: neither created there
  // (INV-SWEEP-7, structural while GRACE > BUCKET) nor inherited there
  // (INV-SWEEP-8, the inheritance module's subject).
  val invNothingBelow = cells.forall(c => c.pos > frontier)

  // No cell is removed while a transaction reading it could still be
  // admitted (INV-SWEEP-3). Holds with no skew bound in it, which is
  // INV-SWEEP-9.
  val invNoLiveRemoval = not(tookLive)

  // Two replicas of one chain reach the same removal set and the same
  // cursor (INV-SWEEP-1).
  val invSameSet =
    val ta = sweepCeilingClock(sweepClockOf(clock, readA))
    val tb = sweepCeilingClock(sweepClockOf(clock, readB))
    and {
      frontierAfter(cells, frontier, ta) == frontierAfter(cells, frontier, tb),
      removalsAt(cells, frontier, ta) == removalsAt(cells, frontier, tb),
    }

  // A committed block's cursor is the one the walk produces
  // (INV-SWEEP-6) — the validity rule verbatim, since that rule is what
  // leaves the frontier a function of committed state with no proposer
  // choice in it. A backlog is *not* bounded state by state: cells
  // become sweepable between blocks and drain over the next few, so the
  // bound the obligation buys is a rate, and the rate is bounded by the
  // creation cap rather than by anything a trace can show.
  val invMandatory = not(stoppedShort)

  // A block removes at most the cap, and what it leaves is exactly where
  // the next block resumes — nothing skipped, nothing repeated
  // (INV-SWEEP-4).
  val invCapped =
    val t = sweepCeilingClock(sweepClockOf(clock, readA))
    and {
      removalsAt(cells, frontier, t).size() <= CAP,
      removalsAt(cells, frontier, t)
        .forall(c => c.pos > frontier and c.pos <= frontierAfter(cells, frontier, t)),
    }

  val invAll = and {
    invNothingBelow,
    invNoLiveRemoval,
    invSameSet,
    invMandatory,
    invCapped,
  }
}

// What a successor's cursor starts at. A frontier is safe carried
// *down* and not up: creation is one claim (INV-SWEEP-7) and inheritance
// another (INV-SWEEP-8), and the second is invisible to the first.
module sweep_inheritance {
  // Positions the predecessors' cells may take.
  const MAX_POS: int
  // Broken twin: the successor adopts the higher of its predecessors'
  // cursors. False is the shipped zero.
  const ADOPT_MAX: bool

  // Each predecessor's cursor and the cells it still holds.
  var frontierLeft: int
  var frontierRight: int
  var cellsLeft: Set[int]
  var cellsRight: Set[int]
  // The successor, once composed.
  var merged: bool
  var frontierMerged: int
  var cellsMerged: Set[int]

  action init = all {
    frontierLeft' = 0,
    frontierRight' = 0,
    cellsLeft' = Set(),
    cellsRight' = Set(),
    merged' = false,
    frontierMerged' = 0,
    cellsMerged' = Set(),
  }

  // A predecessor sweeps: its cursor advances and it drops what it
  // passed. Each chain's cursor is its own, over its own keyspace, and
  // nothing relates them.
  action advanceLeft = {
    nondet f = (frontierLeft + 1).to(MAX_POS).oneOf()
    all {
      not(merged),
      frontierLeft' = f,
      cellsLeft' = cellsLeft.filter(p => p > f),
      frontierRight' = frontierRight, cellsRight' = cellsRight,
      merged' = merged, frontierMerged' = frontierMerged,
      cellsMerged' = cellsMerged,
    }
  }

  action advanceRight = {
    nondet f = (frontierRight + 1).to(MAX_POS).oneOf()
    all {
      not(merged),
      frontierRight' = f,
      cellsRight' = cellsRight.filter(p => p > f),
      frontierLeft' = frontierLeft, cellsLeft' = cellsLeft,
      merged' = merged, frontierMerged' = frontierMerged,
      cellsMerged' = cellsMerged,
    }
  }

  // A predecessor takes on a cell above its own cursor — the only place
  // one can be born (INV-SWEEP-7).
  action writeLeft = {
    nondet p = (frontierLeft + 1).to(MAX_POS).oneOf()
    all {
      not(merged),
      cellsLeft' = cellsLeft.union(Set(p)),
      frontierLeft' = frontierLeft, frontierRight' = frontierRight,
      cellsRight' = cellsRight, merged' = merged,
      frontierMerged' = frontierMerged, cellsMerged' = cellsMerged,
    }
  }

  action writeRight = {
    nondet p = (frontierRight + 1).to(MAX_POS).oneOf()
    all {
      not(merged),
      cellsRight' = cellsRight.union(Set(p)),
      frontierRight' = frontierRight, frontierLeft' = frontierLeft,
      cellsLeft' = cellsLeft, merged' = merged,
      frontierMerged' = frontierMerged, cellsMerged' = cellsMerged,
    }
  }

  // The cut: two chains over disjoint keyspaces become one, holding
  // both their cells. What its cursor starts at is the whole question.
  action compose = all {
    not(merged),
    merged' = true,
    cellsMerged' = cellsLeft.union(cellsRight),
    frontierMerged' =
      if (ADOPT_MAX)
        (if (frontierLeft > frontierRight) frontierLeft else frontierRight)
      else 0,
    frontierLeft' = frontierLeft, frontierRight' = frontierRight,
    cellsLeft' = cellsLeft, cellsRight' = cellsRight,
  }

  action step = any { advanceLeft, advanceRight, writeLeft, writeRight, compose }

  // Nothing the successor inherited sits at or below its cursor. Under
  // the shipped zero this is free; under ADOPT_MAX every cell the lower
  // predecessor held between the two cursors is stranded — unswept and
  // unreachable, and created nowhere near the cursor that now hides it.
  val invNothingBelow =
    merged implies cellsMerged.forall(p => p > frontierMerged)
}

module substate_sweep_ok {
  import substate_sweep(
    BUCKET = 2, GRACE = 3, WINDOW = 2, SEQ_WIDTH = 4, CAP = 2,
    MAX_CLOCK = 14, MAX_DRIFT = 2,
    GRACE_EARLY = false, LOCAL_CLOCK = false, OPTIONAL = false,
    SPLIT_CLOCK = false
  ).*
}

// Control, not a twin: it holds, and that is the result.
module substate_sweep_grace_early {
  import substate_sweep(
    BUCKET = 2, GRACE = 3, WINDOW = 2, SEQ_WIDTH = 4, CAP = 2,
    MAX_CLOCK = 14, MAX_DRIFT = 2,
    GRACE_EARLY = true, LOCAL_CLOCK = false, OPTIONAL = false,
    SPLIT_CLOCK = false
  ).*
}

module substate_sweep_split_clock {
  import substate_sweep(
    BUCKET = 2, GRACE = 3, WINDOW = 2, SEQ_WIDTH = 4, CAP = 2,
    MAX_CLOCK = 14, MAX_DRIFT = 6,
    GRACE_EARLY = false, LOCAL_CLOCK = false, OPTIONAL = false,
    SPLIT_CLOCK = true
  ).*
}

module substate_sweep_local_clock {
  import substate_sweep(
    BUCKET = 2, GRACE = 3, WINDOW = 2, SEQ_WIDTH = 4, CAP = 2,
    MAX_CLOCK = 14, MAX_DRIFT = 2,
    GRACE_EARLY = false, LOCAL_CLOCK = true, OPTIONAL = false,
    SPLIT_CLOCK = false
  ).*
}

module substate_sweep_optional {
  import substate_sweep(
    BUCKET = 2, GRACE = 3, WINDOW = 2, SEQ_WIDTH = 4, CAP = 2,
    MAX_CLOCK = 14, MAX_DRIFT = 2,
    GRACE_EARLY = false, LOCAL_CLOCK = false, OPTIONAL = true,
    SPLIT_CLOCK = false
  ).*
}

module sweep_inheritance_ok {
  import sweep_inheritance(MAX_POS = 4, ADOPT_MAX = false).*
}

module sweep_inheritance_adopt_max {
  import sweep_inheritance(MAX_POS = 4, ADOPT_MAX = true).*
}
