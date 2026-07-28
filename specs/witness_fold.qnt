// Model H (register: specs/README.md) — the two folds a shard's boundary
// crossing drives, split because they no longer share a coordinate. The
// `witness_fold` module checks the governance-leaf accumulator's window
// arithmetic (INV-BEACON-6). The `reveal_fold` module checks the epoch
// randomness the crossing header's reveal chain carries (INV-BEACON-10,
// INV-BEACON-11). The `witness_window` module checks the window-base
// freeze that keeps a window's two schedule writes byte-identical
// (INV-BEACON-3's bookend discipline applied to `witness_window_bases`).
//
// One subject shard, folded by the committed beacon epoch sequence. The
// environment plays that sequence: the chain grows, a proposer reports a
// fresh crossing or re-rides the recorded one, a halt recovery stamps its
// frontier, the shard terminates into a successor.
//
// Code anchors:
//   - Chunk bounds: crates/beacon/src/rules.rs `chunk_bounds` — the fold
//     consumes `[watermark, min(watermark + CAP, boundary_count))`; the
//     range derives from the local watermark, never from the reporter.
//   - Leaf application: crates/beacon/src/state/witness.rs
//     `apply_contribution_witnesses` — exact-range, all-or-nothing.
//   - Reveal collection: crates/beacon/src/state/epoch.rs
//     `record_boundaries` — a fresh crossing contributes
//     `header.reveal_chain()`, the closed chain of the anchor epoch that
//     crossing ends; a drain re-fold of the recorded crossing contributes
//     nothing, so exactly-once needs no watermark.
//   - Chain derivation: crates/types/src/shard/roots/reveal_chain.rs
//     `next_reveal_chain` — reseeds when the block's anchor epoch differs
//     from its parent's, so an epoch's chain descends from no earlier
//     epoch's reveals.
//   - Fence: crates/beacon/src/state/epoch.rs `reveal_chain_fenced` — a
//     crossing above a pending recovery's `attested_frontier` contributes
//     nothing. No band persists: per-epoch chains are independent, so
//     dropping one epoch's value excises it permanently.
//   - Reseat: crates/beacon/src/state/epoch.rs `seed_split_children` /
//     `compose_merge_parent` — successors start fresh accumulators; the
//     predecessor's terminal record keeps draining until consumed.
//   - Base freeze: crates/beacon/src/state/epoch.rs `apply_epoch` —
//     `witness_window_bases = live_witness_bases()` at the epoch's open,
//     before the fold advances any watermark; the lookahead projection
//     (crates/types/src/beacon/state.rs `derive_next_topology_snapshot`)
//     reads the same live value at the prior epoch's close.
//
// Properties:
//   invNoDoubleFold  — no leaf is applied twice, across fresh crossings,
//                      backlog drains, and the terminal drain of a
//                      reshaped-away predecessor. The
//                      `witness_fold_reporter_range` twin lets the
//                      reporter choose the chunk's start instead of the
//                      local watermark and violates it.
//   invCoverage      — every leaf below a generation's watermark is
//                      applied exactly once, and nothing at or above it
//                      is: the accumulator is a pure function of the
//                      committed sequence with no skipped or phantom
//                      leaves.
//   invRevealFence   — no reveal a beyond-frontier crossing produced ever
//                      enters the seed, in its own epoch or carried
//                      forward by a later one. The `reveal_fold_no_fence`
//                      twin folds the fenced crossing directly; the
//                      `reveal_fold_cumulative_chain` twin never reseeds
//                      the chain, so a later epoch's fold drags the
//                      fenced epoch's reveals in by descent — the design
//                      alternative this split exists to reject.
//   invRevealOnce    — each epoch's chain folds at most once. The
//                      `reveal_fold_drain_refolds` twin re-folds the
//                      recorded crossing and violates it.
//
// Assumptions:
//   - Content validation is out of scope. A contribution matches its
//     header exactly or is rejected whole (`contribution_chunk_valid`),
//     so the model applies ranges, never forged leaves. The adversary's
//     levers are scheduling ones: which certified crossing is reported
//     when, how much backlog accumulates, when the chain grows (minted
//     post-halt growth included — the fence, not content validation, is
//     what keeps it out of the seed).
//   - Heights collapse onto leaf counts in `witness_fold` and onto epochs
//     in `reveal_fold`. Each fence compares positions along one chain, so
//     the collapse preserves every checked ordering.
//   - One reshape generation in `witness_fold` (the reseat guard),
//     draining concurrently with the successor. `reveal_fold` carries no
//     reseat: a per-epoch chain already restarts at every epoch, so a
//     reshape adds no case the epoch boundary does not.
//   - Terminal drains bypass the fence: halt detection exempts terminal
//     records, so no recovery ever pends on the draining predecessor.
//   - Seed determinism itself is not a separate invariant: the fold is a
//     pure function of the committed sequence by construction
//     (INV-BEACON-2), mirroring Model G's treatment.
//
// Run:
//   quint typecheck specs/witness_fold.qnt
//   quint verify --main=witness_fold_ok --invariant=invAll --max-steps=10 specs/witness_fold.qnt
//   quint verify --main=reveal_fold_ok --invariant=invRevealAll --max-steps=10 specs/witness_fold.qnt
//   quint verify --main=witness_window_ok --invariant=invWindowAgree --max-steps=8 specs/witness_fold.qnt
//
//   Twins (each must produce a counterexample):
//   quint verify --main=witness_fold_reporter_range --invariant=invNoDoubleFold --max-steps=10 specs/witness_fold.qnt
//   quint verify --main=reveal_fold_no_fence --invariant=invRevealFence --max-steps=10 specs/witness_fold.qnt
//   quint verify --main=reveal_fold_cumulative_chain --invariant=invRevealFence --max-steps=8 specs/witness_fold.qnt
//   quint verify --main=reveal_fold_drain_refolds --invariant=invRevealOnce --max-steps=10 specs/witness_fold.qnt

module witness_fold {
  // Chunk cap per fold (`MAX_WITNESSES_PER_SHARD`; model scale).
  const CAP: int
  // Per-step chain growth bound (leaves; model scale).
  const MAX_GROW: int
  // Constant leaf-index band per generation (Apalache: state-relative
  // ranges are filtered from a constant band).
  const MAX_LEAVES: int
  // Broken twin: the chunk's start comes from the reporter rather than
  // the local watermark. False is the shipped `chunk_bounds`.
  const REPORTER_RANGE: bool

  // ── Current-generation accumulator ──
  // Reshape generation: 0 = the original shard, 1 = its successor.
  var gen: int
  // The chain's committed leaf count (ground truth).
  var chainCount: int
  // `ShardBoundary::witness_leaf_count` — the applied watermark.
  var watermark: int
  // The recorded crossing's leaf count (`boundary_count`).
  var recorded: int

  // ── The reshaped-away predecessor's terminal drain slot ──
  var oldActive: bool
  var oldWatermark: int
  var oldRecorded: int

  // ── Ghosts ──
  // Leaves the fold applied, as (generation, leaf index).
  var applied: Set[(int, int)]
  // A leaf was applied twice.
  var dupSeen: bool

  pure def minInt(a: int, b: int): int = if (a < b) a else b

  // The leaf band [lo, hi) of generation g, drawn from the constant band.
  pure def leafBand(g: int, lo: int, hi: int): Set[(int, int)] =
    0.to(MAX_LEAVES).filter(i => lo <= i and i < hi).map(i => (g, i))

  action init = all {
    gen' = 0,
    chainCount' = 0,
    watermark' = 0,
    recorded' = 0,
    oldActive' = false,
    oldWatermark' = 0,
    oldRecorded' = 0,
    applied' = Set(),
    dupSeen' = false,
  }

  // The chain commits more blocks, some carrying governance leaves.
  action grow = {
    nondet n = 1.to(MAX_GROW).oneOf()
    all {
      chainCount + n <= MAX_LEAVES,
      chainCount' = chainCount + n,
      gen' = gen, watermark' = watermark, recorded' = recorded,
      oldActive' = oldActive, oldWatermark' = oldWatermark,
      oldRecorded' = oldRecorded,
      applied' = applied, dupSeen' = dupSeen,
    }
  }

  // A fresh crossing at count c records the shard's new boundary and
  // applies the chunk `[watermark, min(watermark + CAP, c))`.
  action foldFresh = {
    nondet c = 0.to(MAX_LEAVES).oneOf()
    nondet s = 0.to(MAX_LEAVES).oneOf()
    val start = if (REPORTER_RANGE) s else watermark
    val end = minInt(watermark + CAP, c)
    val newLeaves = leafBand(gen, start, end)
    all {
      c > recorded,
      c <= chainCount,
      c >= watermark,
      s <= watermark,
      applied' = applied.union(newLeaves),
      dupSeen' = if (newLeaves.intersect(applied) != Set()) true else dupSeen,
      watermark' = end,
      recorded' = c,
      gen' = gen, chainCount' = chainCount,
      oldActive' = oldActive, oldWatermark' = oldWatermark,
      oldRecorded' = oldRecorded,
    }
  }

  // A drain re-fold of the recorded crossing: watermark-only progress, no
  // record refresh.
  action foldDrain = {
    val end = minInt(watermark + CAP, recorded)
    val newLeaves = leafBand(gen, watermark, end)
    all {
      watermark < recorded,
      applied' = applied.union(newLeaves),
      dupSeen' = if (newLeaves.intersect(applied) != Set()) true else dupSeen,
      watermark' = end,
      gen' = gen, chainCount' = chainCount, recorded' = recorded,
      oldActive' = oldActive, oldWatermark' = oldWatermark,
      oldRecorded' = oldRecorded,
    }
  }

  // The shard terminates at its recorded crossing and a successor seats
  // with a fresh accumulator; the terminal record keeps draining. Leaves
  // past the terminal cut are coast content no fold consumes. One
  // generation suffices — further reseats repeat the same shape.
  action reseat = all {
    gen == 0,
    recorded > 0,
    oldActive' = watermark < recorded,
    oldWatermark' = watermark,
    oldRecorded' = recorded,
    gen' = gen + 1,
    chainCount' = 0,
    watermark' = 0,
    recorded' = 0,
    applied' = applied, dupSeen' = dupSeen,
  }

  // The predecessor's terminal record drains: re-sourced every fold until
  // consumed, then dropped.
  action foldOldDrain = {
    val end = minInt(oldWatermark + CAP, oldRecorded)
    val newLeaves = leafBand(gen - 1, oldWatermark, end)
    all {
      oldActive,
      oldWatermark < oldRecorded,
      applied' = applied.union(newLeaves),
      dupSeen' = if (newLeaves.intersect(applied) != Set()) true else dupSeen,
      oldWatermark' = end,
      oldActive' = end < oldRecorded,
      gen' = gen, chainCount' = chainCount, watermark' = watermark,
      recorded' = recorded, oldRecorded' = oldRecorded,
    }
  }

  action step = any {
    grow,
    foldFresh,
    foldDrain,
    foldOldDrain,
    reseat,
  }

  // ── Invariants ──

  val invNoDoubleFold = not(dupSeen)

  // Exactly-once accounting: below a generation's watermark every leaf is
  // applied; at or above it, nothing is.
  val invCoverage = and {
    0.to(MAX_LEAVES).forall(i =>
      (i < watermark) implies applied.contains((gen, i))),
    0.to(MAX_LEAVES).forall(i =>
      (oldActive and i < oldWatermark) implies applied.contains((gen - 1, i))),
    applied.forall(l =>
      if (l._1 == gen) l._2 < watermark
      else l._1 == gen - 1 and l._2 < oldWatermark),
  }

  // The fold never runs ahead of certified content.
  val invWatermarkSound = and {
    watermark <= recorded or recorded == 0,
    recorded <= chainCount,
    watermark <= chainCount,
    oldWatermark <= oldRecorded,
  }

  val invAll = and {
    invNoDoubleFold,
    invCoverage,
    invWatermarkSound,
  }
}

module reveal_fold {
  // Epoch band the chain can reach (model scale).
  const MAX_EPOCH: int
  // Broken twin: the recovery fence is off entirely. False is the shipped
  // `reveal_chain_fenced`.
  const FENCE_OFF: bool
  // Broken twin: the chain never reseeds at an epoch boundary, so folding
  // one epoch's value carries every earlier epoch's reveals with it —
  // the cumulative-chain design this model exists to reject. False is the
  // shipped per-epoch reseed.
  const CHAIN_CUMULATIVE: bool
  // Broken twin: a drain re-fold of the recorded crossing folds its chain
  // again instead of contributing nothing. False is the shipped
  // `drain_refold` gate.
  const DRAIN_REFOLDS: bool

  // The shard's current anchor epoch. Its blocks fold into one chain,
  // which reseeds when the anchor epoch changes.
  var epoch: int
  // The last epoch whose crossing the beacon recorded. -1 before any.
  var recordedEpoch: int
  var recoveryPending: bool
  // `ShardRecovery::attested_frontier`, in the collapsed height≈epoch
  // coordinates.
  var frontier: int
  // Epochs whose chain reached `state.randomness`.
  var seededEpochs: Set[int]
  // Ground truth: epochs a beyond-frontier committee produced into, whose
  // reveals must never seed however the fold's bookkeeping evolves.
  var taintedEpochs: Set[int]
  // An epoch's chain folded twice.
  var dupFold: bool

  action init = all {
    epoch' = 0,
    recordedEpoch' = -1,
    recoveryPending' = false,
    frontier' = 0,
    seededEpochs' = Set(),
    taintedEpochs' = Set(),
    dupFold' = false,
  }

  // The chain crosses into its next anchor epoch.
  action advanceEpoch = all {
    epoch + 1 <= MAX_EPOCH,
    epoch' = epoch + 1,
    recordedEpoch' = recordedEpoch,
    recoveryPending' = recoveryPending, frontier' = frontier,
    seededEpochs' = seededEpochs, taintedEpochs' = taintedEpochs,
    dupFold' = dupFold,
  }

  // The beacon folds the crossing that closes epoch e — one chain value,
  // read off the crossing block's header. A crossing above a pending
  // recovery's frontier is fenced: its epoch contributes nothing.
  action foldCrossing = {
    nondet e = 0.to(MAX_EPOCH).oneOf()
    val beyond = recoveryPending and e > frontier
    val fenced = beyond and not(FENCE_OFF)
    // What the folded value carries. A per-epoch chain carries only its
    // own epoch; a cumulative one drags every earlier epoch in with it.
    val carried =
      if (CHAIN_CUMULATIVE) 0.to(MAX_EPOCH).filter(i => i <= e)
      else Set(e)
    all {
      e > recordedEpoch,
      e <= epoch,
      seededEpochs' =
        if (fenced) seededEpochs else seededEpochs.union(carried),
      dupFold' =
        if (not(fenced) and carried.intersect(seededEpochs) != Set()) true
        else dupFold,
      taintedEpochs' =
        if (beyond) taintedEpochs.union(Set(e)) else taintedEpochs,
      recordedEpoch' = e,
      recoveryPending' = false,
      epoch' = epoch, frontier' = frontier,
    }
  }

  // A committed re-fold of the crossing already recorded. The shipped fold
  // recognises it and contributes nothing; the twin folds its chain again.
  action drainRefold = all {
    recordedEpoch >= 0,
    seededEpochs' =
      if (DRAIN_REFOLDS) seededEpochs.union(Set(recordedEpoch))
      else seededEpochs,
    dupFold' =
      if (DRAIN_REFOLDS and seededEpochs.contains(recordedEpoch)) true
      else dupFold,
    epoch' = epoch, recordedEpoch' = recordedEpoch,
    recoveryPending' = recoveryPending, frontier' = frontier,
    taintedEpochs' = taintedEpochs,
  }

  // The detector flags the shard and the re-draw stamps a recovery whose
  // frontier is the recorded crossing's position. Detection calibration is
  // Model G's subject; here any moment is a valid stamp.
  action stampRecovery = all {
    not(recoveryPending),
    recoveryPending' = true,
    frontier' = if (recordedEpoch > 0) recordedEpoch else 0,
    epoch' = epoch, recordedEpoch' = recordedEpoch,
    seededEpochs' = seededEpochs, taintedEpochs' = taintedEpochs,
    dupFold' = dupFold,
  }

  action step = any {
    advanceEpoch,
    foldCrossing,
    drainRefold,
    stampRecovery,
  }

  // ── Invariants ──

  // No reveal a beyond-frontier crossing produced ever enters the seed —
  // not in the epoch it lands in, and not carried forward by any later
  // epoch's fold. The `CHAIN_CUMULATIVE` twin violates it by descent.
  val invRevealFence = seededEpochs.intersect(taintedEpochs) == Set()

  // Each epoch's chain folds at most once. The `DRAIN_REFOLDS` twin
  // violates it by re-folding the recorded crossing.
  val invRevealOnce = not(dupFold)

  // The fold never seeds an epoch the chain has not reached.
  val invRevealSound = seededEpochs.forall(e => e <= epoch)

  val invRevealAll = and {
    invRevealFence,
    invRevealOnce,
    invRevealSound,
  }
}

module witness_window {
  // Per-epoch chunk bound (model scale).
  const MAX_CHUNK: int
  // Broken twin: the active write reads the watermark after the epoch's
  // fold advanced it. False is the shipped freeze-at-open.
  const FREEZE_AFTER_FOLD: bool

  // The shard's applied watermark (`live_witness_bases()` proxy).
  var watermark: int
  // The window base the lookahead projection wrote at the prior epoch's
  // close (`derive_next_topology_snapshot`).
  var baseLookahead: int
  // The window base the promotion wrote at this epoch's open
  // (`apply_epoch`: `witness_window_bases = live_witness_bases()`).
  var baseActive: int
  // 0 = awaiting the epoch open, 1 = awaiting the fold, 2 = awaiting the
  // close.
  var phase: int

  action init = all {
    watermark' = 0,
    baseLookahead' = 0,
    baseActive' = 0,
    phase' = 0,
  }

  // Epoch open: promote — the shipped rule freezes this window's active
  // base before the fold; the twin defers it.
  action openWindow = all {
    phase == 0,
    baseActive' = if (FREEZE_AFTER_FOLD) baseActive else watermark,
    phase' = 1,
    watermark' = watermark, baseLookahead' = baseLookahead,
  }

  // The epoch's boundary fold advances the watermark; the twin writes the
  // active base from the post-fold value.
  action foldEpoch = {
    nondet chunk = 0.to(MAX_CHUNK).oneOf()
    all {
      phase == 1,
      watermark' = watermark + chunk,
      baseActive' = if (FREEZE_AFTER_FOLD) watermark + chunk else baseActive,
      phase' = 2,
      baseLookahead' = baseLookahead,
    }
  }

  // Epoch close: the lookahead projection writes the next window's base
  // from the live post-fold watermark.
  action closeWindow = all {
    phase == 2,
    baseLookahead' = watermark,
    phase' = 0,
    watermark' = watermark, baseActive' = baseActive,
  }

  action step = any { openWindow, foldEpoch, closeWindow }

  // A window's two schedule writes agree — the proposer's claimed base
  // (lookahead) and the verifier's resolved base (active) are one value.
  val invWindowAgree = (phase != 0) implies (baseActive == baseLookahead)
}

module witness_fold_ok {
  import witness_fold(CAP = 2, MAX_GROW = 2, MAX_LEAVES = 6, REPORTER_RANGE = false).*
}

module witness_fold_reporter_range {
  import witness_fold(CAP = 2, MAX_GROW = 2, MAX_LEAVES = 6, REPORTER_RANGE = true).*
}

module reveal_fold_ok {
  import reveal_fold(
    MAX_EPOCH = 4, FENCE_OFF = false, CHAIN_CUMULATIVE = false,
    DRAIN_REFOLDS = false
  ).*
}

module reveal_fold_no_fence {
  import reveal_fold(
    MAX_EPOCH = 4, FENCE_OFF = true, CHAIN_CUMULATIVE = false,
    DRAIN_REFOLDS = false
  ).*
}

module reveal_fold_cumulative_chain {
  import reveal_fold(
    MAX_EPOCH = 4, FENCE_OFF = false, CHAIN_CUMULATIVE = true,
    DRAIN_REFOLDS = false
  ).*
}

module reveal_fold_drain_refolds {
  import reveal_fold(
    MAX_EPOCH = 4, FENCE_OFF = false, CHAIN_CUMULATIVE = false,
    DRAIN_REFOLDS = true
  ).*
}

module witness_window_ok {
  import witness_window(MAX_CHUNK = 2, FREEZE_AFTER_FOLD = false).*
}

module witness_window_live_base {
  import witness_window(MAX_CHUNK = 2, FREEZE_AFTER_FOLD = true).*
}
