// Drives the wasm session and renders its event stream.
//
// The page holds no protocol knowledge: it reacts to typed events and derives
// the topology from shard paths (one path parents another when it prefixes
// it). Anything it cannot render still reaches the log rather than vanishing.

import init, { DemoSession, last_panic } from "./vendor/hyperscale_demo.js";

const SEED = 42;
const SHARD_SIZE = 4;
const MAX_SHARDS = 2;
const STEP_MS = 500;      // simulated ms per animation frame at 1x
// Width of the visible weighted-time window. A shard commits roughly three
// blocks a second of attested time, so much past this and the marks merge
// into a bar instead of reading as discrete blocks.
const WINDOW_MS = 45_000;
// The simulation keeps every committed block in memory — the in-memory
// storage backend has no GC — so a session grows by roughly 7 MiB per
// simulated minute and never gives it back. Stop at a bound rather than let
// a tab left open overnight take the browser down with it.
const MAX_SIM_MS = 20 * 60_000;

// Colour by trie path so a shard keeps its identity across a split; the two
// children of a shard never collide because their paths differ in the last bit.
const PALETTE = ["#3E8DCB", "#CC6390", "#8A66DB", "#2EA871", "#C77B4C"];
const BEACON_COLOR = "#B5872F";
const colorOf = (path) => {
  let h = 0;
  for (const ch of `s${path}`) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
};
const labelOf = (path) => (path === "" ? "ROOT" : path);

const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const el = (name, attrs, parent) => {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  parent?.appendChild(node);
  return node;
};
const fmtWt = (ms) => {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}.${String(Math.floor(ms % 1000)).padStart(3, "0")}`;
};

let dirtyTopology = true;
const params = new URLSearchParams(location.search);
const SPEEDS = [1, 2, 4, 8, 16];

const state = {
  session: null,
  playing: true,
  speed: SPEEDS.includes(Number(params.get("speed"))) ? Number(params.get("speed")) : 1,
  wt: 0,
  shards: [],           // live trie leaves, in order
  beacon: [],           // { wt, epoch } — one per committed epoch
  lanes: new Map(),     // path -> { blocks: [{wt, fallback}], retiredAt: number|null }
  splits: [],           // { wt, appeared, retired }
  txs: new Map(),       // label -> { status, height, submittedWt }
  log: [],
  events: 0,
};

function laneFor(path) {
  if (!state.lanes.has(path)) {
    state.lanes.set(path, { blocks: [], retiredAt: null });
    // A child's lane appears when it commits its first block, which is later
    // than the partition change that created it — so the trie and legend key
    // off lanes rather than off topology events alone.
    dirtyTopology = true;
  }
  return state.lanes.get(path);
}

function note(wt, text, cls = "") {
  state.log.push({ wt, text, cls });
  if (state.log.length > 300) state.log.shift();
}

function apply(event) {
  state.events++;
  state.wt = Math.max(state.wt, event.wt);
  const k = event.kind;
  switch (k.type) {
    case "blockCommitted": {
      const lane = laneFor(k.shard);
      lane.blocks.push({ wt: event.wt, fallback: k.fallback });
      // Drop what has scrolled off: this page is meant to run for hours.
      const floor = state.wt - WINDOW_MS * 2;
      if (lane.blocks.length > 64 && lane.blocks[0].wt < floor) {
        lane.blocks = lane.blocks.filter((b) => b.wt >= floor);
      }
      break;
    }
    case "beaconBlockCommitted":
      state.beacon.push({ wt: event.wt, epoch: k.epoch });
      if (state.beacon.length > 200) state.beacon.shift();
      note(event.wt, `beacon committed epoch ${k.epoch}`);
      break;
    case "topologyChanged":
      state.shards = k.shards.map((s) => s);
      state.splits.push({ wt: event.wt, appeared: k.appeared, retired: k.retired });
      for (const path of k.retired) laneFor(path).retiredAt = event.wt;
      note(
        event.wt,
        `SPLIT — ${k.retired.map(labelOf).join(", ")} → ${k.appeared.map(labelOf).join(" + ")}`,
        "split",
      );
      break;
    case "txSubmitted":
      state.txs.set(k.tx, { status: "pending", height: null, submittedWt: event.wt });
      note(event.wt, `tx ${k.tx} submitted`);
      break;
    case "txStatusChanged": {
      const tx = state.txs.get(k.tx) ?? { submittedWt: event.wt };
      tx.status = k.status;
      tx.height = k.height ?? tx.height;
      state.txs.set(k.tx, tx);
      note(
        event.wt,
        `tx ${k.tx} ${k.status}${k.height != null ? ` at h${k.height}` : ""}`,
        k.status === "succeeded" ? "ok" : "",
      );
      break;
    }
    default:
      // Unknown to this build: surface it rather than dropping it.
      note(event.wt, `unhandled event: ${k.type}`);
  }
}

// ── rendering ────────────────────────────────────────────────────────────
function renderTrie() {
  const svg = $("trie");
  svg.replaceChildren();
  // Every path that has ever existed, so a retired parent still shows.
  const paths = [...state.lanes.keys()].sort((a, b) => a.length - b.length || a.localeCompare(b));
  if (!paths.length) return;
  const byDepth = new Map();
  for (const p of paths) {
    if (!byDepth.has(p.length)) byDepth.set(p.length, []);
    byDepth.get(p.length).push(p);
  }
  const depths = [...byDepth.keys()].sort((a, b) => a - b);
  const pos = new Map();
  const rowH = depths.length > 1 ? Math.min(90, 150 / (depths.length - 1)) : 0;
  depths.forEach((d, row) => {
    const row_ = byDepth.get(d).sort();
    row_.forEach((p, i) => {
      pos.set(p, [(300 / (row_.length + 1)) * (i + 1), 34 + row * rowH]);
    });
  });
  // Edges: a path parents another when it is its prefix and one bit shorter.
  for (const p of paths) {
    const parent = p.slice(0, -1);
    if (p !== "" && pos.has(parent)) {
      const [px, py] = pos.get(parent);
      const [cx, cy] = pos.get(p);
      el("path", {
        class: "tedge",
        d: `M ${px} ${py + 16} C ${px} ${py + 40}, ${cx} ${cy - 40}, ${cx} ${cy - 16}`,
      }, svg);
    }
  }
  for (const p of paths) {
    const [x, y] = pos.get(p);
    const live = state.shards.includes(p);
    const g = el("g", { class: `tnode${live ? "" : " dead"}` }, svg);
    el("rect", { x: x - 30, y: y - 16, width: 60, height: 32, rx: 6, stroke: colorOf(p) }, g);
    el("text", { x, y: y + 1 }, g).textContent = labelOf(p);
    el("text", { x, y: y + 11, class: "sub" }, g).textContent = live ? "LIVE" : "RETIRED";
  }
}

function renderLanes() {
  const svg = $("lanes");
  const width = svg.clientWidth || 900;
  const paths = [...state.lanes.keys()];
  const rowH = 46;
  // The beacon gets its own lane above the shards: it is the chain that
  // decides who governs them, and it ticks once an epoch rather than
  // continuously, so it reads as a different kind of thing.
  const height = Math.max(120, 26 + (paths.length + 1) * rowH);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();

  const t1 = Math.max(state.wt, WINDOW_MS);
  const t0 = t1 - WINDOW_MS;
  const x = (wt) => 46 + ((wt - t0) / WINDOW_MS) * (width - 60);

  // Time grid every 15s of weighted time.
  const firstTick = Math.ceil(t0 / 15000) * 15000;
  for (let t = firstTick; t <= t1; t += 15000) {
    el("line", { class: "gridline", x1: x(t), y1: 16, x2: x(t), y2: height - 4 }, svg);
    el("text", { class: "gridlab", x: x(t) + 3, y: 12 }, svg).textContent = `${Math.round(t / 1000)}s`;
  }

  const beaconY = 30;
  el("text", { class: "lane-label", x: 4, y: beaconY, fill: BEACON_COLOR }, svg).textContent = "BEACON";
  el("line", {
    class: "baseline", x1: 46, y1: beaconY, x2: x(t1), y2: beaconY, stroke: BEACON_COLOR,
  }, svg);
  for (const b of state.beacon) {
    if (b.wt < t0) continue;
    el("rect", {
      class: "blk", x: x(b.wt) - 5, y: beaconY - 9, width: 10, height: 18,
      fill: BEACON_COLOR, stroke: BEACON_COLOR,
    }, svg);
    el("text", { class: "gridlab", x: x(b.wt), y: beaconY + 20, "text-anchor": "middle" }, svg)
      .textContent = `E${b.epoch}`;
  }

  paths.forEach((path, row) => {
    const lane = state.lanes.get(path);
    const y = 30 + (row + 1) * rowH;
    const c = colorOf(path);
    el("text", { class: "lane-label", x: 4, y, fill: c }, svg).textContent = labelOf(path);
    const endX = lane.retiredAt == null ? x(t1) : x(lane.retiredAt);
    el("line", { class: "baseline", x1: 46, y1: y, x2: endX, y2: y, stroke: c }, svg);
    for (const b of lane.blocks) {
      if (b.wt < t0) continue;
      el("rect", {
        class: `blk${b.fallback ? " fallback" : ""}`,
        x: x(b.wt) - 2, y: y - 7, width: 4, height: 14,
        fill: c, stroke: c,
      }, svg);
    }
  });

  for (const s of state.splits) {
    if (s.wt < t0) continue;
    el("line", { class: "splitmark", x1: x(s.wt), y1: 16, x2: x(s.wt), y2: height - 12 }, svg);
    el("text", { class: "splitlab", x: x(s.wt), y: height - 2 }, svg).textContent = "SPLIT";
  }
}

function renderTxs() {
  const box = $("txs");
  if (!state.txs.size) return;
  const rows = [...state.txs.entries()].slice(-8).reverse();
  box.replaceChildren(...rows.map(([id, tx]) => {
    const row = document.createElement("div");
    row.className = "tx";
    const done = ["succeeded", "aborted", "rejected"].includes(tx.status);
    const pct = done ? 100 : tx.status === "committed" ? 66 : 25;
    row.innerHTML =
      `<span class="id">${id}</span>` +
      `<span class="bar"><i style="width:${pct}%"></i></span>` +
      `<span class="pill ${tx.status}">${tx.status.toUpperCase()}${tx.height != null && !done ? ` h${tx.height}` : ""}</span>`;
    return row;
  }));
}

function renderLog() {
  const box = $("log");
  const wasBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
  box.replaceChildren(...state.log.slice(-80).map((e) => {
    const d = document.createElement("div");
    d.className = e.cls;
    d.innerHTML = `<b>${fmtWt(e.wt)}</b>  ${e.text}`;
    return d;
  }));
  if (wasBottom) box.scrollTop = box.scrollHeight;
}

function renderChrome() {
  $("wt").textContent = `WT ${fmtWt(state.wt)}`;
  $("shardcount").textContent = `${state.shards.length} LIVE`;
  $("evcount").textContent = `${state.events} events`;
  const split = state.splits.length > 0;
  $("triecap").innerHTML = split
    ? `<span class="good">hash(r₀ ∥ r₁) = r_root ✓</span> — the root retired into its children.`
    : `A shard <b>is</b> a prefix subtree. Watch it split.`;
}

function renderLegend() {
  $("legend").replaceChildren(...[...state.lanes.keys()].map((p) => {
    const s = document.createElement("span");
    s.className = "k";
    s.innerHTML = `<span class="sw" style="background:${colorOf(p)}"></span>${labelOf(p)}`;
    return s;
  }));
}

function render() {
  if (dirtyTopology) { renderTrie(); renderLegend(); dirtyTopology = false; }
  renderLanes();
  renderTxs();
  renderLog();
  renderChrome();
}

// ── main loop ────────────────────────────────────────────────────────────
function frame() {
  if (state.playing && state.session) {
    let events;
    try {
      events = state.session.step(STEP_MS * state.speed);
    } catch (err) {
      state.playing = false;
      const msg = last_panic() || String(err);
      $("badge").textContent = `HALTED: ${msg.slice(0, 90)}`;
      $("badge").className = "badge boot";
      requestAnimationFrame(frame);
      return;
    }
    const before = state.shards.length;
    for (const e of events) apply(e);
    if (state.wt >= MAX_SIM_MS) {
      state.playing = false;
      $("play").innerHTML = "&#9654; PLAY";
      $("badge").textContent = `SESSION CAPPED at ${MAX_SIM_MS / 60_000} simulated minutes — reload for a fresh run`;
      $("badge").className = "badge boot";
      note(state.wt, "session capped — reload to start a new one", "split");
    }
    if (state.shards.length !== before || events.some((e) => e.kind.type === "topologyChanged")) {
      dirtyTopology = true;
    }
    render();
  }
  requestAnimationFrame(frame);
}

async function main() {
  await init();
  $("badge").textContent = "booting cluster…";
  $("badge").className = "badge boot";
  const t0 = performance.now();
  state.session = new DemoSession(SEED, SHARD_SIZE, MAX_SHARDS);
  state.shards = state.session.shards();
  for (const p of state.shards) laneFor(p);
  const boot = Math.round(performance.now() - t0);
  $("badge").textContent = `LIVE — booted in ${boot}ms`;
  $("badge").className = "badge";
  $("meta").textContent = `seed ${SEED} · ${SHARD_SIZE} validators per shard`;
  note(0, "genesis — one shard, committee seated", "hl");
  $("speed").textContent = `${state.speed}×`;

  // ?warmup=<seconds> runs the session forward before the first paint, for
  // deep-linking past the wait to a state worth looking at — the split lands
  // around three minutes of attested time. Blocking on purpose: there is
  // nothing to show until it lands.
  const warmup = Number(params.get("warmup")) || 0;
  if (warmup > 0) {
    $("badge").textContent = `skipping to ${warmup}s…`;
    $("badge").className = "badge boot";
    for (let t = 0; t < warmup * 1000; t += STEP_MS * 8) {
      for (const e of state.session.step(STEP_MS * 8)) apply(e);
    }
    dirtyTopology = true;
    $("badge").textContent = `LIVE — skipped to ${warmup}s`;
    $("badge").className = "badge";
  }
  render();
  requestAnimationFrame(frame);
}

$("play").addEventListener("click", () => {
  state.playing = !state.playing;
  $("play").innerHTML = state.playing ? "&#10074;&#10074; PAUSE" : "&#9654; PLAY";
});
$("speed").addEventListener("click", () => {
  state.speed = SPEEDS[(SPEEDS.indexOf(state.speed) + 1) % SPEEDS.length];
  $("speed").textContent = `${state.speed}×`;
});
$("submit").addEventListener("click", () => {
  if (state.session) state.session.submit_transfer();
});

main();
