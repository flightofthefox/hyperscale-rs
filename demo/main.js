// Drives the wasm session and renders its event stream.
//
// The page holds no protocol knowledge: it reacts to typed events and derives
// the topology from shard paths (one path parents another when it prefixes
// it). Anything it cannot render still reaches the log rather than vanishing.

import init, { DemoSession, last_panic } from "./vendor/hyperscale_demo.js";

const SEED = 42;
const SHARD_SIZE = 4;
const MAX_SHARDS = 2;
// Largest real gap a single frame may replay, so returning to a backgrounded
// tab resumes instead of lurching through minutes of simulated time at once.
const MAX_CATCHUP_MS = 250;
// Width of the visible weighted-time window. A shard commits roughly 3.5
// blocks per second of attested time, so this holds ~50 blocks per lane —
// wide enough to show a lane's rhythm and a split's before-and-after, narrow
// enough that each block is still a block rather than a pixel in a bar.
const WINDOW_MS = Number(new URLSearchParams(location.search).get("window")) * 1000 || 15_000;
// The simulation keeps every committed block in memory — the in-memory
// storage backend has no GC — so a session grows by roughly 7 MiB per
// simulated minute and never gives it back. Stop at a bound rather than let
// a tab left open overnight take the browser down with it.
const MAX_SIM_MS = 20 * 60_000;

// Playback above which a delivery no longer resolves as a moving dot: one
// frame then spans more simulated time than a message spends in flight, so
// every dot would be born already landed. Past it the network view switches
// to edge weight, which says the same thing at a rate a frame can carry.
const DOT_SPEED_LIMIT = 4;
// How much simulated time the traffic meter totals over. Long enough to read
// as a rate rather than a flicker, short enough that a spike is still a spike.
const TRAFFIC_WINDOW_MS = 2_000;
// Message classes, urgent first — the order the protocol prioritises them in,
// which is the order the meter stacks and the legend lists.
const CLASSES = ["consensus", "block_completion", "cross_shard_progress", "recovery", "bulk"];

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
// Playback rates, slowest first. The gears below 1× are what make the network
// panel readable: a message spends about 150ms in flight, which at real time
// is nine frames and far too quick to follow, and slowing the playback is the
// honest way to see it — the latency is what it is, only the watching changes.
const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];

const state = {
  session: null,
  playing: true,
  lastClock: null,
  // The viewport's own clock. Advanced by the simulated span every frame,
  // where `wt` only moves when an event happens to arrive — roughly every
  // 293ms of attested time. Panning on `wt` makes the timeline hop between
  // block arrivals; panning on this makes it glide.
  viewWt: 0,
  speed: SPEEDS.includes(Number(params.get("speed"))) ? Number(params.get("speed")) : 1,
  // Simulated milliseconds owed but not yet stepped. The session takes whole
  // milliseconds, so a slow gear asking for 1.6ms a frame would round to 2 and
  // quietly run a quarter faster than the label claims. Carrying the remainder
  // keeps the playback rate exactly what the control says it is.
  stepCarry: 0,
  wt: 0,
  shards: [],           // live trie leaves, in order
  beacon: [],           // { wt, epoch } — one per committed epoch
  lanes: new Map(),     // path -> { blocks, heights, retiredAt, terminal }
  splits: [],           // { wt, appeared, retired }
  // Cross-shard arcs. One per direction per settlement, from the block
  // whose state was provisioned to the block that committed the certified
  // outcome. Endpoints are (shard, height) pairs resolved to positions at
  // render time, so an arc whose blocks have scrolled off simply stops
  // being drawn.
  arcs: [],             // { wt, from, fromHeight, to, toHeight, txs }
  waves: [],            // { wt, shard, height, wave, participants, txs }
  // The harness clock, advanced by exactly what is handed to `step`. The
  // network view runs on this: a message in flight exists on the clock the
  // session is stepping, where `wt` is what consensus has since attested and
  // necessarily trails it. The two are never compared.
  simNow: 0,
  hosts: [],            // { host, shards, pooled } — every host, in host order
  flights: [],          // { from, to, sentAt, deliveredAt, class } in flight
  edges: new Map(),     // "a-b" -> deliveries seen, decayed — degrade-mode weight
  traffic: [],          // { at, byClass: Map, sampled, dropped } per step
  layout: new Map(),    // host -> { x, y, tx, ty } — positions ease to targets
  dirtyLayout: true,
  txs: new Map(),       // label -> { status, height, submittedWt }
  // Which blocks carry each transaction, as `path height` keys. Built
  // from the certificates that name it, so it covers every shard that ran
  // it rather than only the one that reported its status.
  txBlocks: new Map(),  // label -> Set<string>
  selected: null,       // traced transaction label, or null
  log: [],
  events: 0,
};

const blockKey = (path, height) => `${path} ${height}`;

function laneFor(path) {
  if (!state.lanes.has(path)) {
    state.lanes.set(path, {
      blocks: [], heights: new Map(), retiredAt: null, height: 0, terminal: null,
    });
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
  state.viewWt = Math.max(state.viewWt, state.wt);
  const k = event.kind;
  switch (k.type) {
    case "blockCommitted": {
      const lane = laneFor(k.shard);
      lane.blocks.push({ wt: event.wt, height: k.height, fallback: k.fallback });
      lane.heights.set(k.height, event.wt);
      lane.height = k.height;
      // Drop what has scrolled off: this page is meant to run for hours.
      const floor = state.wt - WINDOW_MS * 2;
      if (lane.blocks.length > 64 && lane.blocks[0].wt < floor) {
        for (const b of lane.blocks) if (b.wt < floor) lane.heights.delete(b.height);
        lane.blocks = lane.blocks.filter((b) => b.wt >= floor);
      }
      break;
    }
    case "provisionsVerified":
      state.arcs.push({
        wt: event.wt,
        from: k.from, fromHeight: k.fromHeight, to: k.to, toHeight: k.toHeight, txs: k.txs,
      });
      break;
    case "executionCertified":
      // The certificate rides the same edge the provisions did, so it draws
      // no line of its own. What it adds is which block on which shard ran
      // each transaction — the index the tracer dims everything else
      // against, and the only source that covers both participants.
      for (const [tx] of k.outcomes) {
        if (!state.txBlocks.has(tx)) state.txBlocks.set(tx, new Set());
        state.txBlocks.get(tx).add(blockKey(k.shard, k.height));
      }
      break;
    case "waveFinalized":
      // A wave with one participant never left its shard; its transactions'
      // own status already tells that story, and logging it would bury the
      // settlements that did cross.
      if (k.participants.length > 1) {
        state.waves.push({
          wt: event.wt, shard: k.shard, height: k.height,
          wave: k.wave, participants: k.participants, txs: k.txs,
        });
        note(
          event.wt,
          `wave ${k.wave} settled across ${k.participants.map(labelOf).join(" + ")}` +
            ` at h${k.height} — opened h${k.openedAt}`,
          "ok",
        );
      }
      break;
    case "shardTerminal": {
      const lane = laneFor(k.shard);
      lane.terminal = { height: k.height, handoffFrom: k.handoffFrom };
      note(
        event.wt,
        `${labelOf(k.shard)} terminal at h${k.height}` +
          (k.handoffFrom == null ? "" : ` — certifying its handoff since h${k.handoffFrom}`),
        "split",
      );
      break;
    }
    case "messageDelivered": {
      // Kept only while it is still in the air. A delivery reported after it
      // already landed — which is every delivery once a frame outruns the
      // flight time — is counted by the meter and drawn by nothing.
      if (k.deliveredAt > state.simNow) {
        state.flights.push({
          from: k.from, to: k.to, sentAt: k.sentAt, deliveredAt: k.deliveredAt, cls: k.class,
        });
      }
      const key = k.from < k.to ? `${k.from}-${k.to}` : `${k.to}-${k.from}`;
      state.edges.set(key, (state.edges.get(key) ?? 0) + 1);
      break;
    }
    case "trafficSampled":
      state.traffic.push({
        at: state.simNow,
        byClass: new Map(k.byClass.map(([cls, deliveries]) => [cls, deliveries])),
        sampled: k.sampled,
        dropped: k.dropped,
      });
      break;
    case "hostsChanged":
      state.hosts = k.hosts;
      state.dirtyLayout = true;
      break;
    case "beaconBlockCommitted":
      state.beacon.push({ wt: event.wt, epoch: k.epoch });
      if (state.beacon.length > 200) state.beacon.shift();
      note(event.wt, `beacon committed epoch ${k.epoch}`);
      break;
    case "topologyChanged":
      state.shards = k.shards.map((s) => s);
      // The clusters are the live committees, so a new partition is a new
      // arrangement of the network view as well as of the trie.
      state.dirtyLayout = true;
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

// ── the tracer ───────────────────────────────────────────────────────────
// Selecting a transaction dims every mark that does not carry its label.
// Both predicates answer false while nothing is selected, so the whole
// timeline stays at full opacity until a viewer asks a question of it.
const dimmed = (txs) => state.selected != null && !txs.includes(state.selected);
const dimmedBlock = (path, height) =>
  state.selected != null && !state.txBlocks.get(state.selected)?.has(blockKey(path, height));

// Arcs and convergence points live as long as the blocks they attach to,
// which the lanes already bound by the visible window.
function prune() {
  // Traffic prunes on the harness clock, not attested time: it is what the
  // transport did, and it is measured on the clock the transport runs on.
  state.flights = state.flights.filter((f) => f.deliveredAt > state.simNow);
  const trafficFloor = state.simNow - TRAFFIC_WINDOW_MS;
  if (state.traffic.length && state.traffic[0].at < trafficFloor) {
    state.traffic = state.traffic.filter((t) => t.at >= trafficFloor);
  }
  // Edge weight decays rather than accumulating, so a busy edge that goes
  // quiet fades instead of staying lit for the rest of the session.
  for (const [key, weight] of state.edges) {
    if (weight < 0.5) state.edges.delete(key);
    else state.edges.set(key, weight * 0.94);
  }

  const floor = state.wt - WINDOW_MS * 2;
  if (state.arcs.length > 512) state.arcs = state.arcs.filter((a) => a.wt >= floor);
  if (state.waves.length > 256) state.waves = state.waves.filter((w) => w.wt >= floor);
  // Transactions and the blocks that carry them are kept long past the
  // window so a settled one can still be traced, but not forever. Both maps
  // are insertion-ordered and drop together, so nothing left in the panel
  // loses the marks the tracer would dim against.
  while (state.txs.size > 200) {
    const oldest = state.txs.keys().next().value;
    state.txs.delete(oldest);
    state.txBlocks.delete(oldest);
    txRows.delete(oldest);
    if (state.selected === oldest) state.selected = null;
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

// Reading clientWidth right before writing viewBox forces a layout every
// frame. Cache the measurement and refresh it only when the window resizes.
let laneWidth = 0;
let laneViewBox = "";
const measureLanes = () => { laneWidth = $("lanes").clientWidth || 900; };
window.addEventListener("resize", () => { laneWidth = 0; netWidth = 0; });

function renderLanes() {
  const svg = $("lanes");
  if (!laneWidth) measureLanes();
  const width = laneWidth;
  const paths = [...state.lanes.keys()];
  const rowH = 46;
  // The beacon gets its own lane above the shards: it is the chain that
  // decides who governs them, and it ticks once an epoch rather than
  // continuously, so it reads as a different kind of thing.
  const height = Math.max(120, 26 + (paths.length + 1) * rowH);
  const viewBox = `0 0 ${width} ${height}`;
  if (viewBox !== laneViewBox) {
    laneViewBox = viewBox;
    svg.setAttribute("height", height);
    svg.setAttribute("viewBox", viewBox);
  }
  svg.replaceChildren();

  const t1 = Math.max(state.viewWt, state.wt, WINDOW_MS);
  const t0 = t1 - WINDOW_MS;
  const x = (wt) => 46 + ((wt - t0) / WINDOW_MS) * (width - 60);

  // Grid roughly every fifth of the window, on a round number of seconds.
  const gridStep = Math.max(1000, Math.round(WINDOW_MS / 5 / 1000) * 1000);
  const firstTick = Math.ceil(t0 / gridStep) * gridStep;
  for (let t = firstTick; t <= t1; t += gridStep) {
    el("line", { class: "gridline", x1: x(t), y1: 16, x2: x(t), y2: height - 4 }, svg);
    el("text", { class: "gridlab", x: x(t) + 3, y: 12 }, svg).textContent = `${Math.round(t / 1000)}s`;
  }

  // Lane rows are laid out before anything is drawn, so the arcs between
  // them can go down first and sit under the blocks they connect.
  const laneY = new Map();
  paths.forEach((path, row) => laneY.set(path, 30 + (row + 1) * rowH));

  for (const arc of state.arcs) {
    const y0 = laneY.get(arc.from);
    const y1 = laneY.get(arc.to);
    const wt0 = state.lanes.get(arc.from)?.heights.get(arc.fromHeight);
    const wt1 = state.lanes.get(arc.to)?.heights.get(arc.toHeight);
    // Either end may have scrolled out of the window, or not have arrived
    // yet: draw nothing rather than guess where it belongs.
    if (y0 == null || y1 == null || wt0 == null || wt1 == null) continue;
    if (wt1 < t0 || wt0 > t1) continue;
    const [x0, x1] = [x(wt0), x(wt1)];
    const mid = (x0 + x1) / 2;
    el("path", {
      class: `arc${dimmed(arc.txs) ? " dim" : ""}`,
      d: `M ${x0} ${y0} C ${mid} ${y0}, ${mid} ${y1}, ${x1} ${y1}`,
      stroke: colorOf(arc.from),
    }, svg);
    el("circle", {
      class: `archead${dimmed(arc.txs) ? " dim" : ""}`,
      cx: x1, cy: y1, r: 2.6, fill: colorOf(arc.from),
    }, svg);
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

  paths.forEach((path) => {
    const lane = state.lanes.get(path);
    const y = laneY.get(path);
    const c = colorOf(path);
    el("text", { class: "lane-label", x: 4, y, fill: c }, svg).textContent = labelOf(path);
    const endX = lane.retiredAt == null ? x(t1) : x(lane.retiredAt);
    el("line", { class: "baseline", x1: 46, y1: y, x2: endX, y2: y, stroke: c }, svg);
    // A shard on its way out spends its last epoch certifying the handoff
    // rather than merely running. Marking that stretch is what separates a
    // chain that finished from one that stopped.
    const handoff = lane.terminal?.handoffFrom;
    if (handoff != null) {
      const from = lane.heights.get(handoff);
      el("line", {
        class: "handoff",
        x1: x(Math.max(from ?? t0, t0)), y1: y - 13, x2: endX, y2: y - 13, stroke: c,
      }, svg);
    }
    for (const b of lane.blocks) {
      if (b.wt < t0) continue;
      const inHandoff = handoff != null && b.height >= handoff;
      el("rect", {
        class: `blk${b.fallback ? " fallback" : ""}${inHandoff ? " handoff" : ""}` +
          `${dimmedBlock(path, b.height) ? " dim" : ""}`,
        x: x(b.wt) - 3, y: y - 8, width: 6, height: 16,
        fill: c, stroke: c,
      }, svg);
    }
    if (lane.terminal) {
      el("line", { class: "terminal", x1: endX, y1: y - 11, x2: endX, y2: y + 11, stroke: c }, svg);
      el("text", { class: "tip", x: endX - 4, y: y - 14, fill: c }, svg)
        .textContent = `TERMINAL h${lane.terminal.height}`;
    } else if (lane.height) {
      // The tip height, so the lane carries scale: marks show rhythm, this
      // shows how far the chain has actually got.
      el("text", { class: "tip", x: width - 6, y: y - 12, fill: c }, svg)
        .textContent = `h${lane.height}`;
    }
  });

  // Where a settlement round closed: both sides' arcs land on this block.
  for (const wave of state.waves) {
    const y = laneY.get(wave.shard);
    const wt = state.lanes.get(wave.shard)?.heights.get(wave.height);
    if (y == null || wt == null || wt < t0) continue;
    el("circle", {
      class: `converge${dimmed(wave.txs) ? " dim" : ""}`,
      cx: x(wt), cy: y, r: 6.5, stroke: colorOf(wave.shard),
    }, svg);
  }

  for (const s of state.splits) {
    if (s.wt < t0) continue;
    el("line", { class: "splitmark", x1: x(s.wt), y1: 16, x2: x(s.wt), y2: height - 12 }, svg);
    el("text", { class: "splitlab", x: x(s.wt), y: height - 2 }, svg).textContent = "SPLIT";
  }
}

// ── the network view ─────────────────────────────────────────────────────
// Hosts grouped by the committee they serve, with the free pool as a group of
// its own. Positions are abstract: this simulation has no geography, and
// latency is a configured value with jitter rather than a function of
// distance, so a map would assert a story the run does not have.

// Which group a host belongs to: the live shard it serves, or the pool. A
// grown host keeps its retired parent alongside its live child, so the
// membership is the intersection with the live partition rather than
// whatever the host still carries.
function groupOf(host) {
  const live = host.shards.filter((s) => state.shards.includes(s));
  return live.length ? live[0] : null;
}

const NET_ROW = 108;
const NET_PAD = 30;

// Recompute where each host belongs. Targets only — the drawn positions ease
// toward them, so a split slides its new committees into place instead of
// teleporting every node the instant the partition changes.
function retarget(width) {
  // Every live shard gets a group whether or not it is staffed yet; the pool
  // gets one only while somebody is sitting in it.
  const present = [...state.shards];
  if (state.hosts.some((h) => groupOf(h) === null)) present.push(null);
  const span = (width - NET_PAD * 2) / Math.max(present.length, 1);
  const radius = Math.max(14, Math.min(34, span / 2 - 22));
  const mid = NET_ROW / 2;

  return present.map((group, column) => {
    const cx = NET_PAD + span * (column + 0.5);
    const members = state.hosts.filter((h) => groupOf(h) === group);
    members.forEach((host, i) => {
      // Spread around the group's circle by index, so a host keeps its place
      // within a committee for as long as it stays in one.
      const angle = (i / Math.max(members.length, 1)) * Math.PI * 2 - Math.PI / 2;
      const spot = state.layout.get(host.host) ?? { x: cx, y: mid };
      spot.tx = cx + Math.cos(angle) * radius;
      spot.ty = mid + Math.sin(angle) * radius;
      state.layout.set(host.host, spot);
    });
    return { group, cx, r: radius + 16 };
  });
}

let netWidth = 0;
let netGroups = [];
let netViewBox = "";

function renderNetwork() {
  const svg = $("net");
  if (!netWidth) {
    netWidth = svg.clientWidth || 900;
    state.dirtyLayout = true;
  }
  if (state.dirtyLayout) { netGroups = retarget(netWidth); state.dirtyLayout = false; }
  const viewBox = `0 0 ${netWidth} ${NET_ROW}`;
  if (viewBox !== netViewBox) {
    netViewBox = viewBox;
    svg.setAttribute("height", NET_ROW);
    svg.setAttribute("viewBox", viewBox);
  }
  svg.replaceChildren();
  if (!state.hosts.length) return;

  // Ease toward the targets. Membership animates; the layout is never
  // recomputed out from under a node mid-move.
  for (const spot of state.layout.values()) {
    spot.x += (spot.tx - spot.x) * 0.12;
    spot.y += (spot.ty - spot.y) * 0.12;
  }

  for (const { group, cx, r } of netGroups) {
    el("ellipse", { class: "cluster", cx, cy: NET_ROW / 2, rx: r, ry: r * 0.86 }, svg);
    el("text", {
      class: "cluster-label", x: cx, y: 12,
      fill: group === null ? "var(--muted)" : colorOf(group),
    }, svg).textContent = group === null ? "FREE POOL" : labelOf(group);
  }

  const at = (host) => state.layout.get(host);
  const dots = state.speed <= DOT_SPEED_LIMIT;
  if (!dots) {
    // Past the dot limit, weight each edge by what the sample saw cross it.
    const heaviest = Math.max(...state.edges.values(), 1);
    for (const [key, weight] of state.edges) {
      const [a, b] = key.split("-").map(Number);
      const from = at(a);
      const to = at(b);
      if (!from || !to) continue;
      el("line", {
        class: "wire hot", x1: from.x, y1: from.y, x2: to.x, y2: to.y,
        "stroke-width": 0.4 + (weight / heaviest) * 2.4,
        opacity: 0.15 + (weight / heaviest) * 0.5,
      }, svg);
    }
  }

  for (const host of state.hosts) {
    const spot = at(host.host);
    if (!spot) continue;
    const pooled = groupOf(host) === null;
    const g = el("g", { class: `host${pooled ? " pooled" : ""}` }, svg);
    el("title", {}, g).textContent = pooled
      ? `host ${host.host} — free pool, ${host.pooled} following the beacon`
      : `host ${host.host} — serving ${host.shards.map(labelOf).join(", ")}`;
    el("circle", { cx: spot.x, cy: spot.y, r: 11 }, g);
    el("text", { x: spot.x, y: spot.y }, g).textContent = host.host;
  }

  if (!dots) return;
  for (const flight of state.flights) {
    const from = at(flight.from);
    const to = at(flight.to);
    if (!from || !to || flight.deliveredAt <= flight.sentAt) continue;
    const t = (state.simNow - flight.sentAt) / (flight.deliveredAt - flight.sentAt);
    if (t < 0 || t > 1) continue;
    el("circle", {
      class: `dot ${flight.cls}`,
      cx: from.x + (to.x - from.x) * t,
      cy: from.y + (to.y - from.y) * t,
      r: 2.8,
    }, svg);
  }
}

// Per-class totals over the window, exact — they cover every delivery, not
// just the ones the sample kept.
function renderMeter() {
  const totals = new Map(CLASSES.map((cls) => [cls, 0]));
  let sampled = 0;
  let dropped = 0;
  for (const step of state.traffic) {
    for (const [cls, deliveries] of step.byClass) {
      totals.set(cls, (totals.get(cls) ?? 0) + deliveries);
    }
    sampled += step.sampled;
    dropped += step.dropped;
  }
  const carried = [...totals.values()].reduce((a, b) => a + b, 0);

  const key = document.createElement("div");
  key.className = "meterkey";
  if (!carried) {
    const quiet = document.createElement("span");
    quiet.className = "k none";
    quiet.textContent = "nothing carried";
    key.appendChild(quiet);
  }
  for (const cls of CLASSES) {
    const count = totals.get(cls) ?? 0;
    if (!count) continue;
    const k = document.createElement("span");
    k.className = "k";
    k.innerHTML =
      `<span class="sw m ${cls}"></span>${cls.replace(/_/g, " ")} <b>${count}</b>`;
    key.appendChild(k);
  }
  $("meter").replaceChildren(key);

  const thinned = dropped > 0
    ? ` <span class="thinned">Drawing ${sampled} of ${sampled + dropped}</span> —` +
      ` the counts cover all of them.`
    : "";
  $("netcap").innerHTML =
    `Deliveries over the last ${TRAFFIC_WINDOW_MS / 1000}s of simulated time, ` +
    `urgent class first. Latency is configured, not geographic &mdash; ` +
    `positions carry no distance.${thinned}`;
  $("netclock").textContent = state.speed > DOT_SPEED_LIMIT
    ? `${state.speed}× — edge weight, too fast for single messages`
    : `t = ${fmtWt(state.simNow)} on the harness clock`;
}

// Row elements are built once per transaction and updated in place. Rebuilding
// them would replace the node under the pointer between mousedown and mouseup,
// and a click only fires when both land on the same element — at this panel's
// refresh rate that loses most of them.
const txRows = new Map(); // label -> { root, bar, pill, span, sig }
let txOrder = "";

function txRow(id) {
  if (txRows.has(id)) return txRows.get(id);
  const root = document.createElement("button");
  root.className = "tx";
  root.type = "button";
  root.dataset.tx = id;
  // The grid lives on a wrapper rather than the button itself: WebKit has a
  // long history of not laying out a button's children as grid or flex items,
  // and a row that collapses is a row nobody can aim at.
  const grid = document.createElement("span");
  grid.className = "txgrid";
  const label = document.createElement("span");
  label.className = "id";
  label.textContent = id;
  const track = document.createElement("span");
  track.className = "bar";
  const bar = document.createElement("i");
  track.appendChild(bar);
  const pill = document.createElement("span");
  const span = document.createElement("span");
  span.className = "span";
  for (const child of [label, track, pill, span]) grid.appendChild(child);
  root.appendChild(grid);
  const row = { root, bar, pill, span, sig: null };
  txRows.set(id, row);
  return row;
}

function renderTxs() {
  const box = $("txs");
  if (!state.txs.size) return;
  const rows = [...state.txs.entries()].slice(-8).reverse();
  const nodes = rows.map(([id, tx]) => {
    const row = txRow(id);
    const done = ["succeeded", "aborted", "rejected"].includes(tx.status);
    const pct = done ? 100 : tx.status === "committed" ? 66 : 25;
    const shards = state.txBlocks.get(id)?.size ?? 0;
    const traced = state.selected === id;
    const sig = `${tx.status}|${tx.height}|${shards}|${traced}`;
    if (row.sig !== sig) {
      row.sig = sig;
      row.root.className = `tx${traced ? " traced" : ""}`;
      row.root.setAttribute("aria-pressed", String(traced));
      row.bar.style.width = `${pct}%`;
      row.pill.className = `pill ${tx.status}`;
      row.pill.textContent =
        tx.status.toUpperCase() + (tx.height != null && !done ? ` h${tx.height}` : "");
      row.span.textContent = shards > 1 ? `${shards} shards` : "";
    }
    return row.root;
  });
  // Reordering detaches nodes too, so only touch the container when the
  // visible set actually moved.
  const order = rows.map(([id]) => id).join(",");
  if (order !== txOrder) {
    txOrder = order;
    box.replaceChildren(...nodes);
  }
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
  $("tracing").textContent = state.selected == null
    ? "click a transaction to trace it"
    : `tracing ${state.selected} — click again to clear`;
}

function renderLegend() {
  $("legend").replaceChildren(...[...state.lanes.keys()].map((p) => {
    const s = document.createElement("span");
    s.className = "k";
    s.innerHTML = `<span class="sw" style="background:${colorOf(p)}"></span>${labelOf(p)}`;
    return s;
  }));
}

let lastPanels = 0;
function render(clock = 0) {
  if (dirtyTopology) { renderTrie(); renderLegend(); dirtyTopology = false; }
  renderLanes();
  // The node graph animates messages in flight, so it repaints with the
  // timeline rather than with the text panels.
  renderNetwork();
  // Text panels change far slower than the timeline pans, and rebuilding them
  // costs layout on every row. 8Hz is past the point anyone reads a difference.
  if (clock - lastPanels > 125) {
    lastPanels = clock;
    renderTxs();
    renderLog();
    renderMeter();
    renderChrome();
  }
}

// ── main loop ────────────────────────────────────────────────────────────
function frame(clock) {
  if (state.playing && state.session) {
    // Advance by the real time this frame actually took, so 1× is wall clock:
    // a second of attested time per second of watching. A fixed span per frame
    // would instead run at whatever multiple the frame rate implied, which
    // compressed each beacon epoch's work into a visible hitch.
    const elapsed = state.lastClock == null ? 0 : clock - state.lastClock;
    state.lastClock = clock;
    // A backgrounded tab stops firing frames; on return the gap would be
    // seconds. Cap it so the session resumes rather than lurching forward.
    const simMs = Math.min(elapsed, MAX_CATCHUP_MS) * state.speed;
    state.viewWt += simMs;

    let events = [];
    try {
      state.stepCarry += simMs;
      const stepped = Math.floor(state.stepCarry);
      state.stepCarry -= stepped;
      if (stepped > 0) {
        // Advance the local copy of the harness clock by exactly what the
        // session is given, so a delivery's two instants land on the same
        // timeline the network view interpolates along.
        state.simNow += stepped;
        events = state.session.step(stepped);
      }
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
    prune();
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
    render(clock);
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
  state.hosts = state.session.hosts();
  for (const p of state.shards) laneFor(p);
  const boot = Math.round(performance.now() - t0);
  $("badge").textContent = `LIVE — booted in ${boot}ms`;
  $("badge").className = "badge";
  $("meta").textContent = `seed ${SEED} · ${SHARD_SIZE} validators per shard`;
  note(0, "genesis — one shard, committee seated", "hl");
  buildSpeeds();

  // ?warmup=<seconds> runs the session forward before the first paint, for
  // deep-linking past the wait to a state worth looking at — the split lands
  // around three minutes of attested time. Blocking on purpose: there is
  // nothing to show until it lands.
  const warmup = Number(params.get("warmup")) || 0;
  if (warmup > 0) {
    $("badge").textContent = `skipping to ${warmup}s…`;
    $("badge").className = "badge boot";
    for (let t = 0; t < warmup * 1000; t += 4000) {
      state.simNow += 4000;
      for (const e of state.session.step(4000)) apply(e);
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
  // Drop the gap the pause opened up rather than replaying it on resume.
  state.lastClock = null;
  $("play").innerHTML = state.playing ? "&#10074;&#10074; PAUSE" : "&#9654; PLAY";
});
// Grouped so the slow gears read as a deliberate range rather than a mistake:
// nobody thinks to look below real time until the control says it is there.
function buildSpeeds() {
  const select = $("speed");
  const groups = [
    ["slower than real time", SPEEDS.filter((s) => s < 1)],
    ["real time and faster", SPEEDS.filter((s) => s >= 1)],
  ];
  for (const [label, speeds] of groups) {
    const group = document.createElement("optgroup");
    group.label = label;
    for (const speed of speeds) {
      const option = document.createElement("option");
      option.value = String(speed);
      option.textContent = `${speed}×`;
      group.appendChild(option);
    }
    select.appendChild(group);
  }
  select.value = String(state.speed);
}

$("speed").addEventListener("change", (ev) => {
  state.speed = Number(ev.target.value);
  // Whatever the old gear owed is not what the new one owes.
  state.stepCarry = 0;
});
$("submit").addEventListener("click", () => {
  if (state.session) state.session.submit_transfer();
});
$("txs").addEventListener("click", (ev) => {
  const id = ev.target.closest("[data-tx]")?.dataset.tx;
  if (!id) return;
  state.selected = state.selected === id ? null : id;
  renderTxs();
  renderChrome();
  // The timeline has to answer the tap here rather than wait for the next
  // frame: the animation loop only renders while the session is playing, and
  // on a touch device it is suspended outright for the length of a scroll.
  renderLanes();
});

main();
