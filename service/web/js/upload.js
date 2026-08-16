/**
 * FIRST LIGHT - upload and stage tracker (plan deliverable C3).
 *
 * WHY this file exists: an operator who cannot see the machine working does not
 * trust it, so every image gets a card naming the three pivoted stages. Stage 3
 * is a STORAGE decision, not an analysis gate: a person tile shows stages 1 and
 * 2 complete and stage 3 red, which is the whole design in one line of UI.
 *
 * API fields read (and nothing else):
 *   POST api/upload             multipart, one request per file, field name "files"
 *     response {items: [TileRecord]}   (a bare list or single record is tolerated)
 *   GET  api/tiles           -> {items: [TileRecord]}
 *     TileRecord.filename        matched against the uploaded file name
 *     TileRecord.status          "processed" | "withheld" | "error" | "needs_geo"
 *     TileRecord.bounds          [w, s, e, n] or null
 *     TileRecord.captured_at     epoch seconds, float
 *     TileRecord.latency_ms      int, end to end
 *     TileRecord.buildings[].class   int 0-3, counted for the severe tally
 *     TileRecord.stored          bool, the privacy gate verdict for STORAGE
 *     TileRecord.withheld_reason string or null
 *     TileRecord.needs_geo       bool
 *   Stage 3 renders from stored + withheld_reason, never from status.
 *
 * Host elements, all optional (init never throws when they are absent):
 *   #upload-cards   per-image cards, left truly empty so its :empty rule holds
 *   #stage-1 #stage-2 #stage-3   corpus stage rows, this module writes the
 *                   .dot state and the .n count and never rewrites the .lab
 *   #withheld-row #withheld-count
 * Bus: listens "upload:files" {files} and "data:changed";
 *      emits "data:changed" when a batch settles.
 */

const POLL_MS = 2500;
const CARD_TIMEOUT_MS = 180000;
const STYLE_ID = "fl-style-upload";

const CSS = `
.dot.skip { background:#4a5566; }
#upload-cards.up-hot { outline:1px dashed var(--green,#76b900); outline-offset:2px; }
.up-sum { font-size:11px; color:var(--dim,#8899aa); margin:2px 0 6px; }
.up-card { background:var(--panel,#11161f); border:1px solid var(--line,#1f2733); border-radius:7px;
  padding:7px 9px; margin-bottom:7px; }
.up-card.bad { border-color:#3a2530; background:#180f14; }
.up-head { display:flex; align-items:center; gap:7px; cursor:pointer; }
.up-name { font-size:12.5px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.up-badge { margin-left:auto; font-size:11px; font-weight:700; letter-spacing:.5px; border-radius:3px;
  padding:1px 6px; border:1px solid var(--line,#1f2733); color:var(--dim,#8899aa); white-space:nowrap; }
.up-badge.ok { border-color:var(--green,#76b900); color:var(--green,#76b900); }
.up-badge.warn { border-color:var(--amber,#ffb84c); color:var(--amber,#ffb84c); }
.up-badge.bad { border-color:var(--red,#ff5c5c); color:var(--red,#ff5c5c); }
.up-x { background:transparent; border:1px solid var(--line,#1f2733); color:var(--dim,#8899aa);
  border-radius:3px; width:19px; height:19px; font-size:11px; line-height:1; padding:0; cursor:pointer; flex:none; }
.up-x:hover { border-color:var(--red,#ff5c5c); color:var(--red,#ff5c5c); }
.up-stage { display:flex; align-items:center; gap:7px; padding:3px 0; font-size:12px; }
.up-dot { width:8px; height:8px; border-radius:50%; background:var(--dim,#8899aa); flex:none; }
.up-dot.busy { background:var(--amber,#ffb84c); animation:up-pulse 1.2s infinite; }
.up-dot.done { background:var(--green,#76b900); }
.up-dot.fail { background:var(--red,#ff5c5c); }
.up-dot.skip { background:#4a5566; }
@keyframes up-pulse { 50% { opacity:.35; } }
.up-lab { color:var(--ink,#dde6ee); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
/* An open touch bubble must not be clipped by the label's own ellipsis. */
.up-lab.tip-open, .up-badge.tip-open { overflow:visible; }
.up-stage.fail .up-lab { color:var(--red,#ff5c5c); font-weight:600; }
.up-stage.skip .up-lab, .up-stage.pending .up-lab { color:var(--dim,#8899aa); }
.up-n { margin-left:auto; color:var(--dim,#8899aa); font-size:11px; white-space:nowrap; }
.up-note { font-size:11px; color:var(--amber,#ffb84c); margin-top:4px; }
.up-note:empty { display:none; }
.up-more { display:none; margin-top:5px; border-top:1px solid var(--line,#1f2733); padding-top:5px;
  font:11px Consolas,monospace; color:var(--dim,#8899aa); white-space:pre-wrap; word-break:break-all; }
.up-card.open .up-more { display:block; }
.up-retry { background:transparent; border:1px solid var(--amber,#ffb84c); color:var(--amber,#ffb84c);
  border-radius:4px; font:600 11px inherit; padding:3px 9px; margin-top:5px; cursor:pointer; }
`;

// ---------------------------------------------------------------- state
let ctx = null;
let host = null;
let sumEl = null;
let pollTimer = null;
let inFlight = 0;

/** Cards keyed by normalized file name. A refresh patches them, never rebuilds. */
const cards = new Map();

/**
 * Shell helpers with quiet fallbacks. WHY: the panel must still render when it
 * is loaded standalone or a helper is renamed under us.
 */
function adapt(raw) {
  const c = raw || {};
  const api = c.api || {};
  return {
    hasBus: !!(c.bus && typeof c.bus.on === "function"),
    get: typeof api.get === "function" ? api.get.bind(api) : plainGet,
    form: typeof api.form === "function" ? api.form.bind(api) : plainForm,
    toast: typeof c.toast === "function" ? c.toast.bind(c) : quietToast,
    operator: typeof c.operator === "function" ? c.operator.bind(c) : readOpName,
    on: c.bus && typeof c.bus.on === "function" ? c.bus.on.bind(c.bus) : () => {},
    emit: c.bus && typeof c.bus.emit === "function" ? c.bus.emit.bind(c.bus) : () => 0,
    fmt: c.fmt || {},
    el: typeof c.el === "function" ? c.el.bind(c) : (id) => document.getElementById(id),
  };
}

async function plainGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

async function plainForm(path, fd) {
  const r = await fetch(path, { method: "POST", body: fd });
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

function readOpName() {
  const el = document.getElementById("opname");
  return el && el.value ? el.value.trim() : "";
}

function quietToast(msg) {
  if (typeof console !== "undefined") console.info("[upload] " + msg);
}

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement("style");
  s.id = STYLE_ID;
  s.textContent = CSS;
  document.head.appendChild(s);
}

/**
 * Set a tooltip that survives the shell's touch handler. WHY the data-tip check:
 * while a tip is open the shell has moved the text out of title, and clobbering
 * title then would leave the operator with an empty bubble.
 */
function setTip(node, text) {
  if (!text) return;
  if (node.dataset.tip !== undefined) node.dataset.tip = text;
  else node.title = text;
}

// ---------------------------------------------------------------- tile reading
function baseName(name) {
  return String(name || "").split(/[\\/]/).pop().toLowerCase();
}

function stemOf(name) {
  const b = baseName(name);
  const dot = b.lastIndexOf(".");
  return dot > 0 ? b.slice(0, dot) : b;
}

function listOf(payload) {
  if (!payload || typeof payload !== "object") return [];
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload.items)) return payload.items;
  if (Array.isArray(payload.tiles)) return payload.tiles;
  if (payload.record) return [payload.record];
  if (payload.filename) return [payload];
  return [];
}

function severeCount(rec) {
  const list = (rec && rec.buildings) || [];
  let n = 0;
  for (const b of list) if (Number(b && b.class) >= 2) n += 1;
  return n;
}

/** Human text for withheld_reason without pretending to know every string. */
function withheldText(reason) {
  const r = String(reason || "").toLowerCase();
  if (r.includes("person")) return "person signal, held out of storage";
  if (r.includes("caption")) return "caption mentioned a person, re-withheld";
  if (r.includes("error") || r.includes("detector") || r.includes("weights"))
    return "detector error, withheld by policy";
  return reason || "privacy gate refused storage";
}

/**
 * Derive the three stage rows from a TileRecord.
 *
 * WHY here: the wire carries outcomes, not UI state, so the mapping from
 * outcome to stage name lives in exactly one place.
 */
function stagesFor(card) {
  const pending = (lab, n) => ({ state: "pending", lab, n: n || "" });
  if (card.phase === "uploading")
    return [
      { state: "busy", lab: "1 analyzing (outlines + grades)", n: "sending" },
      pending("2 indexing (join, doubt, caption)"),
      pending("3 storage decision (privacy gate)"),
    ];
  if (card.phase === "upload-failed")
    return [
      { state: "fail", lab: "upload failed, nothing analyzed", n: "" },
      pending("2 indexing (join, doubt, caption)", "not reached"),
      pending("3 storage decision (privacy gate)", "not reached"),
    ];

  const rec = card.record;
  if (!rec)
    return [
      { state: "busy", lab: "1 analyzing (outlines + grades)", n: "on the box" },
      pending("2 indexing (join, doubt, caption)"),
      pending("3 storage decision (privacy gate)"),
    ];

  const nb = ((rec.buildings || []).length) | 0;
  const sev = severeCount(rec);
  const failed = rec.status === "error";

  const s1 = failed
    ? {
        state: "fail",
        lab: "stage 1 failed, fell back to labelled stub",
        n: nb ? nb + " stub outlines" : "no outlines",
      }
    : {
        state: "done",
        lab: "1 analyzing (outlines + grades)",
        n: nb + " buildings outlined, " + sev + " severe",
      };

  const s2 = failed
    ? { state: "fail", lab: "stage 2 incomplete, working from the labelled stub", n: "" }
    : {
        state: "done",
        lab: "2 indexing (join, doubt, caption)",
        n: rec.needs_geo ? "ranked, no location yet" : "ranked",
      };

  let s3;
  if (rec.stored === false) {
    const reason = String(rec.withheld_reason || "").toLowerCase();
    const noGeo = reason.includes("geo") || reason.includes("location");
    s3 = noGeo
      ? { state: "skip", lab: "3 storage decision skipped, no location yet", n: "" }
      : { state: "fail", lab: "withheld: analyzed, not stored", n: withheldText(rec.withheld_reason) };
  } else if (rec.stored === true) {
    s3 = { state: "done", lab: "3 storage decision (privacy gate)", n: "stored and searchable" };
  } else {
    s3 = { state: "busy", lab: "3 storage decision (privacy gate)", n: "deciding" };
  }
  return [s1, s2, s3];
}

/** Plain-English tips per stage row, so the same markup answers mouse and touch. */
const STAGE_TIP = [
  "xView2 outlines every building in the tile and the vision model grades each one 0 to 3.",
  "Each building is joined to its address, nearby care facilities and resident vulnerability, " +
    "then the fast model votes on the grade to produce an uncertainty figure.",
  "The person detector runs before anything can be written to the archive. Person signal, or any " +
    "detector error, withholds the image from storage. The grades it produced are kept, because a " +
    "person in frame is rescue signal.",
];

function badgeFor(card) {
  if (card.phase === "uploading") return { t: "UPLOADING", c: "", tip: "sending this file to the box" };
  if (card.phase === "upload-failed")
    return { t: "UPLOAD FAILED", c: "bad", tip: "the file never reached the box, so no stage ran" };
  const rec = card.record;
  if (!rec) return { t: "ANALYZING", c: "warn", tip: "on the box now, the card fills in as stages land" };
  if (rec.stored === false)
    return {
      t: "WITHHELD",
      c: "bad",
      tip: "analyzed and ranked, never stored: no archive row, no thumbnail, no embedding and not " +
        "searchable. Authorized review only.",
    };
  if (rec.status === "error")
    return { t: "STUB", c: "warn", tip: "a model was unavailable, so a labelled deterministic stub answered" };
  if (rec.needs_geo)
    return { t: "NEEDS GEO", c: "warn", tip: "accepted with no location, drag it onto the map to place it" };
  return { t: "STORED", c: "ok", tip: "cleared the privacy gate, indexed and searchable" };
}

// ---------------------------------------------------------------- card DOM
/** Lazily add the batch summary so #upload-cards stays :empty until it is used. */
function ensureSummary() {
  if (sumEl || !host) return sumEl;
  sumEl = document.createElement("div");
  sumEl.className = "up-sum";
  host.appendChild(sumEl);
  return sumEl;
}

function teardownIfEmpty() {
  if (!host || cards.size) return;
  host.textContent = "";
  sumEl = null;
}

function mountCard(card) {
  const el = document.createElement("div");
  el.className = "up-card";
  el.innerHTML =
    '<div class="up-head"><span class="up-name"></span><span class="up-badge"></span>' +
    '<button class="up-x" type="button" title="dismiss this card, the tile itself is unaffected">x</button></div>' +
    '<div class="up-stage"><span class="up-dot"></span><span class="up-lab"></span><span class="up-n"></span></div>' +
    '<div class="up-stage"><span class="up-dot"></span><span class="up-lab"></span><span class="up-n"></span></div>' +
    '<div class="up-stage"><span class="up-dot"></span><span class="up-lab"></span><span class="up-n"></span></div>' +
    '<div class="up-note"></div><div class="up-more"></div>';
  card.el = el;
  card.refs = {
    badge: el.querySelector(".up-badge"),
    note: el.querySelector(".up-note"),
    more: el.querySelector(".up-more"),
    rows: Array.from(el.querySelectorAll(".up-stage")).map((r) => ({
      row: r,
      dot: r.querySelector(".up-dot"),
      lab: r.querySelector(".up-lab"),
      n: r.querySelector(".up-n"),
    })),
  };
  const nameEl = el.querySelector(".up-name");
  nameEl.textContent = card.name;
  nameEl.title = card.name;

  el.querySelector(".up-head").addEventListener("click", (e) => {
    if (e.target.classList.contains("up-x")) return;
    el.classList.toggle("open");
  });
  el.querySelector(".up-x").addEventListener("click", () => {
    cards.delete(card.key);
    el.remove();
    paintSummary();
    teardownIfEmpty();
  });
  if (host) {
    ensureSummary();
    host.insertBefore(el, sumEl ? sumEl.nextSibling : host.firstChild);
  }
  return card;
}

/**
 * Patch a card in place. WHY never innerHTML on refresh: a periodic poll must
 * not clear a card the operator is reading, and file names may be hostile.
 */
function paintCard(card) {
  if (!card.el) return;
  const stages = stagesFor(card);
  card.el.classList.toggle("bad", stages.some((s) => s.state === "fail"));

  // class="tip" plus title: the shell's delegated handler makes the same text
  // answer a finger as well as a mouse.
  const b = badgeFor(card);
  card.refs.badge.textContent = b.t;
  card.refs.badge.className = "up-badge tip " + b.c;
  setTip(card.refs.badge, b.tip);

  stages.forEach((s, i) => {
    const r = card.refs.rows[i];
    if (!r) return;
    r.row.className = "up-stage " + s.state;
    r.dot.className = "up-dot " + (s.state === "pending" ? "" : s.state);
    r.lab.textContent = s.lab;
    r.lab.className = "up-lab tip";
    setTip(r.lab, STAGE_TIP[i]);
    r.n.textContent = s.n || "";
  });

  const rec = card.record;
  const notes = [];
  if (card.error) notes.push(card.error);
  if (rec && rec.needs_geo) notes.push("no location yet, drag the tile onto the map to place it");
  if (rec && rec.stored === false)
    notes.push("analyzed and ranked, held out of the archive: authorized review only");
  if (card.timedOut) notes.push("no result after 3 minutes, the tile may still be in the queue");
  card.refs.note.textContent = notes.join(" - ");

  const detail = [];
  if (rec) {
    detail.push("status " + rec.status + ", stored " + String(rec.stored));
    if (rec.withheld_reason) detail.push("withheld_reason " + rec.withheld_reason);
    if (typeof rec.latency_ms === "number") detail.push("latency " + rec.latency_ms + " ms");
    if (rec.bounds) detail.push("bounds " + rec.bounds.map((v) => Number(v).toFixed(5)).join(", "));
    if (rec.captured_at) detail.push("captured " + clock(rec.captured_at));
    detail.push("filename " + rec.filename);
  } else {
    detail.push("no tile record yet");
  }
  if (card.operator) detail.push("uploaded by " + card.operator);
  card.refs.more.textContent = detail.join("\n");

  if (card.phase === "upload-failed" && !card.refs.retry) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "up-retry";
    btn.textContent = "retry this file";
    btn.addEventListener("click", () => {
      if (!card.file) {
        ctx.toast("pick the file again, the browser does not keep it after a reload", "warn");
        return;
      }
      card.phase = "uploading";
      card.error = "";
      btn.remove();
      card.refs.retry = null;
      paintCard(card);
      sendOne(card);
    });
    card.el.appendChild(btn);
    card.refs.retry = btn;
  }
}

function clock(epoch) {
  if (typeof ctx.fmt.clock === "function") {
    const s = ctx.fmt.clock(epoch);
    if (s && s !== "-") return s;
  }
  const n = Number(epoch);
  if (!isFinite(n) || n <= 0) return "unknown";
  return new Date(n > 1e12 ? n : n * 1000).toTimeString().slice(0, 5);
}

function paintSummary() {
  if (!cards.size) return;
  const el = ensureSummary();
  if (!el) return;
  let stored = 0;
  let withheld = 0;
  let working = 0;
  let failed = 0;
  for (const c of cards.values()) {
    if (c.phase === "upload-failed") failed += 1;
    else if (!c.record) working += 1;
    else if (c.record.stored === false) withheld += 1;
    else stored += 1;
  }
  const parts = [];
  if (working) parts.push(working + " in flight");
  parts.push(stored + " stored and searchable");
  if (withheld) parts.push(withheld + " withheld: analyzed, not stored");
  if (failed) parts.push(failed + " upload failed");
  el.textContent = "this batch: " + parts.join(" - ");
}

// ------------------------------------------------------------ corpus stage rows
/** Patch one static rail row: the .dot state and the .n count, never the label. */
function setRailRow(id, state, count, title) {
  const row = ctx.el(id);
  if (!row) return;
  const dot = row.querySelector(".dot");
  if (dot) dot.className = "dot " + state;
  const n = row.querySelector(".n");
  if (n) {
    n.textContent = count;
    if (title) n.title = title;
  }
}

function paintRail(tiles) {
  let buildings = 0;
  let severe = 0;
  let stored = 0;
  let withheld = 0;
  let errored = 0;
  let needsGeo = 0;
  for (const t of tiles) {
    buildings += ((t.buildings || []).length) | 0;
    severe += severeCount(t);
    if (t.stored === false) withheld += 1;
    else stored += 1;
    if (t.status === "error") errored += 1;
    if (t.needs_geo) needsGeo += 1;
  }

  if (!tiles.length) {
    setRailRow("stage-1", "", "waiting");
    setRailRow("stage-2", "", "waiting");
    setRailRow("stage-3", "", "waiting");
  } else {
    setRailRow(
      "stage-1",
      errored ? "bad" : "done",
      tiles.length + " tiles, " + buildings + " buildings" +
        (errored ? ", " + errored + " on stubs" : ""),
      errored
        ? errored + " tiles fell back to a labelled stub, so their grades are stub grades"
        : "every tile analyzed, nothing skipped"
    );
    setRailRow(
      "stage-2",
      "done",
      severe + " severe" + (needsGeo ? ", " + needsGeo + " need geo" : ""),
      "severe means damage class 2 or 3"
    );
    setRailRow(
      "stage-3",
      withheld ? "bad" : "done",
      stored + " stored, " + withheld + " withheld",
      "withheld images were analyzed and ranked, and were never written to the archive"
    );
  }

  const row = ctx.el("withheld-row");
  if (row) row.hidden = !withheld;
  const cnt = ctx.el("withheld-count");
  if (cnt) cnt.textContent = String(withheld);
}

// ---------------------------------------------------------------- upload flow
function addFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const who = ctx.operator();
  for (const f of files) {
    const key = baseName(f.name);
    let card = cards.get(key);
    if (card) {
      card.file = f;
      card.phase = "uploading";
      card.record = null;
      card.error = "";
      card.timedOut = false;
      card.at = Date.now();
    } else {
      card = { key, name: f.name, file: f, phase: "uploading", record: null, at: Date.now() };
      cards.set(key, card);
      mountCard(card);
    }
    card.operator = who;
    paintCard(card);
    sendOne(card);
  }
  paintSummary();
  startPoll();
}

/**
 * One request per file. WHY: a single malformed tile must not fail the batch,
 * and each card needs its own verdict.
 */
async function sendOne(card) {
  const fd = new FormData();
  fd.append("files", card.file, card.name);
  inFlight += 1;
  try {
    const out = await ctx.form("api/upload", fd);
    card.phase = "waiting";
    const mine = matchRecord(listOf(out), card);
    if (mine) applyRecord(card, mine);
    else paintCard(card);
  } catch (err) {
    card.phase = "upload-failed";
    card.error = "upload failed: " + shortErr(err);
    paintCard(card);
    ctx.toast("Upload failed for " + card.name + ". The card kept what it has.", "err");
  } finally {
    inFlight -= 1;
    paintSummary();
    if (inFlight <= 0) {
      // Rank, map and the trust strip all move when a tile lands.
      ctx.emit("data:changed", { source: "upload" });
      refresh();
    }
  }
}

function shortErr(err) {
  const s = err && err.message ? err.message : String(err);
  return s.length > 120 ? s.slice(0, 117) + "..." : s;
}

function matchRecord(recs, card) {
  const stem = stemOf(card.name);
  let loose = null;
  for (const r of recs) {
    if (!r || !r.filename) continue;
    const b = baseName(r.filename);
    if (b === card.key) return r;
    if (!loose && stem && b.includes(stem)) loose = r;
  }
  return loose;
}

function applyRecord(card, rec) {
  card.record = rec;
  card.phase = "settled";
  paintCard(card);
}

function pendingCards() {
  const out = [];
  for (const c of cards.values()) if (c.phase !== "upload-failed" && !c.record) out.push(c);
  return out;
}

function startPoll() {
  if (pollTimer) return;
  pollTimer = setInterval(() => {
    if (pendingCards().length) refresh();
    else stopPoll();
  }, POLL_MS);
}

function stopPoll() {
  clearInterval(pollTimer);
  pollTimer = null;
}

// ---------------------------------------------------------------- public API
async function refresh() {
  if (!ctx) return;
  let tiles = [];
  try {
    tiles = listOf(await ctx.get("api/tiles"));
  } catch (err) {
    return; // Quiet: the trust strip already reports service health.
  }
  paintRail(tiles);
  const pend = pendingCards();
  for (const card of pend) {
    const rec = matchRecord(tiles, card);
    if (rec) applyRecord(card, rec);
    else if (Date.now() - card.at > CARD_TIMEOUT_MS && !card.timedOut) {
      card.timedOut = true;
      paintCard(card);
    }
  }
  if (pend.length) paintSummary();
  if (pendingCards().length) startPoll();
  else stopPoll();
}

function init(rawCtx) {
  ctx = adapt(rawCtx);
  injectStyle();
  host = ctx.el("upload-cards") || ctx.el("upload-panel");

  if (ctx.hasBus) {
    // The shell owns the button and the hidden input and hands us the files.
    // WHY: two listeners on one change event would POST every file twice.
    ctx.on("upload:files", (d) => addFiles(d && d.files));
    ctx.on("data:changed", (d) => {
      if (d && d.source === "upload") return;
      refresh();
    });
  } else {
    // dataset guards: a second init must not POST every picked file twice.
    const input = ctx.el("upload-input");
    const btn = ctx.el("btn-upload");
    if (input && !input.dataset.upWired) {
      input.dataset.upWired = "1";
      input.addEventListener("change", () => {
        addFiles(input.files);
        input.value = "";
      });
      if (btn && !btn.dataset.upWired) {
        btn.dataset.upWired = "1";
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          input.click();
        });
      }
    }
  }

  if (host && !host.dataset.upWired) {
    host.dataset.upWired = "1";
    const hold = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };
    ["dragenter", "dragover"].forEach((ev) =>
      host.addEventListener(ev, (e) => {
        hold(e);
        host.classList.add("up-hot");
      })
    );
    ["dragleave", "drop"].forEach((ev) =>
      host.addEventListener(ev, (e) => {
        hold(e);
        host.classList.remove("up-hot");
      })
    );
    host.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
    });
  }

  refresh();
}

export { init, refresh, addFiles };
