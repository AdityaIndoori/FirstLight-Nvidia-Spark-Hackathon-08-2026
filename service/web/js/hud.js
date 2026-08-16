/**
 * FIRST LIGHT - trust strip, uncertainty distribution, aid package
 * (plan deliverables C8 and C7).
 *
 * WHY the numbers live on one strip: every claim the pitch makes is measured on
 * the box in front of the judge, so tiles, latency, tokens per second, memory,
 * the recovery indicator and the OpenShell verdicts are all visible at once. The
 * uncertainty readout is a DISTRIBUTION, not only per-row bars: if every row
 * sits at the 0.05 floor the bars are decoration and a judge will say so.
 *
 * This module owns the api/status poll and emits it on the bus as "status", so
 * every other panel reads one shared payload. Inbound "status" events (the
 * shell's own fallback poll) are accepted too and never double-paint.
 *
 * API fields read (and nothing else):
 *   GET api/status
 *     tiles_analyzed, tiles_stored, tiles_withheld_from_storage, tiles_error
 *     tile_latency_ms_p50
 *     model_versions {gate, damage, planner, lightning, captioner, embedder}
 *     tokens_per_s   {nano, lightning}
 *     memory_gb, memory_total_gb, gpu_power
 *     last_replan_ms, recovery ("model" | "stub" | null)
 *     doubt_distribution {buckets, contested, total, mean}
 *     datasets [{name, last_refreshed, source}]
 *     openshell {policy, denials, allows, audit: [{ts, actor, action, destination, verdict}]}
 *   GET api/rank?limit=1 -> {doubt_distribution}   fallback only, when status omits it
 *   GET api/export/aid-package -> a file, streamed to a download
 *
 * Host elements, all optional (init never throws when they are absent):
 *   #hud with #strip-tiles #strip-latency #strip-replan #strip-tokens
 *            #strip-memory #strip-audit ; five extra spans are appended once
 *            (#hud-withheld #hud-gauge-wrap #hud-recovery #hud-doubt #hud-datasets)
 *   #btn-aid  the top-bar Download aid package anchor
 */

const STYLE_ID = "fl-style-hud";
const POLL_MS = 5000;
const AID_PATH = "api/export/aid-package";

const AID_CONTENTS =
  "One click builds: the FEMA Preliminary Damage Assessment worksheet, one row per damaged " +
  "structure with coordinates, damage class, confidence and who graded it; an ICS-213 general " +
  "message; an ICS-209 incident summary carrying the agency assignments and unit counts; one " +
  "ICS-213 RR for every agency whose ask exceeds the availability you entered; and the decision " +
  "log as JSON. Every document is stamped DRAFT with a signature line, because a machine does " +
  "not file federal paperwork.";

const CSS = `
#hud .hud-x { display:flex; align-items:center; gap:6px; white-space:nowrap; }
#hud .hud-red { color:var(--red,#ff5c5c); }
#hud .hud-amber { color:var(--amber,#ffb84c); }
#hud .hud-green { color:var(--green,#76b900); }
.hud-gauge { width:56px; height:6px; background:var(--sunk,#060a0f); border:1px solid var(--line,#1f2733);
  border-radius:3px; overflow:hidden; flex:none; }
.hud-gauge > i { display:block; height:100%; background:var(--green,#76b900); }
.hud-gauge > i.hot { background:var(--amber,#ffb84c); }
.hud-gauge > i.full { background:var(--red,#ff5c5c); }
.hud-hist { display:flex; align-items:flex-end; gap:2px; height:14px; flex:none; }
.hud-hist i { display:block; width:6px; background:var(--amber,#ffb84c); min-height:1px; }
.hud-hist i.floor { background:#4a5566; }
.hud-chip { border:1px solid var(--line,#1f2733); border-radius:3px; padding:0 5px; font-size:11px;
  color:var(--dim,#8899aa); }
.hud-chip.stale { border-color:var(--amber,#ffb84c); color:var(--amber,#ffb84c); }
.hud-chip.fresh { border-color:var(--green,#76b900); color:var(--green,#76b900); }
#strip-audit { cursor:pointer; }
/* The strip sits on the bottom edge, so its touch bubbles must open upward and
 * stay inside the viewport rather than off the bottom of the screen. */
#hud .tip.tip-open::after { top:auto; bottom:calc(100% + 6px); }
#hud span:nth-last-child(-n+3) .tip.tip-open::after,
#hud .tip.tip-open:nth-last-child(-n+3)::after { left:auto; right:0; }
.hud-pop { position:fixed; bottom:36px; right:12px; z-index:60; width:min(540px, calc(100vw - 24px));
  background:var(--panel2,#0d1219); border:1px solid var(--line,#1f2733); border-radius:8px;
  padding:10px 12px; box-shadow:0 8px 30px rgba(0,0,0,.55); display:none; }
.hud-pop.open { display:block; }
.hud-pop h5 { margin:0 0 6px; font-size:11px; letter-spacing:1.5px; text-transform:uppercase;
  color:var(--blue,#4cc2ff); }
.hud-pop .policy { font:11.5px Consolas,monospace; color:var(--dim,#8899aa); margin-bottom:8px;
  word-break:break-all; }
.hud-v { display:flex; align-items:center; gap:8px; font-size:12px; padding:4px 7px; border-radius:5px;
  border:1px solid var(--line,#1f2733); background:var(--panel,#11161f); margin-bottom:4px; }
.hud-v .n { margin-left:auto; font-weight:700; }
.hud-v.allow { border-color:var(--green,#76b900); }
.hud-v.deny { border-color:var(--red,#ff5c5c); }
.hud-v.missing { opacity:.5; }
.hud-list { max-height:170px; overflow-y:auto; border-top:1px solid var(--line,#1f2733); padding-top:6px; }
.hud-line { font:11px Consolas,monospace; padding:2px 0; color:var(--green,#76b900);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.hud-line.deny { color:var(--red,#ff5c5c); }
.hud-line .ts { color:var(--dim,#8899aa); }
.hud-close { float:right; cursor:pointer; color:var(--dim,#8899aa); font-size:11px; }
.hud-spin { display:inline-block; width:10px; height:10px; border:2px solid rgba(10,13,18,.3);
  border-top-color:var(--bg,#0a0d12); border-radius:50%; animation:hud-spin .7s linear infinite;
  vertical-align:-1px; margin-right:5px; }
@keyframes hud-spin { to { transform:rotate(360deg); } }
`;

// ---------------------------------------------------------------- state
let ctx = null;
let hud = null;
let aidBtn = null;
let aidBusy = false;
let status = null;
let distribution = null;
let pollTimer = null;
let lastPaintKey = "";
/** Replan samples observed this session, so the p95 we print is one we measured. */
const replanSamples = [];
let lastReplanSeen = -1;

function adapt(raw) {
  const c = raw || {};
  const api = c.api || {};
  return {
    get: typeof api.get === "function" ? api.get.bind(api) : plainGet,
    url: typeof api.url === "function" ? api.url.bind(api) : (p) => p,
    toast: typeof c.toast === "function" ? c.toast.bind(c) : quietToast,
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

function quietToast(msg) {
  if (typeof console !== "undefined") console.info("[hud] " + msg);
}

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement("style");
  s.id = STYLE_ID;
  s.textContent = CSS;
  document.head.appendChild(s);
}

function num(v, dec) {
  const n = Number(v);
  if (!isFinite(n)) return dec ? (0).toFixed(dec) : "0";
  return dec ? n.toFixed(dec) : String(Math.round(n));
}

function shortErr(err) {
  const s = err && err.message ? err.message : String(err);
  return s.length > 120 ? s.slice(0, 117) + "..." : s;
}

function setText(id, text, title) {
  const el = ctx.el(id);
  if (!el) return null;
  el.textContent = text;
  if (title) {
    el.title = title;
    el.classList.add("tip");
  }
  return el;
}

/** Append one extra strip span, once, so a reload does not stack duplicates. */
function extraSpan(id, title) {
  let el = document.getElementById(id);
  if (el) return el;
  if (!hud) return null;
  el = document.createElement("span");
  el.id = id;
  el.className = "hud-x" + (title ? " tip" : "");
  if (title) el.title = title;
  hud.appendChild(el);
  return el;
}

function bold(text, cls) {
  const b = document.createElement("b");
  if (cls) b.className = cls;
  b.textContent = String(text);
  return b;
}

// ---------------------------------------------------------------- strip paint
function paintTiles(s) {
  const analyzed = Number(s.tiles_analyzed) || 0;
  const stored = Number(s.tiles_stored) || 0;
  const withheld = Number(s.tiles_withheld_from_storage) || 0;
  const errored = Number(s.tiles_error) || 0;

  setText(
    "strip-tiles",
    analyzed + " analyzed, " + stored + " stored" + (errored ? ", " + errored + " on stubs" : ""),
    "every tile is analyzed because a person in frame is rescue signal. Stored means it also " +
      "cleared the privacy gate and is searchable."
  );

  const el = extraSpan(
    "hud-withheld",
    "withheld from storage: analyzed and ranked, never stored, indexed, thumbnailed or searchable. " +
      "Reachable only from the authorized review endpoint."
  );
  if (!el) return;
  el.textContent = "";
  el.append(
    document.createTextNode("withheld from storage "),
    bold(withheld, withheld ? "hud-red" : "")
  );
}

function paintLatency(s) {
  setText(
    "strip-latency",
    num(Number(s.tile_latency_ms_p50) / 1000, 1) + " s p50",
    "median end to end per tile, measured on this Spark with every model warm"
  );
}

function p95(samples) {
  const sorted = samples.slice().sort((a, b) => a - b);
  return sorted[Math.max(0, Math.ceil(0.95 * sorted.length) - 1)];
}

function paintReplan(s) {
  const last = Number(s.last_replan_ms) || 0;
  if (last > 0 && last !== lastReplanSeen) {
    lastReplanSeen = last;
    replanSamples.push(last);
  }
  if (!replanSamples.length) {
    setText("strip-replan", "not run yet", "the p95 appears once a replan has been measured here");
    return;
  }
  const v = p95(replanSamples);
  const el = setText(
    "strip-replan",
    num(v / 1000, 1) + " s p95" + (replanSamples.length < 3 ? " (n=" + replanSamples.length + ")" : ""),
    "p95 over " + replanSamples.length + " replans measured this session, last " + last +
      " ms. The budget is under 3 s with every model warm."
  );
  if (el) el.className = v > 3000 ? "hud-amber" : "";
}

function paintTokens(s) {
  const el = ctx.el("strip-tokens");
  if (!el) return;
  const tps = s.tokens_per_s || {};
  const mv = s.model_versions || {};
  el.textContent = "";
  const keys = Object.keys(tps);
  if (!keys.length) {
    el.append(document.createTextNode("models "), bold("not measured yet"));
  } else {
    const ordered = ["nano", "lightning"].filter((k) => tps[k] != null);
    for (const k of keys) if (ordered.indexOf(k) < 0) ordered.push(k);
    ordered.forEach((k, i) => {
      if (i) el.appendChild(document.createTextNode(" - "));
      el.append(document.createTextNode(k + " "), bold(num(tps[k], 1)), document.createTextNode(" tok/s"));
    });
  }
  const names = [];
  for (const role of ["planner", "lightning", "damage", "captioner", "gate", "embedder"])
    if (mv[role]) names.push(role + ": " + mv[role]);
  el.title = names.length
    ? "measured on this Spark. " + names.join("; ")
    : "measured on this Spark once the servers report a version";
}

function paintMemory(s) {
  const used = Number(s.memory_gb) || 0;
  const total = Number(s.memory_total_gb) || 0;
  setText(
    "strip-memory",
    num(used, 0) + " / " + num(total, 0) + " GB",
    "unified memory in use with every model resident, measured on this Spark"
  );

  const el = extraSpan("hud-gauge-wrap", "");
  if (!el) return;
  el.textContent = "";
  const g = document.createElement("span");
  g.className = "hud-gauge";
  const i = document.createElement("i");
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  i.style.width = pct.toFixed(1) + "%";
  if (pct > 92) i.className = "full";
  else if (pct > 78) i.className = "hot";
  g.appendChild(i);
  el.appendChild(g);
  if (s.gpu_power) el.append(document.createTextNode("GPU "), bold(s.gpu_power));
  el.title =
    num(pct, 0) + " percent of " + num(total, 0) + " GB in use" +
    (s.gpu_power ? ", GPU power " + s.gpu_power : "") + ", measured on this Spark";
}

function paintRecovery(s) {
  const el = extraSpan(
    "hud-recovery",
    "when a model returns schema-invalid output the agent re-prompts itself with the validation " +
      "error. Model recovered means the retry landed valid; stub engaged means the deterministic " +
      "labelled fallback answered instead."
  );
  if (!el) return;
  el.textContent = "";
  const r = s.recovery;
  if (r === "model") el.appendChild(bold("model recovered", "hud-green"));
  else if (r === "stub") el.appendChild(bold("stub engaged", "hud-amber"));
  else el.appendChild(bold("no recovery event"));
}

// ------------------------------------------------------- uncertainty readout
/**
 * Normalize whatever doubt_distribution shape arrives into ordered buckets.
 * WHY tolerant: a dict of edge to count and a list of pairs both read naturally
 * on the wire, and the readout must not depend on which one B picked.
 */
function normalizeBuckets(dist) {
  const raw = dist && dist.buckets;
  const out = [];
  if (!raw) return out;
  if (Array.isArray(raw)) {
    raw.forEach((b, i) => {
      if (b == null) return;
      if (typeof b === "number") out.push({ label: String(i), count: b });
      else
        out.push({
          label: String(b.label != null ? b.label : b.bucket != null ? b.bucket : i),
          count: Number(b.count) || 0,
        });
    });
  } else if (typeof raw === "object") {
    for (const k of Object.keys(raw)) out.push({ label: k, count: Number(raw[k]) || 0 });
    out.sort((a, b) => parseFloat(a.label) - parseFloat(b.label) || a.label.localeCompare(b.label));
  }
  return out;
}

function paintDoubt(dist) {
  const el = extraSpan(
    "hud-doubt",
    "AI uncertainty is 1 minus the fast model's agreement with itself over 8 votes, floored at 0.05. " +
      "Contested means the votes scattered, which RAISES priority: uncertainty means send someone to look."
  );
  if (!el) return;
  el.textContent = "";
  if (!dist) {
    el.appendChild(bold("uncertainty not voted yet"));
    return;
  }
  const buckets = normalizeBuckets(dist);
  const total = Number(dist.total) || buckets.reduce((a, b) => a + b.count, 0);
  const contested = Number(dist.contested) || 0;

  if (buckets.length) {
    const max = buckets.reduce((m, b) => Math.max(m, b.count), 1);
    const hist = document.createElement("span");
    hist.className = "hud-hist";
    for (const b of buckets) {
      const bar = document.createElement("i");
      bar.style.height = Math.max(1, Math.round((b.count / max) * 14)) + "px";
      // The floor bucket is grey: a column of identical floors is not a signal.
      if (parseFloat(b.label) <= 0.05) bar.className = "floor";
      bar.title = "doubt " + b.label + ": " + b.count + " buildings";
      hist.appendChild(bar);
    }
    el.appendChild(hist);
  }

  el.append(
    bold(contested + " of " + total, contested ? "hud-amber" : ""),
    document.createTextNode(" buildings contested")
  );
  if (dist.mean != null)
    el.append(document.createTextNode(", mean doubt "), bold(num(dist.mean, 3)));
  if (total && !contested)
    el.append(document.createTextNode(", "), bold("every row at the 0.05 floor", "hud-amber"));
}

// ---------------------------------------------------------------- datasets
function whenText(v) {
  if (v == null || v === "") return "never";
  const n = Number(v);
  if (isFinite(n) && n > 1000000) {
    const d = new Date(n > 1e12 ? n : n * 1000);
    const hrs = (Date.now() - d.getTime()) / 3600000;
    return hrs < 48
      ? String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0")
      : d.toISOString().slice(0, 10);
  }
  return String(v).slice(0, 19).replace("T", " ");
}

function isStale(v) {
  const n = Number(v);
  if (!isFinite(n) || n <= 0) return true;
  const ms = n > 1e12 ? n : n * 1000;
  return Date.now() - ms > 7 * 24 * 3600 * 1000;
}

function paintDatasets(list) {
  const el = extraSpan(
    "hud-datasets",
    "local datasets and when the librarian last refreshed each one, asked for by NAME from the " +
      "GET-only allowlist. Everything works at zero connectivity from the local store."
  );
  if (!el) return;
  el.textContent = "";
  if (!list || !list.length) {
    el.appendChild(bold("datasets: local store only"));
    return;
  }
  el.appendChild(document.createTextNode("data "));
  for (const d of list) {
    const chip = document.createElement("span");
    const stale = isStale(d.last_refreshed);
    chip.className = "hud-chip tip " + (stale ? "stale" : "fresh");
    chip.textContent = d.name + " " + whenText(d.last_refreshed);
    chip.title =
      d.name + (d.source ? ", source " + d.source : "") + ", last refreshed " +
      whenText(d.last_refreshed) +
      (stale ? ". Older than a week, still fully usable offline." : "");
    el.appendChild(chip);
  }
}

// ---------------------------------------------------------------- openshell
/**
 * Three verdict classes on one screen: a localhost allow, an approved-source
 * allow, and a deny. WHY all three: without the two allows, "denied" could just
 * mean the cable is out.
 */
function classify(audit) {
  const groups = { localhost: [], approved: [], deny: [] };
  for (const a of audit || []) {
    if (!a) continue;
    const verdict = String(a.verdict || "").toLowerCase();
    const dest = String(a.destination || "");
    if (verdict.indexOf("den") === 0 || verdict === "block" || verdict === "refused")
      groups.deny.push(a);
    else if (/localhost|127\.0\.0\.1|::1/.test(dest)) groups.localhost.push(a);
    else groups.approved.push(a);
  }
  return groups;
}

function paintAudit(shell) {
  const el = ctx.el("strip-audit");
  if (!el) return;
  const g = classify(shell && shell.audit);
  el.textContent = "";
  el.title = "click for the policy state and the full append-only audit list";

  const chunks = [
    ["localhost allow", g.localhost.length, "hud-green"],
    ["approved-source allow", g.approved.length, "hud-green"],
    ["deny", g.deny.length || Number((shell && shell.denials) || 0), "hud-red"],
  ];
  el.appendChild(document.createTextNode("policy: "));
  chunks.forEach(([label, n, cls], i) => {
    if (i) el.appendChild(document.createTextNode(" - "));
    const span = document.createElement("span");
    span.className = n ? cls : "";
    span.style.opacity = n ? "1" : ".5";
    span.textContent = label + " " + n;
    el.appendChild(span);
  });

  const open = document.getElementById("hud-pop");
  if (open && open.classList.contains("open")) paintPop(shell, g);
}

/**
 * Resolved by id on every call rather than cached, because a cached node can
 * outlive its document when the shell re-inits the module.
 */
function ensurePop() {
  let node = document.getElementById("hud-pop");
  if (node) return node;
  node = document.createElement("div");
  node.id = "hud-pop";
  node.className = "hud-pop";
  node.innerHTML =
    '<span class="hud-close">close</span><h5>OpenShell policy state</h5>' +
    '<div class="policy"></div><div class="hud-verdicts"></div>' +
    "<h5>Audit, append-only</h5><div class=\"hud-list\"></div>";
  node.querySelector(".hud-close").addEventListener("click", () => node.classList.remove("open"));
  document.body.appendChild(node);
  return node;
}

function paintPop(shell, g) {
  const box = ensurePop();
  box.querySelector(".policy").textContent =
    (shell && shell.policy) || "policy state not reported by the runtime";

  const rows = [
    ["allow", "localhost inference", "our own vLLM servers on 8000, 8001 and 8002", g.localhost],
    [
      "allow",
      "approved source, GET only",
      "the five named datasets, asked for by NAME and never by URL",
      g.approved,
    ],
    ["deny", "everything else", "including the agent trying to widen its own policy", g.deny],
  ];
  const host = box.querySelector(".hud-verdicts");
  host.textContent = "";
  for (const [kind, label, why, list] of rows) {
    const row = document.createElement("div");
    row.className = "hud-v " + kind + (list.length ? "" : " missing");
    const t = document.createElement("span");
    t.textContent = label;
    t.className = "tip";
    t.title = why;
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = list.length ? String(list.length) : "not witnessed yet";
    row.append(t, n);
    host.appendChild(row);
  }

  const list = box.querySelector(".hud-list");
  const audit = (shell && shell.audit) || [];
  list.textContent = "";
  if (!audit.length) {
    const empty = document.createElement("div");
    empty.className = "hud-line";
    empty.style.color = "var(--dim,#8899aa)";
    empty.textContent = "no audit records yet";
    list.appendChild(empty);
    return;
  }
  for (const a of audit.slice().reverse()) {
    const line = document.createElement("div");
    const deny = String(a.verdict || "").toLowerCase().indexOf("den") === 0;
    line.className = "hud-line" + (deny ? " deny" : "");
    const ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = whenText(a.ts) + " ";
    line.appendChild(ts);
    line.appendChild(
      document.createTextNode(
        (a.verdict || "?") + " actor=" + (a.actor || "?") + " " + (a.action || "") +
          (a.destination ? " -> " + a.destination : "")
      )
    );
    list.appendChild(line);
  }
}

// ---------------------------------------------------------------- aid package
/**
 * Stream the package to a download with a spinner and a real error path. WHY not
 * the plain anchor: a failed export on a plain link looks like nothing happened,
 * which is the worst possible outcome for the one-click paperwork claim.
 */
async function downloadAid() {
  if (aidBusy) return;
  aidBusy = true;
  const label = aidBtn ? aidBtn.textContent : "";
  if (aidBtn) {
    aidBtn.textContent = "";
    const spin = document.createElement("span");
    spin.className = "hud-spin";
    aidBtn.append(spin, document.createTextNode("building the package"));
    aidBtn.setAttribute("aria-busy", "true");
  }
  try {
    const res = await fetch(ctx.url(AID_PATH));
    if (!res.ok) throw new Error("HTTP " + res.status);
    const blob = await res.blob();
    const cd = res.headers.get("content-disposition") || "";
    const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
    const name = m ? decodeURIComponent(m[1]) : "first-light-aid-package.zip";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    ctx.toast("Aid package downloaded as " + name + ". Every document is stamped DRAFT.", "ok");
  } catch (err) {
    ctx.toast("Aid package failed: " + shortErr(err) + ". Nothing was written.", "err");
  } finally {
    aidBusy = false;
    if (aidBtn) {
      aidBtn.textContent = label;
      aidBtn.removeAttribute("aria-busy");
    }
  }
}

// ---------------------------------------------------------------- lifecycle
function applyStatus(payload) {
  if (!payload || typeof payload !== "object") return;
  status = payload;
  if (!hud) return;
  // De-dupe: the shell may also emit the payload we just fetched.
  const key = JSON.stringify([
    payload.tiles_analyzed,
    payload.tiles_stored,
    payload.tiles_withheld_from_storage,
    payload.tile_latency_ms_p50,
    payload.last_replan_ms,
    payload.recovery,
    payload.memory_gb,
    payload.tokens_per_s,
    payload.doubt_distribution,
    payload.datasets,
    payload.openshell,
  ]);
  if (key === lastPaintKey) return;
  lastPaintKey = key;

  paintTiles(payload);
  paintLatency(payload);
  paintReplan(payload);
  paintTokens(payload);
  paintMemory(payload);
  paintRecovery(payload);
  paintDatasets(payload.datasets);
  paintAudit(payload.openshell);

  const dist = payload.doubt_distribution;
  if (dist && (dist.total || dist.buckets)) {
    distribution = dist;
    paintDoubt(distribution);
  } else if (!distribution) {
    paintDoubt(null);
  }
}

/**
 * One poll for the whole console. WHY here: every panel needs the same numbers,
 * and a second poller would double the load on a box that is already full.
 */
async function poll() {
  try {
    const payload = await ctx.get("api/status");
    applyStatus(payload);
    ctx.emit("status", payload);
  } catch (err) {
    // Quiet: the empty states and the chips already say the service is not up.
  }
}

async function refresh() {
  if (!ctx) return;
  await poll();
  const d = status && status.doubt_distribution;
  if (d && (d.total || d.buckets)) return;
  try {
    const out = await ctx.get("api/rank?limit=1");
    if (out && out.doubt_distribution) {
      distribution = out.doubt_distribution;
      paintDoubt(distribution);
    }
  } catch (err) {
    // Quiet for the same reason.
  }
}

function init(rawCtx) {
  ctx = adapt(rawCtx);
  injectStyle();
  hud = ctx.el("hud");

  if (hud) {
    paintRecovery({});
    paintDoubt(null);
    paintDatasets(null);
    const audit = ctx.el("strip-audit");
    // dataset guards: a second init must not double-wire a click.
    if (audit && !audit.dataset.hudWired) {
      audit.dataset.hudWired = "1";
      audit.addEventListener("click", () => {
        const shell = status && status.openshell;
        const box = ensurePop();
        paintPop(shell, classify(shell && shell.audit));
        box.classList.toggle("open");
      });
    }
  }

  aidBtn = ctx.el("btn-aid");
  if (aidBtn) {
    aidBtn.title = AID_CONTENTS;
    if (!aidBtn.dataset.hudWired) {
      aidBtn.dataset.hudWired = "1";
      aidBtn.addEventListener("click", (e) => {
        e.preventDefault();
        downloadAid();
      });
    }
  }

  // Accept the shell's fallback poll too, so the strip is live either way.
  ctx.on("status", applyStatus);

  clearInterval(pollTimer);
  pollTimer = setInterval(poll, POLL_MS);
  refresh();
}

export { init, refresh, applyStatus, downloadAid };
