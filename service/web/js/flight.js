/**
 * FIRST LIGHT - next-flight panel (plan deliverable C5).
 *
 * WHY display plus export only: government teams fly what their ground station
 * reads, so six export formats earn their keep. Waypoint drag/insert/delete and
 * the draw-a-grid tool are STRETCH per C5, and the seam for them is marked below
 * rather than half-built, because no gate or demo beat scores them.
 *
 * API fields read (and nothing else):
 *   GET api/flight -> GeoJSON FeatureCollection, two features:
 *     properties.role == "survey-area"  Polygon, coordinates [[[lng,lat], ...]]
 *     properties.role == "survey-path"  LineString, coordinates [[lng,lat], ...]
 *       properties.altitude_m_agl, .line_spacing_m, .transects, .est_flight_min
 *       optional: .sector, .reason, .speed_m_s, .battery_count
 *   GET api/flight/export?fmt=  plan | waypoints | kml | csv | gpx | geojson
 *   POST api/replan   {operator}   the Replan flight control
 *
 * Host elements, all optional (init never throws when they are absent):
 *   #panel-flight   the panel body
 *   #btn-replan     map control, re-tasks the next flight
 *   #chip-flight    top-bar chip, gets the sector and the estimated minutes
 * Map: calls ctx.mapModule.showFlight(fc) is left to map.js, which reads the
 *      same endpoint; this panel only flies to the area on request.
 * Bus: listens "data:changed"; emits "data:changed" and "plan:changed" on replan.
 */

const STYLE_ID = "fl-style-flight";

const FORMATS = [
  ["plan", "QGroundControl .plan", "PX4 and ArduPilot ground stations"],
  ["waypoints", "Mission Planner .waypoints", "MAVLink text waypoint list"],
  ["kml", "KML", "DJI Pilot 2 and Google Earth"],
  ["csv", "Litchi CSV", "DJI consumer airframes"],
  ["gpx", "GPX", "handheld GPS units and most trackers"],
  ["geojson", "GeoJSON", "GIS, and what this panel drew"],
];

const CSS = `
.fp-bar { display:flex; align-items:center; gap:8px; margin-bottom:9px; }
.fp-bar .t { font-size:11px; letter-spacing:1.5px; color:var(--dim,#8899aa); text-transform:uppercase; flex:1; }
.fp-card { background:var(--panel,#11161f); border:1px solid var(--line,#1f2733); border-radius:8px;
  padding:10px 12px; margin-bottom:9px; }
.fp-title { font-weight:600; font-size:14px; color:var(--green,#76b900); margin-bottom:6px; }
.fp-kv { display:flex; justify-content:space-between; align-items:baseline; font-size:12.5px;
  padding:2px 0; color:var(--dim,#8899aa); }
.fp-kv b { color:var(--ink,#dde6ee); font-weight:600; }
.fp-kv.big b { font-size:16px; }
.fp-reason { font-size:12px; color:var(--blue,#4cc2ff); margin-top:6px; }
.fp-warn { color:var(--amber,#ffb84c); font-size:12px; margin-top:6px; }
.fp-exp { position:relative; }
.fp-menu { display:none; position:absolute; right:0; top:26px; z-index:40; min-width:240px;
  background:var(--panel2,#0d1219); border:1px solid var(--line,#1f2733); border-radius:6px; padding:5px;
  box-shadow:0 8px 24px rgba(0,0,0,.55); }
.fp-menu.open { display:block; }
.fp-menu button { display:block; width:100%; text-align:left; background:transparent; border:0;
  color:var(--ink,#dde6ee); font:12px inherit; padding:5px 8px; border-radius:4px; cursor:pointer; }
.fp-menu button:hover { background:#151c27; color:var(--green,#76b900); }
.fp-menu button small { display:block; color:var(--dim,#8899aa); font-size:11px; }
.fp-wp { font:11px Consolas,monospace; color:var(--dim,#8899aa); max-height:130px; overflow-y:auto;
  border-top:1px solid var(--line,#1f2733); margin-top:8px; padding-top:6px; }
.fp-wp div { padding:1px 0; }
.fp-stretch { font-size:11px; color:var(--dim,#8899aa); border-top:1px solid var(--line,#1f2733);
  margin-top:9px; padding-top:7px; }
`;

// ---------------------------------------------------------------- state
let ctx = null;
let host = null;
let fc = null;
let loaded = false;
let loadError = "";
let menuOpen = false;
let replanning = false;

function adapt(raw) {
  const c = raw || {};
  const api = c.api || {};
  return {
    get: typeof api.get === "function" ? api.get.bind(api) : plainGet,
    post: typeof api.post === "function" ? api.post.bind(api) : plainPost,
    url: typeof api.url === "function" ? api.url.bind(api) : (p) => p,
    toast: typeof c.toast === "function" ? c.toast.bind(c) : quietToast,
    requireOperator:
      typeof c.requireOperator === "function" ? c.requireOperator.bind(c) : fallbackRequireOperator,
    on: c.bus && typeof c.bus.on === "function" ? c.bus.on.bind(c.bus) : () => {},
    emit: c.bus && typeof c.bus.emit === "function" ? c.bus.emit.bind(c.bus) : () => 0,
    el: typeof c.el === "function" ? c.el.bind(c) : (id) => document.getElementById(id),
    get map() {
      return c.mapModule || (c.modules && c.modules.map) || null;
    },
  };
}

async function plainGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

async function plainPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

function fallbackRequireOperator() {
  const el = document.getElementById("opname");
  const who = el && el.value ? el.value.trim() : "";
  if (who) return who;
  if (el) el.focus();
  ctx.toast("Enter your name in the top bar first. A re-task is logged under it.", "warn");
  return null;
}

function quietToast(msg) {
  if (typeof console !== "undefined") console.info("[flight] " + msg);
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
  if (!isFinite(n)) return "-";
  return dec ? n.toFixed(dec) : String(Math.round(n));
}

function shortErr(err) {
  const s = err && err.message ? err.message : String(err);
  return s.length > 120 ? s.slice(0, 117) + "..." : s;
}

// ---------------------------------------------------------------- geometry
function featureByRole(role) {
  for (const f of (fc && fc.features) || []) {
    if (f && f.properties && f.properties.role === role) return f;
  }
  return null;
}

function pathCoords() {
  const g = (featureByRole("survey-path") || {}).geometry;
  if (!g) return [];
  if (g.type === "LineString") return g.coordinates || [];
  if (g.type === "MultiLineString") {
    const out = [];
    for (const seg of g.coordinates || []) for (const c of seg) out.push(c);
    return out;
  }
  return [];
}

/** Great-circle length in metres, so the printed distance is measured not quoted. */
function pathLengthM(coords) {
  const R = 6371000;
  let total = 0;
  for (let i = 1; i < coords.length; i += 1) {
    const p1 = (coords[i - 1][1] * Math.PI) / 180;
    const p2 = (coords[i][1] * Math.PI) / 180;
    const dp = p2 - p1;
    const dl = ((coords[i][0] - coords[i - 1][0]) * Math.PI) / 180;
    const a =
      Math.sin(dp / 2) * Math.sin(dp / 2) +
      Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
    total += 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
  }
  return total;
}

/** Planar shoelace over a county-scale AOI, good to a few percent. */
function areaKm2() {
  const g = (featureByRole("survey-area") || {}).geometry;
  if (!g || g.type !== "Polygon") return 0;
  const ring = (g.coordinates || [])[0] || [];
  if (ring.length < 4) return 0;
  const latRef = ring.reduce((a, c) => a + c[1], 0) / ring.length;
  const mLat = 111320;
  const mLng = 111320 * Math.cos((latRef * Math.PI) / 180);
  let sum = 0;
  for (let i = 0; i < ring.length - 1; i += 1) {
    sum += ring[i][0] * mLng * (ring[i + 1][1] * mLat) - ring[i + 1][0] * mLng * (ring[i][1] * mLat);
  }
  return Math.abs(sum / 2) / 1e6;
}

function areaCenter() {
  const g = (featureByRole("survey-area") || {}).geometry;
  const ring = g && g.type === "Polygon" ? (g.coordinates || [])[0] || [] : [];
  if (!ring.length) {
    const coords = pathCoords();
    return coords.length ? coords[0] : null;
  }
  let x = 0;
  let y = 0;
  for (const c of ring) {
    x += c[0];
    y += c[1];
  }
  return [x / ring.length, y / ring.length];
}

// ---------------------------------------------------------------- export
/**
 * Export by fetching then saving, so a failure toasts instead of navigating the
 * console away to an error page mid-demo.
 */
async function doExport(fmt, label) {
  try {
    const res = await fetch(ctx.url("api/flight/export?fmt=" + encodeURIComponent(fmt)));
    if (!res.ok) throw new Error("HTTP " + res.status);
    const blob = await res.blob();
    const cd = res.headers.get("content-disposition") || "";
    const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
    const name = m ? decodeURIComponent(m[1]) : "first-light-flight." + fmt;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    ctx.toast(label + " saved as " + name + ".", "ok");
  } catch (err) {
    ctx.toast(label + " export failed: " + shortErr(err), "err");
  }
}

// ---------------------------------------------------------------- render
function render() {
  if (!host) return;
  host.textContent = "";

  const bar = document.createElement("div");
  bar.className = "fp-bar";
  const t = document.createElement("span");
  t.className = "t";
  t.textContent = "Next flight";
  bar.appendChild(t);

  const replan = document.createElement("button");
  replan.type = "button";
  replan.className = "mini";
  replan.textContent = replanning ? "re-tasking" : "Replan flight";
  replan.disabled = replanning;
  replan.title = "asks the planner for a new survey over the stalest cut-off sector";
  replan.addEventListener("click", doReplan);
  bar.appendChild(replan);

  const exp = document.createElement("span");
  exp.className = "fp-exp";
  const expBtn = document.createElement("button");
  expBtn.type = "button";
  expBtn.className = "mini on";
  expBtn.textContent = "Export";
  expBtn.title = "six formats, all written locally, no internet";
  const menu = document.createElement("div");
  menu.className = "fp-menu" + (menuOpen ? " open" : "");
  for (const [fmt, label, why] of FORMATS) {
    const b = document.createElement("button");
    b.type = "button";
    const small = document.createElement("small");
    small.textContent = why;
    b.append(document.createTextNode(label), small);
    b.addEventListener("click", () => {
      menuOpen = false;
      menu.classList.remove("open");
      doExport(fmt, label);
    });
    menu.appendChild(b);
  }
  expBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    menuOpen = !menuOpen;
    menu.classList.toggle("open", menuOpen);
  });
  exp.append(expBtn, menu);
  bar.appendChild(exp);
  host.appendChild(bar);

  if (!loaded) {
    host.appendChild(note("loading the proposed survey"));
    return;
  }

  const path = featureByRole("survey-path");
  const area = featureByRole("survey-area");
  if (!path && !area) {
    host.appendChild(
      note(
        loadError
          ? "no flight plan: " + loadError
          : "no flight tasked yet. Replan asks the planner for the stalest cut-off sector."
      )
    );
    return;
  }

  const p = (path && path.properties) || {};
  const areaProps = (area && area.properties) || {};
  const coords = pathCoords();
  const lenM = pathLengthM(coords);
  const km2 = areaKm2();

  const card = document.createElement("div");
  card.className = "fp-card";
  const title = document.createElement("div");
  title.className = "fp-title";
  const sector = p.sector || areaProps.sector;
  title.textContent = sector ? "Sector " + sector + " survey" : "Proposed survey";
  card.appendChild(title);

  const rows = [
    [
      "estimated flight time",
      p.est_flight_min != null ? num(p.est_flight_min) + " min" : "not estimated",
      true,
      "the planner's estimate for this path at the altitude and speed shown",
    ],
    [
      "transects",
      p.transects != null ? num(p.transects) : String(Math.max(0, coords.length - 1)),
      true,
      "number of survey lines in the serpentine",
    ],
    [
      "altitude",
      p.altitude_m_agl != null ? num(p.altitude_m_agl) + " m AGL" : "not set",
      false,
      "above ground level, the figure a ground station wants",
    ],
    [
      "line spacing",
      p.line_spacing_m != null ? num(p.line_spacing_m) + " m" : "not set",
      false,
      "distance between adjacent transects, which sets the image overlap",
    ],
    [
      "path length",
      lenM ? num(lenM / 1000, 2) + " km" : "unknown",
      false,
      "measured from the returned coordinates, not quoted",
    ],
    ["waypoints", String(coords.length), false, "vertices in the survey path"],
  ];
  if (km2) rows.push(["survey area", num(km2, 2) + " km2", false, "planar approximation of the polygon"]);
  if (p.speed_m_s != null)
    rows.push(["speed", num(p.speed_m_s, 1) + " m/s", false, "planned ground speed"]);
  if (p.battery_count != null)
    rows.push(["batteries", num(p.battery_count), false, "packs the planner expects this survey to burn"]);

  for (const [label, value, big, why] of rows) {
    const kv = document.createElement("div");
    kv.className = "fp-kv" + (big ? " big" : "");
    const l = document.createElement("span");
    l.textContent = label;
    l.className = "tip";
    l.title = why;
    const v = document.createElement("b");
    v.textContent = value;
    kv.append(l, v);
    card.appendChild(kv);
  }

  if (p.reason || areaProps.reason) {
    const r = document.createElement("div");
    r.className = "fp-reason";
    r.textContent = p.reason || areaProps.reason;
    card.appendChild(r);
  }
  if (!coords.length) {
    const w = document.createElement("div");
    w.className = "fp-warn";
    w.textContent = "the area is tasked but the path has no coordinates yet, so exports would be empty";
    card.appendChild(w);
  }

  const center = areaCenter();
  const map = ctx.map;
  if (center && map && typeof map.flyTo === "function") {
    const locate = document.createElement("button");
    locate.type = "button";
    locate.className = "mini";
    locate.style.marginTop = "7px";
    locate.textContent = "Show on map";
    locate.addEventListener("click", () => map.flyTo(center));
    card.appendChild(locate);
  }

  if (coords.length) {
    const wp = document.createElement("div");
    wp.className = "fp-wp";
    wp.title = "the waypoints as they will be written into every export format";
    coords.slice(0, 60).forEach((c, i) => {
      const line = document.createElement("div");
      line.textContent =
        String(i + 1).padStart(2, "0") + "  " + Number(c[1]).toFixed(6) + ", " +
        Number(c[0]).toFixed(6) +
        (p.altitude_m_agl != null ? "  " + num(p.altitude_m_agl) + " m" : "");
      wp.appendChild(line);
    });
    if (coords.length > 60) {
      const more = document.createElement("div");
      more.textContent = "and " + (coords.length - 60) + " more waypoints";
      wp.appendChild(more);
    }
    card.appendChild(wp);
  }

  // STRETCH SEAM, deliberately not built (plan C5, cut-list item 3): in-place
  // waypoint editing (click the line to insert, drag to move, click a point to
  // delete) and the draw-a-grid tool. When they arrive they mutate the local `fc`
  // and hand it to ctx.mapModule.showFlight(fc); exports read the endpoint, so an
  // edited plan needs one POST of the edited FeatureCollection before exporting.
  // Nothing else in this module changes.
  const stretch = document.createElement("div");
  stretch.className = "fp-stretch";
  stretch.textContent =
    "Waypoint editing and the draw-a-grid tool are not built: the planner proposes, the operator " +
    "exports, and Replan re-tasks the whole survey.";
  card.appendChild(stretch);

  host.appendChild(card);
}

function note(text) {
  const d = document.createElement("div");
  d.className = "empty";
  d.textContent = text;
  return d;
}

function paintChip() {
  const chip = ctx.el("chip-flight");
  if (!chip) return;
  const p = (featureByRole("survey-path") || {}).properties || {};
  const areaProps = (featureByRole("survey-area") || {}).properties || {};
  const sector = p.sector || areaProps.sector || "";
  if (!sector && p.est_flight_min == null) {
    chip.textContent = "NEXT FLIGHT: not tasked";
    chip.title = "the planner proposes a survey once tiles have been ranked";
    return;
  }
  chip.textContent =
    "NEXT FLIGHT: " + (sector ? "SECTOR " + String(sector).toUpperCase() : "tasked") +
    (p.est_flight_min != null ? " - " + num(p.est_flight_min) + " min" : "");
  chip.title = "the planner's proposed next survey. Re-task it from the Flight tab.";
}

// ---------------------------------------------------------------- actions
async function doReplan() {
  const who = ctx.requireOperator();
  if (!who) return;
  replanning = true;
  render();
  try {
    await ctx.post("api/replan", { operator: who });
    ctx.emit("data:changed", { source: "replan" });
    ctx.emit("plan:changed", { source: "replan" });
    replanning = false;
    await load();
    ctx.toast("Flight re-tasked and the plan re-drafted.", "ok");
  } catch (err) {
    replanning = false;
    render();
    ctx.toast("Replan failed: " + shortErr(err) + ". The current flight stands.", "err");
  }
}

// ---------------------------------------------------------------- lifecycle
async function load() {
  try {
    const out = await ctx.get("api/flight");
    fc = out && out.features ? out : { type: "FeatureCollection", features: [] };
    loadError = "";
  } catch (err) {
    loadError = shortErr(err);
    if (!fc) fc = { type: "FeatureCollection", features: [] };
  }
  loaded = true;
  render();
  paintChip();
}

async function refresh() {
  if (!ctx) return;
  if (menuOpen || replanning) return; // Never close a menu the operator opened.
  await load();
}

function init(rawCtx) {
  ctx = adapt(rawCtx);
  injectStyle();
  host = ctx.el("panel-flight");

  // dataset guard: a second init must not double-wire the control.
  const btn = ctx.el("btn-replan");
  if (btn && !btn.dataset.fpWired) {
    btn.dataset.fpWired = "1";
    btn.addEventListener("click", doReplan);
  }

  if (!document.body.dataset.fpMenuWired) {
    document.body.dataset.fpMenuWired = "1";
    document.addEventListener("click", (e) => {
      if (!menuOpen || !host) return;
      const inside = e.target && e.target.closest ? e.target.closest(".fp-exp") : null;
      if (inside) return;
      menuOpen = false;
      const m = host.querySelector(".fp-menu");
      if (m) m.classList.remove("open");
    });
  }

  ctx.on("data:changed", (d) => {
    if (d && d.source === "replan") return; // load() already ran for it.
    refresh();
  });

  render();
  load();
}

export { init, refresh, doExport, doReplan };
