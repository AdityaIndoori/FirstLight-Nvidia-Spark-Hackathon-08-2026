/**
 * FIRST LIGHT - agency dispatch panel (plan deliverable C4).
 *
 * WHY this shape: an Operations Section Chief acts on assignments grouped by
 * responding agency, never on one flat list. The AI drafts, a named human
 * disposes, and every edit carries that human's name into the append-only log.
 * Availability is operator-entered, so it never silently re-drafts the plan:
 * Save records the numbers, then an explicit Re-draft asks the model again.
 *
 * API fields read (and nothing else):
 *   GET  api/plan -> {agencies: [{agency, units_required, units_available,
 *                     steps: [{n, footprint_id, label, centroid, task, units}]}],
 *                     drafted_by}
 *   GET  api/route?footprint_id=&agency= -> {ok, geometry, steps: [{text, dist_m}],
 *                     distance_m, eta_min, crosses_blockage,
 *                     blocked_roads_avoided, warning}   printed turn-by-turn
 *   GET  api/roads -> GeoJSON FeatureCollection; feature.properties.name and
 *                     feature.geometry close a road by name AND geometry
 *   POST api/plan/edit    {agency, op, step_n, payload, operator}
 *   POST api/availability {agency, units_available, operator}
 *   POST api/replan       {operator}
 *   POST api/roadblock    {road_name, geometry, blocked, operator}
 *
 * Host elements, all optional (init never throws when they are absent):
 *   #panel-dispatch  the plan itself
 *   #avail-list      static availability rows, inputs are input.uin[data-agency]
 *   #btn-avail-save #btn-redraft #avail-note   the Save then Re-draft flow
 *   #btn-block-road  map control
 * Map (all guarded, a missing map module only changes the message):
 *   ctx.mapModule.captureLine({hint}) to draw a closed road, .showRoute(geometry,
 *   {agency, label}) for a resolved route, .flyTo(centroid) when there is no route.
 * Bus: listens "navigate" {footprint_id, label, centroid}, "plan:changed",
 *      "counts" {plan}; emits "plan:changed" and "data:changed" after edits.
 */

const AGENCIES = ["fire", "ems", "police", "public_works"];
const AGENCY_LABEL = { fire: "Fire", ems: "EMS", police: "Police", public_works: "Public Works" };
const AGENCY_COLOR = {
  fire: "var(--fire,#ff6b4a)",
  ems: "var(--ems,#4cc2ff)",
  police: "var(--police,#b08cff)",
  public_works: "var(--works,#ffd24c)",
};
const AGENCY_TINT = {
  fire: "#2a1510",
  ems: "#0f2230",
  police: "#1d1530",
  public_works: "#2c2410",
};
const STYLE_ID = "fl-style-dispatch";

const CSS = `
.dp-bar { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
.dp-bar .t { font-size:11px; letter-spacing:1.5px; color:var(--dim,#8899aa); text-transform:uppercase; flex:1; }
.dp-drafted { font-size:11px; color:var(--dim,#8899aa); margin-bottom:8px; }
.dp-agency { margin-bottom:12px; }
.dp-head { display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:6px 6px 0 0;
  font-weight:700; font-size:13px; }
.dp-units { margin-left:auto; font-size:12px; font-weight:400; display:flex; align-items:center; gap:6px; }
.dp-units b { font-weight:700; }
.dp-over { color:var(--red,#ff5c5c); font-weight:700; }
.dp-order { border:1px solid var(--red,#ff5c5c); color:var(--red,#ff5c5c); border-radius:4px;
  font-size:11px; padding:1px 7px; white-space:nowrap; }
.dp-printa { border:1px solid currentColor; background:transparent; color:inherit; border-radius:4px;
  font-size:11px; padding:2px 8px; cursor:pointer; font-family:inherit; opacity:.85; }
.dp-printa:hover { opacity:1; }
.dp-step { display:flex; align-items:center; gap:7px; padding:6px 8px;
  border:1px solid var(--line,#1f2733); border-top:0; font-size:13px; background:var(--panel,#11161f); }
.dp-num { width:22px; height:22px; border-radius:50%; color:var(--bg,#0a0d12);
  font:700 12px/22px "Segoe UI",system-ui,sans-serif; text-align:center; flex:none; }
/* The numeral is a filled circle, so the shared .tip underline does not belong. */
.dp-num.tip { border-bottom:0; }
.dp-num.tip-open { overflow:visible; }
.dp-u { color:var(--dim,#8899aa); font-size:12px; white-space:nowrap; cursor:pointer;
  border-bottom:1px dotted var(--line,#1f2733); }
.dp-sel { background:var(--sunk,#060a0f); color:var(--ink,#dde6ee); border:1px solid var(--line,#1f2733);
  border-radius:4px; padding:2px 4px; font:11.5px inherit; max-width:100px; }
.dp-ctl { display:flex; gap:2px; }
.dp-ctl button { background:transparent; border:1px solid var(--line,#1f2733); color:var(--dim,#8899aa);
  border-radius:3px; width:20px; height:20px; font-size:11px; line-height:1; cursor:pointer; padding:0; }
.dp-ctl button:hover { border-color:var(--blue,#4cc2ff); color:var(--blue,#4cc2ff); }
.dp-ctl button.del:hover { border-color:var(--red,#ff5c5c); color:var(--red,#ff5c5c); }
.dp-add { width:100%; background:transparent; border:1px dashed var(--line,#1f2733);
  color:var(--dim,#8899aa); border-radius:0 0 6px 6px; border-top:0; padding:5px; font:12px inherit;
  cursor:pointer; }
.dp-add:hover { color:var(--green,#76b900); border-color:var(--green,#76b900); }
.dp-route { border:1px solid var(--line,#1f2733); border-top:0; background:var(--panel2,#0d1219);
  padding:6px 9px; font-size:12px; color:var(--dim,#8899aa); }
.dp-route ol { margin:4px 0 0 16px; padding:0; color:var(--ink,#dde6ee); }
.dp-route li { margin:1px 0; }
.dp-route .warn { color:var(--amber,#ffb84c); }
.dp-newform { display:flex; gap:6px; padding:6px 8px; border:1px solid var(--green,#76b900); border-top:0; }
.dp-newform input { flex:1; min-width:0; background:var(--sunk,#060a0f); border:1px solid var(--line,#1f2733);
  border-radius:4px; color:var(--ink,#dde6ee); padding:3px 7px; font:12px inherit; }
.dp-newform input.u { flex:none; width:44px; text-align:center; }
.dp-need { color:var(--dim,#8899aa); font-size:11px; margin-left:auto; margin-right:8px; }
.dp-need.over { color:var(--red,#ff5c5c); }

#dp-print { display:none; }
@media print {
  body > *:not(#dp-print) { display:none !important; }
  #dp-print { display:block !important; background:#fff; color:#000;
    font:12pt/1.35 "Segoe UI",system-ui,sans-serif; }
  #dp-print .pk { page-break-after:always; }
  #dp-print .pk:last-child { page-break-after:auto; }
  #dp-print h1 { font-size:20pt; margin:0 0 2mm; letter-spacing:1pt; }
  #dp-print h2 { font-size:15pt; margin:4mm 0 2mm; border-bottom:2pt solid #000; }
  #dp-print .draft { border:2pt solid #000; padding:2mm 3mm; font-weight:700; margin-bottom:3mm; }
  #dp-print .meta { font-size:10pt; margin-bottom:3mm; }
  #dp-print ol.steps { margin:0; padding-left:8mm; }
  #dp-print ol.steps > li { margin-bottom:3mm; page-break-inside:avoid; }
  #dp-print .lab { font-size:13pt; font-weight:700; }
  #dp-print .sub { font-size:10.5pt; }
  #dp-print ol.turns { margin:1mm 0 0 6mm; padding:0; font-size:10.5pt; }
  #dp-print .sig { margin-top:8mm; border-top:1pt solid #000; padding-top:2mm; font-size:10pt; }
  #dp-print table { border-collapse:collapse; font-size:10.5pt; }
  #dp-print td, #dp-print th { border:1pt solid #000; padding:1mm 2mm; text-align:left; }
}
`;

// ---------------------------------------------------------------- state
let ctx = null;
let hostPanel = null;

/** Last plan from the server, mutated in place for instant renumbering. */
let plan = { agencies: [], drafted_by: "" };
let planLoaded = false;
let editsInFlight = 0;
let addingTo = null;
/** Route drawers the operator opened, keyed agency + "|" + step index. */
const openRoutes = new Map();
/** Operator-entered availability, dirty until Save. */
const availDraft = new Map();
let availDirty = false;
let redraftPending = false;

function adapt(raw) {
  const c = raw || {};
  const api = c.api || {};
  return {
    get: typeof api.get === "function" ? api.get.bind(api) : plainGet,
    post: typeof api.post === "function" ? api.post.bind(api) : plainPost,
    toast: typeof c.toast === "function" ? c.toast.bind(c) : quietToast,
    operator: typeof c.operator === "function" ? c.operator.bind(c) : readOpName,
    requireOperator:
      typeof c.requireOperator === "function" ? c.requireOperator.bind(c) : fallbackRequireOperator,
    on: c.bus && typeof c.bus.on === "function" ? c.bus.on.bind(c.bus) : () => {},
    emit: c.bus && typeof c.bus.emit === "function" ? c.bus.emit.bind(c.bus) : () => 0,
    showTab: typeof c.showTab === "function" ? c.showTab.bind(c) : () => {},
    fmt: c.fmt || {},
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

function readOpName() {
  const el = document.getElementById("opname");
  return el && el.value ? el.value.trim() : "";
}

/** Refuse the edit when the name field is empty: the log needs an owner. */
function fallbackRequireOperator() {
  const who = readOpName();
  if (who) return who;
  const el = document.getElementById("opname");
  if (el) el.focus();
  ctx.toast("Enter your name in the top bar first. Every edit is logged under it.", "warn");
  return null;
}

function quietToast(msg) {
  if (typeof console !== "undefined") console.info("[dispatch] " + msg);
}

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement("style");
  s.id = STYLE_ID;
  s.textContent = CSS;
  document.head.appendChild(s);
}

function agencyLabel(a) {
  if (typeof ctx.fmt.agencyLabel === "function") {
    const s = ctx.fmt.agencyLabel(a);
    if (s) return s;
  }
  return AGENCY_LABEL[a] || String(a || "unassigned");
}

function agencyColor(a) {
  if (typeof ctx.fmt.agencyColor === "function") {
    const s = ctx.fmt.agencyColor(a);
    if (s) return s;
  }
  return AGENCY_COLOR[a] || "var(--dim,#8899aa)";
}

function nowClock() {
  const d = new Date();
  return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
}

function num(v, dec) {
  const n = Number(v);
  if (!isFinite(n)) return "0";
  return dec ? n.toFixed(dec) : String(Math.round(n));
}

function shortErr(err) {
  const s = err && err.message ? err.message : String(err);
  return s.length > 110 ? s.slice(0, 107) + "..." : s;
}

/**
 * Dialogs go through the window object, never the bare global. WHY: a headless
 * or embedded host may not expose them, and a missing confirm must not turn a
 * remove click into an uncaught error.
 */
function ask(question) {
  return typeof window.confirm === "function" ? window.confirm(question) : true;
}

function askText(question) {
  return typeof window.prompt === "function" ? window.prompt(question) : null;
}

// ---------------------------------------------------------------- plan model
function normalizePlan(payload) {
  const src = (payload && payload.agencies) || [];
  const agencies = [];
  for (const a of src) {
    if (!a) continue;
    const steps = (a.steps || []).map((s, i) => ({
      n: Number(s && s.n) || i + 1,
      footprint_id: (s && s.footprint_id) || "",
      label: (s && s.label) || "",
      centroid: (s && s.centroid) || null,
      task: (s && s.task) || "",
      units: Number(s && s.units) || 0,
    }));
    agencies.push({
      agency: a.agency || "unassigned",
      units_required: Number(a.units_required) || 0,
      units_available: Number(a.units_available) || 0,
      steps,
    });
  }
  agencies.sort((x, y) => AGENCIES.indexOf(x.agency) - AGENCIES.indexOf(y.agency));
  return { agencies, drafted_by: (payload && payload.drafted_by) || "" };
}

function groupOf(agency) {
  return plan.agencies.find((g) => g.agency === agency) || null;
}

/** Renumber in place so the numerals never lie between a click and a response. */
function renumber(group) {
  group.steps.forEach((s, i) => {
    s.n = i + 1;
  });
}

/**
 * units_required is the model's ask; per-step units are its breakdown. Prefer
 * the breakdown once an operator has edited steps so the header cannot drift.
 */
function unitTotals(group) {
  let asked = 0;
  for (const s of group.steps) asked += Number(s.units) || 0;
  return asked || group.units_required;
}

function availableFor(agency) {
  if (availDraft.has(agency)) return Number(availDraft.get(agency)) || 0;
  const g = groupOf(agency);
  return g ? Number(g.units_available) || 0 : 0;
}

// ---------------------------------------------------------------- edit posting
/**
 * Every edit posts with the operator name. WHY optimistic: the numerals must
 * renumber instantly, so the model moves first and the server confirms after.
 */
async function postEdit(agency, op, stepN, payload) {
  const who = ctx.requireOperator();
  if (!who) return false;
  editsInFlight += 1;
  let ok = false;
  try {
    await ctx.post("api/plan/edit", {
      agency,
      op,
      step_n: stepN,
      payload: payload || {},
      operator: who,
    });
    ctx.emit("plan:changed", { source: "dispatch", op, agency });
    ok = true;
  } catch (err) {
    ctx.toast("Edit not logged: " + shortErr(err) + ". Reloading the plan.", "err");
    planLoaded = false;
  } finally {
    editsInFlight -= 1;
    if (editsInFlight <= 0 && !planLoaded) load();
  }
  return ok;
}

// ---------------------------------------------------------------- panel render
function render() {
  if (!hostPanel) return;
  hostPanel.textContent = "";

  const bar = document.createElement("div");
  bar.className = "dp-bar";
  const t = document.createElement("span");
  t.className = "t";
  t.textContent = "Assignments by agency";
  bar.appendChild(t);
  const printAll = document.createElement("button");
  printAll.type = "button";
  printAll.className = "mini on";
  printAll.textContent = "Print ALL";
  printAll.title = "prints a cover sheet plus one numbered packet per agency, with turn-by-turn";
  printAll.addEventListener("click", () => printPackets(AGENCIES));
  bar.appendChild(printAll);
  hostPanel.appendChild(bar);

  const drafted = document.createElement("div");
  drafted.className = "dp-drafted";
  drafted.textContent = planLoaded
    ? "drafted by " + (plan.drafted_by || "unknown drafter") +
      ", disposed by the Operations Section Chief"
    : "loading the plan";
  hostPanel.appendChild(drafted);

  if (!plan.agencies.length) {
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = planLoaded
      ? "no assignments yet, the plan appears once buildings are ranked"
      : "waiting on api/plan";
    hostPanel.appendChild(e);
    return;
  }

  for (const group of plan.agencies) hostPanel.appendChild(renderAgency(group));
}

function renderAgency(group) {
  const wrap = document.createElement("div");
  wrap.className = "dp-agency";
  wrap.dataset.agency = group.agency;

  const color = agencyColor(group.agency);
  const need = unitTotals(group);
  const have = availableFor(group.agency);
  const over = need > have;

  const head = document.createElement("div");
  head.className = "dp-head";
  head.style.background = AGENCY_TINT[group.agency] || "#151c27";
  head.style.color = color;
  head.appendChild(document.createTextNode(agencyLabel(group.agency).toUpperCase()));

  // class="tip" plus title: the shell's delegated handler answers touch too.
  const units = document.createElement("span");
  units.className = "dp-units";
  const asked = document.createElement("span");
  asked.className = "tip";
  asked.append(document.createTextNode("needs "), bold(need));
  asked.title = "what the drafted plan is asking this agency for, summed over its steps";
  const held = document.createElement("span");
  held.className = "tip";
  held.append(document.createTextNode(" / have "), bold(have));
  held.title = "units you entered for this operational period, never a dispatch feed";
  units.append(asked, held);
  if (over) {
    const flag = document.createElement("span");
    flag.className = "dp-over";
    flag.textContent = "OVER";
    const order = document.createElement("span");
    order.className = "dp-order tip";
    order.textContent = "ORDER +" + (need - have) + " ICS-213 RR";
    order.title =
      "the ask exceeds what you entered, so the aid package carries an ICS-213 RR mutual-aid request";
    units.append(flag, order);
  }
  const printBtn = document.createElement("button");
  printBtn.type = "button";
  printBtn.className = "dp-printa";
  printBtn.textContent = "Print";
  printBtn.title = "prints this agency packet: numbered steps plus turn-by-turn";
  printBtn.addEventListener("click", () => printPackets([group.agency]));
  units.appendChild(printBtn);
  head.appendChild(units);
  wrap.appendChild(head);

  group.steps.forEach((step, idx) => {
    wrap.appendChild(renderStep(group, step, idx, color));
    const key = stepKey(step);
    if (openRoutes.has(key)) wrap.appendChild(routeBox(openRoutes.get(key)));
  });

  if (addingTo === group.agency) wrap.appendChild(newStepForm(group));

  const add = document.createElement("button");
  add.type = "button";
  add.className = "dp-add";
  const who = ctx.operator();
  add.textContent = "+ add step" + (who ? " (logged as " + who + ")" : "");
  add.addEventListener("click", () => {
    if (!ctx.requireOperator()) return;
    addingTo = addingTo === group.agency ? null : group.agency;
    render();
    const inp = hostPanel.querySelector(".dp-newform input");
    if (inp) inp.focus();
  });
  wrap.appendChild(add);
  return wrap;
}

function bold(v) {
  const b = document.createElement("b");
  b.textContent = String(v);
  return b;
}

function renderStep(group, step, idx, color) {
  const row = document.createElement("div");
  row.className = "dp-step";

  const n = document.createElement("span");
  n.className = "dp-num tip";
  n.style.background = color;
  n.textContent = String(idx + 1);
  n.title =
    "dispatch order for " + agencyLabel(group.agency) + ". Reorder with the arrows and every " +
    "numeral renumbers, logged under your name.";
  row.appendChild(n);

  const task = document.createElement("span");
  task.className = "dp-task";
  const text = step.label
    ? step.label + (step.task ? " - " + step.task : "")
    : step.task || step.footprint_id || "unlabelled step";
  task.textContent = text;
  task.title = text;
  row.appendChild(task);

  const u = document.createElement("span");
  u.className = "dp-u tip";
  u.textContent = step.units + (step.units === 1 ? " unit" : " units");
  u.title = "click to change the unit count for this step";
  u.addEventListener("click", () => editUnits(group, step, u));
  row.appendChild(u);

  const nav = document.createElement("button");
  nav.type = "button";
  nav.className = "mini";
  nav.textContent = "Nav";
  nav.title = "turn-by-turn that avoids blocked roads";
  nav.addEventListener("click", () => toggleRoute(group, step));
  row.appendChild(nav);

  const sel = document.createElement("select");
  sel.className = "dp-sel";
  sel.title = "reassign this step to another agency, logged under your name";
  for (const a of AGENCIES) {
    const o = document.createElement("option");
    o.value = a;
    o.textContent = agencyLabel(a);
    if (a === group.agency) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => reassign(group, idx, sel.value, sel));
  row.appendChild(sel);

  const ctl = document.createElement("span");
  ctl.className = "dp-ctl";
  const up = mkCtl("up", "\u25b2", "move up", () => move(group, idx, -1));
  const dn = mkCtl("dn", "\u25bc", "move down", () => move(group, idx, 1));
  const del = mkCtl("del", "\u2715", "remove this step", () => remove(group, idx));
  ctl.append(up, dn, del);
  row.appendChild(ctl);
  return row;
}

function mkCtl(cls, glyph, title, fn) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = cls;
  b.textContent = glyph;
  b.title = title;
  b.addEventListener("click", fn);
  return b;
}

// ---------------------------------------------------------------- step edits
function move(group, idx, delta) {
  const to = idx + delta;
  if (to < 0 || to >= group.steps.length) return;
  if (!ctx.requireOperator()) return;
  const step = group.steps[idx];
  const fromN = step.n;
  group.steps.splice(idx, 1);
  group.steps.splice(to, 0, step);
  renumber(group);
  render(); // Numerals renumber before the request leaves.
  postEdit(group.agency, "move", fromN, { to_n: to + 1, footprint_id: step.footprint_id });
}

function remove(group, idx) {
  const step = group.steps[idx];
  if (!step) return;
  const who = ctx.requireOperator();
  if (!who) return;
  const label = step.label || step.task || step.footprint_id;
  if (!ask("Remove step " + (idx + 1) + ", " + label + "? Logged as " + who + ".")) return;
  openRoutes.delete(stepKey(step));
  group.steps.splice(idx, 1);
  renumber(group);
  render();
  postEdit(group.agency, "delete", step.n, { footprint_id: step.footprint_id });
}

function reassign(group, idx, toAgency, sel) {
  const step = group.steps[idx];
  if (!step || toAgency === group.agency) return;
  if (!ctx.requireOperator()) {
    sel.value = group.agency;
    return;
  }
  const target = groupOf(toAgency) || addGroup(toAgency);
  const fromN = step.n;
  group.steps.splice(idx, 1);
  renumber(group);
  target.steps.push(step);
  renumber(target);
  render();
  postEdit(group.agency, "reassign", fromN, {
    to_agency: toAgency,
    footprint_id: step.footprint_id,
  });
}

function addGroup(agency) {
  const g = { agency, units_required: 0, units_available: availableFor(agency), steps: [] };
  plan.agencies.push(g);
  plan.agencies.sort((x, y) => AGENCIES.indexOf(x.agency) - AGENCIES.indexOf(y.agency));
  return g;
}

function editUnits(group, step, cell) {
  if (cell.dataset.editing) return;
  if (!ctx.requireOperator()) return;
  cell.dataset.editing = "1";
  const before = step.units;
  const inp = document.createElement("input");
  inp.className = "dp-sel";
  inp.style.width = "42px";
  inp.value = String(before);
  cell.textContent = "";
  cell.appendChild(inp);
  inp.focus();
  inp.select();
  let done = false;
  const commit = (save) => {
    if (done) return;
    done = true;
    const v = Math.max(0, parseInt(inp.value, 10) || 0);
    delete cell.dataset.editing;
    if (!save || v === before) {
      render();
      return;
    }
    step.units = v;
    render();
    postEdit(group.agency, "edit", step.n, { units: v, footprint_id: step.footprint_id });
  };
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit(true);
    if (e.key === "Escape") commit(false);
  });
  inp.addEventListener("blur", () => commit(true));
}

function newStepForm(group) {
  const form = document.createElement("div");
  form.className = "dp-newform";
  const task = document.createElement("input");
  task.placeholder = "new step, for example 4726 42nd Ave SW - structure fire";
  const units = document.createElement("input");
  units.className = "u";
  units.value = "1";
  units.title = "units this step needs";
  const ok = document.createElement("button");
  ok.type = "button";
  ok.className = "mini on";
  ok.textContent = "Add";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "mini";
  cancel.textContent = "Cancel";
  form.append(task, units, ok, cancel);

  const commit = () => {
    const text = task.value.trim();
    if (!text) {
      ctx.toast("Give the step a task before adding it.", "warn");
      return;
    }
    const step = {
      n: group.steps.length + 1,
      footprint_id: "",
      label: "",
      centroid: null,
      task: text,
      units: Math.max(0, parseInt(units.value, 10) || 1),
    };
    group.steps.push(step);
    renumber(group);
    addingTo = null;
    render();
    postEdit(group.agency, "add", step.n, { task: step.task, units: step.units });
  };
  ok.addEventListener("click", commit);
  task.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    if (e.key === "Escape") {
      addingTo = null;
      render();
    }
  });
  cancel.addEventListener("click", () => {
    addingTo = null;
    render();
  });
  return form;
}

// ---------------------------------------------------------------- routes
function routeBox(route) {
  const box = document.createElement("div");
  box.className = "dp-route";
  if (!route) {
    box.textContent = "resolving the route";
    return box;
  }
  if (!route.ok) {
    const w = document.createElement("div");
    w.className = "warn";
    w.textContent =
      "no route: " + (route.warning || "routing not answering") +
      (route.centroid ? ". Head to " + fmtCentroid(route.centroid) + "." : "");
    box.appendChild(w);
    return box;
  }
  const head = document.createElement("div");
  head.textContent =
    num(route.distance_m) + " m, " + num(route.eta_min, 1) + " min" +
    (route.blocked_roads_avoided && route.blocked_roads_avoided.length
      ? ", avoids " + route.blocked_roads_avoided.join(", ")
      : "");
  box.appendChild(head);
  if (route.crosses_blockage) {
    const w = document.createElement("div");
    w.className = "warn";
    w.textContent = "warning: this route crosses a blocked road, no clean path exists";
    box.appendChild(w);
  }
  const ol = document.createElement("ol");
  for (const s of route.steps || []) {
    const li = document.createElement("li");
    li.textContent = s.text + (s.dist_m ? " (" + num(s.dist_m) + " m)" : "");
    ol.appendChild(li);
  }
  box.appendChild(ol);
  return box;
}

function fmtCentroid(c) {
  if (!c || c.length < 2) return "an unknown location";
  return Number(c[1]).toFixed(5) + ", " + Number(c[0]).toFixed(5);
}

/**
 * B4's Dijkstra may not be wired yet, so a missing or negative answer degrades
 * to coordinates plus a stated reason rather than an empty packet.
 */
async function fetchRoute(agency, step) {
  const qs =
    "api/route?footprint_id=" + encodeURIComponent(step.footprint_id || "") +
    "&agency=" + encodeURIComponent(agency || "");
  try {
    const r = await ctx.get(qs);
    if (r && typeof r === "object") return Object.assign({ centroid: step.centroid }, r);
  } catch (err) {
    return {
      ok: false,
      centroid: step.centroid,
      warning: "turn-by-turn unavailable, route service not answering",
    };
  }
  return { ok: false, centroid: step.centroid, warning: "route service returned nothing" };
}

/**
 * Route drawers are keyed by the STEP, not its position, so a reorder or a
 * reassign keeps the drawer attached to the door it describes.
 */
function stepKey(step) {
  return step.footprint_id || "task:" + (step.task || step.label || "");
}

async function toggleRoute(group, step) {
  const key = stepKey(step);
  if (openRoutes.has(key)) {
    openRoutes.delete(key);
    const map = ctx.map;
    if (map && typeof map.showRoute === "function") map.showRoute(null);
    render();
    return;
  }
  openRoutes.set(key, null);
  render();
  const route = await fetchRoute(group.agency, step);
  openRoutes.set(key, route);
  render();
  const map = ctx.map;
  if (route.ok && route.geometry && map && typeof map.showRoute === "function")
    map.showRoute(route.geometry, { agency: group.agency, label: step.label || step.task });
  else if (!route.ok && step.centroid && map && typeof map.flyTo === "function")
    map.flyTo(step.centroid);
}

/**
 * Re-resolve every drawer the operator has open. WHY after a closure: the whole
 * point of blocking a road is that the routes change, so a blanked drawer would
 * hide the one answer the operator just asked for.
 */
async function refreshOpenRoutes() {
  if (!openRoutes.size) return;
  for (const group of plan.agencies) {
    for (const step of group.steps) {
      const key = stepKey(step);
      if (!openRoutes.has(key)) continue;
      openRoutes.set(key, await fetchRoute(group.agency, step));
    }
  }
  render();
}

/** Rank cards emit "navigate"; the route drawer is the dispatch answer to it. */
async function onNavigate(detail) {
  const fid = detail && detail.footprint_id;
  if (!fid) return;
  ctx.showTab("dispatch");
  if (!planLoaded) await load();
  for (const group of plan.agencies) {
    const step = group.steps.find((s) => s.footprint_id === fid);
    if (!step) continue;
    const key = stepKey(step);
    if (openRoutes.has(key)) openRoutes.set(key, await fetchRoute(group.agency, step));
    else await toggleRoute(group, step);
    const drawn = openRoutes.get(key);
    render();
    if (drawn && !drawn.ok)
      ctx.toast(
        "No route to " + (step.label || step.task || fid) + ": " +
          (drawn.warning || "routing unavailable") + ". Coordinates are on the step.",
        "warn"
      );
    return;
  }
  // Not on the plan yet: still resolve a route so Navigate always answers.
  const route = await fetchRoute("", { footprint_id: fid, centroid: detail.centroid });
  const map = ctx.map;
  if (route.ok && route.geometry && map && typeof map.showRoute === "function") {
    map.showRoute(route.geometry, { label: detail.label || fid });
    ctx.toast("Route drawn. This door is not on the agency plan yet.", "info");
  } else {
    ctx.toast(
      "No route for " + (detail.label || fid) + ": " + (route.warning || "routing unavailable"),
      "warn"
    );
  }
}

// ---------------------------------------------------------------- availability
function availInputs() {
  return Array.from(document.querySelectorAll("#avail-list input.uin[data-agency]"));
}

/** Wire the shell's static availability rows once, then only patch values. */
function wireAvail() {
  for (const inp of availInputs()) {
    if (inp.dataset.dpWired) continue;
    inp.dataset.dpWired = "1";
    inp.title = "units you have for this operational period, entered by you";
    inp.addEventListener("input", () => {
      availDraft.set(inp.dataset.agency, Math.max(0, parseInt(inp.value, 10) || 0));
      markDirty();
      paintAvailNeeds();
      render();
    });
  }
  const save = ctx.el("btn-avail-save");
  if (save && !save.dataset.dpWired) {
    save.dataset.dpWired = "1";
    save.title = "records the numbers in the log under your name, it does not re-draft";
    save.addEventListener("click", saveAvail);
  }
  const rd = ctx.el("btn-redraft");
  if (rd && !rd.dataset.dpWired) {
    rd.dataset.dpWired = "1";
    rd.title = "asks the planner to re-draft against the availability you just saved";
    rd.addEventListener("click", doReplan);
  }
}

function paintAvail() {
  wireAvail();
  for (const inp of availInputs()) {
    // Never overwrite the box the operator is typing into.
    if (document.activeElement === inp) continue;
    if (!availDirty || !availDraft.has(inp.dataset.agency))
      inp.value = String(availableFor(inp.dataset.agency));
  }
  paintAvailNeeds();
  const rd = ctx.el("btn-redraft");
  if (rd) rd.classList.toggle("show", redraftPending);
}

/** The over-commitment figure belongs beside the input the operator just typed. */
function paintAvailNeeds() {
  for (const inp of availInputs()) {
    const agency = inp.dataset.agency;
    const row = inp.parentElement;
    if (!row) continue;
    let cell = row.querySelector(".dp-need");
    if (!cell) {
      cell = document.createElement("span");
      cell.className = "dp-need";
      cell.title = "what the plan is asking this agency for";
      row.insertBefore(cell, inp);
    }
    const g = groupOf(agency);
    const need = g ? unitTotals(g) : 0;
    cell.textContent = need ? "needs " + need : "";
    cell.classList.toggle("over", need > availableFor(agency));
  }
}

function markDirty() {
  availDirty = true;
  const save = ctx.el("btn-avail-save");
  if (save) {
    save.classList.add("dirty");
    save.textContent = "Save*";
  }
  const note = ctx.el("avail-note");
  if (note) note.textContent = "unsaved changes";
}

/**
 * Save records the numbers; it never re-drafts. WHY: an availability edit that
 * silently rewrote the plan would move assignments under the operator's hands.
 */
async function saveAvail() {
  const who = ctx.requireOperator();
  if (!who) return;
  const save = ctx.el("btn-avail-save");
  const note = ctx.el("avail-note");
  if (save) save.disabled = true;
  if (note) note.textContent = "saving";

  let failed = 0;
  for (const inp of availInputs()) {
    const agency = inp.dataset.agency;
    const units = Math.max(0, parseInt(inp.value, 10) || 0);
    try {
      await ctx.post("api/availability", { agency, units_available: units, operator: who });
      const g = groupOf(agency);
      if (g) g.units_available = units;
      availDraft.set(agency, units);
    } catch (err) {
      failed += 1;
    }
  }

  availDirty = false;
  if (save) {
    save.disabled = false;
    save.classList.remove("dirty");
    save.textContent = "Save";
  }
  if (failed) {
    if (note) note.textContent = failed + " agencies did not save, try again";
    ctx.toast("Availability partly saved: " + failed + " failed.", "err");
  } else {
    if (note) note.textContent = "saved " + nowClock() + " by " + who + " (logged)";
    redraftPending = true;
    ctx.toast("Availability saved. The plan is unchanged until you re-draft.", "ok");
  }
  paintAvail();
  render();
}

async function doReplan() {
  const who = ctx.requireOperator();
  if (!who) return;
  const rd = ctx.el("btn-redraft");
  const note = ctx.el("avail-note");
  if (rd) {
    rd.disabled = true;
    rd.textContent = "re-drafting";
  }
  try {
    await ctx.post("api/replan", { operator: who });
    redraftPending = false;
    if (note) note.textContent = "re-drafted " + nowClock() + " for " + who;
    ctx.emit("plan:changed", { source: "replan" });
    ctx.emit("data:changed", { source: "replan" });
    await load();
    await refreshOpenRoutes();
    ctx.toast("Plan re-drafted against your availability.", "ok");
  } catch (err) {
    ctx.toast("Re-draft failed: " + shortErr(err) + ". The saved numbers stand.", "err");
  } finally {
    if (rd) {
      rd.disabled = false;
      rd.textContent = "Re-draft plan";
      rd.classList.toggle("show", redraftPending);
    }
  }
}

// ---------------------------------------------------------------- road block
async function roadGeometry(name) {
  try {
    const fc = await ctx.get("api/roads");
    const want = String(name).toLowerCase();
    for (const f of (fc && fc.features) || []) {
      const p = (f && f.properties) || {};
      const n = String(p.name || p.road_name || "").toLowerCase();
      if (n && (n === want || n.includes(want) || want.includes(n))) return f.geometry || null;
    }
  } catch (err) {
    return null;
  }
  return null;
}

/**
 * Close a road by name AND geometry, because B bans on both.
 *
 * Geometry comes from the map's draw tool when it is available, since the
 * operator can see exactly which segment is out. When the map is not there we
 * fall back to matching the name against the local road table, so a closure
 * still carries a line rather than a bare string.
 */
async function blockRoad() {
  const who = ctx.requireOperator();
  if (!who) return;
  const name = askText("Road name to close, exactly as it reads on the map:");
  if (!name) return;
  const road = name.trim();

  let geometry = null;
  let drawn = false;
  const map = ctx.map;
  if (map && typeof map.captureLine === "function") {
    geometry = await map.captureLine({ hint: "Click along " + road + ", double click to finish" });
    drawn = !!geometry;
  }
  if (!geometry) geometry = await roadGeometry(road);

  try {
    await ctx.post("api/roadblock", { road_name: road, geometry, blocked: true, operator: who });
    ctx.toast(
      geometry
        ? road + " closed by name and " + (drawn ? "the line you drew" : "its line from the road table") +
          ". Routes re-planned."
        : road + " closed by name only: no matching line in the local road table.",
      geometry ? "ok" : "warn"
    );
    ctx.emit("data:changed", { source: "roadblock" });
    ctx.emit("plan:changed", { source: "roadblock" });
    await load();
    await refreshOpenRoutes();
  } catch (err) {
    ctx.toast("Road block failed: " + shortErr(err), "err");
  }
}

// ---------------------------------------------------------------- print
/**
 * The print target lives outside the console so the print stylesheet can hide
 * everything else. Resolved by id on every call rather than cached: a cached
 * node can outlive its document when the shell re-inits the module.
 */
function ensurePrintHost() {
  let node = document.getElementById("dp-print");
  if (!node) {
    node = document.createElement("div");
    node.id = "dp-print";
    document.body.appendChild(node);
  }
  return node;
}

function escapeText(s) {
  const d = document.createElement("div");
  d.textContent = String(s == null ? "" : s);
  return d.innerHTML;
}

function coverSheet(list) {
  const rows = list
    .map((g) => {
      const need = unitTotals(g);
      const have = availableFor(g.agency);
      return (
        "<tr><td>" + escapeText(agencyLabel(g.agency)) + "</td><td>" + g.steps.length +
        "</td><td>" + need + "</td><td>" + have + "</td><td>" +
        (need > have ? "ORDER +" + (need - have) + ", ICS-213 RR" : "within resources") +
        "</td></tr>"
      );
    })
    .join("");
  return (
    '<div class="pk"><div class="draft">DRAFT - requires approval by the Planning Section Chief</div>' +
    "<h1>FIRST LIGHT dispatch packet</h1>" +
    '<div class="meta">Printed ' + escapeText(new Date().toString()) +
    " by " + escapeText(ctx.operator() || "unnamed operator") +
    ". Drafted by " + escapeText(plan.drafted_by || "unknown drafter") +
    ". Units available were entered by the operator for this operational period.</div>" +
    "<h2>Cover sheet</h2><table><tr><th>Agency</th><th>Steps</th><th>Units asked</th>" +
    "<th>Units available</th><th>Resource status</th></tr>" + rows + "</table>" +
    '<div class="sig">Approved by (print name, sign, time): ' +
    "_______________________________________________</div></div>"
  );
}

function packetHTML(group, routes) {
  const need = unitTotals(group);
  const have = availableFor(group.agency);
  const steps = group.steps
    .map((s, i) => {
      const route = routes[i];
      let turns;
      if (route && route.ok && (route.steps || []).length) {
        turns =
          '<ol class="turns">' +
          route.steps
            .map(
              (t) =>
                "<li>" + escapeText(t.text) + (t.dist_m ? " (" + num(t.dist_m) + " m)" : "") + "</li>"
            )
            .join("") +
          "</ol>";
        if (route.blocked_roads_avoided && route.blocked_roads_avoided.length)
          turns +=
            '<div class="sub">avoids ' + escapeText(route.blocked_roads_avoided.join(", ")) + "</div>";
        if (route.crosses_blockage)
          turns += '<div class="sub">WARNING: no clean path, this route crosses a closure</div>';
      } else {
        turns =
          '<div class="sub">turn-by-turn unavailable, ' +
          escapeText((route && route.warning) || "route service not answering") +
          ". Proceed to " + escapeText(fmtCentroid(s.centroid)) + ".</div>";
      }
      return (
        '<li><span class="lab">' + escapeText(s.label || s.task || s.footprint_id) + "</span>" +
        '<div class="sub">' + escapeText(s.task || "") +
        " | units " + s.units + " | " + escapeText(fmtCentroid(s.centroid)) +
        (s.footprint_id ? " | " + escapeText(s.footprint_id) : "") + "</div>" +
        turns + "</li>"
      );
    })
    .join("");
  return (
    '<div class="pk"><div class="draft">DRAFT - requires approval by the Planning Section Chief</div>' +
    "<h1>" + escapeText(agencyLabel(group.agency).toUpperCase()) + " assignment packet</h1>" +
    '<div class="meta">Units asked ' + need + ", units available " + have +
    (need > have ? ". OVER by " + (need - have) + ", ICS-213 RR required." : ".") +
    " Printed " + escapeText(new Date().toString()) +
    " by " + escapeText(ctx.operator() || "unnamed operator") + ".</div>" +
    "<h2>Numbered steps, in dispatch order</h2>" +
    (steps ? '<ol class="steps">' + steps + "</ol>" : "<p>No steps assigned.</p>") +
    '<div class="sig">Operations Section Chief (print name, sign, time): ' +
    "_______________________________________________</div></div>"
  );
}

/**
 * Print one agency packet, or every packet behind a cover sheet.
 *
 * The cover is keyed on how many agencies were ASKED for, not how many survived
 * the steps filter: Print ALL with three of four agencies empty is still a Print
 * ALL, and an Ops Chief needs the empty rows to see the whole picture. Turn-by-
 * turn is fetched per step, because the paper packet is what a crew leaves with.
 */
async function printPackets(agencies) {
  const printable = plan.agencies.filter((g) => agencies.indexOf(g.agency) >= 0 && g.steps.length);
  if (!printable.length) {
    ctx.toast("Nothing to print yet: no steps assigned.", "warn");
    return;
  }
  const host = ensurePrintHost();
  host.innerHTML = "<h1>Preparing packets, resolving turn-by-turn</h1>";
  const parts = [];
  if (agencies.length > 1)
    parts.push(coverSheet(agencies.map((a) => groupOf(a) || emptyGroup(a))));
  for (const group of printable) {
    const routes = [];
    for (const s of group.steps) routes.push(await fetchRoute(group.agency, s));
    parts.push(packetHTML(group, routes));
  }
  host.innerHTML = parts.join("");
  window.print();
}

/** A cover-sheet row for an agency the plan gave nothing to. */
function emptyGroup(agency) {
  return { agency, units_required: 0, units_available: availableFor(agency), steps: [] };
}

// ---------------------------------------------------------------- lifecycle
/** True while the operator is mid-edit, so a refresh cannot pull the rug. */
function busy() {
  if (editsInFlight > 0 || addingTo) return true;
  const a = document.activeElement;
  if (!a) return false;
  if (a.tagName === "INPUT" && a.classList.contains("uin")) return true;
  if (!hostPanel) return false;
  return hostPanel.contains(a) && (a.tagName === "INPUT" || a.tagName === "SELECT");
}

async function load() {
  try {
    plan = normalizePlan(await ctx.get("api/plan"));
    planLoaded = true;
  } catch (err) {
    planLoaded = true;
    return;
  }
  render();
  paintAvail();
}

async function refresh() {
  if (!ctx) return;
  if (busy()) return;
  await load();
}

function init(rawCtx) {
  ctx = adapt(rawCtx);
  injectStyle();
  hostPanel = ctx.el("panel-dispatch");
  ensurePrintHost();

  // dataset guard: a second init must not double-wire the control.
  const blockBtn = ctx.el("btn-block-road");
  if (blockBtn && !blockBtn.dataset.dpWired) {
    blockBtn.dataset.dpWired = "1";
    blockBtn.addEventListener("click", blockRoad);
  }

  ctx.on("navigate", onNavigate);
  ctx.on("plan:changed", (d) => {
    if (d && (d.source === "dispatch" || d.source === "replan" || d.source === "roadblock")) return;
    refresh();
  });
  // map.js already fetches api/plan for its routes, so reuse that read.
  ctx.on("counts", (c) => {
    if (!c || !c.plan || busy()) return;
    if (!c.plan.agencies) return;
    plan = normalizePlan(c.plan);
    planLoaded = true;
    render();
    paintAvail();
  });

  render();
  paintAvail();
  load();
}

export { init, refresh, printPackets, blockRoad };
