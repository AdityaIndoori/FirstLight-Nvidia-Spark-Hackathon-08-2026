/**
 * FIRST LIGHT - searchable image archive panel (plan deliverable C6).
 *
 * WHY this panel exists: it turns a pile of photos into an asset, and it is the
 * visible proof of the privacy claim. Only gate-cleared images are ever indexed,
 * so a withheld tile cannot appear here even when it is topically relevant. The
 * add-image button posts through the SAME ingest door, so the gate runs again on
 * anything an operator tries to add, and a refusal is stated plainly.
 *
 * API fields read (and nothing else):
 *   GET  api/archive/search?q=&limit= -> {items: [ArchiveItem], resolved_by: [...], took_ms}
 *     ArchiveItem.image_id, .thumb_path, .captured_at, .centroid, .needs_geo,
 *                 .caption, .tags, .class_max, .key_evidence, .footprint_ids
 *     resolved_by holds any of "location", "filter", "semantic", in applied order
 *   POST api/upload            multipart "files", the one ingest door
 *     response {items: [TileRecord]}; TileRecord.stored and .withheld_reason
 *     are what the refusal message is built from
 *   POST api/archive/edit      {image_id, caption, tags, key_evidence, operator}
 *
 * Host elements, all optional (init never throws when they are absent):
 *   #panel-archive      the panel body
 *   #tab-archive-count  result count badge on the tab
 * Map: calls ctx.mapModule.showPins(items) when available, [] to clear.
 * Bus: listens "archive:show" {image_ids}, emits "data:changed" after an add.
 */

const STYLE_ID = "fl-style-archive";
const DEFAULT_LIMIT = 60;

const CSS = `
.ar-bar { display:flex; gap:6px; margin-bottom:8px; }
.ar-bar input { flex:1; min-width:0; background:var(--sunk,#060a0f); border:1px solid var(--line,#1f2733);
  border-radius:5px; color:var(--ink,#dde6ee); padding:6px 9px; font:13px inherit; }
.ar-bar input:focus { border-color:var(--blue,#4cc2ff); outline:none; }
.ar-meta { display:flex; align-items:center; flex-wrap:wrap; gap:6px; font-size:11px;
  color:var(--dim,#8899aa); margin-bottom:8px; }
.ar-chip { border:1px solid var(--line,#1f2733); border-radius:3px; padding:1px 6px; }
.ar-chip.on { border-color:var(--green,#76b900); color:var(--green,#76b900); }
.ar-chip.off { opacity:.4; }
.ar-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(110px,1fr)); gap:7px; }
.ar-cell { background:var(--panel,#11161f); border:1px solid var(--line,#1f2733); border-radius:6px;
  overflow:hidden; cursor:pointer; }
.ar-cell.sel { border-color:var(--blue,#4cc2ff); }
.ar-cell.key { border-color:var(--green,#76b900); }
.ar-thumb { width:100%; height:74px; object-fit:cover; display:block; background:var(--sunk,#060a0f); }
.ar-miss { width:100%; height:74px; display:flex; align-items:center; justify-content:center;
  font-size:11px; color:var(--dim,#8899aa); background:var(--sunk,#060a0f); text-align:center; }
.ar-cap { font-size:11px; line-height:1.3; padding:4px 5px; color:var(--ink,#dde6ee);
  max-height:44px; overflow:hidden; }
.ar-cellfoot { display:flex; align-items:center; gap:4px; padding:0 5px 4px; font-size:11px;
  color:var(--dim,#8899aa); }
.ar-cls { border-radius:2px; padding:0 4px; font-weight:700; color:var(--bg,#0a0d12); }
.ar-detail { background:var(--panel,#11161f); border:1px solid var(--blue,#4cc2ff); border-radius:7px;
  padding:9px 10px; margin-bottom:9px; }
.ar-detail .dh { display:flex; align-items:center; gap:7px; margin-bottom:6px; font-size:12px; }
.ar-detail textarea { width:100%; background:var(--sunk,#060a0f); border:1px solid var(--line,#1f2733);
  border-radius:4px; color:var(--ink,#dde6ee); padding:5px 7px; font:12px inherit; resize:vertical;
  min-height:52px; }
.ar-detail input.tags { width:100%; background:var(--sunk,#060a0f); border:1px solid var(--line,#1f2733);
  border-radius:4px; color:var(--ink,#dde6ee); padding:4px 7px; font:12px inherit; }
.ar-lab { font-size:11px; letter-spacing:1px; text-transform:uppercase; color:var(--dim,#8899aa);
  margin:7px 0 3px; }
.ar-drow { display:flex; align-items:center; gap:8px; margin-top:8px; }
.ar-drow label { font-size:11.5px; color:var(--dim,#8899aa); display:flex; align-items:center; gap:5px; }
.ar-guide { color:var(--dim,#8899aa); font-size:12.5px; line-height:1.5; }
.ar-guide h4 { color:var(--ink,#dde6ee); font-size:11.5px; letter-spacing:1px; text-transform:uppercase;
  margin:10px 0 3px; font-weight:600; }
.ar-ex { display:inline-block; background:var(--sunk,#060a0f); border:1px solid var(--line,#1f2733);
  border-radius:3px; padding:1px 6px; margin:3px 4px 0 0; font:11.5px Consolas,monospace;
  color:var(--blue,#4cc2ff); cursor:pointer; }
.ar-ex:hover { border-color:var(--blue,#4cc2ff); }
.ar-refused { border:1px solid #3a2530; background:#180f14; border-radius:6px; padding:8px 10px;
  margin-bottom:9px; font-size:12.5px; }
.ar-refused b { color:var(--red,#ff5c5c); }
.ar-refused ul { margin:5px 0 0 16px; padding:0; }
.ar-refused .x { float:right; cursor:pointer; color:var(--dim,#8899aa); }
.ar-note { font-size:11px; color:var(--amber,#ffb84c); margin-bottom:8px; }
`;

const CLASS_COLOR = { 0: "var(--c0,#7fbf5f)", 1: "var(--c1,#f2e35c)", 2: "var(--c2,#ff9f45)", 3: "var(--c3,#ff5c5c)" };

const RESOLVERS = [
  ["location", "geocoded against the local road and facility tables, then bbox filtered"],
  ["filter", "SQL over the tile and building tables"],
  ["semantic", "cosine over caption embeddings, ranks whatever survived the filters"],
];

const QUERY_KINDS = [
  [
    "Location",
    "geocoded against the local road and facility tables, then bbox filtered",
    ["35th Ave SW", "near Providence Mount", "47.558, -122.377"],
  ],
  [
    "Semantic tag",
    "cosine over the caption embeddings, ranks whatever survived the filters",
    ["buildings on fire", "flooded intersections", "collapsed roof"],
  ],
  [
    "Structured filter",
    "SQL over the tile and building tables, and it chains onto either of the above",
    ["class:3 after:06:00 sector:C", "key:true", "class:2 buildings on fire"],
  ],
];

// ---------------------------------------------------------------- state
let ctx = null;
let host = null;
let query = "";
let results = [];
let resolvedBy = [];
let tookMs = 0;
let searched = false;
let searching = false;
let selected = null;
let refusal = null;
let addNote = "";

function adapt(raw) {
  const c = raw || {};
  const api = c.api || {};
  return {
    get: typeof api.get === "function" ? api.get.bind(api) : plainGet,
    post: typeof api.post === "function" ? api.post.bind(api) : plainPost,
    form: typeof api.form === "function" ? api.form.bind(api) : plainForm,
    url: typeof api.url === "function" ? api.url.bind(api) : (p) => p,
    toast: typeof c.toast === "function" ? c.toast.bind(c) : quietToast,
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

async function plainForm(path, fd) {
  const r = await fetch(path, { method: "POST", body: fd });
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

function fallbackRequireOperator() {
  const el = document.getElementById("opname");
  const who = el && el.value ? el.value.trim() : "";
  if (who) return who;
  if (el) el.focus();
  ctx.toast("Enter your name in the top bar first. Every metadata edit is logged under it.", "warn");
  return null;
}

function quietToast(msg) {
  if (typeof console !== "undefined") console.info("[archive] " + msg);
}

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement("style");
  s.id = STYLE_ID;
  s.textContent = CSS;
  document.head.appendChild(s);
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

function classLabel(n) {
  if (typeof ctx.fmt.classLabel === "function") return ctx.fmt.classLabel(n);
  return ["no damage", "minor damage", "major damage", "destroyed"][Number(n)] || "unknown";
}

/** thumb_path is already root-relative, so only relative paths need the base. */
function thumbURL(item) {
  const p = item && item.thumb_path;
  if (!p) return "";
  return String(p).charAt(0) === "/" ? p : ctx.url(p);
}

/** The full-resolution image behind a row. Same enforcement as the thumbnail:
 *  the archive row must exist, so a withheld image has no reachable copy. */
function imageURL(item) {
  if (!item || !item.image_id) return "";
  return "/api/archive/image/" + encodeURIComponent(item.image_id) + ".jpg";
}

/** Open the actual photograph, because a caption is a claim and the pixels are
 *  the evidence: an operator overriding an AI grade needs to see what it saw.
 *  Closes on the backdrop, the button, or Escape. */
export function openImage(item) {
  if (!item || !item.image_id) return;
  const prior = document.getElementById("fl-lightbox");
  if (prior) prior.remove();

  const back = document.createElement("div");
  back.id = "fl-lightbox";
  back.setAttribute("role", "dialog");
  back.setAttribute("aria-label", "Archive image " + item.image_id);
  back.style.cssText =
    "position:fixed;inset:0;z-index:9000;background:rgba(4,6,10,.92);" +
    "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:24px;";

  const img = document.createElement("img");
  img.src = imageURL(item);
  img.alt = item.caption || item.image_id;
  img.style.cssText =
    "max-width:92vw;max-height:78vh;border:1px solid #1f2733;border-radius:6px;background:#0a0d12;";

  const cap = document.createElement("div");
  cap.style.cssText =
    "max-width:92vw;color:#dde6ee;font-size:13px;text-align:center;line-height:1.5;";
  const capText = document.createElement("div");
  capText.textContent = item.caption || "no caption";
  const meta = document.createElement("div");
  meta.style.cssText = "color:#8899aa;font-size:11.5px;margin-top:4px;";
  const where = item.centroid
    ? Number(item.centroid[1]).toFixed(5) + ", " + Number(item.centroid[0]).toFixed(5)
    : "no location";
  meta.textContent =
    item.image_id +
    " · " +
    clock(item.captured_at) +
    " · " +
    where +
    " · worst grade " +
    classLabel(item.class_max);
  cap.append(capText, meta);

  const close = document.createElement("button");
  close.type = "button";
  close.className = "mini";
  close.textContent = "close";
  close.style.cssText = "position:absolute;top:16px;right:20px;";

  img.addEventListener("error", () => {
    img.remove();
    const fail = document.createElement("div");
    fail.style.cssText = "color:#ff5c5c;font-size:13px;";
    fail.textContent =
      "the stored image is not on disk any more, so only its caption and grade remain";
    back.insertBefore(fail, cap);
  });

  const shut = () => {
    document.removeEventListener("keydown", onKey);
    back.remove();
  };
  function onKey(e) {
    if (e.key === "Escape") shut();
  }
  back.addEventListener("click", (e) => {
    if (e.target === back) shut();
  });
  close.addEventListener("click", shut);
  document.addEventListener("keydown", onKey);

  back.append(close, img, cap);
  document.body.appendChild(back);
}

function shortErr(err) {
  const s = err && err.message ? err.message : String(err);
  return s.length > 120 ? s.slice(0, 117) + "..." : s;
}

// ---------------------------------------------------------------- search
async function runSearch(q) {
  if (q != null) query = q;
  searching = true;
  render();
  const qs = "api/archive/search?q=" + encodeURIComponent(query) + "&limit=" + DEFAULT_LIMIT;
  try {
    const out = await ctx.get(qs);
    results = (out && out.items) || [];
    resolvedBy = (out && out.resolved_by) || [];
    tookMs = Number(out && out.took_ms) || 0;
  } catch (err) {
    results = [];
    resolvedBy = [];
    tookMs = 0;
    ctx.toast("Search failed: " + shortErr(err), "err");
  } finally {
    searching = false;
    searched = true;
  }
  if (selected && !results.some((i) => i.image_id === selected)) selected = null;
  pushPins();
  render();
  paintTabCount();
}

/** A query is also a spatial answer, so results go to the map as pins. */
function pushPins() {
  const map = ctx.map;
  if (!map || typeof map.showPins !== "function") return;
  map.showPins(results.filter((i) => i && i.centroid));
}

function paintTabCount() {
  const badge = ctx.el("tab-archive-count");
  if (badge) badge.textContent = searched ? "(" + results.length + ")" : "";
}

// ---------------------------------------------------------------- add image
/**
 * The hidden picker is resolved by id, so a re-init cannot leave a second one
 * behind that would fire the ingest door twice for one pick.
 */
function ensureAddInput() {
  let input = document.getElementById("ar-add-input");
  if (input) return input;
  input = document.createElement("input");
  input.id = "ar-add-input";
  input.type = "file";
  input.multiple = true;
  input.accept = "image/*,.tif,.tiff";
  input.hidden = true;
  input.addEventListener("change", () => {
    addImages(input.files);
    input.value = "";
  });
  document.body.appendChild(input);
  return input;
}

/**
 * Add goes through the ingest door, never a direct archive write. WHY: there is
 * one storage door and the gate lives in the writer, so an operator adding a
 * person tile by hand gets refused exactly as a card dump would be.
 */
async function addImages(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  if (!ctx.requireOperator()) return;
  refusal = null;
  addNote =
    "sending " + files.length + " image" + (files.length === 1 ? "" : "s") +
    " through the ingest door, the gate runs on each one";
  render();

  let stored = 0;
  const refused = [];
  for (const f of files) {
    const fd = new FormData();
    fd.append("files", f, f.name);
    try {
      const out = await ctx.form("api/upload", fd);
      const recs = (out && out.items) || (Array.isArray(out) ? out : out && out.filename ? [out] : []);
      if (!recs.length) refused.push({ name: f.name, why: "no tile record came back" });
      for (const rec of recs) {
        if (!rec) continue;
        if (rec.stored === false)
          refused.push({ name: rec.filename || f.name, why: rec.withheld_reason });
        else stored += 1;
      }
    } catch (err) {
      refused.push({ name: f.name, why: "upload failed: " + shortErr(err) });
    }
  }

  addNote = "";
  if (refused.length) {
    refusal = {
      stored,
      lines: refused.map((r) => r.name + ": " + (r.why || "person signal, withheld from storage")),
    };
    ctx.toast(
      "The gate ran again and refused " + refused.length + " image" +
        (refused.length === 1 ? "" : "s") + ".",
      "warn"
    );
  } else {
    ctx.toast(stored + " image" + (stored === 1 ? "" : "s") + " stored and searchable.", "ok");
  }
  ctx.emit("data:changed", { source: "archive-add" });
  await runSearch(query);
}

// ---------------------------------------------------------------- metadata edit
async function saveMetadata(item, caption, tagsText, keyEvidence) {
  const who = ctx.requireOperator();
  if (!who) return;
  const tags = tagsText
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  try {
    await ctx.post("api/archive/edit", {
      image_id: item.image_id,
      caption,
      tags,
      key_evidence: !!keyEvidence,
      operator: who,
    });
    item.caption = caption;
    item.tags = tags;
    item.key_evidence = !!keyEvidence;
    ctx.toast("Metadata saved and logged under " + who + ".", "ok");
    render();
  } catch (err) {
    ctx.toast("Metadata not saved: " + shortErr(err), "err");
  }
}

// ---------------------------------------------------------------- render
function render() {
  if (!host) return;
  host.textContent = "";

  const bar = document.createElement("div");
  bar.className = "ar-bar";
  const input = document.createElement("input");
  input.placeholder = "search a location, a tag or a filter";
  input.value = query;
  input.title = "one bar, three resolvers: location and filters narrow, semantic ranks";
  const go = document.createElement("button");
  go.type = "button";
  go.className = "mini on";
  go.textContent = searching ? "searching" : "Search";
  go.disabled = searching;
  const add = document.createElement("button");
  add.type = "button";
  add.className = "mini";
  add.textContent = "Add image";
  add.title = "adds through the same ingest door, so the privacy gate runs on it again";
  bar.append(input, go, add);
  host.appendChild(bar);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runSearch(input.value.trim());
  });
  go.addEventListener("click", () => runSearch(input.value.trim()));
  add.addEventListener("click", () => ensureAddInput().click());

  if (refusal) host.appendChild(refusalBox());
  if (addNote) {
    const n = document.createElement("div");
    n.className = "ar-note";
    n.textContent = addNote;
    host.appendChild(n);
  }

  if (searched) host.appendChild(metaRow());

  if (selected) {
    const item = results.find((i) => i.image_id === selected);
    if (item) host.appendChild(detailCard(item));
  }

  if (!searched) {
    host.appendChild(guide("Three kinds of query, all offline, all local."));
    return;
  }
  if (!results.length) {
    host.appendChild(
      guide(
        query
          ? "Nothing matched " + query + ". Withheld imagery is never indexed, so no query reaches it."
          : "The archive has no stored images yet."
      )
    );
    return;
  }

  const grid = document.createElement("div");
  grid.className = "ar-grid";
  for (const item of results) grid.appendChild(cell(item));
  host.appendChild(grid);
}

function metaRow() {
  const meta = document.createElement("div");
  meta.className = "ar-meta";
  const count = document.createElement("span");
  count.textContent =
    results.length + " image" + (results.length === 1 ? "" : "s") + " in " + tookMs + " ms";
  count.className = "tip";
  count.title = "measured on this box, cosine over the local vectors with no service and no network";
  meta.appendChild(count);
  for (const [name, why] of RESOLVERS) {
    const chip = document.createElement("span");
    const fired = resolvedBy.indexOf(name) >= 0;
    chip.className = "ar-chip tip " + (fired ? "on" : "off");
    chip.textContent = name + (fired ? " fired" : " idle");
    chip.title = why;
    meta.appendChild(chip);
  }
  const pins = document.createElement("span");
  pins.className = "ar-chip";
  pins.textContent = results.filter((i) => i && i.centroid).length + " pinned on the map";
  meta.appendChild(pins);
  return meta;
}

function refusalBox() {
  const box = document.createElement("div");
  box.className = "ar-refused";
  const x = document.createElement("span");
  x.className = "x";
  x.textContent = "x";
  x.title = "dismiss";
  x.addEventListener("click", () => {
    refusal = null;
    render();
  });
  box.appendChild(x);
  const b = document.createElement("b");
  b.textContent = "The gate ran again and refused it. ";
  box.appendChild(b);
  box.appendChild(
    document.createTextNode(
      "Analyzed, never stored: no archive row, no thumbnail, no embedding and not searchable. " +
        "Authorized review only." +
        (refusal.stored ? " " + refusal.stored + " other image(s) were stored." : "")
    )
  );
  const ul = document.createElement("ul");
  for (const line of refusal.lines) {
    const li = document.createElement("li");
    li.textContent = line;
    ul.appendChild(li);
  }
  box.appendChild(ul);
  return box;
}

function cell(item) {
  const el = document.createElement("div");
  el.className =
    "ar-cell" + (item.image_id === selected ? " sel" : "") + (item.key_evidence ? " key" : "");
  const url = thumbURL(item);
  if (url) {
    const img = document.createElement("img");
    img.className = "ar-thumb";
    img.loading = "lazy";
    img.alt = item.caption || item.image_id;
    img.src = url;
    img.addEventListener("error", () => {
      const miss = document.createElement("div");
      miss.className = "ar-miss";
      miss.textContent = "thumbnail missing";
      if (img.parentNode) img.parentNode.replaceChild(miss, img);
    });
    el.appendChild(img);
  } else {
    const miss = document.createElement("div");
    miss.className = "ar-miss";
    miss.textContent = "no thumbnail";
    el.appendChild(miss);
  }

  const cap = document.createElement("div");
  cap.className = "ar-cap";
  cap.textContent = item.caption || "no caption yet";
  el.appendChild(cap);

  const foot = document.createElement("div");
  foot.className = "ar-cellfoot";
  const cls = document.createElement("span");
  cls.className = "ar-cls tip";
  cls.style.background = CLASS_COLOR[Number(item.class_max)] || "#4a5566";
  cls.textContent = "c" + (Number(item.class_max) || 0);
  cls.title = "worst damage in this image: " + classLabel(item.class_max);
  foot.appendChild(cls);
  const t = document.createElement("span");
  t.textContent = clock(item.captured_at);
  foot.appendChild(t);
  if (typeof item.score === "number") {
    // The score, because an ordering asks the operator to trust it while a number
    // can be argued with. Measured on real captions: a topical hit sits around
    // 0.8, a loose association around 0.6, and the embedder's noise band is under
    // 0.5, which is where the relevance floor sits.
    const sc = document.createElement("span");
    sc.className = "tip";
    sc.textContent = "match " + item.score.toFixed(2);
    sc.style.color = item.score >= 0.7 ? "var(--green,#76b900)" : "var(--dim,#8899aa)";
    sc.title =
      "cosine similarity between your query and this caption, 1.00 is identical. " +
      "Results under the relevance floor are not returned at all.";
    foot.appendChild(sc);
  }
  if (item.needs_geo) {
    const g = document.createElement("span");
    g.textContent = "no geo";
    g.title = "no location yet, so it cannot be pinned";
    foot.appendChild(g);
  }
  if (item.key_evidence) {
    const k = document.createElement("span");
    k.textContent = "key";
    k.style.color = "var(--green,#76b900)";
    foot.appendChild(k);
  }
  el.appendChild(foot);

  el.title = (item.tags || []).join(", ") || item.image_id;
  el.addEventListener("click", (e) => {
    // Clicking the picture opens the picture. Clicking anywhere else on the card
    // selects it and shows the editable detail, which is the metadata workflow.
    if (e.target && e.target.tagName === "IMG") {
      openImage(item);
      return;
    }
    selected = selected === item.image_id ? null : item.image_id;
    render();
    const map = ctx.map;
    if (selected && item.centroid && map && typeof map.flyTo === "function") map.flyTo(item.centroid);
  });
  return el;
}

function detailCard(item) {
  const box = document.createElement("div");
  box.className = "ar-detail";

  const dh = document.createElement("div");
  dh.className = "dh";
  const b = document.createElement("b");
  b.textContent = item.image_id;
  const when = document.createElement("span");
  when.style.color = "var(--dim,#8899aa)";
  when.textContent =
    clock(item.captured_at) +
    (item.centroid
      ? ", " + Number(item.centroid[1]).toFixed(5) + ", " + Number(item.centroid[0]).toFixed(5)
      : ", no location");
  const close = document.createElement("button");
  close.type = "button";
  close.className = "mini";
  close.style.marginLeft = "auto";
  close.textContent = "close";
  close.addEventListener("click", () => {
    selected = null;
    render();
  });
  dh.append(b, when, close);
  box.appendChild(dh);

  const capLab = document.createElement("div");
  capLab.className = "ar-lab tip";
  capLab.textContent = "Caption";
  capLab.title = "written by the vision model, constrained to structures, terrain and water";
  box.appendChild(capLab);
  const ta = document.createElement("textarea");
  ta.value = item.caption || "";
  box.appendChild(ta);

  const tagLab = document.createElement("div");
  tagLab.className = "ar-lab tip";
  tagLab.textContent = "Tags, comma separated";
  tagLab.title = "extracted from the caption by the fast model, correct them freely";
  box.appendChild(tagLab);
  const tags = document.createElement("input");
  tags.className = "tags";
  tags.value = (item.tags || []).join(", ");
  box.appendChild(tags);

  const row = document.createElement("div");
  row.className = "ar-drow";
  const keyWrap = document.createElement("label");
  keyWrap.className = "tip";
  keyWrap.title = "marks this image for the aid package";
  const key = document.createElement("input");
  key.type = "checkbox";
  key.checked = !!item.key_evidence;
  keyWrap.append(key, document.createTextNode("key evidence"));
  row.appendChild(keyWrap);

  const save = document.createElement("button");
  save.type = "button";
  save.className = "mini confirm";
  save.textContent = "Save metadata";
  save.addEventListener("click", () => saveMetadata(item, ta.value.trim(), tags.value, key.checked));
  row.appendChild(save);

  const map = ctx.map;
  if (item.centroid && map && typeof map.flyTo === "function") {
    const loc = document.createElement("button");
    loc.type = "button";
    loc.className = "mini";
    loc.textContent = "Locate";
    loc.addEventListener("click", () => map.flyTo(item.centroid));
    row.appendChild(loc);
  }
  box.appendChild(row);

  if ((item.footprint_ids || []).length) {
    const fp = document.createElement("div");
    fp.className = "ar-lab";
    fp.textContent = "buildings in this image: " + item.footprint_ids.length;
    fp.title = item.footprint_ids.join(", ");
    box.appendChild(fp);
  }
  return box;
}

/** The empty state teaches the three query kinds, with runnable examples. */
function guide(lead) {
  const box = document.createElement("div");
  box.className = "ar-guide";
  const p = document.createElement("div");
  p.textContent = lead;
  box.appendChild(p);

  for (const [name, how, examples] of QUERY_KINDS) {
    const h = document.createElement("h4");
    h.textContent = name;
    box.appendChild(h);
    const line = document.createElement("div");
    line.textContent = how;
    box.appendChild(line);
    const ex = document.createElement("div");
    for (const e of examples) {
      const chip = document.createElement("span");
      chip.className = "ar-ex";
      chip.textContent = e;
      chip.title = "click to run this query";
      chip.addEventListener("click", () => runSearch(e));
      ex.appendChild(chip);
    }
    box.appendChild(ex);
  }

  const rule = document.createElement("div");
  rule.style.marginTop = "10px";
  rule.textContent =
    "Filter, then rank: location and filters narrow the corpus, semantic ranks within it. " +
    "Withheld imagery is never indexed, so no query can reach it.";
  box.appendChild(rule);
  return box;
}

// ---------------------------------------------------------------- lifecycle
/** Rank cards emit "archive:show" with the image ids behind a building. */
function onShow(detail) {
  const ids = (detail && detail.image_ids) || [];
  if (!ids.length) {
    ctx.toast("No stored imagery for that building. It may have been withheld.", "warn");
    return;
  }
  runSearch(ids[0]);
}

/** True while the operator is typing or editing, so a refresh cannot clobber it. */
function busy() {
  const a = document.activeElement;
  if (!a || !host || !host.contains(a)) return false;
  return a.tagName === "INPUT" || a.tagName === "TEXTAREA";
}

async function refresh() {
  if (!ctx) return;
  if (!searched) {
    render();
    return;
  }
  if (busy() || searching) return;
  await runSearch(null);
}

function init(rawCtx) {
  ctx = adapt(rawCtx);
  injectStyle();
  host = ctx.el("panel-archive");

  ensureAddInput();

  // The map owns the pins; this panel owns showing an image. Registering the
  // handler keeps the map from needing to know how a lightbox is built, and means
  // clicking a pin and clicking a thumbnail land in exactly the same place.
  const map = ctx.map || (rawCtx && rawCtx.mapModule);
  if (map && typeof map.setPinHandler === "function") {
    map.setPinHandler((item) => openImage(item));
  }

  ctx.on("archive:show", onShow);
  render();
}

export { init, refresh, runSearch, addImages };
