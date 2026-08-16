/* FIRST LIGHT map console (deliverable C1).
 *
 * Offline by construction: the style is built inline and points only at raster
 * tiles cached under web/tiles. No style URL, no sprite server, no glyph server,
 * no CDN. Because there is no glyph server, nothing on the map is drawn with a
 * text layer: numerals and the medical cross are canvas images registered with
 * map.addImage, and every name lives in a click popup or in the legend.
 *
 * ---------------------------------------------------------------------------
 * API FIELDS EACH LAYER READS
 * ---------------------------------------------------------------------------
 * GET api/buildings -> FeatureCollection of Polygon
 *   properties.damage_class  int 0-3   picks the fill layer and its colour
 *   properties.confirmed     bool      draws the white operator-confirmed outline
 *   properties.label         string    popup title (street address, never an id)
 *   properties.footprint_id  string    popup id line, emitted on bus "building:click"
 *   properties.graded_by     string    popup provenance line ("nemotron-vl", "operator:<name>")
 *   properties.confidence    float     popup confidence line, optional
 *
 * GET api/roads -> FeatureCollection of LineString
 *   properties.blocked                 true routes the feature to the red dashed layer
 *   properties.road_name or .name      popup title and blocked-road legend note
 *
 * GET api/facilities -> FeatureCollection of Point
 *   geometry.coordinates     [lng,lat] medical cross marker position
 *   properties.name          string    popup title
 *   properties.type          string    nursing_home | dialysis | hospital, popup subtitle
 *   properties.beds or .patients       popup capacity line when present
 *
 * GET api/plan -> {agencies:[{agency, units_required, units_available, route?, steps:[...]}], drafted_by}
 *   agency                   string    picks the route colour (fire|ems|police|public_works)
 *   route.geometry           LineString  drawn SOLID, the real road-following route
 *   route.warning            string    surfaced in the legend footnote when set
 *   steps[].centroid         [lng,lat] numbered circle position, and the approximate
 *                                      dashed connector when route.geometry is absent
 *   steps[].n                int       the numeral inside the circle
 *   steps[].label            string    popup title
 *   steps[].task             string    popup task line
 *   steps[].units            int       popup unit count
 *   steps[].footprint_id     string    emitted on bus "building:click" from the popup
 *
 * GET api/flight -> FeatureCollection
 *   properties.role == "survey-area"   Polygon, green dashed box plus faint fill
 *   properties.role == "survey-path"   LineString, the serpentine path
 *   properties.transects, .est_flight_min, .line_spacing_m, .altitude_m_agl
 *                                      legend footnote text
 *
 * showPins(items) reads ArchiveItem.centroid and .class_max and .caption.
 * showRoute(geometry) takes a LineString for the Navigate highlight.
 * ---------------------------------------------------------------------------
 *
 * PUBLIC API
 *   init(ctx)                     wire the console context, build the map
 *   refresh()                     refetch every layer source and repaint counts
 *   flyTo(centroid, opts)         centroid is [lng, lat]
 *   setCounts(partial)            merge externally known counts into the legend
 *   applyPreset(name)             "triage" | "dispatch" | "all"
 *   showPins(items)               archive search hits, [] clears
 *   showFlight(featureCollection) override the flight layer, null returns to the endpoint
 *   showRoute(geometry, opts)     draw one highlighted route, null clears
 *   captureLine(opts)             operator clicks a line, resolves LineString or null
 *   handle()                      raw MapLibre map, or null
 *   available()                   false when the vendored map engine is missing
 */

// Colours duplicated from css/app.css because MapLibre paint values cannot read
// CSS custom properties. Keep the two in step.
const COLORS = {
  c0: '#7fbf5f', c1: '#f2e35c', c2: '#ff9f45', c3: '#ff5c5c',
  fire: '#ff6b4a', ems: '#4cc2ff', police: '#b08cff', public_works: '#ffd24c',
  green: '#76b900', blue: '#4cc2ff', red: '#ff5c5c', amber: '#ffb84c',
  ink: '#dde6ee', dim: '#8899aa', bg: '#0c1016', road: '#2b3646',
};

const AGENCIES = ['fire', 'ems', 'police', 'public_works'];
const AGENCY_LABEL = { fire: 'Fire', ems: 'EMS', police: 'Police', public_works: 'Public Works' };
const CLASS_LABEL = { 0: 'no damage', 1: 'minor damage', 2: 'major damage', 3: 'destroyed' };
const FACILITY_LABEL = { nursing_home: 'nursing home', dialysis: 'dialysis centre', hospital: 'hospital' };

// Used only until real geometry arrives, so the first view is the demo area of
// operations rather than the whole planet. Mirrors config.AOI [w, s, e, n].
const AOI_FALLBACK = [-122.42, 47.52, -122.36, 47.58];

const TILES_TACTICAL = 'tiles/{z}/{x}/{y}.png';
const TILES_SATELLITE = 'tiles/sat/{z}/{x}/{y}.png';

const EMPTY = { type: 'FeatureCollection', features: [] };

/* The legend IS the filter. One table drives the rows, the toggles, the counts
 * and the three presets, so a new layer cannot appear on the map without
 * appearing in the operator's control of it. */
const LAYER_GROUPS = [
  {
    section: 'Building damage',
    rows: [
      {
        id: 'c3', name: 'destroyed (class 3)', count: 'c3', on: true,
        swatch: { kind: 'sw', color: COLORS.c3 }, layers: ['bld-c3'],
        presets: ['triage', 'dispatch', 'all'],
      },
      {
        id: 'c2', name: 'major damage (class 2)', count: 'c2', on: true,
        swatch: { kind: 'sw', color: COLORS.c2 }, layers: ['bld-c2'],
        presets: ['triage', 'all'],
      },
      {
        id: 'c1', name: 'minor damage (class 1)', count: 'c1', on: true,
        swatch: { kind: 'sw', color: COLORS.c1 }, layers: ['bld-c1'],
        presets: ['triage', 'all'],
      },
      {
        // Off by default on purpose: hundreds of green rectangles are noise.
        id: 'c0', name: 'no damage (class 0)', count: 'c0', on: false,
        swatch: { kind: 'sw', color: COLORS.c0 }, layers: ['bld-c0'],
        presets: ['all'],
        tip: 'Intact buildings are hidden by default because they crowd out the damage. Turn them on to check coverage, which is the one thing they are good for.',
      },
      {
        id: 'confirmed', name: 'operator confirmed', count: 'confirmed', on: true,
        swatch: { kind: 'ring', color: '#ffffff' }, layers: ['bld-confirmed'],
        presets: ['triage', 'dispatch', 'all'],
        tip: 'A white outline means a named operator confirmed or overrode the grade on this building, so it is no longer only the AI talking.',
      },
    ],
  },
  {
    section: 'Assignments by agency',
    rows: AGENCIES.map((a) => ({
      id: 'route-' + a,
      name: AGENCY_LABEL[a] + (a === 'police' ? ' posts' : ' route'),
      count: 'route-' + a,
      on: true,
      swatch: { kind: 'pin', color: COLORS[a] },
      layers: ['route-' + a + '-real', 'route-' + a + '-approx', 'step-' + a],
      presets: ['dispatch', 'all'],
    })),
  },
  {
    section: 'Hazards and tasking',
    rows: [
      {
        id: 'blocked', name: 'blocked roads', count: 'blocked', on: true,
        swatch: { kind: 'dash', color: COLORS.red }, layers: ['roads-blocked'],
        presets: ['triage', 'dispatch', 'all'],
      },
      {
        id: 'facilities', name: 'care facilities', count: 'facilities', on: true,
        swatch: { kind: 'cross' }, layers: ['facilities'],
        presets: ['triage', 'dispatch', 'all'],
        tip: 'Nursing homes, dialysis centres and hospitals. These carry residents who cannot self-evacuate, which is why they get a marker of their own.',
      },
      {
        id: 'flight', name: 'next flight area', count: 'flight', on: true,
        swatch: { kind: 'dash', color: COLORS.green },
        layers: ['flight-area-fill', 'flight-area-line', 'flight-path'],
        presets: ['dispatch', 'all'],
      },
      {
        id: 'roads', name: 'open roads', count: 'roads', on: false,
        swatch: { kind: 'solid', color: COLORS.road }, layers: ['roads-open'],
        presets: ['all'],
      },
    ],
  },
  {
    section: 'Overlays',
    rows: [
      {
        id: 'pins', name: 'archive hits', count: 'pins', on: true, dynamic: true,
        swatch: { kind: 'sw', color: COLORS.blue }, layers: ['pins'],
        presets: ['triage', 'dispatch', 'all'],
      },
      {
        id: 'nav', name: 'route being navigated', count: 'nav', on: true, dynamic: true,
        swatch: { kind: 'solid', color: COLORS.green }, layers: ['nav-line'],
        presets: ['triage', 'dispatch', 'all'],
      },
    ],
  },
];

const PRESET_NAMES = ['triage', 'dispatch', 'all'];

// ------------------------------------------------------------------ module state
let ctx = null;
let map = null;
let mapReady = false;
let engineMissing = false;
let firstFitDone = false;
let flightOverride = null;
let inFlight = false;
let satelliteUsable = false;
let drawing = false;

const rowState = new Map();   // legend row id -> bool
const counts = {};            // legend row id -> number
const popups = [];
const legendNotes = { plan: '', flight: '', tiles: '' };

// ---------------------------------------------------------------------- helpers
function q(id) { return document.getElementById(id); }

function num(v) { const n = Number(v); return isFinite(n) ? n : 0; }

function labelOfClass(c) { return CLASS_LABEL[num(c)] || 'unknown'; }

function agencyColor(a) { return COLORS[a] || COLORS.dim; }

function esc(text) {
  return String(text === undefined || text === null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function toast(text, kind) {
  if (ctx && typeof ctx.toast === 'function') ctx.toast(text, kind);
}

/** Walk any GeoJSON coordinate nesting and widen the running bbox. */
function growBounds(box, coords) {
  if (!Array.isArray(coords)) return;
  if (typeof coords[0] === 'number' && typeof coords[1] === 'number') {
    const [lng, lat] = coords;
    if (!isFinite(lng) || !isFinite(lat)) return;
    if (lng < box[0]) box[0] = lng;
    if (lat < box[1]) box[1] = lat;
    if (lng > box[2]) box[2] = lng;
    if (lat > box[3]) box[3] = lat;
    return;
  }
  for (const child of coords) growBounds(box, child);
}

function bboxOf(collections) {
  const box = [Infinity, Infinity, -Infinity, -Infinity];
  for (const fc of collections) {
    if (!fc || !Array.isArray(fc.features)) continue;
    for (const f of fc.features) {
      if (f && f.geometry) growBounds(box, f.geometry.coordinates);
    }
  }
  return isFinite(box[0]) ? box : null;
}

function pad(box, frac) {
  const dx = Math.max((box[2] - box[0]) * frac, 0.0008);
  const dy = Math.max((box[3] - box[1]) * frac, 0.0008);
  return [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy];
}

// ------------------------------------------------------------- canvas icons
// No glyph server offline, so every numeral and symbol on the map is a small
// canvas image. Registered once per name, drawn at 2x for a crisp numeral.
const registeredIcons = new Set();

function canvasOf(side) {
  const cv = document.createElement('canvas');
  cv.width = side;
  cv.height = side;
  return cv;
}

function addIcon(name, side, draw) {
  if (!map || registeredIcons.has(name)) return name;
  if (map.hasImage && map.hasImage(name)) { registeredIcons.add(name); return name; }
  const cv = canvasOf(side);
  const g = cv.getContext('2d');
  if (!g) return name;
  draw(g, side);
  const data = g.getImageData(0, 0, side, side);
  map.addImage(name, { width: side, height: side, data: new Uint8Array(data.data) }, { pixelRatio: 2 });
  registeredIcons.add(name);
  return name;
}

/** Big legible route numeral in the agency colour, readable across a room. */
function numberIcon(agency, n) {
  const name = 'num-' + agency + '-' + n;
  return addIcon(name, 56, (g, side) => {
    const r = side / 2;
    g.beginPath();
    g.arc(r, r, r - 4, 0, Math.PI * 2);
    g.fillStyle = agencyColor(agency);
    g.fill();
    g.lineWidth = 4;
    g.strokeStyle = 'rgba(10,13,18,0.9)';
    g.stroke();
    g.fillStyle = COLORS.bg;
    g.font = '700 ' + (String(n).length > 2 ? 22 : 30) + 'px "Segoe UI", system-ui, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(String(n), r, r + 1);
  });
}

/** A medical cross, never a blue dot: a care facility must read as medical. */
function crossIcon() {
  return addIcon('medcross', 44, (g, side) => {
    const m = 4;
    g.fillStyle = 'rgba(10,13,18,0.85)';
    g.fillRect(m - 2, m - 2, side - 2 * m + 4, side - 2 * m + 4);
    g.fillStyle = '#ffffff';
    g.fillRect(m, m, side - 2 * m, side - 2 * m);
    g.fillStyle = '#c62828';
    const arm = 7;
    const c = side / 2;
    g.fillRect(c - arm / 2, m + 5, arm, side - 2 * m - 10);
    g.fillRect(m + 5, c - arm / 2, side - 2 * m - 10, arm);
  });
}

// ------------------------------------------------------------------ map style
function buildStyle() {
  return {
    version: 8,
    // No glyphs and no sprite entries: nothing here needs a server to render.
    sources: {
      'base-tactical': {
        type: 'raster', tiles: [TILES_TACTICAL], tileSize: 256, minzoom: 0, maxzoom: 19,
        attribution: 'local tile cache',
      },
      'base-sat': {
        type: 'raster', tiles: [TILES_SATELLITE], tileSize: 256, minzoom: 0, maxzoom: 19,
        attribution: 'local tile cache',
      },
      buildings: { type: 'geojson', data: EMPTY },
      roads: { type: 'geojson', data: EMPTY },
      facilities: { type: 'geojson', data: EMPTY },
      routes: { type: 'geojson', data: EMPTY },
      steps: { type: 'geojson', data: EMPTY },
      flight: { type: 'geojson', data: EMPTY },
      pins: { type: 'geojson', data: EMPTY },
      nav: { type: 'geojson', data: EMPTY },
      draw: { type: 'geojson', data: EMPTY },
    },
    layers: [
      { id: 'bg', type: 'background', paint: { 'background-color': COLORS.bg } },
      { id: 'base-tactical', type: 'raster', source: 'base-tactical', paint: { 'raster-opacity': 0.9 } },
      {
        id: 'base-sat', type: 'raster', source: 'base-sat',
        layout: { visibility: 'none' }, paint: { 'raster-opacity': 1 },
      },

      {
        id: 'roads-open', type: 'line', source: 'roads',
        filter: ['!=', ['get', 'blocked'], true],
        layout: { visibility: 'none', 'line-cap': 'round' },
        paint: { 'line-color': COLORS.road, 'line-width': ['interpolate', ['linear'], ['zoom'], 12, 1.2, 17, 5] },
      },

      ...[0, 1, 2, 3].map((c) => ({
        id: 'bld-c' + c, type: 'fill', source: 'buildings',
        filter: ['==', ['to-number', ['get', 'damage_class']], c],
        paint: {
          'fill-color': COLORS['c' + c],
          'fill-opacity': 0.72,
          'fill-outline-color': 'rgba(10,13,18,0.8)',
        },
      })),
      // Operator-confirmed rows carry a white halo: it has to read at a glance
      // against a red or amber fill, which a hairline does not.
      {
        id: 'bld-confirmed', type: 'line', source: 'buildings',
        filter: ['==', ['get', 'confirmed'], true],
        paint: {
          'line-color': '#ffffff',
          'line-width': ['interpolate', ['linear'], ['zoom'], 12, 2, 15, 3, 18, 4],
        },
      },

      {
        id: 'flight-area-fill', type: 'fill', source: 'flight',
        filter: ['==', ['get', 'role'], 'survey-area'],
        paint: { 'fill-color': COLORS.green, 'fill-opacity': 0.07 },
      },
      {
        id: 'flight-area-line', type: 'line', source: 'flight',
        filter: ['==', ['get', 'role'], 'survey-area'],
        paint: { 'line-color': COLORS.green, 'line-width': 2.4, 'line-dasharray': [5, 3] },
      },
      {
        id: 'flight-path', type: 'line', source: 'flight',
        filter: ['==', ['get', 'role'], 'survey-path'],
        paint: { 'line-color': COLORS.green, 'line-width': 2, 'line-opacity': 0.8 },
      },

      {
        id: 'roads-blocked', type: 'line', source: 'roads',
        filter: ['==', ['get', 'blocked'], true],
        layout: { 'line-cap': 'butt' },
        paint: { 'line-color': COLORS.red, 'line-width': 5, 'line-dasharray': [2.4, 1.6] },
      },

      // Real routed geometry is solid; the straight connector is dashed and
      // labelled approximate, so the map never implies a road-following line
      // it does not have.
      ...AGENCIES.flatMap((a) => ([
        {
          id: 'route-' + a + '-real', type: 'line', source: 'routes',
          filter: ['all', ['==', ['get', 'agency'], a], ['!=', ['get', 'approx'], true]],
          layout: { 'line-cap': 'round', 'line-join': 'round' },
          paint: { 'line-color': agencyColor(a), 'line-width': 4, 'line-opacity': 0.95 },
        },
        {
          id: 'route-' + a + '-approx', type: 'line', source: 'routes',
          filter: ['all', ['==', ['get', 'agency'], a], ['==', ['get', 'approx'], true]],
          paint: {
            'line-color': agencyColor(a), 'line-width': 2.4,
            'line-opacity': 0.7, 'line-dasharray': [3, 3],
          },
        },
      ])),

      {
        id: 'nav-line', type: 'line', source: 'nav',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': COLORS.green, 'line-width': 6, 'line-opacity': 0.9 },
      },

      // Line the operator is drawing for a road closure. Above nav so the
      // vertices stay visible while they click.
      {
        id: 'draw-line', type: 'line', source: 'draw',
        filter: ['==', ['geometry-type'], 'LineString'],
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': COLORS.red, 'line-width': 4, 'line-dasharray': [2, 1.5] },
      },
      {
        id: 'draw-vertex', type: 'circle', source: 'draw',
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
          'circle-radius': 5, 'circle-color': COLORS.red,
          'circle-stroke-color': COLORS.bg, 'circle-stroke-width': 2,
        },
      },

      {
        id: 'pins', type: 'circle', source: 'pins',
        paint: {
          'circle-radius': 6,
          'circle-color': ['get', 'color'],
          'circle-stroke-color': COLORS.bg,
          'circle-stroke-width': 2,
        },
      },

      ...AGENCIES.map((a) => ({
        id: 'step-' + a, type: 'symbol', source: 'steps',
        filter: ['==', ['get', 'agency'], a],
        layout: {
          'icon-image': ['get', 'icon'],
          // Numerals are the one thing a crew reads from across the room, so
          // they grow with zoom instead of staying a pinprick.
          'icon-size': ['interpolate', ['linear'], ['zoom'], 11, 0.8, 14, 1.1, 17, 1.4],
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
        },
      })),

      {
        id: 'facilities', type: 'symbol', source: 'facilities',
        layout: {
          'icon-image': 'medcross',
          'icon-size': ['interpolate', ['linear'], ['zoom'], 11, 0.9, 14, 1.2, 17, 1.5],
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
        },
      },
    ],
  };
}

// ------------------------------------------------------------------- legend UI
function swatchNode(swatch) {
  const s = document.createElement('span');
  if (!swatch) return s;
  if (swatch.kind === 'sw') { s.className = 'sw'; s.style.background = swatch.color; }
  else if (swatch.kind === 'dash') { s.className = 'dash'; s.style.borderColor = swatch.color; }
  else if (swatch.kind === 'solid') { s.className = 'solid'; s.style.borderColor = swatch.color; }
  else if (swatch.kind === 'cross') { s.className = 'cross'; }
  else if (swatch.kind === 'ring') {
    s.className = 'sw';
    s.style.border = '2px solid ' + swatch.color;
    s.style.background = 'transparent';
  } else if (swatch.kind === 'pin') {
    s.className = 'pin';
    s.style.background = swatch.color;
    s.textContent = '1';
  }
  return s;
}

function allRows() {
  const out = [];
  for (const group of LAYER_GROUPS) for (const row of group.rows) out.push(row);
  return out;
}

function buildLegend() {
  const host = q('legend-body');
  if (!host) return;
  host.textContent = '';
  for (const group of LAYER_GROUPS) {
    const title = document.createElement('div');
    title.className = 'lt';
    title.textContent = group.section;
    title.dataset.section = group.section;
    host.appendChild(title);
    for (const row of group.rows) {
      if (!rowState.has(row.id)) rowState.set(row.id, !!row.on);
      const node = document.createElement('div');
      node.className = 'lrow';
      node.dataset.row = row.id;
      node.appendChild(Object.assign(document.createElement('span'), { className: 'cb' }));
      node.appendChild(swatchNode(row.swatch));
      const name = document.createElement('span');
      name.className = 'lname';
      name.textContent = row.name;
      if (row.tip) {
        name.classList.add('tip');
        name.title = row.tip;
      }
      node.appendChild(name);
      const cnt = document.createElement('span');
      cnt.className = 'cnt';
      node.appendChild(cnt);
      node.addEventListener('click', (ev) => {
        // A tap on the tooltip text explains the layer, it does not toggle it.
        if (ev.target && ev.target.classList && ev.target.classList.contains('tip')) return;
        setRow(row.id, !rowState.get(row.id));
      });
      host.appendChild(node);
    }
  }
  paintLegend();
}

function paintLegend() {
  const host = q('legend-body');
  if (!host) return;
  const bySection = new Map();
  for (const group of LAYER_GROUPS) {
    let visible = 0;
    for (const row of group.rows) {
      const node = host.querySelector('.lrow[data-row="' + row.id + '"]');
      if (!node) continue;
      const count = counts[row.count];
      const known = count !== undefined && count !== null;
      const hide = row.dynamic && (!known || count === 0);
      node.hidden = hide;
      if (!hide) visible += 1;
      const on = !!rowState.get(row.id);
      node.classList.toggle('on', on);
      node.classList.toggle('off', !on);
      node.classList.toggle('empty-layer', known && count === 0);
      const cnt = node.querySelector('.cnt');
      if (cnt) cnt.textContent = known ? String(count) : '';
    }
    bySection.set(group.section, visible);
  }
  for (const title of host.querySelectorAll('.lt')) {
    title.hidden = bySection.get(title.dataset.section) === 0;
  }
  paintPresetState();
  paintLegendFoot();
}

function paintPresetState() {
  for (const btn of document.querySelectorAll('.lpreset')) {
    const want = new Set();
    for (const row of allRows()) {
      if ((row.presets || []).indexOf(btn.dataset.preset) >= 0) want.add(row.id);
    }
    let same = true;
    for (const row of allRows()) {
      const node = q('legend-body') && q('legend-body').querySelector('.lrow[data-row="' + row.id + '"]');
      if (node && node.hidden) continue;
      if (!!rowState.get(row.id) !== want.has(row.id)) { same = false; break; }
    }
    btn.classList.toggle('on', same);
  }
}

function paintLegendFoot() {
  const foot = q('legend-foot');
  if (!foot) return;
  foot.textContent = '';
  // The footnotes describe how layers are drawn. With no engine there is
  // nothing drawn, so saying it would only confuse.
  if (engineMissing) return;
  const lines = [];
  if (legendNotes.tiles) lines.push({ text: legendNotes.tiles, warn: true });
  if (legendNotes.plan) lines.push({ text: legendNotes.plan, warn: true });
  if (legendNotes.flight) lines.push({ text: legendNotes.flight, warn: false });
  const off = allRows().filter((r) => {
    const node = q('legend-body') && q('legend-body').querySelector('.lrow[data-row="' + r.id + '"]');
    if (node && node.hidden) return false;
    return !rowState.get(r.id);
  });
  if (off.length) lines.push({ text: off.length + ' layer(s) hidden: ' + off.map((r) => r.name).join(', '), warn: false });
  for (const line of lines) {
    const div = document.createElement('div');
    if (line.warn) div.className = 'approx';
    div.textContent = line.text;
    foot.appendChild(div);
  }
}

function applyRowToMap(row) {
  if (!map || !mapReady) return;
  const visibility = rowState.get(row.id) ? 'visible' : 'none';
  for (const id of row.layers) {
    if (!map.getLayer(id)) continue;
    try { map.setLayoutProperty(id, 'visibility', visibility); }
    catch (err) { /* a layer removed mid-restyle is not worth a broken console */ }
  }
}

function setRow(id, on) {
  const row = allRows().find((r) => r.id === id);
  if (!row) return;
  rowState.set(id, !!on);
  applyRowToMap(row);
  paintLegend();
}

function applyAllRows() {
  for (const row of allRows()) applyRowToMap(row);
}

// ------------------------------------------------------------------- popups
function popup(lngLat, html) {
  if (!map || !window.maplibregl) return;
  const p = new window.maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: '280px' })
    .setLngLat(lngLat)
    .setHTML('<div class="popup">' + html + '</div>')
    .addTo(map);
  popups.push(p);
  if (popups.length > 4) { const old = popups.shift(); try { old.remove(); } catch (err) { /* already gone */ } }
}

function buildingPopup(feature, lngLat) {
  const p = feature.properties || {};
  const cls = num(p.damage_class);
  const parts = [
    '<h4>' + esc(p.label || p.footprint_id || 'building') + '</h4>',
    '<div class="sub">' + esc(labelOfClass(cls)) + ' (class ' + cls + ')'
      + (p.confidence !== undefined && p.confidence !== null && p.confidence !== ''
        ? ', confidence ' + Number(p.confidence).toFixed(2) : '') + '</div>',
  ];
  if (p.graded_by) parts.push('<div class="sub">graded by ' + esc(p.graded_by) + '</div>');
  if (p.confirmed === true || p.confirmed === 'true') parts.push('<div class="sub">confirmed by an operator</div>');
  if (p.footprint_id) parts.push('<div class="sub">' + esc(p.footprint_id) + '</div>');
  popup(lngLat, parts.join(''));
  if (p.footprint_id && ctx) ctx.bus.emit('building:click', { footprint_id: p.footprint_id });
}

function wireInteractions() {
  if (!map) return;
  const clickable = [
    'bld-c0', 'bld-c1', 'bld-c2', 'bld-c3', 'facilities', 'pins',
    'roads-blocked', ...AGENCIES.map((a) => 'step-' + a),
  ];
  for (const id of clickable) {
    if (!map.getLayer(id)) continue;
    map.on('mouseenter', id, () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', id, () => { map.getCanvas().style.cursor = ''; });
  }

  for (const id of ['bld-c0', 'bld-c1', 'bld-c2', 'bld-c3']) {
    if (!map.getLayer(id)) continue;
    map.on('click', id, (ev) => {
      const f = ev.features && ev.features[0];
      if (f) buildingPopup(f, ev.lngLat);
    });
  }

  if (map.getLayer('facilities')) {
    map.on('click', 'facilities', (ev) => {
      const f = ev.features && ev.features[0];
      if (!f) return;
      const p = f.properties || {};
      const type = FACILITY_LABEL[p.type] || String(p.type || 'care facility').replace(/_/g, ' ');
      const cap = p.beds || p.patients || p.capacity;
      popup(ev.lngLat, '<h4>' + esc(p.name || 'care facility') + '</h4>'
        + '<div class="sub">' + esc(type) + (cap ? ', ' + esc(cap) + ' residents' : '') + '</div>'
        + '<div class="sub">residents here cannot self-evacuate</div>');
    });
  }

  if (map.getLayer('roads-blocked')) {
    map.on('click', 'roads-blocked', (ev) => {
      const f = ev.features && ev.features[0];
      if (!f) return;
      const p = f.properties || {};
      popup(ev.lngLat, '<h4>' + esc(p.road_name || p.name || 'road') + '</h4>'
        + '<div class="sub">blocked, routes avoid this segment</div>'
        + (p.operator ? '<div class="sub">closed by ' + esc(p.operator) + '</div>' : ''));
    });
  }

  if (map.getLayer('pins')) {
    map.on('click', 'pins', (ev) => {
      const f = ev.features && ev.features[0];
      if (!f) return;
      const p = f.properties || {};
      popup(ev.lngLat, '<h4>' + esc(p.image_id || 'image') + '</h4>'
        + '<div class="sub">' + esc(p.caption || 'no caption') + '</div>');
    });
  }

  for (const a of AGENCIES) {
    const id = 'step-' + a;
    if (!map.getLayer(id)) continue;
    map.on('click', id, (ev) => {
      const f = ev.features && ev.features[0];
      if (!f) return;
      const p = f.properties || {};
      popup(ev.lngLat, '<h4>' + esc(AGENCY_LABEL[a] || a) + ' ' + esc(p.n) + '</h4>'
        + '<div class="sub">' + esc(p.label || '') + '</div>'
        + '<div class="sub">' + esc(p.task || '') + '</div>'
        + (p.units ? '<div class="sub">' + esc(p.units) + ' unit(s) assigned</div>' : ''));
      if (p.footprint_id && ctx) ctx.bus.emit('building:click', { footprint_id: p.footprint_id });
    });
  }
}

// -------------------------------------------------------------- tile probing
/** Probe one cached tile so an empty tile directory is reported, not guessed. */
function probeTile(template) {
  return new Promise((resolve) => {
    const z = 12;
    const lng = (AOI_FALLBACK[0] + AOI_FALLBACK[2]) / 2;
    const lat = (AOI_FALLBACK[1] + AOI_FALLBACK[3]) / 2;
    const n = Math.pow(2, z);
    const x = Math.floor((lng + 180) / 360 * n);
    const latRad = lat * Math.PI / 180;
    const y = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);
    const img = new Image();
    const done = (ok) => { img.onload = null; img.onerror = null; resolve(ok); };
    img.onload = () => done(true);
    img.onerror = () => done(false);
    img.src = template.replace('{z}', z).replace('{x}', x).replace('{y}', y);
    setTimeout(() => done(false), 4000);
  });
}

async function probeBasemaps() {
  const [tactical, satellite] = await Promise.all([
    probeTile(TILES_TACTICAL), probeTile(TILES_SATELLITE),
  ]);
  satelliteUsable = satellite;
  const satBtn = q('btn-basemap-satellite');
  if (satBtn && !satellite) {
    satBtn.disabled = true;
    satBtn.title = 'No satellite tiles are cached under web/tiles/sat, so this basemap is not available offline.';
  }
  if (!tactical) {
    legendNotes.tiles = 'basemap tiles not cached, geometry still draws on the dark background';
    paintLegendFoot();
  }
}

// ------------------------------------------------------------------ fetching
async function fetchJson(path) {
  if (!ctx) return null;
  try { return await ctx.api.get(path, { timeoutMs: 12000 }); }
  catch (err) { return null; }
}

function asFC(value) {
  if (value && value.type === 'FeatureCollection' && Array.isArray(value.features)) return value;
  if (value && Array.isArray(value.features)) return { type: 'FeatureCollection', features: value.features };
  return EMPTY;
}

/** Build the route lines and the numbered step points out of the agency plan. */
function planFeatures(plan) {
  const routes = [];
  const steps = [];
  const perAgency = {};
  const warnings = [];
  let approxUsed = false;
  const agencies = plan && Array.isArray(plan.agencies) ? plan.agencies : [];

  for (const entry of agencies) {
    const agency = entry && entry.agency;
    if (AGENCIES.indexOf(agency) < 0) continue;
    const list = Array.isArray(entry.steps) ? entry.steps : [];
    perAgency[agency] = list.length;

    const route = entry.route;
    const geom = route && route.geometry;
    if (geom && geom.type === 'LineString' && Array.isArray(geom.coordinates) && geom.coordinates.length > 1) {
      routes.push({ type: 'Feature', geometry: geom, properties: { agency, approx: false } });
      if (route.warning) warnings.push(AGENCY_LABEL[agency] + ': ' + route.warning);
      if (route.ok === false) warnings.push(AGENCY_LABEL[agency] + ': no clean route, the drawn line is not usable as given');
    } else if (agency !== 'police') {
      // No routed geometry yet, or none exists. Draw the connector and say so,
      // rather than letting a straight diagonal pass for a road-following route.
      //
      // Police are excluded on purpose: their steps are static posts, most often
      // the two ends of one closure, and a line between them would draw traffic
      // along the very road that is shut.
      const line = list
        .map((s) => s && s.centroid)
        .filter((c) => Array.isArray(c) && c.length >= 2);
      if (line.length > 1) {
        routes.push({
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: line },
          properties: { agency, approx: true },
        });
        approxUsed = true;
      }
    }

    for (const s of list) {
      if (!s || !Array.isArray(s.centroid) || s.centroid.length < 2) continue;
      const n = s.n === undefined || s.n === null ? steps.length + 1 : s.n;
      steps.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [Number(s.centroid[0]), Number(s.centroid[1])] },
        properties: {
          agency, n,
          icon: numberIcon(agency, n),
          label: s.label || '',
          task: s.task || '',
          units: s.units || 0,
          footprint_id: s.footprint_id || '',
        },
      });
    }
  }

  if (approxUsed) {
    warnings.unshift('dashed agency lines are straight connectors between stops, not routed roads');
  }
  legendNotes.plan = warnings.join('; ');
  return {
    routes: { type: 'FeatureCollection', features: routes },
    steps: { type: 'FeatureCollection', features: steps },
    perAgency,
  };
}

function flightNote(fc) {
  const path = (fc.features || []).find((f) => f && f.properties && f.properties.role === 'survey-path');
  if (!path) return '';
  const p = path.properties || {};
  const bits = [];
  if (p.transects) bits.push(p.transects + ' transects');
  if (p.line_spacing_m) bits.push(p.line_spacing_m + ' m spacing');
  if (p.altitude_m_agl) bits.push(p.altitude_m_agl + ' m above ground');
  if (p.est_flight_min) bits.push(Math.round(Number(p.est_flight_min)) + ' min');
  return bits.length ? 'next flight: ' + bits.join(', ') : '';
}

function setData(id, data) {
  if (!map || !mapReady) return;
  const src = map.getSource(id);
  if (src && typeof src.setData === 'function') src.setData(data || EMPTY);
}

function countBuildings(fc) {
  const out = { c0: 0, c1: 0, c2: 0, c3: 0, confirmed: 0 };
  for (const f of fc.features || []) {
    const p = (f && f.properties) || {};
    const c = num(p.damage_class);
    if (c >= 0 && c <= 3) out['c' + c] += 1;
    if (p.confirmed === true || p.confirmed === 'true') out.confirmed += 1;
  }
  return out;
}

async function loadAll() {
  if (inFlight) return;
  inFlight = true;
  try {
    const [buildingsRaw, roadsRaw, facilitiesRaw, planRaw, flightRaw] = await Promise.all([
      fetchJson('api/buildings'), fetchJson('api/roads'), fetchJson('api/facilities'),
      fetchJson('api/plan'), flightOverride ? Promise.resolve(null) : fetchJson('api/flight'),
    ]);

    const buildings = asFC(buildingsRaw);
    const roads = asFC(roadsRaw);
    const facilities = asFC(facilitiesRaw);
    const flight = flightOverride ? asFC(flightOverride) : asFC(flightRaw);
    const plan = planRaw && Array.isArray(planRaw.agencies) ? planRaw : { agencies: [], drafted_by: '' };

    const built = planFeatures(plan);

    setData('buildings', buildings);
    setData('roads', roads);
    setData('facilities', facilities);
    setData('routes', built.routes);
    setData('steps', built.steps);
    setData('flight', flight);

    const bcounts = countBuildings(buildings);
    let blocked = 0;
    let open = 0;
    for (const f of roads.features || []) {
      const p = (f && f.properties) || {};
      if (p.blocked === true || p.blocked === 'true') blocked += 1; else open += 1;
    }
    Object.assign(counts, bcounts, {
      roads: open,
      blocked,
      facilities: (facilities.features || []).length,
      flight: (flight.features || []).length ? 1 : 0,
    });
    for (const a of AGENCIES) counts['route-' + a] = built.perAgency[a] || 0;

    legendNotes.flight = flightNote(flight);
    paintLegend();
    applyAllRows();

    if (!firstFitDone) {
      const box = bboxOf([buildings, flight, built.steps, facilities]);
      if (box) { fitBox(box); firstFitDone = true; }
    }

    if (ctx) {
      // counts.plan carries the RAW api/plan payload so the dispatch panel can
      // render from this one read instead of fetching the endpoint again.
      ctx.bus.emit('counts', Object.assign({}, counts, {
        routes: AGENCIES.reduce((acc, a) => { acc[a] = built.perAgency[a] || 0; return acc; }, {}),
        plan,
        agency_count: plan.agencies.length,
      }));
      ctx.bus.emit('plan', plan);
      ctx.bus.emit('flight', flight);
    }
  } finally {
    inFlight = false;
  }
}

function fitBox(box) {
  if (!map || !mapReady) return;
  const padded = pad(box, 0.08);
  try {
    map.fitBounds([[padded[0], padded[1]], [padded[2], padded[3]]], { padding: 40, duration: 600, maxZoom: 17 });
  } catch (err) { /* a degenerate bbox is not worth a broken console */ }
}

// -------------------------------------------------------------------- basemap
function setBasemap(which) {
  const tacticalBtn = q('btn-basemap-tactical');
  const satBtn = q('btn-basemap-satellite');
  if (which === 'satellite' && !satelliteUsable) {
    toast('No satellite tiles are cached on this box, staying on the tactical basemap.', 'warn');
    return;
  }
  if (tacticalBtn) tacticalBtn.classList.toggle('on', which !== 'satellite');
  if (satBtn) satBtn.classList.toggle('on', which === 'satellite');
  if (!map || !mapReady) return;
  try {
    map.setLayoutProperty('base-tactical', 'visibility', which === 'satellite' ? 'none' : 'visible');
    map.setLayoutProperty('base-sat', 'visibility', which === 'satellite' ? 'visible' : 'none');
  } catch (err) { /* style not ready, the buttons already show intent */ }
}

// ----------------------------------------------------------------- public API
export function available() {
  return !!map && !engineMissing;
}

export function refresh() {
  return loadAll();
}

export function flyTo(centroid, opts) {
  if (!Array.isArray(centroid) || centroid.length < 2) return;
  const center = [Number(centroid[0]), Number(centroid[1])];
  if (!isFinite(center[0]) || !isFinite(center[1])) return;
  if (!map || !mapReady) {
    toast('The map is not available, so the location cannot be shown.', 'warn');
    return;
  }
  map.flyTo({ center, zoom: (opts && opts.zoom) || 17, duration: (opts && opts.duration) || 900 });
}

export function setCounts(partial) {
  if (!partial) return;
  Object.assign(counts, partial);
  paintLegend();
}

export function applyPreset(name) {
  if (PRESET_NAMES.indexOf(name) < 0) return;
  for (const row of allRows()) {
    rowState.set(row.id, (row.presets || []).indexOf(name) >= 0);
  }
  applyAllRows();
  paintLegend();
}

export function showPins(items) {
  const list = Array.isArray(items) ? items : [];
  const features = [];
  for (const item of list) {
    const c = item && item.centroid;
    if (!Array.isArray(c) || c.length < 2) continue;
    features.push({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [Number(c[0]), Number(c[1])] },
      properties: {
        image_id: item.image_id || '',
        caption: item.caption || '',
        color: COLORS['c' + num(item.class_max)] || COLORS.blue,
      },
    });
  }
  const fc = { type: 'FeatureCollection', features };
  setData('pins', fc);
  counts.pins = features.length;
  if (features.length) rowState.set('pins', true);
  paintLegend();
  applyRowToMap(allRows().find((r) => r.id === 'pins'));
  if (features.length) {
    const box = bboxOf([fc]);
    if (box) fitBox(box);
  }
  return features.length;
}

export function showFlight(featureCollection) {
  flightOverride = featureCollection || null;
  const fc = asFC(flightOverride);
  setData('flight', flightOverride ? fc : EMPTY);
  counts.flight = (fc.features || []).length ? 1 : 0;
  legendNotes.flight = flightNote(fc);
  paintLegend();
  if (!flightOverride) loadAll();
}

export function showRoute(geometry, opts) {
  if (!geometry || geometry.type !== 'LineString' || !Array.isArray(geometry.coordinates)) {
    setData('nav', EMPTY);
    counts.nav = 0;
    paintLegend();
    return;
  }
  const fc = {
    type: 'FeatureCollection',
    features: [{ type: 'Feature', geometry, properties: { label: (opts && opts.label) || '' } }],
  };
  setData('nav', fc);
  counts.nav = 1;
  rowState.set('nav', true);
  paintLegend();
  const navRow = allRows().find((r) => r.id === 'nav');
  if (navRow && map && mapReady) {
    const color = (opts && opts.agency && agencyColor(opts.agency)) || COLORS.green;
    try { map.setPaintProperty('nav-line', 'line-color', color); } catch (err) { /* style busy */ }
    applyRowToMap(navRow);
  }
  const box = bboxOf([fc]);
  if (box) fitBox(box);
}

/** Raw MapLibre handle, or null. Sibling panels that need a control the module
 *  does not wrap can reach it, and the smoke test asserts against the live
 *  layer state rather than our own CSS classes. */
export function handle() {
  return mapReady ? map : null;
}

/** Let the operator click a line on the map, for a road closure geometry.
 *  Resolves with a LineString, or null when they cancel with Escape or a
 *  right click. Never rejects, so the caller needs no try/catch. */
export function captureLine(opts) {
  if (!map || !mapReady) {
    toast('The map is not available, so a road cannot be drawn.', 'warn');
    return Promise.resolve(null);
  }
  if (drawing) {
    toast('A road is already being drawn. Finish it or press Escape.', 'warn');
    return Promise.resolve(null);
  }
  drawing = true;
  const coords = [];
  const canvas = map.getCanvas();
  const priorCursor = canvas.style.cursor;
  canvas.style.cursor = 'crosshair';
  const hint = (opts && opts.hint)
    || 'Click along the closed road, then double click or press Enter to finish. Escape cancels.';
  if (ctx) ctx.bus.emit('banner', { text: hint });

  const repaint = () => {
    const features = coords.map((c) => ({
      type: 'Feature', geometry: { type: 'Point', coordinates: c }, properties: {},
    }));
    if (coords.length > 1) {
      features.push({
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: coords.slice() },
        properties: {},
      });
    }
    setData('draw', { type: 'FeatureCollection', features });
  };

  return new Promise((resolve) => {
    const finish = (result) => {
      drawing = false;
      canvas.style.cursor = priorCursor;
      map.off('click', onClick);
      map.off('dblclick', onDouble);
      map.off('contextmenu', onCancel);
      document.removeEventListener('keydown', onKey);
      setData('draw', EMPTY);
      if (ctx) ctx.bus.emit('banner', null);
      resolve(result);
    };
    const onClick = (ev) => { coords.push([ev.lngLat.lng, ev.lngLat.lat]); repaint(); };
    const done = () => {
      if (coords.length < 2) {
        toast('A road closure needs at least two points.', 'warn');
        finish(null);
        return;
      }
      finish({ type: 'LineString', coordinates: coords.slice() });
    };
    const onDouble = (ev) => {
      if (ev && ev.preventDefault) ev.preventDefault();
      done();
    };
    const onCancel = (ev) => { if (ev && ev.preventDefault) ev.preventDefault(); finish(null); };
    const onKey = (ev) => {
      if (ev.key === 'Escape') finish(null);
      else if (ev.key === 'Enter') done();
    };
    map.on('click', onClick);
    map.on('dblclick', onDouble);
    map.on('contextmenu', onCancel);
    document.addEventListener('keydown', onKey);
  });
}

export async function init(context) {
  ctx = context;

  buildLegend();

  const chevron = q('legend-chevron');
  const legend = q('legend');
  if (chevron && legend) {
    chevron.addEventListener('click', () => {
      const collapsed = legend.classList.toggle('collapsed');
      chevron.title = collapsed ? 'Show the layer list' : 'Collapse the layer list and give the map back';
    });
  }
  for (const btn of document.querySelectorAll('.lpreset')) {
    btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
  }

  const tacticalBtn = q('btn-basemap-tactical');
  const satBtn = q('btn-basemap-satellite');
  if (tacticalBtn) tacticalBtn.addEventListener('click', () => setBasemap('tactical'));
  if (satBtn) satBtn.addEventListener('click', () => setBasemap('satellite'));
  const fitBtn = q('btn-fit');
  if (fitBtn) {
    fitBtn.addEventListener('click', () => {
      firstFitDone = false;
      loadAll();
    });
  }

  ctx.bus.on('data:changed', () => loadAll());
  ctx.bus.on('plan:changed', () => loadAll());
  ctx.bus.on('locate', (detail) => { if (detail && detail.centroid) flyTo(detail.centroid); });

  if (!window.maplibregl || typeof window.maplibregl.Map !== 'function') {
    engineMissing = true;
    const box = q('map-fallback');
    if (box) box.hidden = false;
    const body = q('legend-body');
    if (body) body.innerHTML = '<div class="empty">no map engine, so no layers to filter</div>';
    // Presets, basemaps and fit all act on a map that is not there. Hide the
    // controls rather than leaving dead buttons an operator will click twice.
    for (const btn of document.querySelectorAll('.lpreset')) btn.hidden = true;
    const chev = q('legend-chevron');
    if (chev) chev.hidden = true;
    for (const btn of [tacticalBtn, satBtn, fitBtn]) if (btn) btn.disabled = true;
    // The counts still matter to the rail and the rank panel, so keep fetching.
    await loadAll();
    return;
  }

  const host = q('map');
  if (!host) return;

  try {
    map = new window.maplibregl.Map({
      container: host,
      style: buildStyle(),
      center: [(AOI_FALLBACK[0] + AOI_FALLBACK[2]) / 2, (AOI_FALLBACK[1] + AOI_FALLBACK[3]) / 2],
      zoom: 13,
      attributionControl: false,
      // Fonts would need a glyph server, and there is none offline.
      localIdeographFontFamily: false,
    });
    map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
    map.addControl(new window.maplibregl.ScaleControl({ maxWidth: 110, unit: 'imperial' }), 'bottom-left');
  } catch (err) {
    engineMissing = true;
    map = null;
    const box = q('map-fallback');
    const why = q('map-fallback-why');
    if (why) why.textContent = 'The map engine failed to start: ' + (err && err.message ? err.message : err);
    if (box) box.hidden = false;
    await loadAll();
    return;
  }

  map.on('error', (ev) => {
    // Missing tiles are expected when a cache is thin. Never throw over them.
    const msg = ev && ev.error && ev.error.message ? ev.error.message : '';
    if (msg) console.warn('map:', msg);
  });

  await new Promise((resolve) => {
    if (map.loaded()) { resolve(); return; }
    map.once('load', resolve);
    setTimeout(resolve, 8000);
  });

  mapReady = true;
  // Sibling panels and the smoke test read the live layer state from here
  // rather than trusting our own CSS classes.
  window.__flMap = map;
  crossIcon();
  wireInteractions();
  applyAllRows();
  setBasemap('tactical');
  probeBasemaps();
  await loadAll();
}
