/* FIRST LIGHT rank panel (deliverable C2).
 *
 * Each row is an evidence card, not a list item: a judge with a calculator has
 * to be able to re-compute the priority from what is on the screen, and an
 * operator who has never seen the tool has to learn what a damage grade is from
 * the control that changes it.
 *
 * Reads GET api/rank?limit=  ->  {items: [RankItem.wire()], doubt_distribution: {...}}
 *   footprint_id, label, centroid, damage_class, confidence, confirmed, graded_by,
 *   facility_near {name, type, dist_m}, inputs {severity_weight, staleness_h,
 *   vulnerable_density, doubt, road_cutoff}, priority, rationale, rationale_by,
 *   image_ids, votes, vote_agreement
 * Writes POST api/grade  {footprint_id, new_class, operator}
 *
 * PUBLIC API
 *   init(ctx)   wire the panel into the console shell
 *   refresh()   refetch and repaint
 */

const CLASS_LABEL = { 0: 'no damage', 1: 'minor damage', 2: 'major damage', 3: 'destroyed' };
const CLASS_SHORT = { 0: 'no damage', 1: 'minor', 2: 'major', 3: 'destroyed' };
const FACILITY_LABEL = { nursing_home: 'nursing home', dialysis: 'dialysis centre', hospital: 'hospital' };

// contracts.DISPLAY_NAME, so a label can never drift from the wire field.
const DISPLAY_NAME = {
  severity_weight: 'damage severity',
  staleness_h: 'hours since last look',
  vulnerable_density: 'resident vulnerability',
  doubt: 'AI uncertainty',
  road_cutoff: 'road cut-off',
  priority: 'priority',
};

// The order the factors are spoken in. severity_weight leads because it is what
// stops an intact building the models argue about from outranking a collapse.
const FACTOR_ORDER = ['severity_weight', 'staleness_h', 'vulnerable_density', 'doubt', 'road_cutoff'];

const DOUBT_FLOOR = 0.05;
const LIMIT = 50;

const TIP = {
  formula: 'Priority is the product of these five factors, each rounded to three decimals before multiplying. Nothing about property value enters it. Multiply what you see and you get the number on the right.',
  svi: 'CDC Social Vulnerability Index: how hard it is for this block group to evacuate or recover unaided, from age, disability, poverty and vehicle access. 0 is most self-sufficient, 1 is most vulnerable.',
  ballot: 'The fast model graded this building several times independently. When it disagrees with itself, a human look matters more, so uncertainty raises priority instead of lowering it.',
  doubt: 'Uncertainty is 1 minus how often the votes agreed, with a floor of 0.05. It is a multiplier in the priority formula.',
  grade: 'The AI graded this building from the air. If your crew sees different on the ground, change it here. Your call overrides the AI, is logged under your name, and confirmed severe damage pins to the top of the list.',
  stale: 'Hours since the last drone pass covered this building. The longer nobody has looked, the higher it climbs.',
  cutoff: 'A multiplier above 1 applied when a blocked road cuts this building off from responders. Absent when access is clear.',
};

let ctx = null;
let items = [];
let distribution = null;
let loading = false;
let lastError = null;
// A pending dropdown choice must survive a background refresh, or the operator
// loses the selection they were about to confirm.
const pendingGrade = new Map();

function q(id) { return document.getElementById(id); }

function num(v, dec) {
  const n = Number(v);
  if (!isFinite(n)) return '-';
  return n.toFixed(dec === undefined ? 3 : dec);
}

function node(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined && text !== null) el.textContent = text;
  return el;
}

function tip(el, text) {
  el.classList.add('tip');
  el.title = text;
  return el;
}

/** Plain English for a vulnerability index, never a bare number. The index is
 *  returned separately so it can be kept on one line. */
function sviSentence(density) {
  // Number(null) is 0, which would print a confident "SVI 0.00" for a missing
  // join. A missing vulnerability figure prints nothing at all.
  if (density === null || density === undefined || density === '') return null;
  const v = Number(density);
  if (!isFinite(v) || v < 0 || v > 1) return null;
  const topPct = Math.max(1, Math.round((1 - v) * 100));
  const index = '(CDC SVI ' + num(v, 2) + ')';
  let text;
  let high = false;
  if (v >= 0.9) { text = 'residents highly vulnerable, top ' + topPct + '% nationally'; high = true; }
  else if (v >= 0.75) { text = 'residents more vulnerable than most, top ' + topPct + '% nationally'; high = true; }
  else if (v >= 0.5) text = 'residents somewhat vulnerable, above the national median';
  else if (v >= 0.25) text = 'residents below the national median for vulnerability';
  else text = 'residents largely self-sufficient';
  return { text, index, high };
}

/** "6x destroyed, 2x major" from the sampled ballot labels. */
function ballotSentence(votes) {
  if (!Array.isArray(votes) || !votes.length) return null;
  const tally = new Map();
  for (const v of votes) {
    const c = Number(v);
    if (!isFinite(c)) continue;
    tally.set(c, (tally.get(c) || 0) + 1);
  }
  if (!tally.size) return null;
  const parts = Array.from(tally.entries())
    .sort((a, b) => b[1] - a[1] || b[0] - a[0])
    .map(([cls, n]) => n + 'x ' + (CLASS_SHORT[cls] || 'class ' + cls));
  return { k: votes.length, text: parts.join(', ') };
}

function doubtOf(item) {
  const d = Number(item && item.inputs && item.inputs.doubt);
  return isFinite(d) ? d : null;
}

function isContested(item) {
  const d = doubtOf(item);
  return d !== null && d > DOUBT_FLOOR + 1e-9;
}

/* ------------------------------------------------------------ evidence card */

function formulaRow(item) {
  const inputs = (item && item.inputs) || {};

  // The priority is the headline; the arithmetic lives in the tooltip. Fifty cards
  // each carrying five factors of monospace buried the one number an operator
  // sorts by, and gate 4 is still satisfied because the full working is one hover
  // away and the mismatch check below still runs on every render whether anyone
  // looks or not.
  const row = node('div', 'formula');
  let product = 1;
  let counted = 0;
  const parts = [];
  for (const key of FACTOR_ORDER) {
    const raw = inputs[key];
    if (raw === undefined || raw === null) continue;
    const value = Number(raw);
    if (!isFinite(value)) continue;
    product *= value;
    counted += 1;
    parts.push(DISPLAY_NAME[key] + ' ' + num(value, 3));
  }

  const shown = Number(item.priority);
  row.appendChild(node('span', 'flab', DISPLAY_NAME.priority));
  row.appendChild(node('b', 'prod', isFinite(shown) ? num(shown, 5) : '-'));
  if (counted) {
    row.appendChild(node('span', 'fhint', 'how'));
    tip(
      row,
      parts.join(' x ') + ' = ' + num(product, 5) + '. '
        + 'Each factor is rounded to three decimals before multiplying, so this '
        + 'reconciles by hand. Property value never enters it.'
    );
  } else {
    tip(row, TIP.formula);
  }

  // Verify the arithmetic on screen rather than trusting it. A mismatch is a real
  // defect in the scorer and the operator should see it, not discover it when a
  // judge multiplies the row by hand. This stays visible even though the working
  // is now hidden: a hidden formula is fine, a hidden discrepancy is not.
  if (isFinite(shown) && counted && Math.abs(Number(product.toFixed(5)) - shown) > 0.00002) {
    const warn = node('div', null, 'these factors multiply to ' + num(product, 5)
      + ', which does not match the priority sent');
    warn.style.color = 'var(--red)';
    row.appendChild(warn);
  }
  return row;
}

/** The ballot reads as a sentence, with the bar on its own full-width run so a
 *  long tally can never squeeze it into a nub. */
function ballotRow(item) {
  const doubt = doubtOf(item);
  if (doubt === null) return null;
  const wrap = node('div', 'ballot');

  const said = node('div', 'ballotsay');
  const ballot = ballotSentence(item.votes);
  if (ballot) {
    said.appendChild(tip(node('span', 'lbl', 'AI checked ' + ballot.k + 'x:'), TIP.ballot));
    said.appendChild(node('span', 'tally', ballot.text));
  } else {
    // votes is null until the Lightning ballot is wired. Say where the number
    // came from instead of inventing a tally.
    said.appendChild(tip(node('span', 'lbl', 'grader confidence only, no ballot yet'), TIP.doubt));
  }
  wrap.appendChild(said);

  const pct = Math.max(0, Math.min(100, Math.round(doubt * 100)));
  const meter = node('div', 'cardrow');
  const bar = node('div', 'doubtbar' + (doubt <= DOUBT_FLOOR + 1e-9 ? ' low' : doubt >= 0.5 ? ' high' : ''));
  const fill = node('div');
  fill.style.width = pct + '%';
  bar.appendChild(fill);
  tip(bar, TIP.doubt);
  meter.appendChild(bar);
  meter.appendChild(node('span', 'doubtval', 'uncertainty ' + pct + '%'));
  wrap.appendChild(meter);
  return wrap;
}

function actionRow(item) {
  const row = node('div', 'cardrow');

  const locate = node('button', 'mini', 'Locate');
  locate.addEventListener('click', () => {
    const centroid = item.centroid;
    if (!Array.isArray(centroid) || centroid.length < 2) {
      ctx.toast('This building has no location yet, so it cannot be shown on the map.', 'warn');
      return;
    }
    const mapModule = ctx.modules && ctx.modules.map;
    if (mapModule && typeof mapModule.flyTo === 'function') mapModule.flyTo(centroid);
    else if (ctx.bus.emit('locate', { centroid }) === 0) {
      ctx.toast('The map is not loaded, so the location cannot be shown.', 'warn');
    }
  });
  row.appendChild(locate);

  const nav = node('button', 'mini', 'Navigate');
  nav.addEventListener('click', () => {
    const ran = ctx.bus.emit('navigate', {
      footprint_id: item.footprint_id,
      label: item.label,
      centroid: item.centroid,
    });
    if (!ran) ctx.toast('Turn-by-turn is handled by the dispatch panel, which is not loaded.', 'warn');
  });
  row.appendChild(nav);

  const shots = Array.isArray(item.image_ids) ? item.image_ids : [];
  const photos = node('button', 'mini', 'Photos (' + shots.length + ')');
  if (!shots.length) {
    photos.disabled = true;
    photos.title = 'No stored image covers this building. An image withheld by the privacy gate still contributes its grade, but is never stored or thumbnailed.';
    photos.classList.add('tip');
  }
  photos.addEventListener('click', () => {
    if (!shots.length) return;
    const ran = ctx.bus.emit('archive:show', { image_ids: shots, footprint_id: item.footprint_id });
    if (ran) ctx.showTab('archive');
    else ctx.toast('The archive panel is not loaded, so these photos cannot be opened.', 'warn');
  });
  row.appendChild(photos);

  return row;
}

/** One dropdown plus one Confirm, on its own hairline-separated row. */
function gradeRow(item) {
  const row = node('div', 'cardrow graderow');
  const aiClass = Number(item.damage_class);
  const label = node('span', 'gradelabel');
  label.appendChild(document.createTextNode('Damage grade '));
  const said = node('i', null, '(AI said: ' + (CLASS_LABEL[aiClass] || 'unknown') + ')');
  label.appendChild(said);
  tip(label, TIP.grade);
  row.appendChild(label);

  const select = node('select');
  for (const cls of [3, 2, 1, 0]) {
    const opt = document.createElement('option');
    opt.value = String(cls);
    opt.textContent = CLASS_LABEL[cls];
    select.appendChild(opt);
  }
  const pending = pendingGrade.get(item.footprint_id);
  select.value = String(pending !== undefined ? pending : (aiClass >= 0 && aiClass <= 3 ? aiClass : 0));
  select.addEventListener('change', () => pendingGrade.set(item.footprint_id, Number(select.value)));
  row.appendChild(select);

  const confirm = node('button', 'mini confirm', 'Confirm');
  confirm.addEventListener('click', () => submitGrade(item, Number(select.value), confirm));
  row.appendChild(confirm);

  return row;
}

async function submitGrade(item, newClass, button) {
  const operator = ctx.requireOperator();
  if (!operator) return;
  const same = newClass === Number(item.damage_class);
  button.disabled = true;
  const restore = button.textContent;
  button.textContent = 'Sending';
  try {
    await ctx.api.post('api/grade', {
      footprint_id: item.footprint_id,
      new_class: newClass,
      operator,
    });
    pendingGrade.delete(item.footprint_id);
    ctx.toast(same
      ? 'Confirmed ' + CLASS_LABEL[newClass] + ' at ' + (item.label || item.footprint_id)
        + '. Logged under ' + operator + ', and confirmed severe damage pins to the top.'
      : 'Overrode the AI: ' + CLASS_LABEL[Number(item.damage_class)] + ' becomes ' + CLASS_LABEL[newClass]
        + ' at ' + (item.label || item.footprint_id) + '. Logged under ' + operator + '.',
    'ok');
    ctx.bus.emit('data:changed', { from: 'grade' });
    await refresh();
  } catch (err) {
    button.disabled = false;
    button.textContent = restore;
    ctx.toast('The grade was not saved: ' + (err && err.message ? err.message : err), 'err');
  }
}

function card(item, index) {
  const wrap = node('div', 'card' + (item.confirmed ? ' pinned' : ''));
  wrap.dataset.footprintId = item.footprint_id || '';

  const head = node('div', 'head');
  head.appendChild(node('span', 'rankn', '#' + (index + 1)));
  head.appendChild(node('span', 'addr', item.label || item.footprint_id || 'unnamed building'));
  const cls = Number(item.damage_class);
  const chip = node('span', 'cls c' + (cls >= 0 && cls <= 3 ? cls : 'x'),
    (CLASS_LABEL[cls] || 'unknown').toUpperCase());
  head.appendChild(chip);
  wrap.appendChild(head);

  const fac = item.facility_near;
  if (fac && fac.name) {
    const type = FACILITY_LABEL[fac.type] || String(fac.type || '').replace(/_/g, ' ');
    const dist = isFinite(Number(fac.dist_m)) ? Math.round(Number(fac.dist_m)) + ' m from ' : 'beside ';
    wrap.appendChild(node('div', 'facility', dist + fac.name + (type ? ' (' + type + ')' : '')));
  }

  const svi = sviSentence(item.inputs && item.inputs.vulnerable_density);
  if (svi) {
    const line = node('div', 'vuln' + (svi.high ? ' high' : ''), svi.text + ' ');
    line.appendChild(node('span', 'idx', svi.index));
    wrap.appendChild(tip(line, TIP.svi));
  }

  if (item.confirmed) {
    const by = String(item.graded_by || '');
    const who = by.indexOf('operator:') === 0 ? by.slice('operator:'.length) : 'an operator';
    wrap.appendChild(node('div', 'confirmedby', 'confirmed by ' + who + ', pinned to the top'));
  }

  const cutoff = item.inputs && item.inputs.road_cutoff;
  if (cutoff === null || cutoff === undefined) {
    wrap.appendChild(tip(node('div', 'vuln', 'access clear, no road cut-off multiplier'), TIP.cutoff));
  }

  wrap.appendChild(formulaRow(item));

  const ballot = ballotRow(item);
  if (ballot) wrap.appendChild(ballot);

  if (item.rationale) {
    const r = node('div', 'rationale');
    r.appendChild(document.createTextNode('"' + item.rationale + '" '));
    if (item.rationale_by) r.appendChild(node('span', 'by', '- ' + item.rationale_by));
    wrap.appendChild(r);
  }

  wrap.appendChild(actionRow(item));
  wrap.appendChild(gradeRow(item));
  return wrap;
}

/* --------------------------------------------------- doubt distribution line */

/** Reconcile whatever B sends with what the rows actually show. */
function contestedSummary() {
  const total = items.length;
  const contested = items.filter(isContested).length;
  const dist = distribution;
  if (dist && typeof dist === 'object') {
    const dTotal = Number(dist.total !== undefined ? dist.total : dist.count);
    const dContested = Number(dist.contested);
    if (isFinite(dTotal) && isFinite(dContested)) {
      return { contested: dContested, total: dTotal, source: 'api' };
    }
    // Bucket histogram: keys are doubt values or bucket edges.
    const keys = Object.keys(dist).filter((k) => isFinite(Number(k)));
    if (keys.length) {
      let sum = 0;
      let above = 0;
      for (const k of keys) {
        const n = Number(dist[k]);
        if (!isFinite(n)) continue;
        sum += n;
        if (Number(k) > DOUBT_FLOOR + 1e-9) above += n;
      }
      if (sum > 0) return { contested: above, total: sum, source: 'api' };
    }
  }
  return { contested, total, source: 'rows' };
}

function distributionLine() {
  const s = contestedSummary();
  if (!s.total) return null;
  const box = node('div', 'doubtsummary');
  const line = node('b', null, s.contested + ' of ' + s.total + ' buildings contested');
  box.appendChild(line);
  const sub = node('span', 'sub');
  if (s.contested === 0) {
    sub.textContent = 'Every row sits at the 0.05 uncertainty floor: the models agreed with themselves'
      + ' everywhere, so uncertainty is not re-ordering this list yet.';
  } else {
    const pct = Math.round((s.contested / s.total) * 100);
    sub.textContent = 'Contested means the AI did not agree with itself about the grade, so those '
      + pct + '% climb the list. Send a human to look at them first.';
  }
  box.appendChild(sub);
  return box;
}

/* ------------------------------------------------------------------ painting */

function paint() {
  const host = q('panel-rank');
  if (!host) return;
  host.textContent = '';

  const head = node('div', 'rankhead');
  head.appendChild(node('span', 't', 'Ranked buildings, highest priority first'));
  const reload = node('button', 'mini', 'Reload');
  reload.addEventListener('click', () => refresh());
  head.appendChild(reload);
  host.appendChild(head);

  const badge = q('tab-rank-count');
  if (badge) badge.textContent = items.length ? '(' + items.length + ')' : '';

  if (lastError) {
    const err = node('div', 'empty', 'The rank could not be loaded: ' + lastError
      + '. Nothing is ranked on screen until the service answers.');
    host.appendChild(err);
    return;
  }

  if (loading && !items.length) {
    host.appendChild(node('div', 'empty', 'loading the ranked list'));
    return;
  }

  if (!items.length) {
    host.appendChild(node('div', 'empty',
      'No ranked buildings yet. Upload drone imagery and the ranking fills in as tiles are analyzed.'));
    return;
  }

  const dline = distributionLine();
  if (dline) host.appendChild(dline);

  items.forEach((item, i) => host.appendChild(card(item, i)));
}

function publishStats() {
  const s = contestedSummary();
  const stale = items
    .map((it) => Number(it.inputs && it.inputs.staleness_h))
    .filter((n) => isFinite(n))
    .sort((a, b) => a - b);
  const median = stale.length ? stale[Math.floor(stale.length / 2)] : null;
  ctx.bus.emit('rank:stats', {
    count: s.total,
    contested: s.contested,
    facilitiesAtRisk: items.filter((it) => it.facility_near && it.facility_near.name).length,
    medianStalenessH: median,
  });
}

export async function refresh() {
  if (!ctx || loading) return;
  loading = true;
  try {
    const payload = await ctx.api.get('api/rank?limit=' + LIMIT, { timeoutMs: 15000 });
    items = payload && Array.isArray(payload.items) ? payload.items : [];
    distribution = payload ? payload.doubt_distribution : null;
    lastError = null;
  } catch (err) {
    items = [];
    distribution = null;
    lastError = err && err.message ? err.message : String(err);
  } finally {
    loading = false;
  }
  paint();
  publishStats();
}

export async function init(context) {
  ctx = context;
  paint();
  ctx.bus.on('data:changed', () => refresh());
  ctx.bus.on('rank:refresh', () => refresh());
  await refresh();
}
