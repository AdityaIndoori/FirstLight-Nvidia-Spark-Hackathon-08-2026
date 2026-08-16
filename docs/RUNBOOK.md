# Running FIRST LIGHT: clean start to analysed plan

Everything below was run against the box at `10.10.0.3` and produced the output
shown. Numbers are measured, not quoted.

## 0. The box is clean right now

```
tiles analyzed : 0      buildings : 0      archive : 0      dispatch steps : 0
```

The console shows all three stage cards as `waiting`, every agency at
`needs 0 / have 0`, and the footer reads `per-tile not measured yet`. That is the
correct empty state — nothing is broken, nothing has been analysed.

Reference data is still loaded, because it is not a result: 6,211 roads (2,089
named), 4 care facilities, 20,429 building footprints, and the basemap tile cache.

---

## 1. Open the console

    http://10.10.0.3:8081/

If you have had it open through a redeploy, hard-reload once (`Ctrl+Shift+R`).
Static assets are served `no-store` now, so this should not be necessary again.

## 2. Enter your operator name — do this first

Top bar, `Operator name (required to edit)`. Type e.g. `R. Alvarez`.

**Nothing you change is accepted without it.** Grade flips, reassignments, road
closures and availability all refuse and tell you why. Every one is written to an
append-only log under that name — the database physically rejects `UPDATE` and
`DELETE` on that table, so the audit trail cannot be rewritten, not even by the
maintenance tooling.

## 3. Point it at imagery

### Option A — the Upload button (what you asked for)

1. Click **Upload drone images**.
2. Navigate to the test-image folder (`demo/test_images/` in the working
   checkout; regenerate it on the box with `service/tools/pick_test_images.py`).
3. Select **all files** (`Ctrl+A` is fine).
4. Open.

**Select the `.bounds.json` sidecars too.** These NOAA frames carry no EXIF GPS,
and the geo chain is GeoTIFF transform → EXIF GPS → sidecar. Without the sidecar
a tile lands in `NEEDS GEO` and cannot be graded until you drag it onto the map.
`MANIFEST.json` and `README.md` are filtered client-side and get no card.

### Option B — the watch folder (how a real downlink feeds it)

Drop images into `~/fl/service/data/watch/` on the box. The poller picks them up
within a couple of seconds; no clicking. Sidecars are read off disk here, so
`frame.jpg` + `frame.jpg.bounds.json` both go in.

```bash
scp "<test-images>/drone_00_68688_108010.jpg"* spark:~/fl/service/data/watch/
```

Either path runs the identical pipeline. Upload deliberately does **not**
deduplicate — the privacy gate must re-run on every submission rather than trust a
cached verdict from a possibly different build.

### Option C — your own imagery

Any `.jpg/.png/.tif/.tiff/.webp/.jp2`. To be gradeable it needs a location, from
one of: a GeoTIFF transform, EXIF GPS tags, a `.bounds.json` sidecar
(`{"bounds": [west, south, east, north]}`), or a manual drag onto the map.

It must also fall inside the configured AOI (`FIRSTLIGHT_AOI=bay`, Panama City:
`-85.72, 30.13, -85.62, 30.22`), because the vulnerability join reads county
footprints, parcels, facilities and SVI for that area. Imagery from elsewhere
analyses but joins to nothing.

## 4. Watch it work — expect ~10-15 s per tile

Each image gets a card with three stages: `privacy check → damage spotting →
vulnerability indexing`, and a live elapsed counter (`on the box · 6s`).

Measured on these six frames: **p50 14.0 s**, min 10.3 s, max 17.8 s, end to end —
privacy gate, VL grading (8 crops per tile), k=8 uncertainty ballot, vulnerability
join, archive write.

| File | Time | Outcome | Buildings |
|---|---|---|---|
| `drone_00_68688_108010` | 11.6 s | stored | 14 |
| `drone_01_68686_108003` | 17.8 s | stored | 12 |
| `drone_02_68688_108004` | 16.6 s | stored | 12 |
| `drone_03_68690_108003` | 16.5 s | stored | 12 |
| `drone_04_68690_108008` | 10.9 s | stored | 7 |
| `drone_05_68690_108007` | 10.3 s | **WITHHELD** | 5 |

**`drone_05` is the one to show a judge.** The privacy gate fires on it for real
(`person-signal conf=0.55`, stable across four runs). It is **withheld from
storage** — no thumbnail, no archive row — but **still analysed**: 5 buildings
reach the rank and the ballot. Analysis and storage are separate decisions, and
the gate blocks only the second. The log records the withhold without the filename.

Two of these tiles now legitimately return **0 buildings**: they are over open
ground, and the footprint layer says so. The card reads `no buildings in frame
(open ground)`. That is a fix, not a failure — the pipeline used to invent twelve
rectangles there and label them with real street addresses.

## 5. Read the result

- **RANK** — ordered worklist. Each card shows one `PRIORITY` figure; hover `how`
  for the full arithmetic, which reconciles by hand. A mismatch renders red.
- **DISPATCH** — stops grouped by agency, with lat/long under each address (click
  to copy). `Nav` draws the route, centres on it, rings the destination, and opens
  turn-by-turn that avoids blocked roads. The `#1` stop of each agency pulses.
  Reassign with the agency dropdown; edits now persist across polls.
- **ARCHIVE** — searchable thumbnails. The pin sits on the building the caption
  describes, and the caption says which of the frame's buildings that is.
- **FLIGHT** — next survey area over the least recently seen ground, sized in
  metres from a stated sensor model. `Replan flight` re-tasks it.
- **Footer** — every figure measured on this box, scoped to the current grading
  settings, with sample size shown (`n=4`) so a thin median cannot pass for a fat
  one.

Zoom past z14 and the roads carry **street names**, rendered from committed SDF
glyphs — the same names the turn-by-turn reads out. Closed roads are labelled red.

## 6. Reset to clean again

```bash
ssh spark 'cd ~/fl/service && FIRSTLIGHT_AOI=bay FIRSTLIGHT_DATA=$PWD/data \
  PYTHONPATH=$PWD .venv/bin/python tools/reset_clean.py'          # dry run
```

Add `--apply` to act. It clears tiles, buildings, archive, operator plan edits,
road closures, availability, and the analyzed/withheld/watch/thumbs directories.

It keeps datasets, basemap tiles, glyphs, model weights — and the decision log,
always. There is deliberately no flag to clear the log: it is append-only enforced
by SQL triggers, and an audit trail a maintenance script can erase is not an audit
trail. The reset appends its own entry. For a demo against an empty log, point
`FIRSTLIGHT_DATA` at a fresh directory instead.

Restart the service with:

```bash
ssh spark 'bash ~/fl/service/tools/restart.sh < /dev/null'
```

## 7. Re-measure anything

On the box, under `service/tools/`:

| Tool | What it measures |
|---|---|
| `measure_models.py` | per-model decode tok/s and VL call latency |
| `measure_budget.py` | per-tile p50 across VL budgets, to pick the cap |
| `measure_tiles.py` | serial vs concurrent grading on real tiles |
| `measure_for_deck.py` | every number the slide deck asserts |
| `pick_test_images.py` | a fresh spread of test frames from the NOAA cache |
| `profile_status.py` | which contributor makes `/api/status` slow |
| `reset_clean.py` | back to a clean slate |

`measure_for_deck.py` rewrites `demo/measured.json`, which is where the deck's
figures come from — so the slides cannot drift from the box again.
