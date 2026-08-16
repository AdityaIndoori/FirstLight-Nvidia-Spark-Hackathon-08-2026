# Data provisioning runbook

Everything FIRST LIGHT needs from the network, in the order to fetch it, with
the command, the licence, the expected size and the way to check it landed.

**Run this while a link exists.** The box is designed to work at zero
connectivity, which only holds if the local store was filled first. Every
script here is resumable, so an interrupted run continues rather than starting
over, and every script verifies what it got rather than trusting a 200.

All sizes and counts below were measured on the Spark (`gn100-2714`) on
2026-08-16. Where a number is an estimate it says so.

---

## 0. The order, if the network is about to go away

```bash
cd ~/fl/service

# 1. The eval set. Smallest, and it is the only thing blocking A5's published
#    gate-recall number. About 375 MB of source zips, 16 MB kept.
./.venv/bin/python scripts/fetch_visdrone_eval.py --limit 100

# 2. The basemap. Without it the map is a dark rectangle with polygons on it.
#    z12-16 first, because that is a legible map in 3 MB and 2.5 minutes.
./.venv/bin/python scripts/fetch_tiles.py --aoi pinellas --zoom 12-16

# 3. County GIS. Refuses to write an empty file and exits non-zero on failure,
#    so a silent wrong-county fetch cannot happen again. See section 3.
./.venv/bin/python scripts/fetch_aoi.py

# 4. Deep zoom, only if there is time and bandwidth left. z17-18 is where a
#    crew can see roofs and driveways, and it is 93% of the tile count.
./.venv/bin/python scripts/fetch_tiles.py --aoi pinellas --zoom 17-18 --yes
```

Anything left unfetched degrades a named feature rather than breaking the
system. Section 5 says exactly what.

---

## 1. Basemap tiles: `scripts/fetch_tiles.py`

Writes `web/tiles/{z}/{x}/{y}.png`, which is exactly where `web/js/map.js`
looks. The MapLibre style is local-tiles-only: no style URL, no sprite server,
no glyph server, no CDN fallback. When the directory is empty the legend prints
`basemap tiles not cached`. `map.js` probes one z12 tile at the centre of the
AOI served by `GET /api/status`, so **always include z12** or the legend will
report a complete cache as absent.

Satellite tiles go to `web/tiles/sat/{z}/{x}/{y}.png` under `--sat`, and are
**off by default**.

### Commands, one per AOI preset

The three presets need three separate caches. Switch AOI with
`FIRSTLIGHT_AOI`; the tile cache does not follow automatically, and a cache for
the wrong county paints the wrong county.

```bash
# Pinellas County FL (Hurricane Milton) - the default AOI and the demo AOI
./.venv/bin/python scripts/fetch_tiles.py --aoi pinellas --zoom 12-16
./.venv/bin/python scripts/fetch_tiles.py --aoi pinellas --zoom 17-18 --yes

# Bay County FL (Hurricane Michael, Panama City) - the AOI the xView2 cls path
# needs, because it is the only one with a true pre/post aerial pair from one
# source. Fetch this cache if you intend to demo six-channel cls grading.
./.venv/bin/python scripts/fetch_tiles.py --aoi bay --zoom 12-16
./.venv/bin/python scripts/fetch_tiles.py --aoi bay --zoom 17-18 --yes

# Sarasota County FL (Milton) - the county with 34,620 real building footprints
./.venv/bin/python scripts/fetch_tiles.py --aoi sarasota --zoom 12-16
./.venv/bin/python scripts/fetch_tiles.py --aoi sarasota --zoom 17-18 --yes

# See the cost before spending the bandwidth. Always do this first.
./.venv/bin/python scripts/fetch_tiles.py --aoi bay --zoom 12-18 --dry-run
```

`--out` defaults to `web/tiles`. Pass a different `--out` per preset if you
want all three resident at once; otherwise the caches overlay in one tree and
`MANIFEST.json` records every run, so you can tell what is in there.

### Measured: tile counts and bytes per zoom

Pinellas, provider `carto_dark`, measured:

| Zoom | Tiles | Bytes on disk | Note |
|---|---|---|---|
| 12 | 4 | 58.1 KB | the zoom `map.js` probes; never omit it |
| 13 | 9 | 108.2 KB | |
| 14 | 25 | 281.4 KB | |
| 15 | 72 | 814.3 KB | |
| 16 | 240 | 1.7 MB | legible street map |
| 17 | 900 | 3.6 MB | |
| 18 | 3,481 | 8.2 MB | roofs and driveways. Cheaper per tile than z16 because a dark basemap at high zoom is mostly flat fill, which PNG compresses hard |
| **12-16** | **350** | **3.0 MB measured, 146 s at `--rate 12`** | the practical minimum |
| **12-18** | **4,731** | **14.8 MB measured (15,516,939 bytes), 33 min total at `--rate 14`** | the full demo cache, fetched with 0 failures and 0 refusals |

Planned counts for the other two presets, from `--dry-run` (tile counts are
exact arithmetic; bytes are estimated at the measured 11.7 KB per tile):

| AOI | z12-16 tiles | z12-18 tiles | z12-18 est bytes |
|---|---|---|---|
| pinellas | 350 | 4,731 | 54.1 MB |
| bay | 523 | 7,664 | 87.7 MB |
| sarasota | 365 | 5,279 | 60.4 MB |

Satellite, Pinellas z12-15, measured: **110 tiles, 13.8 MB**. Esri serves JPEG
and the script re-encodes to PNG so the `.png` path `map.js` requests holds
honest bytes; photographic tiles balloon about 6x doing that, which is why
satellite is roughly 10x the size of the tactical basemap per tile.

### Providers, licences and attribution

The script writes `ATTRIBUTION.txt` into the output directory on every run.
Shipping a tile cache without attribution is a licence problem on stage, and
the file travels with the tiles wherever they are copied.

| Provider | Default | Licence and attribution |
|---|---|---|
| `carto_dark` | **yes**, tactical | Basemap tiles (c) CARTO, style `dark_all`. Map data (c) OpenStreetMap contributors, ODbL 1.0, <https://www.openstreetmap.org/copyright>. CARTO terms: <https://carto.com/legal/> |
| `carto_dark_nolabels` | no | Same as above, style `dark_nolabels`. Useful because there is no glyph server offline, so labels baked into the raster are the only labels there are, and some rooms prefer them off |
| `osm` | no, **and it does not work from here** | Map data and tiles (c) OpenStreetMap contributors, ODbL 1.0. Tile policy: <https://operations.osmfoundation.org/policies/tiles/> |
| `esri_imagery` | no, `--sat` only | Esri World Imagery. Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community. Terms: <https://www.esri.com/en-us/legal/terms/full-master-agreement> and the item page <https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9>. Permitted for internal display and non-commercial use with attribution; caching for offline use is a grey area under the master agreement, which is why it is opt-in and why the tactical basemap is the default. If the licence matters more than the imagery, ship without `--sat`: cut-list item 7 already drops the satellite layer |

**Finding: `tile.openstreetmap.org` refuses this client.** Measured from the
Spark, every request returns HTTP **200** with a body of exactly **6,987
bytes** and the header `x-blocked: Access denied. See
https://operations.osmfoundation.org/policies/tiles/`. The body is a valid PNG
that says so, so a magic-byte check alone would happily cache 4,731 copies of a
refusal notice. The script detects this two ways, gives up after 5 consecutive
refusals rather than hammering a volunteer service, and writes zero files.
Verified: `--provider osm` exits 1 with `ABORTED: provider refused this client`
and an empty output directory. Do not use `osm`; `carto_dark` serves the same
OpenStreetMap data under the same ODbL terms and is dark, which is what the
tactical basemap wants anyway.

### Safety rails

- **Pre-flight count.** Prints tiles and estimated bytes per zoom before a
  single request, plus the wall-clock estimate at the chosen rate.
- **Cap.** Refuses to start above 20,000 tiles without `--yes`. Verified: bay
  z12-19 is 30,002 tiles and exits 2 with the count and the advice.
- **Disk check.** Refuses if free space is under twice the estimate.
- **Rate limit.** `--rate` requests per second, default 5. Retries with
  exponential backoff, 3 attempts. A 404 is treated as a real answer, not
  retried, and reported as `not published`.
- **Resumable.** Tiles already on disk are skipped. Verified: re-running
  `--zoom 12-14` over a complete cache reports `38 already cached` in 0.0s and
  makes no requests.
- **Atomic writes.** Every tile lands via a `.part` file and `os.replace`, so
  an interrupted run never leaves a truncated tile that a resume would skip.

### Verify it landed

```bash
# Sample the cache for real PNGs, without re-fetching anything
./.venv/bin/python scripts/fetch_tiles.py --verify-only --zoom 12-18

# Count and size
find web/tiles -name '*.png' | wc -l
du -sh web/tiles

# The one tile map.js actually probes (z12, AOI centre). If this is missing,
# the legend says "basemap tiles not cached" no matter what else is cached.
ls -la web/tiles/12/1106/1718.png     # Pinellas, 18,111 bytes measured
```

Expected for the full Pinellas z12-18 cache, as it stands on the box now:
**4,731 tactical files, 14.8 MB**, reported as `cached files: 4731, sampled 40`
with no `BAD` lines and exit 0. For the z12-16 minimum it is **350 files,
3.0 MB**. The verifier checks PNG magic bytes AND flags byte sizes that repeat
across most of the sample, because one canned image served for every request is
exactly what a refusal looks like and it is a valid PNG.

`MANIFEST.json` in the output directory records `aoi`, `aoi_name`, provider,
zoom list, per-zoom counts and bytes, and a timestamp for every run. Compare
its `aoi_name` against `aoi_name` from `GET /api/status` before a demo: if they
disagree, the cache is for a different county than the one configured.

---

## 2. Privacy-gate eval set: `scripts/fetch_visdrone_eval.py`

Assembles the held-out set A5 needs: **100 real aerial tiles with a boolean
person label per tile**. This is the only thing blocking a published
gate-recall number.

It has to be real imagery. The synthetic person fixture in `make_demo_kit.py`
produces **zero detections** from the real detector weights at two scales,
which is correct behaviour from a VisDrone-trained model and a defect in the
fixture. A recall number measured on drawn people would be worse than no
number, so **this script never generates a synthetic tile**: if no source is
reachable it exits non-zero with what it tried.

```bash
./.venv/bin/python scripts/fetch_visdrone_eval.py --limit 100
./.venv/bin/python scripts/fetch_visdrone_eval.py --limit 100 --keep-cache  # keep the 375 MB of zips
```

Output, exactly the format `scripts/gate_eval.py` already expects:

```
data/gate_eval/tiles/*.jpg          100 real VisDrone frames
data/gate_eval/labels.json          {"<filename>": true|false}
data/gate_eval/PROVENANCE.json      per image: source, split, original filename,
                                    original annotation path, annotation basis, licence
```

### Source, and the one actually used

| Source | Reachable from the Spark | Used |
|---|---|---|
| Ultralytics assets release: `VisDrone2019-DET-val.zip` (81,638,851 bytes) and `VisDrone2019-DET-test-dev.zip` (311,251,787 bytes) | **yes**, HTTP 200, full download in 2.1 s and 7.7 s | **yes, this is the one used**. Carries the CANONICAL VisDrone annotation format, not a re-export |
| `huggingface.co/datasets/banu4prasad/VisDrone-Dataset` | yes, per-file HTTP 200 | fallback only. Same imagery re-exported to YOLO txt, which **drops the score column**, so ignored regions cannot be excluded. Recorded in `PROVENANCE.json` as a weaker annotation basis if it is ever used |
| `aiskyeye.com` / `github.com/VisDrone/VisDrone-Dataset` (canonical) | host answers 200, but the image payloads sit behind Google Drive interstitials that are not a scriptable GET | no |

**Annotation basis.** Canonical VisDrone is one comma-separated line per
object: `x,y,w,h,score,category,truncation,occlusion`. Category **1 is
pedestrian**, **2 is people**. `has_person` is true when any object has
category in {1, 2} **and score == 1**. Score 0 marks an *ignored region*, which
is the part of the frame the benchmark tells you not to score, so counting it
would put a person label on a tile whose person the annotators disclaimed.

**Measured census of the sources:**

| Split | Frames | With a person | Without |
|---|---|---|---|
| validation | 548 | 531 | **17** |
| test-dev | 1,610 | 1,267 | **343** |

Validation alone **cannot balance the set**: 17 negatives against a target of
50. That is why the script pulls test-dev as well, in that order, and stops
early once the balance is satisfiable so a small `--limit` never pays for the
311 MB split.

**Frame selection is spread across flights.** VisDrone frames come from
continuous drone flights, so the first N filenames in sorted order are often
near-duplicates of one scene, and a recall number measured on fifty frames of
the same intersection is not a recall number. Selection is round-robin across
sequence ids, and it is deterministic, so a resumed run picks the same frames.
The delivered set spans **70 distinct flight sequences** across 100 frames.

### Measured result

```
100 tiles: 50 with a person, 50 without (100 written now, 0 already present, 0 failed)
  15,835,211 bytes of imagery
  sources: visdrone_testdev_ultralytics, visdrone_val_ultralytics
  splits: test-dev, validation
```

Split by source: 26 person + 15 clear from validation, 24 person + 35 clear
from test-dev. An exact **50/50** balance.

### Verify it landed

```bash
ls data/gate_eval/tiles | wc -l                      # expect 100
python - <<'EOF'
import json
L = json.load(open("data/gate_eval/labels.json"))
P = json.load(open("data/gate_eval/PROVENANCE.json"))
print(len(L), "labels,", sum(L.values()), "person tiles")
print("provenance covers every label:", set(L) == {r["filename"] for r in P["images"]})
EOF
```

Expect `100 labels, 50 person tiles` and `True`. Then measure the gate, which
is the whole point of the set:

```bash
./.venv/bin/python scripts/gate_eval.py \
  --tiles data/gate_eval/tiles --labels data/gate_eval/labels.json \
  --sweep 0.15,0.2,0.25,0.3
```

**Measured on this set, on the box:**

| conf | Person recall | Withhold precision | False clears | False withholds |
|---|---|---|---|---|
| 0.15 | 98.0% | 68.1% | 1 | 23 |
| 0.20 | 98.0% | 69.0% | 1 | 22 |
| **0.25** | **98.0%** | **76.6%** | **1** | **15** |
| 0.30 | 98.0% | 79.0% | 1 | 13 |

Latency p50 57 ms, p95 165 ms, mean 1.16 tiles scanned per image through the
tiled 1280 px / 20 % overlap path. Recall does not move across the sweep, so
0.25 is the right threshold: it buys the best precision without costing a
single additional false clear.

The one false clear is `0000026_00000_d_0000024.jpg`: a single truncated
pedestrian, box 36x33 px in a 1360x765 frame (0.11 % of frame area), at y=732
of 765, so it is clipped by the bottom edge with `truncation=1`. That is a
genuinely hard case, not a defect in the set, and it is exactly the residual
the README's privacy claim is bounded by. Publish it.

### Licence

VisDrone benchmark, Lab of Machine Learning and Data Mining, Tianjin
University. Free for **academic and non-commercial research use**. Cite:

> Zhu et al., "Detection and Tracking Meet Drones Challenge",
> arXiv:2001.06303. See also arXiv:2105.02440.

The eval set is used here to measure a privacy control, is not redistributed,
and the mirror is `github.com/ultralytics/assets` release `v0.0.0`. The
Hugging Face fallback mirror additionally declares `cc-by-nc-sa-3.0` on its
dataset card. The full licence string is recorded per image in
`PROVENANCE.json`, because a published recall number has to be able to name its
test set.

---

## 3. County GIS: `scripts/fetch_aoi.py`

Writes `data/datasets/{footprints,parcels,roads,svi}.geojson` and
`facilities.csv`. Owner-name columns are dropped in the loader, at ingest,
before anything downstream can read them. Scrubbing is substring-based, not an
exact-match list, because the field names differ per county and a tuple written
for one county silently passes another county's owner fields straight through.
Each dataset records the columns it actually dropped, so the write-up quotes a
measured number.

### The bug this runbook caught, and what it cost

Worth keeping, because it is the reason the checks below exist. As originally
committed the script hardcoded King County layers,
`services.arcgis.com/.../Building_Outlines_2023` and
`gismaps.kingcounty.gov/.../KingCo_Parcels`, while the AOI had moved to
Pinellas County, Florida. It ran to completion and wrote
`footprints.geojson` and `parcels.geojson` of **45 bytes each**, literally
`{"type":"FeatureCollection","features":[]}`. Counted live with
`returnCountOnly`:

| Query | Count |
|---|---|
| Seattle footprints against the **Pinellas** bbox | **0** |
| Seattle footprints against the **Seattle** bbox | 27,250 |
| King County parcels against the **Pinellas** bbox | **0** |

The endpoints were alive; the geography was wrong. An empty FeatureCollection
reads downstream as "this county publishes none" rather than "the fetch
failed", which is the kind of silence that ends up in a FEMA export.

**Now fixed.** Sources are a per-AOI table keyed by `AOI_NAME`, the script
probes `returnCountOnly` BEFORE paging and refuses to page when a
should-be-populated layer returns zero, it never writes an empty file, it exits
non-zero listing failures, and an unregistered AOI is a hard error naming the
registered ones instead of querying another county with this bbox.

A second and worse defect surfaced with it: `DROP_COLUMNS` was an exact-match
list written for King County, so against Pinellas it silently passed `OWNER1`,
`OWNER2`, `MAILTO`, `OWNADD_1`, `OWNADD_2`, `OWNCITY`, `OWNSTATE`,
`OWNCOUNTRY` and `OWNZIP` straight through into the loaded properties.
Scrubbing is now substring-based (`own`, `taxpay`, `mail`, `grantee`,
`grantor`, `deed`) and each dataset records the columns actually dropped: 9 for
parcels, 7 for facilities.

**Measured after the fix:** parcels **39,166** written with `SITE_ADDRESS`
present, facilities **36**, road closures **0** (expected, see below), zero
owner fields surviving, 36 useful fields kept.

### The Pinellas sources, each counted live against the Pinellas bbox

| What | Endpoint | Count |
|---|---|---|
| Parcels | `https://egis.pinellas.gov/gis/rest/services/WebGIS/Parcels/MapServer/1/query` | **39,166** |
| Fire stations | `PublicWebGIS/General/MapServer/1/query` | 5 |
| Hospitals | `PublicWebGIS/General/MapServer/2/query` | 3 |
| Police stations | `PublicWebGIS/General/MapServer/0/query` | 2 |
| Public elementary schools | `PublicWebGIS/General/MapServer/4/query` | 8 |
| Private schools | `PublicWebGIS/General/MapServer/9/query` | 18 |
| Building footprints | none published by Pinellas | **0 sources** |
| Gray-sky road closures | `RoadClosures/GraySkyRoadClosures_Public/MapServer`, all 12 layers | all **0** |

Three traps in that table:

1. **Parcels is layer 1, not layer 0.** Layer 0 returns HTTP 400 `Invalid or
   missing input parameters`. Layers 3 and 18 of `PublicWebGIS/General` also
   return 400. Enumerate the service and skip failures rather than assuming
   every layer id answers.
2. **Pinellas publishes no building-footprint service**, exactly as planned.
   The footprint tier for this AOI is Microsoft GlobalMLBuildingFootprints, or
   switch to `FIRSTLIGHT_AOI=sarasota` where the county layer has 34,620.
   Sarasota's footprint layer returns 0 against the Pinellas bbox: it is a
   different county, not a fallback.
3. **The gray-sky closure service is up and empty.** All 12 layers answer and
   all return 0 features. That is correct and is now recorded in the source
   table as `expect=0`: it is a live during-the-event feed and there is no storm
   today. B4 will not get real blocked roads from it, so the operator-entered
   `road_block` path is the only source that will have data on stage.

**Roads are still a gap, and it is nobody's current lane.**
`data/datasets/roads.geojson` remains the 926-byte, 5-feature fixture. Real
road geometry for this AOI needs an OSM extract, which no script here fetches.
Any routing number measured today is measured against a 5-edge fixture graph and
must say so.

---

## 4. The librarian allowlist: five datasets, refreshed **by name**

`app/librarian.py` is the only component in FIRST LIGHT that touches the
network at runtime. The agent's entire network surface is one tool with one
enum parameter, `refresh_dataset(name)`.

| Name | Source | Feeds |
|---|---|---|
| `noaa_storm_imagery` | <https://storms.ngs.noaa.gov/> | post-event aerial imagery, the tile source |
| `xview2_labels` | <https://xview2.org/> | damage-grade reference set |
| `ms_building_footprints` | <https://github.com/microsoft/GlobalMLBuildingFootprints> | footprints where a county publishes none, which is most counties and is exactly the Pinellas case above |
| `cms_facilities` | <https://data.cms.gov/provider-data/topics/nursing-homes> | `facility_near` |
| `cdc_svi` | <https://www.atsdr.cdc.gov/place-health/php/svi/index.html> | `vulnerable_density` |

**The agent refreshes these by NAME, never by URL.** That is the containment
property, not a style preference: a name is resolved to a URL inside the
librarian, GET-only, redirects locked to the allowlisted host, with a byte cap
and an atomic swap into the local store. A hijacked agent has no fetch
primitive to point at an address of its choosing, because the parameter is an
enum. Adding a source means adding an entry to `ALLOWLIST`, not passing a URL.

```bash
# by name, the only way
python -c "from app import librarian; print(librarian.refresh('cdc_svi'))"
python -c "from app import librarian; print(librarian.catalog())"
```

`catalog()` reports `last_refreshed` per name, which the HUD dataset strip
shows. Everything works at zero connectivity from the local store; a refresh
is an improvement, never a dependency.

---

## 5. What degrades when a dataset is missing

Nothing here crashes the system. Each gap costs one named, visible thing.

| Missing | What degrades | What the operator sees |
|---|---|---|
| `web/tiles` | The basemap. Damage polygons, routes, facilities and the flight box still draw, on a dark background with no streets. Navigation instructions still compute; they are just harder to follow visually | Legend prints `basemap tiles not cached, geometry still draws on the dark background` |
| `web/tiles/12/...` only | Nothing real, but the probe fails and the legend lies about a cache that is present | The stale-looking "not cached" note, with tiles on disk |
| `web/tiles/sat` | The satellite basemap toggle. This is cut-list item 7 and costs no gate | Satellite button disabled with a tooltip naming the missing directory |
| `data/gate_eval` | **A5's published gate-recall number, and gate 9.** The privacy claim's residual becomes unbounded, and the README cannot say what the recall is. The gate itself still runs and still withholds | No recall figure to publish. Do not substitute a synthetic one |
| County footprints | Building identity and street addresses in the rank panel. The rank still ranks, but rows read as raw IDs instead of addresses | Rank rows without real address labels |
| County parcels | The `SITE_ADDRESS` join, and the property-value columns the pitch promises to load and then visibly refuse to use | Same as above |
| `roads.geojson` (real) | B4's offline routing falls back to a 5-edge fixture graph, so routes cannot follow real turns, which is the one claim that feature exists to make. **This is the current state**, see section 3 | Routes drawn as dashed approximate connectors rather than solid routed lines |
| Gray-sky closures | Real blocked roads. Operator-entered closures still work and are still banned by name and geometry | No pre-populated closures |
| `cdc_svi` | `vulnerable_density` falls back to a default, so the vulnerability term stops discriminating between block groups | SVI reads as a default rather than "top 5% nationally" |
| `cms_facilities` | `facility_near`, so EMS assignment loses its strongest signal and the medical-cross markers vanish | No facility proximity in the evidence card |
| `xview2_labels` | Nothing at runtime: the weights are already on the box. It is the reference set for grading accuracy | B8-c2 has no reference |
| `noaa_storm_imagery` | Post-event imagery to ingest. The demo tile pool is local, so the demo survives | Nothing, unless you wanted fresh imagery |

---

## 6. Disk budget

| What | Size | Note |
|---|---|---|
| `web/tiles` Pinellas z12-16 | 3.0 MB measured | the practical minimum |
| `web/tiles` Pinellas z12-18 | 14.8 MB measured | full demo cache, 4,731 tiles |
| `web/tiles/sat` Pinellas z12-15 | 13.8 MB measured | opt-in, 110 tiles |
| `data/gate_eval/tiles` | 16 MB measured (15,835,211 bytes) | 100 frames |
| VisDrone source zips | 375 MB (392,890,638 bytes measured) | deleted after assembly unless `--keep-cache` |
| `data/datasets` | 28 KB before the GIS fix, a few MB with 39,166 parcels loaded | see section 3 |

Budget about **440 MB of transfer** and **45 MB kept** for a full provisioning
run with satellite, or **~400 MB transfer and 31 MB kept** without. The eval
set dominates the transfer and the tiles dominate the wall clock.
