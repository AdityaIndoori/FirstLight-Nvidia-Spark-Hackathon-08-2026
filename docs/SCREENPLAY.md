# FIRST LIGHT — demo screenplay

**Target 3:30. Structure follows the recommended flow.**
Every number spoken here was measured on the box and is reproduced in
`demo/measured.json`. Anything a judge can check on screen is marked ✅.

Two columns: what you **do**, what you **say**. Say the words in the right column
close to verbatim — they are timed, and every claim in them is one the box can
defend.

---

## Before you hit record

```bash
# 1. Clean slate, so the console starts empty and fills in live
ssh spark 'cd ~/fl/service && FIRSTLIGHT_AOI=bay FIRSTLIGHT_DATA=$PWD/data \
  PYTHONPATH=$PWD .venv/bin/python tools/reset_clean.py --apply'
ssh spark 'bash ~/fl/service/tools/restart.sh < /dev/null'

# 2. Confirm all three model servers are warm (first call after boot is slow)
ssh spark 'for p in 8000 8001 8002; do curl -s -m 3 http://127.0.0.1:$p/v1/models \
  >/dev/null && echo "$p ok"; done'
```

- Browser at `http://10.10.0.3:8081/`, hard-reload once, window at 1600×1000.
- **Leave the operator name blank.** You will type it on camera — it is the
  cheapest possible proof that every edit is attributed.
- Have the test-image folder open in Explorer, sorted by name.
- Zoom the map to the AOI but do **not** pre-zoom past z14; letting the street
  names appear as you zoom is a beat.

---

## 1 · Team — 0:00–0:25

| Do | Say |
|---|---|
| Camera on you. No screen yet. | "We're **FIRST LIGHT** — three builders, one weekend, one DGX Spark. |
| | I took perception and data: the privacy gate, damage grading, and the county data joins. |
| | *[B]* took the decision layer: the uncertainty ballot, agency tasking and containment. |
| | *[C]* took the operator console and everything you're about to see on screen. |
| | We split by contract, froze the interfaces on Friday night, and integrated Sunday morning." |

> Swap the bracketed names. If you are presenting solo, say "I built the
> perception and data layer; my teammates built the decision layer and the
> console" — do not claim all three.

---

## 2 · Elevator pitch — 0:25–1:05

| Do | Say |
|---|---|
| Cut to the empty console. Let it sit — it is visibly empty. | "After Hurricane Helene, roughly **74% of cell towers** in the worst-hit counties failed. That is exactly when drone teams are collecting hundreds of gigabytes of damage imagery that can reach no cloud. |
| Gesture at the empty stage cards. | FIRST LIGHT is a **county-owned box** that turns a drone downlink into a ranked, navigable rescue plan with **zero connectivity**. Aerial photos in, door-by-door rescue plan out. |
| | Five models sit co-resident in 128 GB of unified memory — a vision-language grader, a 30-billion-parameter model that cross-examines it, a planner, an embedder and a person detector. |
| | The box is **offline because policy says so, not because luck held**. And every number you're about to see is measured on the machine behind me, not quoted from a datasheet." |

**✅ On screen:** empty stage cards, `needs 0 / have 0` per agency, footer reading
`per-tile not measured yet`.

> The empty footer is doing work here. It says *not measured yet* rather than a
> fake zero, which sets up everything that follows.

---

## 3 · Live demo — 1:05–2:05

**One continuous take. Do not cut.** The whole batch settles in **27 seconds** ✅,
so you narrate over live inference rather than over an edit.

| Do | Say |
|---|---|
| Type `R. Alvarez` in the operator field. | "First, my name. Nothing in this system can be changed without it, and every change is written to an append-only log." |
| Click **Upload drone images**, select **all** files in the test-image folder, Open. | "Six real frames from the NOAA survey of Panama City after Hurricane Michael. I'm selecting the images *and* their location sidecars — these frames carry no GPS in EXIF." |
| Let the cards mount. Point at the counters. | "Six tiles, in flight together. Each card counts up so you can see it working — that is real inference, not a progress bar." |
| **Wait.** Say nothing for ~5 s. Let the first card land. | — |
| First card settles. | "There's the first: **14 buildings outlined, 5 severe.**" |
| Point at the amber card as it settles. | "And there's the one I want you to see. That tile is **withheld from storage** — the person detector fired, confidence 0.55. |
| Tap the withheld card so its stages show. | No thumbnail, no archive row, not searchable. **But it was still analysed** — five buildings from that frame are in the rank list right now. |
| | A person in frame is rescue signal. Withholding it from *grading* would throw away the exact information triage needs. So the gate guards **storage**, not analysis. Those are two different decisions and we made them separately." |
| Point at two cards reading `no buildings in frame (open ground)`. | "Two of these say *no buildings in frame*. That's the footprint layer telling the truth about woodland — earlier this week that path invented twelve rectangles and gave them real street addresses. We'd rather return nothing than invent a door to kick." |
| Footer. | "Footer: **10.8 seconds per tile, p50, n equals 6** — measured, with the sample size shown." |
| Click **RANK**. | "Ranked worklist. One priority number per card — hover, and you get the arithmetic." |
| Hover the `how` hint on the top card. | "Severity, staleness, vulnerable density, doubt. Multiply the four and you get the number on the left — and if they ever *don't* multiply out, the card says so in red. Bring a calculator." |
| Click **DISPATCH**. | "Grouped by agency, with coordinates under every address, because half of these have no street name a crew can drive to." |
| Click **Nav** on Fire #1. | "Nav draws the route, centres on it, rings the destination, and gives turn-by-turn that **avoids blocked roads** — and it tells you when the last 50 metres leave the road network, instead of pretending a driveway is a street." |
| Zoom in two notches. Street names appear. | "Street names, rendered offline. Same names the turn-by-turn reads out." |

**✅ Verified this exact run:** 6 tiles / 27 s wall / 1 withheld on real
person-signal / 26 buildings / tally 7 none, 12 minor, 6 major, 1 destroyed.

> **If a tile is slow on stage:** say *"that's a real 2-second-per-crop
> vision-language call, eight crops a tile"* and keep talking. Do not apologise
> for latency you have already published.

---

## 4 · How we built it — 2:05–3:10

Stay on the console; switch to the memory slide only for the middle third, then
come back.

> **Getting there:** the deck reveals sub-steps within a slide, so arrow-right
> advances a *step*, not a slide. Verified: **8 presses** from the hero lands on
> `05 / 7`, "Every model fits in 128 GB". Better on stage — open the deck in a
> second window already parked on that slide and alt-tab, so you never count
> keypresses on camera.

| Do | Say |
|---|---|
| Console still up. | "Architecture. Ingest is a watch folder and an HTTP door, both hitting one pipeline: **privacy gate → damage grading → vulnerability join → uncertainty ballot → archive**. FastAPI, SQLite, vanilla-JS console, MapLibre. No cloud SDK anywhere in it." |
| | "Five models, and none of them does another's job. **Nemotron Nano 12B v2 VL** grades each building crop 0 to 3 with guided JSON. **Nemotron 3.5 Lightning 30B** never sees pixels — it votes eight times on the grader's own caption, and the spread becomes a *doubt* column. **Nano 9B** drafts the agency tasking. **BGE-small** embeds captions for search, pinned to **CPU** — with three vLLM pools resident the GPU allocator is full and it OOMs. And a VisDrone YOLO guards storage." |
| Switch to the memory slide. | "The hardware shaped every one of those choices. 128 GB unified memory is why five models are co-resident instead of swapped: **80.1 GB of weights** — Lightning 43, the planner 21, the VL grader 15 — plus KV pools, **124 of 128 GB** on that gauge. That is the whole reason this is a car-park box and not a rack." |
| | "Three real bottlenecks. First: per-tile latency was **33 seconds** against a 10-second budget. The grader was issuing twelve vision calls **strictly serially** while the GPU sat with spare batch width. We made them concurrent, then measured the knee — 17.5 seconds at two lanes, 12.6 at four, 11.5 at eight, flattening right where the captioner's own `max-num-seqs` is set. **33 down to 10.8.**" |
| | "Second: the privacy gate was crashing under exactly the load you just watched. Ultralytics mutates the model during inference, so six concurrent uploads raced and two tiles came back as detector errors. Because the gate **fails closed**, working imagery was silently becoming unstorable — a bug that *looks* like the gate doing its job. One lock at the right boundary, and a test that fails if two threads are ever inside the detector at once." |
| Back to console. | "Third, and this is the one I'd want a judge to poke at: the box was **contradicting its own slide deck** on every number. Deck said 4.2 seconds a tile; the box said 33. Said 240 watts; the box draws **65 median, 85 peak**. So we wrote the measurement harness first, made the deck read from its JSON output, and scoped the latency percentile to the grading settings that produced it — change the VL budget and it starts a fresh measurement instead of averaging the old build in forever." |
| | "Tradeoffs we took deliberately: eight vision calls a tile, not forty — the rest get a **labelled** pixel-statistic stub, so you can always tell what looked and what guessed. Upload deliberately does **not** deduplicate, because the gate must re-run rather than trust a cached verdict. And the decision log is append-only enforced by SQL triggers — I know, because it refused my own reset script." |

**✅ Checkable:** `service/tools/measure_budget.py` reproduces the lane sweep;
317 tests pass; `probe_append_only.py` shows `DELETE` and `UPDATE` both abort.

---

## 5 · So what — 3:10–3:35

| Do | Say |
|---|---|
| Console up, rank list visible. | "So what. Today, the morning after a storm, this is four people with a paper map and a radio, and the imagery sits on an SD card until connectivity comes back. |
| | This box does it in ten seconds a frame with the network unplugged, and it shows its work: every grade labelled with who made it, every number measured on the machine, every operator edit signed and unerasable. |
| | We didn't build a dashboard over an API. We put five models in 128 gigabytes, ran them against real post-hurricane imagery, and **published the numbers that made us look slow** until we fixed them. |
| Beat. | That's a box a county can own. Thank you." |

---

## Judge questions, with answers the box supports

**"Is the person detection actually any good?"**
Recall is measured on 100 held-out real aerial tiles through the tiled path, not a
single downscaled pass — a person at survey altitude is about 5 px in a 640 px
frame, which is why we run overlapping 1280 px crops and take the union. The
number is published, and the threshold was set *from* that measurement.

**"Why do three cards have the identical priority?"**
Because they are genuinely identical inputs: same street, same damage class, same
staleness, same doubt. The formula is deterministic, so equal inputs give equal
outputs — that is a feature, not a rounding artifact, and the tie is broken by
footprint ID so the ordering is stable across re-ranks rather than shuffling. Do
not improvise here: say "same inputs, same score", show the hover on two of them,
and move on.

**"Why is a major-damage building above a destroyed one in the rank?"**
Be straight: doubt is a multiplier, so an uncertain class-2 can outrank a
confident class-3. That is deliberate for *recon* priority — uncertainty means
send someone to look. Say so, then point at the Fire list, where
`collapse search, possible entrapment` is assigned on **severity**. If they push,
concede it is a live design tension and that the two priorities want to be two
columns.

**"What is stubbed?"**
Say it plainly: the agency tasking is a labelled deterministic rule set, not the
planner model, and it says `stub-rules-v1` on screen. Pixel-statistic grades are
labelled `stub-pixelstat-v1`. Nothing on screen claims a model that did not run.

**"Prove it's offline."**
Unplug the ethernet and upload another tile. Nothing in the pipeline reaches the
network — tiles, glyphs, footprints and weights are all local.

**"Can you fake this?"**
Hand them the keyboard. Upload their own georeferenced image, or re-upload the
withheld tile and watch the gate reach the same verdict independently.

---

## Recording notes

- **Do not cut during section 3.** The 27-second batch is the credibility of the
  whole demo; an edit there reads as hiding latency.
- Zoom past z14 **on camera** so labels appear as a visible consequence.
- If the browser has stale JS, hard-reload before recording — assets are
  `no-store` now, but a tab opened before the last deploy can still be holding old
  code.
- Total spoken words ≈ 620. At a measured 175 wpm that is 3:32. Read slower than
  feels natural.
