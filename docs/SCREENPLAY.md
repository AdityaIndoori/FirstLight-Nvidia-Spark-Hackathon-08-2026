# FIRST LIGHT — demo screenplay

**Runs 4:34 as written. The recommended flow totals 3:30 — read the timing note
before you record.**
Every number spoken here was measured on the box and is reproduced in
`demo/measured.json`. Anything a judge can check on screen is marked ✅.

Two columns: what you **do**, what you **say**. Say the words in the right column
close to verbatim — they are word-counted, and every claim in them is one the box
can defend. The Q&A at the end is preparation, not script; it is not in the count.

### Timing, counted not guessed

| Section | Words | At 175 wpm | Recommended |
|---|---|---|---|
| 1 Team | 65 | 22 s | 20–30 s ✅ |
| 2 Pitch | 118 | 40 s | 30–40 s ✅ |
| 3 Demo | 231 | 79 s | 45–60 s |
| 4 Build | 275 | 94 s | 60–90 s ✅ |
| 5 So what | 107 | 37 s | 20–30 s |
| **Total** | **801** | **4:34** | **3:30** |

Sections 1, 2 and 4 land inside the guidance. 3 and 5 run long, and the total does
not fit 3:30 — the demo has more measured substance than 3:30 of speech. Two honest
ways to record it:

**A. Ship 4:34.** The guidance calls 3:30 *ideal*, not a cap. Every extra second is
a measured number or a named bug, which is the opposite of padding.

**B. Cut to 3:44** using the four `[CUT-n]` markers in the script, in this order.
Verified by word count: dropping all four (CUT-3 replaced by its one-line
substitute) leaves **656 words → 3:44 at 175 wpm**. That is as tight as this gets
without dropping a measured claim, and it is 14 s over the ideal rather than 63.
1. `[CUT-1]` the open-ground beat. Strongest with a reviewer who knows the old
   behaviour, weakest with a judge who does not.
2. `[CUT-2]` the DISPATCH/Nav narration. Keep the clicks — the ringed destination
   and the turn list read without words.
3. `[CUT-3]` the per-model detail. Say the one-line substitute instead; the
   breakdown is on the memory slide anyway.
4. `[CUT-4]` closer paragraph 2. Ends on "…signed and unerasable. That's a box a
   county can own. Thank you."

Do **not** cut the withheld-tile beat or the 33→10.8 bottleneck. They are the two
moments that separate this from a dashboard.

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
| Type `R. Alvarez` in the operator field. | "First, my name. Nothing changes in this system without it, and every change is logged append-only." |
| Click **Upload drone images**, select **all** files in the test-image folder, Open. | "Six real NOAA frames from Panama City after Hurricane Michael — images plus location sidecars, because these carry no GPS in EXIF." |
| Let the cards mount. | "Six tiles in flight together. Each card counts up, so you're watching real inference." |
| **Wait ~5 s. Say nothing.** | — |
| First card settles. | "**14 buildings outlined, 5 severe.**" |
| Point at the amber card. | "And there's the one to watch. **Withheld from storage** — the person detector fired at 0.55. No thumbnail, no archive row, not searchable. |
| Tap it so the stages show. | **But it was still analysed.** Five buildings from that frame are in the rank right now. A person in frame is rescue signal — the gate guards **storage**, not analysis. **98% recall on 50 held-out tiles**, one false clear, and we'll name it." |
| `[CUT-1]` Point at the two `open ground` cards. | "Two say *no buildings in frame*. That's woodland, and the footprint layer saying so. This path used to invent twelve rectangles and give them real street addresses." |
| Footer. | "**10.8 seconds a tile, p50, n=6.** Measured, sample size shown." |
| Click **RANK**, hover `how` on the top card. | "One priority per card. Hover: severity, staleness, vulnerable density, doubt — multiply the four and you get the number. If they ever don't, the card says so in red." |
| `[CUT-2]` Click **DISPATCH**, then **Nav** on Fire #1. *(keep the clicks, drop the words)* | "Grouped by agency, coordinates under every address. Nav routes it, rings the destination, and gives turn-by-turn that avoids blocked roads — and says when the last 50 metres leave the road network." |
| Zoom in two notches. | "Street names, rendered offline. Same names the turn-by-turn reads out." |

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
| Console still up. | "One pipeline: **privacy gate → damage grading → vulnerability join → uncertainty ballot → archive**. FastAPI, SQLite, vanilla JS, MapLibre. No cloud SDK anywhere in it." |
| `[CUT-3]` *(if cutting, say only: "Five models — a VL grader, a 30B that cross-examines it, a planner, an embedder, a person detector.")* | "Five models, none doing another's job. **Nano 12B VL** grades each crop 0–3 with guided JSON. **Lightning 30B** never sees pixels — it votes eight times on the grader's own caption, and the spread becomes the doubt column. **Nano 9B** drafts tasking. **BGE-small** embeds captions, pinned to **CPU**, because with three vLLM pools resident the GPU allocator is full and it OOMs. And a VisDrone YOLO guards storage." |
| Switch to the memory slide. | "The hardware chose that shape. **80.1 GB of weights** — Lightning 43, planner 21, VL grader 15 — plus KV pools: **124 of 128 GB**, co-resident, nothing swapping. That's why this is a car-park box and not a rack." |
| | "Two bottlenecks worth naming. Per-tile was **33 seconds** against a 10-second budget: the grader issued its twelve vision calls **serially** while the GPU had spare batch width. Concurrent, then measured the knee — 17.5 s at two lanes, 12.6 at four, 11.5 at eight, flattening exactly where the server's own `max-num-seqs` sits. **33 down to 10.8.**" |
| | "And the privacy gate was crashing under the load you just watched. Ultralytics mutates the model mid-inference, so six concurrent uploads raced — and because the gate **fails closed**, working imagery silently became unstorable. A bug that *looks* like the gate working." |
| Back to console. | "Which is the real theme: the box was contradicting its own deck on every number. Deck said 4.2 seconds a tile, box said 33. Said 240 watts, box draws **65**. So we wrote the measurement harness first and made the deck read from it." |

**✅ Checkable:** `service/tools/measure_budget.py` reproduces the lane sweep;
318 tests pass; `probe_append_only.py` shows `DELETE` and `UPDATE` both abort.

---

## 5 · So what — 3:10–3:30

| Do | Say |
|---|---|
| Console up, rank list visible. | "So what. Today this is four people, a paper map and a radio, and the imagery sits on an SD card until the towers come back. |
| | This box does it in **eleven seconds a frame with the network unplugged** — and it shows its work: every grade labelled with who made it, every number measured on the machine, every operator edit signed and unerasable. |
| `[CUT-4]` | We didn't wrap a dashboard round an API. We put five models in 128 gigabytes, ran them on real post-hurricane imagery, and **published the numbers that made us look slow** until we'd fixed them. |
| Beat. | That's a box a county can own. Thank you." |

---

## Judge questions, with answers the box supports

**"Is the person detection actually any good?"**
Measured, don't hand-wave: **98% recall on 50 held-out person tiles** through the
tiled 1280 px / 20 %-overlap path at conf 0.50, with 50 clear tiles alongside —
withhold precision 86%, gate latency 59 ms p50. Say the failure out loud: **one
false clear**, and we know which tile it is. A person at survey altitude is about
5 px in a downscaled 640 px frame, which is why we scan overlapping full-res crops
and take the union rather than one downscaled pass. Reproduce with
`scripts/gate_eval.py --tiles data/gate_eval/tiles --labels data/gate_eval/labels.json`.

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

**"What is stubbed?"** — *and volunteer this if nobody asks*
Say it plainly: the agency tasking is a labelled deterministic rule set, not the
planner model, and it says `stub-rules-v1` on screen. Pixel-statistic grades are
labelled `stub-pixelstat-v1`. Nothing on screen claims a model that did not run.

The tradeoffs behind that are deliberate and worth saying in the same breath:
**eight** vision calls a tile, not forty — the rest take the labelled stub, so you
can always tell what looked from what guessed. Upload deliberately does **not**
deduplicate, because the gate must re-run rather than trust a cached verdict from a
possibly different build. And the decision log is append-only enforced by SQL
triggers — I know, because it refused my own reset script.

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
- **801 spoken words as written = 4:34 at 175 wpm; 656 = 3:44 with all four cuts.**
  Counted, not estimated. Rehearse against a stopwatch once: if you land under
  4:00 uncut you are rushing the two beats that matter.
- Also fix the header numbers if you edit any spoken line — the counts above are
  the only reason the cut list can be trusted.
