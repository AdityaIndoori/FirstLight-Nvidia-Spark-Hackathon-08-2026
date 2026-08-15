# FIRST LIGHT

**Offline disaster triage on one NVIDIA DGX Spark. Aerial photos in, ranked rescue plan out — with the room's network left plugged in, because policy keeps this box offline, not luck.**

Team plan for the NVIDIA DGX Spark Hackathon, August 2026. Three builders, one weekend, one box.

---

## 1. The pitch

After Hurricane Helene, roughly 74% of cell towers in the worst-hit counties failed ([FCC report](https://docs.fcc.gov/public/attachments/DOC-406055A1.pdf)). That is exactly when drone teams are collecting hundreds of gigabytes of damage imagery that can reach no cloud.

FIRST LIGHT is a county-owned box that turns a live drone downlink into a ranked, navigable rescue plan with zero connectivity:

`privacy gate -> damage grading -> vulnerability join -> auditable ranking -> next-flight tasking -> FEMA paperwork`

Two things make it a winner rather than a dashboard:

1. **It closes the loop.** The deployed state of the art (TAMU's CLARKE) emits one damage map and stops. We rank the doors, task the next flight, and re-rank on what comes back.
2. **It is contained on purpose.** An agent that tasks drones and drafts federal forms is an agent worth containing. Everything runs as a NemoClaw agent inside an NVIDIA OpenShell sandbox, and we leave the venue network **on** so the judges can watch OpenShell refuse an outbound request in real time, audit line printed on screen.

**Track:** See (streaming perception) + Do (contained agent).
**Bounties targeted:** Nemotron, Nemotron Lightning, NemoClaw + OpenShell.

> **Rules check, first 30 minutes.** We prototyped this concept before the event to de-risk the design. Confirm the event's fresh-code rules at check-in, write all submission code during the event, and disclose the earlier spike wherever the rules ask. The design is proven; the build is the weekend's work. Nobody claims otherwise on stage.

---

## 2. Judge criteria, answered before we write a line

| Criterion | Our answer | What the judge sees |
|---|---|---|
| Completeness: full workflow, no crash | Downlink to export, end to end | Tiles arrive and paint one by one |
| Streaming input (See track) | Frames stream to the box **during** the demo and are processed on arrival | Per-tile end-to-end latency on the HUD |
| Multi-step agent, branching (Do track) | rank -> find stale sectors -> task flight -> re-rank; invalid model output triggers a **self re-prompt with the validation error** before any fallback | Replan beat plus a visible self-correction, labelled "model recovered" vs "stub engaged" |
| NVIDIA stack | Nemotron Nano 9B v2 and Nemotron 3.5 Lightning, both local vLLM; NemoClaw agent inside OpenShell | Status bar names every model; audit panel is live |
| Spark story | **Six models** plus county GIS resident together, zero swap, in a 240 W box a county can own and run in a parking lot with the internet gone. Imagery never leaves the county. | Memory gauge and power figure on the HUD |
| Value and impact | An EOC gets a door-by-door plan the first morning, and every rank is auditable. Property value never enters life-safety ranking. | Rank rows show their formula inputs; the judge checks the arithmetic |
| Usable tomorrow | Locate, Navigate (printable turn-by-turn that avoids blocked roads), one-click FEMA PDA and ICS-213 | The judge drives it unaided |
| Technical depth: retrieval | A real local RAG path — vision caption, tag extraction, embeddings, cosine retrieval, cited answers — with the privacy rule enforced *in the index writer* so withheld imagery is unreachable by construction | Judge types "buildings on fire" and gets pins; then searches for the withheld image and finds nothing |
| Usability: dispatch-shaped output | The plan is grouped by agency with unit counts, not one undifferentiated list — Fire, EMS, Police and Public Works each with numbered routes and printable directions | Judge reassigns a step and watches the numerals and unit totals update |
| Innovation | A closed perception-decision loop nobody deploys, plus a live containment drill | Spoken contrast and a witnessed policy denial |
| Performance | **Measured, never quoted**: speculative-decoding before/after tokens per second, co-resident replan p95, per-tile latency | HUD numbers captioned "measured on this Spark" |

**Name the rival before a judge does.** NVIDIA's VSS blueprint does excellent streaming video analytics, but as shipped it assumes NGC keys, an AI Enterprise licence and datacenter GPUs — a hard fit for an air-gapped county EOC. We ship the decision loop and the federal paperwork VSS does not, fully offline. That is the answer to "why not just use VSS?"

---

## 3. Architecture

```mermaid
flowchart LR
    DL[Drone downlink<br/>streamed frames + SD card fallback] --> GATE[Privacy gate<br/>person detector<br/>withhold by default]
    GATE -->|clear tiles only| SEG[Damage grading<br/>xView2 seg baseline<br/>RescueNet 4 levels]
    GATE -->|withheld| REVIEW[Restricted review queue]
    SEG --> JOIN[Vulnerability join<br/>footprints + CMS + SVI + roads]
    JOIN --> SCORE[Priority scorer<br/>staleness x vulnerability x doubt<br/>append only decision log]
    SCORE --> AGENT[NemoClaw agent - Nemotron Nano 9B v2<br/>flight tasking, hero rationale, ICS-213<br/>reasoning ON for replan]
    AGENT --> LIGHT[Nemotron 3.5 Lightning<br/>k=8 self consistency vote over 50 grades<br/>batch rationales, FEMA rows]
    LIGHT --> UI[Operator console<br/>MapLibre offline, rank, navigate, exports]
    AGENT --> UI
    UI -->|grade flips, road blocks| SCORE
    AGENT -->|next flight box| DL

    subgraph SHELL[OpenShell sandbox - network live, egress denied by policy]
        GATE
        SEG
        JOIN
        SCORE
        AGENT
        LIGHT
    end
```

### Memory and bandwidth, honestly

The Spark is 128 GB unified memory at **273 GB/s**. Bandwidth is the real constraint, not capacity, and we say so before a hardware-literate judge says it for us.

| Component | Footprint | Note |
|---|---|---|
| Nemotron Nano 9B v2, FP8 | ~12 GB | Mamba-hybrid, only 4 attention layers, so the KV cache is unusually small |
| Nemotron 3.5 Lightning, 4-bit (NVFP4-class) weights | ~17 GB | Use whichever quantized build NVIDIA publishes for this box; confirm the exact name at check-in before saying it on stage |
| Damage seg + person detector | ~3 GB | |
| Caption VLM (small vision-language model, FP8) | ~8 GB | Writes the archive caption at stage 3. A text model cannot caption an image; this is why the roster is six models, not four |
| Text embedder (BGE-small class, 384-dim) | ~1 GB | Embeds captions and queries for semantic search |
| County GIS, map tiles, SQLite, archive vectors | ~20 GB | A few thousand 384-dim vectors is under 10 MB — the tiles dominate |
| KV pools, both LLM servers | ~48 GB | vLLM pre-allocates to fill its utilization fraction |
| **Reserved total** | **~109 GB of 128 GB** | Zero swap, everything warm, ~19 GB free |

**Do the arithmetic the way the gauge will.** vLLM does not reserve only the weights; it pre-allocates a KV pool to fill its `--gpu-memory-utilization` fraction. We set Nano to 0.25 (~32 GB total allocation: 12 weights + 20 KV) and Lightning to 0.35 (~45 GB total: 17 weights + 28 KV) — never the ~0.9 default, which would collide on the first request. Add it up: 12 + 17 + 3 + 8 + 1 + 20 + 48 = **~109 GB reserved, ~19 GB free**, and that is exactly what the HUD gauge shows. Bring a calculator; the slide and the gauge agree. Want a smaller number? Cap `--max-model-len` and shrink the fractions — then update this table in the same commit. The rule is that the table and the gauge never disagree. Shared 273 GB/s bandwidth is why every latency number is measured with **both** models warm — a single-model benchmark on this box is a lie by omission.

**Latency targets (so "blows its budget" has a threshold):** replan p95 **under 3 s** with both models warm, per-tile end-to-end under 10 s, hard stub timeout at 10 s.

---

## 4. Model roles, each earning its seat

| Model | Job | Why this model | Bounty |
|---|---|---|---|
| **Nemotron Nano 9B v2** (FP8, vLLM) | Decision-maker: flight tasking with reasoning **visibly on** (thinking trace streams during the replan beat), the one hero rationale, ICS-213 drafting. Cheap structured calls use `/no_think`. | We chose a reasoning model, so we show it reasoning once, where it matters, inside the demo beat | Best Use of Nemotron |
| **Nemotron 3.5 Lightning** (4-bit weights, vLLM) | The job only its throughput unlocks: **k=8 self-consistency voting that computes the `doubt` term** (see the ballot below), plus batch rationales for ranks 2-50, **archive tag extraction across the whole corpus**, and FEMA PDA row fill. Speed buys *calibrated uncertainty* the slow model cannot afford at demo tempo. | 3B active of 30B MoE with multi-token prediction and speculative decoding. We configure it and measure the delta on this box. | Best Use of Nemotron Lightning |
| xView2 first-place seg | Building damage classes | Public weights, RescueNet taxonomy. **Baseline must work**; the RescueNet fine-tune is a Saturday stretch goal behind the same interface, never on the critical path | — |
| Person detector (YOLO class 0) | Privacy gate, before anything else reads a tile | Fast, conservative, and its recall is measurable | — |
| Caption VLM (small, FP8) | Writes one factual caption per **cleared** image at stage 3, constrained to structures, terrain and water | A text LLM cannot see an image. This is the only vision-language model in the stack and it is small on purpose | — |
| Text embedder (BGE-small class) | Embeds captions and search queries, 384-dim, normalized | Search must work offline with no service. 1 GB buys the whole semantic surface | — |
| **NemoClaw + OpenShell** | The planner runs as a NemoClaw agent inside the OpenShell sandbox. Policy: deny all egress, filesystem scoped to `./data`, localhost inference only, rules at binary and destination level, scrollable audit feed in the UI | Out-of-process enforcement the agent cannot override, and we prove it live | NemoClaw + OpenShell |

### The Lightning ballot — what exactly is voted, and why it changes the ranking

The seg model owns the *initial* damage class. Lightning owns the *confidence in it*, and that is what feeds the priority formula.

| Step | Detail |
|---|---|
| Input per building | seg class + seg confidence + join context (footprint area, facility proximity, neighbour classes) |
| Ballot | Lightning samples the severity label **k=8 times at temperature 0.7**, structured-decoded to a single integer 0-3 |
| Output | `voted_class` = the modal label; `vote_agreement` = modal count / 8 |
| Wired into the rank | `doubt = 1 - vote_agreement`, floored at 0.05. So a building the fast model cannot agree with itself about **rises in the ranking**, which is exactly the behaviour we want: uncertainty means send someone to look |
| Eval, two distinct numbers | **Self-agreement**: mean `vote_agreement` across the 50 buildings (how sure Lightning is of itself). **Cross-model agreement**: how often Lightning's `voted_class` matches Nano's single-shot label on the same 50. Both published at gate 9 |

This is why the throughput matters and is impossible to hand-wave: 50 buildings × 8 samples = 400 structured generations inside a demo beat. Nano cannot do that and still stream a thinking trace. The speedup is not a vanity metric; it is the thing that produces the uncertainty column.

**If the measured speedup disappoints on Friday night** (acceptance rate low): drop k from 8 to 4, or vote only on the top 15 most-uncertain buildings. The mechanism survives; only the sample count moves. Never bet the beat on an unmeasured number.

### Two containment beats (this bounty is won on stage, not in prose)

The agent has real tools — `write_flight_plan`, `write_export`, `fetch_context(url)` — registered as NemoClaw tool-calls. Every beat below is the agent invoking one of its own tools and the runtime intercepting it. Nothing is a scripted `curl`; the audit line names `actor=agent`, which is the whole point.

**Beat 1 — positive control, so the denial means something.** Same network stack, two destinations. The agent's inference traffic to `localhost:8000` flows (allowed by destination rule, tokens visibly streaming). The same agent's `fetch_context` to the judge's external server is denied. One screen, one policy, two verdicts: *the network is up and policy discriminates by destination.* Without this control, "denied" could just mean "unplugged".

**Beat 2 — witnessed exfiltration denial.** A judge-pool tile carries a hostile caption instructing the agent to POST the parcel table to an external address. The agent, doing what its context told it, calls `fetch_context` on that address. OpenShell denies it out-of-process: binary, destination, verdict, timestamp, printed in the audit panel. Policy protected the agent from being turned into somebody else's tool.

**Beat 3 — the agent tries to widen its own cage, and cannot.** Still under the injected instruction, the agent attempts to rewrite the egress rule (and to read outside `./data` for the dropped owner-name file). Both denied. Audit shows `actor=agent, action=policy-write, verdict=deny` and `actor=agent, action=fs-read, path=../, verdict=deny`. This is the money shot: enforcement lives outside the process, so a hijacked agent cannot unhook it. **Non-cuttable.**

**Beat 4 — a human can widen it, with a receipt.** The operator needs a USB export. OpenShell denies the unapproved path; a named human approves a scoped policy delta; the audit records who, what and when; the export succeeds. Autonomy inside boundaries, changed only by a person. *Pre-record this one* so a time crunch cannot take it off the table.

**Prove the audit panel is not our own UI lying.** Beside the styled panel we tail OpenShell's raw audit stream in a terminal, same source, append-only. A judge can watch a line appear in the runtime log and then in our UI. Publish one measured number too: OpenShell's enforcement overhead on localhost inference in added milliseconds, captioned "measured on this Spark" — a sandbox that costs nothing is a sandbox nobody will remove.

**Threat model, stated up front.** We treat the agent's context and every model output as fully attacker-controlled: any tile caption, EXIF field or filename may be hostile. We trust the OpenShell runtime and the host kernel. Out of scope for the weekend: kernel exploits, a malicious host, and side-channel egress. Output corruption is partly covered (structured decoding + the k=8 vote) and we say so rather than claiming it is solved.

**Say it in the rubric's own words in the write-up:** the policy bounds the blast radius so that even a fully hijacked agent cannot exfiltrate, escalate, or act outside `./data`. OpenShell protects the agent from being weaponised, and it protects the county's data from the agent.

### Where containment stops, and what covers the rest

OpenShell contains **exfiltration**. It does not contain **output corruption**, and Lightning sits on the untrusted-content path, so we say this plainly and cover it separately: structured-only decoding, Lightning's k=8 vote, and an injection fixture in the eval suite. A hostile caption must not flip a damage grade or a FEMA field.

### How Lightning is actually optimized (not just "we ran it")

| Knob | What we do | Evidence on the HUD |
|---|---|---|
| Speculative decoding | Configure a draft-target pair (use NVIDIA's published drafter for this model if one ships; otherwise a small Nemotron as drafter), record token acceptance rate | Before/after tokens per second |
| Quantization | 4-bit weights as the default, not a panic fallback | Memory and throughput it buys |
| Batching | Sweep batch size to find the throughput knee, then pin it | Chosen batch size, stated with the reason |
| Spend of the speedup | k=8 self-consistency instead of k=1 | Vote agreement rate versus Nano |

The one sentence that wins this bounty on stage: *"Speculative decoding took this batch from X to Y tokens per second, and that headroom is exactly what lets us vote eight times on all fifty buildings inside a three-minute demo."*

---

## 5. The operator workflow — one upload, five panels

This is the product as an operator experiences it. Everything below is reachable from one screen, and the whole thing works with the network denied.

### 5.0 Upload and watch it think

One **Upload drone images** button (multi-select, or the SD-card watcher, or the live RTSP downlink — same pipeline). Hit submit and a per-image progress card shows the three stages by name, because an operator who cannot see the machine working does not trust it:

| Stage | What it does | What the operator sees |
|---|---|---|
| **1. Privacy check — humans** | Person detector runs first, always. Any signal, or any detector error, withholds the image | `12 clear, 1 withheld for authorized review` |
| **2. Damage spotting — buildings** | Seg model outlines every building and grades it 0-3 | `47 buildings outlined, 9 severe` |
| **3. Vulnerability indexing — people in buildings** | Join to footprints, care facilities, SVI; Lightning's ballot computes doubt; the archive gets its caption and embedding | `ranked, indexed, searchable` |

**When a stage fails, the card says so.** A stage never fails silently: the card shows `stage 2 failed — seg model unavailable, fell back to labelled stub` or `stage 3 skipped — no location yet`, and the image stays in the list with what it does have. An operator can always see which of the three stages an image actually cleared.

Geo metadata is extracted from GeoTIFF transform, then EXIF GPS, then a sidecar file, in that order. An image with no location is accepted, flagged `needs_geo`, and the operator can drag it onto the map to place it — never silently dropped.

### 5.1 Part one — the tactical map, with a numbered plan per agency

Two basemaps, one toggle: **tactical** (dark, offline vector-style raster) and **satellite** (real cached imagery, legible to z18 so a crew can see roofs and driveways).

The plan is **divided by responding agency first**, because that is the shape an Operations Section Chief can act on. Nemotron drafts it; a named human owns it.

| Agency | What it gets assigned | Unit accounting |
|---|---|---|
| **Fire** | Structure fires, collapse with entrapment, hazmat | `3 engines, 1 ladder, 1 rescue` |
| **EMS** | Care facilities, dialysis, high-casualty structures | `4 ambulances, 1 supervisor` |
| **Police** | Road closures, perimeter, evacuation escort | `6 units` |
| **Public Works / Heavy Rescue** | Debris clearance, heavy equipment, isolated sectors | `2 high-water vehicles, 1 loader` |

Each agency gets its own **numbered route** on the map — big legible numerals, one colour per agency, in dispatch order — so a crew reads "Fire 1 → Fire 2 → Fire 3" and goes. Every step is **editable**: add, reorder, modify, delete, reassign to another agency, change the unit count. Every edit lands in the append-only log with the operator's name, and the numerals renumber instantly. Each step carries turn-by-turn navigation that avoids blocked roads, printable per agency so a crew leaves with paper.

**Units, with honest provenance.** The plan shows `units_required` — the AI's *ask*. `units_available` is **entered by the operator** for the current operational period and labelled as such on screen; we never invent a resource roster or imply a CAD feed we do not have. When the ask exceeds what the operator entered, the row turns red and offers the third column a real EOC lives on: **Order** — the mutual-aid ask, exported as an **ICS-213 RR resource request** in the aid package, because a disaster by definition overwhelms local resources. A resource light backed by a hardcoded constant would be worse than no light at all.

**What this is, precisely.** Not "how an EOC dispatches" — it is a triage worksheet the Operations Section Chief turns into assignments. We do not model NIMS resource typing, staging and check-in, strike-team assembly or span of control. The named decision owner is the **Operations Section Chief**; the AI drafts, that person disposes, and the log records which of them did what.

### 5.2 Part two — the drone flight plan, editable and exportable

The planner proposes the next survey area and its serpentine path. The operator then edits it like a real mission planner:

- **Add, drag, insert, delete waypoints** — click the line to insert mid-path, drag to move, click a point to delete.
- **Drop a grid** — draw a box, choose line spacing and altitude, get a serpentine or crosshatch pattern with estimated flight time and battery count.
- **Set altitude and speed** per leg; the estimated duration updates live.

**Export to what government teams actually fly:** QGroundControl `.plan` (PX4 / ArduPilot), Mission Planner `.waypoints` (MAVLink), KML for DJI Pilot 2 and Google Earth, Litchi CSV for DJI consumer airframes, GPX, and GeoJSON for GIS. One menu, six files, no internet.

### 5.3 Part three — the searchable image archive

**This is the panel that turns a pile of photos into an asset**, and it is built to be searchable *without* violating the privacy rule: only images that passed stage 1 are ever indexed. Withheld images exist in the restricted queue and are unreachable from search, by construction.

One search bar. Three kinds of query, all offline, all local:

| Query type | Example | How it resolves |
|---|---|---|
| **Location** | `35th Ave SW`, `near Providence Mount`, `47.558, -122.377` | Geocode against the local OSM road and facility tables, then bbox filter |
| **Semantic tag** | `buildings on fire`, `flooded intersections`, `hospital`, `school`, `collapsed roof` | Text embedding of the query against per-image caption embeddings, cosine top-k |
| **Structured filter** | `class:3 after:06:00 sector:C`, chained onto either of the above | SQL over the tile and building tables |

**How the three resolvers combine: filter, then rank.** Location and structured resolvers *narrow* — bbox from the geocode, SQL for the filters. Semantic *ranks* by cosine within whatever survived. A pure semantic query ranks the whole corpus; `buildings on fire near 35th Ave` narrows to the street, then ranks by caption similarity. One rule, no ambiguity about precedence.

**How the index is built.** At stage 3, the caption VLM writes a short factual caption per cleared image ("two-storey wood structure, roof collapsed, standing water in the street"), **Lightning** extracts the tag list from every caption in one batched sweep — thousands of short structured generations is precisely its sweet spot, and it keeps the reasoning model free for the replan beat — and both go into SQLite alongside a normalized embedding vector. Search is cosine similarity in NumPy over a few thousand vectors — no vector database, no service, no network. Results show as a thumbnail grid **and** as pins on the map, so a query is also a spatial answer.

**Editable metadata.** Operators can correct a caption, add or remove tags, fix a location, and mark an image as key evidence. Adding an image goes through **one door only**: the same ingest pipeline, so the privacy gate runs on it exactly as it does on a card dump. There is no writer that reaches the index without passing stage 1 — and a fixture test proves it by trying to add a person image through the archive's own button and watching it get withheld.

**The honest version of the privacy claim.** Two channels, two controls, and we state the residual out loud:

| Channel | Control | Residual |
|---|---|---|
| Pixels | Withheld images are never written to the index. One ingest door, gate first. | Bounded by the measured gate recall (published, A5) |
| **Captions** | The captioner is prompted and post-filtered to describe **structures, terrain and water only**. Any caption mentioning a person, body or clothing is dropped and the image is re-withheld for review. | A caption is a second chance to catch what the detector missed, not a second way to leak |

So the claim we make on stage is precise: *withheld pixels are never indexed, the add and edit paths re-run the same gate, and the captioner is constrained not to write about people.* Not "unreachable, full stop" — a judge can falsify absolutes, and this one does not need them.

### 5.4 Part four — the grounded assistant (optional, and only if it is not a chatbot)

A question box over the archive and the current damage state, answered by Nemotron **with citations only**: every claim comes back with the image IDs and building references it used, and if retrieval returns nothing the answer is "no data covers that yet" rather than a guess. Ask *"which care facilities are cut off and what did the last flight see there?"* and get three cited images plus the road blocks in force.

Grounding rules, non-negotiable: retrieval-only context, no free recall; refuse when retrieval is empty; never surface a withheld image, even when it is topically relevant — the assistant queries the same index as search, which by construction contains no withheld imagery.

This is part four and optional because a thin chatbot layer wins nothing. Build it only after parts one to three are gate-green; it is second on the cut list.

### 5.5 Part five — one button, prefilled federal paperwork

**Download aid package** produces, in one click: the **FEMA Preliminary Damage Assessment** worksheet (one row per damaged structure, with coordinates, category, confidence and who graded it), an **ICS-213 general message**, an **ICS-209 incident summary** (agency assignments and unit counts pulled straight from part one), an **ICS-213 RR** for every agency whose ask exceeds what the operator entered, and the decision log as JSON. Every document carries a "DRAFT — requires approval by the Planning Section Chief" header and a signature line, because a machine does not file federal paperwork.

---

## 6. Three people, three streams, zero waiting

Each member pairs with an AI coding agent. Section 8 is the technique; Appendix A holds a ready-to-paste seed prompt per member. Streams share only the contracts in section 7, frozen in hour one.

### Member A — Perception and Data
*Motto: nothing unsafe, nothing stale.*

| # | Deliverable | Detail |
|---|---|---|
| A1 | Streaming ingest | Receive frames over **RTSP** (frozen choice — the defensible "real downlink" story) plus watch-folder and SD-card fallback; content-hash dedup; emit per-tile end-to-end latency |
| A2 | Privacy gate | Person detection **before any other component reads the tile**; withhold to a restricted queue; detector error also withholds; fixture test proves a person tile never appears in any API response except the authorized review endpoint |
| A3 | Damage grading | xView2 baseline that must work; RescueNet fine-tune only as a stretch goal behind the same function signature |
| A4 | Data joins | Building footprints, CMS Care Compare facilities (facility-level only), CDC SVI block groups, county parcels with owner names dropped at ingest |
| A5 | Gate eval | Person-recall on 100 held-out tiles, number published in the README |
| A6 | Archive indexer | For cleared images only: local vision caption, tag extraction, normalized embedding, all written to SQLite. Withheld images are never indexed — enforce it in the writer, not by convention |
| A7 | Geo fallback chain | GeoTIFF transform, then EXIF GPS, then sidecar, then `needs_geo` for operator drag-to-place. Never silently drop an image |

### Member B — Decision and Agent
*Motto: every rank auditable, every number measured.*

| # | Deliverable | Detail |
|---|---|---|
| B1 | Priority scorer | `priority = staleness x vulnerable_density x doubt [x road_cutoff]`. Round each factor to 3 decimals **first**, then multiply, so the on-screen product reconciles exactly. Operator-confirmed severe damage (class >= 2) pins to the top. Append-only log enforced by SQL triggers |
| B2 | NemoClaw agent | Nano 9B v2 on vLLM at `--gpu-memory-utilization 0.25`; reasoning on for flight tasking, `/no_think` elsewhere; guided JSON; on schema-invalid output **re-prompt once with the validation error** before any stub; expose a demo flag that forces one invalid first attempt so the recovery path is demonstrable on demand; measure co-resident p95 |
| B3 | Lightning layer | vLLM at `--gpu-memory-utilization 0.35`, 4-bit weights; speculative decoding configured and measured; batch-size sweep; k=8 self-consistency vote; before/after numbers to the HUD |
| B4 | Offline routing | Dijkstra over the OSM node graph; blocked roads banned at edge level (by name and geometrically); when no clean route exists, say so loudly and never silently |
| B5 | Containment | NemoClaw inside OpenShell; policy file; both stage beats; audit feed API; the bounty write-up |
| B6 | Agency plan builder | Nemotron drafts assignments grouped by agency (Fire / EMS / Police / Public Works) with `units_required`; flags over-commitment against `units_available`; every operator edit re-logged. Guided JSON, schema in section 7 |
| B7 | Archive search + batch tagging (Lightning) | Three resolvers behind one endpoint: location (geocode against local OSM tables), semantic (cosine over caption embeddings in NumPy), structured filter (SQL). Returns thumbnails + map pins. No vector DB, no service |
| B8 | LLM eval, gate 9 | (a) rationale faithfulness: cited inputs equal scorer inputs, auto-checked; (b) FEMA field accuracy on a small labeled set; (c) two agreement numbers: Lightning self-agreement and Lightning-versus-Nano agreement; (d) agency-plan correctness on a small labeled set (fires to Fire, care facilities to EMS) plus unit-count sanity; (e) tag precision and recall against the caption; **(e2) search recall@k and precision@k on 20 held-out queries with known-relevant image IDs** — the number that turns "we built retrieval" into "we measured it"; (f) assistant citation faithfulness — sampled answers whose cited image IDs actually support the claim, and refusal verified on empty retrieval; (g) injection battery: N hostile captions must produce **0 altered grades and 0 altered FEMA fields**, plus a policy-tamper attempt that OpenShell denies. All four published with their pass criteria |

### Member C — Operator Console and Demo
*Motto: readable from the back of the room.*

| # | Deliverable | Detail |
|---|---|---|
| C1 | Map console | MapLibre with pre-downloaded dark and satellite tiles; damage polygons; facility markers as a medical cross, never a blue dot; blocked roads red; flight box plus survey path |
| C2 | Rank panel | Address labels from real data, not raw IDs; formula inputs visible; confidence and doubt bars; grade-flip logged by operator name; Locate and Navigate buttons per row |
| C3 | Upload + stage tracker | The **Upload drone images** button and the three-stage progress card (privacy check, damage spotting, vulnerability indexing) with live per-image counts |
| C4 | Agency plan panel | Numbered routes per agency, one colour each, big legible numerals; **reassign first** (that is the demoed edit), then add / reorder / edit / delete; operator-entered availability with the Order column; printable per agency |
| C5 | Flight plan panel | Show the proposed area and survey path, plus the export menu. **Waypoint drag/insert/delete and the draw-a-grid tool are stretch** — build them only after gates 1-8 are green, because nothing scores them |
| C6 | Archive panel | One search bar, thumbnail grid plus map pins, add-image (through the ingest door), and caption/tag editing. Re-embed on save is stretch |
| C7 | Aid package | One **Download aid package** button: FEMA PDA, ICS-213, ICS-209, decision log — every document stamped DRAFT with a signature line |
| C8 | HUD + vote column | Tiles processed and withheld, per-tile latency, model names with measured tokens per second, memory gauge, OpenShell policy state, scrollable audit records, and a "model recovered" versus "stub engaged" indicator. **Plus the vote column in the rank panel: 8 tally pips per row filling live, and a doubt bar** — this is Lightning's only visible moment, so it must read from the back of the room |
| C9 | Demo kit | The 3-minute script, a pre-gated judge tile pool, the hostile-caption fixture tile, a `--demo-force-invalid-first-replan` flag (guarantees the self-recovery beat fires live), pre-seeded per-agency availability so the over-commitment flag fires on cue, a canned recording one keypress away, and a one-command reseed |

---

## 7. Interface contracts — complete, frozen, code against these

Copy these into every AI prompt. If a field is missing here, add it here first, then tell the other two.

**Damage classes.** `class` is an integer: `0 = no-damage, 1 = minor, 2 = major, 3 = destroyed`. "Severe" means `class >= 2`. Never use strings on the wire.

**Coordinates.** Every coordinate pair is `[lng, lat]`, GeoJSON order, matching MapLibre. Bounds are `[west, south, east, north]`.

| Contract | Direction | Exact shape |
|---|---|---|
| Tile record | A to B | `{filename: str, bounds: [w,s,e,n], status: "processed" or "withheld" or "error", captured_at: float, latency_ms: int, buildings: [{id: str, class: 0-3, conf: 0.0-1.0}]}` |
| Rank item | B to C | `{footprint_id: str, label: str, centroid: [lng,lat], damage_class: 0-3, confidence: float, confirmed: bool, graded_by: str, facility_near: {name: str, type: str, dist_m: int} or null, inputs: {staleness_h: float, vulnerable_density: float, doubt: float, road_cutoff: float or null}, priority: float, rationale: str, rationale_by: "nano" or "lightning"}` |
| Flight plan | B to C | GeoJSON FeatureCollection with two features: `properties.role = "survey-area"` (Polygon) and `properties.role = "survey-path"` (LineString with `altitude_m_agl`, `line_spacing_m`, `transects`, `est_flight_min`) |
| Route | B to C | `{ok: bool, geometry: LineString, steps: [{text: str, dist_m: int}], distance_m: int, eta_min: float, crosses_blockage: bool, blocked_roads_avoided: [str], warning: str or null}` |
| Status | all to C | `{tiles_processed: int, tiles_withheld: int, tile_latency_ms_p50: int, model_versions: {gate, damage, planner, lightning, captioner, embedder}, tokens_per_s: {nano: float, lightning: float}, memory_gb: float, last_replan_ms: int, recovery: "model" or "stub" or null, openshell: {policy: str, denials: int, audit: [{ts, actor, action, destination, verdict}]}}` |
| Archive item | A to B, B to C | `{image_id: str, thumb_path: str, captured_at: float, centroid: [lng,lat] or null, needs_geo: bool, caption: str, tags: [str], class_max: 0-3, key_evidence: bool}` |
| Search request | C to B | `{q: str, limit: int}` — B decides which resolvers fire; a query may hit all three |
| Search result | B to C | `{items: [ArchiveItem], resolved_by: ["location","semantic","filter"], took_ms: int}` |
| Agency plan | B to C | `{agencies: [{agency: "fire" or "ems" or "police" or "public_works", units_required: int, units_available: int, steps: [{n: int, footprint_id: str, label: str, centroid: [lng,lat], task: str, units: int}]}], drafted_by: str}` |
| Set availability | **C to B** | `{agency: str, units_available: int, operator: str}` — the only source of availability, so B's over-commitment flag has a real input |
| Plan edit | **C to B** | `{agency: str, op: "add" or "move" or "edit" or "delete" or "reassign", step_n: int, payload: {...}, operator: str}` |
| Grade flip | **C to B** | `{footprint_id: str, new_class: 0-3, operator: str}` |
| Road block | **C to B** | `{road_name: str, geometry: LineString, blocked: bool, operator: str}` — B bans by name **and** geometry |

**Field semantics, so nobody invents them:**

- `graded_by` — `"xview2"` until an operator flips it, then `"operator:<name>"`. Lightning never grades the class; it owns the `doubt` term only.
- `road_cutoff` — a multiplier **>= 1** that *raises* priority for buildings cut off by a blocked road; `null` when access is clear.
- `facility_near.type` — one of `nursing_home`, `dialysis`, `hospital`. Those three get the medical-cross marker.
- `captured_at` — epoch seconds, float.
- **List ordering is B's job.** B returns the rank list already sorted: confirmed-severe first as a *sort tiebreaker*, then priority descending. Pinning never inflates `priority`, so gate 4's arithmetic still reconciles. C renders in the order received.

**Where `doubt` comes from:** `doubt = max(0.05, 1 - vote_agreement)` from the Lightning ballot. Before Lightning is wired, use `1 - seg_confidence` so B1 is never blocked on B3.

**Reconciliation rule (gate 4 depends on it):** `priority == round(staleness_h,3) * round(vulnerable_density,3) * round(doubt,3) * (road_cutoff or 1)`, itself rounded to 5 decimals. Member C displays those same rounded factors, so a judge with a calculator always agrees.

---

## 8. How we build this fast with AI pairs

Seven techniques that earned their place on the design spike. Use them verbatim.

1. **Contract-first prompting.** Paste the relevant contract from section 7 into every prompt, plus the non-negotiable. Example: *"Build the ingest watcher. It must emit exactly this Tile record shape. The privacy gate runs before any other read of the file — this is non-negotiable."* The agent then keeps modules compatible without you coordinating.
2. **One module per session.** Fresh context for the gate, the scorer, the routing graph. Long sessions drift; contracts keep the pieces aligned anyway.
3. **Demand the test with the feature.** *"...and write the fixture test proving a person tile never appears in any API response except the authorized review endpoint."* Non-negotiables become executable.
4. **Adversarial review loop.** After each milestone, open a second session as a hostile judge: *"You are a demo-day skeptic. Here is the API and the UI. Find everything that reads as fake, breaks under fast clicking, or contradicts our pitch."* On the prototype this loop caught the three worst bugs: on-screen rank arithmetic that did not reconcile, the hero building vanishing from the list after an operator confirmed it, and routes that silently crossed blocked roads. Humans found none of those.
5. **Geometric truth over string truth.** For anything spatial, assert on geometry — route line intersected with the blocked-road buffer must be under 30 m — never on step text. AI-written tests love string matching, which passes while the map lies.
6. **Stub behind the real interface.** Every model gets a deterministic fallback with an identical signature, labelled in the status bar. The demo never dies, honesty is preserved, and the real model drops in with no code change.
7. **Reseed script from hour one.** One command returns the box to pristine demo state. You will run it fifty times.

---

## 9. Weekend timeline

| When | Member A | Member B | Member C |
|---|---|---|---|
| Fri, first hour | Rules check together, then contracts frozen (section 7) | — | — |
| Fri evening | Download every weight, dataset and map tile — last internet! | Both vLLM instances up; verify Nano + Lightning co-residency; configure speculative decoding; capture baseline numbers | Repo scaffold, offline tile cache, static UI shell |
| Sat morning | Streaming ingest + privacy gate + fixture test | Scorer + decision log + first Nano rationale | Map painting against a stub API |
| Sat afternoon | Seg inference wired; data joins; **archive captions + embeddings** | Flight tasking with guided JSON; Lightning k=8 sweep; **agency plan builder** | Upload + stage tracker; **agency plan panel** |
| Sat evening | Gate recall eval; geo fallback chain; caption + tag pipeline running | Routing with blocked roads; **archive search resolvers**; OpenShell policy + all beats | **Archive panel** and the aid package. Flight-path *display* only; Navigate and print |
| **Sun 09:00** | **Integration starts, no matter what is unfinished** | | |
| Sun morning | 200-tile end-to-end run, fix everything found | p95 with both models warm; LLM eval suite (B8); failure drills | Rehearse the script three times; build the judge tile pool |
| Sun afternoon | Freeze. Reseed. Rehearse. Record the backup video. Write the bounty submissions. | | |

**The 80% rule:** a working 80% beats a broken 100%. Integration starts Sunday 9 AM sharp even if something is half-built.

### Cut list — drop in this order when behind

1. Part four, the grounded assistant — cut it whole, it wins nothing on its own
2. Semantic archive search and the caption VLM (keep location + structured filter, which need no model at all — gate 7's hard floor is written to survive exactly this cut)
3. Manual waypoint editing and the draw-a-grid tool (keep the proposed path + export — no gate or beat touches manual editing)
4. Archive caption re-embed on save (keep add-image and text edits)
5. Agency step CRUD beyond reassign (reassign is the demoed edit)
6. Extra flight-plan formats (keep QGroundControl `.plan` and GeoJSON)
7. Satellite basemap layer (keep the dark tactical map)
8. RescueNet fine-tune (keep the xView2 baseline)
9. SD-card auto-mount (keep the watch folder and the live stream)
10. Live delivery of the human-approved policy delta — play the pre-recorded version instead (the denial and self-tamper beats stay live; they are the bounty)

**Never cuttable:** the privacy gate, containment beats 1-3, and gates 1-8. Gate 9 (evals) degrades gracefully — publish whatever eval subset is measured and mark the rest "not reached" rather than declaring the system unfinished.

---

## 10. Risks, each with a pre-decided answer

| Risk | Answer |
|---|---|
| Both models will not fit or contend for bandwidth | Measured Friday night with explicit utilization splits. If tight, Lightning stays (it is the bounty centrepiece) and Nano drops to 4-bit. We never discover this on Sunday. |
| Replan blows its latency budget | Token cap plus prefill cap already sized; p95 measured with both models warm; narration covers any gap; a hard 10 s timeout fires the stub, and the HUD says which happened |
| xView2 weights misbehave on this imagery | Vision-model grading fallback behind the same interface, disclosed in the status bar |
| OpenShell blocks our own vLLM sockets | Policy allows localhost:8000 and :8001 explicitly, tested Friday. If it still fights us, run inference inside the sandbox too |
| A judge does something unexpected in the UI | Everything visible is safe to click; one-command reseed; recorded backup a keypress away |
| The new archive/agency scope crowds out the core | Cut list items 1-2 exist for exactly this: the assistant goes first, then semantic search. Location and structured search need no model and stay. Gates 1-8 are the line we defend |
| The confident-wrong grade appears on stage | That is the scripted operator-override beat: "high confidence and wrong, which is exactly why a named human owns the final call." Flip it live, watch the tally update |

---

## 11. Definition of done — nine gates

Run all nine Sunday afternoon. Any red light means we are not finished.

| # | Gate |
|---|---|
| 1 | One command brings the whole system up with the network policy denying egress |
| 2 | Ten tiles streamed in paint damage polygons on the map, each within seconds of arrival |
| 3 | A fixture tile containing a person is withheld and appears in no UI or API surface except the authorized review endpoint |
| 4 | Every rank row shows its formula inputs, the displayed product reconciles, and a grade flip updates the tally and pins confirmed severe damage to the top |
| 5 | Blocking a road and replanning yields a new flight area plus a flyable survey path, and Navigate produces turn-by-turn that geometrically avoids the blocked road or warns loudly — printed on paper and legible at arm's length |
| 6 | One **Download aid package** click yields a FEMA PDA that opens in a spreadsheet with one row per damaged building, plus ICS-213, ICS-209 and the decision log |
| 7 | Archive search answers a location query and a structured filter, plus semantic tag search when it is built; a withheld image appears in none of them, and adding it through the archive's own button re-runs the gate and withholds it again |
| 8 | The plan is grouped by agency with unit counts, and an operator can add, reorder, edit, delete and reassign a step, with every edit in the log |
| 9 | The eval numbers are published: gate recall, rationale faithfulness, FEMA field accuracy, Lightning-versus-Nano agreement, and the injection fixture result |

---

## 12. The three-minute demo

| Time | Beat | On screen |
|---|---|---|
| 0:00 | "~109 GB resident, six models warm, and the venue network is up. Watch: our own inference traffic flows to localhost, and the same agent's request to your server is refused. Policy, not an unplugged cable." | HUD: policy `deny-all-egress`, one allow and one deny side by side in the audit panel, memory gauge, both models warm |
| 0:20 | Judge starts the downlink from the pre-gated pool | Tiles arrive live, polygons paint, per-tile latency ticks, withheld counter increments once: "the privacy gate held that tile back; it never reached a screen" |
| 0:50 | **The plan is dispatch-shaped.** Fire 1-2-3, EMS 1-2, Police 1 on the map with unit counts. Judge reassigns a step to EMS; numerals and unit totals update, edit logged by name | Numbered routes per agency, over-commitment flagged red |
| 1:00 | **Lightning's beat.** The uncertainty column fills in live: fifty rows, each showing 8 tally pips and a doubt bar, completing in one visible sweep | "The fast model just voted eight times on all fifty buildings — four hundred generations — in the time the reasoning model wrote one sentence. That column is what re-orders the list." Spec-decode before/after tokens per second on the HUD |
| 1:35 | **"Both bridges on the arterial are out. Replan."** The demo flag forces the agent's first flight plan to come back schema-invalid, so this beat fires every time; the HUD flips to **model recovered** as it re-prompts itself with the validation error, the retry is guided-JSON constrained so it lands valid, then it streams its thinking trace and produces the plan. Navigate reroutes with printable directions | Route detours live, flight box jumps to the cut-off sector, replan p95 on the HUD, recovery indicator witnessed |
| 2:15 | **Containment.** The poisoned caption fires. The agent calls its own fetch tool on an external address — denied. It tries to rewrite its egress rule — denied. It tries to read outside `./data` — denied. | Three audit lines print: `actor=agent`, action, destination, verdict, timestamp |
| 2:30 | **Archive.** Judge types `buildings on fire` — thumbnails and map pins. Then they try to sneak it in: add the withheld image through the archive's own add button. The gate runs again and withholds it again. "One door, gate first. Search never had it, and you just watched it get refused." | Result grid plus pins; empty result for the withheld one |
| 2:40 | One click: **Download aid package** | FEMA PDA, ICS-213, ICS-209 and the log, each stamped DRAFT with a signature line |
| 2:50 | Close | "One box. No cloud. The county owns it. The same policy that stops a hijacked agent is what makes this thing work when the network is gone." |

**Rehearsed contingencies:** every model beat has a stub that keeps the demo moving and labels itself as such; the policy-delta beat is pre-recorded; a full run recording is one keypress away; and the confident-wrong grade, if it appears, becomes the operator-override beat rather than an accident.

---

## Appendix A — Seed prompts

Paste these into your AI coding agent at the start of each session, with the relevant contract from section 7 appended verbatim.

### Member A

```
You are building the perception front end of FIRST LIGHT, an offline disaster-triage
system that runs on one NVIDIA DGX Spark with no internet access.

Build ONE module at a time. Right now: <module name>.

Non-negotiables:
1. The privacy gate runs BEFORE any other component reads a tile. If the person
   detector errors or is uncertain, the tile is withheld. Withheld tiles are
   reachable only from an authorized review endpoint.
2. Emit exactly this Tile record shape: <paste contract>
3. Damage classes are integers 0-3 (0 no-damage, 1 minor, 2 major, 3 destroyed).
4. Coordinates are [lng, lat]; bounds are [w, s, e, n].
5. No network calls to anything except localhost.
6. The archive indexer writes ONLY images that passed the privacy gate. Enforce it
   in the writer itself, so a withheld image is unreachable from search by
   construction, not by a query-time filter.
7. Geo extraction order is GeoTIFF transform, then EXIF GPS, then sidecar file,
   then flag needs_geo. Never drop an image for missing location.

Deliver the module plus a runnable test proving non-negotiables 1 and 6. Use pytest,
no new frameworks, no scaffolding for later.
```

### Member B

```
You are building the decision layer of FIRST LIGHT, an offline disaster-triage
system on one NVIDIA DGX Spark. Two local vLLM servers are available:
Nemotron Nano 9B v2 on :8000 and Nemotron 3.5 Lightning on :8001.

Build ONE module at a time. Right now: <module name>.

Non-negotiables:
1. priority = staleness_h x vulnerable_density x doubt x (road_cutoff or 1).
   Round each factor to 3 decimals BEFORE multiplying so the on-screen product
   reconciles exactly. Property value never enters the formula.
2. The decision log is append-only, enforced by SQL triggers, not by convention.
3. Every model call has a hard timeout and a deterministic fallback with an
   identical signature. Label which one ran in the response.
4. Consume/emit exactly these contracts: <paste contracts>
5. When the model returns schema-invalid JSON, re-prompt ONCE with the validation
   error before falling back.
6. Archive search runs three resolvers behind ONE endpoint: location (geocode
   against the local OSM tables), semantic (cosine over caption embeddings in
   NumPy), structured filter (SQL). No vector database, no external service.
7. The rescue plan is grouped by responding agency (fire, ems, police,
   public_works) with units_required per agency, never one flat list.
8. The assistant answers only from retrieved context, cites the image IDs it
   used, and says "no data covers that yet" when retrieval is empty.

Deliver the module plus tests. For anything spatial, assert on geometry
(intersection lengths, buffers), never on step text.
```

### Member C

```
You are building the operator console for FIRST LIGHT, an offline disaster-triage
system. Single-page app, MapLibre GL, vendored dependencies, no build step.
Dark theme on #0a0d12, one green accent #76b900, one blue secondary #4cc2ff.

Build ONE panel at a time. Right now: <panel name>.

Non-negotiables:
1. Every screen readable from across a room. No purple gradients, no template look.
2. Every button and every non-obvious term has a plain-English tooltip that works
   on hover AND on touch.
3. Consume exactly these contracts, and never invent fields: <paste contracts>
4. Basemap tiles come from the local cache only. No CDN, no font server, no
   external style URL.
5. A 5-second refresh must never close a panel the operator has open.
6. Upload shows the three stages by name as they complete: privacy check,
   damage spotting, vulnerability indexing, with live per-image counts.
7. Agency routes use large legible numerals, one colour per agency, and every
   step is add / reorder / edit / delete / reassign with the numerals renumbering
   instantly.
8. The flight path is editable in place: drag a waypoint, click the line to
   insert one, click a point to delete, draw a box to generate a grid.

Deliver the panel and tell me exactly which API fields it reads.
```

### The judge prompt (all three, after every milestone)

```
You are a hostile hackathon judge for the NVIDIA DGX Spark hackathon. Criteria:
technical depth over wrappers, real NVIDIA stack usage, streaming input, a
multi-step agent with branching and error recovery, and a credible "why a Spark"
story. Here is our API and UI: <paste>. Exercise it, do not just read it.

Find: anything that reads as fake or unfinished on screen, anything that breaks
under fast or repeated clicking, any claim in our pitch the artifact does not
support, and any number we assert without measuring. Return findings as
P0 (demo killer) / P1 (credibility) / P2 (polish), each with a concrete fix.
```

---

*The design in this document was de-risked before the event, so the weekend is spent building rather than discovering. Every number that reaches a judge's eyes is measured on the box in front of them.*
