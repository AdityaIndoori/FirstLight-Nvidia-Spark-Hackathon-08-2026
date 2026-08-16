"""Archive indexer and search. THE privacy enforcement point (A6).

WHY this module is the only writer: the privacy claim we make on stage is that a
tile with people in it is analyzed and ranked, because that is rescue signal, and
is never stored, indexed, thumbnailed or searchable. A claim like that survives a
hostile judge only if there is exactly one door. So `try_store` runs the gate
before it touches the filesystem or the database, `add_via_ingest_door` routes the
archive panel's add-image button back through the same chokepoint, and
`update_metadata` re-runs the caption filter on every edit. There is no code path
that reaches the index without passing here, which is why gate 3 and gate 7 can
be asserted by a fixture rather than promised in prose.

WHY there is no status column to filter on: query-time exclusion is the design we
rejected. A withheld row cannot exist, so `search` never asks about storage
state; forgetting a WHERE clause therefore cannot leak anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from . import config, contracts, db, embed

# PUBLIC API
# try_store(image_path: Path, tile_record: contracts.TileRecord, buildings: list)
#     -> tuple[bool, str | None]                 (stored, withheld_reason); never raises
# add_via_ingest_door(upload_path: Path) -> contracts.TileRecord
# search(q: str = "", limit: int = 50) -> dict   {items, resolved_by, took_ms}
# update_metadata(image_id: str, *, caption=None, tags=None, centroid=None,
#                 key_evidence=None, operator: str) -> dict
# evict(image_id: str) -> bool                   deletes row AND thumbnail file
# stats() -> dict                                HUD counters
# get(image_id: str) -> dict | None              one wired ArchiveItem
# withheld_review(token: str) -> list[dict]      raises PermissionError; ONLY surface
#                                                where a withheld filename appears
# review_configured() -> bool                    False when config.REVIEW_TOKEN is unset
# caption_mentions_person(text: str) -> bool
# reset_geocode_cache() -> None                  call after a librarian atomic swap

THUMB_PX = 320
THUMB_QUALITY = 80
THUMB_URL_PREFIX = "/thumbs"

# Sector grid over the AOI: letter columns west to east, digit rows north to
# south, so "sector:C2" reads like a paper map to an operator with a printout.
SECTOR_COLS = 4
SECTOR_ROWS = 4
_SECTOR_LETTERS = "ABCDEFGH"

# Radius applied to a point geocode (a facility, or a typed "lat, lng" pair).
GEOCODE_RADIUS_M = 400.0
# Corridor half-width around a geocoded road. A street's own bbox is a line, and
# a line contains no centroids, so "near 35th Ave SW" has to mean a corridor.
ROAD_CORRIDOR_M = 150.0

# ------------------------------------------------------------ caption filter
# WHY a word list and not a model: this filter is a control, and a control has to
# be auditable and instant. A caption is a second chance to catch what the
# detector missed, not a second way to leak, so it fails toward withholding: a
# false withhold costs an operator one review click, a false clear costs the
# whole privacy claim.
_PERSON_WORDS = (
    "person persons people human humans man men woman women child children kid kids "
    "boy boys girl girls baby babies body bodies corpse corpses victim victims "
    "survivor survivors casualty casualties crowd crowds pedestrian pedestrians "
    "bystander bystanders occupant occupants resident residents rescuer rescuers "
    "firefighter firefighters worker workers face faces silhouette silhouettes "
    "clothing clothes shirt shirts jacket jackets coat coats hat hats helmet "
    "shoe shoes boots jeans trousers dress uniform backpack stretcher"
).split()
_PERSON_RE = re.compile(r"\b(?:" + "|".join(_PERSON_WORDS) + r")\b", re.IGNORECASE)
# The one exemption, because the captioner is prompted to describe water: "a body
# of water" is terrain, not a person. Narrow and explicit, so widening the
# exemption requires an edit somebody has to justify.
_NOT_A_PERSON_RE = re.compile(r"\bbod(?:y|ies)\s+of\s+water\b", re.IGNORECASE)

# ------------------------------------------------------------ tag vocabulary
# Deterministic caption tags. Lightning's batched sweep (B7) is the real path and
# drops in behind `_extract_tags` with the same signature; this keeps archive
# search answering when the text model is not up, per technique 6.
_TAG_VOCAB: tuple[tuple[str, str], ...] = (
    ("fire", r"\b(fire|burning|burnt|charred|flame|flames|smoke|smoldering)\b"),
    ("flood", r"\b(flood|flooded|flooding|inundat\w+|standing water|submerged)\b"),
    ("water", r"\b(water|river|creek|canal|shoreline|surge)\b"),
    ("collapse", r"\b(collapse\w*|caved|rubble|flattened|destroyed)\b"),
    ("roof-damage", r"\b(roof\w*|shingle\w*|tarp\w*)\b"),
    ("debris", r"\b(debris|wreckage|scattered|strewn)\b"),
    ("tree-down", r"\b(tree\w*|branch\w*|timber)\b"),
    ("road", r"\b(road|street|avenue|intersection|driveway|highway|arterial)\b"),
    ("bridge", r"\b(bridge|overpass|culvert)\b"),
    ("vehicle", r"\b(vehicle\w*|car|cars|truck\w*|bus|van)\b"),
    ("structure", r"\b(structure\w*|building\w*|house\w*|storey|story|residential)\b"),
    ("commercial", r"\b(commercial|warehouse|retail|storefront|industrial)\b"),
    ("mud", r"\b(mud|mudslide|silt|sediment)\b"),
    ("power", r"\b(power line\w*|utility pole\w*|transformer|pylon)\b"),
    ("intact", r"\b(intact|undamaged|no visible damage|no damage)\b"),
)
_TAG_RES = tuple((tag, re.compile(rx, re.IGNORECASE)) for tag, rx in _TAG_VOCAB)

_ARCHIVE_COLS = (
    "image_id, filename, thumb_path, captured_at, centroid_json, needs_geo, "
    "caption, caption_by, tags_json, class_max, key_evidence, embedding, footprints_json, "
    "caption_anchor"
)


# =============================================================== gate plumbing
def _gate_check(image_path: Path):
    """Run the privacy gate. Import is lazy and failures are values, not raises,
    so a missing detector withholds instead of crashing ingest."""
    from . import privacy_gate

    return privacy_gate.check(image_path)


def _withheld_reason(verdict: Any) -> Optional[str]:
    """Name the channel that refused the image, because "withheld" with no
    channel is not an auditable statement."""
    if verdict is None:
        return "detector-error: gate returned nothing"
    dets = list(getattr(verdict, "person_detections", None) or [])
    if dets:
        confs = [float(d.get("conf", 0.0)) for d in dets if isinstance(d, dict)]
        if confs:
            return f"person-signal conf={max(confs):.2f}"
        return f"person-signal detections={len(dets)}"
    err = getattr(verdict, "detector_error", None)
    if err:
        return f"detector-error: {str(err)[:120]}"
    if not getattr(verdict, "store_ok", False):
        return "detector-error: gate refused without a reason"
    return None


def _safe_log(actor: str, action: str, payload: dict) -> None:
    """Log without letting a locked or broken database change a storage verdict.

    The verdict is the product here; the audit line is evidence about it. Losing
    the evidence is bad, turning a refusal into an exception is worse.
    """
    try:
        db.log(actor, action, payload)
    except Exception:  # noqa: BLE001
        pass


def _gate_log(verdict: Any, extra: dict) -> None:
    """Log the gate outcome WITHOUT the filename: the decision log exports
    unauthenticated, so a withheld filename here would leak the thing the gate
    just refused to store."""
    payload = dict(extra)
    summary = getattr(verdict, "summary", None)
    if callable(summary):
        try:
            got = summary()
            if isinstance(got, dict):
                payload.update(got)
        except Exception:
            pass
    else:
        payload.update(
            {
                "store_ok": bool(getattr(verdict, "store_ok", False)),
                "person_detections": len(getattr(verdict, "person_detections", None) or []),
                "detector_error": getattr(verdict, "detector_error", None),
                "took_ms": int(getattr(verdict, "took_ms", 0) or 0),
                "tiles_scanned": int(getattr(verdict, "tiles_scanned", 0) or 0),
            }
        )
    _safe_log("gate", "storage-decision", payload)


def caption_mentions_person(text: str) -> bool:
    """True when the caption mentions a person, body, clothing or crowd.

    Consults the captioner's own checker as well when it is importable, so the
    two lists can only widen the filter, never narrow it.
    """
    if not text:
        return False
    scan = _NOT_A_PERSON_RE.sub(" water ", text)
    if _PERSON_RE.search(scan):
        return True
    try:
        from . import vlm

        checker = getattr(vlm, "caption_mentions_person", None)
        if callable(checker) and bool(checker(text)):
            return True
    except Exception:
        pass
    return False


# =================================================================== the writer
def try_store(
    image_path: Path,
    tile_record: contracts.TileRecord,
    buildings: Iterable[Any] = (),
) -> tuple[bool, Optional[str]]:
    """Index a tile, or refuse to. Returns (stored, withheld_reason). Never raises.

    Order matters and is the whole control: the gate runs first, then the caption
    filter, and only then does anything reach the disk or the database. Nothing
    partial is left behind on refusal, and any row or thumbnail from an earlier
    pass over the same bytes is evicted, because "we stopped writing" is not the
    same as "it is not stored".

    The outer guard is not defensive padding: this is a decision function whose
    two outcomes are "stored" and "withheld". A raised exception is a third
    outcome a caller could mishandle into the wrong one, so anything unexpected
    here resolves to a refusal.
    """
    try:
        return _try_store(Path(image_path), tile_record, list(buildings or []))
    except Exception as exc:  # noqa: BLE001 - a refusal, never a raise
        return False, f"index-error: {type(exc).__name__}: {exc}"[:160]


def _try_store(
    path: Path, tile_record: contracts.TileRecord, blds: list[Any]
) -> tuple[bool, Optional[str]]:
    verdict = None
    try:
        verdict = _gate_check(path)
    except Exception as exc:
        reason = f"detector-error: {type(exc).__name__}: {exc}"[:160]
        _safe_log("gate", "storage-decision", {"store_ok": False, "channel": "gate-unavailable"})
        return False, reason

    if not getattr(verdict, "store_ok", False):
        reason = _withheld_reason(verdict) or "detector-error"
        _gate_log(verdict, {"stored": False, "channel": "pixels"})
        # An identical image may have cleared an earlier, less careful pass.
        _evict_keeping_reason(_image_id(path), "gate", reason)
        return False, reason

    caption, caption_by, caption_anchor = _tile_caption(blds)
    if caption_mentions_person(caption):
        reason = "caption-person-language"
        _gate_log(verdict, {"stored": False, "channel": "caption"})
        _evict_keeping_reason(_image_id(path), "caption-filter", reason)
        _safe_log("archive", "caption-withhold", {"caption_by": caption_by})
        return False, reason

    try:
        image_id = _image_id(path)
        thumb_rel = _write_thumb(path, image_id)
        tags = _extract_tags(caption)
        vec = embed.encode([caption])[0] if caption else np.zeros(embed.dim(), np.float32)
        _insert(
            image_id=image_id,
            filename=path.name,
            thumb_rel=thumb_rel,
            tile_record=tile_record,
            buildings=blds,
            caption=caption,
            caption_by=caption_by,
            tags=tags,
            caption_anchor=caption_anchor,
            vec=vec,
        )
    except Exception as exc:
        # Fail closed: an indexer that half-writes is worse than one that refuses.
        reason = f"index-error: {type(exc).__name__}: {exc}"[:160]
        _evict_keeping_reason(_image_id(path), "archive", "index-error")
        _safe_log("archive", "index-error", {"error": type(exc).__name__})
        return False, reason

    _gate_log(verdict, {"stored": True, "channel": "pixels"})
    return True, None


def add_via_ingest_door(upload_path: Path) -> contracts.TileRecord:
    """The archive panel's add-image button. One door only.

    It copies the upload into the watch folder and hands it to the ingest
    chokepoint with source "archive-add", so the full pipeline including the
    privacy gate runs on it exactly as it does on a card dump. Deliberately no
    shortcut into `try_store`: a second entry point is a second thing to forget.
    """
    src = Path(upload_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    if src.parent.resolve() == config.WATCH_DIR.resolve():
        # Already staged. Copying it would leave a second copy for the watcher to
        # analyze all over again.
        dest = src
    else:
        dest = _unique_in(config.WATCH_DIR, src.name)
        shutil.copy2(src, dest)

    from . import ingest

    rec = ingest.analyze_tile(dest, source="archive-add")
    # Logged AFTER the verdict, and the filename appears only when the image was
    # stored: naming it up front would leak through the unauthenticated log export
    # exactly the file the gate is about to refuse.
    db.log(
        "operator",
        "archive-add",
        {"filename": dest.name} if rec.stored else {"stored": False},
    )
    return rec


def _tile_caption(buildings: list[Any]) -> tuple[str, str, Optional[str]]:
    """One caption per image, reusing the VL pass the grader already paid for.

    A6 is explicit: never call the VLM twice per crop. The grader owns the
    caption; the archive borrows it. The third element is the footprint the caption
    describes, so the archive dot can be placed on that building.
    """
    try:
        from . import grading

        picker = getattr(grading, "tile_caption", None)
        if callable(picker):
            picked = picker(buildings)
            # A patched grader in a test may still return the old 2-tuple.
            caption, caption_by = picked[0], picked[1]
            anchor = picked[2] if len(picked) > 2 else None
            return (caption or ""), (caption_by or "unknown"), anchor
    except Exception:
        pass
    # Standalone fallback with the same rule: describe the worst thing in frame.
    best, best_cls, best_id = "", -1, None
    for b in buildings:
        cap = (getattr(b, "caption", "") or "").strip()
        cls = int(getattr(b, "cls", 0) or 0)
        if cap and cls > best_cls:
            best, best_cls, best_id = cap, cls, getattr(b, "footprint_id", None)
    if best:
        return best, str(getattr(buildings[0], "graded_by", "unknown")), best_id
    return "", "none", None


def _extract_tags(caption: str) -> list[str]:
    """Tags for one caption, deterministic and ordered by the vocabulary."""
    if not caption:
        return []
    return [tag for tag, rx in _TAG_RES if rx.search(caption)]


def _image_id(path: Path) -> str:
    """Content-addressed, so re-adding the same bytes is the same row and the
    fixture test can assert the same image is refused twice."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        h.update(str(path.name).encode("utf-8", "replace"))
    return "img_" + h.hexdigest()[:16]


def _thumb_file(image_id: str) -> Path:
    return config.THUMB_DIR / f"{image_id}.jpg"


def _write_thumb(path: Path, image_id: str) -> str:
    from PIL import Image

    config.THUMB_DIR.mkdir(parents=True, exist_ok=True)
    out = _thumb_file(image_id)
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX))
        im.save(out, "JPEG", quality=THUMB_QUALITY)
    return f"{THUMB_URL_PREFIX}/{out.name}"


def _unique_in(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    n = 2
    while (folder / f"{stem}-{n}{suffix}").exists():
        n += 1
    return folder / f"{stem}-{n}{suffix}"


def _centroid_for(
    tile_record: Any, buildings: list[Any], anchor_id: Optional[str] = None
) -> Optional[list[float]]:
    """Where the archive dot goes.

    WHY not the tile centre, which is what this used to return unconditionally: a
    tile is ~525 m square and holds tens of buildings, so its geometric centre lands
    wherever the arithmetic falls - routinely on a parking lot, a retention pond or
    a road. An operator clicking that dot got a photo captioned "partial collapse
    and large roof holes" centred on empty asphalt, because the dot and the caption
    were describing different things.

    So the dot anchors on the building the caption is ABOUT (anchor_id, the one
    tile_caption picked), falling back to the worst-graded building, then to the
    mean of the footprints, and only then to the tile centre for a tile with no
    footprints at all.
    """
    def _pt(b: Any) -> Optional[list[float]]:
        c = getattr(b, "centroid", None)
        if c and len(c) == 2:
            try:
                return [round(float(c[0]), 6), round(float(c[1]), 6)]
            except (TypeError, ValueError):
                return None
        return None

    if anchor_id:
        for b in buildings:
            if getattr(b, "footprint_id", None) == anchor_id:
                p = _pt(b)
                if p:
                    return p

    # No named anchor: the worst thing in the frame is what the caption would have
    # described anyway, so land on that rather than on the average of everything.
    graded = [b for b in buildings if _pt(b)]
    if graded:
        worst = max(
            graded,
            key=lambda b: (
                int(getattr(b, "cls", 0) or 0),
                float(getattr(b, "conf", 0.0) or 0.0),
                float(getattr(b, "area_m2", 0.0) or 0.0),
            ),
        )
        return _pt(worst)

    pts = [p for p in (_pt(b) for b in buildings) if p]
    if pts:
        return [
            round(sum(p[0] for p in pts) / len(pts), 6),
            round(sum(p[1] for p in pts) / len(pts), 6),
        ]

    bounds = getattr(tile_record, "bounds", None)
    if bounds and len(bounds) == 4:
        w, s, e, n = (float(x) for x in bounds)
        return [round((w + e) / 2.0, 6), round((s + n) / 2.0, 6)]
    return None


def _insert(
    *,
    image_id: str,
    filename: str,
    thumb_rel: str,
    tile_record: Any,
    buildings: list[Any],
    caption: str,
    caption_by: str,
    tags: list[str],
    vec: np.ndarray,
    caption_anchor: Optional[str] = None,
) -> None:
    """Write the row. Only reachable with a cleared gate verdict above it."""
    prior = db.q1("SELECT caption, caption_by, key_evidence FROM archive WHERE image_id=?", (image_id,))
    key_evidence = int(prior["key_evidence"]) if prior else 0
    if prior and str(prior["caption_by"] or "").startswith("operator:"):
        # An operator correction outranks a re-run of the model on the same bytes.
        caption, caption_by = prior["caption"] or caption, prior["caption_by"]
        tags = _extract_tags(caption)
        vec = embed.encode([caption])[0] if caption else vec

    classes = [int(getattr(b, "cls", 0) or 0) for b in buildings]
    footprints = [str(getattr(b, "footprint_id", "")) for b in buildings]
    footprints = [f for f in footprints if f]
    centroid = _centroid_for(tile_record, buildings, caption_anchor)
    db.run(
        f"INSERT OR REPLACE INTO archive ({_ARCHIVE_COLS}) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            image_id,
            filename,
            thumb_rel,
            float(getattr(tile_record, "captured_at", 0.0) or time.time()),
            json.dumps(centroid) if centroid else None,
            1 if getattr(tile_record, "needs_geo", False) else 0,
            caption,
            caption_by,
            json.dumps(tags),
            max(classes) if classes else 0,
            key_evidence,
            np.asarray(vec, dtype=np.float32).tobytes(),
            json.dumps(footprints),
            caption_anchor,
        ),
    )
    db.log(
        "archive",
        "indexed",
        {
            "image_id": image_id,
            "class_max": max(classes) if classes else 0,
            "tags": tags,
            "caption_by": caption_by,
            "embedder": embed.model_version(),
            "footprints": len(footprints),
        },
    )


# =================================================================== eviction
def evict(image_id: str, *, actor: str = "archive", reason: str = "evicted") -> bool:
    """Delete the row AND the thumbnail file. Returns True when a row existed.

    Pixels go first, then the row: if the process dies between the two steps the
    survivor is an index entry pointing at nothing, which the UI renders as a
    broken tile. The other order would leave the pixels reachable, which is
    precisely the leak this module exists to prevent.
    """
    row = db.q1("SELECT image_id FROM archive WHERE image_id=?", (image_id,))
    thumb = _thumb_file(image_id)
    had_thumb = thumb.exists()
    if had_thumb:
        try:
            thumb.unlink()
        except OSError:
            pass
    if row is None:
        return False
    db.run("DELETE FROM archive WHERE image_id=?", (image_id,))
    _safe_log(
        actor,
        "archive-evict",
        {"image_id": image_id, "reason": reason, "thumb_deleted": had_thumb},
    )
    return True


def _evict_keeping_reason(image_id: str, actor: str, reason: str) -> None:
    """Evict during a refusal without letting the eviction's own failure rename
    the verdict: "person-signal conf=0.87" is the audit line we owe, not
    "index-error" from a locked database on the cleanup step."""
    try:
        evict(image_id, actor=actor, reason=reason)
    except Exception as exc:  # noqa: BLE001
        _safe_log("archive", "evict-failed", {"image_id": image_id, "error": type(exc).__name__})


# ==================================================================== metadata
def update_metadata(
    image_id: str,
    *,
    caption: Optional[str] = None,
    tags: Optional[list[str]] = None,
    centroid: Optional[list[float]] = None,
    key_evidence: Optional[bool] = None,
    operator: str,
) -> dict:
    """Operator edits, with the caption filter re-run on the way in.

    The edit path is a writer, so it inherits the writer's control: an edit that
    introduces person language evicts the row and its thumbnail instead of being
    saved. Returns {"ok": bool, "evicted": bool, ...}.
    """
    if not (operator or "").strip():
        raise ValueError("operator name required")
    row = db.q1(f"SELECT {_ARCHIVE_COLS} FROM archive WHERE image_id=?", (image_id,))
    if row is None:
        return {"ok": False, "evicted": False, "error": "unknown image_id"}

    new_caption = row["caption"] if caption is None else str(caption)
    new_tags = db.jload(row["tags_json"], []) if tags is None else [str(t) for t in tags]
    if caption_mentions_person(new_caption) or any(caption_mentions_person(t) for t in new_tags):
        evict(image_id, actor=f"operator:{operator}", reason="caption-person-language")
        db.log("caption-filter", "edit-withhold", {"image_id": image_id})
        return {
            "ok": False,
            "evicted": True,
            "image_id": image_id,
            "reason": "caption-person-language",
        }

    if centroid is not None and len(centroid) != 2:
        raise ValueError("centroid must be [lng, lat]")
    new_centroid = db.jload(row["centroid_json"], None) if centroid is None else [
        float(centroid[0]),
        float(centroid[1]),
    ]
    new_key = int(row["key_evidence"]) if key_evidence is None else int(bool(key_evidence))
    caption_by = row["caption_by"]
    vec_blob = row["embedding"]
    if caption is not None and new_caption != (row["caption"] or ""):
        caption_by = f"operator:{operator}"
        if tags is None:
            new_tags = _extract_tags(new_caption)
        vec_blob = np.asarray(embed.encode([new_caption])[0], dtype=np.float32).tobytes()

    db.run(
        "UPDATE archive SET caption=?, caption_by=?, tags_json=?, centroid_json=?, "
        "needs_geo=?, key_evidence=?, embedding=? WHERE image_id=?",
        (
            new_caption,
            caption_by,
            json.dumps(new_tags),
            json.dumps(new_centroid) if new_centroid else None,
            0 if new_centroid else int(row["needs_geo"]),
            new_key,
            vec_blob,
            image_id,
        ),
    )
    db.log(
        f"operator:{operator}",
        "archive-edit",
        {
            "image_id": image_id,
            "caption_changed": caption is not None,
            "tags": new_tags,
            "centroid": new_centroid,
            "key_evidence": bool(new_key),
        },
    )
    return {"ok": True, "evicted": False, "image_id": image_id, "item": get(image_id)}


def get(image_id: str) -> Optional[dict]:
    row = db.q1(f"SELECT {_ARCHIVE_COLS} FROM archive WHERE image_id=?", (image_id,))
    return _item(row).wire() if row is not None else None


# ====================================================================== search
@dataclass
class _Parsed:
    """One query, split into what narrows and what ranks."""

    text: str
    where: list[str]
    args: list[Any]
    bbox: Optional[list[float]]
    location_label: str
    used_filter: bool
    # A sector bbox is a structured filter per the plan's own table, a geocoded
    # street or coordinate pair is the location resolver. Both narrow by bbox, so
    # only the reported name differs.
    used_location: bool


def search(q: str = "", limit: int = 50) -> dict:
    """Three resolvers behind one entry point: filter, then rank.

    Location and structured tokens NARROW (bbox, SQL). Semantic RANKS by cosine
    within whatever survived. A pure semantic query ranks the whole corpus.

    Storage state is never a query term. A withheld image has no row here, so
    search cannot exclude it and cannot accidentally include it either.
    """
    t0 = time.perf_counter()
    limit = max(1, min(500, int(limit or 50)))
    parsed = _parse(q or "")
    resolved: list[str] = []

    sql = f"SELECT {_ARCHIVE_COLS} FROM archive"
    if parsed.where:
        sql += " WHERE " + " AND ".join(parsed.where)
    sql += " ORDER BY captured_at DESC"
    try:
        rows = db.q(sql, parsed.args)
    except sqlite3.Error:
        rows = []
    if parsed.used_filter:
        resolved.append("filter")

    if parsed.bbox:
        rows = [r for r in rows if _in_bbox(db.jload(r["centroid_json"], None), parsed.bbox)]
        if parsed.used_location:
            resolved.append("location")

    items = [_item(r) for r in rows]
    if parsed.text.strip() and items:
        items, ranked = _rank_semantic(parsed.text, rows, items)
        # Only claim the resolver that actually ran: an unrankable query (every
        # token a stopword, or no embedder) leaves the set in recency order, and
        # saying "semantic" then would be the UI asserting something untrue.
        if ranked:
            resolved.append("semantic")

    took = int((time.perf_counter() - t0) * 1000)
    return {
        "items": [it.wire() for it in items[:limit]],
        "resolved_by": resolved,
        "took_ms": took,
        "total": len(items),
        "location": parsed.location_label or None,
        "embedder": embed.model_version(),
    }


_TOKEN_RE = re.compile(r"\b(class|after|before|sector|key):([A-Za-z0-9:.]+)", re.IGNORECASE)
_LATLNG_RE = re.compile(r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)")


def _parse(q: str) -> _Parsed:
    where: list[str] = []
    args: list[Any] = []
    used_filter = False
    used_location = False
    bbox: Optional[list[float]] = None
    label = ""
    text = q

    for m in _TOKEN_RE.finditer(q):
        key, raw = m.group(1).lower(), m.group(2)
        if key == "class":
            try:
                where.append("class_max >= ?")
                args.append(max(0, min(3, int(raw))))
                used_filter = True
            except ValueError:
                continue
        elif key in ("after", "before"):
            clock = _clock(raw)
            if clock:
                # NULL captured_at compares false, so an undated row is excluded
                # from a time-filtered query rather than silently included.
                op = ">=" if key == "after" else "<="
                where.append(f"strftime('%H:%M', captured_at, 'unixepoch', 'localtime') {op} ?")
                args.append(clock)
                used_filter = True
        elif key == "key":
            if raw.lower() in ("true", "1", "yes"):
                where.append("key_evidence = 1")
                used_filter = True
        elif key == "sector":
            got = sector_bounds(raw)
            if got:
                bbox = _bbox_narrow(bbox, got)
                label = f"sector {raw.upper()}"
                used_filter = True
        text = text.replace(m.group(0), " ")

    ll = _LATLNG_RE.search(text)
    if ll:
        a, b = float(ll.group(1)), float(ll.group(2))
        # Humans type "lat, lng"; we store [lng, lat]. A first value out of
        # latitude range means they typed it the GeoJSON way instead.
        lat, lng = (a, b) if abs(a) <= 90 else (b, a)
        pt_box = _point_bbox(lng, lat, GEOCODE_RADIUS_M)
        bbox = _bbox_narrow(bbox, pt_box)
        used_location = True
        label = label or f"{lat:.5f}, {lng:.5f}"
        text = text.replace(ll.group(0), " ")

    text = re.sub(r"\b(near|around|at)\b", " ", text, flags=re.IGNORECASE)
    place = _match_place(text)
    if place:
        name, pbox = place
        bbox = _bbox_narrow(bbox, pbox)
        used_location = True
        label = label or name
        # Remove the matched phrase so the semantic resolver ranks on the rest
        # ("buildings on fire near 35th Ave" ranks on "buildings on fire").
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)

    return _Parsed(
        text=" ".join(text.split()),
        where=where,
        args=args,
        bbox=bbox,
        location_label=label,
        used_filter=used_filter,
        used_location=used_location,
    )


def _clock(raw: str) -> Optional[str]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", raw)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if hh > 23 or mm > 59:
        return None
    return f"{hh:02d}:{mm:02d}"


def sector_bounds(code: str) -> Optional[list[float]]:
    """Translate a map-grid code over the AOI into [w, s, e, n]."""
    m = re.match(r"^([A-Za-z])(\d+)?$", (code or "").strip())
    if not m:
        return None
    col = _SECTOR_LETTERS.find(m.group(1).upper())
    if col < 0 or col >= SECTOR_COLS:
        return None
    w, s, e, n = (float(x) for x in config.AOI)
    cw = (e - w) / SECTOR_COLS
    x0, x1 = w + col * cw, w + (col + 1) * cw
    if m.group(2) is None:
        return [x0, s, x1, n]
    row = int(m.group(2))
    if row < 1 or row > SECTOR_ROWS:
        return None
    ch = (n - s) / SECTOR_ROWS
    y1 = n - (row - 1) * ch  # row 1 is the northern band, as on a paper grid
    return [x0, y1 - ch, x1, y1]


def _point_bbox(lng: float, lat: float, radius_m: float) -> list[float]:
    dlat = radius_m / 111320.0
    dlng = radius_m / max(1.0, 111320.0 * math.cos(math.radians(lat)))
    return [lng - dlng, lat - dlat, lng + dlng, lat + dlat]


def _grow_bbox(box: list[float], pad_m: float) -> list[float]:
    """Widen a bbox by a metric margin on every side.

    A LineString road bbox is zero-width in one axis, so growing it is what turns
    a street into the searchable corridor an operator means by "near".
    """
    if pad_m <= 0:
        return box
    mid_lat = (box[1] + box[3]) / 2.0
    dlat = pad_m / 111320.0
    dlng = pad_m / max(1.0, 111320.0 * math.cos(math.radians(mid_lat)))
    return [box[0] - dlng, box[1] - dlat, box[2] + dlng, box[3] + dlat]


def _bbox_union(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _bbox_narrow(current: Optional[list[float]], box: list[float]) -> list[float]:
    """Combine two spatial constraints from one query by INTERSECTING them.

    "sector:B2 near 35th Ave SW" means both, so the constraints narrow. A union
    would widen the result set with every extra term the operator typed, which is
    the opposite of what filter-then-rank promises. An empty intersection is kept
    as-is and simply matches nothing, which is the honest answer.
    """
    if current is None:
        return box
    return [
        max(current[0], box[0]),
        max(current[1], box[1]),
        min(current[2], box[2]),
        min(current[3], box[3]),
    ]


def _in_bbox(centroid: Optional[list[float]], bbox: list[float]) -> bool:
    if not centroid or len(centroid) != 2:
        return False
    lng, lat = float(centroid[0]), float(centroid[1])
    return bbox[0] <= lng <= bbox[2] and bbox[1] <= lat <= bbox[3]


_places_cache: Optional[list[tuple[str, list[float]]]] = None


def reset_geocode_cache() -> None:
    """Called after a librarian atomic swap: new road and facility files mean the
    geocoder's name table is stale."""
    global _places_cache
    _places_cache = None


def _places() -> list[tuple[str, list[float]]]:
    """Local gazetteer: road and facility names with their bounds, longest first.

    Built from the datasets module's normalized `properties.name`, so every key
    alias it tolerates is inherited here for free.
    """
    global _places_cache
    if _places_cache is not None:
        return _places_cache
    out: list[tuple[str, list[float]]] = []
    try:
        from . import datasets

        for getter, pad in (
            ("roads_geojson", ROAD_CORRIDOR_M),
            ("facilities_geojson", GEOCODE_RADIUS_M),
        ):
            fn = getattr(datasets, getter, None)
            if not callable(fn):
                continue
            try:
                fc = fn() or {}
            except Exception:
                continue
            for feat in fc.get("features", []) or []:
                props = feat.get("properties") or {}
                name = str(props.get("name") or "").strip()
                if not name:
                    continue
                box = _geom_bbox(feat.get("geometry") or {})
                if box is None:
                    continue
                out.append((name.lower(), _grow_bbox(box, pad)))
    except Exception:
        out = []
    merged: dict[str, list[float]] = {}
    for name, box in out:
        merged[name] = _bbox_union(merged[name], box) if name in merged else box
    _places_cache = sorted(merged.items(), key=lambda kv: -len(kv[0]))
    return _places_cache


def _geom_bbox(geom: dict) -> Optional[list[float]]:
    coords: list[float] = []
    lats: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (int, float)) and len(node) >= 2:
                coords.append(float(node[0]))
                lats.append(float(node[1]))
                return
            for child in node:
                walk(child)

    walk((geom or {}).get("coordinates"))
    if not coords:
        return None
    return [min(coords), min(lats), max(coords), max(lats)]


def _match_place(text: str) -> Optional[tuple[str, list[float]]]:
    low = " ".join((text or "").lower().split())
    if not low:
        return None
    for name, box in _places():
        if len(name) >= 3 and name in low:
            return name, box
    # Nothing matched locally: the datasets module may still know the name.
    try:
        from . import datasets

        fn = getattr(datasets, "geocode", None)
        if callable(fn):
            got = fn(low)
            if got and len(got) == 4:
                # Padded for the same reason: a road bbox from anywhere is a line.
                return low, _grow_bbox([float(x) for x in got], ROAD_CORRIDOR_M)
    except Exception:
        pass
    return None


def _rank_semantic(
    text: str, rows: list[sqlite3.Row], items: list[contracts.ArchiveItem]
) -> tuple[list[contracts.ArchiveItem], bool]:
    """Cosine over stored caption vectors, one matmul, no vector database.

    Returns (ordered_items, ranked). `ranked` is False when the query produced no
    usable vector, so the caller does not claim a resolver that did not fire.

    Rows whose embedding is missing or was written at a different dimension score
    0.0 and sink; they are never dropped, because the narrowing resolvers already
    decided who is a candidate.

    A RELEVANCE FLOOR applies, and it is a correctness matter rather than polish.
    Ranking without a floor returns the whole corpus for every query, so "penguins
    in antarctica" comes back with the same hits as "roof damage" in the same order
    as each other's tails, and the panel looks like it matched something. Measured
    on real captions: "roof damage" scores 0.81 to 0.84, "buildings on fire" 0.58
    to 0.66, "penguins in antarctica" 0.41 to 0.47, gibberish 0.38 to 0.45. So a
    floor near 0.5 separates a real topical hit from the embedder's noise band. The
    score rides on each item, so the panel can show what it matched on rather than
    asking an operator to trust an ordering.
    """
    dim = embed.dim()
    qv = embed.encode([text])
    if qv.shape[0] == 0 or not np.any(qv):
        return items, False
    mat = np.zeros((len(rows), dim), dtype=np.float32)
    scorable = [False] * len(rows)
    for i, row in enumerate(rows):
        blob = row["embedding"]
        if not blob:
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.size == dim:
            mat[i] = vec
            scorable[i] = True
    scores = mat @ qv[0]
    order = np.argsort(-scores, kind="stable")
    floor = semantic_floor()
    kept: list[contracts.ArchiveItem] = []
    unscorable: list[contracts.ArchiveItem] = []
    for i in order:
        if not scorable[i]:
            # A missing or wrong-dimension vector means UNKNOWN relevance, not
            # irrelevance, and the narrowing resolvers already decided this row is
            # a candidate. Dropping it would hide a real image because its
            # embedding failed to write, so it sinks to the bottom with no score
            # rather than being deleted by a threshold it was never measured
            # against.
            unscorable.append(items[i])
            continue
        score = round(float(scores[i]), 3)
        if score < floor:
            continue
        items[i].score = score
        kept.append(items[i])
    return kept + unscorable, True


def semantic_floor() -> float:
    """Minimum cosine for a semantic hit, or 0 to keep everything.

    Measured with the real BGE embedder on real captions: a topical hit scores 0.77
    to 0.84, a loose association 0.58 to 0.66, and nonsense or gibberish 0.38 to
    0.47. So a floor near 0.5 separates a real match from the noise band, and
    without one every query returns the whole corpus in some order, which makes
    "penguins in antarctica" look like it matched.

    The floor is DISABLED for the stub embedder, because a hash-derived pseudo
    vector has no meaningful similarity scale and a threshold tuned for BGE would
    silently empty the archive. When the stub is running, ordering is the only
    signal available and it is better to return a ranked list the status strip
    already labels "stub-hash-v1" than to return nothing.
    """
    raw = os.environ.get("FIRSTLIGHT_SEMANTIC_FLOOR")
    if raw is not None:
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            pass
    if getattr(embed, "STUB_VERSION", "stub-hash-v1") == embed.model_version():
        return 0.0
    return 0.5


def _col(row: sqlite3.Row, name: str) -> Optional[str]:
    """A column that may be absent from an older row object. Selects elsewhere in
    this module name columns explicitly, and a stale prepared query would otherwise
    turn a schema addition into an IndexError on read."""
    try:
        value = row[name]
    except (IndexError, KeyError):
        return None
    return str(value) if value else None


def _item(row: sqlite3.Row) -> contracts.ArchiveItem:
    return contracts.ArchiveItem(
        image_id=row["image_id"],
        thumb_path=row["thumb_path"] or "",
        captured_at=float(row["captured_at"] or 0.0),
        centroid=db.jload(row["centroid_json"], None),
        needs_geo=bool(row["needs_geo"]),
        caption=row["caption"] or "",
        tags=db.jload(row["tags_json"], []) or [],
        class_max=int(row["class_max"] or 0),
        key_evidence=bool(row["key_evidence"]),
        footprint_ids=db.jload(row["footprints_json"], []) or [],
        caption_anchor=_col(row, "caption_anchor"),
    )


# ======================================================================== HUD
def stats() -> dict:
    """Counters for C8. `withheld_from_storage` comes from the tiles table,
    because by construction the archive has nothing to count there."""
    indexed = db.q1("SELECT COUNT(*) AS n FROM archive")
    key = db.q1("SELECT COUNT(*) AS n FROM archive WHERE key_evidence=1")
    withheld = db.q1("SELECT COUNT(*) AS n FROM tiles WHERE stored=0")
    newest = db.q1("SELECT MAX(captured_at) AS t FROM archive")
    thumbs = sum(1 for _ in config.THUMB_DIR.glob("*.jpg")) if config.THUMB_DIR.exists() else 0
    return {
        "indexed": int(indexed["n"] if indexed else 0),
        "key_evidence": int(key["n"] if key else 0),
        "withheld_from_storage": int(withheld["n"] if withheld else 0),
        "thumbnails": thumbs,
        "newest_captured_at": float(newest["t"]) if newest and newest["t"] else None,
        "embedder": embed.model_version(),
        "embedder_stub": not embed.available(),
        "embed_dim": embed.dim(),
    }


# ====================================================== authorized review only
def review_configured() -> bool:
    return bool((config.REVIEW_TOKEN or "").strip())


def withheld_review(token: str) -> list[dict]:
    """The ONLY surface where a withheld filename may appear.

    Fails closed: an unset REVIEW_TOKEN refuses every caller rather than
    defaulting to an open door.
    """
    expected = (config.REVIEW_TOKEN or "").strip()
    if not expected or (token or "").strip() != expected:
        db.log("review", "withheld-review-denied", {})
        raise PermissionError("review token required")
    rows = db.q(
        "SELECT filename, withheld_reason, captured_at, analyzed_at, stored_path "
        "FROM tiles WHERE stored=0 ORDER BY analyzed_at DESC"
    )
    db.log("review", "withheld-review", {"count": len(rows)})
    return [
        {
            "filename": r["filename"],
            "withheld_reason": r["withheld_reason"],
            "captured_at": r["captured_at"],
            "analyzed_at": r["analyzed_at"],
            "path": r["stored_path"],
        }
        for r in rows
    ]


__all__ = [
    "add_via_ingest_door",
    "caption_mentions_person",
    "evict",
    "get",
    "reset_geocode_cache",
    "review_configured",
    "search",
    "sector_bounds",
    "stats",
    "try_store",
    "update_metadata",
    "withheld_review",
]
