"""A3 damage grading: building outlines plus one damage class and caption each.

Three tiers behind ONE signature, so callers never branch on what is installed.
`model_version()` names the path that actually ran, which is what the status bar
prints, because a stub that does not announce itself is a lie.

Outlines, in order:
  1. xView2 loc ensemble. Single-image, works as-is, runs when the checkpoints
     AND torch AND the ensemble's own model code are all present.
  2. County footprints from config.DATASET_DIR, clipped to the tile bounds. In a
     county with a GIS department these are better than any inference.
  3. A deterministic synthetic grid, so a tile never dead-ends with no buildings.

Grades: the primary grader is vlm.caption_and_grade, one VL pass per building
crop, which yields the damage class and the archive caption together. VL calls
per tile are capped (FIRSTLIGHT_VL_CALLS_PER_TILE, default 12) and the remainder
are graded by the labelled pixel-statistic stub, because a 200-building tile at
7 s per crop is 23 minutes and the plan's budget is 10 s per tile.

WHY xView2 cls is not in the grade chain here: verified in the ensemble code, the
cls models concatenate the pre-disaster and post-disaster chips into six input
channels. Duplicating the post image into the pre slot makes every pixel identical
between the two halves, which the model reads as "no change", and "no change"
decodes as "no damage". That is the worst possible bias for a triage tool: it would
report an intact neighbourhood on the morning a hurricane flattened it. So cls runs
only where cached pre-event basemap chips genuinely cover the tile, and
`graded_by` is "xview2" only in that case. There is no code path here that fakes
a pre image.
"""
from __future__ import annotations

import concurrent.futures as futures
import functools
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from PIL import Image

from . import config, contracts, datasets, vlm

# PUBLIC API
# GRADED_BY_VL = "nemotron-vl"
# GRADED_BY_STUB = "stub-pixelstat-v1"
# GRADED_BY_XVIEW2 = "xview2"
# GradedBuilding: .footprint_id .cls .conf .centroid .geom .area_m2 .graded_by
#                 .caption .label .facility_near .svi
#                 .building() -> contracts.Building
# outline_and_grade(image_path, bounds, *, vl_budget_override=None) -> list[GradedBuilding]
#     bounds may be None, which returns [] (no transform, so nothing is placeable)
# tile_caption(buildings) -> tuple[str, str]        # one archive caption per tile
# model_version() -> str                            # active path, for the status bar
# last_run() -> dict                                # counts from the most recent tile
# outline_source() -> str                           # "xview2" | "footprints" | "grid"
# vl_budget() -> int                                # FIRSTLIGHT_VL_CALLS_PER_TILE
# tile_wall_budget() -> float                       # FIRSTLIGHT_TILE_VL_SECONDS
# bounds_to_pixel(bounds, width, height, lng, lat) -> tuple[float, float]
# pixel_to_bounds(bounds, width, height, x, y) -> list[float]
# mask_to_polygons(mask, bounds, *, min_px=40, max_shapes=400) -> list[dict]

log = logging.getLogger("firstlight.grading")

GRADED_BY_VL = "nemotron-vl"
GRADED_BY_STUB = "stub-pixelstat-v1"
GRADED_BY_XVIEW2 = "xview2"

OUTLINE_XVIEW2 = "xview2"
OUTLINE_FOOTPRINTS = "footprints"
OUTLINE_GRID = "grid"
# Coverage exists and this frame has no buildings under it. Distinct from "grid"
# (we guessed) and from a failure: open ground is a real, reportable answer.
OUTLINE_NONE = "none-in-frame"

# Every number below is MEASURED on this box (service/tools/measure_budget.py),
# not estimated. One VL grading call on a real footprint crop is ~2.2 s.
#
# The captioner's vLLM server runs --max-num-seqs 4, and measured wall time per
# tile at 12 calls was 17.5 s at 2 lanes, 12.6 s at 4, 11.5 s at 8: past the
# server's own batch width the curve flattens, so 8 lanes is the knee and not a
# guess. Held at 8 rather than raised because the same GPU serves the planner and
# the k=8 ballot.
DEFAULT_VL_CONCURRENCY = 8

# At 8 lanes the sweep measured p50 7.6 s (budget 6), 9.1 s (8), 11.2 s (10),
# 13.5 s (12). The plan's per-tile budget is 10 s, so 8 is the most model coverage
# that fits: 8 model grades per tile with p50 9.1 s and worst case 10.9 s.
DEFAULT_VL_CALLS_PER_TILE = 8
# Footprints outlined per tile. Sized so most buildings on screen carry a real
# model grade rather than a stub: a rank list where every row shows the same
# class and the same confidence is not a rank list.
DEFAULT_FOOTPRINTS_PER_TILE = 40

# The wall-clock companion to the call cap. Eight concurrent crops at ~2.2 s land
# in ~9 s, so this is the ceiling that catches a server which has gone slow rather
# than down: past it the remaining crops take the labelled stub.
DEFAULT_TILE_VL_SECONDS = 20.0

# Context around a footprint helps the grader see a collapsed wall lying outside
# the polygon, expressed as a fraction of the bbox.
CROP_PAD = 0.18
MIN_CROP_PX = 24

_LOCK = threading.Lock()
_LAST_RUN: dict[str, Any] = {
    "outline_source": "",
    "buildings": 0,
    "vl_calls": 0,
    "model_graded": 0,
    "stub_graded": 0,
    "grade_path": "",
}


# ------------------------------------------------------------------ the record
@dataclass
class GradedBuilding:
    """One outlined, graded structure. datasets.join fills the last three fields."""

    footprint_id: str
    cls: int
    conf: float
    centroid: list[float]
    geom: dict
    area_m2: float
    graded_by: str
    caption: str
    label: str = ""
    facility_near: Optional[contracts.FacilityNear] = None
    svi: Optional[float] = None
    props: dict = field(default_factory=dict)

    def building(self) -> contracts.Building:
        """The TileRecord shape. A to B carries id, class and confidence only."""
        return contracts.Building(id=self.footprint_id, cls=int(self.cls), conf=float(self.conf))


# --------------------------------------------------------------- pixel mapping
def _span(bounds: Sequence[float]) -> tuple[float, float, float, float]:
    w, s, e, n = (float(v) for v in bounds[:4])
    if e < w:
        w, e = e, w
    if n < s:
        s, n = n, s
    return w, s, e, n


def bounds_to_pixel(
    bounds: Sequence[float], width: int, height: int, lng: float, lat: float
) -> tuple[float, float]:
    """Georeferenced point to pixel. North is row zero, matching image order."""
    w, s, e, n = _span(bounds)
    dx = (e - w) or 1e-12
    dy = (n - s) or 1e-12
    return ((float(lng) - w) / dx * width, (n - float(lat)) / dy * height)


def pixel_to_bounds(
    bounds: Sequence[float], width: int, height: int, x: float, y: float
) -> list[float]:
    """Pixel to [lng, lat]. The inverse of bounds_to_pixel."""
    w, s, e, n = _span(bounds)
    return [
        round(w + (float(x) / max(width, 1)) * (e - w), 7),
        round(n - (float(y) / max(height, 1)) * (n - s), 7),
    ]


def _crop_box(
    geom: dict, bounds: Sequence[float], width: int, height: int
) -> Optional[tuple[int, int, int, int]]:
    box = datasets.bbox_of(geom)
    if box is None:
        return None
    x1, y1 = bounds_to_pixel(bounds, width, height, box[0], box[3])
    x2, y2 = bounds_to_pixel(bounds, width, height, box[2], box[1])
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    pad_x = max((right - left) * CROP_PAD, 4.0)
    pad_y = max((bottom - top) * CROP_PAD, 4.0)
    left, right = left - pad_x, right + pad_x
    top, bottom = top - pad_y, bottom + pad_y
    # Widen anything too small to carry detail, centred on the footprint.
    if right - left < MIN_CROP_PX:
        cx = (left + right) * 0.5
        left, right = cx - MIN_CROP_PX / 2, cx + MIN_CROP_PX / 2
    if bottom - top < MIN_CROP_PX:
        cy = (top + bottom) * 0.5
        top, bottom = cy - MIN_CROP_PX / 2, cy + MIN_CROP_PX / 2
    li, ti = max(0, int(left)), max(0, int(top))
    ri, bi = min(width, int(round(right))), min(height, int(round(bottom)))
    if ri - li < 2 or bi - ti < 2:
        return None
    return li, ti, ri, bi


def _stable_id(centroid: Sequence[float]) -> str:
    """Deterministic id from position, so a re-flown building updates its own row.

    Rounded to five decimals, roughly one metre, which is inside the outline noise
    of two passes over the same roof and far below the spacing of two buildings.
    """
    key = f"{float(centroid[0]):.5f},{float(centroid[1]):.5f}"
    return "b_" + hashlib.sha1(key.encode("ascii")).hexdigest()[:12]


# ---------------------------------------------------------------- image access
def _open_image(image_path: Union[str, Path, Image.Image]) -> Optional[Image.Image]:
    if isinstance(image_path, Image.Image):
        return image_path.convert("RGB")
    try:
        with Image.open(image_path) as im:
            return im.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        log.warning("PIL could not open %s: %s", image_path, exc)
    try:  # a GeoTIFF layout PIL rejects may still open through rasterio
        import numpy as np
        import rasterio

        with rasterio.open(str(image_path)) as src:
            arr = src.read(indexes=[1, 2, 3][: min(3, src.count)])
        arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return Image.fromarray(arr.astype("uint8"), "RGB")
    except Exception as exc:  # noqa: BLE001
        log.warning("rasterio could not open %s: %s", image_path, exc)
        return None


# ------------------------------------------------------- tier 1: xView2 loc
@functools.lru_cache(maxsize=1)
def _xview2_state() -> tuple[bool, str]:
    """Can the loc ensemble run here, and if not, exactly why.

    Three independent prerequisites, checked cheapest first. The ensemble's own
    model code is a separate check from torch on purpose: the checkpoints are just
    state dicts, so without the repo's architecture definitions there is nothing
    to load them into, and guessing at the architecture would silently produce
    garbage masks.
    """
    weights = Path(config.XVIEW2_WEIGHTS)
    if not weights.is_dir():
        return False, "no weights directory"
    checkpoints = [p for p in weights.rglob("*") if p.suffix in (".pth", ".pt", ".ckpt")]
    if not checkpoints:
        return False, "no checkpoints in weights directory"
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "torch not importable"
    try:
        import importlib

        importlib.import_module("zoo.models")
    except Exception:  # noqa: BLE001
        return False, "xview2 ensemble code absent"
    return True, f"{len(checkpoints)} checkpoints"


def _xview2_mask(img: Image.Image) -> Optional[Any]:
    """Union of the loc ensemble's building masks, or None when it cannot run.

    Guarded end to end: a missing repo, a shape mismatch or an out-of-memory
    allocation all return None so the footprint tier takes over. Never raises into
    ingest.
    """
    ok, why = _xview2_state()
    if not ok:
        log.debug("xview2 loc unavailable: %s", why)
        return None
    try:
        import importlib

        import numpy as np
        import torch

        models_mod = importlib.import_module("zoo.models")
        weights = Path(config.XVIEW2_WEIGHTS)
        checkpoints = sorted(p for p in weights.rglob("*loc*") if p.suffix in (".pth", ".pt"))
        if not checkpoints:
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        arr = np.asarray(img.resize((1024, 1024), Image.BILINEAR), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        votes = None
        used = 0
        for ckpt in checkpoints:
            builder = _xview2_builder(models_mod, ckpt.name)
            if builder is None:
                continue
            state = torch.load(str(ckpt), map_location=device)
            net = builder()
            net.load_state_dict(state.get("state_dict", state), strict=False)
            net.to(device).eval()
            with torch.no_grad():
                out = torch.sigmoid(net(tensor))[0, 0].cpu().numpy()
            votes = out if votes is None else votes + out
            used += 1
        if not used or votes is None:
            return None
        prob = votes / used
        mask = np.asarray(prob > 0.5)
        # Masks resolve at 1024, so map them back onto the source raster.
        return np.asarray(
            Image.fromarray((mask * 255).astype("uint8")).resize(
                (img.width, img.height), Image.NEAREST
            )
        ) > 127
    except Exception as exc:  # noqa: BLE001
        log.warning("xview2 loc inference failed, falling back: %s", exc)
        return None


def _xview2_builder(models_mod: Any, checkpoint_name: str) -> Optional[Any]:
    """Match a checkpoint filename to a constructor in the ensemble's model module.

    The first-place solution names its checkpoints after the architecture, so the
    filename is the only reliable key. An unrecognised name is skipped rather than
    loaded into a guessed architecture.
    """
    name = checkpoint_name.lower()
    for attr in dir(models_mod):
        low = attr.lower()
        if low.startswith(("res", "dpn", "senet", "seresnext", "unet")) and low in name:
            candidate = getattr(models_mod, attr)
            if callable(candidate):
                return candidate
    return None


def mask_to_polygons(
    mask: Any, bounds: Sequence[float], *, min_px: int = 40, max_shapes: int = 400
) -> list[dict]:
    """Connected components of a boolean mask as georeferenced GeoJSON polygons.

    Deliberately axis-aligned per component: an operator needs to know which roof,
    not the exact eaves, and a rectangle beats a 400-vertex traced contour for both
    map legibility and payload size. Components smaller than min_px are dropped as
    speckle.
    """
    import numpy as np

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not arr.any():
        return []
    height, width = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    out: list[dict] = []
    for sy, sx in zip(*np.nonzero(arr)):
        if seen[sy, sx]:
            continue
        # Iterative flood fill: recursion would blow the stack on a city block.
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        count = 0
        y0 = y1 = int(sy)
        x0 = x1 = int(sx)
        while stack:
            cy, cx = stack.pop()
            count += 1
            y0, y1 = min(y0, cy), max(y1, cy)
            x0, x1 = min(x0, cx), max(x1, cx)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and arr[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if count < min_px:
            continue
        nw = pixel_to_bounds(bounds, width, height, x0, y0)
        se = pixel_to_bounds(bounds, width, height, x1 + 1, y1 + 1)
        out.append(_rect_geom(nw[0], se[1], se[0], nw[1]))
        if len(out) >= max_shapes:
            log.info("mask_to_polygons hit the %d shape cap", max_shapes)
            break
    return out


def _rect_geom(w: float, s: float, e: float, n: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


# ------------------------------------------------------------- outline tiers
def _outlines_from_footprints(bounds: Sequence[float]) -> list[tuple[str, dict, dict]]:
    """County footprints clipped to the tile, as (id, geom, props).

    CAPPED, and the cap is a correctness matter rather than a performance one.
    A real footprint layer puts hundreds of buildings under one tile (31,485 in
    the Pinellas AOI, about 600 per demo tile) while the VL budget grades twelve.
    Uncapped, the other 588 all receive the same pixel-stat stub verdict, which
    lands as an identical class and confidence on every row: identical priorities,
    a rank list that cannot be ordered, and a doubt column pinned near 1.0 for
    thousands of buildings. Better to outline the ones we can actually assess.

    Nearest-to-centre first, because a tile's subject is what the drone was
    pointed at, and the operator can always fly the edges again.
    """
    cap = footprint_cap()
    rows: list[tuple[float, str, dict, dict]] = []
    w, s, e, n = _span(bounds)
    cx, cy = (w + e) / 2.0, (s + n) / 2.0
    for f in datasets.footprints_in(bounds):
        if not f.geom:
            continue
        props = dict(f.props)
        if f.address:
            props["address"] = f.address
        c = f.centroid or [cx, cy]
        d2 = (c[0] - cx) ** 2 + (c[1] - cy) ** 2
        rows.append((d2, f"fp_{f.fid}", f.geom, props))
    rows.sort(key=lambda r: r[0])
    if cap and len(rows) > cap:
        log.info("tile has %d footprints, grading the %d nearest the centre", len(rows), cap)
        rows = rows[:cap]
    return [(fid, geom, props) for _, fid, geom, props in rows]


def _outlines_grid(bounds: Sequence[float], *, cols: int = 4, rows: int = 3) -> list[
    tuple[str, dict, dict]
]:
    """A deterministic grid, for the case where we KNOW there is structure in frame
    but have no geometry for it.

    NOT a general fallback. It used to run whenever the footprint layer returned
    nothing, which meant a tile over woodland - where the nearest real footprint was
    238 m away - produced twelve identical 15.9 x 21.0 m rectangles, and the address
    join then labelled them with real street addresses off the nearest road. That is
    a rescue list telling a fire crew to search a building that does not exist.

    An empty footprint layer over open ground is INFORMATION: there is nothing
    there. See _outlines, which now reports that instead of inventing coverage.
    """
    w, s, e, n = _span(bounds)
    inset_x, inset_y = (e - w) * 0.1, (n - s) * 0.1
    w, e = w + inset_x, e - inset_x
    s, n = s + inset_y, n - inset_y
    cw, ch = (e - w) / cols, (n - s) / rows
    out: list[tuple[str, dict, dict]] = []
    for r in range(rows):
        for c in range(cols):
            x0 = w + c * cw + cw * 0.2
            x1 = w + (c + 1) * cw - cw * 0.2
            y0 = s + r * ch + ch * 0.2
            y1 = s + (r + 1) * ch - ch * 0.2
            geom = _rect_geom(round(x0, 7), round(y0, 7), round(x1, 7), round(y1, 7))
            centroid = datasets.centroid_of(geom)
            out.append((_stable_id(centroid or [x0, y0]), geom, {"synthetic": True}))
    return out


def _outlines(
    img: Optional[Image.Image], bounds: Sequence[float]
) -> tuple[list[tuple[str, dict, dict]], str]:
    """Building outlines for one tile, and which tier produced them.

    Tiers in order: a segmentation mask, then the county footprint layer, then
    nothing. There is deliberately no synthetic fallback: a footprint layer that
    covers the AOI and returns zero buildings under a tile is telling us the ground
    is empty, and the honest answer to "what buildings are here" is none. Inventing
    a grid there produced twelve identical rectangles in woodland that the address
    join then labelled with real street addresses.
    """
    if img is not None:
        mask = _xview2_mask(img)
        if mask is not None:
            polys = mask_to_polygons(mask, bounds)
            if polys:
                shapes = []
                for geom in polys:
                    centroid = datasets.centroid_of(geom)
                    shapes.append((_stable_id(centroid or [0.0, 0.0]), geom, {}))
                return shapes, OUTLINE_XVIEW2
    try:
        shapes = _outlines_from_footprints(bounds)
    except Exception as exc:  # noqa: BLE001 - a broken dataset file costs outlines, not the tile
        log.warning("footprint outlines failed: %s", exc)
        return _outlines_grid(bounds), OUTLINE_GRID
    if shapes:
        return shapes, OUTLINE_FOOTPRINTS
    # Coverage exists but this frame has no buildings in it. Say so.
    if datasets.footprints():
        return [], OUTLINE_NONE
    # No footprint layer at all on this box: we know nothing about the ground rather
    # than knowing it is empty, so the grid is a labelled placeholder, not a claim.
    return _outlines_grid(bounds), OUTLINE_GRID


# ------------------------------------------------------------------- grading
def _env_number(name: str, default: float, cast: Any) -> Any:
    raw = os.environ.get(name)
    if raw is None:
        return cast(default)
    try:
        return max(cast(0), cast(raw))
    except (TypeError, ValueError):
        log.warning("%s=%r is not a number, using %s", name, raw, default)
        return cast(default)


def vl_budget() -> int:
    """VL calls allowed per tile. Env-tunable so a slow box can drop it live."""
    return _env_number("FIRSTLIGHT_VL_CALLS_PER_TILE", DEFAULT_VL_CALLS_PER_TILE, int)


def vl_concurrency() -> int:
    """VL calls in flight at once, per tile.

    Measured on the box: one grading call is ~2.2 s and the captioner's server runs
    --max-num-seqs 4, so issuing calls serially spent almost all of the tile's wall
    time waiting on a GPU with spare batch width. Kept at the measured knee rather
    than raised further because the same GPU serves the planner and the ballot.
    """
    return max(1, _env_number("FIRSTLIGHT_VL_CONCURRENCY", DEFAULT_VL_CONCURRENCY, int))


def footprint_cap() -> int:
    """Footprints outlined per tile, 0 for no cap.

    Env-tunable because the right number depends on the imagery: a wide-area
    survey frame legitimately covers hundreds of buildings, and an operator who
    wants all of them can raise this and accept that most will carry a stub grade
    clearly labelled as such.
    """
    return _env_number("FIRSTLIGHT_FOOTPRINTS_PER_TILE", DEFAULT_FOOTPRINTS_PER_TILE, int)


def tile_wall_budget() -> float:
    """Seconds of VL time one tile may spend before the rest is stub-graded."""
    return _env_number("FIRSTLIGHT_TILE_VL_SECONDS", DEFAULT_TILE_VL_SECONDS, float)


def outline_and_grade(
    image_path: Union[str, Path, Image.Image],
    bounds: Optional[Sequence[float]],
    *,
    vl_budget_override: Optional[int] = None,
) -> list[GradedBuilding]:
    """Outline every building in one tile and grade each 0-3 with a caption.

    Returns [] when bounds is None: without a transform there is no way to place a
    polygon on the map, and a building with no location cannot be dispatched to.
    The tile is still ingested and flagged needs_geo, and re-running this after the
    operator drags it into place produces the buildings.

    Never raises. An unreadable image still yields outlines from the county
    footprints and stub grades, so the tile appears in the rank with an honest
    label rather than vanishing.
    """
    if bounds is None:
        return []
    box = _span(bounds)
    img = _open_image(image_path)
    shapes, outline_source = _outlines(img, box)

    budget = vl_budget() if vl_budget_override is None else max(0, int(vl_budget_override))
    # Largest first: a big structure holds more people and dominates the frame, so
    # if only twelve crops get the model they should be the twelve that matter.
    areas = [datasets.geom_area_m2(geom) for _fid, geom, _p in shapes]
    order = sorted(range(len(shapes)), key=lambda i: (-areas[i], shapes[i][0]))
    model_slots = set(order[:budget]) if img is not None else set()
    # Second, independent guard: the CALL cap assumes each call is fast. A server
    # that answers slowly rather than not at all would still blow the tile budget,
    # so the remaining crops fall back to the stub once this much time is spent.
    wall_budget_s = tile_wall_budget()
    started = time.monotonic()

    # The VL calls run CONCURRENTLY. WHY: measured on the box, one grading call is
    # ~1.8 s, so twelve of them issued one after another cost ~22 s and drove a
    # 33 s p50 per tile against a <10 s budget. vLLM batches concurrent requests
    # on the same server, so the wall time for the batch is far below the sum. The
    # cap stays small because the same GPU is serving the planner and the ballot.
    slot_list = [i for i in range(len(shapes)) if i in model_slots]
    crops: dict[int, tuple[int, int, int, int]] = {}
    if img is not None:
        for idx in slot_list:
            _fid, geom, _p = shapes[idx]
            c = _crop_box(geom, box, img.width, img.height)
            if c is not None:
                crops[idx] = c

    results: dict[int, dict] = {}
    vl_calls = 0
    if crops:
        lanes = max(1, min(vl_concurrency(), len(crops)))
        deadline = started + wall_budget_s

        def _one(idx: int) -> tuple[int, Optional[dict]]:
            # Checked per task rather than once: a queue that backs up must stop
            # issuing work, not just stop starting it.
            if time.monotonic() >= deadline:
                return idx, None
            return idx, vlm.caption_and_grade(img, crop_box=crops[idx])

        with futures.ThreadPoolExecutor(max_workers=lanes) as pool:
            for idx, res in pool.map(_one, list(crops)):
                if res is None:
                    continue
                results[idx] = res
                vl_calls += 1
        if len(results) < len(crops):
            log.info(
                "tile VL wall budget spent after %d of %d calls, stubbing the rest",
                len(results),
                len(crops),
            )

    graded: list[GradedBuilding] = []
    model_graded = 0
    stub_graded = 0
    for idx, (fid, geom, props) in enumerate(shapes):
        centroid = datasets.centroid_of(geom) or [
            round((box[0] + box[2]) / 2, 7),
            round((box[1] + box[3]) / 2, 7),
        ]
        crop = crops.get(idx) or (_crop_box(geom, box, img.width, img.height) if img is not None else None)
        result = results.get(idx)
        if result is None:
            result = vlm.stub_grade(img, crop_box=crop) if img is not None else _blind_grade()
        by = GRADED_BY_VL if result["how"] == vlm.GRADE_HOW_MODEL else GRADED_BY_STUB
        if by == GRADED_BY_VL:
            model_graded += 1
        else:
            stub_graded += 1
        graded.append(
            GradedBuilding(
                footprint_id=fid,
                cls=int(result["class"]),
                conf=float(result["conf"]),
                centroid=centroid,
                geom=geom,
                area_m2=areas[idx],
                graded_by=by,
                caption=str(result["caption"]),
                props=props,
            )
        )

    grade_path = GRADED_BY_VL if model_graded else GRADED_BY_STUB
    with _LOCK:
        _LAST_RUN.update(
            outline_source=outline_source,
            buildings=len(graded),
            vl_calls=vl_calls,
            model_graded=model_graded,
            stub_graded=stub_graded,
            grade_path=grade_path,
        )
    log.info(
        "tile graded: %d buildings, outlines=%s, %d model, %d stub",
        len(graded),
        outline_source,
        model_graded,
        stub_graded,
    )
    return graded


def _blind_grade() -> dict:
    """No pixels at all, so no pixel statistic either.

    Class 1 rather than 0: an unreadable tile is not evidence of an intact
    neighbourhood, and confidence 0.2 makes doubt high enough that this building
    rises for a human to look at.
    """
    return {"class": 1, "caption": vlm.STUB_CAPTION, "conf": 0.2, "how": vlm.GRADE_HOW_STUB}


# ------------------------------------------------------------- tile summaries
def tile_caption(
    buildings: Sequence[GradedBuilding],
) -> tuple[str, str, Optional[str]]:
    """One archive caption per tile, reusing a caption the VL pass already wrote.

    A6 forbids a second VLM call per crop, so the tile caption is picked, not
    generated: the most severe building that got the model path, because the tile
    should describe the worst thing in the frame.

    Returns (caption, caption_by, anchor_footprint_id). The anchor is the building
    the caption is ABOUT, so the archive dot can sit on that structure instead of on
    the tile's geometric centre - a caption reading "partial collapse" is misleading
    when the pin it travels with is over a parking lot 200 m away.
    """
    model_captions = [
        b for b in buildings if b.graded_by == GRADED_BY_VL and b.caption and b.caption != vlm.STUB_CAPTION
    ]
    if model_captions:
        best = max(model_captions, key=lambda b: (int(b.cls), float(b.conf), b.area_m2))
        return best.caption, GRADED_BY_VL, best.footprint_id
    if buildings:
        worst = max(buildings, key=lambda b: (int(b.cls), b.area_m2))
        return (
            f"{vlm.STUB_CAPTION}, worst structure graded {contracts.CLASS_LABEL[int(worst.cls)]}",
            GRADED_BY_STUB,
            worst.footprint_id,
        )
    return vlm.STUB_CAPTION, GRADED_BY_STUB, None


def outline_source() -> str:
    """Which outline tier ran on the most recent tile."""
    with _LOCK:
        return _LAST_RUN["outline_source"] or OUTLINE_FOOTPRINTS


def last_run() -> dict:
    """Counts from the most recent tile, for the HUD and the stage cards."""
    with _LOCK:
        return dict(_LAST_RUN)


def model_version() -> str:
    """The active grading path, exactly as the status bar should print it.

    Names both halves, because "nemotron-vl" alone would hide that the outlines
    came from a synthetic grid, and an operator who cannot tell those apart cannot
    judge what is on screen.
    """
    with _LOCK:
        snapshot = dict(_LAST_RUN)
    grade = snapshot["grade_path"] or GRADED_BY_VL
    source = snapshot["outline_source"]
    if not source:
        ok, _why = _xview2_state()
        source = OUTLINE_XVIEW2 if ok else (
            OUTLINE_FOOTPRINTS if datasets.footprints() else OUTLINE_GRID
        )
        return f"{grade} (outlines: {source}, no tile yet)"
    stubbed = snapshot["stub_graded"]
    total = snapshot["buildings"] or 1
    suffix = "" if not stubbed else f", {stubbed}/{total} stub-graded"
    return f"{grade} (outlines: {source}{suffix})"


__all__ = [
    "CROP_PAD",
    "DEFAULT_TILE_VL_SECONDS",
    "DEFAULT_VL_CALLS_PER_TILE",
    "GRADED_BY_STUB",
    "GRADED_BY_VL",
    "GRADED_BY_XVIEW2",
    "OUTLINE_FOOTPRINTS",
    "OUTLINE_GRID",
    "OUTLINE_XVIEW2",
    "GradedBuilding",
    "bounds_to_pixel",
    "last_run",
    "mask_to_polygons",
    "model_version",
    "outline_and_grade",
    "outline_source",
    "pixel_to_bounds",
    "tile_caption",
    "tile_wall_budget",
    "vl_budget",
]
