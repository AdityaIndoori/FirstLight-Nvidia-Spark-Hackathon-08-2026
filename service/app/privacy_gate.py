"""Privacy gate, plan deliverable A2. It guards STORAGE, never analysis.

Every tile is analyzed, because a person in frame is rescue signal and throwing
that away would blind the triage this system exists to produce. The gate sits in
front of the archive writer instead: person signal, or any detector failure at
all, means the image is never stored, never indexed, never thumbnailed and never
searchable.

Two rules make the published privacy claim defensible:

1. Fail-safe. `check()` never raises and never returns None. Doubt withholds, so
   a wedged detector, a corrupt JPEG or a missing weights file costs an operator
   review click rather than costing the claim.
2. Tiled inference. A person in a 4000 px aerial tile downscaled to a 640 px
   detector input is about 5 px tall and no detector sees that. We slice the tile
   into overlapping 1280 px crops, infer at crop resolution, translate the boxes
   back to full-image coordinates and union the verdicts. The tiled path is the
   one measured in A5, so the recall number we publish is the recall operators
   actually get.
"""
from __future__ import annotations

# PUBLIC API
#   check(image, *, conf: float | None = None, tiled: bool = True) -> GateVerdict
#       image is a path (str / os.PathLike) or a PIL.Image.Image. Never raises,
#       never returns None. Call this FIRST in the archive writer and refuse to
#       write any row, thumbnail, caption or embedding when store_ok is False.
#   GateVerdict(store_ok, person_detections, all_detections, detector_error,
#               took_ms, tiles_scanned, conf_threshold, tiled)
#       .withheld_reason() -> str | None   human-readable, for the withheld vault
#       .summary()         -> dict         log-safe counts, no filename, no boxes
#   available() -> bool          detector loadable right now (loads it, warms it)
#   model_version() -> str      status-strip string, cheap, never loads the model
#   is_person_class(det) -> bool
#   reset() -> None             drop the cached model, for tests and the eval CLI
#   GateUnavailable             raised internally, always becomes a withhold
#
# Detection dicts are {"cls": int, "name": str, "conf": float,
#                      "bbox": [x1, y1, x2, y2]} in FULL-image pixel coordinates.
# No return value here ever carries a filename or a path: withheld_reason() and
# summary() are path-scrubbed, because the only surface allowed to name a
# withheld image is the token-guarded review endpoint.

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

from . import config

ImageInput = Union[str, "os.PathLike[str]", Image.Image]


# A detector error message routinely quotes the path it choked on, and a withheld
# filename belongs in exactly one place: the token-guarded review endpoint. Every
# error that reaches a log row, a tile row or the UI goes through here first.
_PATHY = re.compile(r"""['"(]?(?:[A-Za-z]:)?[^\s'"()]*[\\/][^\s'"()]*['")]?""")
_FILEY = re.compile(r"""['"]?[\w.\-]+\.(?:jpe?g|png|tiff?|webp|bmp|jp2|pt)['"]?""", re.I)


def _scrub(msg: str) -> str:
    return _FILEY.sub("<path>", _PATHY.sub("<path>", str(msg))).strip()


def _scrub_named(msg: str, image: ImageInput) -> str:
    """Same, plus the exact source string when we have it. A filename with spaces
    in it survives the generic pattern, and 'Mrs Alvarez 3rd Ave.jpg' is precisely
    the kind of name that must not reach an unauthenticated export."""
    text = str(msg)
    if not isinstance(image, Image.Image):
        try:
            raw = os.fspath(image)
        except TypeError:
            return _scrub(text)
        for token in (raw, str(Path(raw)), Path(raw).name, Path(raw).stem):
            if token:
                text = text.replace(token, "<path>")
    return _scrub(text)


class GateUnavailable(RuntimeError):
    """The detector cannot run at all. Raised inside `_detect`, caught by
    `check()`, and it always resolves to a withhold: no detector, no storage."""


@dataclass
class GateVerdict:
    """The storage decision for one image, plus the evidence behind it."""

    store_ok: bool
    person_detections: list[dict] = field(default_factory=list)
    all_detections: list[dict] = field(default_factory=list)
    detector_error: Optional[str] = None
    took_ms: int = 0
    tiles_scanned: int = 0
    conf_threshold: float = 0.0
    tiled: bool = False

    def withheld_reason(self) -> Optional[str]:
        """None when the image may be stored, else the vault's reason string.

        Path-scrubbed: this string is persisted on the tile row and rendered in
        the operator UI, and the only surface allowed to name a withheld file is
        the authorized review endpoint.
        """
        if self.store_ok:
            return None
        if self.detector_error:
            return f"detector error: {_scrub(self.detector_error)}"
        n = len(self.person_detections)
        if n:
            return f"person signal: {n} detection{'' if n == 1 else 's'}"
        return "withheld by policy"

    def summary(self) -> dict:
        """Log-safe. GET /api/export/decision-log.json is unauthenticated by
        design, so this carries counts and a channel name only: no filename, no
        boxes, and no per-detection confidence that could fingerprint one
        withheld image. A bounding box is a description of where a person was
        standing, and a confidence vector is nearly as identifying."""
        return {
            "store_ok": bool(self.store_ok),
            "channel": "pixels",
            "persons": len(self.person_detections),
            "detections": len(self.all_detections),
            "tiles_scanned": int(self.tiles_scanned),
            "took_ms": int(self.took_ms),
            "conf": float(self.conf_threshold),
            "tiled": bool(self.tiled),
            "detector_error": _scrub(self.detector_error) if self.detector_error else None,
            "gate": model_version(),
        }


# ---------------------------------------------------------------- model state
# Loaded lazily and exactly once. Ingest runs several tiles concurrently, and
# without the lock two threads would each pull the weights into memory on a box
# whose real constraint is bandwidth, not capacity.
_model: Any = None
_model_lock = threading.Lock()
_load_error: Optional[str] = None
_load_done = False


def _load() -> tuple[Any, Optional[str]]:
    """Import ultralytics and open the weights. Returns (model, error): a
    missing detector is a state, not a crash, because the service must still
    boot and still withhold."""
    weights = Path(config.GATE_WEIGHTS)
    if not weights.is_file():
        return None, f"weights missing: {weights}"
    # This box is offline by policy. Say so to ultralytics too, so a version
    # ping or an auto-install cannot make the first tile wait on a DNS timeout.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_VERBOSE", "false")
    try:
        from ultralytics import YOLO
    except Exception as exc:  # ImportError, but a broken install raises others
        return None, f"ultralytics unavailable: {exc.__class__.__name__}: {exc}"
    try:
        return YOLO(str(weights)), None
    except Exception as exc:
        return None, f"weights load failed: {exc.__class__.__name__}: {exc}"


def _ensure_model() -> Any:
    """Double-checked lazy load. Raises GateUnavailable when the detector is
    unusable, which the caller turns into a withhold."""
    global _model, _load_error, _load_done
    if not _load_done:
        with _model_lock:
            if not _load_done:
                _model, _load_error = _load()
                _load_done = True
    if _load_error:
        raise GateUnavailable(_load_error)
    if _model is None:
        raise GateUnavailable("detector not loaded")
    return _model


def reset() -> None:
    """Forget the cached model and any cached load failure. Tests and the eval
    harness use this; nothing on the request path should."""
    global _model, _load_error, _load_done
    with _model_lock:
        _model = None
        _load_error = None
        _load_done = False


def available() -> bool:
    """True when the detector is loadable. Loads it, so the first status poll
    doubles as a warm-up."""
    try:
        _ensure_model()
        return True
    except Exception:
        return False


def model_version() -> str:
    """Status-strip string naming the actual weights file, the person classes and
    the threshold in force. Deliberately cheap: a status poll must never block on
    a model load, so this reports configured state, not a live handle."""
    name = Path(config.GATE_WEIGHTS).name
    unusable = (_load_done and _load_error) or not Path(config.GATE_WEIGHTS).is_file()
    if unusable:
        return f"{name} UNAVAILABLE, withholding all storage"
    classes = ",".join(str(c) for c in sorted(config.GATE_PERSON_CLASSES))
    overlap = int(round(float(config.GATE_TILE_OVERLAP) * 100))
    return (
        f"{name}, person classes {{{classes}}}, conf>={float(config.GATE_CONF):.2f}, "
        f"tiled {int(config.GATE_TILE)}px/{overlap}%"
    )


def is_person_class(det: dict) -> bool:
    """Config is read per call: the A5 sweep moves these values, and a threshold
    frozen at import time would make the sweep report numbers nobody gets."""
    try:
        return int(det.get("cls", -1)) in config.GATE_PERSON_CLASSES
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- inference
def _imgsz() -> int:
    """Infer at crop resolution. Feeding a 1280 px crop to a 640 px network
    would re-introduce the exact downscale that tiling exists to remove."""
    tile = max(320, int(config.GATE_TILE))
    return ((tile + 31) // 32) * 32


def _class_name(names: Any, cid: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(cid, cid))
        if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
            return str(names[cid])
    except Exception:
        pass
    return str(cid)


def _detect(image: Image.Image, conf: float) -> list[dict]:
    """One detector pass over one image or crop. Boxes come back in the
    coordinates of what was passed in; `check()` translates them. This is the
    single seam tests monkeypatch, so nothing else may talk to the model."""
    model = _ensure_model()
    # PIL in, not numpy: ultralytics reads numpy arrays as BGR, and a silent
    # channel swap in the one model that guards privacy is not a risk worth
    # taking for the copy it would save.
    results = model.predict(source=image, conf=conf, imgsz=_imgsz(), verbose=False)
    names = getattr(model, "names", None)
    out: list[dict] = []
    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
            cid = int(cls_t)
            x1, y1, x2, y2 = (float(v) for v in xyxy_t)
            out.append(
                {
                    "cls": cid,
                    "name": _class_name(names, cid),
                    "conf": float(conf_t),
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return out


def _starts(extent: int, tile: int, step: int) -> list[int]:
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, step))
    if starts[-1] + tile < extent:
        # Flush the final crop against the edge. An uncovered strip is exactly
        # where an unseen person ends up, and a partial crop would be worse.
        starts.append(extent - tile)
    return starts


def _crop_boxes(w: int, h: int, tiled: bool) -> list[tuple[int, int, int, int]]:
    """SAHI-style overlapping slice plan, or one whole-image box when the tile
    is small enough that a single pass already resolves people."""
    tile = max(32, int(config.GATE_TILE))
    if not tiled or max(w, h) <= int(config.GATE_TILE_MIN_SIDE):
        return [(0, 0, w, h)]
    overlap = min(max(float(config.GATE_TILE_OVERLAP), 0.0), 0.9)
    step = max(1, int(round(tile * (1.0 - overlap))))
    return [
        (x, y, min(x + tile, w), min(y + tile, h))
        for y in _starts(h, tile, step)
        for x in _starts(w, tile, step)
    ]


def _iou(a: Any, b: Any) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge(dets: list[dict], iou_min: float = 0.55) -> list[dict]:
    """Overlapping crops see the same person twice. Fold the duplicates so a
    published count means people, not passes. This can never change a verdict:
    the merge of two person boxes is still a person box."""
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: -float(x.get("conf", 0.0))):
        dup = any(
            d.get("cls") == k.get("cls") and _iou(d.get("bbox"), k.get("bbox")) >= iou_min
            for k in kept
        )
        if not dup:
            kept.append(d)
    return kept


def _open(image: ImageInput) -> tuple[Image.Image, bool]:
    """Returns (rgb_image, owned). We never close an image the caller owns."""
    if isinstance(image, Image.Image):
        if image.mode == "RGB":
            return image, False
        return image.convert("RGB"), True
    img = Image.open(os.fspath(image))
    # Force the decode here, where a truncated file becomes a withhold instead
    # of an exception thrown at the detector three frames later.
    img.load()
    if img.mode != "RGB":
        rgb = img.convert("RGB")
        img.close()
        return rgb, True
    return img, True


def check(
    image: ImageInput,
    *,
    conf: Optional[float] = None,
    tiled: bool = True,
) -> GateVerdict:
    """Decide whether one image may be STORED. Never raises, never returns None.

    store_ok is False when any detection of a class in config.GATE_PERSON_CLASSES
    lands at or above the confidence threshold, and equally on any exception at
    all: unreadable file, missing weights, detector fault, malformed output.

    The whole slice plan is scanned even after the first person is found. The
    verdict is already settled at that point, but the authorized review endpoint
    and the A5 eval both need every detection, not just the earliest one.
    """
    t0 = time.perf_counter()
    threshold = float(config.GATE_CONF) if conf is None else float(conf)
    dets: list[dict] = []
    error: Optional[str] = None
    scanned = 0
    tiling_used = False
    img: Optional[Image.Image] = None
    owned = False
    try:
        img, owned = _open(image)
        boxes = _crop_boxes(img.width, img.height, tiled)
        tiling_used = len(boxes) > 1
        whole = (0, 0, img.width, img.height)
        for box in boxes:
            crop = img if box == whole else img.crop(box)
            try:
                found = _detect(crop, threshold)
            finally:
                if crop is not img:
                    crop.close()
            ox, oy = box[0], box[1]
            for d in found:
                x1, y1, x2, y2 = d["bbox"]
                d["bbox"] = [
                    round(x1 + ox, 1),
                    round(y1 + oy, 1),
                    round(x2 + ox, 1),
                    round(y2 + oy, 1),
                ]
                dets.append(d)
            scanned += 1
    except Exception as exc:
        # Scrub with the source name in hand. The regex catches paths in general,
        # but a filename with spaces in it needs the exact string, and we have it
        # right here. Nothing downstream ever sees the name.
        error = _scrub_named(f"{exc.__class__.__name__}: {exc}", image)
    finally:
        if img is not None and owned:
            img.close()

    merged = _merge(dets)
    persons = [
        d for d in merged if is_person_class(d) and float(d.get("conf", 0.0)) >= threshold
    ]
    return GateVerdict(
        store_ok=(error is None and not persons),
        person_detections=persons,
        all_detections=merged,
        detector_error=error,
        took_ms=int(round((time.perf_counter() - t0) * 1000)),
        tiles_scanned=scanned,
        conf_threshold=round(threshold, 3),
        tiled=tiling_used,
    )


__all__ = [
    "GateUnavailable",
    "GateVerdict",
    "available",
    "check",
    "is_person_class",
    "model_version",
    "reset",
]
