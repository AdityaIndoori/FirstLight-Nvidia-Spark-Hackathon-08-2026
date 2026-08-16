"""Paths, thresholds and model endpoints. Everything overridable by env."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FIRSTLIGHT_DATA", ROOT / "data"))
WEB = ROOT / "web"

WATCH_DIR = DATA / "watch"
ANALYZED_DIR = DATA / "analyzed"
WITHHELD_DIR = DATA / "withheld"
THUMB_DIR = DATA / "thumbs"
DATASET_DIR = DATA / "datasets"
MODEL_DIR = Path(os.environ.get("FIRSTLIGHT_MODELS", DATA / "models"))
DB_PATH = Path(os.environ.get("FIRSTLIGHT_DB", DATA / "firstlight.db"))

for d in (WATCH_DIR, ANALYZED_DIR, WITHHELD_DIR, THUMB_DIR, DATASET_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- inference
# Three local vLLM servers, per the measured Friday-night configuration.
NANO_URL = os.environ.get("FIRSTLIGHT_NANO_URL", "http://127.0.0.1:8000/v1")
NANO_MODEL = os.environ.get("FIRSTLIGHT_NANO_MODEL", "nano")
LIGHTNING_URL = os.environ.get("FIRSTLIGHT_LIGHTNING_URL", "http://127.0.0.1:8001/v1")
LIGHTNING_MODEL = os.environ.get("FIRSTLIGHT_LIGHTNING_MODEL", "lightning")
VL_URL = os.environ.get("FIRSTLIGHT_VL_URL", "http://127.0.0.1:8002/v1")
VL_MODEL = os.environ.get("FIRSTLIGHT_VL_MODEL", "captioner")

# Hard cap on every model call. A wedged endpoint must never stall ingest.
LLM_TIMEOUT_S = float(os.environ.get("FIRSTLIGHT_LLM_TIMEOUT", "10"))
VL_TIMEOUT_S = float(os.environ.get("FIRSTLIGHT_VL_TIMEOUT", "20"))

# ---------------------------------------------------------------- privacy gate
# A2/A5: conservative by construction. Person classes only; any signal or any
# detector error withholds the image FROM STORAGE (analysis always proceeds).
GATE_WEIGHTS = os.environ.get(
    "FIRSTLIGHT_GATE_WEIGHTS", str(Path.home() / "firstlight" / "visdrone-yolov8x.pt")
)
# SET FROM THE MEASUREMENT, which is what A5 exists for. Swept on 100 held-out
# real aerial frames, 50 with people and 50 without, through the tiled path:
#
#   conf   recall   precision   false clears   false withholds
#   0.25    98.0%      76.6%          1              15
#   0.50    98.0%      86.0%          1               8
#   0.60    96.0%      87.3%          2               7
#   0.70    88.0%      95.7%          6               2
#
# 0.50 dominates 0.25: same recall, same single false clear, and precision rises
# from 76.6 to 86.0 while false withholds halve. Going higher starts trading away
# recall, and a false clear costs the privacy claim while a false withhold costs an
# operator one review click, so recall is the side to protect.
GATE_CONF = float(os.environ.get("FIRSTLIGHT_GATE_CONF", "0.5"))
# VisDrone: 0 pedestrian, 1 people. COCO fallback: 0 person.
GATE_PERSON_CLASSES = {
    int(c) for c in os.environ.get("FIRSTLIGHT_GATE_CLASSES", "0,1").split(",") if c.strip()
}
# A2: tiled inference. A person downscaled to 640 px is about 5 px tall.
GATE_TILE = int(os.environ.get("FIRSTLIGHT_GATE_TILE", "1280"))
GATE_TILE_OVERLAP = float(os.environ.get("FIRSTLIGHT_GATE_OVERLAP", "0.2"))
GATE_TILE_MIN_SIDE = int(os.environ.get("FIRSTLIGHT_GATE_TILE_MIN", "1600"))

EMBED_MODEL = os.environ.get("FIRSTLIGHT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
EMBED_DEVICE = os.environ.get("FIRSTLIGHT_EMBED_DEVICE", "cpu")  # A6: pinned, GPU OOMs

XVIEW2_WEIGHTS = Path(
    os.environ.get("FIRSTLIGHT_XVIEW2", str(Path.home() / "firstlight" / "weights"))
)

REVIEW_TOKEN = os.environ.get("FIRSTLIGHT_REVIEW_TOKEN", "")

# ---------------------------------------------------------------- AOI
# Area of operations, [w, s, e, n]. See README section 9 for why these counties.
#
# Default is Pinellas County FL over ground Hurricane Milton actually crossed:
# the county publishes a live gray-sky road-closure service, which is the only
# real blocked-roads feed any candidate county had.
#
# Named alternatives, switch with FIRSTLIGHT_AOI:
#   Bay County FL (Michael 2018, Panama City): -85.72,30.13,-85.62,30.22
#     the one AOI with a true pre/post aerial pair from one source, so the
#     xView2 cls path can actually run on six-channel input.
#   Sarasota FL (Milton 2024):                 -82.56,27.30,-82.48,27.38
#     34,620 county building footprints, where Pinellas has none.
AOI_PRESETS = {
    "pinellas": [-82.78, 27.75, -82.70, 27.82],
    "bay": [-85.72, 30.13, -85.62, 30.22],
    "sarasota": [-82.56, 27.30, -82.48, 27.38],
}
_aoi = os.environ.get("FIRSTLIGHT_AOI", "pinellas")
AOI = AOI_PRESETS.get(_aoi.strip().lower()) or [float(x) for x in _aoi.split(",")]
AOI_NAME = _aoi if _aoi in AOI_PRESETS else "custom"

STALENESS_CAP_H = 12.0
