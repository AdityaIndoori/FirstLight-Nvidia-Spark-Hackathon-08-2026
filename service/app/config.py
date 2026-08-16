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
GATE_CONF = float(os.environ.get("FIRSTLIGHT_GATE_CONF", "0.25"))
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
# West Seattle demo area of operations, [w, s, e, n].
AOI = [
    float(x)
    for x in os.environ.get("FIRSTLIGHT_AOI", "-122.42,47.52,-122.36,47.58").split(",")
]

STALENESS_CAP_H = 12.0
