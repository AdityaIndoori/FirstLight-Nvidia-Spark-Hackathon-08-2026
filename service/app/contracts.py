"""FIRST LIGHT wire contracts, section 7 of the plan, frozen.

Every module consumes and emits these shapes. If a field is missing here, add it
here first, then tell the other two members.

Damage classes are integers 0-3: 0 no-damage, 1 minor, 2 major, 3 destroyed.
"Severe" means class >= 2. Never strings on the wire.
Coordinates are [lng, lat], GeoJSON order. Bounds are [west, south, east, north].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

# ---------------------------------------------------------------- damage class
CLASS_NO_DAMAGE = 0
CLASS_MINOR = 1
CLASS_MAJOR = 2
CLASS_DESTROYED = 3
SEVERE_FROM = CLASS_MAJOR

CLASS_LABEL = {0: "no damage", 1: "minor damage", 2: "major damage", 3: "destroyed"}

# B1: severity_weight is non-optional. Without it an intact class-0 building the
# models cannot agree on outranks an unconfirmed class-3 collapse they agree about.
SEVERITY_WEIGHT = {0: 0.25, 1: 0.5, 2: 1.0, 3: 1.5}

# Section 7 display-name mapping, so UI labels never drift from wire fields.
DISPLAY_NAME = {
    "severity_weight": "damage severity",
    "staleness_h": "hours since last look",
    "vulnerable_density": "resident vulnerability",
    "doubt": "AI uncertainty",
    "road_cutoff": "road cut-off",
    "priority": "priority",
}

DOUBT_FLOOR = 0.05

AGENCIES = ("fire", "ems", "police", "public_works")
FACILITY_TYPES = ("nursing_home", "dialysis", "hospital")

TileStatus = Literal["processed", "withheld", "error", "needs_geo"]


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None or k in _KEEP_NULL}


_KEEP_NULL = {"road_cutoff", "facility_near", "centroid", "warning", "recovery"}


@dataclass
class Building:
    id: str
    cls: int
    conf: float

    def wire(self) -> dict:
        return {"id": self.id, "class": int(self.cls), "conf": round(float(self.conf), 3)}


@dataclass
class TileRecord:
    """A to B. `status` describes the ANALYSIS outcome.

    Storage is a separate decision: `stored` is False when the privacy gate
    withheld the image from the archive. A withheld tile still carries its
    buildings, because a person in frame is rescue signal, not a reason to
    discard data.
    """

    filename: str
    bounds: Optional[list[float]]
    status: TileStatus
    captured_at: float
    latency_ms: int
    buildings: list[Building] = field(default_factory=list)
    stored: bool = True
    withheld_reason: Optional[str] = None
    needs_geo: bool = False

    def wire(self) -> dict:
        return {
            "filename": self.filename,
            "bounds": self.bounds,
            "status": self.status,
            "captured_at": float(self.captured_at),
            "latency_ms": int(self.latency_ms),
            "buildings": [b.wire() for b in self.buildings],
            "stored": bool(self.stored),
            "withheld_reason": self.withheld_reason,
            "needs_geo": bool(self.needs_geo),
        }


@dataclass
class RankInputs:
    severity_weight: float
    staleness_h: float
    vulnerable_density: float
    doubt: float
    road_cutoff: Optional[float] = None

    def wire(self) -> dict:
        return {
            "severity_weight": round(self.severity_weight, 3),
            "staleness_h": round(self.staleness_h, 3),
            "vulnerable_density": round(self.vulnerable_density, 3),
            "doubt": round(self.doubt, 3),
            "road_cutoff": None if self.road_cutoff is None else round(self.road_cutoff, 3),
        }


def priority_of(inputs: RankInputs) -> float:
    """Reconciliation rule, gate 4 depends on it.

    Round each factor to 3 decimals FIRST, then multiply, then round the product
    to 5. Member C displays these same rounded factors so a judge with a
    calculator always agrees.
    """
    w = inputs.wire()
    product = (
        w["severity_weight"]
        * w["staleness_h"]
        * w["vulnerable_density"]
        * w["doubt"]
        * (w["road_cutoff"] if w["road_cutoff"] is not None else 1.0)
    )
    return round(product, 5)


@dataclass
class FacilityNear:
    name: str
    type: str
    dist_m: int

    def wire(self) -> dict:
        return {"name": self.name, "type": self.type, "dist_m": int(self.dist_m)}


@dataclass
class RankItem:
    footprint_id: str
    label: str
    centroid: list[float]
    damage_class: int
    confidence: float
    confirmed: bool
    graded_by: str
    inputs: RankInputs
    priority: float
    rationale: str = ""
    rationale_by: Optional[str] = None
    facility_near: Optional[FacilityNear] = None
    image_ids: list[str] = field(default_factory=list)
    votes: Optional[list[int]] = None
    vote_agreement: Optional[float] = None

    def wire(self) -> dict:
        return {
            "footprint_id": self.footprint_id,
            "label": self.label,
            "centroid": self.centroid,
            "damage_class": int(self.damage_class),
            "confidence": round(float(self.confidence), 3),
            "confirmed": bool(self.confirmed),
            "graded_by": self.graded_by,
            "facility_near": self.facility_near.wire() if self.facility_near else None,
            "inputs": self.inputs.wire(),
            "priority": float(self.priority),
            "rationale": self.rationale,
            "rationale_by": self.rationale_by,
            "image_ids": self.image_ids,
            "votes": self.votes,
            "vote_agreement": self.vote_agreement,
        }


@dataclass
class ArchiveItem:
    image_id: str
    thumb_path: str
    captured_at: float
    centroid: Optional[list[float]]
    needs_geo: bool
    caption: str
    tags: list[str]
    class_max: int
    key_evidence: bool = False
    footprint_ids: list[str] = field(default_factory=list)
    # Cosine similarity to the query when the semantic resolver ranked this row,
    # None otherwise. Shown in the panel so an operator sees WHY a result placed
    # where it did rather than being asked to trust an ordering.
    score: Optional[float] = None

    def wire(self) -> dict:
        return {
            "image_id": self.image_id,
            "thumb_path": self.thumb_path,
            "captured_at": float(self.captured_at),
            "centroid": self.centroid,
            "needs_geo": bool(self.needs_geo),
            "caption": self.caption,
            "tags": list(self.tags),
            "class_max": int(self.class_max),
            "key_evidence": bool(self.key_evidence),
            "footprint_ids": list(self.footprint_ids),
            "score": self.score,
        }


def status_payload(**kw: Any) -> dict:
    """Status, all to C. Shape is fixed; missing pieces report as zero/None."""
    return {
        "tiles_analyzed": kw.get("tiles_analyzed", 0),
        "tiles_stored": kw.get("tiles_stored", 0),
        "tiles_withheld_from_storage": kw.get("tiles_withheld_from_storage", 0),
        "tiles_error": kw.get("tiles_error", 0),
        "tile_latency_ms_p50": kw.get("tile_latency_ms_p50", 0),
        "tally": kw.get("tally", {}),
        "model_versions": kw.get("model_versions", {}),
        "tokens_per_s": kw.get("tokens_per_s", {}),
        "memory_gb": kw.get("memory_gb", 0.0),
        "memory_total_gb": kw.get("memory_total_gb", 0.0),
        "gpu_power": kw.get("gpu_power", ""),
        "last_replan_ms": kw.get("last_replan_ms", 0),
        "recovery": kw.get("recovery"),
        "doubt_distribution": kw.get("doubt_distribution", {}),
        "datasets": kw.get("datasets", []),
        # The AOI is served, never hardcoded in the frontend: a stale copy there
        # opens the map over the wrong ocean and makes the basemap-cache note lie.
        "aoi": kw.get("aoi"),
        "aoi_name": kw.get("aoi_name", "custom"),
        "openshell": kw.get(
            "openshell", {"policy": "not-wired", "denials": 0, "allows": 0, "audit": []}
        ),
    }


__all__ = [
    "AGENCIES",
    "ArchiveItem",
    "Building",
    "CLASS_LABEL",
    "DISPLAY_NAME",
    "DOUBT_FLOOR",
    "FACILITY_TYPES",
    "FacilityNear",
    "RankInputs",
    "RankItem",
    "SEVERE_FROM",
    "SEVERITY_WEIGHT",
    "TileRecord",
    "priority_of",
    "status_payload",
]
