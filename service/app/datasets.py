"""A4 data joins: local GeoJSON/CSV loaders and the vulnerability join.

WHY everything is local and lru_cached: at zero connectivity the joins still have
to answer, and the only component allowed to touch the network is the librarian
(by dataset NAME, never a URL). Call reset_cache() after a librarian atomic swap
so the new file is visible without a restart.

WHY DROP_COLUMNS exists at load time and not at render time: the King County
assessor join carries owner names. A filter at the API boundary is one forgotten
endpoint away from leaking them, so the columns are deleted while the file is
being parsed and never enter a process object at all. Life-safety ranking does
not need to know who owns a building, and property value never enters the
formula.

Geometry uses shapely when importable and a pure-python fallback otherwise,
because the join is on the critical path of every tile and must not depend on a
compiled wheel being present.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import config, contracts

# PUBLIC API
# DROP_COLUMNS: tuple[str, ...]                 # owner-name fields killed at load time
# footprints() -> list[Feature]
# roads() -> list[Feature]
# facilities() -> list[Feature]                 # .ftype in contracts.FACILITY_TYPES
# svi() -> list[Feature]                        # .value carries the block-group index
# parcels() -> list[Feature]
# join(buildings, bounds) -> None               # mutates .label .facility_near .svi in place
# vulnerable_density(building) -> float         # 0..1
# vulnerable_density_from(svi_value, facility) -> float   # THE definition, dict or dataclass
# footprints_in(bounds) -> list[Feature]
# facilities_geojson() -> dict                  # GeoJSON FeatureCollection
# roads_geojson() -> dict                       # GeoJSON FeatureCollection
# footprints_geojson(bounds=None) -> dict
# nearest_road(centroid, max_m=400.0) -> tuple[Optional[str], float]
# road_names() -> list[str]
# geocode(q, pad_m=250.0) -> Optional[list[float]]        # [w, s, e, n] or None
# svi_at(centroid) -> float                     # DEFAULT_SVI when no coverage
# facility_near(centroid, radius_m=300.0) -> Optional[contracts.FacilityNear]
# label_for(centroid, max_m=60.0) -> str
# available() -> dict[str, int]                  # per-set feature counts, for the HUD
# dropped_columns_seen() -> list[str]            # measured, for the privacy write-up
# reset_cache() -> None                          # after a librarian atomic swap
# Geometry helpers shared with grading: bbox_of, centroid_of, point_in,
# bbox_overlaps, geom_area_m2, meters_between, classify_facility

log = logging.getLogger("firstlight.datasets")

# ------------------------------------------------------------------- privacy
# A4 names the exact fields to drop. These are the King County assessor and
# address-point owner columns, plus the mailing-address variants that identify a
# person just as well as a name does.
DROP_COLUMNS: tuple[str, ...] = (
    "owner",
    "owner_name",
    "ownername",
    "owner_1",
    "owner_2",
    "ownr_name",
    "OWNER",
    "OWNER_NAME",
    "OWNERNAME",
    "TAXPAYER",
    "taxpayer",
    "taxpayer_name",
    "TaxpayerName",
    "contact_name",
    "ContactName",
    "mailing_address",
    "MAILING_ADDRESS",
    "mail_address",
    "MAILADDR",
    "mailaddr",
    "mail_addr1",
    "mail_addr2",
    "mail_city",
    "mail_state",
    "mail_zip",
    "MAILINGCITY",
    "MAILINGSTATE",
    "MAILINGZIP",
    "resident_name",
    "occupant",
    "occupant_name",
    "phone",
    "PHONE",
    "email",
    "EMAIL",
)
_DROP_LOWER = frozenset(c.lower() for c in DROP_COLUMNS)
_SEEN_DROPPED: set[str] = set()

DEFAULT_SVI = 0.5
FACILITY_RADIUS_M = 300.0

# One degree of latitude, and of longitude at the equator, in metres. Good to a
# fraction of a percent over a county-sized AOI and it needs no projection.
_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LNG = 111320.0

try:  # pragma: no cover - presence depends on the box, both paths are exercised
    from shapely.geometry import shape as _shapely_shape

    _HAVE_SHAPELY = True
except Exception:  # noqa: BLE001
    _shapely_shape = None
    _HAVE_SHAPELY = False


# --------------------------------------------------------------------- feature
@dataclass
class Feature:
    """One normalized local-dataset row.

    `name` and `ftype` are normalized at load time so callers never have to know
    which of a dozen upstream spellings a given county file used.
    """

    fid: str
    name: str = ""
    ftype: str = ""
    address: str = ""
    value: Optional[float] = None
    centroid: Optional[list[float]] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    geom: Optional[dict] = None
    props: dict = field(default_factory=dict)

    def as_geojson(self) -> dict:
        props = dict(self.props)
        props.update({"id": self.fid, "name": self.name})
        if self.ftype:
            props["type"] = self.ftype
        if self.address:
            props["address"] = self.address
        if self.value is not None:
            props["value"] = self.value
        geom = self.geom
        if geom is None and self.centroid is not None:
            geom = {"type": "Point", "coordinates": list(self.centroid)}
        return {"type": "Feature", "geometry": geom, "properties": props}


# ----------------------------------------------------------------- key aliases
_NAME_KEYS = (
    "name",
    "NAME",
    "facility_name",
    "FACILITY_NAME",
    "provider_name",
    "PROVIDER_NAME",
    "legal_business_name",
    "LEGAL_BUSINESS_NAME",
    "street_name",
    "ST_NAME",
    "STREETNAME",
    "full_name",
    "FULLNAME",
    "street",
    "STREET",
    "road",
    "ROAD",
    "label",
)
_ADDRESS_KEYS = (
    "address",
    "ADDRESS",
    "addr",
    "ADDR",
    "full_address",
    "FULL_ADDRESS",
    "addr_full",
    "ADDR_FULL",
    "site_address",
    "SITEADDRESS",
    "situs_address",
    "SITUS_ADDRESS",
    "situsaddress",
    "street_address",
    "STREET_ADDRESS",
    "addr1",
    "ADDR1",
    "address_1",
)
_TYPE_KEYS = (
    # scripts/fetch_aoi.py stamps this when it enumerates a county's facility
    # layers, because the layer id is the only place the kind is recorded: a
    # Pinellas fire-station row carries NAME "09" and no type column at all.
    "firstlight_kind",
    "type",
    "TYPE",
    "facility_type",
    "FACILITY_TYPE",
    "provider_type",
    "PROVIDER_TYPE",
    "category",
    "CATEGORY",
    "amenity",
    "healthcare",
    "subtype",
    "FACILITYTYPE",
    "CRITICALFACILITY",
)
_SVI_KEYS = (
    "svi",
    "SVI",
    "rpl_themes",
    "RPL_THEMES",
    "svi_index",
    "SVI_INDEX",
    "overall_svi",
    "value",
)
_PIN_KEYS = ("PIN", "pin", "parcel_pin", "PARCEL_PIN", "parcel_id", "PARCELID", "apn")
_ID_KEYS = ("id", "ID", "OBJECTID", "objectid", "fid", "FID", "gid", "ccn", "CCN")
_LNG_KEYS = ("lng", "lon", "long", "longitude", "x", "LONGITUDE", "Longitude", "LON", "X")
_LAT_KEYS = ("lat", "latitude", "y", "LATITUDE", "Latitude", "LAT", "Y")

# CMS Care Compare and OSM both spell these differently. Order matters: a
# dialysis unit inside a hospital campus is a dialysis need first.
_FACILITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dialysis", ("dialysis", "esrd", "renal")),
    (
        "nursing_home",
        (
            "nursing",
            "skilled nursing",
            "snf",
            "long term care",
            "long-term care",
            "assisted living",
            "residential care",
            "adult family home",
            "memory care",
            "hospice",
            "rehabilitation",
            "convalescent",
        ),
    ),
    ("hospital", ("hospital", "medical center", "medical centre", "critical access", "clinic")),
)


def _first(props: dict, keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        v = props.get(k)
        if v not in (None, "", "null", "NULL"):
            return v
    return None


def _as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _scrub(props: dict) -> dict:
    """Delete owner-name columns while parsing, before anything can reference them."""
    out = {}
    for k, v in props.items():
        if str(k).strip().lower() in _DROP_LOWER:
            _SEEN_DROPPED.add(str(k))
            continue
        out[k] = v
    return out


def dropped_columns_seen() -> list[str]:
    """Owner columns actually encountered in the loaded files, for the write-up."""
    return sorted(_SEEN_DROPPED)


def classify_facility(raw: Any, name: str = "") -> str:
    """Map an upstream type string onto contracts.FACILITY_TYPES.

    Returns "" when nothing matches, and such rows are dropped: the medical-cross
    marker means one of three specific care needs, so a coffee shop tagged
    `amenity` must not inherit it.
    """
    blob = f"{raw or ''} {name or ''}".lower()
    for ftype, needles in _FACILITY_PATTERNS:
        if any(n in blob for n in needles):
            return ftype
    return ""


# ------------------------------------------------------------------- geometry
def meters_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Distance in metres between two [lng, lat] pairs, equirectangular."""
    mean_lat = math.radians((float(a[1]) + float(b[1])) * 0.5)
    dx = (float(b[0]) - float(a[0])) * _M_PER_DEG_LNG * math.cos(mean_lat)
    dy = (float(b[1]) - float(a[1])) * _M_PER_DEG_LAT
    return math.hypot(dx, dy)


def _coords(geom: Any) -> Iterable[Sequence[float]]:
    """Every position in a GeoJSON geometry, at any nesting depth."""
    if isinstance(geom, dict):
        if geom.get("type") == "GeometryCollection":
            for g in geom.get("geometries") or ():
                yield from _coords(g)
            return
        yield from _coords(geom.get("coordinates"))
        return
    if isinstance(geom, (list, tuple)):
        if geom and isinstance(geom[0], (int, float)) and len(geom) >= 2:
            yield geom  # type: ignore[misc]
            return
        for part in geom:
            yield from _coords(part)


def bbox_of(geom: Any) -> Optional[tuple[float, float, float, float]]:
    """Bounding box [w, s, e, n] of any GeoJSON geometry, None when it has no positions."""
    xs: list[float] = []
    ys: list[float] = []
    for lng, lat, *_ in _coords(geom):
        xs.append(float(lng))
        ys.append(float(lat))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def centroid_of(geom: Any) -> Optional[list[float]]:
    """Mean of the positions as [lng, lat], enough for a nearest-neighbour key.

    Not the area centroid: for a building footprint the difference is under a
    metre and this costs no shapely import on the ingest path.
    """
    n = 0
    sx = sy = 0.0
    for lng, lat, *_ in _coords(geom):
        sx += float(lng)
        sy += float(lat)
        n += 1
    if not n:
        return None
    return [round(sx / n, 7), round(sy / n, 7)]


def _polygons(geom: dict) -> list[list[list[Sequence[float]]]]:
    """Rings grouped per polygon: element 0 is the shell, the rest are holes."""
    kind = geom.get("type")
    coords = geom.get("coordinates") or []
    if kind == "Polygon":
        return [[list(r) for r in coords]]
    if kind == "MultiPolygon":
        return [[list(r) for r in poly] for poly in coords]
    return []


def _rings(geom: dict) -> list[list[Sequence[float]]]:
    """Every ring, flattened. Even-odd over the flat list handles shells and holes."""
    return [ring for poly in _polygons(geom) for ring in poly]


def _ring_contains(ring: Sequence[Sequence[float]], x: float, y: float) -> bool:
    """Even-odd ray cast. Pure python so the join never needs a compiled wheel."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        if (yi > y) != (yj > y):
            t = (y - yi) / (yj - yi) if yj != yi else 0.0
            if x < xi + t * (xj - xi):
                inside = not inside
        j = i
    return inside


def point_in(geom: Optional[dict], point: Sequence[float]) -> bool:
    """Point-in-polygon with shapely when available, pure python otherwise."""
    if not geom:
        return False
    x, y = float(point[0]), float(point[1])
    box = bbox_of(geom)
    if box and not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
        return False  # bbox reject first, it kills almost every candidate
    if _HAVE_SHAPELY:
        try:
            return bool(_shapely_shape(geom).covers(_shapely_shape({
                "type": "Point",
                "coordinates": [x, y],
            })))
        except Exception:  # noqa: BLE001 - a malformed ring falls through to ray casting
            pass
    hits = sum(1 for ring in _rings(geom) if _ring_contains(ring, x, y))
    return hits % 2 == 1


def bbox_overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _ring_area_m2(ring: Sequence[Sequence[float]], kx: float) -> float:
    """Shoelace area of one ring on a local metre grid."""
    n = len(ring)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x1, y1 = float(ring[i][0]) * kx, float(ring[i][1]) * _M_PER_DEG_LAT
        x2 = float(ring[(i + 1) % n][0]) * kx
        y2 = float(ring[(i + 1) % n][1]) * _M_PER_DEG_LAT
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def geom_area_m2(geom: Optional[dict]) -> float:
    """Footprint area in square metres, shell minus holes, zero for non-polygons.

    Equirectangular rather than projected: over one county the scale error is a
    fraction of a percent, and area only feeds the join context and a size hint,
    never a life-safety threshold.
    """
    if not geom:
        return 0.0
    polys = _polygons(geom)
    if not polys:
        return 0.0
    anchor = centroid_of(geom)
    if anchor is None:
        return 0.0
    kx = _M_PER_DEG_LNG * math.cos(math.radians(anchor[1]))
    total = 0.0
    for rings in polys:
        if not rings:
            continue
        total += _ring_area_m2(rings[0], kx)
        for hole in rings[1:]:
            total -= _ring_area_m2(hole, kx)
    return round(max(total, 0.0), 2)


# ------------------------------------------------------------------ grid index
class _Grid:
    """Coarse lat/lng bucket index.

    WHY: nearest-neighbour over 27,250 footprints for every building of every
    tile is 5 million distance calls a tile. Bucketing at roughly 200 m turns it
    into a handful.
    """

    CELL = 0.002  # about 220 m of latitude

    def __init__(self, feats: Sequence[Feature]) -> None:
        self.feats = feats
        self.cells: dict[tuple[int, int], list[int]] = {}
        for i, f in enumerate(feats):
            c = f.centroid
            if c is None:
                continue
            self.cells.setdefault(self._key(c[0], c[1]), []).append(i)

    def _key(self, lng: float, lat: float) -> tuple[int, int]:
        return (int(math.floor(lng / self.CELL)), int(math.floor(lat / self.CELL)))

    def near(self, point: Sequence[float], radius_m: float) -> Iterable[Feature]:
        if not self.cells:
            return ()
        span = max(1, int(math.ceil(radius_m / (self.CELL * _M_PER_DEG_LAT))))
        cx, cy = self._key(float(point[0]), float(point[1]))
        out: list[Feature] = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for i in self.cells.get((cx + dx, cy + dy), ()):
                    out.append(self.feats[i])
        return out

    def nearest(
        self, point: Sequence[float], max_m: float
    ) -> tuple[Optional[Feature], float]:
        """Expanding-ring search so a sparse area still finds its neighbour."""
        radius = min(max_m, self.CELL * _M_PER_DEG_LAT)
        while True:
            best: Optional[Feature] = None
            best_d = float("inf")
            for f in self.near(point, radius):
                d = meters_between(point, f.centroid)  # type: ignore[arg-type]
                if d < best_d:
                    best, best_d = f, d
            if best is not None and best_d <= radius:
                return best, best_d
            if radius >= max_m:
                return (best, best_d) if best is not None and best_d <= max_m else (None, best_d)
            radius = min(max_m, radius * 3.0)


# ---------------------------------------------------------------------- loading
def _candidates(*stems: str) -> list[Path]:
    out: list[Path] = []
    for stem in stems:
        for ext in (".geojson", ".json", ".csv"):
            p = config.DATASET_DIR / f"{stem}{ext}"
            if p.exists():
                out.append(p)
    return out


def _read_geojson(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a truncated download must not kill ingest
        log.warning("dataset %s unreadable: %s", path.name, exc)
        return []
    if isinstance(data, dict) and isinstance(data.get("features"), list):
        return [f for f in data["features"] if isinstance(f, dict)]
    if isinstance(data, list):
        return [f for f in data if isinstance(f, dict)]
    if isinstance(data, dict) and data.get("type") == "Feature":
        return [data]
    return []


def _read_csv(path: Path) -> list[dict]:
    """CSV rows as pseudo-features, with lat/lng promoted to a Point geometry."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001
        log.warning("dataset %s unreadable: %s", path.name, exc)
        return []
    feats: list[dict] = []
    for row in rows:
        props = {k: v for k, v in row.items() if k}
        lng = _as_float(_first(props, _LNG_KEYS))
        lat = _as_float(_first(props, _LAT_KEYS))
        geom = {"type": "Point", "coordinates": [lng, lat]} if None not in (lng, lat) else None
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    return feats


def _raw(*stems: str) -> list[dict]:
    for path in _candidates(*stems):
        feats = _read_geojson(path) if path.suffix != ".csv" else _read_csv(path)
        if feats:
            log.info("loaded %d features from %s", len(feats), path.name)
            return feats
    return []


def _to_feature(raw: dict, prefix: str, index: int) -> Feature:
    props = _scrub(dict(raw.get("properties") or {}))
    geom = raw.get("geometry") if isinstance(raw.get("geometry"), dict) else None
    fid = _first(props, _ID_KEYS)
    name = _first(props, _NAME_KEYS)
    addr = _first(props, _ADDRESS_KEYS)
    return Feature(
        fid=str(fid) if fid is not None else f"{prefix}_{index}",
        name=str(name) if name is not None else "",
        address=str(addr) if addr is not None else "",
        centroid=centroid_of(geom) if geom else None,
        bbox=bbox_of(geom) if geom else None,
        geom=geom,
        props=props,
    )


@lru_cache(maxsize=1)
def footprints() -> list[Feature]:
    """Building outlines. Seattle 2023 locally, MS GlobalMLBuildingFootprints rurally."""
    feats = [
        _to_feature(r, "fp", i)
        for i, r in enumerate(_raw("footprints", "building_footprints", "ms_building_footprints"))
    ]
    # A footprint with no geometry cannot be joined to anything.
    return [f for f in feats if f.centroid is not None]


@lru_cache(maxsize=1)
def roads() -> list[Feature]:
    return [
        f
        for f in (_to_feature(r, "road", i) for i, r in enumerate(_raw("roads", "osm_roads")))
        if f.centroid is not None
    ]


@lru_cache(maxsize=1)
def facilities() -> list[Feature]:
    """Care facilities, restricted to the three types that get a medical cross."""
    out: list[Feature] = []
    for i, raw in enumerate(_raw("facilities", "cms_facilities", "care_facilities")):
        f = _to_feature(raw, "fac", i)
        if f.centroid is None:
            continue
        f.ftype = classify_facility(_first(f.props, _TYPE_KEYS), f.name)
        if f.ftype not in contracts.FACILITY_TYPES:
            continue
        if not f.name:
            f.name = f"{f.ftype.replace('_', ' ')} {f.fid}"
        out.append(f)
    return out


@lru_cache(maxsize=1)
def svi() -> list[Feature]:
    """CDC SVI block groups. `.value` is the 0..1 overall percentile."""
    out: list[Feature] = []
    for i, raw in enumerate(_raw("svi", "cdc_svi", "svi_block_groups")):
        f = _to_feature(raw, "svi", i)
        v = _as_float(_first(f.props, _SVI_KEYS))
        # CDC publishes -999 for suppressed block groups. Treated as no coverage.
        if v is None or v < 0.0:
            continue
        f.value = min(1.0, v / 100.0 if v > 1.0 else v)
        out.append(f)
    return out


@lru_cache(maxsize=1)
def parcels() -> list[Feature]:
    """King County parcels. Owner columns are already gone: see DROP_COLUMNS."""
    return [
        _to_feature(r, "parcel", i)
        for i, r in enumerate(_raw("parcels", "kc_parcels", "address_points"))
    ]


@lru_cache(maxsize=1)
def _footprint_grid() -> _Grid:
    return _Grid(footprints())


@lru_cache(maxsize=1)
def _facility_grid() -> _Grid:
    return _Grid(facilities())


@lru_cache(maxsize=1)
def _road_grid() -> _Grid:
    return _Grid(roads())


@lru_cache(maxsize=1)
def _parcel_address_by_pin() -> dict[str, str]:
    """PIN to street address. Identity join, per A4: a key, not spatial matching."""
    out: dict[str, str] = {}
    for p in parcels():
        pin = _first(p.props, _PIN_KEYS)
        if pin is None or not p.address:
            continue
        out.setdefault(str(pin), p.address)
    return out


@lru_cache(maxsize=1)
def _parcel_grid() -> _Grid:
    return _Grid([p for p in parcels() if p.centroid is not None and p.address])


def reset_cache() -> None:
    """Drop every loader cache. Call after the librarian swaps a dataset in."""
    for fn in (
        footprints,
        roads,
        facilities,
        svi,
        parcels,
        _footprint_grid,
        _facility_grid,
        _road_grid,
        _parcel_address_by_pin,
        _parcel_grid,
    ):
        fn.cache_clear()


def available() -> dict[str, int]:
    """Feature counts per set, so the HUD can say what the joins actually have."""
    return {
        "footprints": len(footprints()),
        "roads": len(roads()),
        "facilities": len(facilities()),
        "svi": len(svi()),
        "parcels": len(parcels()),
    }


# ------------------------------------------------------------------- lookups
def footprints_in(bounds: Optional[Sequence[float]]) -> list[Feature]:
    """Footprints whose bbox intersects [w, s, e, n]. All of them when bounds is None."""
    if bounds is None:
        return footprints()
    box = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    return [f for f in footprints() if f.bbox and bbox_overlaps(f.bbox, box)]


def nearest_road(centroid: Sequence[float], max_m: float = 400.0) -> tuple[Optional[str], float]:
    """Nearest named road centre and its distance, for the fallback label."""
    if not roads():
        return None, float("inf")
    f, d = _road_grid().nearest(centroid, max_m)
    if f is None or not f.name:
        return None, d
    return f.name, d


def road_names() -> list[str]:
    return sorted({f.name for f in roads() if f.name})


def svi_at(centroid: Optional[Sequence[float]]) -> float:
    """SVI of the containing block group, DEFAULT_SVI where there is no coverage.

    WHY a 0.5 default and not 0: vulnerable_density multiplies into priority, so a
    zero would silently delete a building from the ranking for a data gap.
    """
    if centroid is None:
        return DEFAULT_SVI
    for f in svi():
        if f.bbox and bbox_overlaps(f.bbox, (centroid[0], centroid[1], centroid[0], centroid[1])):
            if point_in(f.geom, centroid) and f.value is not None:
                return float(f.value)
    return DEFAULT_SVI


def facility_near(
    centroid: Optional[Sequence[float]], radius_m: float = FACILITY_RADIUS_M
) -> Optional[contracts.FacilityNear]:
    """Nearest care facility within radius_m, or None. Type is one of three."""
    if centroid is None or not facilities():
        return None
    f, d = _facility_grid().nearest(centroid, radius_m)
    if f is None or d > radius_m:
        return None
    return contracts.FacilityNear(name=f.name, type=f.ftype, dist_m=int(round(d)))


def _footprint_address(f: Feature) -> str:
    if f.address:
        return f.address
    pin = _first(f.props, _PIN_KEYS)
    if pin is not None:
        addr = _parcel_address_by_pin().get(str(pin))
        if addr:
            return addr
    return ""


def label_for(centroid: Optional[Sequence[float]], *, max_m: float = 60.0) -> str:
    """Human-readable address for a graded building.

    Order: the address on the nearest footprint, then the address of the nearest
    address-carrying parcel, then a road-relative description. Never a raw ID: an
    operator dispatches crews to streets, not to footprint hashes.
    """
    if centroid is None:
        return "unlocated structure"
    fp, d = _footprint_grid().nearest(centroid, max_m) if footprints() else (None, float("inf"))
    if fp is not None and d <= max_m:
        addr = _footprint_address(fp)
        if addr:
            return addr
    parcel, pd = _parcel_grid().nearest(centroid, max_m) if parcels() else (None, float("inf"))
    if parcel is not None and pd <= max_m and parcel.address:
        return parcel.address
    road, _rd = nearest_road(centroid)
    if road:
        return f"unnamed structure near {road}"
    return f"unnamed structure at {centroid[1]:.5f}, {centroid[0]:.5f}"


_COORD_RE = re.compile(r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)")


# Below this a query is too short to identify a street: "sw" would match half of
# West Seattle and the location resolver would narrow to nothing useful.
MIN_GEOCODE_TERM = 3


def _name_matches(name: str, query_low: str) -> bool:
    low = name.lower()
    if len(low) < MIN_GEOCODE_TERM:
        return False
    return low in query_low or (len(query_low) >= MIN_GEOCODE_TERM and query_low in low)


def geocode(q: str, *, pad_m: float = 250.0) -> Optional[list[float]]:
    """Resolve a location phrase to [w, s, e, n] against the local tables only.

    Accepts "47.558, -122.377", a road name, or a facility name. Returns None when
    nothing local matches, so the caller can skip the location resolver rather
    than silently searching the whole corpus.
    """
    text = (q or "").strip()
    if not text:
        return None
    m = _COORD_RE.search(text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        # Whichever value cannot be a latitude is the longitude.
        lat, lng = (a, b) if abs(a) <= 90.0 else (b, a)
        return _pad([lng, lat], pad_m)
    low = text.lower()
    for f in facilities():
        if f.centroid and _name_matches(f.name, low):
            return _pad(f.centroid, pad_m)
    boxes = [f.bbox for f in roads() if f.bbox and _name_matches(f.name, low)]
    if boxes:
        return _grow(
            [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ],
            pad_m,
        )
    return None


def _grow(box: Sequence[float], pad_m: float) -> list[float]:
    """Widen a bbox by pad_m on every side.

    WHY this is not optional for a road: a north-south street has zero longitude
    width, so an unpadded bbox filter matches no building at all and the location
    resolver silently returns nothing. Buildings sit beside a street, never on it.
    """
    lat = (float(box[1]) + float(box[3])) * 0.5
    dlat = pad_m / _M_PER_DEG_LAT
    dlng = pad_m / (_M_PER_DEG_LNG * max(math.cos(math.radians(lat)), 1e-6))
    return [
        round(float(box[0]) - dlng, 7),
        round(float(box[1]) - dlat, 7),
        round(float(box[2]) + dlng, 7),
        round(float(box[3]) + dlat, 7),
    ]


def _pad(centroid: Sequence[float], pad_m: float) -> list[float]:
    return _grow(
        [float(centroid[0]), float(centroid[1]), float(centroid[0]), float(centroid[1])], pad_m
    )


# ------------------------------------------------------------------ the join
# Facility bumps. A dialysis patient misses treatment in days, a nursing-home
# resident cannot self-evacuate at all, so those outrank a staffed hospital.
_FACILITY_BUMP = {"nursing_home": 0.35, "dialysis": 0.30, "hospital": 0.25}
VULN_FLOOR = 0.05


def vulnerable_density_from(svi_value: Optional[float], facility: Optional[Any]) -> float:
    """THE definition of resident vulnerability, 0..1. Everything else delegates here.

    `facility` is the persisted facility dict {name, type, dist_m}, a
    contracts.FacilityNear, or None. B's scorer reads rows back out of SQLite where
    it is a dict, and the join path holds the dataclass, so both are accepted
    rather than duplicating this arithmetic on either side.

    SVI is the base and a nearby care facility adds a bump scaled by how close it
    is, so a dialysis unit next door counts fully and one at the 300 m edge counts
    for nothing. Floored at VULN_FLOOR because this term multiplies into priority
    and a zero would erase a building from the ranking rather than rank it last.
    """
    score = DEFAULT_SVI if svi_value is None else max(0.0, min(1.0, float(svi_value)))
    if facility is not None:
        if isinstance(facility, dict):
            ftype = str(facility.get("type") or "")
            dist = facility.get("dist_m")
        else:
            ftype = str(getattr(facility, "type", "") or "")
            dist = getattr(facility, "dist_m", None)
        dist_m = FACILITY_RADIUS_M if dist is None else max(0.0, float(dist))
        closeness = max(0.0, 1.0 - dist_m / FACILITY_RADIUS_M)
        score += _FACILITY_BUMP.get(ftype, 0.2) * closeness
    return round(max(VULN_FLOOR, min(1.0, score)), 3)


def vulnerable_density(building: Any) -> float:
    """Resident vulnerability for one graded building. Thin read of the fields."""
    return vulnerable_density_from(
        getattr(building, "svi", None), getattr(building, "facility_near", None)
    )


def join(buildings: Iterable[Any], bounds: Optional[Sequence[float]] = None) -> None:
    """Attach address label, nearest care facility and SVI to each building.

    Mutates in place and returns None, per the frozen cross-slice interface. Every
    lookup degrades to a labelled default rather than raising, because a missing
    county file must cost a label, never a tile.
    """
    for b in buildings:
        centroid = getattr(b, "centroid", None)
        try:
            b.label = label_for(centroid)
        except Exception as exc:  # noqa: BLE001
            log.warning("label join failed for %s: %s", getattr(b, "footprint_id", "?"), exc)
            b.label = "unnamed structure"
        try:
            b.facility_near = facility_near(centroid)
        except Exception as exc:  # noqa: BLE001
            log.warning("facility join failed: %s", exc)
            b.facility_near = None
        try:
            b.svi = svi_at(centroid)
        except Exception as exc:  # noqa: BLE001
            log.warning("svi join failed: %s", exc)
            b.svi = DEFAULT_SVI


# ------------------------------------------------------------------ API shapes
def _collection(feats: Sequence[Feature]) -> dict:
    return {"type": "FeatureCollection", "features": [f.as_geojson() for f in feats]}


def facilities_geojson() -> dict:
    """Care facilities for the map. properties.type drives the medical-cross marker."""
    return _collection(facilities())


def roads_geojson() -> dict:
    """Roads for the map and for the local geocoder. properties.name is normalized."""
    return _collection(roads())


def footprints_geojson(bounds: Optional[Sequence[float]] = None) -> dict:
    return _collection(footprints_in(bounds))


__all__ = [
    "DEFAULT_SVI",
    "DROP_COLUMNS",
    "FACILITY_RADIUS_M",
    "Feature",
    "VULN_FLOOR",
    "available",
    "bbox_of",
    "bbox_overlaps",
    "centroid_of",
    "classify_facility",
    "dropped_columns_seen",
    "facilities",
    "facilities_geojson",
    "facility_near",
    "footprints",
    "footprints_geojson",
    "footprints_in",
    "geocode",
    "geom_area_m2",
    "join",
    "label_for",
    "meters_between",
    "nearest_road",
    "parcels",
    "point_in",
    "reset_cache",
    "road_names",
    "roads",
    "roads_geojson",
    "svi",
    "svi_at",
    "vulnerable_density",
    "vulnerable_density_from",
]
