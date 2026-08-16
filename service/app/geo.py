"""A7: the geo fallback chain. GeoTIFF transform, then EXIF GPS, then sidecar.

Why a chain at all: a card dump from a county drone team is never uniform. Some
frames are orthorectified GeoTIFFs, most are JPEGs with EXIF GPS, and a few are
screenshots or exports that lost everything. An image with no location is still
evidence, so the last link is `needs_geo` for operator drag-to-place, never a
silent drop. Every optional dependency is imported defensively: a box without
rasterio must degrade to the EXIF path, not fail to import.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Rasterio carries GDAL. If it is missing we lose GeoTIFF transforms only.
try:  # pragma: no cover - presence depends on the box, both paths are exercised
    import rasterio
    from rasterio.warp import transform_bounds
except Exception:  # noqa: BLE001 - any import failure degrades the same way
    rasterio = None
    transform_bounds = None

try:  # pragma: no cover - piexif is the pinned reader, Pillow is the fallback
    import piexif
except Exception:  # noqa: BLE001
    piexif = None

GEO_SUFFIXES = {".tif", ".tiff", ".gtiff", ".jp2", ".vrt"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}

# A single EXIF point has no footprint, so we synthesize one. 150 m across is
# what a survey frame covers at a typical 120 m AGL pass with a 4:3 sensor.
EXIF_FOOTPRINT_M = 150.0

# A tile is not a continent. Anything wider than this came from a broken
# transform, not from a drone.
MAX_SPAN_DEG = 10.0
# Null island: a bbox centred here means EXIF decoded to zeros or the transform
# was never applied. Real imagery in the Gulf of Guinea is not our failure mode.
NULL_ISLAND_DEG = 0.01
_ORIGIN_EPS = 1e-9

SOURCE_GEOTIFF = "geotiff"
SOURCE_EXIF = "exif"
SOURCE_SIDECAR = "sidecar"
SOURCE_NONE = "none"


@dataclass
class GeoResult:
    """Where a tile's bounds came from, so the console can say so out loud.

    `detail` explains a miss (no CRS, no GPS tags, implausible box) because
    "needs_geo" with no reason is the kind of UI an operator stops trusting.
    """

    bounds: Optional[list[float]]
    source: str
    needs_geo: bool
    detail: Optional[str] = None

    @property
    def centroid(self) -> Optional[list[float]]:
        if not self.bounds:
            return None
        w, s, e, n = self.bounds
        return [round((w + e) / 2.0, 7), round((s + n) / 2.0, 7)]

    def wire(self) -> dict:
        return {
            "bounds": self.bounds,
            "geo_source": self.source,
            "needs_geo": bool(self.needs_geo),
            "detail": self.detail,
        }


def bbox_around(lng: float, lat: float, meters: float = EXIF_FOOTPRINT_M) -> list[float]:
    """A small square bbox in degrees around a point, [w, s, e, n].

    Local flat-earth conversion: at tile scale the error is centimetres, and a
    projection library for a 150 m box would be ceremony.
    """
    half = max(1.0, float(meters)) / 2.0
    dlat = half / 111_320.0
    dlng = half / (111_320.0 * max(0.05, math.cos(math.radians(lat))))
    return [
        round(lng - dlng, 7),
        round(lat - dlat, 7),
        round(lng + dlng, 7),
        round(lat + dlat, 7),
    ]


def _touches_origin(w: float, s: float, e: float, n: float) -> bool:
    """An untransformed pixel grid pins one corner at exactly 0, 0."""
    return any(
        abs(x) < _ORIGIN_EPS and abs(y) < _ORIGIN_EPS
        for x, y in ((w, s), (w, n), (e, s), (e, n))
    )


def _plausible(bounds: Any) -> bool:
    """Reject boxes that are syntactically fine and point nowhere.

    Four real failure modes, all seen on real data: a zeroed or identity GeoTIFF
    transform, EXIF that decoded to null island, a bbox assembled in the wrong
    order (inverted), and degrees that are actually pixels or radians.
    """
    if bounds is None:
        return False
    try:
        if len(bounds) != 4:
            return False
        w, s, e, n = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return False
    if any(math.isnan(v) or math.isinf(v) for v in (w, s, e, n)):
        return False
    if not (-180.0 <= w < e <= 180.0):
        return False
    if not (-90.0 <= s < n <= 90.0):
        return False
    if (e - w) > MAX_SPAN_DEG or (n - s) > MAX_SPAN_DEG:
        return False
    if _touches_origin(w, s, e, n):
        return False
    if abs((w + e) / 2.0) < NULL_ISLAND_DEG and abs((s + n) / 2.0) < NULL_ISLAND_DEG:
        return False
    return True


def _clean(bounds: Any) -> Optional[list[float]]:
    if not _plausible(bounds):
        return None
    return [round(float(v), 7) for v in bounds]


# ------------------------------------------------------------------- geotiff
def _from_geotiff(path: Path) -> tuple[Optional[list[float]], Optional[str]]:
    if path.suffix.lower() not in GEO_SUFFIXES:
        return None, None
    if rasterio is None:
        return None, "rasterio unavailable"
    try:
        with rasterio.open(str(path)) as ds:
            if ds.crs is None:
                return None, "geotiff carries no CRS"
            b = ds.bounds
            epsg = None
            try:
                epsg = ds.crs.to_epsg()
            except Exception:  # noqa: BLE001 - exotic CRS, let warp decide
                epsg = None
            if epsg == 4326:
                raw = [b.left, b.bottom, b.right, b.top]
            elif transform_bounds is not None:
                raw = list(
                    transform_bounds(
                        ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
                    )
                )
            else:  # pragma: no cover - transform_bounds ships with rasterio
                return None, "cannot reproject without rasterio.warp"
    except Exception as exc:  # noqa: BLE001 - a corrupt tif must not kill ingest
        return None, f"geotiff read failed: {type(exc).__name__}"
    cleaned = _clean(raw)
    if cleaned is None:
        return None, "geotiff transform implausible"
    return cleaned, None


# ---------------------------------------------------------------------- exif
def _rational(value: Any) -> float:
    """EXIF rationals arrive as (num, den), IFDRational or float depending on
    which reader got there first."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("not a rational")
        num, den = value
        return float(num) / float(den) if float(den) else 0.0
    return float(value)


def _dms_to_deg(dms: Any, ref: Any) -> float:
    d, m, s = (_rational(v) for v in tuple(dms)[:3])
    deg = d + m / 60.0 + s / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", "ignore")
    if str(ref).strip().upper() in ("S", "W"):
        deg = -deg
    return deg


def _gps_tags(path: Path) -> Optional[dict]:
    """Return {lat_ref, lat, lng_ref, lng} raw EXIF values, or None.

    piexif is the pinned reader; Pillow is the fallback so a box without piexif
    still resolves locations instead of sending every JPEG to needs_geo.
    """
    if piexif is not None:
        try:
            gps = piexif.load(str(path)).get("GPS") or {}
            if gps.get(2) and gps.get(4):
                return {
                    "lat_ref": gps.get(1, b"N"),
                    "lat": gps[2],
                    "lng_ref": gps.get(3, b"E"),
                    "lng": gps[4],
                }
        except Exception:  # noqa: BLE001 - fall through to Pillow
            pass
    try:
        from PIL import Image

        with Image.open(str(path)) as im:
            ifd = dict(im.getexif().get_ifd(0x8825) or {})
    except Exception:  # noqa: BLE001
        return None
    if ifd.get(2) and ifd.get(4):
        return {
            "lat_ref": ifd.get(1, "N"),
            "lat": ifd[2],
            "lng_ref": ifd.get(3, "E"),
            "lng": ifd[4],
        }
    return None


def _from_exif(path: Path) -> tuple[Optional[list[float]], Optional[str]]:
    if path.suffix.lower() not in JPEG_SUFFIXES:
        return None, None
    try:
        tags = _gps_tags(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"exif read failed: {type(exc).__name__}"
    if not tags:
        return None, "no EXIF GPS tags"
    try:
        lat = _dms_to_deg(tags["lat"], tags["lat_ref"])
        lng = _dms_to_deg(tags["lng"], tags["lng_ref"])
    except Exception as exc:  # noqa: BLE001 - hostile EXIF is in the threat model
        return None, f"exif GPS unreadable: {type(exc).__name__}"
    cleaned = _clean(bbox_around(lng, lat))
    if cleaned is None:
        return None, "exif GPS implausible"
    return cleaned, None


# ------------------------------------------------------------------- sidecar
def sidecar_paths(path: Path) -> list[Path]:
    """Both spellings, because operators write both: `frame.bounds.json` beside
    `frame.jpg`, and `frame.jpg.bounds.json` from scripted exports."""
    return [
        path.with_suffix(".bounds.json"),
        path.parent / (path.name + ".bounds.json"),
    ]


def _from_sidecar(path: Path) -> tuple[Optional[list[float]], Optional[str]]:
    for cand in sidecar_paths(path):
        try:
            if not cand.is_file():
                continue
            payload = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return None, f"sidecar unreadable: {type(exc).__name__}"
        if not isinstance(payload, dict):
            return None, "sidecar is not an object"
        if payload.get("bounds") is not None:
            cleaned = _clean(payload.get("bounds"))
            return (cleaned, None) if cleaned else (None, "sidecar bounds implausible")
        lng = payload.get("lng", payload.get("lon", payload.get("longitude")))
        lat = payload.get("lat", payload.get("latitude"))
        if lng is None or lat is None:
            return None, "sidecar has neither bounds nor lng/lat"
        try:
            meters = float(payload.get("footprint_m", EXIF_FOOTPRINT_M))
            cleaned = _clean(bbox_around(float(lng), float(lat), meters))
        except (TypeError, ValueError):
            return None, "sidecar lng/lat not numeric"
        return (cleaned, None) if cleaned else (None, "sidecar point implausible")
    return None, None


def write_sidecar(path: Path, bounds: list[float], by: str = "operator") -> Path:
    """Persist an operator drag-to-place so a re-ingest of the same file resolves.

    Written beside the image, which is why ingest reads geo before it moves the
    file and moves the sidecar with it.
    """
    target = path.with_suffix(".bounds.json")
    target.write_text(
        json.dumps({"bounds": [round(float(v), 7) for v in bounds], "placed_by": by}),
        encoding="utf-8",
    )
    return target


# ----------------------------------------------------------------- the chain
def extract(path: Path | str) -> GeoResult:
    """Run the chain in order and never raise. Missing location is a flag, not
    an error: A7 says an image is never dropped for want of coordinates."""
    p = Path(path)
    notes: list[str] = []
    for source, fn in (
        (SOURCE_GEOTIFF, _from_geotiff),
        (SOURCE_EXIF, _from_exif),
        (SOURCE_SIDECAR, _from_sidecar),
    ):
        try:
            bounds, detail = fn(p)
        except Exception as exc:  # noqa: BLE001 - a broken link must not break the chain
            bounds, detail = None, f"{source} raised {type(exc).__name__}"
        if bounds is not None:
            return GeoResult(bounds=bounds, source=source, needs_geo=False, detail=detail)
        if detail:
            notes.append(detail)
    return GeoResult(
        bounds=None,
        source=SOURCE_NONE,
        needs_geo=True,
        detail="; ".join(notes) if notes else "no GeoTIFF transform, EXIF GPS or sidecar",
    )


__all__ = [
    "EXIF_FOOTPRINT_M",
    "GeoResult",
    "SOURCE_EXIF",
    "SOURCE_GEOTIFF",
    "SOURCE_NONE",
    "SOURCE_SIDECAR",
    "bbox_around",
    "extract",
    "sidecar_paths",
    "write_sidecar",
]
