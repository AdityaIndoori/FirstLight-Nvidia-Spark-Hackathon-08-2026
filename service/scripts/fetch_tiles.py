#!/usr/bin/env python3
"""Fetch raster basemap tiles for the AOI into web/tiles/{z}/{x}/{y}.png.

That path is exactly where web/js/map.js looks. The style is local-tiles-only:
there is no CDN fallback, and the legend prints "basemap tiles not cached" when
the directory is empty. So this script is the difference between a map and a
dark rectangle, and it has to run while a link still exists.

Run it once per AOI before the network goes away. It is resumable: tiles already
on disk are skipped, so an interrupted run continues where it stopped.

Providers, all measured from the Spark on 2026-08-16:

  carto_dark  (default) CARTO "dark_all", raster, OpenStreetMap data under
              ODbL with CARTO styling. Dark by design, which is the tactical
              basemap map.js wants. 5.9 KB to 18 KB per tile measured.
  carto_dark_nolabels  same, no place labels. Useful because the offline style
              has no glyph server, so labels baked into the raster are the only
              labels the operator gets, and some rooms prefer them off.
  osm         the openstreetmap.org standard layer. REACHABLE BUT REFUSED from
              this network: every request returned HTTP 200 with a 6987-byte
              body and the header `x-blocked: Access denied. See
              https://operations.osmfoundation.org/policies/tiles/`. The bytes
              are a real PNG saying so, which is why --verify checks more than
              the magic number. Left in as a named provider so the failure is
              documented rather than rediscovered.
  esri_imagery  satellite, --sat only, OFF by default. See docs/DATA.md for its
              terms. JPEG on the wire, re-encoded to PNG on disk so the .png
              path map.js requests is honest about its bytes.

Attribution is written to ATTRIBUTION.txt in the output directory, because
shipping a tile cache without attribution is a licence problem on stage.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# Run as `python scripts/fetch_tiles.py` from service/ as well as `python -m`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

# PUBLIC API
# PROVIDERS: dict[str, Provider]
# tile_xy(lng: float, lat: float, z: int) -> tuple[int, int]
# tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]
# tile_range(aoi, z) -> tuple[int, int, int, int]      (x0, y0, x1, y1) inclusive
# plan(aoi, zooms, provider) -> dict                   counts and byte estimate
# fetch_one(provider, z, x, y, dest, *, timeout, retries) -> tuple[str, int]
# download(aoi, zooms, provider, out, *, rate, retries, timeout, log) -> dict
# verify(out, zooms, *, sample) -> dict
# main(argv=None) -> int

USER_AGENT = (
    "FIRSTLIGHT-tilecache/1.0 "
    "(offline county EOC basemap cache; NVIDIA DGX Spark Hackathon 2026; one-time AOI fetch)"
)

# Above this many tiles the run refuses to start without --yes. z18 over a wide
# AOI is tens of thousands of tiles and nobody means to do that by accident.
DEFAULT_CAP = 20000

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


@dataclass(frozen=True)
class Provider:
    name: str
    url: str  # format string with {z} {x} {y}
    attribution: str
    avg_bytes: int  # measured on this box, used only for the pre-flight estimate
    satellite: bool = False
    max_zoom: int = 19
    # Sizes at or below this, byte-for-byte identical across distinct tiles, mean
    # the provider is serving one canned refusal image rather than map data.
    refusal_bytes: tuple[int, ...] = ()


PROVIDERS: dict[str, Provider] = {
    "carto_dark": Provider(
        name="carto_dark",
        url="https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attribution=(
            "Basemap tiles (c) CARTO, style 'dark_all'. "
            "Map data (c) OpenStreetMap contributors, ODbL 1.0 "
            "(https://www.openstreetmap.org/copyright). "
            "CARTO basemap terms: https://carto.com/legal/"
        ),
        avg_bytes=12000,
        max_zoom=20,
    ),
    "carto_dark_nolabels": Provider(
        name="carto_dark_nolabels",
        url="https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
        attribution=(
            "Basemap tiles (c) CARTO, style 'dark_nolabels'. "
            "Map data (c) OpenStreetMap contributors, ODbL 1.0 "
            "(https://www.openstreetmap.org/copyright). "
            "CARTO basemap terms: https://carto.com/legal/"
        ),
        avg_bytes=9000,
        max_zoom=20,
    ),
    "osm": Provider(
        name="osm",
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution=(
            "Map data and tiles (c) OpenStreetMap contributors, ODbL 1.0 "
            "(https://www.openstreetmap.org/copyright). "
            "Tile usage policy: https://operations.osmfoundation.org/policies/tiles/"
        ),
        avg_bytes=14000,
        max_zoom=19,
        refusal_bytes=(6987,),
    ),
    "esri_imagery": Provider(
        name="esri_imagery",
        url=(
            "https://services.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attribution=(
            "Satellite imagery: Esri World Imagery. "
            "Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community. "
            "Terms: https://www.esri.com/en-us/legal/terms/full-master-agreement "
            "and https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9"
        ),
        # 20 KB on the wire as JPEG, but we re-encode to PNG so the .png path
        # map.js requests holds honest bytes, and photographic imagery balloons
        # about 6x doing that. 128 KB is the measured on-disk average.
        avg_bytes=128000,
        satellite=True,
        max_zoom=19,
    ),
}

DEFAULT_TACTICAL = "carto_dark"
DEFAULT_SATELLITE = "esri_imagery"

# ------------------------------------------------------------ slippy-map math
def tile_xy(lng: float, lat: float, z: int) -> tuple[int, int]:
    """Standard Web Mercator tile index for one coordinate. Latitude is clamped
    to the Mercator limit so an AOI typed with a pole in it cannot produce a
    negative row."""
    n = 1 << z
    lat = max(-85.05112878, min(85.05112878, lat))
    x = int((lng + 180.0) / 360.0 * n)
    rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """Inverse of tile_xy, as [w, s, e, n]. Used by the verifier's report only."""
    n = 1 << z

    def lat_of(row: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * row / n))))

    return (x / n * 360.0 - 180.0, lat_of(y + 1), (x + 1) / n * 360.0 - 180.0, lat_of(y))


def tile_range(aoi: list[float], z: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) inclusive for AOI [w, s, e, n]. Tile rows run north to
    south, so the north edge gives the smaller y."""
    w, s, e, n = aoi
    x0, y0 = tile_xy(min(w, e), max(s, n), z)
    x1, y1 = tile_xy(max(w, e), min(s, n), z)
    return x0, y0, x1, y1


def parse_zooms(spec: str) -> list[int]:
    """Accepts "12-18", "12,14,16" or a mix. Rejects anything outside 0-22."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo_s, _, hi_s = part.partition("-")
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"zoom range runs backwards: {part}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("no zoom levels given")
    if min(out) < 0 or max(out) > 22:
        raise ValueError("zoom levels must sit in 0-22")
    return sorted(out)


# config.AOI_PRESETS arrived with README section 9. A box still carrying the
# pre-section-9 config would otherwise crash on import, and the tile cache is
# exactly the thing you need working on a box that is behind.
PRESETS: dict[str, list[float]] = dict(getattr(config, "AOI_PRESETS", None) or {
    "pinellas": [-82.78, 27.75, -82.70, 27.82],
    "bay": [-85.72, 30.13, -85.62, 30.22],
    "sarasota": [-82.56, 27.30, -82.48, 27.38],
})


def parse_aoi(spec: str) -> list[float]:
    """"w,s,e,n" or one of PRESETS by name."""
    preset = PRESETS.get(spec.strip().lower())
    if preset:
        return list(preset)
    parts = [p for p in spec.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise ValueError(f"AOI needs w,s,e,n or a preset name, got {spec!r}")
    w, s, e, n = (float(p) for p in parts)
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValueError("longitudes must sit in -180..180")
    if not (-85.05 <= s <= 85.05 and -85.05 <= n <= 85.05):
        raise ValueError("latitudes must sit inside the Mercator limit, -85.05..85.05")
    return [min(w, e), min(s, n), max(w, e), max(s, n)]


def tiles_at(aoi: list[float], z: int) -> Iterator[tuple[int, int, int]]:
    x0, y0, x1, y1 = tile_range(aoi, z)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            yield z, x, y


# ------------------------------------------------------------------ pre-flight
def plan(aoi: list[float], zooms: list[int], provider: Provider) -> dict:
    """Tile count and byte estimate per zoom. Printed before a single request,
    because the operator deciding whether to spend the last of the venue's
    bandwidth deserves the number first."""
    per_zoom = []
    total = 0
    for z in zooms:
        x0, y0, x1, y1 = tile_range(aoi, z)
        count = (x1 - x0 + 1) * (y1 - y0 + 1)
        total += count
        per_zoom.append(
            {
                "zoom": z,
                "count": count,
                "x": [x0, x1],
                "y": [y0, y1],
                "est_bytes": count * provider.avg_bytes,
                "over_max_zoom": z > provider.max_zoom,
            }
        )
    return {
        "aoi": aoi,
        "provider": provider.name,
        "zooms": zooms,
        "per_zoom": per_zoom,
        "tiles": total,
        "est_bytes": total * provider.avg_bytes,
        "avg_bytes_assumed": provider.avg_bytes,
    }


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


# ------------------------------------------------------------------ downloading
def _to_png(blob: bytes) -> bytes:
    """A .png path must hold PNG bytes. Esri serves JPEG, and MapLibre decodes
    raster tiles from the bytes, not the extension, so re-encode rather than
    lie. Pillow is already a hard dependency for thumbnails."""
    if blob[:8] == PNG_MAGIC:
        return blob
    import io

    from PIL import Image

    with Image.open(io.BytesIO(blob)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "PNG", optimize=True)
        return buf.getvalue()


def fetch_one(
    provider: Provider,
    z: int,
    x: int,
    y: int,
    dest: Path,
    *,
    timeout: float = 20.0,
    retries: int = 3,
) -> tuple[str, int]:
    """Fetch one tile to dest. Returns (outcome, bytes_on_disk) where outcome is
    "ok", "refused", "missing" or "error: <reason>". Writes through a temp file so
    an interrupted run never leaves a half tile that a resume would skip."""
    url = provider.url.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 1.0
    last = "error"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
                blocked = r.headers.get("x-blocked")
        except urllib.error.HTTPError as exc:
            # 404 is a real answer from a provider whose coverage stops here.
            # Retrying it just burns the clock.
            if exc.code in (404, 204):
                return "missing", 0
            last = f"http {exc.code}"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = f"net {type(exc).__name__}"
        else:
            if blocked or (provider.refusal_bytes and len(blob) in provider.refusal_bytes):
                return "refused", len(blob)
            if not blob or (blob[:8] != PNG_MAGIC and blob[:3] != JPEG_MAGIC):
                last = f"not an image, {len(blob)} bytes"
            else:
                try:
                    png = _to_png(blob)
                except Exception:  # Pillow raises a wide family on bad payloads
                    return "error", 0
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(png)
                tmp.replace(dest)
                return "ok", len(png)
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return f"error: {last}", 0


def download(
    aoi: list[float],
    zooms: list[int],
    provider: Provider,
    out: Path,
    *,
    rate: float = 5.0,
    retries: int = 3,
    timeout: float = 20.0,
    log=print,
) -> dict:
    """Walk every tile in the AOI at each zoom. Skips what is already on disk,
    so this is the resume path as well as the first-run path."""
    interval = 1.0 / rate if rate > 0 else 0.0
    per_zoom: list[dict] = []
    refused_seen = 0
    totals = {"ok": 0, "skipped": 0, "missing": 0, "refused": 0, "error": 0, "bytes": 0}
    started = time.time()
    for z in zooms:
        if z > provider.max_zoom:
            log(f"  z{z}: skipped, above {provider.name} max zoom {provider.max_zoom}")
            per_zoom.append({"zoom": z, "skipped_zoom": True})
            continue
        row = {"zoom": z, "ok": 0, "skipped": 0, "missing": 0, "refused": 0, "error": 0, "bytes": 0}
        first_error = ""
        for _, x, y in tiles_at(aoi, z):
            dest = out / str(z) / str(x) / f"{y}.png"
            if dest.exists() and dest.stat().st_size > 0:
                row["skipped"] += 1
                row["bytes"] += dest.stat().st_size
                continue
            outcome, nbytes = fetch_one(
                provider, z, x, y, dest, timeout=timeout, retries=retries
            )
            # fetch_one returns "error: <reason>"; bucket it, keep the first reason.
            bucket = outcome.split(":", 1)[0]
            row[bucket] = row.get(bucket, 0) + 1
            if bucket == "error" and not first_error:
                first_error = outcome
            row["bytes"] += nbytes if bucket == "ok" else 0
            if outcome == "refused":
                refused_seen += 1
                if refused_seen >= 5:
                    log(
                        f"  z{z}: provider {provider.name} is refusing this client "
                        "(5 refusals in a row). Stopping rather than hammering it."
                    )
                    per_zoom.append(row)
                    for k in totals:
                        totals[k] += row.get(k, 0)
                    return {
                        "provider": provider.name,
                        "per_zoom": per_zoom,
                        "totals": totals,
                        "elapsed_s": round(time.time() - started, 1),
                        "aborted": "provider refused this client",
                    }
            else:
                refused_seen = 0
            if interval:
                time.sleep(interval)
        log(
            f"  z{z}: {row['ok']} fetched, {row['skipped']} already cached, "
            f"{row['missing']} not published, {row['error']} failed, {human(row['bytes'])}"
        )
        if first_error:
            log(f"       first failure at z{z}: {first_error}")
            row["first_error"] = first_error
        per_zoom.append(row)
        for k in totals:
            totals[k] += row.get(k, 0)
    return {
        "provider": provider.name,
        "per_zoom": per_zoom,
        "totals": totals,
        "elapsed_s": round(time.time() - started, 1),
        "aborted": None,
    }


# ------------------------------------------------------------------- verifying
def verify(out: Path, zooms: list[int], *, sample: int = 40) -> dict:
    """Check a sample of cached files really are PNGs and not error pages.

    Magic bytes alone are not enough: OSM's refusal is itself a valid PNG. So we
    also flag files that are byte-identical in size to many others, which is the
    signature of one canned image served for every request."""
    bad: list[str] = []
    checked = 0
    sizes: dict[int, int] = {}
    files: list[Path] = []
    for z in zooms:
        zdir = out / str(z)
        if zdir.is_dir():
            files.extend(sorted(zdir.rglob("*.png")))
    if not files:
        return {"checked": 0, "bad": [], "files": 0, "note": "no tiles on disk"}
    step = max(1, len(files) // max(1, sample))
    for path in files[::step][:sample]:
        checked += 1
        try:
            head = path.open("rb").read(8)
        except OSError as exc:
            bad.append(f"{path}: unreadable, {exc}")
            continue
        if head != PNG_MAGIC:
            bad.append(f"{path}: not a PNG, starts {head[:4]!r}")
            continue
        size = path.stat().st_size
        sizes[size] = sizes.get(size, 0) + 1
    identical = [s for s, n in sizes.items() if n >= max(3, checked // 2)]
    note = ""
    if identical and checked >= 4:
        note = (
            f"{len(identical)} byte size(s) repeat across most of the sample "
            f"({identical}), which is what a canned refusal image looks like"
        )
    return {
        "files": len(files),
        "checked": checked,
        "bad": bad,
        "suspicious_uniform_sizes": identical,
        "note": note,
    }


def write_attribution(out: Path, providers: list[Provider]) -> Path:
    """Attribution lands beside the tiles, so the cache carries its own licence
    wherever it is copied."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ATTRIBUTION.txt"
    lines = [
        "FIRST LIGHT offline basemap tile cache",
        "",
        "These tiles were fetched once, for one AOI, for offline disaster-triage use.",
        "Every source below requires its attribution to be displayed with the map.",
        "",
    ]
    for p in providers:
        lines.append(f"[{p.name}]{' (satellite)' if p.satellite else ''}")
        lines.append(f"  {p.attribution}")
        lines.append("")
    lines.append(f"Cached: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def aoi_name_of(aoi: list[float]) -> str:
    """Name the AOI if it matches a preset, so the manifest reads as English."""
    for name, box in PRESETS.items():
        if all(abs(a - b) < 1e-9 for a, b in zip(aoi, box)):
            return name
    return "custom"


def write_manifest(out: Path, aoi: list[float], provider: Provider, result: dict) -> Path:
    """Record which AOI this cache covers, so the reseed script can warn when the
    cache and the configured AOI disagree instead of painting the wrong county."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "MANIFEST.json"
    prior: list[dict] = []
    if path.is_file():
        try:
            prior = json.loads(path.read_text(encoding="utf-8")).get("runs") or []
        except (OSError, ValueError):
            prior = []
    run = {
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "aoi": aoi,
        "aoi_name": aoi_name_of(aoi),
        "provider": provider.name,
        "satellite": provider.satellite,
        "attribution": provider.attribution,
        "zooms": [r["zoom"] for r in result["per_zoom"]],
        "per_zoom": result["per_zoom"],
        "totals": result["totals"],
        "elapsed_s": result["elapsed_s"],
        "aborted": result["aborted"],
    }
    body = {
        "note": (
            "Offline basemap tile cache for FIRST LIGHT. Compare aoi against "
            "config.AOI before a demo: a cache for the wrong county paints the "
            "wrong county."
        ),
        "aoi": aoi,
        "aoi_name": run["aoi_name"],
        "runs": prior + [run],
    }
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fetch_tiles",
        description="Cache raster basemap tiles for the AOI into web/tiles, for offline use.",
    )
    ap.add_argument(
        "--aoi",
        default=",".join(str(v) for v in config.AOI),
        help="w,s,e,n or a preset name (%s); default is config.AOI"
        % ", ".join(PRESETS),
    )
    ap.add_argument("--zoom", default="12-16", help="e.g. 12-18 or 12,14,16; default 12-16")
    ap.add_argument("--out", type=Path, default=config.WEB / "tiles", help="output directory")
    ap.add_argument(
        "--sat",
        action="store_true",
        help="fetch satellite imagery into <out>/sat instead of the tactical basemap; OFF by default",
    )
    ap.add_argument(
        "--provider",
        default=None,
        help="one of %s; default %s, or %s with --sat"
        % (", ".join(PROVIDERS), DEFAULT_TACTICAL, DEFAULT_SATELLITE),
    )
    ap.add_argument("--rate", type=float, default=5.0, help="requests per second; default 5")
    ap.add_argument("--retries", type=int, default=3, help="attempts per tile; default 3")
    ap.add_argument("--timeout", type=float, default=20.0, help="per-request seconds; default 20")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP, help=f"refuse above this many tiles without --yes; default {DEFAULT_CAP}")
    ap.add_argument("--yes", action="store_true", help="proceed past the tile cap")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--verify-only", action="store_true", help="check what is already cached and stop")
    args = ap.parse_args(argv)

    try:
        aoi = parse_aoi(args.aoi)
        zooms = parse_zooms(args.zoom)
    except ValueError as exc:
        print(f"fetch_tiles: {exc}")
        return 2

    name = args.provider or (DEFAULT_SATELLITE if args.sat else DEFAULT_TACTICAL)
    provider = PROVIDERS.get(name)
    if provider is None:
        print(f"fetch_tiles: unknown provider {name!r}, choose from {', '.join(PROVIDERS)}")
        return 2
    if args.sat and not provider.satellite:
        print(f"fetch_tiles: --sat needs a satellite provider, {provider.name} is not one")
        return 2
    if provider.satellite and not args.sat:
        print(f"fetch_tiles: {provider.name} is satellite imagery, pass --sat to place it under <out>/sat")
        return 2

    out = args.out / "sat" if args.sat else args.out
    if args.verify_only:
        rep = verify(out, zooms)
        print(f"cached files: {rep['files']}, sampled {rep['checked']}")
        for line in rep["bad"]:
            print(f"  BAD  {line}")
        if rep.get("note"):
            print(f"  note: {rep['note']}")
        return 1 if rep["bad"] or not rep["files"] else 0

    p = plan(aoi, zooms, provider)
    print(f"AOI {aoi}  provider {provider.name}  -> {out}")
    for row in p["per_zoom"]:
        flag = "  (above provider max zoom, will be skipped)" if row["over_max_zoom"] else ""
        print(
            f"  z{row['zoom']:>2}: {row['count']:>7} tiles  "
            f"x {row['x'][0]}-{row['x'][1]}  y {row['y'][0]}-{row['y'][1]}  "
            f"est {human(row['est_bytes'])}{flag}"
        )
    print(
        f"  total: {p['tiles']} tiles, est {human(p['est_bytes'])} "
        f"(at a measured {human(provider.avg_bytes)} per tile)"
    )
    if args.rate > 0:
        print(f"  at {args.rate:g} req/s that is about {p['tiles'] / args.rate / 60:.1f} minutes if nothing is cached")

    if args.dry_run:
        print("dry run, nothing fetched")
        return 0
    if p["tiles"] > args.cap and not args.yes:
        print(
            f"fetch_tiles: {p['tiles']} tiles is above the {args.cap} cap. "
            "Narrow the AOI or the zoom range, or pass --yes if you mean it."
        )
        return 2

    free = shutil.disk_usage(args.out.parent if args.out.parent.exists() else Path(".")).free
    if free < p["est_bytes"] * 2:
        print(f"fetch_tiles: only {human(free)} free, estimate is {human(p['est_bytes'])}. Refusing.")
        return 2

    attribution = write_attribution(out, [provider])
    print(f"wrote {attribution}")
    print("fetching:")
    result = download(
        aoi,
        zooms,
        provider,
        out,
        rate=args.rate,
        retries=args.retries,
        timeout=args.timeout,
        log=print,
    )
    t = result["totals"]
    print(
        f"\ndone in {result['elapsed_s']}s: {t['ok']} fetched, {t['skipped']} already cached, "
        f"{t['missing']} not published, {t['refused']} refused, {t['error']} failed"
    )
    print(f"cache size for these zooms: {human(t['bytes'])}")
    if result["aborted"]:
        print(f"ABORTED: {result['aborted']}")
    manifest = write_manifest(out, aoi, provider, result)
    print(f"wrote {manifest}")

    rep = verify(out, zooms)
    print(f"verify: {rep['files']} files on disk, sampled {rep['checked']}")
    for line in rep["bad"]:
        print(f"  BAD  {line}")
    if rep.get("note"):
        print(f"  note: {rep['note']}")
    if not rep["files"]:
        print("map.js will print 'basemap tiles not cached'")
        return 1
    probe = out / "12"
    if not args.sat and not probe.is_dir():
        print(
            "note: map.js probes z12 at the AOI centre to decide whether tiles exist. "
            "Include z12 in --zoom or the legend will report the cache as absent."
        )
    return 1 if (rep["bad"] or result["aborted"] or t["error"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
