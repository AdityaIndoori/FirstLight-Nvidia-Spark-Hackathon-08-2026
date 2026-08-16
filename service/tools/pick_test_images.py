#!/usr/bin/env python3
"""Pick a spread of real NOAA post-Michael tiles for manual upload testing.

WHY curated rather than "the first five": a demo set has to exercise the paths a
judge will poke at. This scores every candidate tile on how much its footprints
vary in pixel texture (a proxy for damage spread), then picks across the range so
the operator sees intact, damaged and mixed frames rather than five of the same.
Also emits the sidecar bounds each tile needs and a README of what to expect.

Run on the box; copies land in --out.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def score(path: Path) -> tuple[float, float]:
    """(texture energy, mean luma) for a tile: cheap, deterministic, no model."""
    from PIL import Image, ImageFilter, ImageStat

    with Image.open(path) as im:
        g = im.convert("L")
        energy = ImageStat.Stat(g.filter(ImageFilter.FIND_EDGES)).mean[0]
        luma = ImageStat.Stat(g).mean[0]
    return energy, luma


def main() -> int:
    ap = argparse.ArgumentParser()
    # noaa_frames holds 4x4 MOSAICS (1024 px, ~525 m across). noaa_z18 holds the raw
    # 256 px map tiles they were built from, which is 16x less ground per frame: a
    # building crop out of one is a few dozen pixels, and the privacy gate's 1280 px
    # tiled sweep degenerates to a single downscaled pass. Grading is only meaningful
    # on the mosaics, so those are the default.
    ap.add_argument("--src", default="data/noaa_frames")
    ap.add_argument("--out", default="/tmp/firstlight_test_images")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument(
        "--min-px",
        type=int,
        default=512,
        help="refuse frames smaller than this; guards against picking raw map tiles",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tiles = sorted(p for p in src.glob("*.jpg") if (src / f"{p.name}.bounds.json").exists())
    if not tiles:
        print(f"no tiles with bounds in {src}", file=sys.stderr)
        return 1

    scored = []
    skipped_small = 0
    for p in tiles:
        try:
            from PIL import Image

            with Image.open(p) as probe:
                if min(probe.size) < args.min_px:
                    skipped_small += 1
                    continue
            energy, luma = score(p)
        except Exception:
            continue
        scored.append((energy, luma, p))
    scored.sort()

    if skipped_small:
        print(f"skipped {skipped_small} frame(s) under {args.min_px} px", file=sys.stderr)
    if not scored:
        print(
            f"no frames of at least {args.min_px} px in {src}. Raw map tiles are 256 px; "
            f"mosaic them first with scripts/fetch_noaa_aerials.py --span 4, or point "
            f"--src at data/noaa_frames.",
            file=sys.stderr,
        )
        return 1

    # Spread across the texture range: low energy is intact roofs and open ground,
    # high energy is debris fields and broken roofs.
    n = min(args.n, len(scored))
    picks = [scored[round(i * (len(scored) - 1) / max(1, n - 1))] for i in range(n)]

    manifest = []
    for i, (energy, luma, p) in enumerate(picks):
        bounds = json.loads((src / f"{p.name}.bounds.json").read_text())
        # A plain, obviously-a-test name, and the sidecar the ingest path looks for.
        name = f"drone_{i:02d}_{p.stem.split('_')[-2]}_{p.stem.split('_')[-1]}.jpg"
        shutil.copy2(p, out / name)
        (out / f"{name}.bounds.json").write_text(json.dumps(bounds))
        manifest.append(
            {
                "file": name,
                "source": p.name,
                "bounds": bounds,
                "edge_energy": round(energy, 2),
                "mean_luma": round(luma, 1),
            }
        )
        print(f"{name}: energy={energy:7.2f} luma={luma:5.1f} bounds={bounds}")

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} tiles in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
