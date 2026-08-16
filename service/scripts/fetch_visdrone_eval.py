#!/usr/bin/env python3
"""Assemble the held-out person-detection eval set A5 needs: real aerial tiles
with a boolean person label per tile.

This is the only thing blocking a published gate-recall number. The synthetic
person fixture in make_demo_kit.py provably does not fool the real detector
(drawn figures, two scales, zero detections on the box), so the number has to be
measured on real aerial frames with real people in them.

Output is exactly what `scripts/gate_eval.py --tiles DIR --labels labels.json`
already expects:

    data/gate_eval/tiles/*.jpg
    data/gate_eval/labels.json        filename -> bool has_person
    data/gate_eval/PROVENANCE.json    per image: source, original name, basis

Source, verified reachable from the Spark on 2026-08-16:

  visdrone_val_ultralytics    (default positives)
      VisDrone2019-DET-val.zip, 81,638,851 bytes, from the Ultralytics assets
      release. This is the CANONICAL VisDrone annotation format, not a
      re-export: one comma-separated line per object,
      `x,y,w,h,score,category,truncation,occlusion`, where category 1 is
      pedestrian and 2 is people, and score 0 marks an ignored region. Measured
      census of its 548 frames: 531 contain a person, 17 do not.
  visdrone_testdev_ultralytics  (negatives)
      VisDrone2019-DET-test-dev.zip, 311,251,787 bytes, same format. Measured
      census of its 1610 frames: 1267 with a person, 343 without. Val alone
      cannot balance the set, which is the whole reason this split is here.
  visdrone_hf_banu4prasad      (fallback)
      huggingface.co/datasets/banu4prasad/VisDrone-Dataset, per-file HTTP, the
      same imagery re-exported to YOLO txt where class 0 is pedestrian and 1 is
      people. The re-export drops the score column, so the ignored-region
      distinction is gone; that is recorded in PROVENANCE.json as a weaker
      annotation basis rather than glossed over.

The canonical aiskyeye.com host answers, but its actual image payloads sit
behind Google Drive interstitials that are not a scriptable GET, which is why
the Ultralytics release mirror is the primary.

Never fabricates an image or a label. If no source is reachable this exits non
zero with what it tried, because a recall number measured on synthetic people
would be worse than no number at all.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Run as `python scripts/fetch_visdrone_eval.py` from service/ as well as `python -m`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

# PUBLIC API
# SPLITS: dict[str, Split]
# has_person_canonical(text: str) -> tuple[bool, int]
# has_person_yolo(text: str) -> tuple[bool, int]
# fetch_zip(url, dest, *, timeout, log) -> Path          resumable via HTTP Range
# census(zip_path, split) -> dict[str, dict]             stem -> label record
# pick_balanced(records, want_pos, want_neg) -> tuple[list, list]
# assemble(out, limit, *, splits, timeout, log) -> dict
# main(argv=None) -> int

USER_AGENT = (
    "FIRSTLIGHT-eval-fetch/1.0 "
    "(offline county EOC privacy-gate eval set; NVIDIA DGX Spark Hackathon 2026)"
)

JPEG_MAGIC = b"\xff\xd8\xff"

# VisDrone category ids in the CANONICAL annotation format.
CANONICAL_PERSON_CATS = {1, 2}  # 1 pedestrian, 2 people
# ...and in the YOLO re-export, where the ignored class is dropped and everything
# shifts down by one.
YOLO_PERSON_CLASSES = {0, 1}  # 0 pedestrian, 1 people


@dataclass(frozen=True)
class Split:
    name: str
    url: str
    kind: str  # "canonical_zip" or "hf_yolo"
    dataset: str
    split: str
    annotation_basis: str
    licence: str
    expected_bytes: int = 0
    # Measured census, printed so the operator knows what the split can supply
    # before spending bandwidth on it.
    measured: str = ""


SPLITS: dict[str, Split] = {
    "val": Split(
        name="val",
        url="https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
        kind="canonical_zip",
        dataset="VisDrone2019-DET",
        split="validation",
        annotation_basis=(
            "canonical VisDrone per-image txt, x,y,w,h,score,category,truncation,occlusion; "
            "has_person = any object with category in {1 pedestrian, 2 people} and score == 1 "
            "(score 0 marks an ignored region and is deliberately not counted)"
        ),
        licence=(
            "VisDrone benchmark, Lab of Machine Learning and Data Mining, Tianjin University. "
            "Free for academic and non-commercial research use; cite Zhu et al., "
            "'Detection and Tracking Meet Drones Challenge', arXiv:2001.06303. "
            "Mirror: github.com/ultralytics/assets release v0.0.0."
        ),
        expected_bytes=81638851,
        measured="548 frames: 531 with a person, 17 without",
    ),
    "test-dev": Split(
        name="test-dev",
        url="https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
        kind="canonical_zip",
        dataset="VisDrone2019-DET",
        split="test-dev",
        annotation_basis=(
            "canonical VisDrone per-image txt, x,y,w,h,score,category,truncation,occlusion; "
            "has_person = any object with category in {1 pedestrian, 2 people} and score == 1"
        ),
        licence=(
            "VisDrone benchmark, Lab of Machine Learning and Data Mining, Tianjin University. "
            "Free for academic and non-commercial research use; cite Zhu et al., arXiv:2001.06303. "
            "Mirror: github.com/ultralytics/assets release v0.0.0."
        ),
        expected_bytes=311251787,
        measured="1610 frames: 1267 with a person, 343 without",
    ),
}

# Fallback mirror, used per-file over HTTP when the zips cannot be reached.
HF_REPO = "banu4prasad/VisDrone-Dataset"
HF_TREE = "https://huggingface.co/api/datasets/%s/tree/main/%%s?recursive=true&limit=1000" % HF_REPO
HF_RESOLVE = "https://huggingface.co/datasets/%s/resolve/main/%%s" % HF_REPO
HF_SPLIT_DIR = {"val": "VisDrone2019-DET-val", "test-dev": "VisDrone2019-DET-test-dev"}
HF_LICENCE = (
    "VisDrone imagery re-exported to YOLO format by Hugging Face user banu4prasad, "
    "declared cc-by-nc-sa-3.0 on the dataset card. Underlying benchmark terms still apply: "
    "academic and non-commercial use, cite Zhu et al., arXiv:2001.06303."
)


# ------------------------------------------------------------------- labelling
def has_person_canonical(text: str) -> tuple[bool, int]:
    """Canonical VisDrone annotation to (has_person, person_object_count).

    Ignored regions carry score 0 and are excluded on purpose: they are the parts
    of the frame the benchmark tells you not to score, so counting them would put
    a person label on a tile whose person the annotators disclaimed."""
    count = 0
    for line in text.splitlines():
        parts = line.strip().rstrip(",").split(",")
        if len(parts) < 6:
            continue
        try:
            score = int(parts[4])
            category = int(parts[5])
        except ValueError:
            continue
        if category in CANONICAL_PERSON_CATS and score == 1:
            count += 1
    return count > 0, count


def has_person_yolo(text: str) -> tuple[bool, int]:
    """YOLO re-export to (has_person, person_object_count). No score column
    exists here, so every listed person counts."""
    count = 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls in YOLO_PERSON_CLASSES:
            count += 1
    return count > 0, count


# ------------------------------------------------------------------ downloading
def fetch_zip(url: str, dest: Path, *, timeout: float = 120.0, log=print) -> Path:
    """Download to dest, resuming a partial file with an HTTP Range request.

    The venue's link is the thing most likely to die mid-run, so a 311 MB
    download that has to restart from zero is a real risk, not a hypothetical."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.is_file() and dest.stat().st_size > 0:
        log(f"  cached: {dest.name}, {dest.stat().st_size} bytes")
        return dest
    have = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
        log(f"  resuming {dest.name} at {have} bytes")
    req = urllib.request.Request(url, headers=headers)
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # A server that ignores Range answers 200 with the whole file; starting
        # over is correct then, and appending would corrupt the archive.
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        with part.open(mode) as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                have += len(chunk)
    part.replace(dest)
    log(f"  fetched {dest.name}: {have} bytes in {time.time() - started:.1f}s")
    return dest


# --------------------------------------------------------------------- census
@dataclass
class Record:
    stem: str
    has_person: bool
    person_objects: int
    source: str
    split: str
    original_image: str
    original_annotation: str
    annotation_basis: str
    licence: str
    sequence: str = field(default="")


def census(zip_path: Path, split: Split) -> dict[str, Record]:
    """Read every annotation in a canonical VisDrone zip and label its frame.
    Returns stem -> Record for frames that have BOTH an annotation and an image."""
    out: dict[str, Record] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        images = {Path(n).stem: n for n in names if n.lower().endswith(".jpg")}
        for name in names:
            if "/annotations/" not in name or not name.endswith(".txt"):
                continue
            stem = Path(name).stem
            image = images.get(stem)
            if not image:
                continue
            flag, count = has_person_canonical(z.read(name).decode("utf-8", "replace"))
            out[stem] = Record(
                stem=stem,
                has_person=flag,
                person_objects=count,
                source=f"visdrone_{split.name.replace('-', '')}_ultralytics",
                split=split.split,
                original_image=image,
                original_annotation=name,
                annotation_basis=split.annotation_basis,
                licence=split.licence,
                sequence=stem.split("_")[0],
            )
    return out


def _hf_get(url: str, *, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def census_hf(split_name: str, *, timeout: float = 60.0) -> dict[str, Record]:
    """Fallback census over the Hugging Face mirror, one label file per request.

    Only the labels are pulled here; images come later and only for the frames
    that survive selection, because pulling 548 full-resolution frames to keep
    100 is a waste of a link that may be about to disappear."""
    sub = HF_SPLIT_DIR[split_name]
    files: list[dict] = []
    cursor = ""
    while True:
        url = HF_TREE % sub
        if cursor:
            url += "&cursor=" + cursor
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            page = json.load(r)
            link = r.headers.get("Link", "")
        files += [f for f in page if f.get("type") == "file"]
        if "cursor=" not in link:
            break
        cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
    labels = {Path(f["path"]).stem: f["path"] for f in files if f["path"].endswith(".txt")}
    images = {Path(f["path"]).stem: f["path"] for f in files if f["path"].lower().endswith(".jpg")}
    out: dict[str, Record] = {}
    for stem, lab in sorted(labels.items()):
        image = images.get(stem)
        if not image:
            continue
        text = _hf_get(HF_RESOLVE % lab, timeout=timeout).decode("utf-8", "replace")
        flag, count = has_person_yolo(text)
        out[stem] = Record(
            stem=stem,
            has_person=flag,
            person_objects=count,
            source="visdrone_hf_banu4prasad",
            split=split_name,
            original_image=image,
            original_annotation=lab,
            annotation_basis=(
                "YOLO re-export txt, class 0 pedestrian and 1 people; has_person = any such box. "
                "The re-export drops VisDrone's score column, so ignored regions cannot be "
                "excluded: a weaker basis than the canonical annotation"
            ),
            licence=HF_LICENCE,
            sequence=stem.split("_")[0],
        )
    return out


# ------------------------------------------------------------------- selection
def pick_balanced(
    records: list[Record], want_pos: int, want_neg: int
) -> tuple[list[Record], list[Record]]:
    """Choose frames round-robin across drone sequences.

    VisDrone frames come from continuous flights, so the first N filenames in
    sorted order are often near-duplicates of one scene. A recall number measured
    on fifty frames of the same intersection is not a recall number. Round-robin
    by sequence id spreads the set across flights, and it is deterministic, so a
    resumed run picks the same frames."""

    def spread(pool: list[Record], want: int) -> list[Record]:
        by_seq: dict[str, list[Record]] = {}
        for r in sorted(pool, key=lambda r: r.stem):
            by_seq.setdefault(r.sequence, []).append(r)
        order = sorted(by_seq)
        picked: list[Record] = []
        depth = 0
        while len(picked) < want:
            added = False
            for seq in order:
                if depth < len(by_seq[seq]):
                    picked.append(by_seq[seq][depth])
                    added = True
                    if len(picked) >= want:
                        break
            if not added:
                break
            depth += 1
        return picked

    pos = spread([r for r in records if r.has_person], want_pos)
    neg = spread([r for r in records if not r.has_person], want_neg)
    return pos, neg


# -------------------------------------------------------------------- assembly
def _extract(zip_path: Path, member: str, dest: Path) -> int:
    with zipfile.ZipFile(zip_path) as z:
        blob = z.read(member)
    if blob[:3] != JPEG_MAGIC:
        raise ValueError(f"{member} is not a JPEG, starts {blob[:4]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(blob)
    tmp.replace(dest)
    return len(blob)


def assemble(
    out: Path,
    limit: int,
    *,
    splits: list[str],
    timeout: float = 300.0,
    cache: Optional[Path] = None,
    log=print,
) -> dict:
    """Fetch, label, balance and write the eval set. Resumable: images already on
    disk are kept, and labels.json plus PROVENANCE.json are rewritten from the
    full selection each time so the three artifacts can never drift apart."""
    tiles = out / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    cache = cache or (out / "_cache")
    want_pos = limit // 2
    want_neg = limit - want_pos

    tried: list[str] = []
    records: list[Record] = []
    zips: dict[str, Path] = {}

    # Canonical zips first, in the order given. Stop early once the balance is
    # satisfiable, so a small --limit never pays for the 311 MB test-dev split.
    for name in splits:
        split = SPLITS.get(name)
        if split is None:
            log(f"  unknown split {name!r}, skipping")
            continue
        have_pos = sum(1 for r in records if r.has_person)
        have_neg = sum(1 for r in records if not r.has_person)
        if have_pos >= want_pos and have_neg >= want_neg:
            log(f"  balance already satisfiable, not downloading {name}")
            break
        log(f"source {split.name} ({split.measured}):")
        tried.append(f"{split.url} -> ")
        try:
            path = fetch_zip(split.url, cache / Path(split.url).name, timeout=timeout, log=log)
            zips[split.name] = path
            found = census(path, split)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, zipfile.BadZipFile) as exc:
            tried[-1] += f"FAILED {type(exc).__name__}: {exc}"
            log(f"  unreachable: {type(exc).__name__}: {exc}")
            continue
        tried[-1] += f"ok, {len(found)} labelled frames"
        pos = sum(1 for r in found.values() if r.has_person)
        log(f"  census: {len(found)} frames, {pos} with a person, {len(found) - pos} without")
        records += list(found.values())

    if not records:
        log("no canonical zip was reachable, trying the Hugging Face mirror")
        for name in splits:
            if name not in HF_SPLIT_DIR:
                continue
            tried.append(f"{HF_RESOLVE % HF_SPLIT_DIR[name]} -> ")
            try:
                found = census_hf(name, timeout=min(timeout, 60.0))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
                tried[-1] += f"FAILED {type(exc).__name__}: {exc}"
                log(f"  mirror unreachable: {type(exc).__name__}: {exc}")
                continue
            tried[-1] += f"ok, {len(found)} labelled frames"
            records += list(found.values())
            pos = sum(1 for r in found.values() if r.has_person)
            log(f"  census: {len(found)} frames, {pos} with a person, {len(found) - pos} without")
            if pos >= want_pos and len(found) - pos >= want_neg:
                break

    if not records:
        return {"ok": False, "tried": tried, "reason": "no source reachable"}

    pos, neg = pick_balanced(records, want_pos, want_neg)
    chosen = pos + neg
    log(
        f"selected {len(chosen)} frames: {len(pos)} with a person, {len(neg)} without "
        f"(asked for {want_pos}/{want_neg})"
    )
    if len(pos) < want_pos or len(neg) < want_neg:
        log(
            "  the sources could not fill the requested balance; the actual split "
            "is reported below and written to PROVENANCE.json"
        )

    labels: dict[str, bool] = {}
    provenance: list[dict] = []
    written = reused = failed = 0
    total_bytes = 0
    for rec in chosen:
        dest = tiles / f"{rec.stem}.jpg"
        if dest.is_file() and dest.stat().st_size > 0:
            reused += 1
            total_bytes += dest.stat().st_size
        else:
            zp = zips.get("val") if rec.split == "validation" else zips.get("test-dev")
            try:
                if zp and rec.source.startswith("visdrone_"):
                    total_bytes += _extract(zp, rec.original_image, dest)
                else:
                    blob = _hf_get(HF_RESOLVE % rec.original_image, timeout=min(timeout, 120.0))
                    if blob[:3] != JPEG_MAGIC:
                        raise ValueError("mirror served a non-JPEG payload")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".part")
                    tmp.write_bytes(blob)
                    tmp.replace(dest)
                    total_bytes += len(blob)
                written += 1
            except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
                log(f"  could not write {rec.stem}: {type(exc).__name__}: {exc}")
                failed += 1
                continue
        labels[dest.name] = rec.has_person
        provenance.append(
            {
                "filename": dest.name,
                "has_person": rec.has_person,
                "person_objects": rec.person_objects,
                "source_dataset": rec.source,
                "split": rec.split,
                "original_image": rec.original_image,
                "original_annotation": rec.original_annotation,
                "annotation_basis": rec.annotation_basis,
                "licence": rec.licence,
            }
        )

    n_pos = sum(1 for v in labels.values() if v)
    summary = {
        "ok": bool(labels),
        "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "images": len(labels),
        "with_person": n_pos,
        "without_person": len(labels) - n_pos,
        "requested": limit,
        "written_this_run": written,
        "already_present": reused,
        "failed": failed,
        "bytes_on_disk": total_bytes,
        "splits_used": sorted({r["split"] for r in provenance}),
        "sources_used": sorted({r["source_dataset"] for r in provenance}),
        "tried": tried,
    }
    if not labels:
        return {"ok": False, "tried": tried, "reason": "every image write failed"}

    (out / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "note": (
                    "Held-out eval set for the FIRST LIGHT privacy gate (A5 person recall). "
                    "Every image and every label comes from a published benchmark; nothing "
                    "here is synthetic. A recall number has to be able to name its test set."
                ),
                "summary": summary,
                "images": sorted(provenance, key=lambda r: r["filename"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


# ------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fetch_visdrone_eval",
        description="Assemble the real-aerial person-detection eval set for A5's gate recall.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=config.DATA / "gate_eval",
        help="output directory; default data/gate_eval",
    )
    ap.add_argument("--limit", type=int, default=100, help="tiles to assemble; default 100")
    ap.add_argument(
        "--splits",
        default="val,test-dev",
        help="comma-separated, in order of use: %s. val alone cannot balance the set, "
        "it has only 17 no-person frames" % ", ".join(SPLITS),
    )
    ap.add_argument("--cache", type=Path, default=None, help="where to keep the source zips")
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request seconds; default 300")
    ap.add_argument(
        "--keep-cache",
        action="store_true",
        help="keep the downloaded source zips (about 375 MB) for a later re-assembly",
    )
    args = ap.parse_args(argv)

    if args.limit < 2:
        print("fetch_visdrone_eval: --limit must be at least 2 to hold both classes")
        return 2
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in SPLITS]
    if unknown:
        print(f"fetch_visdrone_eval: unknown split(s) {unknown}, choose from {', '.join(SPLITS)}")
        return 2

    cache = args.cache or (args.out / "_cache")
    need = sum(SPLITS[s].expected_bytes for s in splits)
    free = shutil.disk_usage(args.out.parent if args.out.parent.exists() else Path(".")).free
    if free < need * 2:
        print(f"fetch_visdrone_eval: {free} bytes free, sources need about {need}. Refusing.")
        return 2

    print(f"assembling {args.limit} tiles into {args.out}")
    result = assemble(
        args.out, args.limit, splits=splits, timeout=args.timeout, cache=cache, log=print
    )

    if not result.get("ok"):
        print(f"\nFAILED: {result.get('reason')}")
        for line in result.get("tried", []):
            print(f"  tried {line}")
        print(
            "\nNo eval set was written. This script never generates synthetic tiles: "
            "a recall number measured on drawn people would be worse than no number. "
            "Get a link, or copy data/gate_eval from a box that has one."
        )
        return 1

    print(
        f"\n{result['images']} tiles: {result['with_person']} with a person, "
        f"{result['without_person']} without "
        f"({result['written_this_run']} written now, {result['already_present']} already present, "
        f"{result['failed']} failed)"
    )
    print(f"  {result['bytes_on_disk']} bytes of imagery")
    print(f"  sources: {', '.join(result['sources_used'])}")
    print(f"  splits: {', '.join(result['splits_used'])}")
    print(f"  wrote {args.out / 'labels.json'} and {args.out / 'PROVENANCE.json'}")
    if result["images"] < result["requested"]:
        print(
            f"  NOTE: asked for {result['requested']}, got {result['images']}. "
            "The split above is the real one; publish it, do not round it up."
        )
    print(
        f"\nmeasure the gate with:\n"
        f"  python scripts/gate_eval.py --tiles {args.out / 'tiles'} "
        f"--labels {args.out / 'labels.json'} --sweep 0.15,0.2,0.25,0.3"
    )

    if not args.keep_cache and cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)
        print(f"  removed the source zip cache at {cache} (pass --keep-cache to keep it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
