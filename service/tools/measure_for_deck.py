#!/usr/bin/env python3
"""Measure the numbers the slide deck asserts, on this box, right now.

WHY this exists: the deck claimed 4.2 s per tile, 98 GB and 240 W while the box
measured 33 s, 120 GB and 10 W idle. Numbers that reach a judge have to come from
here. Writes JSON to stdout; demo/measured.json is generated from it.

Power is sampled UNDER LOAD, because an idle draw on a slide next to a throughput
claim is the kind of contradiction the deck is being fixed for.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8081"


def api(path: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as fh:
        return json.load(fh)


def smi(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


class PowerSampler(threading.Thread):
    """Sample GPU power until stopped, so we can report draw under load."""

    def __init__(self, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[float] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            rows = smi("power.draw")
            for r in rows:
                try:
                    self.samples.append(float(r))
                except ValueError:
                    pass
            self._halt.wait(self.interval)

    def finish(self) -> dict:
        # NOT named stop(): threading.Thread._stop is an internal attribute and
        # shadowing it breaks join().
        self._halt.set()
        self.join(timeout=5)
        if not self.samples:
            return {"error": "nvidia-smi reported no power readings"}
        return {
            "median_w": round(statistics.median(self.samples), 1),
            "peak_w": round(max(self.samples), 1),
            "samples": len(self.samples),
        }


def weights_on_disk() -> dict:
    """Actual bytes of the model weights the servers loaded.

    The captioner is served from a container mount rather than the HF cache, so its
    host path is resolved from docker instead of guessed - reporting "/model" told a
    judge nothing and hid a 15 GB VL model from the memory table.
    """
    hub = Path.home() / ".cache/huggingface/hub"
    out: dict = {}
    total = 0.0

    def add(label: str, path: Path) -> None:
        nonlocal total
        if not path.is_dir() and not path.is_file():
            out[label] = None
            return
        if path.is_file():
            n = path.stat().st_size
        else:
            n = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        gb = round(n / 1e9, 1)
        out[label] = gb
        total += gb

    add("planner_nano", hub / "models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-FP8")
    add("ballot_lightning", hub / "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    add("embedder", hub / "models--BAAI--bge-small-en-v1.5")
    add("gate_yolo", Path.home() / "firstlight/visdrone-yolov8x.pt")

    # The VL captioner: ask docker where /model actually comes from.
    try:
        res = subprocess.run(
            ["docker", "inspect", "fl-vl", "--format",
             "{{range .Mounts}}{{.Source}}:{{.Destination}}\n{{end}}"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        for line in res.stdout.splitlines():
            src, _, dest = line.partition(":")
            if dest.strip() == "/model" and src:
                add("captioner_vl", Path(src))
                out["captioner_vl_name"] = Path(src).name.replace("nvidia--", "")
                break
    except (OSError, subprocess.SubprocessError):
        out["captioner_vl"] = None

    out["total_gb"] = round(total, 1)
    return out


def upload_pass(images: list[Path]) -> dict:
    """Re-ingest real tiles through the HTTP path and time each one."""
    import mimetypes
    import uuid

    lat, rows = [], []
    for img in images:
        side = img.parent / f"{img.name}.bounds.json"
        parts = [side, img] if side.is_file() else [img]
        boundary = "----fl" + uuid.uuid4().hex
        body = b""
        for p in parts:
            ct = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="files"; filename="{p.name}"\r\n'.encode()
            body += f"Content-Type: {ct}\r\n\r\n".encode()
            body += p.read_bytes() + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{BASE}/api/upload", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=900) as fh:
            res = json.load(fh)
        it = (res.get("items") or [{}])[0]
        ms = int(it.get("latency_ms") or 0)
        if ms:
            lat.append(ms)
        rows.append({
            "file": img.name,
            "ms": ms,
            "status": it.get("status"),
            "geo_source": it.get("geo_source"),
            "stored": it.get("stored"),
            "withheld_reason": it.get("withheld_reason"),
            "buildings": len(it.get("buildings") or []),
        })
    if not lat:
        return {"error": "no tile produced a latency"}
    return {
        "p50_ms": int(statistics.median(lat)),
        "min_ms": min(lat),
        "max_ms": max(lat),
        "tiles": len(lat),
        "detail": rows,
    }


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/firstlight_test_images")
    images = sorted(p for p in src.glob("*.jpg"))
    if not images:
        print(f"no images in {src}", file=sys.stderr)
        return 1

    before = api("/api/status")
    sampler = PowerSampler()
    sampler.start()
    ingest = upload_pass(images)
    power = sampler.finish()
    after = api("/api/status")

    mem_rows = smi("memory.used,memory.total")
    out = {
        "measured_at": time.time(),
        "box": (smi("name") or ["unknown"])[0],
        "ingest_e2e": ingest,
        "gpu_power_under_load": power,
        "gpu_power_idle_w": before.get("gpu_power"),
        "memory_gb_used": round(float(after.get("memory_gb") or 0), 1),
        "memory_gb_total": round(float(after.get("memory_total_gb") or 0), 1),
        "nvidia_smi_memory": mem_rows[0] if mem_rows else None,
        "weights_gb": weights_on_disk(),
        "model_versions": after.get("model_versions"),
        "doubt_distribution": after.get("doubt_distribution"),
        "tally": after.get("tally"),
        "tiles_analyzed": after.get("tiles_analyzed"),
        "tiles_stored": after.get("tiles_stored"),
        "tiles_withheld_from_storage": after.get("tiles_withheld_from_storage"),
    }
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
