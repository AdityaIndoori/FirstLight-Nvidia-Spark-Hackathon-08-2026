#!/usr/bin/env python3
"""Measure what the three model servers actually do on THIS box.

Every number the console and the deck print has to come from here rather than
from a plan, so this writes a JSON block that the slide build reads directly.
Run it on the Spark; it never reaches the network.
"""
from __future__ import annotations

import base64
import io
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

SERVERS = {"nano": 8000, "lightning": 8001, "captioner": 8002}


def chat(port: int, body: dict, timeout: float = 600.0) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh), time.time() - t0


def root_of(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as fh:
        return json.load(fh)["data"][0].get("root", "?")


def text_throughput(name: str, port: int, n_tokens: int = 256, runs: int = 3) -> dict:
    """Decode throughput on a fixed generation length: the honest tok/s number."""
    body = {
        "model": name,
        "messages": [{"role": "user", "content": "Write a plain paragraph about coastal storm damage."}],
        "max_tokens": n_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    rates, lats = [], []
    for _ in range(runs):
        d, dt = chat(port, body)
        out = int(d.get("usage", {}).get("completion_tokens") or 0)
        if out > 0 and dt > 0:
            rates.append(out / dt)
            lats.append(dt * 1000.0)
    if not rates:
        return {"error": "no successful runs"}
    return {
        "model_root": root_of(port),
        "decode_tok_s_median": round(statistics.median(rates), 1),
        "latency_ms_median": int(statistics.median(lats)),
        "output_tokens": n_tokens,
        "runs": len(rates),
    }


def _test_jpeg() -> str:
    """A synthetic aerial-ish crop, so the VL timing needs no dataset on disk."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 640), (96, 104, 84))
    d = ImageDraw.Draw(img)
    for x in range(0, 640, 80):
        d.line([(x, 0), (x, 640)], fill=(70, 74, 62), width=3)
    d.rectangle([180, 190, 430, 420], fill=(150, 146, 138))
    d.polygon([(250, 250), (360, 260), (330, 350), (240, 330)], fill=(60, 55, 50))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def vl_latency(name: str, port: int, runs: int = 3) -> dict:
    """One image + one short structured answer: the per-building grading call."""
    b64 = _test_jpeg()
    body = {
        "model": name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":
                        "Grade this structure 0-3 for storm damage and caption it in one sentence. "
                        "Reply as JSON: {\"class\": int, \"caption\": str}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 96,
        "temperature": 0.0,
    }
    lats, outs, sample = [], [], ""
    for _ in range(runs):
        try:
            d, dt = chat(port, body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:160]}
        lats.append(dt * 1000.0)
        outs.append(int(d.get("usage", {}).get("completion_tokens") or 0))
        sample = d["choices"][0]["message"]["content"][:160]
    return {
        "model_root": root_of(port),
        "per_call_ms_median": int(statistics.median(lats)),
        "per_call_ms_min": int(min(lats)),
        "output_tokens_median": int(statistics.median(outs)),
        "runs": len(lats),
        "sample": sample,
    }


def main() -> int:
    out: dict = {"measured_at": time.time(), "host": "dgx-spark"}
    for name, port in SERVERS.items():
        try:
            root_of(port)
        except Exception as exc:  # noqa: BLE001 - an absent server is a result, not a crash
            out[name] = {"error": f"unreachable: {type(exc).__name__}"}
            continue
        if name == "captioner":
            out[name] = vl_latency(name, port)
        else:
            out[name] = text_throughput(name, port)
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
