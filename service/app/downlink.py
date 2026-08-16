"""A1: the drone downlink. RTSP is the frozen choice, simulate is the safety net.

CRITICAL invariant: the downlink writes frames into config.WATCH_DIR and stops.
It never calls ingest.analyze_tile itself. Two writers on one folder race the
poller, and the loser reads a half-written JPEG and reports a bogus grading
error on stage 1 - in front of judges, on the one number the See track is scored
on. One producer, one consumer, one path in.

Frames land under a temporary name and are renamed into place, so the watcher's
settle check never sees a partial file at all.
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config, db

# PUBLIC API
# ---------------------------------------------------------------------------
# start(source: str) -> None    "simulate:<dir>" or an rtsp:// URL. Idempotent
#                               only in the sense that starting twice raises
#                               RuntimeError instead of spawning a second thread.
# stop() -> None                Signals the thread and joins briefly. Safe to
#                               call when nothing is running.
# state() -> dict               {source, running, frames_received, frames_ingested,
#                                latency_ms_p50, last_error, started_at, mode}
# ---------------------------------------------------------------------------

SIM_PREFIX = "simulate:"
SIM_INTERVAL_S = 1.5
SIM_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_JOIN_S = 5.0


@dataclass
class _State:
    source: str = ""
    mode: str = ""
    running: bool = False
    frames_received: int = 0
    frames_ingested: int = 0
    last_error: Optional[str] = None
    started_at: Optional[float] = None
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)


_lock = threading.Lock()
_s = _State()


def _bump(received: int = 0, error: Optional[str] = None) -> None:
    with _lock:
        _s.frames_received += received
        if error is not None:
            _s.last_error = error


def _deliver(data: bytes, name: str) -> bool:
    """Atomic-rename a frame into the watch folder.

    The temp name carries no image suffix, so ingest._candidates ignores it until
    the rename completes. That is the whole reason the poller never sees a torn
    frame.
    """
    try:
        config.WATCH_DIR.mkdir(parents=True, exist_ok=True)
        tmp = config.WATCH_DIR / f".{name}.part"
        tmp.write_bytes(data)
        tmp.replace(config.WATCH_DIR / name)
        return True
    except OSError as exc:
        _bump(error=f"write failed: {exc}")
        return False


def _deliver_file(src: Path, name: str) -> bool:
    try:
        config.WATCH_DIR.mkdir(parents=True, exist_ok=True)
        tmp = config.WATCH_DIR / f".{name}.part"
        shutil.copy2(src, tmp)
        tmp.replace(config.WATCH_DIR / name)
        return True
    except OSError as exc:
        _bump(error=f"copy failed: {exc}")
        return False


# ------------------------------------------------------------------- simulate
def _simulate(directory: Path, stop: threading.Event) -> None:
    """Replay a sample directory at downlink tempo, looping until stopped.

    Looping matters for the demo: the judge pool is a couple of dozen tiles and a
    three-minute beat outlasts one pass. Each lap gets a fresh sequence number so
    the watcher treats it as a new file, while the content-hash dedup in ingest
    keeps the second lap from double-counting buildings.
    """
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SIM_SUFFIXES)
    if not files:
        _bump(error=f"no sample images in {directory}")
        return
    seq = 0
    while not stop.is_set():
        for src in files:
            if stop.is_set():
                return
            seq += 1
            name = f"dl_{int(time.time())}_{seq:05d}{src.suffix.lower()}"
            for sidecar in (src.with_suffix(".bounds.json"), Path(str(src) + ".bounds.json")):
                if sidecar.is_file():
                    _deliver_file(
                        sidecar, f"{Path(name).stem}.bounds.json"
                    )
                    break
            if _deliver_file(src, name):
                _bump(received=1)
            stop.wait(SIM_INTERVAL_S)


# ----------------------------------------------------------------------- rtsp
def _rtsp(url: str, stop: threading.Event) -> None:
    """Decode keyframes to JPEG with PyAV. Keyframes only, deliberately.

    A survey pass at 30 fps is 1800 near-identical frames a minute, and grading
    every one buys nothing but a queue an operator watches grow. Keyframes give
    scene changes, which is what a tile is.
    """
    try:
        import av  # noqa: PLC0415 - optional, and its absence is a labelled state
    except Exception:  # noqa: BLE001
        _bump(error="PyAV not installed, RTSP unavailable")
        return
    try:
        container = av.open(
            url,
            timeout=10.0,
            options={"rtsp_transport": "tcp", "stimeout": "5000000"},
        )
    except Exception as exc:  # noqa: BLE001 - unreachable camera is a state, not a crash
        _bump(error=f"rtsp open failed: {type(exc).__name__}: {exc}"[:200])
        return
    seq = 0
    try:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            if stop.is_set():
                return
            seq += 1
            try:
                buf = frame.to_image()
                import io  # noqa: PLC0415 - only the RTSP path pays for this

                raw = io.BytesIO()
                buf.save(raw, format="JPEG", quality=92)
                data = raw.getvalue()
            except Exception as exc:  # noqa: BLE001
                _bump(error=f"frame encode failed: {type(exc).__name__}")
                continue
            if _deliver(data, f"rtsp_{int(time.time())}_{seq:05d}.jpg"):
                _bump(received=1)
    except Exception as exc:  # noqa: BLE001 - a dropped stream ends the run cleanly
        _bump(error=f"rtsp stream ended: {type(exc).__name__}: {exc}"[:200])
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------- control
def _run(source: str, stop: threading.Event) -> None:
    try:
        if source.startswith(SIM_PREFIX):
            _simulate(Path(source[len(SIM_PREFIX) :]).expanduser(), stop)
        else:
            _rtsp(source, stop)
    finally:
        with _lock:
            _s.running = False


def start(source: str) -> None:
    """Begin producing frames into the watch folder.

    `source` is either "simulate:<dir>" or an rtsp:// URL. Anything else is
    rejected here rather than handed to a decoder, because the source string
    reaches this function from an API body.
    """
    src = (source or "").strip()
    if not src:
        raise ValueError("source is required")
    if src.startswith(SIM_PREFIX):
        directory = Path(src[len(SIM_PREFIX) :]).expanduser()
        if not directory.is_dir():
            raise ValueError(f"simulate directory not found: {directory}")
        mode = "simulate"
    elif src.startswith("rtsp://") or src.startswith("rtsps://"):
        mode = "rtsp"
    else:
        raise ValueError("source must be 'simulate:<dir>' or an rtsp:// URL")

    with _lock:
        if _s.running:
            raise RuntimeError("downlink already running")
        _s.stop_event = threading.Event()
        _s.source = src
        _s.mode = mode
        _s.running = True
        _s.last_error = None
        _s.started_at = time.time()
        _s.frames_received = 0
        _s.frames_ingested = 0
        stop_event = _s.stop_event
        thread = threading.Thread(
            target=_run, args=(src, stop_event), name="firstlight-downlink", daemon=True
        )
        _s.thread = thread
    thread.start()
    db.log("downlink", "downlink-start", {"mode": mode})


def stop() -> None:
    with _lock:
        thread, stop_event, was = _s.thread, _s.stop_event, _s.running
        received = _s.frames_received
        _s.running = False
    stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=_JOIN_S)
    with _lock:
        _s.thread = None
    if was:
        db.log("downlink", "downlink-stop", {"frames_received": received})


def state() -> dict:
    """What the HUD reads.

    `frames_ingested` is counted from the tiles table, not from a counter in
    here: the downlink hands off to the watcher and must not claim credit for
    work the watcher has not finished. `backlog` is MEASURED - files sitting in
    the watch folder plus tiles mid-pipeline - not inferred from
    received-minus-ingested, which drifts upward forever once the dedup starts
    absorbing a looped replay and would show a queue that does not exist.
    """
    with _lock:
        source, mode, started_at = _s.source, _s.mode, _s.started_at
        received, last_error = _s.frames_received, _s.last_error
        thread = _s.thread
        running = _s.running and (thread is None or thread.is_alive())
        _s.running = running
    ingested, p50, backlog = _ingest_view(started_at)
    with _lock:
        _s.frames_ingested = ingested
    return {
        "source": source,
        "mode": mode,
        "running": running,
        "frames_received": received,
        "frames_ingested": ingested,
        "latency_ms_p50": p50,
        "last_error": last_error,
        "started_at": started_at,
        "backlog": backlog,
    }


def _ingest_view(since: Optional[float]) -> tuple[int, int, int]:
    """(tiles analyzed since this run started, p50 ms, measured backlog).

    Imported lazily so a partially built tree still starts, and scoped to the run
    window so an earlier card dump does not inflate the downlink's own numbers.
    """
    try:
        from . import ingest

        p50 = ingest.latency_p50()  # also guarantees the schema exists
        waiting = sum(
            1
            for p in config.WATCH_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in ingest.IMAGE_SUFFIXES
        )
        backlog = waiting + ingest.in_flight()
        if since is None:
            return 0, int(p50), backlog
        row = db.q1("SELECT COUNT(*) AS n FROM tiles WHERE analyzed_at >= ?", (since,))
        return int((row["n"] if row else 0) or 0), int(p50), backlog
    except Exception:  # noqa: BLE001 - the HUD gets zeros, never a 500
        return 0, 0, 0


__all__ = ["SIM_INTERVAL_S", "SIM_PREFIX", "start", "state", "stop"]
