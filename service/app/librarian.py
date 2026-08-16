"""The data librarian: the only component in FIRST LIGHT that touches the network.

WHY the API takes a NAME and never a URL: the agent's context is treated as fully
attacker-controlled (any tile caption, EXIF field or filename may be hostile). An
injected instruction can only reach the network through a primitive that accepts a
destination, so this module does not provide one. `refresh(name)` looks the URL up
in a module-level allowlist, an unknown name raises, and there is no function
anywhere in the codebase that takes a URL. OpenShell then denies off-allowlist
destinations out of process, so the two controls are independent: even if this
file were rewritten, the sandbox still refuses.

Everything downstream reads the local store, so a refresh that fails, or a box
with no connectivity at all, degrades to "last_refreshed is old" and nothing else.
Requests are GET only, size capped, checksummed, written to a temp file and
atomically swapped, so a truncated transfer can never replace a good local copy.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlsplit

from . import config, db

# PUBLIC API
# ALLOWLIST: dict[str, dict]                 name -> {url, kind, note, filename, sha256}
# refresh(name: str, *, actor: str = "agent") -> dict
#     {ok, name, source, filename, bytes, sha256, last_refreshed, changed, error}
#     raises ValueError on an unknown name; never accepts a URL
# catalog() -> list[dict]                    every allowlisted name plus last_refreshed
# agent_tool_schema() -> dict                one tool, one enum parameter
# names() -> tuple[str, ...]
# local_path(name: str) -> Path

FETCH_TIMEOUT_S = float(os.environ.get("FIRSTLIGHT_FETCH_TIMEOUT", "20"))
# A hostile or misconfigured source must not be able to fill the box's disk.
MAX_BYTES = int(os.environ.get("FIRSTLIGHT_FETCH_MAX_BYTES", str(256 * 1024 * 1024)))
USER_AGENT = "FIRSTLIGHT-librarian/1.0 (offline county EOC; GET only)"

# The five approved sources, keyed by name. `sha256` pins a checksum when the
# source publishes a stable one; None means "record what arrived and compare
# against the previous refresh", which is still enough to report changed vs
# unchanged without pretending we verified a digest nobody gave us.
ALLOWLIST: dict[str, dict] = {
    "noaa_storm_imagery": {
        "url": "https://storms.ngs.noaa.gov/",
        "kind": "post-event-imagery-index",
        "filename": "noaa_storm_imagery.index.html",
        "sha256": None,
        "note": "NOAA emergency response imagery index, the post-event tile source.",
    },
    "xview2_labels": {
        "url": "https://xview2.org/",
        "kind": "damage-labels-index",
        "filename": "xview2_labels.index.html",
        "sha256": None,
        "note": "xView2 challenge labels, the damage-grade reference set.",
    },
    "ms_building_footprints": {
        "url": "https://github.com/microsoft/GlobalMLBuildingFootprints",
        "kind": "footprints-index",
        "filename": "ms_building_footprints.index.html",
        "sha256": None,
        "note": "Rural fallback footprints, geometry only, for counties with no GIS department.",
    },
    "cms_facilities": {
        "url": "https://data.cms.gov/provider-data/topics/nursing-homes",
        "kind": "care-facilities-index",
        "filename": "cms_facilities.index.html",
        "sha256": None,
        "note": "CMS Care Compare, facility level only, feeds facility_near.",
    },
    "cdc_svi": {
        "url": "https://www.atsdr.cdc.gov/place-health/php/svi/index.html",
        "kind": "svi-index",
        "filename": "cdc_svi.index.html",
        "sha256": None,
        "note": "CDC Social Vulnerability Index block groups, feeds vulnerable_density.",
    },
}


def names() -> tuple[str, ...]:
    return tuple(ALLOWLIST)


def _entry(name: str) -> dict:
    """Resolve a name to its allowlist entry. The single point where a URL is
    ever produced, and it is produced from a name, never from a caller."""
    key = (name or "").strip()
    if key not in ALLOWLIST:
        raise ValueError(
            f"dataset not on the allowlist: {key!r}. Allowed names: {', '.join(ALLOWLIST)}"
        )
    return ALLOWLIST[key]


def local_path(name: str) -> "os.PathLike[str]":
    entry = _entry(name)
    return config.DATASET_DIR / str(entry["filename"])


class _HostLockedRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects only while they stay on the allowlisted host.

    A 302 to somewhere else is exactly how an approved source turns into an
    unapproved one, so the host is checked at every hop rather than only on the
    first request.
    """

    def __init__(self, host: str) -> None:
        self.host = host

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        if urlsplit(newurl).hostname != self.host:
            raise urllib.error.URLError(f"redirect off the allowlisted host: {self.host}")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.method = "GET"  # never let a redirect change the verb
        return new


def _fetch(name: str) -> tuple[bytes, str]:
    """GET the named source. Takes a NAME on purpose: see the module docstring."""
    entry = _entry(name)
    url = str(entry["url"])
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise urllib.error.URLError(f"allowlist entry for {name} is not https")

    req = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_HostLockedRedirect(parts.hostname))
    with opener.open(req, timeout=FETCH_TIMEOUT_S) as resp:
        # Read one byte past the cap so an oversized body is detected, not
        # silently truncated into a file we would then checksum as good.
        blob = resp.read(MAX_BYTES + 1)
        ctype = resp.headers.get("Content-Type", "") or ""
    if len(blob) > MAX_BYTES:
        raise urllib.error.URLError(f"{name} exceeds the {MAX_BYTES} byte cap")
    if not blob:
        raise urllib.error.URLError(f"{name} returned an empty body")
    return blob, ctype


def refresh(name: str, *, actor: str = "agent") -> dict:
    """Fetch one allowlisted dataset by NAME and atomically swap it into place.

    Unknown names raise ValueError. Network failures are values, not exceptions,
    because the box is expected to run with no connectivity and a failed refresh
    must not take a panel down.
    """
    entry = _entry(name)
    dest = config.DATASET_DIR / str(entry["filename"])
    prior = db.q1("SELECT sha256, last_refreshed FROM datasets WHERE name=?", (name,))
    prior_sha = prior["sha256"] if prior else None

    t0 = time.perf_counter()
    try:
        blob, ctype = _fetch(name)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:200]
        db.log(actor, "dataset-refresh-failed", {"name": name, "error": err})
        return {
            "ok": False,
            "name": name,
            "source": entry["url"],
            "filename": dest.name,
            "bytes": 0,
            "sha256": prior_sha,
            "last_refreshed": (prior["last_refreshed"] if prior else None),
            "changed": False,
            "error": err,
        }

    digest = hashlib.sha256(blob).hexdigest()
    pinned = entry.get("sha256")
    if pinned and digest != pinned:
        # Refuse the swap: the local copy on disk is still the trusted one.
        db.log(
            actor,
            "dataset-checksum-mismatch",
            {"name": name, "expected": pinned, "got": digest, "bytes": len(blob)},
        )
        return {
            "ok": False,
            "name": name,
            "source": entry["url"],
            "filename": dest.name,
            "bytes": len(blob),
            "sha256": digest,
            "last_refreshed": (prior["last_refreshed"] if prior else None),
            "changed": False,
            "error": f"sha256 mismatch: expected {pinned}, got {digest}",
        }

    try:
        _atomic_write(dest, blob)
    except OSError as exc:
        err = f"{type(exc).__name__}: {exc}"[:200]
        db.log(actor, "dataset-refresh-failed", {"name": name, "error": err})
        return {
            "ok": False,
            "name": name,
            "source": entry["url"],
            "filename": dest.name,
            "bytes": len(blob),
            "sha256": digest,
            "last_refreshed": (prior["last_refreshed"] if prior else None),
            "changed": False,
            "error": err,
        }

    now = time.time()
    db.run(
        "INSERT OR REPLACE INTO datasets (name, source, last_refreshed, sha256, bytes, note) "
        "VALUES (?,?,?,?,?,?)",
        (name, entry["url"], now, digest, len(blob), entry.get("note", "")),
    )
    _invalidate_caches()
    took = int((time.perf_counter() - t0) * 1000)
    changed = digest != prior_sha
    db.log(
        actor,
        "dataset-refresh",
        {
            "name": name,
            "source": entry["url"],
            "sha256": digest,
            "bytes": len(blob),
            "changed": changed,
            "content_type": ctype[:60],
            "took_ms": took,
        },
    )
    return {
        "ok": True,
        "name": name,
        "source": entry["url"],
        "filename": dest.name,
        "bytes": len(blob),
        "sha256": digest,
        "last_refreshed": now,
        "changed": changed,
        "took_ms": took,
        "error": None,
    }


def _atomic_write(dest, blob: bytes) -> None:
    """Temp file in the destination directory, fsync, then os.replace.

    Same filesystem, so the replace is atomic: a reader either sees the whole old
    file or the whole new one, never a half-written dataset.
    """
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(config.DATASET_DIR), prefix=".refresh-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _invalidate_caches() -> None:
    """A swapped file behind an lru_cache is a stale file. Best effort: a missing
    consumer must not turn a good refresh into a failure."""
    for module_name, fn_name in (("datasets", "reset_cache"), ("archive", "reset_geocode_cache")):
        try:
            mod = __import__(f"{__package__}.{module_name}", fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn()
        except Exception:
            pass


def catalog() -> list[dict]:
    """Every allowlisted name with its local state, for the HUD dataset strip.

    Names with no row yet report last_refreshed None and present False, which is
    the honest reading of "this box has never fetched it".
    """
    rows = {r["name"]: r for r in db.q("SELECT name, last_refreshed, sha256, bytes FROM datasets")}
    out: list[dict] = []
    for name, entry in ALLOWLIST.items():
        row = rows.get(name)
        path = config.DATASET_DIR / str(entry["filename"])
        out.append(
            {
                "name": name,
                "source": entry["url"],
                "kind": entry["kind"],
                "note": entry.get("note", ""),
                "filename": entry["filename"],
                "present": path.exists(),
                "last_refreshed": (float(row["last_refreshed"]) if row and row["last_refreshed"] else None),
                "sha256": (row["sha256"] if row else None),
                "bytes": (int(row["bytes"]) if row and row["bytes"] else 0),
            }
        )
    return out


def agent_tool_schema() -> dict:
    """The agent's entire network surface: one tool, one enum parameter.

    Shaped for a NemoClaw tool registration. There is no `url` field to fill in,
    which is the point: a hijacked agent has no fetch primitive to abuse.
    """
    return {
        "type": "function",
        "function": {
            "name": "refresh_dataset",
            "description": (
                "Refresh one approved local dataset by name. GET only, checksum verified, "
                "atomically swapped into the local store. Names are fixed; URLs cannot be supplied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": list(ALLOWLIST),
                        "description": "Which approved dataset to refresh.",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    }


__all__ = [
    "ALLOWLIST",
    "FETCH_TIMEOUT_S",
    "MAX_BYTES",
    "agent_tool_schema",
    "catalog",
    "local_path",
    "names",
    "refresh",
]
