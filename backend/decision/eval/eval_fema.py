"""B8 Part B: FEMA field accuracy.

Previously deferred, and the reason given was precise: no FEMA row builder
existed in `backend/`. One exists in the service tree (`app/exports.py`
fema_pda_csv), so this measures it instead of explaining its absence.

WHAT IS BEING MEASURED, precisely, because a FEMA number that overstates itself
is worse than a deferral: the row builder is a DETERMINISTIC projection of the
buildings table, not a model output. So this metric checks field-level
correctness of that projection against a labelled fixture: every damaged
structure appears exactly once, coordinates round-trip, the damage category
matches the integer class, provenance names who graded it, and no owner-identity
column reaches the worksheet. It does NOT claim to measure a model's accuracy at
filling FEMA fields, because no model fills them.
"""
from __future__ import annotations

import csv
import importlib
import io
import sys
from pathlib import Path

from backend.decision.eval.report import (
    STATUS_FAIL,
    STATUS_PASS,
    deferred_metric,
    make_metric,
)

_REASON_MISSING = (
    "no FEMA PDA row builder is importable from this tree (expected "
    "app.exports.fema_pda_csv in the service tree)"
)

# One row per labelled structure, with the worksheet values each must produce.
FIXTURE = [
    {
        "footprint_id": "fp-1",
        "label": "7665 Sun Island Dr S",
        "centroid": [-82.7412, 27.7788],
        "damage_class": 3,
        "expect_category": "destroyed",
        "graded_by": "nemotron-vl",
        "confirmed": 0,
    },
    {
        "footprint_id": "fp-2",
        "label": "1200 Pasadena Ave S",
        "centroid": [-82.7355, 27.7701],
        "damage_class": 2,
        "expect_category": "major damage",
        "graded_by": "operator:R. Alvarez",
        "confirmed": 1,
    },
    {
        "footprint_id": "fp-3",
        "label": "455 62nd Ave",
        "centroid": [-82.7290, 27.7654],
        "damage_class": 1,
        "expect_category": "minor damage",
        "graded_by": "stub-pixelstat-v1",
        "confirmed": 0,
    },
    {
        # Class 0 must NOT appear: the worksheet is damaged structures only.
        "footprint_id": "fp-4",
        "label": "900 Gulfport Blvd",
        "centroid": [-82.7201, 27.7512],
        "damage_class": 0,
        "expect_category": None,
        "graded_by": "nemotron-vl",
        "confirmed": 0,
    },
]

# A column carrying any of these in the worksheet is a privacy failure, not a
# field-accuracy quibble, so it fails the whole metric.
FORBIDDEN_SUBSTRINGS = ("owner", "mailto", "mailing", "taxpay", "grantee")


def _import_exports():
    for name in ("service.app.exports", "app.exports"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    service = Path(__file__).resolve().parents[3] / "service"
    if service.is_dir():
        sys.path.insert(0, str(service))
        try:
            return importlib.import_module("app.exports")
        except Exception:
            return None
    return None


def _seed(db, fixture) -> None:
    import json

    db.init()
    db.run("DELETE FROM buildings")
    for row in fixture:
        db.run(
            """INSERT INTO buildings
                 (footprint_id, label, centroid_json, damage_class, confidence,
                  graded_by, confirmed, doubt)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                row["footprint_id"],
                row["label"],
                json.dumps(row["centroid"]),
                row["damage_class"],
                0.72,
                row["graded_by"],
                row["confirmed"],
                0.25,
            ),
        )


def evaluate_fema_field_accuracy() -> dict:
    exports = _import_exports()
    if exports is None:
        return deferred_metric("fema_field_accuracy", _REASON_MISSING)
    pkg = exports.__name__.rsplit(".", 1)[0]
    try:
        db = importlib.import_module(f"{pkg}.db")
        config = importlib.import_module(f"{pkg}.config")
    except Exception as exc:
        return deferred_metric(
            "fema_field_accuracy", f"the service db/config modules are not importable: {exc}"
        )

    import tempfile

    # ignore_cleanup_errors: on Windows the sqlite handle must be closed before
    # the directory can be removed, and a cleanup race must not fail the metric.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        original = config.DB_PATH
        config.DB_PATH = Path(tmp) / "eval.db"
        db._local = type(db._local)()  # a fresh thread-local so the new path is used
        try:
            _seed(db, FIXTURE)
            text = exports.fema_pda_csv()
        except Exception as exc:
            return make_metric(
                "fema_field_accuracy",
                STATUS_FAIL,
                details={"error": f"the row builder raised: {exc}"},
            )
        finally:
            existing = getattr(db._local, "conn", None)
            if existing is not None:
                try:
                    existing.close()
                except Exception:
                    pass
            config.DB_PATH = original
            db._local = type(db._local)()

    # Skip the DRAFT stamp block: the header row is the one with our field names.
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("structure_id")), None)
    if start is None:
        return make_metric(
            "fema_field_accuracy",
            STATUS_FAIL,
            details={"error": "no structure_id header row in the worksheet"},
        )

    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    by_id = {r["structure_id"]: r for r in rows}
    expected = [f for f in FIXTURE if f["damage_class"] >= 1]

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check(
        "one row per damaged structure",
        len(rows) == len(expected),
        f"{len(rows)} rows for {len(expected)} damaged structures",
    )
    check(
        "undamaged structures excluded",
        "fp-4" not in by_id,
        "fp-4 is class 0 and must not appear on a damage worksheet",
    )
    header = ",".join(rows[0].keys()).lower() if rows else ""
    leaked = [s for s in FORBIDDEN_SUBSTRINGS if s in header]
    check("no owner-identity column", not leaked, f"leaked: {leaked}" if leaked else "")

    for want in expected:
        got = by_id.get(want["footprint_id"])
        if got is None:
            check(f"{want['footprint_id']} present", False, "missing from the worksheet")
            continue
        check(f"{want['footprint_id']} address", got.get("address") == want["label"])
        check(
            f"{want['footprint_id']} category",
            got.get("damage_category") == want["expect_category"],
            f"got {got.get('damage_category')!r}, want {want['expect_category']!r}",
        )
        check(
            f"{want['footprint_id']} class",
            str(got.get("damage_class")) == str(want["damage_class"]),
        )
        check(
            f"{want['footprint_id']} longitude",
            abs(float(got.get("longitude", 0)) - want["centroid"][0]) < 1e-5,
        )
        check(
            f"{want['footprint_id']} latitude",
            abs(float(got.get("latitude", 0)) - want["centroid"][1]) < 1e-5,
        )
        check(
            f"{want['footprint_id']} provenance",
            got.get("graded_by") == want["graded_by"],
            f"got {got.get('graded_by')!r}",
        )
        check(
            f"{want['footprint_id']} operator confirmation",
            got.get("operator_confirmed") == ("yes" if want["confirmed"] else "no"),
        )

    failed = [c for c in checks if not c["ok"]]
    accuracy = round(1.0 - len(failed) / len(checks), 4) if checks else 0.0
    return make_metric(
        "fema_field_accuracy",
        STATUS_PASS if not failed else STATUS_FAIL,
        value=accuracy,
        threshold="every field correct on every damaged structure, 0 owner columns",
        sample_count=len(checks),
        details={
            "rows_emitted": len(rows),
            "damaged_structures_expected": len(expected),
            "failed_checks": failed,
            "note": (
                "the worksheet is a deterministic projection of the buildings table, "
                "not a model output, so this measures the projection's field "
                "correctness and its privacy scrub, not a model's FEMA accuracy"
            ),
        },
    )
