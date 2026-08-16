"""FEMA and ICS export pack. One click, every document a real EOC needs.

Every document is stamped DRAFT with a signature line, because a machine does
not file federal paperwork.
"""
from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from typing import Optional

from . import contracts, db, scorer

DRAFT_HEADER = "DRAFT - requires approval by the Planning Section Chief"


def _stamp(title: str) -> list[str]:
    return [
        DRAFT_HEADER,
        f"{title}",
        f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} by FIRST LIGHT, offline",
        "prepared by: ______________________  signature: ______________________",
        "",
    ]


def fema_pda_csv() -> str:
    """One row per damaged structure, with who graded it and how sure."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for line in _stamp("FEMA Preliminary Damage Assessment worksheet"):
        w.writerow([line])
    w.writerow(
        [
            "structure_id",
            "address",
            "longitude",
            "latitude",
            "damage_category",
            "damage_class",
            "confidence",
            "ai_uncertainty",
            "graded_by",
            "operator_confirmed",
            "care_facility_within_300m",
        ]
    )
    rows = db.q(
        """SELECT footprint_id, label, centroid_json, damage_class, confidence,
                  graded_by, confirmed, doubt, facility_json
             FROM buildings
            WHERE damage_class IS NOT NULL AND damage_class >= 1
            ORDER BY damage_class DESC"""
    )
    for r in rows:
        c = db.jload(r["centroid_json"], [0.0, 0.0])
        fac = db.jload(r["facility_json"])
        w.writerow(
            [
                r["footprint_id"],
                r["label"] or "",
                f"{c[0]:.6f}",
                f"{c[1]:.6f}",
                contracts.CLASS_LABEL.get(int(r["damage_class"] or 0), ""),
                int(r["damage_class"] or 0),
                round(float(r["confidence"] or 0), 3),
                round(float(r["doubt"] or 0), 3),
                r["graded_by"] or "",
                "yes" if r["confirmed"] else "no",
                (fac or {}).get("name", ""),
            ]
        )
    return out.getvalue()


def ics_213(message: str = "", operator: str = "") -> str:
    plan = scorer.build_plan()
    severe = db.q1(
        "SELECT COUNT(*) AS n FROM buildings WHERE damage_class >= ?",
        (contracts.SEVERE_FROM,),
    )
    body = message or (
        f"First-morning triage summary. {int(severe['n']) if severe else 0} structures "
        f"graded major or destroyed. Agency asks: "
        + ", ".join(
            f"{a['agency']} {a['units_required']}" for a in plan["agencies"] if a["units_required"]
        )
        + ". Next drone survey tasked to the sector with the least recent look."
    )
    lines = _stamp("ICS-213 General Message")
    lines += [
        "1. INCIDENT NAME: FIRST LIGHT county response",
        "2. TO: County EOC",
        f"3. FROM: {operator or 'Operations Section Chief'}",
        "4. SUBJECT: First-morning triage summary",
        f"5. DATE/TIME: {time.strftime('%Y-%m-%d %H:%M')}",
        "6. MESSAGE:",
        f"   {body}",
        "",
        "7. SIGNATURE: ______________________  POSITION: ______________________",
    ]
    return "\n".join(lines) + "\n"


def ics_209() -> str:
    plan = scorer.build_plan()
    tally = {}
    for r in db.q("SELECT damage_class, COUNT(*) AS n FROM buildings GROUP BY damage_class"):
        tally[contracts.CLASS_LABEL.get(int(r["damage_class"] or 0), "unknown")] = int(r["n"])
    lines = _stamp("ICS-209 Incident Status Summary")
    lines += ["DAMAGE TALLY:"]
    lines += [f"  {k}: {v}" for k, v in sorted(tally.items())]
    lines += ["", "AGENCY ASSIGNMENTS:"]
    for a in plan["agencies"]:
        if not a["steps"]:
            continue
        short = a["units_required"] - a["units_available"]
        flag = f"  SHORT BY {short}" if short > 0 else ""
        lines.append(
            f"  {a['agency'].upper()}: {len(a['steps'])} assignments, "
            f"{a['units_required']} units required, {a['units_available']} available{flag}"
        )
        for s in a["steps"]:
            lines.append(f"    {s['n']}. {s['label']} - {s['task']} ({s['units']} units)")
    lines += ["", "SIGNATURE: ______________________"]
    return "\n".join(lines) + "\n"


def ics_213_rr() -> list[tuple[str, str]]:
    """One resource request per agency whose ask exceeds what the operator entered.

    A disaster by definition overwhelms local resources, so this is the column a
    real EOC lives on.
    """
    out = []
    for a in scorer.build_plan()["agencies"]:
        short = a["units_required"] - a["units_available"]
        if short <= 0:
            continue
        lines = _stamp(f"ICS-213 RR Resource Request - {a['agency'].upper()}")
        lines += [
            f"1. REQUEST FOR: {a['agency'].replace('_', ' ')} units",
            f"2. QUANTITY: {short}",
            f"3. REQUIRED: {a['units_required']}   ON HAND (operator entered): {a['units_available']}",
            "4. NEEDED BY: current operational period",
            "5. JUSTIFICATION: assignments drafted from aerial triage exceed local resources",
            "",
            "REQUESTED BY: ______________________  APPROVED BY: ______________________",
        ]
        out.append((f"ICS-213-RR-{a['agency']}.txt", "\n".join(lines) + "\n"))
    return out


def decision_log_json() -> str:
    rows = db.q("SELECT id, ts, actor, action, payload FROM decision_log ORDER BY id")
    return json.dumps(
        [
            {
                "id": r["id"],
                "ts": r["ts"],
                "actor": r["actor"],
                "action": r["action"],
                "payload": db.jload(r["payload"], {}),
            }
            for r in rows
        ],
        indent=2,
    )


def aid_package(operator: Optional[str] = None) -> bytes:
    """One click: every document, zipped, offline."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("FEMA-PDA.csv", fema_pda_csv())
        z.writestr("ICS-213-general-message.txt", ics_213(operator=operator or ""))
        z.writestr("ICS-209-incident-summary.txt", ics_209())
        for name, text in ics_213_rr():
            z.writestr(name, text)
        z.writestr("decision-log.json", decision_log_json())
        z.writestr(
            "README.txt",
            "\n".join(
                _stamp("FIRST LIGHT aid package")
                + [
                    "Contents:",
                    "  FEMA-PDA.csv                  one row per damaged structure",
                    "  ICS-213-general-message.txt   general message to the EOC",
                    "  ICS-209-incident-summary.txt  agency assignments and unit counts",
                    "  ICS-213-RR-<agency>.txt       resource request per over-committed agency",
                    "  decision-log.json             append-only record of every decision",
                    "",
                    "Every grade in this package names who produced it. Operator overrides",
                    "are recorded as operator:<name>. Property value never entered the",
                    "life-safety ranking.",
                ]
            )
            + "\n",
        )
    db.log(f"operator:{operator or 'unknown'}", "export-aid-package", {"docs": 5})
    return buf.getvalue()
