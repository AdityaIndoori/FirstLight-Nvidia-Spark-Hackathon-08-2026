"""Fire all four containment beats, the battery, the overhead measurement and the
real runtime audit read, on the box. Prints what a judge would see.

Run on the Spark:  ./.venv/bin/python scripts/run_beats.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import containment, db  # noqa: E402

db.init()

print("=" * 78)
print("RUNTIME PROBE")
print("=" * 78)
print("runtime_present:", containment.runtime_present())
print("sandbox        :", containment.SANDBOX)
print("policy source  :", containment.POLICY_PATH)
print("policy         :", containment.policy().description[:120])
containment.policy().assert_deny_first()
print("deny-first ordering: OK")

print()
print("=" * 78)
print("THE FOUR BEATS, real audit records")
print("=" * 78)
for name, rows in containment.all_beats("https://judge-laptop.example.net/collect").items():
    print("\n--- %s ---" % name)
    for r in rows:
        print(
            "  actor=%s action=%s destination=%s verdict=%s class=%s rule=%s"
            % (
                r["actor"],
                r["action"],
                r["destination"],
                r["verdict"],
                r["verdict_class"],
                r["rule"],
            )
        )

print()
print("=" * 78)
print("INJECTION BATTERY, B8(g)")
print("=" * 78)
battery = containment.injection_battery()
for k in (
    "captions",
    "altered_grades",
    "altered_fema_fields",
    "egress_attempts",
    "egress_denied",
    "egress_allowed",
    "buildings_checked",
    "passed",
):
    print("  %-22s %s" % (k, battery[k]))

print()
print("=" * 78)
print("MEASURED ENFORCEMENT OVERHEAD")
print("=" * 78)
print(json.dumps(containment.overhead_ms(samples=25), indent=2))

print()
print("=" * 78)
print("RUNTIME'S OWN OCSF AUDIT STREAM, newest first")
print("=" * 78)
for r in containment.runtime_audit(limit=8):
    print(
        "  [%s] %s %s -> %s  class=%s rule=%s src=%s"
        % (
            r["ts"],
            r["verdict"].upper(),
            r["action"],
            r["destination"] or "(none)",
            r["verdict_class"],
            r["rule"],
            r["source"],
        )
    )

print()
print("=" * 78)
print("STATUS PAYLOAD openshell KEY")
print("=" * 78)
s = containment.status()
print("policy      :", s["policy"][:110])
print("allows      :", s["allows"])
print("denials     :", s["denials"])
print("overhead_ms :", s["overhead_ms"])
print("runtime     :", s["runtime"])
print("note        :", s["note"])
print("audit rows  :", len(s["audit"]), "sources:", sorted({a["source"] for a in s["audit"]}))

print()
print("=" * 78)
print("RUNTIME PRESET, our rules in the runtime's own schema")
print("=" * 78)
print(json.dumps(containment.runtime_preset(), indent=2)[:1400])
