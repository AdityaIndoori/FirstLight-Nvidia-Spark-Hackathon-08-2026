#!/usr/bin/env python3
"""Reset the box to a clean slate: no imagery, no gradings, no operator edits.

WHY a script and not a few rm commands: "clean" has to mean the same thing every
time, and there are six places state hides. Miss one and the console opens with a
stale rank list or a p50 from imagery that is no longer on disk - which is exactly
the box-contradicts-itself problem the measurement work was about.

What it clears:
  - tiles, buildings, archive, plan_overrides, availability, road_blocks
  - the analyzed / withheld / watch / thumbs directories
  - the decision log, optionally

What it KEEPS:
  - datasets (footprints, roads, facilities, SVI) - reference data, not results
  - basemap tiles and glyphs
  - model weights
  - the decision log, ALWAYS

The log cannot be cleared, and this tool deliberately offers no flag to try. It is
append-only enforced by SQL triggers, not by convention, and that is a claim the
pitch makes to judges: an audit trail a maintenance script can erase is not an
audit trail. Resetting appends a `reset-to-clean` entry instead, so the wipe itself
is on the record.

To demo against an empty log, use a fresh FIRSTLIGHT_DATA directory - a different
database, rather than a mutilated one.

Dry run by default. --apply to act.
"""
from __future__ import annotations


import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, db  # noqa: E402

# Result tables. Order matters only for readability; none of these have FKs to
# each other that would make a plain DELETE fail.
RESULT_TABLES = (
    "tiles",
    "buildings",
    "archive",
    "plan_overrides",
    "road_blocks",
    "availability",
)

# Directories holding produced artifacts. `watch` is included because a file left
# there would be picked up by the poller seconds after the reset.
RESULT_DIRS = ("ANALYZED_DIR", "WITHHELD_DIR", "WATCH_DIR", "THUMB_DIR")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete")

    args = ap.parse_args()

    db.init()
    print("current state")
    counts = {}
    for table in RESULT_TABLES:
        try:
            row = db.q1(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = int(row["n"] if row else 0)
        except Exception as exc:  # noqa: BLE001 - a missing table is already clean
            counts[table] = f"absent ({type(exc).__name__})"
        print(f"  {table:16} {counts[table]}")

    files = {}
    for name in RESULT_DIRS:
        d = getattr(config, name, None)
        n = len(list(d.iterdir())) if d and d.is_dir() else 0
        files[name] = (d, n)
        print(f"  {name:16} {n} files")

    log_row = db.q1("SELECT COUNT(*) AS n FROM decision_log")
    log_n = int(log_row["n"] if log_row else 0)
    print(f"  decision_log     {log_n}  (kept: append-only)")

    if not args.apply:
        print("\ndry run: pass --apply to reset")
        return 0

    for table in RESULT_TABLES:
        try:
            db.run(f"DELETE FROM {table}")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not clear {table}: {type(exc).__name__}")

    for name, (d, _n) in files.items():
        if not d or not d.is_dir():
            continue
        for child in d.iterdir():
            try:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
            except OSError as exc:
                print(f"  could not remove {child.name}: {exc}")

    # Reclaim the pages so the file on disk reflects the reset too: a 272 KB db
    # after a wipe invites the question of what is still in it.
    try:
        db.conn().execute("VACUUM")
    except Exception:  # noqa: BLE001 - a locked db still reset correctly
        pass

    db.log("maintenance", "reset-to-clean", {"tables": list(RESULT_TABLES)})

    print("\nclean. Verify with:  curl -s localhost:8081/api/status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
