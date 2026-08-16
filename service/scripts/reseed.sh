#!/usr/bin/env bash
# One command back to a pristine demo state. You will run this fifty times.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${FIRSTLIGHT_DATA:-$PWD/data}"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

echo "reseed: clearing state under $DATA"
rm -f  "$DATA/firstlight.db" "$DATA/firstlight.db-wal" "$DATA/firstlight.db-shm"
rm -rf "$DATA/watch" "$DATA/analyzed" "$DATA/withheld" "$DATA/thumbs"
mkdir -p "$DATA/watch" "$DATA/analyzed" "$DATA/withheld" "$DATA/thumbs"

if [ ! -f "$DATA/datasets/footprints.geojson" ]; then
  echo "reseed: no datasets, writing labelled fakes"
  "$PY" scripts/make_demo_kit.py --datasets --tiles 0 >/dev/null
fi

if [ ! -f "$DATA/sample_tiles/fixture_person.jpg" ]; then
  echo "reseed: building the demo kit"
  "$PY" scripts/make_demo_kit.py --tiles 10 >/dev/null
fi

echo "reseed: seeding operator-entered availability"
"$PY" scripts/make_demo_kit.py --tiles 0 --seed-availability >/dev/null

echo "reseed: staging tiles into the watch folder"
cp "$DATA"/sample_tiles/*.jpg "$DATA"/sample_tiles/*.bounds.json "$DATA/watch/" 2>/dev/null || true

echo
echo "reseed done. Start the box with ./run.sh"
echo "The watch folder holds the judge pool, including fixture_person.jpg:"
echo "  it MUST be analyzed and its buildings MUST appear in the rank,"
echo "  and it MUST NOT appear in the archive, search, or any thumbnail."
