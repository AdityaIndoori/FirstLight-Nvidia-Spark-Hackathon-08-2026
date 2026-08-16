#!/usr/bin/env bash
# FIRST LIGHT startup. Works with no connectivity: everything it needs is local.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "ERROR: .venv missing. Provision once, with network:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# Ultralytics pings home for update checks unless told it is offline.
export YOLO_OFFLINE=true
export FIRSTLIGHT_DATA="${FIRSTLIGHT_DATA:-$PWD/data}"

# Seed the demo datasets when the AOI has never been fetched.
if [ ! -f data/datasets/footprints.geojson ]; then
  echo "no datasets yet: run scripts/fetch_aoi.py once with network, or scripts/make_sample_data.py offline"
fi

exec .venv/bin/python -m uvicorn app.main:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}" --log-level warning
