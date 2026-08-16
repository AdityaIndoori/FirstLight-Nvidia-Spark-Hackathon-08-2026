#!/usr/bin/env bash
# Restart the FIRST LIGHT service on the box.
#
# WHY a script: the inline ssh form kept hanging because a backgrounded process
# holding the sshd session's stdout keeps the channel open, so the client waits
# forever for EOF on a server that started fine. setsid plus a full redirect of
# all three streams detaches it properly.
set -u
cd "$HOME/fl/service" || exit 1

pkill -f "uvicorn app.main" >/dev/null 2>&1
sleep 2

setsid env \
  YOLO_OFFLINE=true \
  FIRSTLIGHT_AOI=bay \
  FIRSTLIGHT_DATA="$PWD/data" \
  FIRSTLIGHT_REVIEW_TOKEN=demo-review-token \
  PYTHONPATH="$PWD" \
  "$PWD/.venv/bin/python" -m uvicorn app.main:app \
    --host 0.0.0.0 --port 8081 --log-level warning \
  >/tmp/fl.log 2>&1 </dev/null &
# Readiness is observed, not assumed. The per-probe timeout is generous because the
# first /api/status can load a model singleton: a 2 s cap here once reported "NOT
# READY" for a server that was healthy and answering in 13 s.
for _ in $(seq 1 30); do
  if curl -fsS -m 20 http://127.0.0.1:8081/api/status >/dev/null 2>&1; then
    echo "ready"
    exit 0
  fi
  sleep 1
done
echo "NOT READY - last log lines:"
tail -20 /tmp/fl.log
exit 1
