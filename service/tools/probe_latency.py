import sys
sys.path.insert(0, ".")
from app import db, ingest

rows = db.q("SELECT filename, latency_ms, captured_at FROM tiles ORDER BY captured_at DESC LIMIT 10")
for r in rows:
    print(f"  {r['latency_ms']:>7} ms  ts={r['captured_at']}  {r['filename'][:46]}")
print("total tiles:", db.q1("SELECT COUNT(*) AS n FROM tiles")["n"])
print("null captured_at:", db.q1("SELECT COUNT(*) AS n FROM tiles WHERE captured_at IS NULL")["n"])
print("window:", ingest.LATENCY_WINDOW)
print("_latencies():", ingest._latencies())
print("p50:", ingest.latency_p50())
