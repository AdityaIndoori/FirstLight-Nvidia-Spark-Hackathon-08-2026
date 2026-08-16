import sys
import time

sys.path.insert(0, ".")
from app import main  # noqa: E402

# Time each contributor to /api/status separately, so the slow one is named rather
# than guessed at.
from app import db, embed  # noqa: E402

steps = {
    "scorer.rank": lambda: main.scorer.rank(),
    "_tokens_per_s": lambda: main._tokens_per_s(),
    "_mem": lambda: main._mem(),
    "_power": lambda: main._power(),
    "ingest.latency_p50": lambda: main._mod("ingest").latency_p50(),
    "ingest.counts": lambda: main._mod("ingest").counts(),
    "archive.stats": lambda: main._mod("archive").stats(),
    "ballot.model_version": lambda: main._mod("ballot").model_version(),
    "grading.model_version": lambda: main.grading.model_version(),
    "gate.model_version": lambda: main.gate.model_version() if main.gate else None,
    "embed.model_version": lambda: embed.model_version(),
}
for name, fn in steps.items():
    t = time.time()
    try:
        fn()
        print(f"  {name:26} {time.time()-t:7.3f}s")
    except Exception as exc:
        print(f"  {name:26} FAILED {type(exc).__name__}: {str(exc)[:70]}")

t = time.time()
main.status()
print(f"  {'FULL status()':26} {time.time()-t:7.3f}s")
