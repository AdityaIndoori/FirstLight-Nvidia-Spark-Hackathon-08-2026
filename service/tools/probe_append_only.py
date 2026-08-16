import sys

sys.path.insert(0, ".")
from app import db  # noqa: E402

before = db.q1("SELECT COUNT(*) AS n FROM decision_log")["n"]
for sql in ("DELETE FROM decision_log", "UPDATE decision_log SET actor='x'"):
    try:
        db.run(sql)
        print(f"  {sql[:28]:30} ALLOWED  <-- audit trail is not protected")
    except Exception as exc:
        print(f"  {sql[:28]:30} refused: {str(exc)[:44]}")
after = db.q1("SELECT COUNT(*) AS n FROM decision_log")["n"]
print(f"  rows before={before} after={after} (must be equal)")
