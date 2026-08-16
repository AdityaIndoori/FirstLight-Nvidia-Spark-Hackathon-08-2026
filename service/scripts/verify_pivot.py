"""End-to-end proof of the storage pivot, run on the box.

A tile the gate withholds must STILL rank, because a person in frame is rescue
signal, and it must reach no archive surface at all. The detector seam is driven
here because the synthetic fixture is not a real person; A5 measures real recall
on real aerial imagery separately.
"""
import shutil

from app import archive, config, db, ingest, privacy_gate, scorer

db.init()
src = config.DATA / "sample_tiles" / "fixture_person.jpg"
tile = config.WATCH_DIR / "person_e2e.jpg"
shutil.copy(src, tile)
shutil.copy(str(src) + ".bounds.json", str(tile) + ".bounds.json")

privacy_gate._detect = lambda image, conf: [
    {"cls": 0, "name": "pedestrian", "conf": 0.74, "bbox": [310, 300, 322, 330]},
    {"cls": 0, "name": "pedestrian", "conf": 0.41, "bbox": [356, 340, 368, 372]},
    {"cls": 3, "name": "car", "conf": 0.88, "bbox": [700, 200, 780, 250]},
]

rec = ingest.analyze_tile(tile, source="upload")
print("status:", rec.status, "| stored:", rec.stored, "| reason:", rec.withheld_reason)
print("buildings analyzed:", len(rec.buildings))

ids = {b.id for b in rec.buildings}
ranked = {i["footprint_id"] for i in scorer.rank(limit=2000)["items"]}
print("ITS BUILDINGS IN THE RANK:", len(ids & ranked), "of", len(ids))

hits = archive.search("", 500)
row = db.q1("SELECT COUNT(*) AS n FROM archive WHERE filename = ?", ("person_e2e.jpg",))
print("ARCHIVE ROWS FOR IT:", int(row["n"]))
print(
    "searchable via any query:",
    any("person_e2e" in str(i.get("thumb_path", "")) for i in hits["items"]),
)
print(
    "placed in:",
    "WITHHELD_DIR" if (config.WITHHELD_DIR / "person_e2e.jpg").exists() else "ANALYZED_DIR",
)
leak = db.q("SELECT COUNT(*) AS n FROM decision_log WHERE payload LIKE ?", ("%person_e2e%",))
print("log payloads naming it:", int(leak[0]["n"]))

# The add-image door must refuse the same bytes again. The file now lives in the
# vault, so hand the door a fresh copy exactly as the archive panel would.
again_src = config.WITHHELD_DIR / "person_e2e.jpg"
retry = config.DATA / "readd_person.jpg"
shutil.copy(again_src, retry)
again = archive.add_via_ingest_door(retry)
print("re-added via archive door, stored:", getattr(again, "stored", None))
print("archive rows after re-add:", int(db.q1("SELECT COUNT(*) AS n FROM archive")["n"]))
