"""A6/A8 fixture tests: the privacy claim, in both directions, plus the librarian.

The headline test is gate 3 and gate 7 in executable form: a tile the gate refuses
still contributes buildings to the rank, and appears in no archive row, no
thumbnail, and no search result for any query, including the empty query that
returns the whole corpus. Then the same bytes are pushed back in through the
archive panel's own add-image button and refused again.

The gate itself belongs to another module, so it is monkeypatched here: these
tests are about the writer's behaviour given a verdict, which is the thing A6
owns. `ingest` and `grading` are stubbed for the same reason, and the stubs mirror
the frozen cross-slice signatures.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, contracts, db  # noqa: E402
from app import archive, embed, librarian  # noqa: E402

BOUNDS = [-122.400, 47.540, -122.390, 47.550]
CENTER = [-122.395, 47.545]


# ------------------------------------------------------------------- fixtures
@dataclass
class FakeVerdict:
    """Mirrors privacy_gate.GateVerdict, the frozen shape."""

    store_ok: bool
    person_detections: list[dict] = field(default_factory=list)
    detector_error: Optional[str] = None
    took_ms: int = 12
    tiles_scanned: int = 4

    def summary(self) -> dict:
        return {
            "store_ok": self.store_ok,
            "person_detections": len(self.person_detections),
            "detector_error": self.detector_error,
            "took_ms": self.took_ms,
            "tiles_scanned": self.tiles_scanned,
        }


@dataclass
class FakeBuilding:
    """Mirrors grading.GradedBuilding."""

    footprint_id: str
    cls: int
    conf: float
    caption: str
    centroid: list[float]
    graded_by: str = "nemotron-vl"
    geom: dict = field(default_factory=dict)
    area_m2: float = 120.0
    label: str = ""
    facility_near: object = None
    svi: float = 0.5


class GateSpy:
    """A programmable gate. `verdict` is what the next check returns."""

    def __init__(self) -> None:
        self.verdict = FakeVerdict(store_ok=True)
        self.calls: list[str] = []

    def check(self, image_path, **_):
        self.calls.append(str(image_path))
        return self.verdict

    def available(self) -> bool:
        return True

    def model_version(self) -> str:
        return "visdrone-yolov8x-fake"


@pytest.fixture()
def gate(monkeypatch):
    spy = GateSpy()
    module = types.ModuleType("app.privacy_gate")
    module.check = spy.check
    module.available = spy.available
    module.model_version = spy.model_version
    import app

    monkeypatch.setitem(sys.modules, "app.privacy_gate", module)
    monkeypatch.setattr(app, "privacy_gate", module, raising=False)
    return spy


@pytest.fixture()
def store(monkeypatch, tmp_path):
    """A private data root plus a fresh database per test."""
    for name in ("WATCH_DIR", "ANALYZED_DIR", "WITHHELD_DIR", "THUMB_DIR", "DATASET_DIR"):
        d = tmp_path / name.lower()
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, d)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "AOI", [-122.42, 47.52, -122.36, 47.58])
    monkeypatch.setattr(config, "REVIEW_TOKEN", "")
    monkeypatch.setattr(db, "_local", threading.local())
    db.init()

    # Deterministic ordering with no model weights on the box.
    embed.force_stub(True)

    # grading owns the caption; the archive borrows it and never re-calls the VLM.
    grading = types.ModuleType("app.grading")
    grading.vl_calls = 0

    def tile_caption(buildings):
        grading.vl_calls += 1
        best, best_cls, best_id = "", -1, None
        for b in buildings:
            if b.caption and b.cls > best_cls:
                best, best_cls, best_id = b.caption, b.cls, b.footprint_id
        return best, ("nemotron-vl" if best else "stub-pixelstat-v1"), best_id

    grading.tile_caption = tile_caption

    datasets = types.ModuleType("app.datasets")
    datasets.roads_geojson = lambda: {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "35th Ave SW"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-122.376, 47.550], [-122.376, 47.570]],
                },
            }
        ],
    }
    datasets.facilities_geojson = lambda: {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Providence Mount", "type": "nursing_home"},
                "geometry": {"type": "Point", "coordinates": [-122.395, 47.545]},
            }
        ],
    }
    datasets.reset_cache = lambda: None

    import app

    monkeypatch.setitem(sys.modules, "app.grading", grading)
    monkeypatch.setattr(app, "grading", grading, raising=False)
    monkeypatch.setitem(sys.modules, "app.datasets", datasets)
    monkeypatch.setattr(app, "datasets", datasets, raising=False)
    archive.reset_geocode_cache()

    yield tmp_path

    embed.force_stub(False)
    archive.reset_geocode_cache()


@pytest.fixture()
def pipeline(monkeypatch, store):
    """A stand-in for ingest.analyze_tile: the chokepoint IngestGeo owns.

    It does what the real one does in the order the contract fixes: write the
    tile and building rows, then hand the image to the archive writer and record
    the storage decision. Stubbed because this test file owns the writer, not the
    pipeline, and the door test must not depend on another module's progress.
    """
    calls: list[tuple[str, str]] = []

    def analyze_tile(path: Path, *, source: str = "watch") -> contracts.TileRecord:
        path = Path(path)
        calls.append((path.name, source))
        blds = _buildings(path.stem)
        rec = contracts.TileRecord(
            filename=path.name,
            bounds=list(BOUNDS),
            status="processed",
            captured_at=time.time(),
            latency_ms=42,
            buildings=[contracts.Building(id=b.footprint_id, cls=b.cls, conf=b.conf) for b in blds],
        )
        for b in blds:
            db.run(
                "INSERT OR REPLACE INTO buildings (footprint_id, label, centroid_json, "
                "damage_class, confidence, graded_by, last_seen_at, source_tile) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    b.footprint_id,
                    b.label or b.footprint_id,
                    f"[{b.centroid[0]},{b.centroid[1]}]",
                    b.cls,
                    b.conf,
                    b.graded_by,
                    time.time(),
                    path.name,
                ),
            )
        stored, reason = archive.try_store(path, rec, blds)
        rec.stored, rec.withheld_reason = stored, reason
        db.run(
            "INSERT OR REPLACE INTO tiles (filename, status, stored, withheld_reason, "
            "needs_geo, bounds_json, captured_at, analyzed_at, latency_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                path.name,
                "processed",
                1 if stored else 0,
                reason,
                0,
                "[-122.4,47.54,-122.39,47.55]",
                rec.captured_at,
                time.time(),
                rec.latency_ms,
            ),
        )
        return rec

    ingest = types.ModuleType("app.ingest")
    ingest.analyze_tile = analyze_tile
    ingest.calls = calls
    import app

    monkeypatch.setitem(sys.modules, "app.ingest", ingest)
    monkeypatch.setattr(app, "ingest", ingest, raising=False)
    return ingest


def test_dot_sits_on_the_captioned_building_not_the_tile_centre(gate, pipeline, tmp_path, store):
    """The archive dot must mark the structure the caption describes.

    A tile spans hundreds of metres and holds tens of buildings. This used to return
    the tile's geometric centre unconditionally, which put the pin on whatever
    happened to be in the middle - routinely a parking lot - while the caption beside
    it read "partial collapse". Clicking that dot showed an operator asphalt and told
    them a building had fallen down.
    """
    w, s, e, n = BOUNDS
    tile_centre = [(w + e) / 2.0, (s + n) / 2.0]
    # The damaged structure is in a corner, far from the middle of the frame.
    damaged = [w + (e - w) * 0.15, s + (n - s) * 0.85]
    intact = [w + (e - w) * 0.80, s + (n - s) * 0.20]

    import app.archive as arch

    blds = [
        FakeBuilding("t-worst", 3, 0.90, "roof collapsed into the ground floor", damaged),
        FakeBuilding("t-fine", 0, 0.70, "intact roof, no visible damage", intact),
    ]
    rec = contracts.TileRecord(
        filename="t.jpg", bounds=list(BOUNDS), status="processed",
        captured_at=time.time(), latency_ms=10,
        buildings=[contracts.Building(id=b.footprint_id, cls=b.cls, conf=b.conf) for b in blds],
    )
    stored, _ = arch.try_store(_image(tmp_path, "t.jpg"), rec, blds)
    assert stored

    rows = arch.search("", limit=10)["items"]
    row = next(r for r in rows if r["caption"])
    assert row["caption"] == "roof collapsed into the ground floor"
    assert row["caption_anchor"] == "t-worst"
    # On the damaged building, and NOT on the middle of the frame.
    assert row["centroid"] == [round(damaged[0], 6), round(damaged[1], 6)]
    assert row["centroid"] != [round(tile_centre[0], 6), round(tile_centre[1], 6)]


# --------------------------------------------------------------------- helpers
def _buildings(prefix: str, caption: str = "two-storey wood structure, roof collapsed") -> list:
    return [
        FakeBuilding(f"{prefix}-b1", 3, 0.81, caption, [CENTER[0], CENTER[1]]),
        FakeBuilding(f"{prefix}-b2", 1, 0.64, "single-storey structure, intact roof", CENTER),
    ]


def _image(tmp_path: Path, name: str, colour: tuple[int, int, int] = (90, 110, 70)) -> Path:
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (64, 48), colour).save(path, "JPEG", quality=92)
    return path


def _record(path: Path, bounds=BOUNDS) -> contracts.TileRecord:
    return contracts.TileRecord(
        filename=path.name,
        bounds=None if bounds is None else list(bounds),
        status="processed",
        captured_at=time.time(),
        latency_ms=30,
        buildings=[],
    )


def _archive_count() -> int:
    return int(db.q1("SELECT COUNT(*) AS n FROM archive")["n"])


def _thumbs() -> list[Path]:
    return sorted(config.THUMB_DIR.glob("*.jpg"))


def _store_cleared(tmp_path: Path, name: str, caption: str, cls: int = 3, colour=(80, 100, 60)):
    """Index one image that the gate clears, and return its id."""
    path = _image(tmp_path, name, colour)
    blds = _buildings(Path(name).stem, caption)
    for b in blds:
        b.cls = cls if b is blds[0] else 1
    rec = _record(path)
    stored, reason = archive.try_store(path, rec, blds)
    assert stored, reason
    return archive._image_id(path), path


# ================================================== the headline, both directions
def test_withheld_tile_ranks_but_is_unstorable_and_unsearchable(gate, store, pipeline):
    """Gate 3 and gate 7 in one test: analyzed and ranked, never stored."""
    cleared_id, _ = _store_cleared(store, "clear.jpg", "warehouse roof torn off, debris in street")
    assert _archive_count() == 1
    thumbs_before = set(_thumbs())

    person_tile = _image(store, "person_tile.jpg", (140, 60, 60))
    gate.verdict = FakeVerdict(
        store_ok=False,
        person_detections=[
            {"cls": 0, "name": "pedestrian", "conf": 0.87, "bbox": [10, 10, 20, 30]},
            {"cls": 1, "name": "people", "conf": 0.41, "bbox": [30, 10, 40, 30]},
        ],
    )
    rec = pipeline.analyze_tile(person_tile, source="downlink")
    person_id = archive._image_id(person_tile)

    # Direction one: it was analyzed, and its buildings are in the rank surface.
    assert rec.stored is False
    assert rec.withheld_reason == "person-signal conf=0.87"
    rows = db.q("SELECT footprint_id FROM buildings WHERE source_tile=?", (person_tile.name,))
    assert len(rows) == 2, "a person in frame is rescue signal, the buildings must survive"

    # Direction two: it reached no storage surface at all.
    assert _archive_count() == 1
    assert db.q1("SELECT image_id FROM archive WHERE image_id=?", (person_id,)) is None
    assert set(_thumbs()) == thumbs_before
    assert not (config.THUMB_DIR / f"{person_id}.jpg").exists()

    for query in (
        "",
        "person",
        "people in the street",
        "roof collapsed",
        "class:0",
        "class:3",
        "key:true",
        "after:00:00",
        "sector:B",
        "47.545, -122.395",
        "near Providence Mount",
        person_tile.stem,
    ):
        got = archive.search(query, limit=200)
        ids = [it["image_id"] for it in got["items"]]
        assert person_id not in ids, f"withheld image surfaced for query {query!r}"
    # The corpus is not empty, so the assertion above is about exclusion, not silence.
    assert cleared_id in [it["image_id"] for it in archive.search("", limit=200)["items"]]


def test_add_via_ingest_door_is_refused_again(gate, store, pipeline):
    """The archive panel's own button routes through the same gate. Same bytes,
    same deterministic id, refused twice."""
    person_tile = _image(store, "person_tile.jpg", (140, 60, 60))
    gate.verdict = FakeVerdict(
        store_ok=False,
        person_detections=[{"cls": 0, "name": "pedestrian", "conf": 0.91, "bbox": [1, 2, 3, 4]}],
    )
    first = pipeline.analyze_tile(person_tile, source="downlink")
    assert first.stored is False

    second = archive.add_via_ingest_door(person_tile)
    assert second.stored is False
    assert second.withheld_reason == "person-signal conf=0.91"
    assert pipeline.calls[-1][1] == "archive-add", "the door must re-run the whole pipeline"
    assert len(gate.calls) == 2, "the gate ran again on the re-add"
    assert _archive_count() == 0
    assert _thumbs() == []
    # The door itself must not name the file it just failed to store.
    for row in db.q("SELECT payload FROM decision_log"):
        assert person_tile.name not in (row["payload"] or "")


def test_add_via_ingest_door_names_a_stored_file(gate, store, pipeline):
    """The mirror of the test above: a cleared add IS named in the log, so the
    audit trail is not silent about what an operator added."""
    tile = _image(store, "clean_add.jpg", (30, 90, 60))
    rec = archive.add_via_ingest_door(tile)
    assert rec.stored is True
    payloads = [r["payload"] or "" for r in db.q("SELECT payload FROM decision_log")]
    assert any("clean_add.jpg" in p for p in payloads)
    assert _archive_count() == 1


def test_cleared_image_is_searchable(gate, store):
    image_id, _ = _store_cleared(store, "clear.jpg", "two-storey structure on fire, smoke rising")
    assert (config.THUMB_DIR / f"{image_id}.jpg").exists()

    got = archive.search("buildings on fire", limit=10)
    assert [it["image_id"] for it in got["items"]] == [image_id]
    assert "semantic" in got["resolved_by"]

    item = got["items"][0]
    assert item["thumb_path"] == f"/thumbs/{image_id}.jpg"
    assert item["class_max"] == 3
    assert "fire" in item["tags"]
    assert item["footprint_ids"] == ["clear-b1", "clear-b2"]
    assert set(item) == set(
        contracts.ArchiveItem(
            image_id="x",
            thumb_path="",
            captured_at=0.0,
            centroid=None,
            needs_geo=False,
            caption="",
            tags=[],
            class_max=0,
        ).wire()
    )


# ================================================= caption is the second control
def test_caption_person_language_rewithholds_and_deletes_thumbnail(gate, store):
    """A caption is a second chance to catch what the detector missed."""
    path = _image(store, "cleared_but_captioned.jpg")
    image_id = archive._image_id(path)
    # Simulate a thumbnail left by an earlier, less careful pass over these bytes.
    stale = config.THUMB_DIR / f"{image_id}.jpg"
    stale.write_bytes(b"stale-thumb")
    assert stale.exists()

    blds = _buildings("cap", "a person in a red jacket beside a collapsed porch")
    stored, reason = archive.try_store(path, _record(path), blds)

    assert stored is False
    assert reason == "caption-person-language"
    assert _archive_count() == 0
    assert not stale.exists(), "the writer must delete a thumbnail it re-withholds"
    assert archive.search("collapsed porch", limit=10)["items"] == []


def test_update_metadata_person_language_evicts_row_and_thumbnail(gate, store):
    image_id, _ = _store_cleared(store, "clear.jpg", "flooded intersection, debris on the roadway")
    thumb = config.THUMB_DIR / f"{image_id}.jpg"
    assert thumb.exists()

    ok = archive.update_metadata(image_id, caption="water up to the roofline", operator="chief")
    assert ok["ok"] is True
    assert archive.get(image_id)["caption"] == "water up to the roofline"

    bad = archive.update_metadata(
        image_id, caption="two survivors on the roof waiting", operator="chief"
    )
    assert bad["evicted"] is True
    assert bad["reason"] == "caption-person-language"
    assert _archive_count() == 0
    assert not thumb.exists()
    assert archive.get(image_id) is None
    assert archive.search("water up to the roofline", limit=10)["items"] == []


def test_update_metadata_person_language_in_a_tag_also_evicts(gate, store):
    image_id, _ = _store_cleared(store, "clear.jpg", "storefront window blown in, glass on sidewalk")
    bad = archive.update_metadata(image_id, tags=["storefront", "crowd"], operator="chief")
    assert bad["evicted"] is True
    assert _archive_count() == 0


def test_gate_error_is_treated_as_a_refusal(gate, store):
    path = _image(store, "err.jpg")
    gate.verdict = FakeVerdict(store_ok=False, detector_error="weights missing")
    stored, reason = archive.try_store(path, _record(path), _buildings("err"))
    assert stored is False
    assert reason.startswith("detector-error")
    assert _archive_count() == 0
    assert _thumbs() == []


def test_missing_gate_module_still_refuses(store, monkeypatch):
    """No detector at all is a refusal, not an open door."""
    import app

    monkeypatch.setattr(app, "privacy_gate", None, raising=False)
    monkeypatch.setattr(archive, "_gate_check", lambda p: (_ for _ in ()).throw(ImportError("nope")))
    path = _image(store, "nogate.jpg")
    stored, reason = archive.try_store(path, _record(path), _buildings("nogate"))
    assert stored is False
    assert "detector-error" in reason
    assert _archive_count() == 0


def test_try_store_never_raises(gate, store, monkeypatch):
    """A decision function has two outcomes. An exception would be a third one a
    caller could mishandle into "stored"."""
    path = _image(store, "boom.jpg")

    def explode(*a, **kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(archive, "_write_thumb", explode)
    stored, reason = archive.try_store(path, _record(path), _buildings("boom"))
    assert stored is False
    assert "index-error" in reason
    assert _archive_count() == 0
    assert _thumbs() == []

    # Even a broken gate verdict object, not just a broken call.
    monkeypatch.setattr(archive, "_gate_check", lambda p: object())
    stored, reason = archive.try_store(path, _record(path), _buildings("boom"))
    assert stored is False
    assert reason == "detector-error: gate refused without a reason"


def test_a_failed_cleanup_does_not_rename_the_verdict(gate, store, monkeypatch):
    """The audit line must say why the image was refused, not report the failure
    of the tidy-up step that ran afterwards."""
    path = _image(store, "person_tile.jpg", (140, 60, 60))
    gate.verdict = FakeVerdict(
        store_ok=False, person_detections=[{"cls": 0, "name": "pedestrian", "conf": 0.63}]
    )
    monkeypatch.setattr(
        archive, "evict", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("db locked"))
    )
    stored, reason = archive.try_store(path, _record(path), _buildings("p"))
    assert stored is False
    assert reason == "person-signal conf=0.63"
    assert _archive_count() == 0


def test_evict_removes_row_and_thumbnail(gate, store):
    image_id, _ = _store_cleared(store, "clear.jpg", "warehouse roof torn off")
    thumb = config.THUMB_DIR / f"{image_id}.jpg"
    assert thumb.exists()
    assert archive.evict(image_id) is True
    assert not thumb.exists()
    assert _archive_count() == 0
    assert archive.evict(image_id) is False


def test_one_vl_pass_per_tile(gate, store):
    """A6: never call the VLM twice per crop. The archive borrows the grade pass."""
    grading = sys.modules["app.grading"]
    grading.vl_calls = 0
    _store_cleared(store, "clear.jpg", "roof collapsed, standing water in the street")
    assert grading.vl_calls == 1


# ====================================================== filter, then rank
def test_resolvers_narrow_then_rank(gate, store):
    ids = {}
    ids["fire3"], _ = _store_cleared(
        store, "a.jpg", "commercial structure on fire, heavy smoke", cls=3, colour=(10, 20, 30)
    )
    ids["fire1"], _ = _store_cleared(
        store, "b.jpg", "small fire on a shed roof", cls=1, colour=(40, 20, 30)
    )
    ids["flood3"], _ = _store_cleared(
        store, "c.jpg", "flooded intersection, standing water over the kerb", cls=3, colour=(10, 60, 30)
    )
    assert _archive_count() == 3

    everything = archive.search("", limit=50)
    assert len(everything["items"]) == 3
    assert everything["resolved_by"] == []

    # Structured filter NARROWS.
    narrowed = archive.search("class:3", limit=50)
    assert narrowed["resolved_by"] == ["filter"]
    got = {it["image_id"] for it in narrowed["items"]}
    assert got == {ids["fire3"], ids["flood3"]}
    assert ids["fire1"] not in got

    # Semantic RANKS whatever survived, and does not re-widen it.
    combined = archive.search("class:3 fire smoke", limit=50)
    assert combined["resolved_by"] == ["filter", "semantic"]
    order = [it["image_id"] for it in combined["items"]]
    assert set(order) == {ids["fire3"], ids["flood3"]}
    assert order[0] == ids["fire3"], "cosine must reorder the narrowed set"

    # Pure semantic ranks the whole corpus, in a different order per query.
    water_first = [it["image_id"] for it in archive.search("standing water", limit=50)["items"]]
    fire_first = [it["image_id"] for it in archive.search("heavy smoke", limit=50)["items"]]
    assert water_first[0] == ids["flood3"]
    assert fire_first[0] == ids["fire3"]
    assert water_first != fire_first


def test_structured_tokens_after_and_key(gate, store):
    old_id, old_path = _store_cleared(store, "old.jpg", "roof damage on a bungalow", colour=(1, 2, 3))
    new_id, _ = _store_cleared(store, "new.jpg", "roof damage on a duplex", colour=(3, 2, 1))
    # 06:00 local on the day the row was written, so the filter has a real edge.
    lt = time.localtime(time.time())
    six = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 6, 0, 0, 0, 0, -1))
    db.run("UPDATE archive SET captured_at=? WHERE image_id=?", (six - 3600, old_id))
    db.run("UPDATE archive SET captured_at=? WHERE image_id=?", (six + 3600, new_id))

    after = archive.search("after:06:00", limit=50)
    assert [it["image_id"] for it in after["items"]] == [new_id]
    before = archive.search("before:06:00", limit=50)
    assert [it["image_id"] for it in before["items"]] == [old_id]

    assert archive.search("key:true", limit=50)["items"] == []
    archive.update_metadata(new_id, key_evidence=True, operator="chief")
    keyed = archive.search("key:true", limit=50)
    assert [it["image_id"] for it in keyed["items"]] == [new_id]


def test_location_resolver_narrows_by_bbox(gate, store):
    inside, _ = _store_cleared(store, "inside.jpg", "roof torn off a duplex", colour=(9, 9, 9))
    outside, _ = _store_cleared(store, "outside.jpg", "roof torn off a bungalow", colour=(9, 9, 8))
    db.run("UPDATE archive SET centroid_json=? WHERE image_id=?", ("[-122.3765,47.5601]", inside))
    db.run("UPDATE archive SET centroid_json=? WHERE image_id=?", ("[-122.410,47.5300]", outside))

    street = archive.search("35th Ave SW", limit=50)
    assert street["resolved_by"] == ["location"]
    assert [it["image_id"] for it in street["items"]] == [inside]

    # A road name plus semantic terms: narrow to the street, then rank.
    both = archive.search("roof torn off near 35th Ave SW", limit=50)
    assert both["resolved_by"] == ["location", "semantic"]
    assert [it["image_id"] for it in both["items"]] == [inside]

    typed = archive.search("47.5601, -122.3765", limit=50)
    assert typed["resolved_by"] == ["location"]
    assert [it["image_id"] for it in typed["items"]] == [inside]


def test_sector_grid_covers_the_aoi(gate, store):
    w, s, e, n = config.AOI
    a1 = archive.sector_bounds("A1")
    d4 = archive.sector_bounds("D4")
    assert a1[0] == pytest.approx(w) and a1[3] == pytest.approx(n)
    assert d4[2] == pytest.approx(e) and d4[1] == pytest.approx(s)
    assert archive.sector_bounds("Z9") is None
    column = archive.sector_bounds("C")
    assert column[1] == pytest.approx(s) and column[3] == pytest.approx(n)


def test_sector_is_a_structured_filter_not_the_location_resolver(gate, store):
    """The plan's own table lists sector:C under the structured filter, so the
    resolver strip must say "filter" for it and reserve "location" for a geocode."""
    inside, _ = _store_cleared(store, "in.jpg", "roof torn off", colour=(6, 6, 6))
    outside, _ = _store_cleared(store, "out.jpg", "roof torn off", colour=(6, 6, 5))
    b2 = archive.sector_bounds("B2")
    mid = [(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2]
    db.run("UPDATE archive SET centroid_json=? WHERE image_id=?", (f"[{mid[0]},{mid[1]}]", inside))
    db.run("UPDATE archive SET centroid_json=? WHERE image_id=?", ("[-122.365,47.525]", outside))

    got = archive.search("sector:B2", limit=50)
    assert got["resolved_by"] == ["filter"]
    assert [it["image_id"] for it in got["items"]] == [inside]
    assert got["location"] == "sector B2"


def test_two_spatial_terms_intersect_rather_than_widen(gate, store):
    """More typed terms must never mean more results."""
    on_street, _ = _store_cleared(store, "street.jpg", "roof torn off", colour=(7, 7, 7))
    elsewhere, _ = _store_cleared(store, "away.jpg", "roof torn off", colour=(7, 7, 6))
    # On 35th Ave SW, which the fixture puts in the eastern column, north half.
    db.run("UPDATE archive SET centroid_json=? WHERE image_id=?", ("[-122.3765,47.5601]", on_street))
    # In sector A1 but nowhere near that street.
    db.run("UPDATE archive SET centroid_json=? WHERE image_id=?", ("[-122.415,47.575]", elsewhere))

    assert len(archive.search("sector:A1", limit=50)["items"]) == 1
    assert len(archive.search("35th Ave SW", limit=50)["items"]) == 1
    both = archive.search("sector:A1 near 35th Ave SW", limit=50)
    assert both["resolved_by"] == ["filter", "location"]
    assert both["items"] == [], "an impossible pair of constraints matches nothing"


def test_search_survives_a_missing_embedding(gate, store):
    image_id, _ = _store_cleared(store, "clear.jpg", "debris field across the parking lot")
    db.run("UPDATE archive SET embedding=NULL WHERE image_id=?", (image_id,))
    got = archive.search("debris", limit=10)
    assert [it["image_id"] for it in got["items"]] == [image_id], "a candidate must not be dropped"


def test_resolved_by_only_claims_resolvers_that_fired(gate, store):
    """The strip must not say "semantic" when nothing was ranked."""
    image_id, _ = _store_cleared(store, "clear.jpg", "debris field across the parking lot")

    # Every token a stopword, so the query vector is all zeros: no ranking happened.
    stopwords = archive.search("the and of", limit=10)
    assert stopwords["resolved_by"] == []
    assert [it["image_id"] for it in stopwords["items"]] == [image_id]

    # A road name consumed by the geocoder leaves no text behind to rank on.
    only_place = archive.search("35th Ave SW", limit=10)
    assert only_place["resolved_by"] == ["location"]


def test_class_filter_is_a_severity_floor(gate, store):
    """class:2 means "at least major", the same reading as SEVERE_FROM."""
    minor, _ = _store_cleared(store, "minor.jpg", "cracked kerb", cls=1, colour=(2, 2, 2))
    major, _ = _store_cleared(store, "major.jpg", "wall sheared away", cls=2, colour=(3, 3, 3))
    gone, _ = _store_cleared(store, "gone.jpg", "structure flattened", cls=3, colour=(4, 4, 4))

    assert {it["image_id"] for it in archive.search("class:2", limit=50)["items"]} == {major, gone}
    assert {it["image_id"] for it in archive.search("class:0", limit=50)["items"]} == {
        minor,
        major,
        gone,
    }
    # Garbage in a token is ignored rather than raising or silently matching all.
    assert archive.search("class:banana", limit=50)["resolved_by"] == []


def test_a_body_of_water_is_terrain_not_a_person(gate, store):
    """The captioner is prompted to describe water, so the filter must not read
    "body of water" as a body. The exemption is narrow: "a body on the roof" is
    still a refusal."""
    assert archive.caption_mentions_person("a large body of water covers the road") is False
    assert archive.caption_mentions_person("bodies of water on both sides") is False
    assert archive.caption_mentions_person("a body on the roof of the garage") is True

    image_id, _ = _store_cleared(store, "water.jpg", "a body of water covers the intersection")
    assert archive.get(image_id) is not None


# ============================================================ HUD and review
def test_stats_counts_indexed_and_withheld(gate, store, pipeline):
    _store_cleared(store, "clear.jpg", "roof collapsed on a duplex")
    gate.verdict = FakeVerdict(
        store_ok=False, person_detections=[{"cls": 0, "name": "pedestrian", "conf": 0.7}]
    )
    pipeline.analyze_tile(_image(store, "person_tile.jpg", (140, 60, 60)))
    st = archive.stats()
    assert st["indexed"] == 1
    assert st["withheld_from_storage"] == 1
    assert st["thumbnails"] == 1
    assert st["embedder"] == embed.STUB_VERSION
    assert st["embedder_stub"] is True


def test_withheld_review_is_the_only_filename_surface(gate, store, pipeline, monkeypatch):
    gate.verdict = FakeVerdict(
        store_ok=False, person_detections=[{"cls": 0, "name": "pedestrian", "conf": 0.7}]
    )
    pipeline.analyze_tile(_image(store, "person_tile.jpg", (140, 60, 60)))

    assert archive.review_configured() is False
    with pytest.raises(PermissionError):
        archive.withheld_review("")
    with pytest.raises(PermissionError):
        archive.withheld_review("guess")

    monkeypatch.setattr(config, "REVIEW_TOKEN", "s3cret")
    assert archive.review_configured() is True
    with pytest.raises(PermissionError):
        archive.withheld_review("wrong")
    rows = archive.withheld_review("s3cret")
    assert [r["filename"] for r in rows] == ["person_tile.jpg"]
    assert rows[0]["withheld_reason"].startswith("person-signal")


def test_decision_log_never_names_a_withheld_file(gate, store, pipeline):
    gate.verdict = FakeVerdict(
        store_ok=False, person_detections=[{"cls": 0, "name": "pedestrian", "conf": 0.7}]
    )
    tile = _image(store, "person_tile.jpg", (140, 60, 60))
    pipeline.analyze_tile(tile)
    rows = db.q("SELECT actor, action, payload FROM decision_log")
    assert rows, "the storage decision must be logged"
    for r in rows:
        assert tile.name not in (r["payload"] or ""), "a withheld filename must not reach the log"
    assert any(r["action"] == "storage-decision" for r in rows)


# ================================================================= the embedder
def test_embed_rows_are_normalized_and_deterministic(store):
    vecs = embed.encode(["roof collapsed", "roof collapsed", "standing water"])
    assert vecs.shape == (3, embed.dim())
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.array_equal(vecs[0], vecs[1])
    assert float(vecs[0] @ vecs[2]) < float(vecs[0] @ vecs[1])
    assert embed.encode([]).shape == (0, embed.dim())
    assert np.all(embed.encode([""]) == 0.0), "an empty caption scores zero, honestly"


# ================================================================== librarian
def test_librarian_rejects_an_unknown_name(store):
    with pytest.raises(ValueError):
        librarian.refresh("totally_new_source")
    with pytest.raises(ValueError):
        librarian.refresh("")
    assert db.q("SELECT name FROM datasets") == []


def test_librarian_never_accepts_a_url(store):
    """An injected instruction has no fetch primitive to abuse."""
    import inspect

    for url in (
        "https://evil.example.com/exfil",
        "http://127.0.0.1:9999/",
        "file:///etc/passwd",
        "https://storms.ngs.noaa.gov/",  # even a real allowlisted URL is not a name
    ):
        with pytest.raises(ValueError):
            librarian.refresh(url)

    banned = {"url", "uri", "href", "endpoint", "address", "host", "source", "destination"}
    for name in librarian.__all__:
        obj = getattr(librarian, name)
        if not callable(obj):
            continue
        params = set(inspect.signature(obj).parameters)
        assert not (params & banned), f"librarian.{name} exposes a destination parameter"


def test_librarian_allowlist_is_exactly_the_five_named_sources(store):
    assert set(librarian.ALLOWLIST) == {
        "noaa_storm_imagery",
        "xview2_labels",
        "ms_building_footprints",
        "cms_facilities",
        "cdc_svi",
    }
    for name, entry in librarian.ALLOWLIST.items():
        assert entry["url"].startswith("https://"), name
        assert entry["kind"] and entry["note"]


def test_catalog_reports_never_fetched_honestly(store):
    rows = librarian.catalog()
    assert len(rows) == len(librarian.ALLOWLIST)
    for row in rows:
        assert row["last_refreshed"] is None
        assert row["present"] is False


def test_refresh_swaps_atomically_and_records_the_checksum(store, monkeypatch):
    """GET only, checksum verified, temp file then atomic swap, row plus log."""
    import hashlib

    dest = librarian.local_path("cdc_svi")
    dest.write_bytes(b"the previous local copy")
    blob = b'{"svi": "fresh"}'
    monkeypatch.setattr(librarian, "_fetch", lambda name: (blob, "application/json"))

    got = librarian.refresh("cdc_svi", actor="agent")
    assert got["ok"] is True
    assert got["sha256"] == hashlib.sha256(blob).hexdigest()
    assert got["bytes"] == len(blob)
    assert got["changed"] is True
    assert dest.read_bytes() == blob
    assert list(config.DATASET_DIR.glob(".refresh-*")) == [], "no temp files left behind"

    row = db.q1("SELECT * FROM datasets WHERE name=?", ("cdc_svi",))
    assert row["sha256"] == got["sha256"]
    assert row["source"] == librarian.ALLOWLIST["cdc_svi"]["url"]
    assert row["bytes"] == len(blob)
    assert any(r["action"] == "dataset-refresh" for r in db.q("SELECT action FROM decision_log"))

    again = librarian.refresh("cdc_svi")
    assert again["changed"] is False, "the same bytes are not a change"
    assert librarian.catalog()[0]["name"] in librarian.ALLOWLIST


def test_refresh_keeps_the_local_copy_when_the_network_fails(store, monkeypatch):
    dest = librarian.local_path("cms_facilities")
    dest.write_bytes(b"good local copy")

    def boom(name):
        raise OSError("no route to host")

    monkeypatch.setattr(librarian, "_fetch", boom)
    got = librarian.refresh("cms_facilities")
    assert got["ok"] is False
    assert "no route to host" in got["error"]
    assert dest.read_bytes() == b"good local copy", "a failed refresh must not damage the store"
    assert db.q("SELECT name FROM datasets") == []


def test_refresh_refuses_a_checksum_mismatch(store, monkeypatch):
    dest = librarian.local_path("xview2_labels")
    dest.write_bytes(b"trusted local copy")
    monkeypatch.setitem(librarian.ALLOWLIST["xview2_labels"], "sha256", "00" * 32)
    monkeypatch.setattr(librarian, "_fetch", lambda name: (b"tampered", "text/html"))

    got = librarian.refresh("xview2_labels")
    assert got["ok"] is False
    assert "mismatch" in got["error"]
    assert dest.read_bytes() == b"trusted local copy"


def test_agent_tool_schema_offers_names_only(store):
    schema = librarian.agent_tool_schema()
    fn = schema["function"]
    assert fn["name"] == "refresh_dataset"
    props = fn["parameters"]["properties"]
    assert set(props) == {"name"}
    assert props["name"]["enum"] == list(librarian.ALLOWLIST)
    assert fn["parameters"]["additionalProperties"] is False
