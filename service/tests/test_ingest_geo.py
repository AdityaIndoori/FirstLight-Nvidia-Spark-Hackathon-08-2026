"""A1 + A7 fixture tests: the geo fallback chain and the ingest chokepoint.

No network, no models, no real weights. grading, datasets and archive are stubbed
because what is under test is the ORDER and the failure behaviour of the pipeline,
not anybody else's inference.
"""
from __future__ import annotations

import json
import sys
import threading
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, geo, ingest  # noqa: E402


# ------------------------------------------------------------------- fixtures
def _img(path: Path, size: tuple[int, int] = (24, 18), colour=(30, 40, 50)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def _jpeg_with_gps(path: Path, lat: float, lng: float) -> Path:
    """Write a JPEG carrying real EXIF GPS DMS rationals."""

    def dms(value: float) -> tuple[Fraction, Fraction, Fraction]:
        value = abs(value)
        d = int(value)
        m_full = (value - d) * 60.0
        m = int(m_full)
        s = round((m_full - m) * 60.0, 4)
        return (Fraction(d, 1), Fraction(m, 1), Fraction(int(s * 10_000), 10_000))

    exif = Image.Exif()
    exif[0x8825] = {
        1: "N" if lat >= 0 else "S",
        2: dms(lat),
        3: "E" if lng >= 0 else "W",
        4: dms(lng),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), (60, 60, 60)).save(path, exif=exif)
    return path


class FakeBuilding:
    """The shape grading.outline_and_grade returns, per the frozen interface."""

    def __init__(self, fid: str, cls: int = 3, conf: float = 0.8):
        self.footprint_id = fid
        self.cls = cls
        self.conf = conf
        self.centroid = [-122.39, 47.55]
        self.geom = {"type": "Point", "coordinates": [-122.39, 47.55]}
        self.area_m2 = 120.0
        self.graded_by = "nemotron-vl"
        self.caption = "two-storey wood structure, roof collapsed"
        self.label = None
        self.facility_near = None
        self.svi = None


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every config path and the thread-local DB connection at tmp_path."""
    for name in ("WATCH_DIR", "ANALYZED_DIR", "WITHHELD_DIR", "THUMB_DIR", "DATASET_DIR"):
        d = tmp_path / name.lower()
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, d)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "_local", type(db._local)())
    monkeypatch.setattr(ingest, "_inited", set())
    ingest.reset_watch_state()
    db.init()
    return tmp_path


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Install stub grading/datasets/archive modules for the lazy imports.

    ingest imports them as `from . import grading`, so the stubs go into
    sys.modules under the package name and monkeypatch removes them after.
    """
    import types

    calls: dict = {"order": [], "stored": [], "grade_raises": False, "store": (True, None)}

    grading = types.ModuleType("app.grading")

    def outline_and_grade(image_path, bounds):
        calls["order"].append("grade")
        if calls["grade_raises"]:
            raise RuntimeError("seg model unavailable")
        if bounds is None:
            return []
        return [FakeBuilding("fp-1"), FakeBuilding("fp-2", cls=1, conf=0.4)]

    grading.outline_and_grade = outline_and_grade

    datasets = types.ModuleType("app.datasets")

    def join(buildings, bounds):
        calls["order"].append("join")
        for b in buildings:
            b.label = "4200 SW Admiral Way"
            b.svi = 0.91

    datasets.join = join

    archive = types.ModuleType("app.archive")

    def try_store(image_path, tile_record, buildings):
        calls["order"].append("store")
        calls["stored"].append(Path(image_path).name)
        return calls["store"]

    archive.try_store = try_store

    # ingest does `from . import archive`, which reads the attribute on the `app`
    # package, not sys.modules. Once another test file has imported the real
    # module the attribute exists and shadows any sys.modules patch, so patch
    # both or these stubs are silently ignored when the full suite runs.
    import app as app_pkg

    for name, mod in (
        ("app.grading", grading),
        ("app.datasets", datasets),
        ("app.archive", archive),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(app_pkg, name.split(".", 1)[1], mod, raising=False)
    return calls


# ------------------------------------------------------------------ geo: chain
def test_sidecar_wins_when_no_geotiff_or_exif(sandbox):
    """A PNG has no transform and no EXIF GPS, so the sidecar is the live link."""
    img = _img(sandbox / "frame.png")
    (sandbox / "frame.bounds.json").write_text(
        json.dumps({"bounds": [-122.40, 47.54, -122.38, 47.56]})
    )
    result = geo.extract(img)
    assert result.source == geo.SOURCE_SIDECAR
    assert result.needs_geo is False
    assert result.bounds == [-122.40, 47.54, -122.38, 47.56]
    assert result.centroid == pytest.approx([-122.39, 47.55])


def test_sidecar_point_form_synthesizes_a_bbox(sandbox):
    img = _img(sandbox / "point.png")
    (sandbox / "point.bounds.json").write_text(json.dumps({"lng": -122.39, "lat": 47.55}))
    result = geo.extract(img)
    assert result.source == geo.SOURCE_SIDECAR
    w, s, e, n = result.bounds
    assert w < -122.39 < e and s < 47.55 < n
    # A synthesized footprint is a tile, not a county.
    assert 0.0 < (e - w) < 0.01


def test_exif_gps_beats_sidecar(sandbox):
    """Chain order is a contract: EXIF is link two, the sidecar is link three."""
    img = _jpeg_with_gps(sandbox / "gps.jpg", 47.5583, -122.3771)
    (sandbox / "gps.bounds.json").write_text(json.dumps({"bounds": [-1.0, -1.0, 1.0, 1.0]}))
    result = geo.extract(img)
    assert result.source == geo.SOURCE_EXIF
    assert result.needs_geo is False
    lng, lat = result.centroid
    assert lat == pytest.approx(47.5583, abs=1e-3)
    assert lng == pytest.approx(-122.3771, abs=1e-3)


def test_exif_without_piexif_falls_back_to_pillow(sandbox, monkeypatch):
    """The box has piexif pinned, but its absence must cost us locations, not
    turn every JPEG into a needs_geo card."""
    monkeypatch.setattr(geo, "piexif", None)
    img = _jpeg_with_gps(sandbox / "nopiexif.jpg", 47.5583, -122.3771)
    result = geo.extract(img)
    assert result.source == geo.SOURCE_EXIF
    assert result.centroid[1] == pytest.approx(47.5583, abs=1e-3)


def test_jpeg_without_gps_falls_through_to_the_sidecar(sandbox):
    img = _img(sandbox / "plain.jpg")
    (sandbox / "plain.bounds.json").write_text(
        json.dumps({"bounds": [-122.40, 47.54, -122.38, 47.56]})
    )
    assert geo.extract(img).source == geo.SOURCE_SIDECAR


def test_geotiff_transform_is_link_one(sandbox, monkeypatch):
    """rasterio is not installed everywhere, so stub the reader and assert the
    ORDER: a GeoTIFF transform outranks both EXIF and a sidecar."""

    class FakeBounds:
        left, bottom, right, top = -122.40, 47.54, -122.38, 47.56

    class FakeCRS:
        def to_epsg(self):
            return 4326

    class FakeDataset:
        crs = FakeCRS()
        bounds = FakeBounds()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = type("FakeRasterio", (), {"open": staticmethod(lambda path: FakeDataset())})()
    monkeypatch.setattr(geo, "rasterio", fake)
    tif = _img(sandbox / "ortho.tif")
    (sandbox / "ortho.bounds.json").write_text(json.dumps({"bounds": [-1.0, -1.0, 1.0, 1.0]}))

    result = geo.extract(tif)
    assert result.source == geo.SOURCE_GEOTIFF
    assert result.bounds == [-122.40, 47.54, -122.38, 47.56]


def test_geotiff_without_crs_falls_through(sandbox, monkeypatch):
    """A tif carrying pixel coordinates only must not become a location."""

    class NoCRS:
        crs = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        geo, "rasterio", type("R", (), {"open": staticmethod(lambda p: NoCRS())})()
    )
    tif = _img(sandbox / "pixels.tif")
    (sandbox / "pixels.bounds.json").write_text(
        json.dumps({"bounds": [-122.40, 47.54, -122.38, 47.56]})
    )
    result = geo.extract(tif)
    assert result.source == geo.SOURCE_SIDECAR


def test_corrupt_geotiff_does_not_break_the_chain(sandbox, monkeypatch):
    def boom(path):
        raise OSError("not a tif")

    monkeypatch.setattr(geo, "rasterio", type("R", (), {"open": staticmethod(boom)})())
    tif = _img(sandbox / "corrupt.tif")
    result = geo.extract(tif)
    assert result.needs_geo is True
    assert "geotiff read failed" in result.detail


def test_needs_geo_when_nothing_resolves(sandbox):
    result = geo.extract(_img(sandbox / "bare.png"))
    assert result.bounds is None
    assert result.source == geo.SOURCE_NONE
    assert result.needs_geo is True
    assert result.detail  # the console must be able to say why


def test_missing_file_still_returns_a_result(sandbox):
    """A7: never drop an image, and never raise out of the chain."""
    result = geo.extract(sandbox / "does-not-exist.jpg")
    assert result.needs_geo is True
    assert result.bounds is None


def test_sidecar_beside_full_filename_also_resolves(sandbox):
    img = _img(sandbox / "export.png")
    (sandbox / "export.png.bounds.json").write_text(
        json.dumps({"bounds": [-122.40, 47.54, -122.38, 47.56]})
    )
    assert geo.extract(img).source == geo.SOURCE_SIDECAR


def test_write_sidecar_round_trips(sandbox):
    """Operator drag-to-place must survive a re-ingest of the same file."""
    img = _img(sandbox / "placed.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56], by="ops-chief")
    result = geo.extract(img)
    assert result.source == geo.SOURCE_SIDECAR
    assert result.bounds == [-122.40, 47.54, -122.38, 47.56]


# -------------------------------------------------------------- geo: plausible
@pytest.mark.parametrize(
    "bounds",
    [
        None,
        [0.0, 0.0, 0.001, 0.001],  # null island
        [-0.005, -0.005, 0.005, 0.005],  # centred on null island
        [-122.38, 47.54, -122.40, 47.56],  # inverted longitude
        [-122.40, 47.56, -122.38, 47.54],  # inverted latitude
        [0.0, 0.0, 1024.0, 768.0],  # untransformed pixel grid
        [-122.40, 47.54, -122.40, 47.56],  # zero width
        [-181.0, 47.54, -122.38, 47.56],  # out of range longitude
        [-122.40, -91.0, -122.38, 47.56],  # out of range latitude
        [-122.40, 47.54, -100.0, 47.56],  # wider than a tile can be
        [float("nan"), 47.54, -122.38, 47.56],
        [-122.40, 47.54, -122.38],  # wrong arity
        "not-bounds",
    ],
)
def test_plausible_rejects(bounds):
    assert geo._plausible(bounds) is False


def test_plausible_accepts_a_real_tile():
    assert geo._plausible([-122.40, 47.54, -122.38, 47.56]) is True


def test_implausible_sidecar_falls_through_to_needs_geo(sandbox):
    """A zeroed bbox must not masquerade as a location."""
    img = _img(sandbox / "zeroed.png")
    (sandbox / "zeroed.bounds.json").write_text(json.dumps({"bounds": [0, 0, 0, 0]}))
    result = geo.extract(img)
    assert result.needs_geo is True
    assert result.bounds is None


# ------------------------------------------------------------- ingest: happy path
def test_analyze_tile_runs_stages_in_the_pivoted_order(sandbox, stub_pipeline):
    img = _img(config.WATCH_DIR / "tile.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])

    rec = ingest.analyze_tile(img)

    # The gate lives in try_store, so store MUST be last.
    assert stub_pipeline["order"] == ["grade", "join", "store"]
    assert rec.status == "processed"
    assert rec.stored is True
    assert rec.needs_geo is False
    assert [b.id for b in rec.buildings] == ["fp-1", "fp-2"]
    assert rec.latency_ms >= 0
    assert rec.geo_source == geo.SOURCE_SIDECAR

    row = db.q1("SELECT * FROM tiles WHERE filename = ?", ("tile.png",))
    assert row["status"] == "processed"
    assert row["stored"] == 1
    assert row["geo_source"] == geo.SOURCE_SIDECAR
    b = db.q1("SELECT * FROM buildings WHERE footprint_id = 'fp-1'")
    assert b["label"] == "4200 SW Admiral Way"  # the join landed
    assert b["damage_class"] == 3
    # Stored images move to the analyzed folder, sidecar in tow.
    assert (config.ANALYZED_DIR / "tile.png").is_file()
    assert (config.ANALYZED_DIR / "tile.bounds.json").is_file()
    assert not img.exists()


def test_withheld_tile_still_contributes_buildings_and_moves_to_the_vault(
    sandbox, stub_pipeline
):
    """The pivot in one test: analyzed and ranked, never stored."""
    stub_pipeline["store"] = (False, "person signal: 3 detections")
    img = _img(config.WATCH_DIR / "person.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])

    rec = ingest.analyze_tile(img)

    assert rec.stored is False
    assert rec.withheld_reason == "person signal: 3 detections"
    assert rec.status == "processed"  # withholding is not an analysis outcome
    assert len(rec.buildings) == 2
    assert db.q1("SELECT COUNT(*) AS n FROM buildings")["n"] == 2
    assert (config.WITHHELD_DIR / "person.png").is_file()
    assert not (config.ANALYZED_DIR / "person.png").exists()

    # The log export is unauthenticated, so no withheld filename and no reason.
    payloads = [r["payload"] for r in db.q("SELECT payload FROM decision_log")]
    joined = " ".join(payloads)
    assert "person.png" not in joined
    assert "person signal" not in joined


def test_needs_geo_tile_is_accepted_not_dropped(sandbox, stub_pipeline):
    img = _img(config.WATCH_DIR / "nogeo.png")
    rec = ingest.analyze_tile(img)
    assert rec.status == "needs_geo"
    assert rec.needs_geo is True
    assert rec.bounds is None
    assert rec.buildings == []  # no transform, no georeferenced outline
    assert ingest.stages_of(rec).grade == "skipped"
    assert db.q1("SELECT COUNT(*) AS n FROM tiles")["n"] == 1


# ------------------------------------------------------------------ ingest: dedup
def test_byte_identical_second_file_is_deduped(sandbox, stub_pipeline):
    first = _img(config.WATCH_DIR / "a.png")
    geo.write_sidecar(first, [-122.40, 47.54, -122.38, 47.56])
    ingest.analyze_tile(first)
    assert stub_pipeline["order"] == ["grade", "join", "store"]

    second = config.WATCH_DIR / "b.png"
    second.write_bytes((config.ANALYZED_DIR / "a.png").read_bytes())
    rec = ingest.analyze_tile(second)

    # No second pass through the models.
    assert stub_pipeline["order"] == ["grade", "join", "store"]
    assert rec.dedup is True
    assert rec.filename == "a.png"
    assert db.q1("SELECT COUNT(*) AS n FROM tiles")["n"] == 1
    assert not second.exists()


def test_duplicate_of_a_withheld_tile_lands_in_the_vault(sandbox, stub_pipeline):
    stub_pipeline["store"] = (False, "person signal: 1 detections")
    first = _img(config.WATCH_DIR / "p1.png")
    ingest.analyze_tile(first)
    second = config.WATCH_DIR / "p2.png"
    second.write_bytes((config.WITHHELD_DIR / "p1.png").read_bytes())

    ingest.analyze_tile(second)

    assert (config.WITHHELD_DIR / "p2.png").is_file()
    assert not (config.ANALYZED_DIR / "p2.png").exists()


def test_archive_add_source_bypasses_dedup_so_the_gate_reruns(sandbox, stub_pipeline):
    """Gates 3 and 7: re-adding the same bytes through the archive button must
    re-run the gate rather than inherit a cached verdict."""
    first = _img(config.WATCH_DIR / "again.png")
    ingest.analyze_tile(first)
    stub_pipeline["order"].clear()

    second = config.WATCH_DIR / "again2.png"
    second.write_bytes((config.ANALYZED_DIR / "again.png").read_bytes())
    rec = ingest.analyze_tile(second, source="archive-add")

    assert "store" in stub_pipeline["order"]
    assert rec.dedup is False


# ----------------------------------------------------------- ingest: failures
def test_grading_failure_yields_a_record_not_an_exception(sandbox, stub_pipeline):
    stub_pipeline["grade_raises"] = True
    img = _img(config.WATCH_DIR / "broken.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])

    rec = ingest.analyze_tile(img)

    assert rec.status == "error"
    assert rec.buildings == []
    stages = ingest.stages_of(rec)
    assert "seg model unavailable" in stages.grade
    assert "grade" in (stages.failed() or "")
    # The storage decision still happened, because the gate is not optional.
    assert "store" in stub_pipeline["order"]
    assert db.q1("SELECT status FROM tiles WHERE filename = 'broken.png'")["status"] == "error"


def test_join_failure_does_not_fail_the_tile(sandbox, stub_pipeline):
    def boom(buildings, bounds):
        raise RuntimeError("svi geojson missing")

    sys.modules["app.datasets"].join = boom
    img = _img(config.WATCH_DIR / "nojoin.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])

    rec = ingest.analyze_tile(img)

    assert rec.status == "processed"
    assert len(rec.buildings) == 2
    assert "svi geojson missing" in ingest.stages_of(rec).join


def test_storage_writer_failure_fails_closed(sandbox, stub_pipeline):
    """If the writer breaks we withhold. "Stored" must never be the outcome of a
    check that did not complete."""

    def boom(image_path, tile_record, buildings):
        raise RuntimeError("archive writer exploded")

    sys.modules["app.archive"].try_store = boom
    img = _img(config.WATCH_DIR / "nostore.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])

    rec = ingest.analyze_tile(img)

    assert rec.stored is False
    assert rec.withheld_reason == "storage error"
    assert (config.WITHHELD_DIR / "nostore.png").is_file()


def test_missing_grading_module_is_a_stage_error_not_a_crash(sandbox, monkeypatch):
    """A partially built tree must still ingest: the lazy import failing is one
    stage's problem."""
    import types

    import app as app_pkg

    monkeypatch.setitem(sys.modules, "app.archive", types.ModuleType("app.archive"))
    sys.modules["app.archive"].try_store = lambda *a, **k: (True, None)
    monkeypatch.setattr(app_pkg, "archive", sys.modules["app.archive"], raising=False)
    # Patch the package attribute as well: `from . import grading` reads it, and
    # once a sibling test file has imported the real module the attribute wins
    # over any sys.modules entry.
    monkeypatch.setitem(sys.modules, "app.grading", None)
    monkeypatch.setattr(app_pkg, "grading", None, raising=False)

    img = _img(config.WATCH_DIR / "partial.png")
    # The tile needs bounds, otherwise it stops at needs_geo and never reaches
    # the grade stage this test is about.
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])
    rec = ingest.analyze_tile(img)

    assert rec.status == "error"
    assert ingest.stages_of(rec).grade != "ok"


def test_upload_door_and_poller_do_not_double_grade_one_file(sandbox, stub_pipeline):
    """The archive/upload door calls analyze_tile inline on a file that is still
    sitting in the watch folder. The poller must not grade it a second time."""
    img = _img(config.WATCH_DIR / "contended.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])

    entered = threading.Event()
    release = threading.Event()
    real_grade = sys.modules["app.grading"].outline_and_grade

    def slow_grade(image_path, bounds):
        entered.set()
        release.wait(5.0)
        return real_grade(image_path, bounds)

    sys.modules["app.grading"].outline_and_grade = slow_grade

    door = threading.Thread(
        target=lambda: ingest.analyze_tile(img, source="archive-add"), daemon=True
    )
    door.start()
    assert entered.wait(5.0)
    assert ingest.is_in_flight(img) is True
    assert ingest.in_flight() == 1

    # The poller sweeps twice while the door holds the file, and skips it.
    assert ingest.scan_once() == []
    assert ingest.scan_once() == []

    release.set()
    door.join(timeout=5.0)
    assert not door.is_alive()
    assert ingest.is_in_flight(img) is False
    assert stub_pipeline["order"].count("grade") == 1
    assert db.q1("SELECT COUNT(*) AS n FROM tiles")["n"] == 1


# ------------------------------------------------------------- ingest: watcher
def test_scan_once_settles_before_processing(sandbox, stub_pipeline):
    """First sweep records the size, second sweep processes. A frame still being
    written must never be graded."""
    img = _img(config.WATCH_DIR / "settling.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])

    assert ingest.scan_once() == []
    recs = ingest.scan_once()

    assert [r.filename for r in recs] == ["settling.png"]
    assert ingest.scan_once() == []  # seen, not reprocessed


def test_scan_once_ignores_sidecars_and_partials(sandbox, stub_pipeline):
    (config.WATCH_DIR / "x.bounds.json").write_text("{}")
    (config.WATCH_DIR / ".half.png.part").write_bytes(b"\x00\x01")
    assert ingest.scan_once() == []
    assert ingest.scan_once() == []
    assert stub_pipeline["order"] == []


def test_seen_is_added_after_success_so_failures_retry(sandbox, stub_pipeline, monkeypatch):
    img = _img(config.WATCH_DIR / "retry.png")
    calls = {"n": 0}

    def flaky(path, *, source="watch"):
        calls["n"] += 1
        raise RuntimeError("transient")

    monkeypatch.setattr(ingest, "analyze_tile", flaky)
    ingest.scan_once()  # settle read
    ingest.scan_once()
    ingest.scan_once()
    assert calls["n"] == 2  # retried rather than abandoned
    assert "retry.png" not in ingest._state.seen

    ingest.scan_once()
    assert "retry.png" in ingest._state.seen  # abandoned after MAX_ATTEMPTS
    assert img.exists()


def test_watch_loop_stops_on_the_event(sandbox, stub_pipeline):
    stop = threading.Event()
    t = threading.Thread(target=ingest.watch_loop, args=(stop, 0.01), daemon=True)
    t.start()
    img = _img(config.WATCH_DIR / "streamed.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])
    deadline = 5.0
    step = 0.02
    waited = 0.0
    while waited < deadline:
        if db.q1("SELECT COUNT(*) AS n FROM tiles")["n"]:
            break
        threading.Event().wait(step)
        waited += step
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()


# --------------------------------------------------------------- ingest: stats
def test_counts_and_p50_report_the_storage_split(sandbox, stub_pipeline):
    a = _img(config.WATCH_DIR / "s1.png", colour=(1, 2, 3))
    geo.write_sidecar(a, [-122.40, 47.54, -122.38, 47.56])
    ingest.analyze_tile(a)

    stub_pipeline["store"] = (False, "person signal: 2 detections")
    b = _img(config.WATCH_DIR / "s2.png", colour=(9, 9, 9))
    geo.write_sidecar(b, [-122.40, 47.54, -122.38, 47.56])
    ingest.analyze_tile(b)

    c = ingest.counts()
    assert c["tiles_analyzed"] == 2
    assert c["tiles_stored"] == 1
    assert c["tiles_withheld_from_storage"] == 1
    assert c["tiles_error"] == 0
    assert ingest.latency_p50() >= 0
    assert ingest.latency_percentile(95) >= ingest.latency_p50()


def test_wire_with_stages_carries_the_card_fields(sandbox, stub_pipeline):
    img = _img(config.WATCH_DIR / "card.png")
    geo.write_sidecar(img, [-122.40, 47.54, -122.38, 47.56])
    payload = ingest.wire_with_stages(ingest.analyze_tile(img))
    assert payload["stages"] == {"geo": "ok", "grade": "ok", "join": "ok", "store": "ok"}
    assert payload["stage_error"] is None
    assert payload["dedup"] is False
    assert payload["geo_source"] == geo.SOURCE_SIDECAR
    assert payload["buildings"][0]["class"] == 3


def test_confirmed_grade_survives_a_reflight(sandbox, stub_pipeline):
    """An operator's call outranks a second look from the same model."""
    a = _img(config.WATCH_DIR / "r1.png", colour=(5, 5, 5))
    geo.write_sidecar(a, [-122.40, 47.54, -122.38, 47.56])
    ingest.analyze_tile(a)
    db.run(
        "UPDATE buildings SET damage_class = 1, confirmed = 1, graded_by = 'operator:chief'"
        " WHERE footprint_id = 'fp-1'"
    )

    b = _img(config.WATCH_DIR / "r2.png", colour=(7, 7, 7))
    geo.write_sidecar(b, [-122.40, 47.54, -122.38, 47.56])
    ingest.analyze_tile(b)

    row = db.q1("SELECT * FROM buildings WHERE footprint_id = 'fp-1'")
    assert row["damage_class"] == 1
    assert row["graded_by"] == "operator:chief"
    assert row["source_tile"] == "r2.png"  # the fresh look still refreshed the join


# ------------------------------------------------------------------- downlink
def test_downlink_writes_to_watch_dir_and_never_ingests(sandbox, stub_pipeline):
    """CRITICAL: the downlink is a producer only. If it also called analyze_tile
    it would race the poller and report spurious stage-1 errors."""
    from app import downlink

    samples = sandbox / "samples"
    _img(samples / "sample1.png", colour=(11, 22, 33))
    _img(samples / "sample2.png", colour=(44, 55, 66))
    (samples / "sample1.bounds.json").write_text(
        json.dumps({"bounds": [-122.40, 47.54, -122.38, 47.56]})
    )

    downlink.start(f"simulate:{samples}")
    try:
        deadline = 5.0
        waited = 0.0
        while waited < deadline and downlink.state()["frames_received"] < 2:
            threading.Event().wait(0.05)
            waited += 0.05
        st = downlink.state()
        assert st["running"] is True
        assert st["mode"] == "simulate"
        assert st["frames_received"] >= 2
    finally:
        downlink.stop()

    assert downlink.state()["running"] is False
    delivered = sorted(p.name for p in config.WATCH_DIR.glob("dl_*.png"))
    assert len(delivered) >= 2
    # Nothing was analyzed: no models were called and no tile rows exist.
    assert stub_pipeline["order"] == []
    assert db.q1("SELECT COUNT(*) AS n FROM tiles")["n"] == 0
    # Sidecars travel with their frame so the watcher resolves geo.
    assert list(config.WATCH_DIR.glob("dl_*.bounds.json"))
    # No partials left behind for the poller to trip over.
    assert not list(config.WATCH_DIR.glob("*.part"))


def test_downlink_backlog_is_measured_and_drains(sandbox, stub_pipeline):
    """A looped replay dedups, so received-minus-ingested would climb forever and
    show a queue that is not there. Backlog counts real waiting files instead."""
    from app import downlink

    samples = sandbox / "loopsamples"
    _img(samples / "one.png", colour=(3, 3, 3))
    (samples / "one.bounds.json").write_text(
        json.dumps({"bounds": [-122.40, 47.54, -122.38, 47.56]})
    )

    downlink.start(f"simulate:{samples}")
    try:
        waited = 0.0
        while waited < 5.0 and downlink.state()["frames_received"] < 2:
            threading.Event().wait(0.05)
            waited += 0.05
        st = downlink.state()
        assert st["frames_received"] >= 2
        assert st["backlog"] == len(list(config.WATCH_DIR.glob("dl_*.png")))
    finally:
        downlink.stop()

    # Drain the folder through the watcher, then the backlog must read zero even
    # though frames_received stays high.
    ingest.scan_once()
    ingest.scan_once()
    st = downlink.state()
    assert st["backlog"] == 0
    assert st["frames_received"] >= 2
    assert st["frames_ingested"] >= 1


def test_downlink_rejects_a_source_that_is_not_simulate_or_rtsp(sandbox):
    from app import downlink

    for bad in ("", "http://example.com/stream", "simulate:/no/such/dir"):
        with pytest.raises(ValueError):
            downlink.start(bad)
    assert downlink.state()["running"] is False


def test_downlink_refuses_a_second_start(sandbox):
    from app import downlink

    samples = sandbox / "samples2"
    _img(samples / "only.png")
    downlink.start(f"simulate:{samples}")
    try:
        with pytest.raises(RuntimeError):
            downlink.start(f"simulate:{samples}")
    finally:
        downlink.stop()
