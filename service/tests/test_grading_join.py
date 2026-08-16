"""A3 + A4 fixture tests: damage grading and the vulnerability joins.

No network and no real models. vlm.chat is monkeypatched, so the "model" path is
exercised without a vLLM server and the stub path is exercised by making that
same patch fail. What is under test is the contract every other module codes
against: integer classes, [lng, lat] centroids, the exact graded_by strings, the
VL-call cap, and the privacy drop of owner-name columns.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, contracts, datasets, grading, vlm  # noqa: E402

BOUNDS = [-122.40, 47.55, -122.39, 47.56]


# ------------------------------------------------------------------- fixtures
def _poly(w: float, s: float, e: float, n: float) -> dict:
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def _fc(features: list[dict]) -> str:
    return json.dumps({"type": "FeatureCollection", "features": features})


def _tile(path: Path, size: tuple[int, int] = (480, 360)) -> Path:
    """A tile with a flat half and a noisy half, so the pixel statistic has signal."""
    img = Image.new("RGB", size, (110, 118, 112))
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0] // 2, size[0]):
            v = (x * 37 + y * 91) % 256  # deterministic, no RNG seed to leak between tests
            px[x, y] = (v, v, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


@pytest.fixture
def datadir(tmp_path, monkeypatch):
    """Point DATASET_DIR at tmp_path and clear every loader cache around the test."""
    d = tmp_path / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATASET_DIR", d)
    datasets.reset_cache()
    yield d
    datasets.reset_cache()


@pytest.fixture
def no_xview2(monkeypatch):
    """Force the footprint/grid outline tiers.

    The box may genuinely have the checkpoints, and a test that grades differently
    depending on what is installed proves nothing.
    """
    monkeypatch.setattr(grading, "_xview2_mask", lambda img: None)


@pytest.fixture
def fake_vl(monkeypatch):
    """Patch vlm.chat so the VL model path runs with no server.

    Returns the call log. Set log["fail"] = True to make the endpoint look wedged,
    which is how the stub path is exercised.
    """
    log: dict = {"calls": [], "fail": False, "cls": 3, "caption": "roof collapsed, debris field"}
    vlm.reset_breakers()

    def fake_chat(endpoint, messages, **kw):
        log["calls"].append({"endpoint": endpoint.name, "kw": kw, "messages": messages})
        if log["fail"]:
            return json.dumps(vlm._schema_stub(kw.get("schema"))), vlm.GRADE_HOW_STUB
        payload = json.dumps({"class": log["cls"], "caption": log["caption"]})
        return payload, vlm.GRADE_HOW_MODEL

    monkeypatch.setattr(vlm, "chat", fake_chat)
    yield log
    vlm.reset_breakers()


class FakeBuilding:
    """The mutable shape datasets.join fills in, per the frozen interface."""

    def __init__(self, centroid: list[float]):
        self.footprint_id = "b_test"
        self.centroid = centroid
        self.label = ""
        self.facility_near = None
        self.svi = None


# ------------------------------------------------------- grading: the contract
def test_grading_emits_integer_classes_and_lnglat_centroids(
    tmp_path, datadir, no_xview2, fake_vl
):
    """Damage class is an int 0-3 on the wire and coordinates are [lng, lat]."""
    graded = grading.outline_and_grade(_tile(tmp_path / "t.jpg"), BOUNDS)

    assert graded, "the grid tier must keep the pipeline from dead-ending"
    for b in graded:
        assert isinstance(b.cls, int) and not isinstance(b.cls, bool)
        assert b.cls in (0, 1, 2, 3)
        assert 0.0 <= b.conf <= 1.0
        lng, lat = b.centroid
        # Geometric, not string: the point must sit inside the tile bounds in
        # [lng, lat] order. Swapped order lands near the Somali coast and fails.
        assert BOUNDS[0] <= lng <= BOUNDS[2]
        assert BOUNDS[1] <= lat <= BOUNDS[3]
        assert b.geom["type"] in ("Polygon", "MultiPolygon")
        assert b.area_m2 > 0.0
        assert b.building().wire()["class"] == b.cls


def test_bounds_none_returns_empty_rather_than_guessing(tmp_path, datadir, no_xview2):
    """No transform means no placeable polygon, so ingest flags needs_geo instead."""
    assert grading.outline_and_grade(_tile(tmp_path / "t.jpg"), None) == []


def test_vl_call_cap_is_respected_and_the_remainder_is_stubbed(
    tmp_path, datadir, no_xview2, fake_vl
):
    """A 200-building tile must not take an hour, so over-cap buildings get the stub."""
    graded = grading.outline_and_grade(_tile(tmp_path / "t.jpg"), BOUNDS, vl_budget_override=3)

    assert len(graded) > 3, "the fixture needs more buildings than the cap to be meaningful"
    assert len(fake_vl["calls"]) == 3, "the cap bounds VL calls, not just the labels"

    modelled = [b for b in graded if b.graded_by == grading.GRADED_BY_VL]
    stubbed = [b for b in graded if b.graded_by == grading.GRADED_BY_STUB]
    assert len(modelled) == 3
    assert len(stubbed) == len(graded) - 3
    # The stub must not inherit a model caption: the archive would index a
    # sentence about a building nothing looked at.
    for b in stubbed:
        assert b.caption == vlm.STUB_CAPTION
    for b in modelled:
        assert b.caption == fake_vl["caption"]

    run = grading.last_run()
    assert run["vl_calls"] == 3
    assert run["model_graded"] == 3
    assert run["stub_graded"] == len(graded) - 3


def test_zero_budget_grades_everything_with_the_stub(tmp_path, datadir, no_xview2, fake_vl):
    """The cap is tunable to zero, which is the cut-list escape hatch for a slow box."""
    graded = grading.outline_and_grade(_tile(tmp_path / "t.jpg"), BOUNDS, vl_budget_override=0)

    assert graded
    assert not fake_vl["calls"]
    assert {b.graded_by for b in graded} == {grading.GRADED_BY_STUB}


def test_env_tunes_the_cap(monkeypatch):
    """A slow box drops the cap without a code change."""
    monkeypatch.setenv("FIRSTLIGHT_VL_CALLS_PER_TILE", "5")
    assert grading.vl_budget() == 5
    monkeypatch.setenv("FIRSTLIGHT_VL_CALLS_PER_TILE", "not-a-number")
    assert grading.vl_budget() == grading.DEFAULT_VL_CALLS_PER_TILE


def test_graded_by_strings_are_exactly_the_documented_values():
    """Section 7 freezes these. A typo here silently breaks C's provenance column."""
    assert grading.GRADED_BY_VL == "nemotron-vl"
    assert grading.GRADED_BY_STUB == "stub-pixelstat-v1"
    assert grading.GRADED_BY_XVIEW2 == "xview2"


def test_wedged_endpoint_grades_stub_and_never_raises(tmp_path, datadir, no_xview2, fake_vl):
    """A dead VL server costs labelled stubs, never a lost tile."""
    fake_vl["fail"] = True
    graded = grading.outline_and_grade(_tile(tmp_path / "t.jpg"), BOUNDS, vl_budget_override=4)

    assert graded
    assert {b.graded_by for b in graded} == {grading.GRADED_BY_STUB}
    assert grading.GRADED_BY_STUB in grading.model_version()


def test_off_contract_model_output_falls_back_to_stub(tmp_path, datadir, no_xview2, monkeypatch):
    """A class of 7 or a missing caption is not a grade, so it must not reach the wire."""
    monkeypatch.setattr(
        vlm, "chat", lambda ep, msgs, **kw: (json.dumps({"class": 7, "caption": ""}), "model")
    )
    graded = grading.outline_and_grade(_tile(tmp_path / "t.jpg"), BOUNDS, vl_budget_override=2)

    assert graded
    for b in graded:
        assert b.cls in (0, 1, 2, 3)
        assert b.graded_by == grading.GRADED_BY_STUB


def test_model_version_names_the_active_path(tmp_path, datadir, no_xview2, fake_vl):
    """The status bar must not be able to hide that outlines were synthetic."""
    grading.outline_and_grade(_tile(tmp_path / "t.jpg"), BOUNDS, vl_budget_override=2)
    version = grading.model_version()

    assert grading.GRADED_BY_VL in version
    assert grading.outline_source() in version
    assert "stub-graded" in version, "a partly stubbed tile must say so"


def test_footprint_outlines_win_over_the_synthetic_grid(tmp_path, datadir, no_xview2, fake_vl):
    """County footprints beat inference and certainly beat a grid."""
    (datadir / "footprints.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": _poly(-122.3980, 47.5540, -122.3975, 47.5545),
                    "properties": {"PIN": "1", "ADDR_FULL": "4512 35th Ave SW"},
                }
            ]
        )
    )
    datasets.reset_cache()
    graded = grading.outline_and_grade(_tile(tmp_path / "t.jpg"), BOUNDS, vl_budget_override=1)

    assert grading.outline_source() == grading.OUTLINE_FOOTPRINTS
    assert len(graded) == 1
    assert graded[0].footprint_id.startswith("fp_")


def test_pixel_mapping_round_trips(tmp_path):
    """The crop affine is the difference between grading a roof and grading a road."""
    x, y = grading.bounds_to_pixel(BOUNDS, 400, 200, -122.395, 47.555)
    assert x == pytest.approx(200.0)
    assert y == pytest.approx(100.0)
    lng, lat = grading.pixel_to_bounds(BOUNDS, 400, 200, x, y)
    assert lng == pytest.approx(-122.395, abs=1e-6)
    assert lat == pytest.approx(47.555, abs=1e-6)
    # North is row zero: the top-left pixel is the north-west corner.
    assert grading.pixel_to_bounds(BOUNDS, 400, 200, 0, 0) == pytest.approx([BOUNDS[0], BOUNDS[3]])


def test_tile_caption_prefers_a_real_model_caption(tmp_path, datadir, no_xview2, fake_vl):
    """A6 forbids a second VLM call, so the tile caption is picked, not generated."""
    graded = grading.outline_and_grade(_tile(tmp_path / "t.jpg"), BOUNDS, vl_budget_override=2)
    caption, by = grading.tile_caption(graded)

    assert caption == fake_vl["caption"]
    assert by == grading.GRADED_BY_VL

    fake_vl["fail"] = True
    stub_graded = grading.outline_and_grade(
        _tile(tmp_path / "t2.jpg"), BOUNDS, vl_budget_override=2
    )
    stub_caption, stub_by = grading.tile_caption(stub_graded)
    assert stub_by == grading.GRADED_BY_STUB
    assert vlm.STUB_CAPTION in stub_caption


def test_stub_caption_never_reads_as_an_observation():
    """An operator must be able to tell "nothing looked" from "nothing was damaged"."""
    assert "no model caption" in vlm.STUB_CAPTION
    assert not vlm.caption_mentions_person(vlm.STUB_CAPTION)


def test_caption_post_filter_catches_people_but_not_water():
    """A6: a caption is a second chance to catch what the detector missed."""
    assert vlm.caption_mentions_person("two people on the roof of a flooded house")
    assert vlm.caption_mentions_person("a child's jacket in the debris")
    assert not vlm.caption_mentions_person("two-storey wood structure, roof collapsed")
    # "body of water" is the terrain vocabulary the prompt asks for, not a person.
    assert not vlm.caption_mentions_person("standing water, a body of water beside the street")


# ------------------------------------------------------------- datasets: joins
def test_join_attaches_a_label_and_leaves_facility_none_when_nothing_is_near(datadir):
    """300 m is the radius. A hospital across town is not a proximity signal."""
    (datadir / "footprints.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": _poly(-122.4001, 47.5501, -122.3999, 47.5503),
                    "properties": {"ADDR_FULL": "4512 35th Ave SW"},
                }
            ]
        )
    )
    (datadir / "facilities.csv").write_text(
        "name,provider_type,lat,lng\nHarborview Medical Center,Hospital,47.6030,-122.3230\n"
    )
    datasets.reset_cache()

    building = FakeBuilding([-122.4000, 47.5502])
    datasets.join([building], BOUNDS)

    assert building.label == "4512 35th Ave SW"
    assert building.facility_near is None, "Harborview is kilometres away"
    assert building.svi == datasets.DEFAULT_SVI


def test_join_attaches_a_facility_inside_the_radius(datadir):
    (datadir / "facilities.csv").write_text(
        "name,provider_type,lat,lng\n"
        "Providence Mount St Vincent,Skilled Nursing Facility,47.5502,-122.4000\n"
    )
    datasets.reset_cache()

    building = FakeBuilding([-122.4000, 47.5504])
    datasets.join([building], BOUNDS)

    near = building.facility_near
    assert isinstance(near, contracts.FacilityNear)
    assert near.type in contracts.FACILITY_TYPES
    assert near.type == "nursing_home", "a skilled nursing facility is not a hospital"
    assert 0 <= near.dist_m <= int(datasets.FACILITY_RADIUS_M)
    assert near.wire()["dist_m"] == near.dist_m


def test_the_facility_radius_is_a_hard_boundary_at_300_m(datadir):
    """Geometric, not incidental: 300 m is a contract number, so probe both sides.

    A facility 250 m away is a real proximity signal an EMS planner acts on. One
    350 m away is a different block, and letting it through would inflate the
    vulnerability of every building in the county near any clinic.
    """
    # Latitude-only offsets, so metres are exact: one degree of latitude is fixed.
    inside_lat = 47.5500 + 250.0 / 110540.0
    outside_lat = 47.5500 + 350.0 / 110540.0
    (datadir / "facilities.csv").write_text(
        "name,provider_type,lat,lng\nMount St Vincent,Skilled Nursing,47.5500,-122.4000\n"
    )
    datasets.reset_cache()

    inside = datasets.facility_near([-122.4000, inside_lat])
    outside = datasets.facility_near([-122.4000, outside_lat])

    assert inside is not None
    assert inside.dist_m == pytest.approx(250, abs=2)
    assert outside is None, "a facility past the radius must not attach at all"


def test_road_relative_label_when_no_address_exists(datadir):
    """A footprint with no address still has to be dispatchable to."""
    (datadir / "roads.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-122.3960, 47.5520], [-122.3960, 47.5560]],
                    },
                    "properties": {"ST_NAME": "35th Ave SW"},
                }
            ]
        )
    )
    datasets.reset_cache()

    building = FakeBuilding([-122.3962, 47.5540])
    datasets.join([building], BOUNDS)

    assert building.label == "unnamed structure near 35th Ave SW"


def test_label_never_returns_a_raw_id(datadir):
    """An operator dispatches crews to streets, never to a footprint hash."""
    datasets.reset_cache()
    label = datasets.label_for([-122.395, 47.555])
    assert label
    assert "b_" not in label
    assert "fp_" not in label


def test_svi_of_the_containing_block_group(datadir):
    (datadir / "svi.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": _poly(-122.41, 47.54, -122.39, 47.56),
                    "properties": {"RPL_THEMES": 0.95},
                }
            ]
        )
    )
    datasets.reset_cache()

    assert datasets.svi_at([-122.40, 47.55]) == pytest.approx(0.95)
    # Outside every block group is a data gap, not a zero: a zero would delete the
    # building from the ranking, because vulnerability multiplies into priority.
    assert datasets.svi_at([-121.00, 47.55]) == datasets.DEFAULT_SVI
    assert datasets.svi_at(None) == datasets.DEFAULT_SVI


def test_suppressed_svi_rows_are_not_treated_as_zero(datadir):
    """CDC publishes -999 for suppressed block groups."""
    (datadir / "svi.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": _poly(-122.41, 47.54, -122.39, 47.56),
                    "properties": {"RPL_THEMES": -999},
                }
            ]
        )
    )
    datasets.reset_cache()
    assert datasets.svi_at([-122.40, 47.55]) == datasets.DEFAULT_SVI


def test_join_survives_a_corrupt_dataset_file(datadir):
    """A truncated download costs a label, never a tile."""
    (datadir / "footprints.geojson").write_text("{ this is not json")
    datasets.reset_cache()

    building = FakeBuilding([-122.40, 47.55])
    datasets.join([building], BOUNDS)  # must not raise

    assert building.label
    assert building.svi == datasets.DEFAULT_SVI


# --------------------------------------------------------- datasets: privacy
def test_owner_name_columns_are_dropped_from_loaded_parcels(datadir):
    """A4: the assessor join can surface owner names, so they die at load time.

    Checked on the loaded object, not on an API response: a query-time filter is
    one forgotten endpoint away from leaking, which is the whole point of doing
    this in the parser.
    """
    with (datadir / "parcels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["PIN", "SITEADDRESS", "OWNER", "OWNER_NAME", "TAXPAYER", "MAILING_ADDRESS", "lat", "lng"]
        )
        writer.writerow(
            ["5555", "4700 35th Ave SW", "Jane Q Public", "Jane Q Public", "Jane Q Public",
             "PO Box 1, Seattle WA", "47.5541", "-122.3959"]
        )
    datasets.reset_cache()

    loaded = datasets.parcels()
    assert loaded, "the fixture parcel must actually load"
    for parcel in loaded:
        keys = {k.lower() for k in parcel.props}
        assert not keys & {c.lower() for c in datasets.DROP_COLUMNS}
        blob = json.dumps(parcel.props)
        assert "Jane Q Public" not in blob
        assert "PO Box 1" not in blob
        # The join keys and the address survive: dropping identity must not
        # break the identity JOIN, which is a PIN, not a name.
        assert parcel.props["PIN"] == "5555"
        assert parcel.address == "4700 35th Ave SW"

    assert "OWNER" in datasets.dropped_columns_seen()
    assert "MAILING_ADDRESS" in datasets.dropped_columns_seen()


def test_owner_columns_are_dropped_from_every_set_not_just_parcels(datadir):
    """The scrub is in the shared parser, so footprints inherit it."""
    (datadir / "footprints.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": _poly(-122.4001, 47.5501, -122.3999, 47.5503),
                    "properties": {"owner_name": "Bob Roberts", "ADDR_FULL": "1 Main St"},
                }
            ]
        )
    )
    datasets.reset_cache()

    for f in datasets.footprints():
        assert "Bob Roberts" not in json.dumps(f.props)
        assert f.address == "1 Main St"


def test_drop_columns_names_the_assessor_fields():
    """Main cites this tuple in the privacy write-up, so it must be specific."""
    lower = {c.lower() for c in datasets.DROP_COLUMNS}
    for expected in ("owner", "owner_name", "taxpayer", "mailing_address"):
        assert expected in lower


# ----------------------------------------------------- datasets: vulnerability
def test_vulnerable_density_stays_within_zero_and_one(datadir):
    """It multiplies into priority, so an out-of-range value corrupts every rank."""
    cases = [
        (None, None),
        (0.0, None),
        (1.0, None),
        (-5.0, None),
        (99.0, None),
        (1.0, contracts.FacilityNear("Mount", "nursing_home", 0)),
        (0.95, contracts.FacilityNear("Mount", "dialysis", 10)),
        (0.5, contracts.FacilityNear("Harborview", "hospital", 300)),
        (0.5, {"name": "Mount", "type": "nursing_home", "dist_m": 150}),
        (0.5, {"name": "Mount", "type": "unknown-type", "dist_m": 0}),
    ]
    for svi_value, facility in cases:
        score = datasets.vulnerable_density_from(svi_value, facility)
        assert 0.0 <= score <= 1.0, (svi_value, facility, score)
        assert score >= datasets.VULN_FLOOR, "a zero would erase the building from the rank"

    building = FakeBuilding([-122.40, 47.55])
    building.svi = 0.9
    building.facility_near = contracts.FacilityNear("Mount", "nursing_home", 100)
    assert datasets.vulnerable_density(building) == datasets.vulnerable_density_from(
        0.9, building.facility_near
    ), "the one-arg form must delegate, not re-derive"


def test_facility_bump_scales_with_distance(datadir):
    """A facility at the radius edge adds nothing; one next door adds the full bump."""
    at_edge = datasets.vulnerable_density_from(
        0.4, contracts.FacilityNear("x", "nursing_home", int(datasets.FACILITY_RADIUS_M))
    )
    next_door = datasets.vulnerable_density_from(
        0.4, contracts.FacilityNear("x", "nursing_home", 0)
    )
    assert at_edge == pytest.approx(0.4)
    assert next_door > at_edge


def test_a_higher_svi_never_lowers_vulnerability(datadir):
    """Monotonic in SVI, or C's plain-English readout would contradict the number."""
    scores = [datasets.vulnerable_density_from(v / 10.0, None) for v in range(11)]
    assert scores == sorted(scores)


# --------------------------------------------------------- datasets: API shapes
def test_geojson_helpers_emit_valid_feature_collections(datadir):
    (datadir / "facilities.csv").write_text(
        "name,provider_type,lat,lng\nMount St Vincent,Skilled Nursing,47.5502,-122.4000\n"
    )
    (datadir / "roads.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-122.396, 47.552], [-122.396, 47.556]],
                    },
                    "properties": {"ST_NAME": "35th Ave SW"},
                }
            ]
        )
    )
    datasets.reset_cache()

    for collection in (datasets.facilities_geojson(), datasets.roads_geojson()):
        assert collection["type"] == "FeatureCollection"
        for feature in collection["features"]:
            assert feature["type"] == "Feature"
            assert feature["geometry"]
            # C's map keys the medical cross off properties.type and labels off name.
            assert "name" in feature["properties"]

    facility = datasets.facilities_geojson()["features"][0]
    assert facility["properties"]["type"] in contracts.FACILITY_TYPES


def test_footprints_in_clips_to_bounds(datadir):
    (datadir / "footprints.geojson").write_text(
        _fc(
            [
                {"type": "Feature", "geometry": _poly(-122.3999, 47.5501, -122.3997, 47.5503),
                 "properties": {}},
                {"type": "Feature", "geometry": _poly(-121.0, 46.0, -120.9, 46.1),
                 "properties": {}},
            ]
        )
    )
    datasets.reset_cache()

    assert len(datasets.footprints()) == 2
    assert len(datasets.footprints_in(BOUNDS)) == 1
    assert len(datasets.footprints_in(None)) == 2


def test_geocode_bbox_is_never_degenerate(datadir):
    """A north-south street has zero longitude width, which would filter to nothing."""
    (datadir / "roads.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-122.3960, 47.5520], [-122.3960, 47.5560]],
                    },
                    "properties": {"ST_NAME": "35th Ave SW"},
                }
            ]
        )
    )
    datasets.reset_cache()

    box = datasets.geocode("35th Ave SW")
    assert box is not None
    assert box[2] > box[0] and box[3] > box[1]
    # The buildings beside the street must fall inside it, which is the point.
    assert datasets.bbox_overlaps(box, (-122.3962, 47.5540, -122.3961, 47.5541))

    coords = datasets.geocode("47.558, -122.377")
    assert coords is not None and coords[0] < -122.377 < coords[2]
    assert datasets.geocode("Atlantis") is None
    assert datasets.geocode("") is None


def test_geom_area_subtracts_holes():
    """A courtyard is not floor area."""
    solid = datasets.geom_area_m2(_poly(-122.4000, 47.5500, -122.3990, 47.5510))
    with_hole = datasets.geom_area_m2(
        {
            "type": "Polygon",
            "coordinates": [
                [[-122.4000, 47.5500], [-122.3990, 47.5500], [-122.3990, 47.5510],
                 [-122.4000, 47.5510], [-122.4000, 47.5500]],
                [[-122.3997, 47.5503], [-122.3993, 47.5503], [-122.3993, 47.5507],
                 [-122.3997, 47.5507], [-122.3997, 47.5503]],
            ],
        }
    )
    assert solid > with_hole > 0.0
    assert datasets.geom_area_m2({"type": "Point", "coordinates": [-122.4, 47.55]}) == 0.0
    assert datasets.geom_area_m2(None) == 0.0


def test_facility_classification_rejects_non_care_types():
    """Only three types get the medical cross, so everything else must be dropped."""
    assert datasets.classify_facility("Skilled Nursing Facility") == "nursing_home"
    assert datasets.classify_facility("ESRD Dialysis Center") == "dialysis"
    assert datasets.classify_facility("Critical Access Hospital") == "hospital"
    assert datasets.classify_facility("cafe", "Corner Coffee") == ""


def test_reset_cache_makes_a_librarian_swap_visible(datadir):
    """The librarian swaps files atomically, so the loaders must be droppable."""
    (datadir / "roads.geojson").write_text(_fc([]))
    datasets.reset_cache()
    assert datasets.road_names() == []

    (datadir / "roads.geojson").write_text(
        _fc(
            [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-122.396, 47.552], [-122.396, 47.556]],
                    },
                    "properties": {"ST_NAME": "35th Ave SW"},
                }
            ]
        )
    )
    assert datasets.road_names() == [], "still cached until the cache is cleared"
    datasets.reset_cache()
    assert datasets.road_names() == ["35th Ave SW"]


# ------------------------------------------------------------- vlm: the syntax
def test_structured_output_uses_json_schema_never_guided_json(monkeypatch):
    """On this build a top-level guided_json is SILENTLY IGNORED, so it must not appear."""
    sent: dict = {}

    def fake_post(url, payload, timeout):
        sent.update(payload)
        return {"choices": [{"message": {"content": '{"class": 2, "caption": "roof hole"}'}}]}

    monkeypatch.setattr(vlm, "_post", fake_post)
    vlm.reset_breakers()

    text, how = vlm.chat(
        vlm.vl(), [{"role": "user", "content": "grade"}], schema=vlm.GRADE_SCHEMA, schema_name="g"
    )
    assert how == "model"
    assert json.loads(text)["class"] == 2
    assert "guided_json" not in sent
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "g", "schema": vlm.GRADE_SCHEMA},
    }


def test_enumerated_picks_use_guided_choice(monkeypatch):
    sent: dict = {}

    def fake_post(url, payload, timeout):
        sent.update(payload)
        return {"choices": [{"message": {"content": "2"}}]}

    monkeypatch.setattr(vlm, "_post", fake_post)
    vlm.reset_breakers()

    text, how = vlm.chat(vlm.lightning(), [{"role": "user", "content": "vote"}], choice=[0, 1, 2, 3])
    assert (text, how) == ("2", "model")
    assert sent["guided_choice"] == ["0", "1", "2", "3"]
    assert "guided_json" not in sent


def test_lightning_thinking_preamble_is_stripped(monkeypatch):
    """Lightning ignores /no_think and emits its preamble as plain content."""
    monkeypatch.setattr(
        vlm,
        "_post",
        lambda url, payload, timeout: {
            "choices": [{"message": {"content": "<think>hmm, maybe 3</think>```json\n{\"a\": 1}\n```"}}]
        },
    )
    vlm.reset_breakers()
    text, how = vlm.chat(vlm.lightning(), [{"role": "user", "content": "x"}])
    assert how == "model"
    assert json.loads(text) == {"a": 1}


def test_a_wedged_endpoint_returns_a_labelled_stub_and_never_raises(monkeypatch):
    """Technique 6: identical signature, labelled fallback, ingest never stalls."""
    def boom(url, payload, timeout):
        raise TimeoutError("wedged")

    monkeypatch.setattr(vlm, "_post", boom)
    vlm.reset_breakers()

    text, how = vlm.chat(vlm.vl(), [{"role": "user", "content": "x"}], choice=["0", "1", "2", "3"])
    assert how == "stub"
    assert text in ("0", "1", "2", "3")


def test_the_breaker_stops_paying_a_timeout_per_crop(monkeypatch):
    """A dead endpoint must cost the tile a bounded number of round trips, not one each."""
    attempts = {"n": 0}

    def boom(url, payload, timeout):
        attempts["n"] += 1
        raise TimeoutError("wedged")

    monkeypatch.setattr(vlm, "_post", boom)
    vlm.reset_breakers()

    for _ in range(10):
        _text, how = vlm.chat(vlm.vl(), [{"role": "user", "content": "x"}])
        assert how == "stub"
    assert attempts["n"] == vlm.BREAKER_TRIP, "after tripping, no further round trips"

    vlm.reset_breakers()
    _text, _how = vlm.chat(vlm.vl(), [{"role": "user", "content": "x"}])
    assert attempts["n"] == vlm.BREAKER_TRIP + 1, "a reset lets the endpoint heal"


def test_off_menu_guided_pick_is_rejected(monkeypatch):
    """Structured decoding is the defence against a hostile caption flipping a grade."""
    monkeypatch.setattr(
        vlm,
        "_post",
        lambda url, payload, timeout: {"choices": [{"message": {"content": "seventeen"}}]},
    )
    vlm.reset_breakers()
    text, how = vlm.chat(vlm.vl(), [{"role": "user", "content": "x"}], choice=["0", "1"])
    assert how == "stub"
    assert text == "0"


def test_stub_grade_is_deterministic_and_in_range(tmp_path):
    """Identical pixels must produce an identical grade, run to run."""
    path = _tile(tmp_path / "t.jpg")
    first = vlm.stub_grade(path)
    second = vlm.stub_grade(path)

    assert first == second
    assert first["class"] in (0, 1, 2, 3)
    assert 0.0 <= first["conf"] <= 1.0
    assert first["how"] == "stub"
    assert first["caption"] == vlm.STUB_CAPTION


def test_caption_and_grade_sends_the_image_as_a_data_url(tmp_path, monkeypatch):
    """The VL model is the only one that sees pixels, so the pixels must reach it."""
    seen: dict = {}

    def fake_chat(endpoint, messages, **kw):
        seen["messages"] = messages
        seen["kw"] = kw
        return json.dumps({"class": 1, "caption": "scattered debris on the roof"}), "model"

    monkeypatch.setattr(vlm, "chat", fake_chat)

    out = vlm.caption_and_grade(_tile(tmp_path / "t.jpg"), crop_box=(10, 10, 200, 150))
    assert out["class"] == 1
    assert out["how"] == "model"
    assert 0.0 < out["conf"] <= 1.0

    content = seen["messages"][1]["content"]
    image_part = next(p for p in content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert seen["kw"]["schema"] is vlm.GRADE_SCHEMA
    # The prompt must forbid describing people: that is the caption-channel control.
    system = seen["messages"][0]["content"].lower()
    assert "never mention people" in system


def test_grade_confidence_drops_when_grade_and_caption_disagree(tmp_path, monkeypatch):
    """Grade says minor, caption says collapsed: that contradiction must raise doubt.

    doubt falls back to 1 - grader_confidence before Lightning votes, so a constant
    confidence here would make the whole uncertainty column decorative.
    """
    path = _tile(tmp_path / "t.jpg")

    def answer(cls: int, caption: str):
        def fake_chat(endpoint, messages, **kw):
            return json.dumps({"class": cls, "caption": caption}), "model"

        monkeypatch.setattr(vlm, "chat", fake_chat)
        return vlm.caption_and_grade(path)["conf"]

    agreeing = answer(3, "structure destroyed, collapsed into rubble")
    contradicting = answer(1, "structure destroyed, collapsed into rubble")
    assert contradicting < agreeing
