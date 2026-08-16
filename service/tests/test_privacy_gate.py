"""A2 privacy gate tests. The real weights are never loaded here.

`privacy_gate._detect` is the single seam that talks to the model, so every test
monkeypatches that one function and the rest of the gate runs for real: the slice
plan, the coordinate translation, the merge, the fail-safe and the verdict.

Each test names the claim it defends, because these are the assertions standing
behind a privacy statement made on stage.
"""
from __future__ import annotations

import threading
import time

import pytest
from PIL import Image

from app import config, privacy_gate


@pytest.fixture(autouse=True)
def _clean_gate_state():
    """The detector handle is module state. Leaking a loaded-or-failed flag
    between tests would let one test's monkeypatch decide another's verdict."""
    privacy_gate.reset()
    yield
    privacy_gate.reset()


def _img(tmp_path, name="tile.jpg", size=(800, 600), color=(40, 60, 40)):
    p = tmp_path / name
    Image.new("RGB", size, color).save(p, quality=60)
    return p


def _det(cls: int, name: str, conf: float, bbox=(10.0, 10.0, 40.0, 80.0)) -> dict:
    return {"cls": cls, "name": name, "conf": conf, "bbox": list(bbox)}


PERSON = next(iter(sorted(config.GATE_PERSON_CLASSES)))
VEHICLE = 3  # VisDrone car, deliberately not a person class


# ---------------------------------------------------------------- the verdict
def test_person_above_threshold_withholds(tmp_path, monkeypatch):
    """A person at or above the threshold means the image is never stored.

    The fixture confidence is derived from the configured threshold rather than
    written as a literal. A5 sets that threshold from measurement, so it moves, and
    a hardcoded 0.41 quietly stopped testing "above the threshold" the moment the
    measurement raised it to 0.50: the detection fell below the floor and the test
    asserted the opposite of its own name.
    """
    above = config.GATE_CONF + 0.1
    monkeypatch.setattr(
        privacy_gate, "_detect", lambda img, conf: [_det(PERSON, "pedestrian", above)]
    )
    v = privacy_gate.check(_img(tmp_path))
    assert v.store_ok is False
    assert len(v.person_detections) == 1
    assert v.detector_error is None
    assert v.withheld_reason() == "person signal: 1 detection"


def test_person_exactly_at_threshold_withholds(tmp_path, monkeypatch):
    """The threshold is inclusive. A gate that withholds at 0.2501 and clears at
    0.2500 publishes a recall number nobody can reproduce."""
    monkeypatch.setattr(
        privacy_gate,
        "_detect",
        lambda img, conf: [_det(PERSON, "pedestrian", config.GATE_CONF)],
    )
    assert privacy_gate.check(_img(tmp_path)).store_ok is False


def test_person_below_threshold_does_not_withhold(tmp_path, monkeypatch):
    """Below-threshold noise must clear, or every tile is withheld and the
    archive claim becomes vacuous. `_detect` filters at the model, so a stray
    low-confidence box arriving anyway is filtered again here."""
    monkeypatch.setattr(
        privacy_gate, "_detect", lambda img, conf: [_det(PERSON, "pedestrian", 0.05)]
    )
    v = privacy_gate.check(_img(tmp_path), conf=0.25)
    assert v.store_ok is True
    assert v.person_detections == []
    assert len(v.all_detections) == 1


def test_vehicle_only_stores(tmp_path, monkeypatch):
    """Vehicles must not trigger. An aerial tile of a parking lot is exactly the
    imagery the archive exists to hold."""
    monkeypatch.setattr(
        privacy_gate,
        "_detect",
        lambda img, conf: [
            _det(VEHICLE, "car", 0.9, (10, 10, 40, 80)),
            _det(VEHICLE, "car", 0.7, (500, 300, 540, 380)),
        ],
    )
    v = privacy_gate.check(_img(tmp_path))
    assert v.store_ok is True
    assert v.withheld_reason() is None
    assert len(v.all_detections) == 2


def test_person_among_vehicles_still_withholds(tmp_path, monkeypatch):
    """One person in a crowd of cars settles it. The verdict is a union, never a
    majority vote."""

    person_conf = config.GATE_CONF + 0.05

    def detect(img, conf):
        return [
            _det(VEHICLE, "car", 0.9),
            _det(PERSON, "people", person_conf, (5, 5, 20, 40)),
        ]

    monkeypatch.setattr(privacy_gate, "_detect", detect)
    assert privacy_gate.check(_img(tmp_path)).store_ok is False


def test_custom_conf_overrides_config(tmp_path, monkeypatch):
    """A5 sweeps the threshold, so `conf=` must actually move the verdict."""
    monkeypatch.setattr(
        privacy_gate, "_detect", lambda img, conf: [_det(PERSON, "pedestrian", 0.18)]
    )
    path = _img(tmp_path)
    assert privacy_gate.check(path, conf=0.30).store_ok is True
    assert privacy_gate.check(path, conf=0.15).store_ok is False


# ---------------------------------------------------------------- fail-safe
def test_detector_exception_withholds_with_reason(tmp_path, monkeypatch):
    """Doubt withholds. A detector fault costs a review click; a false clear
    costs the whole privacy claim."""

    def boom(img, conf):
        raise RuntimeError("CUDA kernel launch failed")

    monkeypatch.setattr(privacy_gate, "_detect", boom)
    v = privacy_gate.check(_img(tmp_path))
    assert v.store_ok is False
    assert v.detector_error is not None
    assert "CUDA kernel launch failed" in v.detector_error
    assert v.withheld_reason().startswith("detector error:")


def test_malformed_detector_output_withholds(tmp_path, monkeypatch):
    """A model that returns garbage must not be read as 'saw nobody'."""
    monkeypatch.setattr(privacy_gate, "_detect", lambda img, conf: ["not-a-dict"])
    v = privacy_gate.check(_img(tmp_path))
    assert v.store_ok is False
    assert v.detector_error is not None


def test_unreadable_file_withholds(tmp_path):
    """A truncated card dump is a withhold, not a traceback out of ingest."""
    bad = tmp_path / "truncated.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    v = privacy_gate.check(bad)
    assert v.store_ok is False
    assert v.detector_error is not None
    assert v.tiles_scanned == 0


def test_missing_weights_withholds_everything(tmp_path, monkeypatch):
    """The module-level unavailable state. `_detect` is NOT patched here: the
    real load path runs, finds no weights, and every check withholds. A service
    that boots without a detector must still be safe."""
    monkeypatch.setattr(config, "GATE_WEIGHTS", str(tmp_path / "no-such-weights.pt"))
    privacy_gate.reset()
    assert privacy_gate.available() is False
    assert "UNAVAILABLE" in privacy_gate.model_version()
    v = privacy_gate.check(_img(tmp_path))
    assert v.store_ok is False
    assert "weights missing" in v.detector_error


def test_check_never_raises_on_nonsense_input():
    """The archive writer calls this unguarded. It must always get a verdict."""
    for bad in (None, 42, "", "/definitely/not/a/file.jpg"):
        v = privacy_gate.check(bad)
        assert isinstance(v, privacy_gate.GateVerdict)
        assert v.store_ok is False
        assert v.detector_error is not None


def test_reasons_and_log_summary_carry_no_path(tmp_path, monkeypatch):
    """The decision-log export is unauthenticated, and the tile row's reason is
    rendered in the UI. Neither may name a withheld image."""
    secret = "victim-house-DCIM0421.jpg"

    def boom(img, conf):
        raise OSError(f"cannot decode {tmp_path / secret}")

    monkeypatch.setattr(privacy_gate, "_detect", boom)
    v = privacy_gate.check(_img(tmp_path, name=secret))
    blob = repr(v.summary()) + " " + v.withheld_reason()
    assert secret not in blob
    assert "DCIM0421" not in blob
    assert str(tmp_path) not in blob
    assert v.summary()["persons"] == 0
    assert "bbox" not in blob


def test_filename_with_spaces_is_scrubbed(tmp_path, monkeypatch):
    """The regex alone stops at whitespace, and 'Mrs Alvarez 3rd Ave.jpg' is
    exactly the kind of name a card dump carries. check() holds the source
    string, so it scrubs by value, not only by pattern."""
    secret = "Mrs Alvarez 3rd Ave.jpg"

    def boom(img, conf):
        raise OSError(f"decode failed for {tmp_path / secret} at offset 12")

    monkeypatch.setattr(privacy_gate, "_detect", boom)
    v = privacy_gate.check(_img(tmp_path, name=secret))
    blob = repr(v.summary()) + " " + v.withheld_reason()
    assert v.store_ok is False
    assert "Alvarez" not in blob
    assert "3rd Ave" not in blob
    assert "offset 12" in blob  # the diagnostic itself survives, just not the name


def test_summary_omits_per_detection_confidence(tmp_path, monkeypatch):
    """Counts and a channel name, not a fingerprint of one image."""
    monkeypatch.setattr(
        privacy_gate,
        "_detect",
        lambda img, conf: [_det(PERSON, "pedestrian", 0.8137, (11, 12, 33, 90))],
    )
    s = privacy_gate.check(_img(tmp_path)).summary()
    assert s["persons"] == 1 and s["channel"] == "pixels"
    assert "0.8137" not in repr(s)
    assert "11" not in str(s.get("detections"))
    assert set(s) == {
        "store_ok",
        "channel",
        "persons",
        "detections",
        "tiles_scanned",
        "took_ms",
        "conf",
        "tiled",
        "detector_error",
        "gate",
    }


# ---------------------------------------------------------------- tiled path
def test_small_image_runs_one_pass(tmp_path, monkeypatch):
    """No tiling below the minimum side. Six passes over a thumbnail buys
    nothing and costs ingest latency."""
    calls: list[tuple[int, int]] = []

    def detect(img, conf):
        calls.append(img.size)
        return []

    monkeypatch.setattr(privacy_gate, "_detect", detect)
    v = privacy_gate.check(_img(tmp_path, size=(900, 700)))
    assert v.tiles_scanned == 1
    assert v.tiled is False
    assert calls == [(900, 700)]


def test_tiling_activates_and_boxes_land_in_bounds(tmp_path, monkeypatch):
    """The A5 path. Every crop is inferred at crop resolution and every box is
    translated back into full-image pixels, so a detection found in the last
    crop still points at the right roof."""
    w, h = 3000, 2000
    seen: list[tuple[int, int]] = []

    def detect(img, conf):
        seen.append(img.size)
        # A box near the crop's own origin: if the caller forgot to add the crop
        # offset, every box would collapse onto the top-left corner.
        return [
            _det(PERSON, "pedestrian", config.GATE_CONF + 0.1, (12.0, 14.0, 30.0, 60.0))
        ]

    monkeypatch.setattr(privacy_gate, "_detect", detect)
    v = privacy_gate.check(_img(tmp_path, size=(w, h)))

    assert v.tiled is True
    assert v.tiles_scanned == len(privacy_gate._crop_boxes(w, h, True)) > 1
    assert len(seen) == v.tiles_scanned
    assert max(s[0] for s in seen) <= config.GATE_TILE
    assert max(s[1] for s in seen) <= config.GATE_TILE

    assert v.store_ok is False
    assert v.person_detections
    for d in v.person_detections:
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
    # Distinct crops, distinct origins: the offsets were really applied.
    assert len({tuple(d["bbox"]) for d in v.person_detections}) > 1


def test_tiling_covers_every_pixel(tmp_path, monkeypatch):
    """An uncovered strip is exactly where an unseen person ends up, so the
    slice plan must reach both far edges."""
    w, h = 4000, 2600
    boxes = privacy_gate._crop_boxes(w, h, True)
    assert min(b[0] for b in boxes) == 0 and min(b[1] for b in boxes) == 0
    assert max(b[2] for b in boxes) == w
    assert max(b[3] for b in boxes) == h
    xs = sorted({(b[0], b[2]) for b in boxes})
    for (_, prev_end), (next_start, _) in zip(xs, xs[1:]):
        assert next_start < prev_end  # overlapping, never merely adjacent


def test_tiling_can_be_disabled(tmp_path, monkeypatch):
    """The eval CLI compares tiled against single-pass, so the switch is real."""
    monkeypatch.setattr(privacy_gate, "_detect", lambda img, conf: [])
    v = privacy_gate.check(_img(tmp_path, size=(3000, 2000)), tiled=False)
    assert v.tiles_scanned == 1
    assert v.tiled is False


def test_overlap_duplicates_merge_to_one_person(tmp_path, monkeypatch):
    """Overlapping crops see the same person twice. A published count must mean
    people, not passes."""
    w, h = 3000, 2000
    boxes = privacy_gate._crop_boxes(w, h, True)
    assert len(boxes) > 1
    # One fixed full-image box, expressed in each crop's local frame, so every
    # crop reports the same person standing in the same place.
    fx1, fy1, fx2, fy2 = 1100.0, 400.0, 1160.0, 520.0
    order = iter(boxes)

    def detect(img, conf):
        ox, oy = next(order)[:2]
        return [_det(PERSON, "pedestrian", 0.5, (fx1 - ox, fy1 - oy, fx2 - ox, fy2 - oy))]

    monkeypatch.setattr(privacy_gate, "_detect", detect)
    v = privacy_gate.check(_img(tmp_path, size=(w, h)))
    assert v.store_ok is False
    assert len(v.person_detections) == 1
    assert v.person_detections[0]["bbox"] == [fx1, fy1, fx2, fy2]


def test_pil_image_input_is_not_closed(monkeypatch):
    """Ingest already has the tile decoded. Passing the handle in must not leave
    the caller with a closed image."""
    monkeypatch.setattr(privacy_gate, "_detect", lambda img, conf: [])
    img = Image.new("RGB", (700, 500), (10, 10, 10))
    v = privacy_gate.check(img)
    assert v.store_ok is True
    assert img.size == (700, 500)  # raises if the gate closed it


def test_grayscale_input_is_handled(monkeypatch):
    """Some thermal and SAR frames arrive single-channel."""
    monkeypatch.setattr(privacy_gate, "_detect", lambda img, conf: [])
    v = privacy_gate.check(Image.new("L", (700, 500), 90))
    assert v.store_ok is True
    assert v.detector_error is None


# ---------------------------------------------------------------- model state
def test_concurrent_checks_load_the_model_once(monkeypatch):
    """Ingest is threaded. Two loads of a multi-gigabyte detector on a box whose
    real constraint is memory bandwidth is a self-inflicted stall."""
    loads = []
    sentinel = object()

    def slow_load():
        loads.append(1)
        time.sleep(0.05)  # widen the race the double-check has to close
        return sentinel, None

    monkeypatch.setattr(privacy_gate, "_load", slow_load)
    privacy_gate.reset()
    threads = [threading.Thread(target=privacy_gate.available) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(loads) == 1
    assert privacy_gate._ensure_model() is sentinel


def test_model_version_names_weights_classes_and_threshold(tmp_path, monkeypatch):
    """The status strip prints this verbatim, so a judge can read the threshold
    off the screen and check it against the published recall number. Both
    branches are pinned here: a dev box without weights would otherwise let the
    format assertions pass by never running."""
    weights = tmp_path / "visdrone-yolov8x.pt"
    weights.write_bytes(b"stub")  # presence is all model_version() inspects
    monkeypatch.setattr(config, "GATE_WEIGHTS", str(weights))
    privacy_gate.reset()
    s = privacy_gate.model_version()
    assert "visdrone-yolov8x.pt" in s
    assert f"conf>={float(config.GATE_CONF):.2f}" in s
    assert "person classes" in s
    assert f"tiled {int(config.GATE_TILE)}px" in s

    monkeypatch.setattr(config, "GATE_WEIGHTS", str(tmp_path / "gone.pt"))
    privacy_gate.reset()
    assert privacy_gate.model_version() == "gone.pt UNAVAILABLE, withholding all storage"


def test_is_person_class_tolerates_junk():
    assert privacy_gate.is_person_class({"cls": PERSON}) is True
    assert privacy_gate.is_person_class({"cls": VEHICLE}) is False
    assert privacy_gate.is_person_class({}) is False
    assert privacy_gate.is_person_class({"cls": "pedestrian"}) is False


def test_concurrent_checks_never_share_the_model_mid_inference(tmp_path, monkeypatch):
    """Two uploads at once must not corrupt the detector.

    Ultralytics mutates the model during predict() - it fuses layers on the first
    pass - so an unsynchronised handle raced. Measured on the box: six concurrent
    uploads raised "'Conv' object has no attribute 'bn'" on two tiles, and because
    the gate fails CLOSED those two were withheld as detector errors. Working
    imagery silently became unstorable, which is the worst shape this bug could
    take: it looks like the privacy gate doing its job.

    The fake detector below asserts the invariant directly - it fails if a second
    caller is inside predict() while the first is still there.
    """
    inside = 0
    overlapped = False
    guard = threading.Lock()

    def fake_detect(image, conf):
        nonlocal inside, overlapped
        with guard:
            inside += 1
            if inside > 1:
                overlapped = True
        try:
            time.sleep(0.02)
            return []
        finally:
            with guard:
                inside -= 1

    monkeypatch.setattr(privacy_gate, "_detect", fake_detect)
    path = _img(tmp_path, "race.jpg")

    errors: list[str] = []

    def run():
        try:
            verdict = privacy_gate.check(path)
            # A clean image must clear. A detector error here would mean the gate
            # turned a storable tile into a withheld one under nothing but load.
            if not verdict.store_ok:
                errors.append(verdict.withheld_reason or "refused")
        except Exception as exc:  # noqa: BLE001 - the point is that this cannot happen
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent gate checks failed: {errors}"
    assert not overlapped, "two threads were inside the detector at once"
