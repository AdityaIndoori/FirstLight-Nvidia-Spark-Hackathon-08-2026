import copy

from backend.decision.nano_client import NanoRationaleClient, StubNanoRationaleClient
from backend.decision.rationale import generate_rationale

RANK_ITEM_WITH_FACILITY = {
    "footprint_id": "fp-001",
    "label": "412 Elm St",
    "centroid": [-122.4194, 37.7749],
    "damage_class": 3,
    "confidence": 0.91,
    "confirmed": True,
    "graded_by": "operator:jsmith",
    "facility_near": {"name": "Riverside Clinic", "type": "clinic", "dist_m": 180},
    "inputs": {
        "staleness_h": 6.5,
        "vulnerable_density": 2.3,
        "doubt": 0.12,
        "road_cutoff": 1.8,
    },
    "priority": 24.19320,
    "rationale": "",
    "rationale_by": "nano",
}

RANK_ITEM_NO_FACILITY_NO_ROAD_CUTOFF = {
    "footprint_id": "fp-002",
    "label": "88 Oak Ave",
    "centroid": [-122.42, 37.77],
    "damage_class": 1,
    "confidence": 0.64,
    "confirmed": False,
    "graded_by": "xview2",
    "facility_near": None,
    "inputs": {
        "staleness_h": 3.0,
        "vulnerable_density": 0.8,
        "doubt": 0.35,
        "road_cutoff": None,
    },
    "priority": 0.84,
    "rationale": "",
    "rationale_by": "nano",
}


def test_identical_input_produces_identical_rationale():
    first = generate_rationale(copy.deepcopy(RANK_ITEM_WITH_FACILITY))
    second = generate_rationale(copy.deepcopy(RANK_ITEM_WITH_FACILITY))
    assert first == second


def test_rationale_references_only_supplied_facts():
    text = generate_rationale(RANK_ITEM_WITH_FACILITY)

    assert RANK_ITEM_WITH_FACILITY["label"] in text
    assert "destroyed" in text  # damage_class 3
    assert f"{RANK_ITEM_WITH_FACILITY['confidence']:.2f}" in text
    assert "operator-confirmed" in text
    assert f"{RANK_ITEM_WITH_FACILITY['priority']:.5f}" in text

    forbidden_terms = [
        "property value",
        "casualt",  # casualty / casualties
        "occupant",
        "resident",
        "people inside",
        "$",
    ]
    lowered = text.lower()
    for term in forbidden_terms:
        assert term not in lowered


def test_facility_information_included_when_present():
    text = generate_rationale(RANK_ITEM_WITH_FACILITY)
    facility = RANK_ITEM_WITH_FACILITY["facility_near"]
    assert facility["name"] in text
    assert facility["type"] in text
    assert str(facility["dist_m"]) in text


def test_facility_information_not_fabricated_when_absent():
    text = generate_rationale(RANK_ITEM_NO_FACILITY_NO_ROAD_CUTOFF)
    assert "clinic" not in text.lower()
    assert "hospital" not in text.lower()
    assert "facility" not in text.lower()


def test_road_access_concern_included_only_when_road_cutoff_present():
    with_cutoff = generate_rationale(RANK_ITEM_WITH_FACILITY)
    assert "road cutoff" in with_cutoff.lower()

    without_cutoff = generate_rationale(RANK_ITEM_NO_FACILITY_NO_ROAD_CUTOFF)
    assert "road cutoff" not in without_cutoff.lower()
    assert "road" not in without_cutoff.lower()


def test_priority_is_not_mutated():
    item = copy.deepcopy(RANK_ITEM_WITH_FACILITY)
    original_priority = item["priority"]
    generate_rationale(item)
    assert item["priority"] == original_priority


def test_input_rank_item_is_not_mutated():
    item = copy.deepcopy(RANK_ITEM_WITH_FACILITY)
    snapshot = copy.deepcopy(item)
    generate_rationale(item)
    assert item == snapshot


def test_no_invented_property_value_casualties_or_occupancy():
    for item in (RANK_ITEM_WITH_FACILITY, RANK_ITEM_NO_FACILITY_NO_ROAD_CUTOFF):
        text = generate_rationale(item).lower()
        for term in ["property value", "casualt", "occupant", "resident", "trapped", "injured"]:
            assert term not in text


def test_rationale_code_uses_interface_agnostic_of_backend():
    class FakeAlternateClient(NanoRationaleClient):
        """Stands in for a future real client to prove callers only need the interface."""

        def generate_rationale(self, rank_item: dict) -> str:
            return f"fake-backend rationale for {rank_item['footprint_id']}"

    stub_result = generate_rationale(RANK_ITEM_WITH_FACILITY)
    fake_result = generate_rationale(RANK_ITEM_WITH_FACILITY, client=FakeAlternateClient())

    assert stub_result != fake_result
    assert fake_result == "fake-backend rationale for fp-001"
    # Both calls used the exact same generate_rationale(rank_item) call shape.
    assert isinstance(StubNanoRationaleClient(), NanoRationaleClient)
    assert isinstance(FakeAlternateClient(), NanoRationaleClient)
