"""Regression tests for the damage-class and facility faithfulness checks
added to nano_client._faithfulness_violations (previously a documented B8
gap: "invented_facility" and "wrong_damage_class" were undetected).
"""

from backend.decision.nano_client import _faithfulness_violations

_BASE = {
    "footprint_id": "fp-001",
    "label": "412 Elm St",
    "centroid": [-122.4194, 37.7749],
    "confidence": 0.91,
    "confirmed": True,
    "graded_by": "operator:jsmith",
    "facility_near": None,
    "inputs": {"staleness_h": 6.5, "vulnerable_density": 2.3, "doubt": 0.12, "road_cutoff": 1.8},
    "priority": 24.19320,
    "rationale": "",
    "rationale_by": "nano",
}


def _rank_item(damage_class: int, facility_near=None) -> dict:
    return dict(_BASE, damage_class=damage_class, facility_near=facility_near)


# --------------------------------------------------------------------------
# DAMAGE CLASS
# --------------------------------------------------------------------------


def test_1_class_3_destroyed_passes():
    item = _rank_item(3)
    assert _faithfulness_violations("The structure is destroyed.", item) == []
    assert _faithfulness_violations("Destroyed building.", item) == []
    assert _faithfulness_violations("Class 3 damage confirmed.", item) == []


def test_2_class_3_minor_damage_fails():
    item = _rank_item(3)
    assert _faithfulness_violations("This shows only minor damage.", item)


def test_3_class_2_major_damage_passes():
    item = _rank_item(2)
    assert _faithfulness_violations("Major damage observed at the site.", item) == []
    assert _faithfulness_violations("Class 2 damage confirmed.", item) == []


def test_4_class_2_destroyed_fails():
    item = _rank_item(2)
    assert _faithfulness_violations("The building is destroyed.", item)


def test_5_class_0_no_visible_damage_passes():
    item = _rank_item(0)
    assert _faithfulness_violations("No visible damage was observed.", item) == []


def test_6_class_0_major_damage_fails():
    item = _rank_item(0)
    assert _faithfulness_violations("Major damage observed at the site.", item)


def test_7_omitting_damage_wording_is_not_a_violation():
    for damage_class in (0, 1, 2, 3):
        item = _rank_item(damage_class)
        assert _faithfulness_violations("Priority: 24.19320, confidence 0.91.", item) == []


def test_class_1_minor_damage_passes():
    item = _rank_item(1)
    assert _faithfulness_violations("Minor damage was noted.", item) == []
    assert _faithfulness_violations("Class 1 damage.", item) == []


def test_class_1_undamaged_fails():
    item = _rank_item(1)
    assert _faithfulness_violations("The building appears undamaged.", item)


def test_avoids_bare_damage_substring_false_positive():
    # "damage" alone (e.g. inside "vulnerable_density... drives urgency of
    # damage assessment") must never be treated as a class claim by itself.
    item = _rank_item(3)
    assert _faithfulness_violations("Ongoing damage assessment is recommended for this site.", item) == []


# --------------------------------------------------------------------------
# FACILITY
# --------------------------------------------------------------------------


def test_8_no_facility_and_no_mention_passes():
    item = _rank_item(2, facility_near=None)
    assert _faithfulness_violations("412 Elm St: priority 24.19320, confidence 0.91.", item) == []


def test_9_no_facility_but_invents_hospital_fails():
    item = _rank_item(2, facility_near=None)
    assert _faithfulness_violations("Located near Riverside Hospital.", item)


def test_10_dialysis_correct_name_passes():
    item = _rank_item(2, facility_near={"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0})
    assert _faithfulness_violations("Located near Riverside Dialysis Center.", item) == []


def test_11_dialysis_generic_mention_passes():
    item = _rank_item(2, facility_near={"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0})
    assert _faithfulness_violations("Located near a dialysis facility.", item) == []
    assert _faithfulness_violations("Near the dialysis center.", item) == []


def test_12_dialysis_labeled_as_hospital_fails():
    item = _rank_item(2, facility_near={"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0})
    assert _faithfulness_violations("Located near a hospital.", item)


def test_13_hospital_generic_mention_passes():
    item = _rank_item(2, facility_near={"name": "Seattle General Hospital", "type": "hospital", "dist_m": 50})
    assert _faithfulness_violations("Located near the hospital.", item) == []


def test_14_hospital_invented_different_name_fails():
    item = _rank_item(2, facility_near={"name": "Seattle General Hospital", "type": "hospital", "dist_m": 50})
    assert _faithfulness_violations("Located near Mercy Hospital.", item)


def test_15_facility_name_comparison_case_insensitive():
    item = _rank_item(2, facility_near={"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0})
    assert _faithfulness_violations("Located near RIVERSIDE DIALYSIS CENTER.", item) == []
    assert _faithfulness_violations("located near riverside dialysis center", item) == []


def test_16_harmless_ordinary_words_not_misclassified():
    item = _rank_item(3, facility_near=None)
    assert _faithfulness_violations(
        "412 Elm St: destroyed, confidence 0.91, priority 24.19320.", item
    ) == []


def test_facility_type_word_not_matched_inside_unrelated_word():
    # "hospitality" contains "hospital" as a raw substring -- must not
    # false-positive via naive substring matching.
    item = _rank_item(2, facility_near=None)
    assert _faithfulness_violations("The hospitality of local responders was notable.", item) == []


def test_facility_same_type_but_wrong_named_facility_of_that_type_fails():
    # "adjacent to Riverside Hospital" -- correct-sounding proximity
    # phrasing, wrong type AND wrong name, for a dialysis facility.
    item = _rank_item(2, facility_near={"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0})
    assert _faithfulness_violations("Adjacent to Riverside Hospital.", item)


def test_nursing_home_generic_mention_passes():
    item = _rank_item(
        2, facility_near={"name": "Providence Mount St. Vincent", "type": "nursing_home", "dist_m": 10}
    )
    assert _faithfulness_violations("Located near a nursing home.", item) == []


def test_nursing_home_labeled_as_dialysis_fails():
    item = _rank_item(
        2, facility_near={"name": "Providence Mount St. Vincent", "type": "nursing_home", "dist_m": 10}
    )
    assert _faithfulness_violations("Located near a dialysis center.", item)


# --------------------------------------------------------------------------
# COMBINED / EXISTING BEHAVIOR PRESERVED
# --------------------------------------------------------------------------


def test_existing_numeric_faithfulness_checks_still_work_alongside_new_ones():
    item = _rank_item(3, facility_near={"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0})
    # correct facility + correct class, but a fabricated road_cutoff unit
    text = "Destroyed structure near Riverside Dialysis Center; 1.8 miles blocked on the approach road."
    violations = _faithfulness_violations(text, item)
    assert violations
    assert any("road_cutoff" in v for v in violations)
