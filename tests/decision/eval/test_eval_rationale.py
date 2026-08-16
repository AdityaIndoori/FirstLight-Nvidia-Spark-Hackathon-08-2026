from backend.decision.eval.eval_rationale import RATIONALE_FIXTURE, evaluate_rationale_faithfulness


def test_fixture_has_at_least_ten_cases():
    assert len(RATIONALE_FIXTURE) >= 10


def test_fixture_covers_required_categories():
    categories = {case["category"] for case in RATIONALE_FIXTURE}
    required = {
        "correct", "wrong_priority", "wrong_staleness", "wrong_vulnerable_density",
        "wrong_doubt", "wrong_road_cutoff", "invented_facility", "wrong_damage_class",
    }
    assert required.issubset(categories)


def test_evaluate_returns_correct_shape():
    metric = evaluate_rationale_faithfulness()
    assert metric["name"] == "rationale_faithfulness"
    assert metric["sample_count"] == len(RATIONALE_FIXTURE)
    assert 0.0 <= metric["value"] <= 1.0


# the damage-class and facility contradiction checks (nano_client.py's
# _damage_class_contradicted / _facility_contradicted) closed the
# previously-documented gap: every fixture case, including
# invented_facility and wrong_damage_class, is now correctly classified.
def test_all_cases_are_correctly_classified():
    metric = evaluate_rationale_faithfulness()
    incorrect_cases = {c["case_id"] for c in metric["details"]["cases"] if not c["correct"]}
    assert incorrect_cases == set()
    assert metric["status"] == "pass"
    assert metric["value"] == 1.0


def test_valid_cases_produce_no_detected_violation():
    metric = evaluate_rationale_faithfulness()
    for case in metric["details"]["cases"]:
        if case["category"] == "correct":
            assert case["detected_violation"] is False


def test_deterministic_repeated_calls():
    m1 = evaluate_rationale_faithfulness()
    m2 = evaluate_rationale_faithfulness()
    assert m1 == m2
