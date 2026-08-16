from backend.decision.eval.eval_tags import TAG_FIXTURE, _precision_recall, evaluate_tag_precision_recall_offline


def test_fixture_has_at_least_ten_captions():
    assert len(TAG_FIXTURE) >= 10


def test_evaluate_returns_measured_with_precision_and_recall():
    metric = evaluate_tag_precision_recall_offline()
    assert metric["status"] == "measured"
    assert 0.0 <= metric["value"]["precision"] <= 1.0
    assert 0.0 <= metric["value"]["recall"] <= 1.0


def test_no_prohibited_tags():
    metric = evaluate_tag_precision_recall_offline()
    assert metric["details"]["prohibited_tags_total"] == 0


def test_no_grade_mutation():
    metric = evaluate_tag_precision_recall_offline()
    assert metric["details"]["grade_mutation_detected"] is False


def test_deterministic_repeated_calls():
    m1 = evaluate_tag_precision_recall_offline()
    m2 = evaluate_tag_precision_recall_offline()
    assert m1 == m2


# precision/recall arithmetic edge cases
def test_precision_recall_perfect_match():
    p, r = _precision_recall(["fire", "smoke"], {"fire", "smoke"})
    assert p == 1.0 and r == 1.0


def test_precision_recall_partial_overlap():
    p, r = _precision_recall(["fire", "debris"], {"fire", "smoke"})
    assert p == 0.5
    assert r == 0.5


def test_precision_recall_empty_returned_and_empty_expected_is_perfect():
    p, r = _precision_recall([], set())
    assert p == 1.0 and r == 1.0


def test_precision_recall_empty_returned_nonempty_expected():
    p, r = _precision_recall([], {"fire"})
    assert p == 0.0  # no tags returned at all -> precision undefined-as-zero by convention
    assert r == 0.0


def test_precision_recall_nonempty_returned_empty_expected():
    p, r = _precision_recall(["fire"], set())
    assert p == 0.0
    assert r == 0.0


def test_normalization_case_and_whitespace_insensitive():
    p, r = _precision_recall(["  Fire  ", "SMOKE"], {"fire", "smoke"})
    assert p == 1.0 and r == 1.0
