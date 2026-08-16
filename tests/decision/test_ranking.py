import copy

from backend.decision.ranking import rank_items


def make_item(footprint_id, priority, damage_class=0, confirmed=False, **overrides):
    item = {
        "footprint_id": footprint_id,
        "label": f"item-{footprint_id}",
        "centroid": [0.0, 0.0],
        "damage_class": damage_class,
        "confidence": 0.9,
        "confirmed": confirmed,
        "graded_by": "nano",
        "facility_near": None,
        "inputs": {
            "staleness_h": 1.0,
            "vulnerable_density": 1.0,
            "doubt": 1.0,
            "road_cutoff": None,
        },
        "priority": priority,
        "rationale": "because",
        "rationale_by": "nano",
    }
    item.update(overrides)
    return item


def test_ordinary_items_sort_by_priority_descending():
    a = make_item("a", priority=10)
    b = make_item("b", priority=50)
    c = make_item("c", priority=30)

    result = rank_items([a, b, c])

    assert [i["footprint_id"] for i in result] == ["b", "c", "a"]


def test_confirmed_damage_class_2_is_pinned():
    low_confirmed_severe = make_item("severe", priority=20, damage_class=2, confirmed=True)
    high_unconfirmed = make_item("plain", priority=100, damage_class=3, confirmed=False)

    result = rank_items([high_unconfirmed, low_confirmed_severe])

    assert [i["footprint_id"] for i in result] == ["severe", "plain"]
    assert result[0]["priority"] == 20
    assert result[1]["priority"] == 100


def test_confirmed_damage_class_3_is_pinned():
    low_confirmed_severe = make_item("severe", priority=5, damage_class=3, confirmed=True)
    high_unconfirmed = make_item("plain", priority=200, damage_class=3, confirmed=False)

    result = rank_items([high_unconfirmed, low_confirmed_severe])

    assert [i["footprint_id"] for i in result] == ["severe", "plain"]


def test_confirmed_damage_class_1_is_not_pinned():
    confirmed_minor = make_item("minor", priority=5, damage_class=1, confirmed=True)
    plain_high = make_item("plain", priority=100, damage_class=0, confirmed=False)

    result = rank_items([confirmed_minor, plain_high])

    # Neither qualifies as confirmed-severe (damage_class < 2), so ordinary
    # priority-descending order applies.
    assert [i["footprint_id"] for i in result] == ["plain", "minor"]


def test_unconfirmed_damage_class_3_is_not_pinned():
    unconfirmed_destroyed = make_item("destroyed", priority=5, damage_class=3, confirmed=False)
    plain_high = make_item("plain", priority=100, damage_class=0, confirmed=False)

    result = rank_items([unconfirmed_destroyed, plain_high])

    assert [i["footprint_id"] for i in result] == ["plain", "destroyed"]


def test_multiple_confirmed_severe_items_sort_among_themselves_by_priority_descending():
    s1 = make_item("s1", priority=10, damage_class=2, confirmed=True)
    s2 = make_item("s2", priority=90, damage_class=3, confirmed=True)
    s3 = make_item("s3", priority=50, damage_class=2, confirmed=True)
    plain = make_item("plain", priority=1000, damage_class=0, confirmed=False)

    result = rank_items([s1, plain, s2, s3])

    assert [i["footprint_id"] for i in result] == ["s2", "s3", "s1", "plain"]


def test_remaining_items_sort_among_themselves_by_priority_descending():
    severe = make_item("severe", priority=1, damage_class=2, confirmed=True)
    r1 = make_item("r1", priority=20, damage_class=0, confirmed=False)
    r2 = make_item("r2", priority=80, damage_class=1, confirmed=True)
    r3 = make_item("r3", priority=50, damage_class=3, confirmed=False)

    result = rank_items([r1, severe, r2, r3])

    assert [i["footprint_id"] for i in result] == ["severe", "r2", "r3", "r1"]


def test_priorities_identical_before_and_after_sorting():
    items = [
        make_item("a", priority=15, damage_class=2, confirmed=True),
        make_item("b", priority=200, damage_class=3, confirmed=False),
        make_item("c", priority=7, damage_class=1, confirmed=True),
    ]
    before = {item["footprint_id"]: item["priority"] for item in items}

    result = rank_items(items)

    after = {item["footprint_id"]: item["priority"] for item in result}
    assert before == after


def test_input_objects_are_not_mutated():
    items = [
        make_item("a", priority=15, damage_class=2, confirmed=True),
        make_item("b", priority=200, damage_class=3, confirmed=False),
    ]
    snapshot = copy.deepcopy(items)

    rank_items(items)

    assert items == snapshot


def test_repeated_sorting_of_identical_input_produces_identical_output():
    items = [
        make_item("a", priority=15, damage_class=2, confirmed=True),
        make_item("b", priority=200, damage_class=3, confirmed=False),
        make_item("c", priority=15, damage_class=0, confirmed=False),
        make_item("d", priority=90, damage_class=2, confirmed=True),
    ]

    first = [i["footprint_id"] for i in rank_items(items)]
    second = [i["footprint_id"] for i in rank_items(items)]

    assert first == second
