import pytest

from backend.decision.scoring import calculate_priority


def test_factors_rounded_before_multiplication():
    # Unrounded product would differ from the rounded-first product,
    # proving each factor is rounded to 3 decimals before multiplying.
    staleness_h = 2.34567
    vulnerable_density = 1.23456
    doubt = 0.98765

    unrounded_product = staleness_h * vulnerable_density * doubt
    rounded_first_product = round(round(staleness_h, 3) * round(vulnerable_density, 3) * round(doubt, 3), 5)

    result = calculate_priority(staleness_h, vulnerable_density, doubt)

    assert result != round(unrounded_product, 5)
    assert result == rounded_first_product


def test_final_priority_rounded_to_five_decimals():
    result = calculate_priority(1.0001, 3.00009, 0.33333)
    # round(1.0001,3)=1.0, round(3.00009,3)=3.0, round(0.33333,3)=0.333
    # 1.0 * 3.0 * 0.333 = 0.999
    assert result == round(0.999, 5)
    assert result == round(result, 5)


def test_road_cutoff_none_behaves_as_multiplier_one():
    with_none = calculate_priority(2.0, 1.5, 0.5, road_cutoff=None)
    with_one = calculate_priority(2.0, 1.5, 0.5, road_cutoff=1)
    assert with_none == with_one
    assert with_none == round(round(2.0, 3) * round(1.5, 3) * round(0.5, 3), 5)


def test_road_cutoff_greater_than_one_increases_priority():
    baseline = calculate_priority(2.0, 1.5, 0.5, road_cutoff=None)
    boosted = calculate_priority(2.0, 1.5, 0.5, road_cutoff=2.0)
    assert boosted > baseline
    assert boosted == round(baseline * 2.0, 5)


def test_road_cutoff_less_than_one_is_rejected():
    with pytest.raises(ValueError):
        calculate_priority(2.0, 1.5, 0.5, road_cutoff=0.5)


def test_road_cutoff_less_than_one_negative_is_rejected():
    with pytest.raises(ValueError):
        calculate_priority(2.0, 1.5, 0.5, road_cutoff=-1)


def test_repeated_calls_with_identical_inputs_are_identical():
    results = {calculate_priority(3.14159, 2.71828, 0.6667, road_cutoff=1.5) for _ in range(50)}
    assert len(results) == 1


def test_property_value_is_not_an_accepted_argument():
    with pytest.raises(TypeError):
        calculate_priority(2.0, 1.5, 0.5, property_value=250000)
