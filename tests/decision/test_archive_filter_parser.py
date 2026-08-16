from datetime import datetime, timezone

import pytest

from backend.decision.archive_filter_parser import ArchiveSearchError, parse_structured_filters


def test_class_filter_parses_to_parameterized_sql():
    filters, residual = parse_structured_filters("class:3")
    assert filters == [("class_max = ?", 3)]
    assert residual == ""


def test_after_filter_parses_iso_datetime():
    filters, residual = parse_structured_filters("after:2026-08-15T06:00:00")
    assert len(filters) == 1
    fragment, value = filters[0]
    assert fragment == "captured_at >= ?"
    expected = datetime(2026, 8, 15, 6, 0, 0, tzinfo=timezone.utc).timestamp()
    assert value == pytest.approx(expected)


def test_after_filter_parses_bare_time_against_reference_date():
    filters, _ = parse_structured_filters("after:06:00")
    _, value = filters[0]
    expected = datetime(2026, 8, 15, 6, 0, 0, tzinfo=timezone.utc).timestamp()
    assert value == pytest.approx(expected)


def test_before_filter_works():
    filters, residual = parse_structured_filters("before:06:00")
    fragment, value = filters[0]
    assert fragment == "captured_at <= ?"
    expected = datetime(2026, 8, 15, 6, 0, 0, tzinfo=timezone.utc).timestamp()
    assert value == pytest.approx(expected)
    assert residual == ""


def test_key_true_and_false():
    filters, _ = parse_structured_filters("key:true")
    assert filters == [("key_evidence = ?", 1)]
    filters, _ = parse_structured_filters("key:false")
    assert filters == [("key_evidence = ?", 0)]


def test_needs_geo_true_and_false():
    filters, _ = parse_structured_filters("needs_geo:true")
    assert filters == [("needs_geo = ?", 1)]
    filters, _ = parse_structured_filters("needs_geo:false")
    assert filters == [("needs_geo = ?", 0)]


def test_sector_filter_supported():
    filters, residual = parse_structured_filters("sector:C")
    assert filters == [("sector = ?", "C")]
    assert residual == ""


def test_readme_example_combo_parses_all_three_tokens():
    filters, residual = parse_structured_filters("class:3 after:06:00 sector:C")
    fragments = {f for f, _ in filters}
    assert fragments == {"class_max = ?", "captured_at >= ?", "sector = ?"}
    assert residual == ""


def test_recognized_tokens_removed_from_residual_text():
    filters, residual = parse_structured_filters("buildings on fire class:3")
    assert filters == [("class_max = ?", 3)]
    assert residual == "buildings on fire"


def test_no_structured_tokens_leaves_residual_unchanged():
    filters, residual = parse_structured_filters("buildings on fire")
    assert filters == []
    assert residual == "buildings on fire"


# malformed structured filters fail safely -- never silently executed
def test_invalid_class_value_raises():
    with pytest.raises(ArchiveSearchError):
        parse_structured_filters("class:5")


def test_invalid_class_non_numeric_raises():
    with pytest.raises(ArchiveSearchError):
        parse_structured_filters("class:abc")


def test_invalid_key_value_raises():
    with pytest.raises(ArchiveSearchError):
        parse_structured_filters("key:maybe")


def test_invalid_needs_geo_value_raises():
    with pytest.raises(ArchiveSearchError):
        parse_structured_filters("needs_geo:maybe")


def test_invalid_after_value_raises():
    with pytest.raises(ArchiveSearchError):
        parse_structured_filters("after:not-a-date")


def test_unrecognized_token_prefix_stays_in_residual_not_an_error():
    filters, residual = parse_structured_filters("status:urgent buildings on fire")
    assert filters == []
    assert "status:urgent" in residual


# parameterized SQL: a hostile-looking sector value is returned as a plain
# parameter value, never concatenated into the SQL fragment string. (A
# structured token's value is a single non-whitespace run, see
# _TOKEN_PATTERN -- so a "space-containing" injection attempt can't even
# be expressed as one token; this checks the value that CAN be expressed.)
def test_sector_value_is_a_bound_parameter_not_interpolated():
    filters, _ = parse_structured_filters("sector:C';DROP_TABLE--")
    fragment, value = filters[0]
    assert fragment == "sector = ?"
    assert value == "C';DROP_TABLE--"
    assert "DROP_TABLE" not in fragment
