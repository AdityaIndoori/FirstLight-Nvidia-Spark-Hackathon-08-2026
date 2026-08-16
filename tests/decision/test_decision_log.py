import sqlite3

import pytest

from backend.decision.decision_log import append_decision, get_connection, read_decisions


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "decision_log.db"))
    yield connection
    connection.close()


def test_append_creates_a_log_entry(conn):
    append_decision(conn, actor="operator:jsmith", action="grade-flip", entity_id="fp-1", details={}, ts=1000.0)

    rows = read_decisions(conn)
    assert len(rows) == 1


def test_two_appends_produce_two_distinct_rows(conn):
    id1 = append_decision(conn, "operator:a", "grade-flip", "fp-1", {}, ts=1000.0)
    id2 = append_decision(conn, "operator:b", "road-block", "road-9", {}, ts=1001.0)

    assert id1 != id2
    rows = read_decisions(conn)
    assert len(rows) == 2
    assert {row["id"] for row in rows} == {id1, id2}


def test_records_returned_in_chronological_insertion_order(conn):
    # Deterministic ts values, deliberately not monotonically written first-to-last
    # in a way that would matter: insertion order must win, independent of ts.
    append_decision(conn, "operator:a", "grade-flip", "fp-1", {"n": 1}, ts=5000.0)
    append_decision(conn, "operator:b", "road-block", "road-9", {"n": 2}, ts=1000.0)
    append_decision(conn, "operator:c", "plan-edit", "plan-3", {"n": 3}, ts=3000.0)

    rows = read_decisions(conn)
    assert [row["details"]["n"] for row in rows] == [1, 2, 3]
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)


def test_actor_is_preserved(conn):
    append_decision(conn, actor="operator:jsmith", action="grade-flip", entity_id="fp-1", details={}, ts=1000.0)

    row = read_decisions(conn)[0]
    assert row["actor"] == "operator:jsmith"


def test_action_is_preserved(conn):
    append_decision(conn, actor="operator:jsmith", action="road-unblock", entity_id="road-4", details={}, ts=1000.0)

    row = read_decisions(conn)[0]
    assert row["action"] == "road-unblock"


def test_entity_reference_is_preserved(conn):
    append_decision(conn, actor="operator:jsmith", action="plan-edit", entity_id="footprint-fp-42", details={}, ts=1000.0)

    row = read_decisions(conn)[0]
    assert row["entity_id"] == "footprint-fp-42"


def test_structured_details_round_trip_correctly(conn):
    details = {
        "from_class": 1,
        "to_class": 3,
        "nested": {"reason": "field survey", "units": [1, 2, 3]},
        "confirmed": True,
    }
    append_decision(conn, actor="operator:jsmith", action="grade-flip", entity_id="fp-1", details=details, ts=1000.0)

    row = read_decisions(conn)[0]
    assert row["details"] == details


def test_direct_sql_update_is_rejected_by_trigger(conn):
    append_decision(conn, "operator:a", "grade-flip", "fp-1", {"v": 1}, ts=1000.0)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE decision_log SET actor = 'tampered' WHERE id = 1")


def test_direct_sql_delete_is_rejected_by_trigger(conn):
    append_decision(conn, "operator:a", "grade-flip", "fp-1", {"v": 1}, ts=1000.0)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM decision_log WHERE id = 1")


def test_failed_update_does_not_alter_original_row(conn):
    append_decision(conn, "operator:a", "grade-flip", "fp-1", {"v": 1}, ts=1000.0)
    original = read_decisions(conn)[0]

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE decision_log SET actor = 'tampered', details = '{}' WHERE id = 1")

    assert read_decisions(conn)[0] == original


def test_failed_delete_does_not_remove_original_row(conn):
    append_decision(conn, "operator:a", "grade-flip", "fp-1", {"v": 1}, ts=1000.0)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM decision_log WHERE id = 1")

    assert len(read_decisions(conn)) == 1


def test_normal_insert_still_works_after_failed_mutation_attempts(conn):
    append_decision(conn, "operator:a", "grade-flip", "fp-1", {"v": 1}, ts=1000.0)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE decision_log SET actor = 'tampered' WHERE id = 1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM decision_log WHERE id = 1")

    append_decision(conn, "operator:b", "road-block", "road-9", {"v": 2}, ts=1001.0)

    rows = read_decisions(conn)
    assert len(rows) == 2
    assert [row["actor"] for row in rows] == ["operator:a", "operator:b"]
