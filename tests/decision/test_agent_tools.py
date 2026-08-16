import copy
import os
import socket
import urllib.request

import pytest

from backend.decision.agent_tools import (
    StubContextTransport,
    fetch_context,
    write_export,
    write_flight_plan,
)
from backend.decision.flight_client import build_deterministic_flight_plan
from backend.decision.flight_planner import validate_flight_plan
from backend.decision.tool_registry import TOOL_REGISTRY, call_tool, list_tools

PLANNING_INPUT = {
    "footprint_id": "fp-900",
    "centroid": [-122.4194, 37.7749],
    "area_radius_m": 150.0,
    "altitude_m_agl": 60.0,
    "line_spacing_m": 40.0,
}


def _valid_flight_plan():
    return build_deterministic_flight_plan(PLANNING_INPUT)


def _invalid_flight_plan():
    plan = _valid_flight_plan()
    for feature in plan["features"]:
        if feature["properties"]["role"] == "survey-path":
            del feature["properties"]["altitude_m_agl"]
    return plan


# --- write_flight_plan -------------------------------------------------

def test_valid_flight_plan_is_written_successfully(tmp_path):
    result = write_flight_plan(_valid_flight_plan(), data_root=str(tmp_path))

    assert result["success"] is True
    assert result["error"] is None
    assert os.path.isfile(result["path"])


def test_write_flight_plan_reuses_existing_validator(tmp_path):
    plan = _invalid_flight_plan()
    expected_errors = "; ".join(validate_flight_plan(plan))

    result = write_flight_plan(plan, data_root=str(tmp_path))

    assert result["success"] is False
    assert result["error"] == expected_errors


def test_invalid_flight_plan_is_rejected(tmp_path):
    result = write_flight_plan(_invalid_flight_plan(), data_root=str(tmp_path))

    assert result["success"] is False
    assert result["path"] is None
    assert not any(tmp_path.rglob("*.geojson"))


def test_flight_plan_output_stays_inside_data_root(tmp_path):
    result = write_flight_plan(_valid_flight_plan(), data_root=str(tmp_path))

    root_real = os.path.realpath(str(tmp_path))
    assert os.path.commonpath([root_real, result["path"]]) == root_real


def test_write_flight_plan_does_not_mutate_input(tmp_path):
    plan = _valid_flight_plan()
    snapshot = copy.deepcopy(plan)

    write_flight_plan(plan, data_root=str(tmp_path))

    assert plan == snapshot


# --- write_export --------------------------------------------------------

def test_normal_export_writes_inside_configured_root(tmp_path):
    result = write_export("report.txt", "hello", data_root=str(tmp_path))

    assert result["success"] is True
    root_real = os.path.realpath(str(tmp_path))
    assert os.path.commonpath([root_real, result["path"]]) == root_real
    with open(result["path"]) as f:
        assert f.read() == "hello"


def test_export_path_traversal_is_rejected(tmp_path):
    result = write_export("../escape.txt", "hello", data_root=str(tmp_path))

    assert result["success"] is False
    assert result["path"] is None
    assert not (tmp_path.parent / "escape.txt").exists()


def test_export_absolute_path_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / "absolute_escape.txt"
    result = write_export(str(outside), "hello", data_root=str(tmp_path))

    assert result["success"] is False
    assert result["path"] is None
    assert not outside.exists()


def test_export_identical_input_is_deterministic(tmp_path):
    first = write_export("same.txt", "same content", data_root=str(tmp_path))
    second = write_export("same.txt", "same content", data_root=str(tmp_path))

    assert first["success"] is True and second["success"] is True
    assert first["path"] == second["path"]


# --- fetch_context ---------------------------------------------------------

def test_fetch_context_invoked_through_injected_fake_transport():
    transport = StubContextTransport(allowed_hosts={"localhost"})
    result = fetch_context("http://localhost:8000/health", transport=transport)

    assert result["success"] is True
    assert result["url"] == "http://localhost:8000/health"


def test_fetch_context_never_makes_a_real_network_call(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    transport = StubContextTransport(allowed_hosts={"localhost"})
    result = fetch_context("https://exfiltrate.example.com/steal", transport=transport)

    assert result["success"] is False
    assert "exfiltrate.example.com" in result["error"]


def test_fetch_context_returns_fake_transport_result_correctly():
    class RecordingTransport:
        def fetch(self, url):
            return {"success": True, "body": f"canned:{url}", "error": None}

    result = fetch_context("http://localhost/anything", transport=RecordingTransport())

    assert result == {
        "url": "http://localhost/anything",
        "success": True,
        "body": "canned:http://localhost/anything",
        "error": None,
    }


# --- tool registry -----------------------------------------------------

def test_all_required_tool_names_are_discoverable():
    assert list_tools() == sorted(["write_flight_plan", "write_export", "fetch_context"])
    assert set(TOOL_REGISTRY) == {"write_flight_plan", "write_export", "fetch_context"}


def test_unknown_tool_name_is_rejected():
    with pytest.raises(KeyError):
        call_tool("delete_everything")


def test_calling_registered_tool_invokes_actual_implementation(tmp_path):
    via_registry = call_tool("write_export", "via_registry.txt", "content", data_root=str(tmp_path))
    direct = write_export("via_registry.txt", "content", data_root=str(tmp_path))

    assert via_registry["success"] is True
    assert via_registry["path"] == direct["path"]
    with open(via_registry["path"]) as f:
        assert f.read() == "content"
