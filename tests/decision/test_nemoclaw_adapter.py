import copy
import os
import socket
import urllib.request
from unittest.mock import MagicMock

import pytest

from backend.decision import agent_tools, tool_registry
from backend.decision.agent_tools import StubContextTransport
from backend.decision.flight_client import build_deterministic_flight_plan
from backend.decision.nemoclaw_adapter import SUPPORTED_TOOLS, invoke

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


# --- 1-3: adapter exposes exactly the three tool names ---------------------

def test_adapter_exposes_write_flight_plan():
    assert "write_flight_plan" in SUPPORTED_TOOLS


def test_adapter_exposes_write_export():
    assert "write_export" in SUPPORTED_TOOLS


def test_adapter_exposes_fetch_context():
    assert "fetch_context" in SUPPORTED_TOOLS


def test_adapter_exposes_exactly_these_three_and_no_more():
    assert set(SUPPORTED_TOOLS) == {"write_flight_plan", "write_export", "fetch_context"}


# --- 4-6: invoking each tool through the adapter reaches the real thing ----

def test_invoke_write_flight_plan_reaches_underlying_tool(tmp_path):
    result = invoke("write_flight_plan", {"flight_plan": _valid_flight_plan(), "data_root": str(tmp_path)})

    assert result["success"] is True
    assert os.path.isfile(result["path"])


def test_invoke_write_export_reaches_underlying_tool(tmp_path):
    result = invoke("write_export", {"export_name": "report.txt", "content": "hello", "data_root": str(tmp_path)})

    assert result["success"] is True
    with open(result["path"]) as f:
        assert f.read() == "hello"


def test_invoke_fetch_context_reaches_underlying_tool():
    transport = StubContextTransport(allowed_hosts={"localhost"})
    result = invoke("fetch_context", {"url": "http://localhost:8000/health", "transport": transport})

    assert result["success"] is True
    assert result["url"] == "http://localhost:8000/health"


# --- 7: arguments arrive unchanged at the underlying tool -------------------

def test_arguments_arrive_unchanged_at_underlying_tool(monkeypatch, tmp_path):
    spy = MagicMock(return_value={"success": True, "path": "unused", "error": None})
    monkeypatch.setitem(tool_registry.TOOL_REGISTRY, "write_export", spy)

    arguments = {"export_name": "report.txt", "content": "hello", "data_root": str(tmp_path)}
    original_snapshot = copy.deepcopy(arguments)

    invoke("write_export", arguments)

    spy.assert_called_once_with(**original_snapshot)
    assert arguments == original_snapshot  # invoke() did not mutate the caller's dict


# --- 8: unknown tool name is rejected ---------------------------------------

def test_unknown_tool_name_is_rejected():
    with pytest.raises(KeyError):
        invoke("delete_everything", {})


# --- 9: application-level failure propagates cleanly, not transformed ------

def test_tool_failure_propagates_cleanly(tmp_path):
    plan = _invalid_flight_plan()
    direct_result = agent_tools.write_flight_plan(plan, data_root=str(tmp_path))

    adapter_result = invoke("write_flight_plan", {"flight_plan": plan, "data_root": str(tmp_path)})

    assert adapter_result == direct_result
    assert adapter_result["success"] is False
    assert adapter_result["error"]  # concrete error text, not swallowed or replaced


# --- 10: adapter itself makes no real external network access --------------

def test_adapter_makes_no_real_network_access(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    transport = StubContextTransport(allowed_hosts={"localhost"})
    result = invoke("fetch_context", {"url": "https://exfiltrate.example.com/steal", "transport": transport})

    assert result["success"] is False
    assert "exfiltrate.example.com" in result["error"]


# --- 11: the adapter required no changes to the existing tool implementations

def test_adapter_wraps_existing_tools_without_modifying_them():
    assert tool_registry.TOOL_REGISTRY["write_flight_plan"] is agent_tools.write_flight_plan
    assert tool_registry.TOOL_REGISTRY["write_export"] is agent_tools.write_export
    assert tool_registry.TOOL_REGISTRY["fetch_context"] is agent_tools.fetch_context
    # The adapter's tool set is derived from the registry, not redefined.
    assert set(SUPPORTED_TOOLS) == set(tool_registry.TOOL_REGISTRY)
