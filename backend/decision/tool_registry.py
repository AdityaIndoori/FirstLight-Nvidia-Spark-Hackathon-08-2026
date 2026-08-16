"""Smallest possible tool boundary: name -> business function.

future NemoClaw tool call -> call_tool(name, ...) -> the existing application
function. Not a plugin framework, not an agent framework - just a lookup so a
NemoClaw adapter can discover and invoke write_flight_plan, write_export, and
fetch_context without this module knowing anything about NemoClaw.
"""

from backend.decision.agent_tools import fetch_context, write_export, write_flight_plan

TOOL_REGISTRY = {
    "write_flight_plan": write_flight_plan,
    "write_export": write_export,
    "fetch_context": fetch_context,
}


def list_tools() -> list:
    return sorted(TOOL_REGISTRY)


def call_tool(name: str, *args, **kwargs):
    if name not in TOOL_REGISTRY:
        raise KeyError(f"unknown tool: {name!r}")
    return TOOL_REGISTRY[name](*args, **kwargs)
