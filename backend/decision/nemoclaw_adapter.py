"""NemoClaw adapter: translates a future NemoClaw tool invocation into a call
against the existing FIRST LIGHT tool registry (tool_registry.py).

    Nemotron Nano -> NemoClaw agent -> [this adapter] -> tool_registry
                                                        -> write_flight_plan / write_export / fetch_context
                                                        -> OpenShell enforces at runtime

This is the ONLY layer that changes when the real NemoClaw runtime replaces the
placeholder (tool_name, arguments) shape below with its actual tool-call
mechanism. It contains no FIRST LIGHT business logic (that's agent_tools.py),
no model prompting (that's nano_client.py / flight_client.py), and no security
policy (that's OpenShell, later, out-of-process). It does not itself make any
network call.

The (tool_name, arguments) shape here is an internal placeholder for "however
NemoClaw hands us a tool call" - not a new public wire contract.
"""

from backend.decision.tool_registry import TOOL_REGISTRY, call_tool

SUPPORTED_TOOLS = tuple(TOOL_REGISTRY)


def invoke(tool_name: str, arguments: dict):
    """Invoke tool_name with arguments (passed through as keyword arguments,
    unmodified) and return exactly what the underlying FIRST LIGHT tool returns.
    """
    if tool_name not in SUPPORTED_TOOLS:
        raise KeyError(f"unknown tool: {tool_name!r}")
    return call_tool(tool_name, **arguments)
