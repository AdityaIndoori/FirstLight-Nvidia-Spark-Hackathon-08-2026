"""Application tool functions the future NemoClaw agent will register and call.

These are plain business functions, callable directly in tests today. Nothing
here is OpenShell: filesystem containment (deny reads/writes outside `./data`)
and network egress policy are enforced out-of-process by OpenShell later, per
the README security design. This module only does ordinary application-level
path validation so obviously unsafe behavior isn't left lying around while
that real enforcement doesn't exist yet.

No NemoClaw, no OpenShell, no vLLM, no real network calls.
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod

from backend.decision.flight_planner import validate_flight_plan


def _resolve_safe_path(root: str, relative_name: str) -> str:
    """Resolve relative_name under root, raising ValueError if it would escape root."""
    if os.path.isabs(relative_name):
        raise ValueError(f"absolute paths are not allowed: {relative_name!r}")

    root_real = os.path.realpath(root)
    candidate_real = os.path.realpath(os.path.join(root_real, relative_name))

    if os.path.commonpath([root_real, candidate_real]) != root_real:
        raise ValueError(f"path escapes data root: {relative_name!r}")

    return candidate_real


def write_flight_plan(flight_plan: dict, data_root: str) -> dict:
    """Validate flight_plan with the existing FlightPlan validator, then write it
    under data_root/flight_plans/ using a deterministic content-addressed filename.
    """
    errors = validate_flight_plan(flight_plan)
    if errors:
        return {"success": False, "path": None, "error": "; ".join(errors)}

    serialized = json.dumps(flight_plan, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    filename = f"flight_plan_{digest}.geojson"

    try:
        target_path = _resolve_safe_path(os.path.join(data_root, "flight_plans"), filename)
    except ValueError as exc:
        return {"success": False, "path": None, "error": str(exc)}

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write(serialized)

    return {"success": True, "path": target_path, "error": None}


def write_export(export_name: str, content: str, data_root: str) -> dict:
    """Write content under data_root/exports/export_name. Rejects any path that
    would escape data_root (../ traversal, absolute paths).
    """
    try:
        target_path = _resolve_safe_path(os.path.join(data_root, "exports"), export_name)
    except ValueError as exc:
        return {"success": False, "path": None, "error": str(exc)}

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write(content)

    return {"success": True, "path": target_path, "error": None}


class ContextTransport(ABC):
    """Boundary the fetch_context tool calls through.

    application/agent code calls fetch_context(url, transport) without knowing
    whether the destination is ultimately permitted. Right now the only
    implementation is StubContextTransport, a fully fake, deterministic fixture
    that never makes a real network call. The real runtime later sits OpenShell
    outside this process to enforce the actual egress policy; this interface is
    simply what that decision plugs into without changing fetch_context's callers.
    """

    @abstractmethod
    def fetch(self, url: str) -> dict:
        """Return {"success": bool, "body": str or None, "error": str or None}."""
        raise NotImplementedError


class StubContextTransport(ContextTransport):
    """Deterministic test fixture. Never contacts the network.

    Not a stand-in for OpenShell policy — just enough to exercise fetch_context's
    call shape before the real runtime exists. allowed_hosts is a test fixture
    allowlist, not the eventual containment policy.
    """

    is_stub = True

    def __init__(self, allowed_hosts=frozenset({"localhost", "127.0.0.1"})):
        self.allowed_hosts = frozenset(allowed_hosts)

    def fetch(self, url: str) -> dict:
        host = _extract_host(url)
        if host in self.allowed_hosts:
            return {"success": True, "body": f"fixture body for {url}", "error": None}
        return {
            "success": False,
            "body": None,
            "error": f"destination not permitted by test fixture: {host}",
        }


def _extract_host(url: str) -> str:
    without_scheme = url.split("://", 1)[-1]
    return without_scheme.split("/", 1)[0].split(":", 1)[0]


def fetch_context(url: str, transport: ContextTransport) -> dict:
    """Request context for url through transport. transport decides allow/deny;
    this function never contacts the network itself.
    """
    result = transport.fetch(url)
    return {"url": url, **result}
