"""Hero rationale operation.

Application entry point for turning one already-ranked RankItem into an
operator-facing rationale string, agnostic to whether the backend is the
deterministic stub or the real Nemotron Nano 9B v2 client on localhost:8000.
Callers never change when the backend behind NanoRationaleClient changes.
"""

from backend.decision.nano_client import (
    NanoClientError,
    NanoRationaleClient,
    RealNanoRationaleClient,
    StubNanoRationaleClient,
)

_default_client: NanoRationaleClient = StubNanoRationaleClient()


def generate_rationale(rank_item: dict, client: NanoRationaleClient = None) -> str:
    """Return the hero rationale for rank_item via client (default: the deterministic stub)."""
    active_client = client if client is not None else _default_client
    return active_client.generate_rationale(rank_item)


def generate_rationale_with_recovery(
    rank_item: dict,
    real_client: NanoRationaleClient = None,
    fallback_client: NanoRationaleClient = None,
) -> dict:
    """Production entry point: try the real Nano client first; on any
    timeout/network/malformed-response failure (NanoClientError), fall back
    to the deterministic stub and label which path produced the result.

    Reuses the same recovery vocabulary as flight_planner.request_flight_plan
    (None | "stub") instead of inventing a second result format, so a model
    failure is recorded, never silently hidden.

    Returns {"rationale": str, "recovery": None | "stub"}.
    """
    active_real_client = real_client if real_client is not None else RealNanoRationaleClient()
    active_fallback_client = fallback_client if fallback_client is not None else _default_client

    try:
        text = active_real_client.generate_rationale(rank_item)
        return {"rationale": text, "recovery": None}
    except NanoClientError:
        text = active_fallback_client.generate_rationale(rank_item)
        return {"rationale": text, "recovery": "stub"}
