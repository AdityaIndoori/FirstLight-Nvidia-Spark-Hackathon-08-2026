from backend.decision.agency_plan_client import _DEFAULT_TIMEOUT_S, RealAgencyPlanDraftClient


# scripts/agency_plan_parallel_latency_check.py runs the REAL production
# draft_agency_plan_with_diagnostics() (never a reimplementation), which
# constructs RealAgencyPlanDraftClient() with its default timeout unless a
# client is explicitly injected -- this task's per-building requests are
# meant to fit inside the existing 10s budget, not to get a longer one.
# These guard that the production timeout stays exactly as it was.
def test_production_timeout_constant_unchanged():
    assert _DEFAULT_TIMEOUT_S == 10.0


def test_default_client_still_uses_production_timeout():
    client = RealAgencyPlanDraftClient()
    assert client.timeout_s == 10.0
