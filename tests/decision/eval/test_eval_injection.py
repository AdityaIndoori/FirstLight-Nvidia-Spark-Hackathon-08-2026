import socket
import urllib.request

from backend.decision.eval.eval_injection import (
    INJECTION_FIXTURE,
    evaluate_injection_battery_offline,
    evaluate_openshell_tamper,
)


def test_fixture_has_at_least_ten_hostile_captions():
    assert len(INJECTION_FIXTURE) >= 10


def test_fixture_captions_are_distinct():
    captions = [case["caption"] for case in INJECTION_FIXTURE]
    assert len(captions) == len(set(captions))


def test_offline_battery_passes_with_zero_alterations():
    metric = evaluate_injection_battery_offline()
    assert metric["status"] == "pass"
    assert metric["value"]["altered_damage_grades"] == 0
    assert metric["value"]["unsupported_actions_accepted"] == 0
    assert metric["value"]["prohibited_tags"] == 0


def test_offline_battery_reports_untested_surfaces_honestly():
    metric = evaluate_injection_battery_offline()
    assert "vl_caption" in metric["details"]["not_exercised"]
    assert "DEFERRED" in metric["details"]["fema_portion"]
    assert "DEFERRED" in metric["details"]["openshell_portion"]


def test_offline_battery_deterministic():
    m1 = evaluate_injection_battery_offline()
    m2 = evaluate_injection_battery_offline()
    assert m1 == m2


def test_offline_battery_never_touches_network(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("offline injection battery must never touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    metric = evaluate_injection_battery_offline()
    assert metric["status"] == "pass"


# grade immutability -- the hard invariant this battery exists to prove
def test_no_candidate_damage_class_mutated_by_agency_drafting():
    import copy

    from backend.decision.agency_plan_client import StubAgencyPlanDraftClient
    from backend.decision.agency_plan_drafter import draft_agency_plan_with_diagnostics
    from backend.decision.eval.eval_injection import _make_availability, _make_candidate

    candidates = [_make_candidate(f"fp-{i}", case["caption"], case["known_damage_class"]) for i, case in enumerate(INJECTION_FIXTURE)]
    snapshot = copy.deepcopy(candidates)
    stub = StubAgencyPlanDraftClient()

    draft_agency_plan_with_diagnostics(candidates, _make_availability(), client=stub, fallback_client=stub)

    assert candidates == snapshot


def test_openshell_tamper_is_deferred():
    metric = evaluate_openshell_tamper()
    assert metric["status"] == "deferred"
    assert "openshell" in metric["details"]["reason"].lower() or "OpenShell" in metric["details"]["reason"]
