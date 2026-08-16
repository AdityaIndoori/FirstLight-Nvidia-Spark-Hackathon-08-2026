import copy
import threading
import time

import pytest

from backend.decision.agency_plan import AvailabilityRegistry
from backend.decision.agency_plan_client import _ACTION_TASK_TEXT, AgencyPlanDraftClient
from backend.decision.agency_plan_drafter import (
    DEFAULT_MAX_CONCURRENCY,
    build_planner_candidate,
    draft_agency_plan,
    draft_agency_plan_with_diagnostics,
)
from backend.decision.nano_client import NanoClientError

PROCESSED_EVIDENCE_A = {
    "evidence": {
        "footprint_id": "fp-100",
        "image_id": "img-1",
        "label": "412 Elm St",
        "centroid": [-122.400, 47.600],
        "captured_at": 1_700_000_000.0,
        "damage_class": 3,
        "confidence": 0.91,
        "graded_by": "nemotron-vl",
        "vl_caption": "Two-storey structure with visible flames and heavy roof damage.",
        "footprint_area_m2": 120.0,
        "facility_near": None,
        "neighbor_damage_classes": [1, 2, 2],
        "vulnerable_density": 2.31,
    },
    "votes": [3] * 8,
    "voted_class": 3,
    "vote_agreement": 1.0,
    "doubt": 0.05,
    "staleness_h": 6.5,
    "road_cutoff": None,
    "priority": 24.19320,
    "lightning_recovery": "model",
}

PROCESSED_EVIDENCE_B = {
    "evidence": {
        "footprint_id": "fp-200",
        "image_id": "img-2",
        "label": "Riverside Dialysis Center",
        "centroid": [-122.385, 47.605],
        "captured_at": 1_700_000_000.0,
        "damage_class": 2,
        "confidence": 0.78,
        "graded_by": "nemotron-vl",
        "vl_caption": "Building has significant exterior damage and obstructed entrance.",
        "footprint_area_m2": 200.0,
        "facility_near": {"name": "Riverside Dialysis Center", "type": "dialysis", "dist_m": 0},
        "neighbor_damage_classes": [1, 2, 2],
        "vulnerable_density": 2.4,
    },
    "votes": [2] * 8,
    "voted_class": 2,
    "vote_agreement": 1.0,
    "doubt": 0.05,
    "staleness_h": 4.5,
    "road_cutoff": None,
    "priority": 12.5,
    "lightning_recovery": "model",
}

# Section 8 fixture B: damaged commercial structure adjacent to an active
# road closure -- no fire, no collapse evidence.
PROCESSED_EVIDENCE_ROAD_CLOSURE = {
    "evidence": {
        "footprint_id": "fp-400",
        "image_id": "img-4",
        "label": "35th Ave SW & Elm St",
        "centroid": [-122.390, 47.598],
        "captured_at": 1_700_000_000.0,
        "damage_class": 1,
        "confidence": 0.66,
        "graded_by": "nemotron-vl",
        "vl_caption": "Damaged commercial structure adjacent to an active road closure.",
        "footprint_area_m2": 180.0,
        "facility_near": None,
        "neighbor_damage_classes": [0, 1, 1],
        "vulnerable_density": 1.1,
    },
    "votes": [1] * 8,
    "voted_class": 1,
    "vote_agreement": 1.0,
    "doubt": 0.05,
    "staleness_h": 5.5,
    "road_cutoff": None,
    "priority": 1.0,
    "lightning_recovery": "model",
}

# fp-100's caption has flames evidence only, no fire+collapse+debris overlap.
VALID_FIRE_RESPONSE = {"assignments": [{"agency": "fire", "action": "fire_suppression", "units": 2}]}
# fp-200's facility_near is dialysis, so medical_support is supported.
VALID_EMS_RESPONSE = {"assignments": [{"agency": "ems", "action": "medical_support", "units": 2}]}


def make_candidates():
    return [
        build_planner_candidate(PROCESSED_EVIDENCE_A, confirmed=True),
        build_planner_candidate(PROCESSED_EVIDENCE_B, confirmed=False),
    ]


def make_availability() -> AvailabilityRegistry:
    availability = AvailabilityRegistry()
    availability.set_availability("fire", 3, "operator:jsmith")
    availability.set_availability("ems", 1, "operator:jsmith")
    availability.set_availability("police", 1, "operator:jsmith")
    availability.set_availability("public_works", 1, "operator:jsmith")
    return availability


class ScriptedClient(AgencyPlanDraftClient):
    """responses_by_footprint_id: {footprint_id: [attempt1_response, attempt2_response, ...]}.
    Each building's own attempt queue is consumed in order; thread-safe
    since the drafter may call this concurrently across DIFFERENT buildings.
    """

    def __init__(self, responses_by_footprint_id: dict):
        self._responses = {k: list(v) for k, v in responses_by_footprint_id.items()}
        self._next_index = {k: 0 for k in self._responses}
        self._lock = threading.Lock()
        self.calls = []  # (footprint_id, validation_error), append order = call order across all buildings

    def propose_assignments_for_building(self, candidate, validation_error=None):
        footprint_id = candidate["footprint_id"]
        with self._lock:
            self.calls.append((footprint_id, validation_error))
            index = self._next_index[footprint_id]
            self._next_index[footprint_id] += 1
        response = self._responses[footprint_id][index]
        if isinstance(response, Exception):
            raise response
        return response

    def calls_for(self, footprint_id: str) -> int:
        return sum(1 for fid, _ in self.calls if fid == footprint_id)


class SlowRecordingClient(AgencyPlanDraftClient):
    """Always succeeds with a fixed response after an artificial delay;
    tracks peak concurrent in-flight calls.
    """

    def __init__(self, delay_s: float = 0.05):
        self.delay_s = delay_s
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0
        self.calls = []

    def propose_assignments_for_building(self, candidate, validation_error=None):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            self.calls.append(candidate["footprint_id"])
        time.sleep(self.delay_s)
        with self._lock:
            self._in_flight -= 1
        return {"assignments": [{"agency": "fire", "action": "fire_suppression", "units": 1}]}


def group(plan: dict, agency: str) -> dict:
    for g in plan["agencies"]:
        if g["agency"] == agency:
            return g
    raise AssertionError(f"agency {agency!r} not present")


# build_planner_candidate: only the listed fields, nothing invented
def test_build_planner_candidate_extracts_expected_fields():
    candidate = build_planner_candidate(PROCESSED_EVIDENCE_A, confirmed=True)
    assert candidate == {
        "footprint_id": "fp-100",
        "label": "412 Elm St",
        "centroid": [-122.400, 47.600],
        "damage_class": 3,
        "confidence": 0.91,
        "confirmed": True,
        "priority": 24.19320,
        "vl_caption": "Two-storey structure with visible flames and heavy roof damage.",
        "facility_near": None,
    }


def test_build_planner_candidate_does_not_mutate_processed_evidence():
    snapshot = copy.deepcopy(PROCESSED_EVIDENCE_A)
    build_planner_candidate(PROCESSED_EVIDENCE_A, confirmed=True)
    assert PROCESSED_EVIDENCE_A == snapshot


# 1: one Nano request is made per building
def test_one_request_per_building():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    draft_agency_plan(make_candidates(), make_availability(), client=client)

    assert client.calls_for("fp-100") == 1
    assert client.calls_for("fp-200") == 1
    assert len(client.calls) == 2


# 6: authoritative identity/location are reattached by B
def test_authoritative_identity_and_location_reattached():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    plan = draft_agency_plan(make_candidates(), make_availability(), client=client)

    fire_step = group(plan, "fire")["steps"][0]
    assert fire_step["footprint_id"] == "fp-100"
    assert fire_step["label"] == "412 Elm St"
    assert fire_step["centroid"] == [-122.400, 47.600]

    ems_step = group(plan, "ems")["steps"][0]
    assert ems_step["footprint_id"] == "fp-200"
    assert ems_step["label"] == "Riverside Dialysis Center"
    assert ems_step["centroid"] == [-122.385, 47.605]


# 8: requests execute with bounded concurrency
def test_requests_execute_with_bounded_concurrency():
    candidates = [build_planner_candidate(PROCESSED_EVIDENCE_A, confirmed=True) for _ in range(4)]
    for index, candidate in enumerate(candidates):
        candidate["footprint_id"] = f"fp-{index}"
    client = SlowRecordingClient(delay_s=0.05)

    draft_agency_plan(candidates, make_availability(), client=client, max_concurrency=2)

    assert client.peak_in_flight <= 2
    assert client.peak_in_flight > 1  # actually ran concurrently, not serially


def test_concurrency_never_exceeds_candidate_count():
    client = SlowRecordingClient(delay_s=0.03)
    draft_agency_plan(make_candidates(), make_availability(), client=client, max_concurrency=100)
    assert client.peak_in_flight <= 2  # only 2 candidates


def test_default_max_concurrency_is_four():
    assert DEFAULT_MAX_CONCURRENCY == 4


# 9: result ordering remains deterministic (original candidate order),
# regardless of which building's request finishes first
def test_result_ordering_matches_original_candidate_order_regardless_of_completion_order():
    class OrderSensitiveClient(AgencyPlanDraftClient):
        def propose_assignments_for_building(self, candidate, validation_error=None):
            if candidate["footprint_id"] == "fp-100":
                time.sleep(0.08)  # first candidate finishes LAST
                return VALID_FIRE_RESPONSE
            return VALID_EMS_RESPONSE

    plan = draft_agency_plan(make_candidates(), make_availability(), client=OrderSensitiveClient())

    # fp-100 (candidate 0, fire) must still appear before fp-200 (candidate 1,
    # ems) wherever the merge order is observable -- here, via B6a's agency
    # grouping being unaffected and each agency's own step numbering starting
    # at 1 regardless of arrival order.
    assert group(plan, "fire")["steps"][0]["footprint_id"] == "fp-100"
    assert group(plan, "ems")["steps"][0]["footprint_id"] == "fp-200"


def test_building_diagnostics_preserve_original_candidate_order():
    class OrderSensitiveClient(AgencyPlanDraftClient):
        def propose_assignments_for_building(self, candidate, validation_error=None):
            if candidate["footprint_id"] == "fp-100":
                time.sleep(0.08)
            return {"assignments": []}

    result = draft_agency_plan_with_diagnostics(make_candidates(), make_availability(), client=OrderSensitiveClient())
    building_ids = [b["footprint_id"] for b in result["diagnostics"]["buildings"]]
    assert building_ids == ["fp-100", "fp-200"]


# 10: one building failure does not discard other successful buildings
def test_one_building_failure_does_not_discard_others():
    client = ScriptedClient(
        {
            "fp-100": [VALID_FIRE_RESPONSE],
            "fp-200": [NanoClientError("down"), NanoClientError("down")],
        }
    )
    fallback = ScriptedClient({"fp-200": [{"assignments": []}]})

    result = draft_agency_plan_with_diagnostics(
        make_candidates(), make_availability(), client=client, fallback_client=fallback
    )

    assert group(result["plan"], "fire")["steps"][0]["footprint_id"] == "fp-100"
    buildings_by_id = {b["footprint_id"]: b for b in result["diagnostics"]["buildings"]}
    assert buildings_by_id["fp-100"]["recovery"] == "nano"
    assert buildings_by_id["fp-200"]["recovery"] == "stub"
    assert result["diagnostics"]["model_building_count"] == 1
    assert result["diagnostics"]["fallback_building_count"] == 1


# 11: schema-invalid response gets exactly one re-prompt
def test_schema_invalid_response_gets_exactly_one_reprompt():
    invalid_response = {"assignments": [{"agency": "not-real", "action": "fire_suppression", "units": 1}]}
    client = ScriptedClient({"fp-100": [invalid_response, VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})

    result = draft_agency_plan_with_diagnostics(make_candidates(), make_availability(), client=client)

    assert client.calls_for("fp-100") == 2
    fp100_calls = [ve for fid, ve in client.calls if fid == "fp-100"]
    assert fp100_calls[0] is None
    assert isinstance(fp100_calls[1], str) and fp100_calls[1]

    buildings_by_id = {b["footprint_id"]: b for b in result["diagnostics"]["buildings"]}
    assert buildings_by_id["fp-100"]["attempt_count"] == 2
    assert buildings_by_id["fp-100"]["recovery"] == "nano"


def test_second_invalid_response_for_that_building_falls_back():
    invalid_response = {"assignments": [{"agency": "not-real", "action": "fire_suppression", "units": 1}]}
    client = ScriptedClient({"fp-100": [invalid_response, invalid_response], "fp-200": [VALID_EMS_RESPONSE]})
    fallback = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE]})

    result = draft_agency_plan_with_diagnostics(
        make_candidates(), make_availability(), client=client, fallback_client=fallback
    )

    buildings_by_id = {b["footprint_id"]: b for b in result["diagnostics"]["buildings"]}
    assert buildings_by_id["fp-100"]["recovery"] == "stub"
    assert buildings_by_id["fp-100"]["attempt_count"] == 2
    assert buildings_by_id["fp-100"]["fallback_reason"] is not None


# 12: transport timeout falls back only for that building
def test_transport_timeout_falls_back_only_for_that_building():
    client = ScriptedClient({"fp-100": [NanoClientError("timed out")], "fp-200": [VALID_EMS_RESPONSE]})
    fallback = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE]})

    result = draft_agency_plan_with_diagnostics(
        make_candidates(), make_availability(), client=client, fallback_client=fallback
    )

    buildings_by_id = {b["footprint_id"]: b for b in result["diagnostics"]["buildings"]}
    assert buildings_by_id["fp-100"]["recovery"] == "stub"
    assert buildings_by_id["fp-100"]["attempt_count"] == 1  # no retry after a transport failure
    assert buildings_by_id["fp-200"]["recovery"] == "nano"  # untouched by fp-100's failure


def test_all_buildings_via_nano_gives_drafted_by_nano():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    plan = draft_agency_plan(make_candidates(), make_availability(), client=client)
    assert plan["drafted_by"] == "nano"


def test_all_buildings_via_fallback_gives_drafted_by_stub():
    client = ScriptedClient({"fp-100": [NanoClientError("down")], "fp-200": [NanoClientError("down")]})
    fallback = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    plan = draft_agency_plan(make_candidates(), make_availability(), client=client, fallback_client=fallback)
    assert plan["drafted_by"] == "stub"


def test_mixed_provenance_conservatively_reports_drafted_by_stub():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [NanoClientError("down")]})
    fallback = ScriptedClient({"fp-200": [VALID_EMS_RESPONSE]})
    plan = draft_agency_plan(make_candidates(), make_availability(), client=client, fallback_client=fallback)
    # documented contract gap: never overclaim "nano" when any building fell back
    assert plan["drafted_by"] == "stub"


# 13: units_required is still derived by B6a
def test_units_required_still_derived_by_b6a():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    plan = draft_agency_plan(make_candidates(), make_availability(), client=client)

    fire = group(plan, "fire")
    assert fire["units_required"] == sum(s["units"] for s in fire["steps"])
    assert fire["units_required"] == 2


# 14: operator units_available remains untouched
def test_operator_availability_untouched():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    availability = make_availability()
    plan = draft_agency_plan(make_candidates(), availability, client=client)

    for g in plan["agencies"]:
        assert g["units_available"] == availability.get_availability(g["agency"])


def test_units_available_never_accepted_from_model_output():
    response_with_units_available = {
        "assignments": [
            {"agency": "fire", "action": "fire_suppression", "units": 2, "units_available": 999}
        ]
    }
    client = ScriptedClient({"fp-100": [response_with_units_available], "fp-200": [VALID_EMS_RESPONSE]})
    availability = make_availability()
    plan = draft_agency_plan(make_candidates(), availability, client=client)

    assert group(plan, "fire")["units_available"] == availability.get_availability("fire")
    assert group(plan, "fire")["units_available"] != 999


def test_unsupported_agency_rejected():
    bad_response = {"assignments": [{"agency": "hazmat", "action": "fire_suppression", "units": 1}]}
    client = ScriptedClient({"fp-100": [bad_response, bad_response], "fp-200": [VALID_EMS_RESPONSE]})
    fallback = ScriptedClient({"fp-100": [{"assignments": []}]})

    plan = draft_agency_plan(make_candidates(), make_availability(), client=client, fallback_client=fallback)
    assert plan["drafted_by"] == "stub"  # fp-100 fell back


def test_non_positive_units_rejected():
    for bad_units in (0, -1, "two", 1.5):
        bad_response = {"assignments": [{"agency": "fire", "action": "fire_suppression", "units": bad_units}]}
        client = ScriptedClient({"fp-100": [bad_response, bad_response], "fp-200": [VALID_EMS_RESPONSE]})
        fallback = ScriptedClient({"fp-100": [{"assignments": []}]})
        plan = draft_agency_plan(make_candidates(), make_availability(), client=client, fallback_client=fallback)
        assert plan["drafted_by"] == "stub"


# 2: only allowed action enum values are accepted
def test_invalid_action_value_rejected():
    for bad_action in ("extinguish_everything", "", None, 123):
        bad_response = {"assignments": [{"agency": "fire", "action": bad_action, "units": 1}]}
        client = ScriptedClient({"fp-100": [bad_response, bad_response], "fp-200": [VALID_EMS_RESPONSE]})
        fallback = ScriptedClient({"fp-100": [{"assignments": []}]})
        plan = draft_agency_plan(make_candidates(), make_availability(), client=client, fallback_client=fallback)
        assert plan["drafted_by"] == "stub"


# 3: invalid agency/action pair rejected -- e.g. fire cannot do medical_support
def test_invalid_agency_action_pair_rejected():
    bad_response = {"assignments": [{"agency": "fire", "action": "medical_support", "units": 1}]}
    client = ScriptedClient({"fp-100": [bad_response, bad_response], "fp-200": [VALID_EMS_RESPONSE]})
    fallback = ScriptedClient({"fp-100": [{"assignments": []}]})

    plan = draft_agency_plan(make_candidates(), make_availability(), client=client, fallback_client=fallback)
    assert plan["drafted_by"] == "stub"


# 4: fire_suppression without fire evidence rejected -- this is the exact
# live-run bug (fp-002/fp-004 style: damage/obstruction but no flames).
def test_fire_suppression_without_fire_evidence_rejected():
    candidate = build_planner_candidate(PROCESSED_EVIDENCE_B, confirmed=False)  # dialysis, no fire evidence
    bad_response = {"assignments": [{"agency": "fire", "action": "fire_suppression", "units": 2}]}
    client = ScriptedClient({"fp-200": [bad_response, bad_response]})
    fallback = ScriptedClient({"fp-200": [{"assignments": []}]})

    result = draft_agency_plan_with_diagnostics(
        [candidate], make_availability(), client=client, fallback_client=fallback
    )

    building = result["diagnostics"]["buildings"][0]
    assert building["recovery"] == "stub"
    assert "fire_suppression" in building["attempt_1_error"]
    assert group(result["plan"], "fire")["steps"] == []


# 5: collapse_response without collapse evidence rejected
def test_collapse_response_without_collapse_evidence_rejected():
    candidate = build_planner_candidate(PROCESSED_EVIDENCE_A, confirmed=True)  # flames only, no collapse wording
    bad_response = {"assignments": [{"agency": "fire", "action": "collapse_response", "units": 2}]}
    client = ScriptedClient({"fp-100": [bad_response, bad_response]})
    fallback = ScriptedClient({"fp-100": [{"assignments": []}]})

    result = draft_agency_plan_with_diagnostics(
        [candidate], make_availability(), client=client, fallback_client=fallback
    )

    building = result["diagnostics"]["buildings"][0]
    assert building["recovery"] == "stub"
    assert "collapse_response" in building["attempt_1_error"]


# 8: debris_clearance requires debris/access evidence
def test_debris_clearance_without_debris_evidence_rejected():
    candidate = build_planner_candidate(PROCESSED_EVIDENCE_A, confirmed=True)  # flames only, no debris wording
    bad_response = {"assignments": [{"agency": "public_works", "action": "debris_clearance", "units": 1}]}
    client = ScriptedClient({"fp-100": [bad_response, bad_response]})
    fallback = ScriptedClient({"fp-100": [{"assignments": []}]})

    result = draft_agency_plan_with_diagnostics(
        [candidate], make_availability(), client=client, fallback_client=fallback
    )

    building = result["diagnostics"]["buildings"][0]
    assert building["recovery"] == "stub"
    assert "debris_clearance" in building["attempt_1_error"]


# 9: road_closure requires road/closure evidence
def test_road_closure_without_road_evidence_rejected():
    candidate = build_planner_candidate(PROCESSED_EVIDENCE_A, confirmed=True)  # flames only, no road wording
    bad_response = {"assignments": [{"agency": "police", "action": "road_closure", "units": 1}]}
    client = ScriptedClient({"fp-100": [bad_response, bad_response]})
    fallback = ScriptedClient({"fp-100": [{"assignments": []}]})

    result = draft_agency_plan_with_diagnostics(
        [candidate], make_availability(), client=client, fallback_client=fallback
    )

    building = result["diagnostics"]["buildings"][0]
    assert building["recovery"] == "stub"
    assert "road_closure" in building["attempt_1_error"]


def test_road_closure_with_road_evidence_supported():
    candidate = build_planner_candidate(PROCESSED_EVIDENCE_ROAD_CLOSURE, confirmed=False)
    response = {"assignments": [{"agency": "police", "action": "road_closure", "units": 1}]}
    client = ScriptedClient({"fp-400": [response]})

    plan = draft_agency_plan([candidate], make_availability(), client=client)
    assert group(plan, "police")["steps"][0]["footprint_id"] == "fp-400"
    assert group(plan, "police")["steps"][0]["task"] == "Manage road closure"


# empty assignments list is valid (evidence doesn't support an assignment)
def test_empty_assignments_list_is_valid():
    client = ScriptedClient({"fp-100": [{"assignments": []}], "fp-200": [VALID_EMS_RESPONSE]})
    plan = draft_agency_plan(make_candidates(), make_availability(), client=client)
    assert plan["drafted_by"] == "nano"
    assert group(plan, "fire")["steps"] == []


def test_all_four_groups_exist_regardless_of_path():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    plan = draft_agency_plan(make_candidates(), make_availability(), client=client)
    assert {g["agency"] for g in plan["agencies"]} == {"fire", "ems", "police", "public_works"}


def test_candidates_not_mutated_by_draft_agency_plan():
    candidates = make_candidates()
    snapshot = copy.deepcopy(candidates)
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})

    draft_agency_plan(candidates, make_availability(), client=client)

    assert candidates == snapshot


def test_fallback_that_is_itself_invalid_raises_loudly():
    client = ScriptedClient({"fp-100": [NanoClientError("down")], "fp-200": [VALID_EMS_RESPONSE]})
    broken_fallback = ScriptedClient({"fp-100": [{"assignments": [{"agency": "not-real"}]}]})

    with pytest.raises(NanoClientError):
        draft_agency_plan(make_candidates(), make_availability(), client=client, fallback_client=broken_fallback)


def test_diagnostics_not_added_to_public_agency_plan():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    result = draft_agency_plan_with_diagnostics(make_candidates(), make_availability(), client=client)

    assert set(result["plan"].keys()) == {"agencies", "drafted_by"}
    for g in result["plan"]["agencies"]:
        assert set(g.keys()) == {"agency", "units_required", "units_available", "steps"}


def test_draft_agency_plan_and_with_diagnostics_agree_on_plan():
    plan_only = draft_agency_plan(
        make_candidates(),
        make_availability(),
        client=ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]}),
    )
    plan_with_diag = draft_agency_plan_with_diagnostics(
        make_candidates(),
        make_availability(),
        client=ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]}),
    )["plan"]

    assert plan_only == plan_with_diag


def test_per_building_elapsed_s_is_recorded():
    client = ScriptedClient({"fp-100": [VALID_FIRE_RESPONSE], "fp-200": [VALID_EMS_RESPONSE]})
    result = draft_agency_plan_with_diagnostics(make_candidates(), make_availability(), client=client)

    for building in result["diagnostics"]["buildings"]:
        assert isinstance(building["elapsed_s"], float)
        assert building["elapsed_s"] >= 0.0


def test_empty_candidates_list_still_produces_all_four_groups():
    client = ScriptedClient({})
    plan = draft_agency_plan([], make_availability(), client=client)
    assert {g["agency"] for g in plan["agencies"]} == {"fire", "ems", "police", "public_works"}
    assert plan["drafted_by"] == "nano"


# --------------------------------------------------------------------------
# 12: deterministic task text is generated by B, never trusted from Nano
# --------------------------------------------------------------------------


def test_task_text_is_generated_by_b_not_trusted_from_model():
    response_with_extra_task_field = {
        "assignments": [
            {"agency": "fire", "action": "fire_suppression", "units": 2, "task": "SOMETHING NANO MADE UP"}
        ]
    }
    client = ScriptedClient({"fp-100": [response_with_extra_task_field], "fp-200": [VALID_EMS_RESPONSE]})
    plan = draft_agency_plan(make_candidates(), make_availability(), client=client)

    fire_step = group(plan, "fire")["steps"][0]
    assert fire_step["task"] == "Suppress structure fire"
    assert fire_step["task"] != "SOMETHING NANO MADE UP"


def test_deterministic_task_text_mapping():
    assert _ACTION_TASK_TEXT == {
        "fire_suppression": "Suppress structure fire",
        "collapse_response": "Assess structural collapse",
        "medical_support": "Support medical facility",
        "perimeter_control": "Establish scene perimeter",
        "road_closure": "Manage road closure",
        "debris_clearance": "Clear debris from access route",
    }
