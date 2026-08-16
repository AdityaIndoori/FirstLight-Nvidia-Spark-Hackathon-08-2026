"""B6a: deterministic Agency Plan domain layer.

Validated data model, availability state, and copy-on-write editing/
accounting that the later Nemotron planner (not called here) will produce
assignments into. No Nano, no Lightning, no NemoClaw, no routing, no
OpenShell, no UI -- pure deterministic Python.

--------------------------------------------------------------------------
FROZEN PUBLIC CONTRACTS (do not change these shapes)
--------------------------------------------------------------------------

Agency plan, B -> C:
    {
        agencies: [
            {
                agency: "fire" | "ems" | "police" | "public_works",
                units_required: int,
                units_available: int,
                steps: [{n: int, footprint_id: str, label: str,
                          centroid: [lng, lat], task: str, units: int}],
            }, ...  # all four agencies always present, even with 0 steps
        ],
        drafted_by: str,
    }

Set availability, C -> B:
    {agency: str, units_available: int, operator: str}

Plan edit, C -> B:
    {agency: str, op: "add"|"move"|"edit"|"delete"|"reassign",
     step_n: int, payload: {...}, operator: str}
    For "reassign", `agency` is the SOURCE agency; the destination is
    payload["to_agency"].

--------------------------------------------------------------------------
INVARIANTS
--------------------------------------------------------------------------

- units_required is ALWAYS derived as sum(step.units for step in steps) --
  never trusted from a caller, always recomputed after every edit.
- units_available comes ONLY from AvailabilityRegistry.set_availability --
  nothing in this module invents or infers it. Before an operator supplies
  it for an agency, it is 0 (no pre-existing project convention for a
  different default was found when this module was written).
- n is assigned by B, contiguous 1..len(steps), always reflecting current
  order -- never trusted from a caller, always recomputed after every edit.
- is_overcommitted/units_shortfall are INTERNAL helpers over one agency
  group, never added as fields on the public AgencyPlan contract.
- Edits never mutate the caller's input plan (copy-on-write) and never
  touch units_available.
- No second logging system: callers wire results into the existing
  append-only decision_log.append_decision() themselves; this module keeps
  operator identity on every result for that integration point.
"""

import copy

SUPPORTED_AGENCIES = ("fire", "ems", "police", "public_works")

_EDITABLE_STEP_FIELDS = ("label", "centroid", "task", "units")
_VALID_OPS = ("add", "move", "edit", "delete", "reassign")


class AgencyPlanError(ValueError):
    """Raised for an agency, step, or edit that fails contract validation --
    a contract failure, never silently corrected or ignored.
    """


def _validate_agency(agency) -> None:
    if agency not in SUPPORTED_AGENCIES:
        raise AgencyPlanError(f"agency must be one of {SUPPORTED_AGENCIES}, got {agency!r}")


def _step_field_errors(fields: dict) -> list:
    errors = []
    footprint_id = fields.get("footprint_id")
    if not isinstance(footprint_id, str) or not footprint_id.strip():
        errors.append("footprint_id must be a non-empty string")

    label = fields.get("label")
    if not isinstance(label, str) or not label.strip():
        errors.append("label must be a non-empty string")

    centroid = fields.get("centroid")
    if (
        not isinstance(centroid, (list, tuple))
        or len(centroid) != 2
        or not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in centroid)
    ):
        errors.append("centroid must be exactly [lng, lat]")

    task = fields.get("task")
    if not isinstance(task, str) or not task.strip():
        errors.append("task must be a non-empty string")

    units = fields.get("units")
    if not isinstance(units, int) or isinstance(units, bool) or units < 1:
        errors.append("units must be an integer >= 1")

    return errors


def _units_required(steps: list) -> int:
    return sum(step["units"] for step in steps)


def _renumber(steps: list) -> list:
    """New list, new step dicts -- never mutates the originals."""
    return [{**step, "n": index + 1} for index, step in enumerate(steps)]


def _recompute(group: dict) -> None:
    """Renumber group['steps'] and recompute group['units_required'] from
    them. Mutates only the given (already copy-on-write) working group.
    """
    group["steps"] = _renumber(group["steps"])
    group["units_required"] = _units_required(group["steps"])


def _agency_group(plan: dict, agency: str) -> dict:
    for group in plan["agencies"]:
        if group["agency"] == agency:
            return group
    raise AgencyPlanError(f"agency {agency!r} not present in plan")


def _step_index(steps: list, step_n) -> int:
    for index, step in enumerate(steps):
        if step["n"] == step_n:
            return index
    raise AgencyPlanError(f"no step with n={step_n!r}")


def is_overcommitted(agency_plan: dict) -> bool:
    """agency_plan: one entry from AgencyPlan['agencies']. INTERNAL ONLY --
    never added as a field on the public AgencyPlan contract.
    """
    return agency_plan["units_required"] > agency_plan["units_available"]


def units_shortfall(agency_plan: dict) -> int:
    """max(0, units_required - units_available) for one agency group.
    INTERNAL ONLY -- drives the future red UI state / ICS-213 RR export,
    never invents a resource that wasn't operator-supplied.
    """
    return max(0, agency_plan["units_required"] - agency_plan["units_available"])


class AvailabilityRegistry:
    """Member B's owned current availability state -- the ONLY source of
    units_available. Nothing else in this module sets or infers it.
    """

    def __init__(self):
        self._available = {agency: 0 for agency in SUPPORTED_AGENCIES}

    def set_availability(self, agency: str, units_available: int, operator: str) -> dict:
        """Validate and store current availability for agency. Returns the
        frozen Set-availability (C -> B) shape, suitable for the
        decision-log integration point (actor=operator) once that's wired.
        """
        _validate_agency(agency)
        if not isinstance(units_available, int) or isinstance(units_available, bool) or units_available < 0:
            raise AgencyPlanError(f"units_available must be an integer >= 0, got {units_available!r}")
        if not isinstance(operator, str) or not operator.strip():
            raise AgencyPlanError("operator must be a non-empty string")

        self._available[agency] = units_available
        return {"agency": agency, "units_available": units_available, "operator": operator}

    def get_availability(self, agency: str) -> int:
        _validate_agency(agency)
        return self._available[agency]


def build_agency_plan(assignments: list, drafted_by: str, availability: AvailabilityRegistry) -> dict:
    """INTERNAL constructor: build a validated AgencyPlan (frozen B -> C
    contract) from a list of INTERNAL proposed assignments -- NOT a public
    contract, just the smallest shape the later Nemotron planner (not
    called here) will eventually produce into:
        {agency, footprint_id, label, centroid, task, units}

    All four SUPPORTED_AGENCIES are always present in the result, even with
    zero steps. units_required is always derived from the assembled steps.
    units_available comes from `availability` (an AvailabilityRegistry) --
    never invented here. drafted_by is stored verbatim (e.g. "nano",
    "stub", "operator:<name>" -- this function never calls Nano itself).
    Never mutates `assignments`.
    """
    if not isinstance(drafted_by, str) or not drafted_by.strip():
        raise AgencyPlanError("drafted_by must be a non-empty string")

    steps_by_agency = {agency: [] for agency in SUPPORTED_AGENCIES}
    for assignment in assignments:
        _validate_agency(assignment.get("agency"))
        fields = {
            "footprint_id": assignment.get("footprint_id"),
            "label": assignment.get("label"),
            "centroid": assignment.get("centroid"),
            "task": assignment.get("task"),
            "units": assignment.get("units"),
        }
        errors = _step_field_errors(fields)
        if errors:
            raise AgencyPlanError("; ".join(errors))

        step = dict(fields)
        step["centroid"] = [fields["centroid"][0], fields["centroid"][1]]
        step["n"] = 0  # placeholder; renumbered below
        steps_by_agency[assignment["agency"]].append(step)

    agencies = []
    for agency in SUPPORTED_AGENCIES:
        steps = _renumber(steps_by_agency[agency])
        agencies.append(
            {
                "agency": agency,
                "units_required": _units_required(steps),
                "units_available": availability.get_availability(agency),
                "steps": steps,
            }
        )

    return {"agencies": agencies, "drafted_by": drafted_by}


def apply_plan_edit(plan: dict, plan_edit: dict) -> dict:
    """Apply ONE frozen Plan-edit (C -> B) operation to plan and return a
    NEW AgencyPlan (frozen B -> C contract). Never mutates the input plan;
    never touches units_available.

    plan_edit = {agency, op, step_n, payload, operator}. Payload shapes per op:
        add:      {footprint_id, label, centroid, task, units}  (all required)
        move:     {to_n: int}  -- destination position within the same agency
        edit:     any subset of {label, centroid, task, units}; a
                  "footprint_id" key is REJECTED (footprint_id is immutable
                  via edit)
        delete:   payload unused
        reassign: {to_agency: str (required), task?, units?, to_n?}
                  footprint_id/label/centroid are ALWAYS preserved from the
                  source step regardless of payload; task/units are
                  preserved unless payload explicitly overrides them;
                  to_n controls destination position (default: append)

    `operator` is retained on every path for the decision-log integration
    point -- this function does not write to the log itself.
    """
    agency = plan_edit.get("agency")
    op = plan_edit.get("op")
    step_n = plan_edit.get("step_n")
    payload = plan_edit.get("payload") or {}
    operator = plan_edit.get("operator")

    if not isinstance(operator, str) or not operator.strip():
        raise AgencyPlanError("operator must be a non-empty string")
    if op not in _VALID_OPS:
        raise AgencyPlanError(f"op must be one of {_VALID_OPS}, got {op!r}")
    _validate_agency(agency)

    working_plan = copy.deepcopy(plan)

    if op == "add":
        return _apply_add(working_plan, agency, payload)
    if op == "move":
        return _apply_move(working_plan, agency, step_n, payload)
    if op == "edit":
        return _apply_edit(working_plan, agency, step_n, payload)
    if op == "delete":
        return _apply_delete(working_plan, agency, step_n)
    return _apply_reassign(working_plan, agency, step_n, payload)


def _apply_add(plan: dict, agency: str, payload: dict) -> dict:
    fields = {
        "footprint_id": payload.get("footprint_id"),
        "label": payload.get("label"),
        "centroid": payload.get("centroid"),
        "task": payload.get("task"),
        "units": payload.get("units"),
    }
    errors = _step_field_errors(fields)
    if errors:
        raise AgencyPlanError("; ".join(errors))

    new_step = dict(fields)
    new_step["centroid"] = [fields["centroid"][0], fields["centroid"][1]]
    new_step["n"] = 0  # placeholder; renumbered below

    group = _agency_group(plan, agency)
    group["steps"].append(new_step)
    _recompute(group)
    return plan


def _apply_move(plan: dict, agency: str, step_n, payload: dict) -> dict:
    group = _agency_group(plan, agency)
    steps = group["steps"]
    index = _step_index(steps, step_n)

    to_n = payload.get("to_n")
    if not isinstance(to_n, int) or isinstance(to_n, bool) or not (1 <= to_n <= len(steps)):
        raise AgencyPlanError(f"payload.to_n must be an integer in [1, {len(steps)}], got {to_n!r}")

    step = steps.pop(index)
    steps.insert(to_n - 1, step)
    _recompute(group)
    return plan


def _apply_edit(plan: dict, agency: str, step_n, payload: dict) -> dict:
    if "footprint_id" in payload:
        raise AgencyPlanError("footprint_id cannot be changed via edit")

    group = _agency_group(plan, agency)
    steps = group["steps"]
    index = _step_index(steps, step_n)

    updated = dict(steps[index])
    for field in _EDITABLE_STEP_FIELDS:
        if field in payload:
            updated[field] = payload[field]
    if "centroid" in payload and isinstance(payload["centroid"], (list, tuple)) and len(payload["centroid"]) == 2:
        updated["centroid"] = [payload["centroid"][0], payload["centroid"][1]]

    errors = _step_field_errors(updated)
    if errors:
        raise AgencyPlanError("; ".join(errors))

    steps[index] = updated
    _recompute(group)
    return plan


def _apply_delete(plan: dict, agency: str, step_n) -> dict:
    group = _agency_group(plan, agency)
    steps = group["steps"]
    index = _step_index(steps, step_n)
    del steps[index]
    _recompute(group)
    return plan


def _apply_reassign(plan: dict, agency: str, step_n, payload: dict) -> dict:
    to_agency = payload.get("to_agency")
    _validate_agency(to_agency)
    if to_agency == agency:
        raise AgencyPlanError("reassign requires a different destination agency")

    source_group = _agency_group(plan, agency)
    source_steps = source_group["steps"]
    index = _step_index(source_steps, step_n)
    moving_step = source_steps.pop(index)

    reassigned_step = {
        "footprint_id": moving_step["footprint_id"],  # always preserved
        "label": moving_step["label"],  # always preserved
        "centroid": list(moving_step["centroid"]),  # always preserved
        "task": payload.get("task", moving_step["task"]),
        "units": payload.get("units", moving_step["units"]),
        "n": 0,  # placeholder; renumbered below
    }
    errors = _step_field_errors(reassigned_step)
    if errors:
        raise AgencyPlanError("; ".join(errors))

    dest_group = _agency_group(plan, to_agency)
    dest_steps = dest_group["steps"]
    to_n = payload.get("to_n")
    if isinstance(to_n, int) and not isinstance(to_n, bool) and 1 <= to_n <= len(dest_steps) + 1:
        dest_steps.insert(to_n - 1, reassigned_step)
    else:
        dest_steps.append(reassigned_step)

    _recompute(source_group)
    _recompute(dest_group)
    return plan
