"""B8: common evaluation-metric result format and report assembly.

This package (backend/decision/eval/) is EVALUATION-ONLY -- it measures the
production decision/archive modules (B1-B7), it never changes their
behavior and is never imported by them. Nothing here is a new public
FIRST LIGHT wire contract; the report shape is internal, meant for a
future HUD/README publication step, not a frozen A/B/C boundary.

--------------------------------------------------------------------------
METRIC SHAPE
--------------------------------------------------------------------------

Every evaluation function in this package returns one metric dict:
    {
        "name": str,
        "status": "pass" | "fail" | "measured" | "deferred",
        "value": float | int | None,
        "threshold": float | int | str | None,
        "sample_count": int | None,
        "details": dict,
    }

Status meanings (deliberately NOT interchangeable):
    "pass"     -- a HARD deterministic safety/correctness gate was checked
                  and satisfied (e.g. 0 altered grades, 100% of rationale
                  fixture violations detected).
    "fail"     -- the same kind of hard gate was checked and violated.
    "measured" -- a genuine numeric value was computed (e.g. agreement
                  rate, precision@k) but no project-defined pass/fail
                  threshold exists for it -- report the number honestly,
                  never invent a threshold to force a verdict (Part 12).
    "deferred" -- the metric could not be measured at all in this run
                  (component not implemented, or real model/data
                  unavailable) -- details["reason"] always explains why.
                  A deferred metric is NEVER silently scored as passing.

build_report() never claims an overall "pass" just because every
measurable metric happened to pass while others were deferred -- see its
docstring.
"""

import time

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_MEASURED = "measured"
STATUS_DEFERRED = "deferred"

_VALID_STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_MEASURED, STATUS_DEFERRED)

OVERALL_PASS = "pass"
OVERALL_FAIL = "fail"
OVERALL_PARTIAL = "partial"


def make_metric(
    name: str,
    status: str,
    value=None,
    threshold=None,
    sample_count: int = None,
    details: dict = None,
) -> dict:
    """Build one metric dict in the common B8 shape. Raises ValueError for
    an unrecognized status -- never silently accepts a typo'd status string.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {_VALID_STATUSES}, got {status!r}")
    return {
        "name": name,
        "status": status,
        "value": value,
        "threshold": threshold,
        "sample_count": sample_count,
        "details": details if details is not None else {},
    }


def build_report(metrics: list, now_fn=time.time) -> dict:
    """Assemble the final B8 report from a list of metric dicts.

    overall is one of:
        "fail"    -- at least one metric has status "fail"
        "partial" -- no failures, but at least one metric is "deferred"
                     (an incomplete evaluation is never reported as a
                     clean pass, even if every metric that DID run passed)
        "pass"    -- every metric is "pass" or "measured"; none deferred,
                     none failed

    `now_fn` is injectable so callers (and pytest) can get a deterministic
    generated_at without monkeypatching time.time directly.
    """
    statuses = {metric["status"] for metric in metrics}
    if STATUS_FAIL in statuses:
        overall = OVERALL_FAIL
    elif STATUS_DEFERRED in statuses:
        overall = OVERALL_PARTIAL
    else:
        overall = OVERALL_PASS

    return {
        "metrics": metrics,
        "generated_at": now_fn(),
        "overall": overall,
    }


def deferred_metric(name: str, reason: str) -> dict:
    """Shorthand for the common "component not available" case."""
    return make_metric(name, STATUS_DEFERRED, details={"reason": reason})
