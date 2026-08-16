"""B8 Part B: FEMA field accuracy -- DEFERRED.

Checked the repository for a FEMA PDA row builder or Lightning FEMA-field
generator before writing this: README section 5.4/7 describes the FEMA
Preliminary Damage Assessment worksheet as part of the C7 "Download aid
package" feature, but no code implements it anywhere in backend/ --
grep for "fema" across backend/ only matches this module and
archive_tag_extractor.py's docstring (an unrelated example caption). There
is no FEMA row builder, no Lightning FEMA-field prompt, and no FEMA record
shape to label a fixture against.

Per instructions: do not build a FEMA subsystem just to make this metric
non-deferred. This module exists so the B8 report always includes a FEMA
entry (never silently omitted) with an exact, concrete reason.
"""

from backend.decision.eval.report import deferred_metric

_REASON = (
    "FEMA PDA row builder is not implemented in this repository (README C7/A "
    "describes it, but no backend/ module produces a FEMA record) -- no fields "
    "exist to compare against a labeled set."
)


def evaluate_fema_field_accuracy() -> dict:
    return deferred_metric("fema_field_accuracy", _REASON)
