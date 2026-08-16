"""B8 Part F: assistant citation faithfulness -- DEFERRED.

Checked the repository for a grounded-assistant/retrieval-answering
feature before writing this: README section 5.4 describes "a question box
over the archive... answered by Nemotron with citations only," and
explicitly lists it as the FIRST item on the cut list ("Semantic archive
search and the caption VLM... keep location + structured filter... 2.
Semantic archive search and the caption VLM"; the assistant sits above
that). grep across backend/ finds no assistant/question-answering/citation
module -- only unrelated hits (archive_tag_extractor.py's docstring
mentions no such thing either; agency_plan_client.py and lightning_client.py
match only on the word fragments inside unrelated identifiers).

Per instructions: do not build a whole grounded assistant just to make
this metric non-deferred -- the README explicitly allows cutting it. This
module exists so the B8 report always includes an F entry with an exact
reason rather than silently omitting it.
"""

from backend.decision.eval.report import deferred_metric

_REASON = (
    "No grounded-assistant/citation-answering feature exists in this repository "
    "(README explicitly lists it as a cuttable feature, section 6 cut list item 2) -- "
    "there is no cited-answer output to sample or verify refusal behavior against."
)


def evaluate_citation_faithfulness() -> dict:
    return deferred_metric("assistant_citation_faithfulness", _REASON)
