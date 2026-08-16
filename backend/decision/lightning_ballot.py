"""Lightning k=8 severity-voting ballot for ONE building.

INTERNAL, PRE-CONTRACT: the building_context shape below is the smallest
input needed to exercise the ballot. It is NOT the frozen A -> B
joined-building wire contract (README Section 7) -- that shape is still
unresolved between Members A and B. Treat every key here as provisional
until that contract is frozen; do not depend on it from outside this module.

    building_context = {
        "grader_class": int 0-3,             # VL model's primary grade, read-only reference
        "grader_confidence": float 0.0-1.0,  # VL model's confidence in that grade
        "vl_caption": str,                   # VL model's independently generated visual caption
        "footprint_area_m2": float,
        "facility_context": str or None,
        "neighbor_damage_classes": list[int],
    }

The image-capable model is Nemotron Nano 12B v2 VL (:8002) -- it sees the
pixels and produces grader_class/grader_confidence/vl_caption as two
independent observations. Lightning (:8001) is TEXT-ONLY: it never sees
pixels, and cross-examines the grade against the caption and GIS context
rather than assuming the grade is correct (see lightning_client._ballot_prompt).

Semantic split (README "Lightning ballot"):
    The VL model owns: grader_class, grader_confidence, vl_caption, and
    (downstream, in the public RankItem contract) damage_class/confidence/
    graded_by -- never touched here.
    Lightning owns: voted_class, vote_agreement, doubt -- computed here.

request_lightning_ballot() reads building_context["grader_class"] only as
one context input (and, in the stub, as its deterministic default vote); it
never writes to building_context and never produces or touches a
damage_class/confidence/graded_by field itself.

KNOWN CONTRACT GAP: the RankItem sent B -> C today carries inputs.doubt but
not the eight votes, voted_class, or vote_agreement, even though the README
wants an 8-pip tally shown in the rank panel. Not fixed here -- adding
fields to RankItem is out of scope for this task and must be resolved when
B/C contracts are frozen.

THROUGHPUT: request_lightning_ballot() issues its k=8 sample_severity() calls
serially and is left unmodified. request_lightning_ballot_parallel() is an
additive, bounded-concurrency alternative with an identical contract, for
Lightning throughput work (see lightning_perf.py and
scripts/lightning_parallel_benchmark.py / scripts/lightning_batch_sweep.py).
"""

import concurrent.futures

from backend.decision.lightning_client import LightningSeverityClient, StubLightningSeverityClient

K_VOTES = 8
_DEFAULT_TEMPERATURE = 0.7
_VALID_SEVERITY_LABELS = (0, 1, 2, 3)

_default_client: LightningSeverityClient = StubLightningSeverityClient()


def request_lightning_ballot(
    building_context: dict,
    client: LightningSeverityClient = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> dict:
    """Request k=8 severity votes for one building and aggregate them.

    Returns {"votes": [8 ints], "voted_class": int, "vote_agreement": float,
    "doubt": float}. Does not mutate building_context and never reads or
    writes damage_class/confidence/graded_by on any RankItem.
    """
    active_client = client if client is not None else _default_client

    votes = []
    for _ in range(K_VOTES):
        vote = active_client.sample_severity(building_context, temperature=temperature)
        if vote not in _VALID_SEVERITY_LABELS:
            raise ValueError(f"Lightning vote must be one of {_VALID_SEVERITY_LABELS}, got {vote!r}")
        votes.append(vote)

    voted_class = _voted_class(votes)
    vote_agreement = votes.count(voted_class) / K_VOTES
    doubt = max(0.05, 1 - vote_agreement)

    return {
        "votes": votes,
        "voted_class": voted_class,
        "vote_agreement": vote_agreement,
        "doubt": doubt,
    }


def _aggregate_votes(votes: list) -> dict:
    """Validate and aggregate a fixed list of severity votes into the same
    ballot result shape request_lightning_ballot returns. Extracted so the
    parallel ballot path (and the batch-sweep benchmark) share this exact
    validation/aggregation instead of reimplementing it -- request_lightning_ballot's
    own body is left untouched on purpose, so it stays available unmodified
    for direct comparison against the parallel path.
    """
    for vote in votes:
        if vote not in _VALID_SEVERITY_LABELS:
            raise ValueError(f"Lightning vote must be one of {_VALID_SEVERITY_LABELS}, got {vote!r}")

    voted_class = _voted_class(votes)
    vote_agreement = votes.count(voted_class) / len(votes)
    doubt = max(0.05, 1 - vote_agreement)

    return {
        "votes": votes,
        "voted_class": voted_class,
        "vote_agreement": vote_agreement,
        "doubt": doubt,
    }


def request_lightning_ballot_parallel(
    building_context: dict,
    client: LightningSeverityClient = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_concurrency: int = K_VOTES,
) -> dict:
    """Parallel-capable k=8 ballot for ONE building.

    Identical contract to request_lightning_ballot -- the same k=8
    independently sampled Lightning generations (each still its own
    sample_severity() call with the caller's temperature; nothing here
    replaces k=8 with one value repeated eight times) and the same
    aggregation (via _aggregate_votes) -- but the eight calls are issued
    concurrently through a bounded thread pool instead of one at a time.
    request_lightning_ballot's sequential path is unmodified and remains
    available for direct comparison.

    max_concurrency is clamped to [1, K_VOTES]: a single ballot never needs
    more than 8 concurrent requests, which keeps any one ballot well under
    the Lightning server's --max-num-seqs=16 without the caller having to
    think about it.
    """
    active_client = client if client is not None else _default_client
    bounded_workers = max(1, min(max_concurrency, K_VOTES))

    votes = [None] * K_VOTES
    with concurrent.futures.ThreadPoolExecutor(max_workers=bounded_workers) as executor:
        future_to_index = {
            executor.submit(active_client.sample_severity, building_context, temperature): index
            for index in range(K_VOTES)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            votes[future_to_index[future]] = future.result()

    return _aggregate_votes(votes)


def _voted_class(votes: list) -> int:
    """Modal severity label across votes, breaking ties via _break_tie."""
    counts = {label: votes.count(label) for label in _VALID_SEVERITY_LABELS}
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return tied[0]
    return _break_tie(tied)


def _break_tie(tied_classes: list) -> int:
    """UNRESOLVED DESIGN DECISION: when multiple severity labels are tied for
    the modal count, this picks the highest (most severe) tied class.

    The README defines voted_class as "the modal label" but says nothing
    about ties. Choosing max() here is a deliberate, deterministic,
    triage-conservative placeholder (when in doubt, treat the building as
    more damaged) -- NOT a policy the team has actually agreed on. Freeze the
    real tie rule with the team; this helper exists on its own so that
    decision can change without touching vote aggregation.
    """
    return max(tied_classes)
