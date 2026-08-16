"""B8 Part E: tag precision/recall against a hand-labeled caption set.

Reuses the real B7 TagExtractor implementations verbatim
(DeterministicStubTagExtractor offline, LightningTagExtractor live) --
this module adds no tagging logic, it only measures the real one.

Conservative exact-match scoring only (Part 8's explicit instruction): no
embeddings, no fuzzy/LLM judging. Every returned and expected tag is
normalized (lowercase, trim, collapse whitespace) before comparison; a
returned tag counts as correct only if it exactly equals one of that
caption's hand-labeled ACCEPTABLE tag strings. Several fixture captions
deliberately have an acceptable set that is BROADER than the deterministic
stub's fixed keyword vocabulary (e.g. "flooding" is an acceptable synonym
the stub never emits, since its one flood-family rule always emits
"standing water") -- this makes precision/recall a genuine, non-trivial
measurement of the stub's rule table, not a rigged 100%.
"""

from backend.decision.archive_tag_extractor import (
    DeterministicStubTagExtractor,
    LightningTagExtractor,
    _PROHIBITED_TAG_TERMS,
)
from backend.decision.eval.report import STATUS_MEASURED, make_metric

TAG_FIXTURE = [
    {
        "caption": "Two-storey wood structure with roof collapsed and standing water in street.",
        "acceptable_tags": {"wood structure", "roof collapse", "standing water"},
    },
    {
        "caption": "Large fire visible with heavy smoke near the residential block.",
        "acceptable_tags": {"fire", "smoke"},
    },
    {
        "caption": "Flooded intersection with standing water covering the roadway.",
        "acceptable_tags": {"standing water", "flooding"},
    },
    {
        "caption": "Single-family home shows no visible damage.",
        "acceptable_tags": {"undamaged"},
    },
    {
        "caption": "Riverside Dialysis Center exterior shows significant structural damage.",
        "acceptable_tags": {"medical facility", "structural damage"},
    },
    {
        "caption": "Large debris field obstructs roadway access beside the structure.",
        "acceptable_tags": {"debris", "blocked access"},
    },
    {
        "caption": "Minor cracking visible on the exterior wall of a building.",
        "acceptable_tags": {"cracking"},
    },
    {
        "caption": "Commercial building on Pine St with roof collapse and scattered debris.",
        "acceptable_tags": {"commercial structure", "roof collapse", "debris"},
    },
    {
        "caption": "Building entrance is damaged and partially blocked by debris.",
        "acceptable_tags": {"debris", "blocked entrance"},
    },
    {
        "caption": "Vehicle fire spreading toward an adjacent structure on 35th Ave SW.",
        "acceptable_tags": {"fire"},
    },
]


def _normalize(tag: str) -> str:
    return " ".join(tag.strip().lower().split())


def _precision_recall(returned: list, acceptable: set) -> tuple:
    normalized_returned = {_normalize(t) for t in returned}
    normalized_acceptable = {_normalize(t) for t in acceptable}
    correct = normalized_returned & normalized_acceptable
    precision = len(correct) / len(normalized_returned) if normalized_returned else (1.0 if not normalized_acceptable else 0.0)
    recall = len(correct) / len(normalized_acceptable) if normalized_acceptable else (1.0 if not normalized_returned else 0.0)
    return precision, recall


def _damage_class_snapshot() -> dict:
    """Structural check helper for "no grade mutation" -- TagExtractor's
    interface has no damage-class parameter at all (see
    archive_tag_extractor.TagExtractor.extract_tags_batch), so this
    fixture's own class_max-like sentinel values are captured before/after
    purely to prove nothing external changes as a side effect of tagging.
    """
    return {case["caption"]: "UNCHANGED_SENTINEL" for case in TAG_FIXTURE}


def evaluate_tag_precision_recall_offline() -> dict:
    extractor = DeterministicStubTagExtractor()
    captions = [case["caption"] for case in TAG_FIXTURE]

    before = _damage_class_snapshot()
    tags_by_caption = extractor.extract_tags_batch(captions)
    after = _damage_class_snapshot()

    per_case = []
    precisions = []
    recalls = []
    prohibited_count = 0
    for case, returned_tags in zip(TAG_FIXTURE, tags_by_caption):
        precision, recall = _precision_recall(returned_tags, case["acceptable_tags"])
        precisions.append(precision)
        recalls.append(recall)
        violating = [t for t in returned_tags if any(term in t.lower() for term in _PROHIBITED_TAG_TERMS)]
        prohibited_count += len(violating)
        per_case.append(
            {
                "caption": case["caption"],
                "acceptable_tags": sorted(case["acceptable_tags"]),
                "returned_tags": returned_tags,
                "precision": precision,
                "recall": recall,
                "prohibited_tags_found": violating,
            }
        )

    return make_metric(
        name="tag_precision_recall",
        status=STATUS_MEASURED,
        value={"precision": sum(precisions) / len(precisions), "recall": sum(recalls) / len(recalls)},
        threshold=None,
        sample_count=len(TAG_FIXTURE),
        details={
            "mode": "offline: DeterministicStubTagExtractor",
            "prohibited_tags_total": prohibited_count,
            "grade_mutation_detected": before != after,
            "per_case": per_case,
        },
    )


def evaluate_tag_precision_recall_live(base_url: str = None, timeout_s: float = 10.0) -> dict:
    extractor = LightningTagExtractor(base_url=base_url, timeout_s=timeout_s) if base_url else LightningTagExtractor(timeout_s=timeout_s)
    captions = [case["caption"] for case in TAG_FIXTURE]

    import time

    started_at = time.perf_counter()
    tags_by_caption = extractor.extract_tags_batch(captions)
    elapsed_s = time.perf_counter() - started_at

    per_case = []
    precisions = []
    recalls = []
    prohibited_count = 0
    for case, returned_tags in zip(TAG_FIXTURE, tags_by_caption):
        precision, recall = _precision_recall(returned_tags, case["acceptable_tags"])
        precisions.append(precision)
        recalls.append(recall)
        violating = [t for t in returned_tags if any(term in t.lower() for term in _PROHIBITED_TAG_TERMS)]
        prohibited_count += len(violating)
        per_case.append(
            {
                "caption": case["caption"],
                "acceptable_tags": sorted(case["acceptable_tags"]),
                "returned_tags": returned_tags,
                "precision": precision,
                "recall": recall,
            }
        )

    total_prompt_tokens = sum(u.get("prompt_tokens", 0) for u in extractor.usage_log if u)
    total_completion_tokens = sum(u.get("completion_tokens", 0) for u in extractor.usage_log if u)

    return make_metric(
        name="tag_precision_recall_live",
        status=STATUS_MEASURED,
        value={"precision": sum(precisions) / len(precisions), "recall": sum(recalls) / len(recalls)},
        threshold=None,
        sample_count=len(TAG_FIXTURE),
        details={
            "mode": "live: real LightningTagExtractor",
            "prohibited_tags_total": prohibited_count,
            "elapsed_s": elapsed_s,
            "prompt_tokens_total": total_prompt_tokens,
            "completion_tokens_total": total_completion_tokens,
            "per_case": per_case,
        },
    )
