#!/usr/bin/env python3
"""B7 opt-in live check for Lightning batch tag extraction, against the
real Lightning server at localhost:8001.

Not part of the pytest suite. Uses the REAL production LightningTagExtractor
(one batched request, /no_think, json_schema response_format,
temperature=0.0) against a small batch of fixture captions
(tests/decision/archive_fixture_data.FIXTURE_ELIGIBLE_ITEMS) -- nothing
here reimplements the HTTP call or the post-validation pipeline.

Verifies, and prints an explicit PASS/FAIL for each:
  - structured output was valid (extract_tags_batch returned without raising)
  - no prohibited person/body/clothing tag survived post-validation
  - damage grades are untouched (TagExtractor never receives/returns one --
    structural, not just observed)
  - whole-batch latency
  - total prompt/completion token usage (client.usage_log)

Usage:
    python scripts/archive_tagging_live_check.py
    FIRSTLIGHT_LIGHTNING_BASE_URL=http://localhost:8001 python scripts/archive_tagging_live_check.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decision.archive_tag_extractor import (  # noqa: E402
    _PROHIBITED_TAG_TERMS,
    LightningTagExtractor,
)
from backend.decision.lightning_client import LightningClientError  # noqa: E402
from tests.decision.archive_fixture_data import FIXTURE_ELIGIBLE_ITEMS  # noqa: E402

BATCH_SIZE = 8


def _contains_prohibited_term(tag: str) -> bool:
    lowered = tag.lower()
    return any(term in lowered for term in _PROHIBITED_TAG_TERMS)


def main():
    captions = [item["caption"] for item in FIXTURE_ELIGIBLE_ITEMS[:BATCH_SIZE]]
    class_max_before = {item["image_id"]: item["class_max"] for item in FIXTURE_ELIGIBLE_ITEMS[:BATCH_SIZE]}

    client = LightningTagExtractor()
    print(f"Target: {client.base_url}/v1/chat/completions (model=lightning, /no_think, json_schema, temperature=0.0)")
    print(f"Batch size: {len(captions)}\n")

    start = time.perf_counter()
    try:
        tags_by_caption = client.extract_tags_batch(captions)
        structured_output_ok = True
    except LightningClientError as exc:
        print(f"FAIL: Lightning batch tag extraction raised: {exc}")
        sys.exit(1)
    elapsed_s = time.perf_counter() - start

    print(f"structured output: {'PASS' if structured_output_ok else 'FAIL'}\n")

    any_prohibited = False
    for item, tags in zip(FIXTURE_ELIGIBLE_ITEMS[:BATCH_SIZE], tags_by_caption):
        print(f"caption: {item['caption']}")
        print(f"tags: {tags}")
        violating = [t for t in tags if _contains_prohibited_term(t)]
        if violating:
            any_prohibited = True
            print(f"  FAIL: prohibited term(s) survived post-validation: {violating}")
        print()

    print(f"no prohibited person/body/clothing tags: {'FAIL' if any_prohibited else 'PASS'}")

    class_max_after = {item["image_id"]: item["class_max"] for item in FIXTURE_ELIGIBLE_ITEMS[:BATCH_SIZE]}
    grade_unchanged = class_max_before == class_max_after
    print(f"damage grades untouched: {'PASS' if grade_unchanged else 'FAIL'}")

    print(f"\nwhole-batch latency: {elapsed_s:.2f}s")

    total_prompt_tokens = sum(u.get("prompt_tokens", 0) for u in client.usage_log if u)
    total_completion_tokens = sum(u.get("completion_tokens", 0) for u in client.usage_log if u)
    print(f"prompt_tokens_total: {total_prompt_tokens}")
    print(f"completion_tokens_total: {total_completion_tokens}")

    if any_prohibited or not grade_unchanged:
        sys.exit(1)


if __name__ == "__main__":
    main()
