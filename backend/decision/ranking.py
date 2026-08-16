"""Deterministic rank ordering for FIRST LIGHT.

Sorts already-scored RankItems for display. Never recalculates or mutates
priority, never mutates input items, and never calls an LLM.

Ordering rule:
1. confirmed-severe items first (confirmed == True and damage_class >= 2)
2. within confirmed-severe items: priority descending
3. all remaining items afterward
4. within remaining items: priority descending

Pinning a confirmed-severe item to the front never changes its priority value;
only the returned order changes.
"""


def _is_confirmed_severe(item: dict) -> bool:
    return item["confirmed"] is True and item["damage_class"] >= 2


def rank_items(items: list) -> list:
    """Return a new list of items sorted per the FIRST LIGHT ordering rule.

    Does not mutate the input list or any item in it, and does not touch
    item["priority"].
    """
    return sorted(
        items,
        key=lambda item: (not _is_confirmed_severe(item), -item["priority"]),
    )
