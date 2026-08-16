"""B7 Parts 9-10: the ONLY writer B7 exposes for the search index, plus the
minimal edit/reindex primitive C6 will later call.

CRITICAL INVARIANT: index_cleared_archive_item requires the caller to pass
`eligible=True` EXPLICITLY. This is the pending A-side integration seam --
Member A's real privacy gate/archive boundary is what will eventually
supply that boolean (e.g. `archive_item_source["status"] == "cleared"`
computed upstream of this call). B7 never runs, reimplements, or
second-guesses the privacy/person-detector model itself; it only refuses
to write anything the caller has not explicitly asserted is cleared. Any
`eligible` value other than the literal True (False, None, missing,
"true", 1, ...) is refused -- deliberately strict, so a withheld/ineligible
record cannot be indexed by a careless caller doing `eligible=some_status`
where some_status happens to be truthy. See tests/decision/archive_fixture_data.py
for a fixture record that proves this refusal end to end (an
ineligible source record that never becomes a row, and never appears in
any search result).

archive_write.py never talks to Lightning or the embedder's model weights
directly except by calling the SAME Embedder.embed_text() abstraction
(Part 2) every other caller uses -- captions are never hand-embedded here.
"""

from backend.decision import archive_store
from backend.decision.archive_embedder import Embedder

_VALID_CLASS_MAX = (0, 1, 2, 3)


class ArchiveWriteError(ValueError):
    """Raised when index_cleared_archive_item/reindex_archive_item is asked
    to do something that would violate the privacy/eligibility invariant,
    the frozen ArchiveItem shape, or the "edit never creates a new row"
    rule. A contract failure, never silently corrected.
    """


def _validate_archive_item_shape(item: dict) -> list:
    errors = []
    if not isinstance(item.get("image_id"), str) or not item.get("image_id"):
        errors.append("image_id must be a non-empty string")
    if not isinstance(item.get("thumb_path"), str) or not item.get("thumb_path"):
        errors.append("thumb_path must be a non-empty string")

    captured_at = item.get("captured_at")
    if not isinstance(captured_at, (int, float)) or isinstance(captured_at, bool):
        errors.append("captured_at must be a number (unix timestamp)")

    centroid = item.get("centroid")
    if centroid is not None and (
        not isinstance(centroid, (list, tuple))
        or len(centroid) != 2
        or not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in centroid)
    ):
        errors.append("centroid must be null or exactly [lng, lat]")

    if not isinstance(item.get("needs_geo"), bool):
        errors.append("needs_geo must be a bool")

    if not isinstance(item.get("caption"), str) or not item.get("caption"):
        errors.append("caption must be a non-empty string")

    tags = item.get("tags")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        errors.append("tags must be a list of strings")

    if item.get("class_max") not in _VALID_CLASS_MAX:
        errors.append(f"class_max must be one of {_VALID_CLASS_MAX}, got {item.get('class_max')!r}")

    if not isinstance(item.get("key_evidence"), bool):
        errors.append("key_evidence must be a bool")

    return errors


def index_cleared_archive_item(
    conn,
    archive_item: dict,
    embedder: Embedder,
    eligible: bool,
    sector: str = None,
) -> None:
    """Persist ONE already-cleared ArchiveItem (+ its caption embedding,
    computed here via `embedder`) into the search index.

    `eligible` MUST be the literal boolean True -- see module docstring.
    `sector`, if given, is stored as internal-only metadata (never a public
    ArchiveItem field) purely so the `sector:<value>` structured filter
    token can narrow on it.

    Raises ArchiveWriteError (never writes anything) if eligible is not
    True, or if archive_item fails the frozen public ArchiveItem shape
    check.
    """
    if eligible is not True:
        raise ArchiveWriteError(
            f"refusing to index {archive_item.get('image_id')!r}: caller did not assert "
            "eligible=True (a withheld/ineligible archive record must never be indexed)"
        )

    errors = _validate_archive_item_shape(archive_item)
    if errors:
        raise ArchiveWriteError("; ".join(errors))

    embedding = embedder.embed_text(archive_item["caption"])
    archive_store.upsert_item(conn, archive_item, embedding, sector)


def reindex_archive_item(
    conn,
    image_id: str,
    embedder: Embedder,
    caption: str = None,
    tags: list = None,
    key_evidence: bool = None,
) -> dict:
    """Update an ALREADY-INDEXED item's caption/tags/key_evidence (the
    backend primitive C6's caption/tag editing UI will call later -- no UI
    here). Never creates a new row: raises ArchiveWriteError if `image_id`
    is not already present, so this can never become a second ingest path
    around the privacy gate -- only items index_cleared_archive_item
    already accepted can ever be edited.

    If `caption` changes, the embedding is regenerated through the SAME
    Embedder abstraction (never hand-rolled here) -- README marks
    re-embed-on-save as stretch, so this direct primitive is the minimal
    version: synchronous, no queue, no batching. `tags`, if given,
    replaces the stored tag list verbatim (callers wanting Lightning-
    generated tags call archive_tag_extractor themselves and pass the
    result in).

    Returns the updated public ArchiveItem.
    """
    existing = archive_store.get_item(conn, image_id)
    if existing is None:
        raise ArchiveWriteError(f"cannot edit {image_id!r}: not present in the archive index")

    if tags is not None and (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)):
        raise ArchiveWriteError("tags must be a list of strings")

    embedding = embedder.embed_text(caption) if caption is not None else None

    archive_store.update_item_fields(
        conn, image_id, caption=caption, tags=tags, embedding=embedding, key_evidence=key_evidence
    )
    return archive_store.get_item(conn, image_id)
