"""Photo-context extraction shared between inference and the semantic cache.

A photo point in Qdrant has the same shape as any other point
(``text`` / ``source`` / ``score`` / ``payload``) plus a few photo-only
payload fields. This module is the single place that knows how to
recognize a photo payload and reshape it into the user-facing
``{"path", "filename", "source", "score"}`` record.

Used by:

- :func:`rag_qdrant.inference.answer_question` — to attach a
  ``photos`` list to every result dict it returns.
- :func:`rag_qdrant.cache.SemanticCache.lookup` — to attach the same
  field on a cache hit, so the user-facing answer carries the matched
  photos even when the search and LLM call were skipped.

Lives in its own tiny module to avoid a circular import between
:mod:`rag_qdrant.inference` and :mod:`rag_qdrant.cache`.
"""

from __future__ import annotations


def is_photo_payload(payload: dict) -> bool:
    """Return ``True`` if ``payload`` is a photo point (has ``kind == "photo"`` and a path)."""
    return bool(payload) and payload.get("kind") == "photo" and bool(payload.get("photo_path"))


def extract_photos(contexts: list[dict]) -> list[dict]:
    """Return one record per distinct photo path, in first-seen order.

    Each record is shaped ``{"path", "filename", "source", "score"}``.
    Dedupes on ``photo_path`` (the absolute path on disk) so a single
    photo that matches multiple chunks surfaces exactly once. Skips
    contexts whose payload is not a photo point.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for item in contexts:
        payload = item.get("payload") or {}
        if not is_photo_payload(payload):
            continue
        path = payload.get("photo_path") or ""
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(
            {
                "path": path,
                "filename": payload.get("photo_filename", "") or "",
                "source": item.get("source", "") or "",
                "score": item.get("score", 0.0),
            }
        )
    return out
