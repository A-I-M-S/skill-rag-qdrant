from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import settings
from .logging_setup import logger
from .text_processing import chunk_text, extract_text, normalize_text
from .cache import search_cache_invalidate, search_cache_lookup, search_cache_store

_embedding_model: TextEmbedding | None = None
_client: QdrantClient | None = None

ALLOWED_TELEGRAM_IDS_FIELD = "allowed_telegram_ids"
WILDCARD_TELEGRAM_ID = "*"

PUBLIC_ID = "public"


def register_custom_fastembed_model_if_needed() -> None:
    supported = {item["model"] for item in TextEmbedding.list_supported_models()}
    if settings.fastembed_model in supported:
        return
    if settings.fastembed_model == "intfloat/multilingual-e5-small":
        logger.info("embedding_model_register_custom model=%s", settings.fastembed_model)
        TextEmbedding.add_custom_model(
            model="intfloat/multilingual-e5-small",
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="Xenova/multilingual-e5-small"),
            dim=384,
            model_file="onnx/model.onnx",
            description="Custom FastEmbed registration for multilingual E5 small; prefixes query:/passage: are required.",
            license="mit",
            size_in_gb=0.47,
        )
        return
    raise ValueError(
        f"Model {settings.fastembed_model} is not supported by FastEmbed. "
        "Use TextEmbedding.list_supported_models() or add a custom model registration."
    )


def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        register_custom_fastembed_model_if_needed()
        logger.info("embedding_model_load_start model=%s", settings.fastembed_model)
        _embedding_model = TextEmbedding(model_name=settings.fastembed_model)
        logger.info("embedding_model_load_done model=%s", settings.fastembed_model)
    return _embedding_model


def embed_texts(texts: list[str], *, query: bool = False) -> list[list[float]]:
    prefix = "query: " if query else "passage: "
    prepared = [prefix + normalize_text(text) for text in texts]
    vectors = [vector.tolist() for vector in get_embedding_model().embed(prepared)]
    logger.info("embed_texts_done count=%s query=%s", len(vectors), query)
    return vectors


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        settings.require_qdrant()
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60)
    return _client


def _normalize_telegram_id(value: int | str) -> str:
    """Coerce a telegram id to a stable string for the ACL payload field.

    The payload index is a ``keyword`` field. Qdrant's ``match`` filter
    is exact-equality, so ``123`` and ``"123"`` would be treated as
    distinct values. Normalizing to ``str(int)`` (or the input verbatim
    when it is already a non-numeric string such as the ``"*"``
    wildcard) ensures ``grant_access`` / ``search`` compare against
    the same stored form.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text == WILDCARD_TELEGRAM_ID:
        return WILDCARD_TELEGRAM_ID
    try:
        return str(int(text))
    except (TypeError, ValueError):
        return text


def _normalize_allowed_telegram_ids(values: list[int | str] | None) -> list[str] | None:
    """Normalize a list of telegram ids for storage.

    Returns ``None`` when the caller passes ``None`` *or* an empty list
    — both mean "no restriction, public" (per user spec: "if not
    specified, anyone can access"). The payload field is then omitted
    at ingest time so legacy chunks with no field are public by default.
    """
    if values is None:
        return None
    out: list[str] = []
    for v in values:
        norm = _normalize_telegram_id(v)
        if norm not in out:
            out.append(norm)
    if not out:
        return None
    if WILDCARD_TELEGRAM_ID in out and len(out) > 1:
        out = [WILDCARD_TELEGRAM_ID]
    return out


def _build_acl_filter(allowed_telegram_id: int | str | None, *, is_admin: bool) -> models.Filter | None:
    """Build the Qdrant payload filter for the access-control list.

    Semantics (per user spec: "if not specified, anyone can access"):

    - ``is_admin=True`` → no filter (admins see everything).
    - non-admin with an id → chunk is visible if its
      ``allowed_telegram_ids`` field is missing/empty (public/legacy),
      contains that id, or contains the ``"*"`` wildcard.
    - non-admin with no id (CLI default path) → chunk is visible if
      its ``allowed_telegram_ids`` field is missing/empty (public/legacy)
      or contains the ``"*"`` wildcard.

    Note on Qdrant semantics: ``is_empty`` matches both missing fields
    and empty arrays. Since ``_normalize_allowed_telegram_ids`` collapses
    ``[]`` to ``None`` (omitted at ingest), and ``revoke_access`` deletes
    the field when the list becomes empty, in practice
    ``is_empty`` here means "public/legacy" — never an explicit
    no-access mode.
    """
    if is_admin:
        return None
    wildcard = models.FieldCondition(
        key=ALLOWED_TELEGRAM_IDS_FIELD,
        match=models.MatchValue(value=WILDCARD_TELEGRAM_ID),
    )
    is_empty_field = models.FieldCondition(
        key=ALLOWED_TELEGRAM_IDS_FIELD,
        is_empty=True,
    )
    if allowed_telegram_id is None:
        return models.Filter(should=[wildcard, is_empty_field])
    norm = _normalize_telegram_id(allowed_telegram_id)
    user_match = models.FieldCondition(
        key=ALLOWED_TELEGRAM_IDS_FIELD,
        match=models.MatchValue(value=norm),
    )
    return models.Filter(should=[user_match, wildcard, is_empty_field])


def ensure_payload_indexes() -> None:
    client = get_qdrant_client()
    field_schemas = {
        "source": models.PayloadSchemaType.KEYWORD,
        "file_name": models.PayloadSchemaType.KEYWORD,
        "file_type": models.PayloadSchemaType.KEYWORD,
        "kind": models.PayloadSchemaType.KEYWORD,
        ALLOWED_TELEGRAM_IDS_FIELD: models.PayloadSchemaType.KEYWORD,
    }
    for field, schema in field_schemas.items():
        try:
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=schema,
            )
            logger.info("qdrant_payload_index_create_done collection=%s field=%s", settings.qdrant_collection, field)
        except Exception as exc:
            logger.info("qdrant_payload_index_exists_or_skipped collection=%s field=%s error=%s", settings.qdrant_collection, field, exc)


def ensure_collection() -> None:
    client = get_qdrant_client()
    existing = {collection.name for collection in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        logger.info("qdrant_collection_exists collection=%s", settings.qdrant_collection)
        ensure_payload_indexes()
        return
    logger.info(
        "qdrant_collection_create_start collection=%s dim=%s",
        settings.qdrant_collection,
        settings.embedding_dim,
    )
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(size=settings.embedding_dim, distance=models.Distance.COSINE),
    )
    ensure_payload_indexes()
    logger.info("qdrant_collection_create_done collection=%s", settings.qdrant_collection)


def _point_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def ingest_text(
    text: str,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
    allowed_telegram_ids: list[int | str] | None = None,
) -> int:
    ensure_collection()
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("ingest_text_empty source=%s", source)
        return 0

    vectors = embed_texts(chunks, query=False)
    points: list[models.PointStruct] = []
    metadata = metadata or {}
    normalized_ids = _normalize_allowed_telegram_ids(allowed_telegram_ids)
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload: dict[str, Any] = {
            "text": chunk,
            "source": source,
            "chunk_index": index,
            "chunk_count": len(chunks),
            **metadata,
        }
        if normalized_ids is not None:
            payload[ALLOWED_TELEGRAM_IDS_FIELD] = normalized_ids
        points.append(
            models.PointStruct(
                id=_point_id(source, index, chunk),
                vector=vector,
                payload=payload,
            )
        )

    get_qdrant_client().upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
    search_cache_invalidate()
    logger.info("ingest_text_done source=%s chunks=%s", source, len(points))
    return len(points)


def ingest_file(
    path: Path,
    *,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
    allowed_telegram_ids: list[int | str] | None = None,
) -> int:
    text = extract_text(path)
    source_name = source or path.name
    payload_metadata = {"file_name": path.name, "file_type": path.suffix.lower(), **(metadata or {})}
    logger.info("ingest_file_start path=%s source=%s", path, source_name)
    count = ingest_text(
        text,
        source=source_name,
        metadata=payload_metadata,
        allowed_telegram_ids=allowed_telegram_ids,
    )
    logger.info("ingest_file_done path=%s source=%s chunks=%s", path, source_name, count)
    return count


def ingest_photo(
    photo_path: Path,
    *,
    description: str,
    source: str,
    photo_filename: str,
    sha256_hex: str,
    file_type: str,
    allowed_telegram_ids: list[int | str] | None = None,
) -> int:
    """Ingest a photo's description as a single chunk in the Qdrant collection.

    The photo bytes are NOT embedded — only the user-supplied
    description is. The payload carries enough metadata for the
    agent handler to surface the saved photo path on future matches:

    - ``text`` = the description (set by :func:`ingest_text`)
    - ``source`` = ``photo-<sha256[:12]>`` (caller-supplied)
    - ``photo_path`` = absolute path to the bytes on disk
    - ``photo_filename`` = original filename
    - ``file_type`` = lowercase extension including the dot
    - ``kind`` = ``"photo"`` (used by :mod:`rag_qdrant.photo_matching`
      to recognize the point on a search hit)
    - ``sha256`` = full hex digest of the bytes
    - ``allowed_telegram_ids`` (optional) = list of telegram ids
      permitted to retrieve this point; see :mod:`rag_qdrant.config`
      for the access-control model.
    """
    resolved = photo_path.resolve()
    payload_metadata = {
        "photo_path": str(resolved),
        "photo_filename": photo_filename,
        "file_type": file_type,
        "kind": "photo",
        "sha256": sha256_hex,
    }
    logger.info(
        "ingest_photo_start path=%s source=%s description_chars=%s",
        resolved, source, len(description or ""),
    )
    count = ingest_text(
        description,
        source=source,
        metadata=payload_metadata,
        allowed_telegram_ids=allowed_telegram_ids,
    )
    logger.info("ingest_photo_done path=%s source=%s chunks=%s", resolved, source, count)
    return count


def search(
    question: str,
    *,
    top_k: int | None = None,
    query_vector: list[float] | None = None,
    allowed_telegram_id: int | str | None = None,
    is_admin: bool = False,
) -> list[dict[str, Any]]:
    ensure_collection()
    top_k = top_k or settings.top_k
    bypass_cache = allowed_telegram_id is not None and not is_admin
    if settings.search_cache_enabled and not bypass_cache:
        cached = search_cache_lookup(question, top_k=top_k)
        if cached is not None:
            logger.info("search_done question_chars=%s top_k=%s results=%s source=cache", len(question), top_k, len(cached))
            return cached
    if query_vector is None:
        query_vector = embed_texts([question], query=True)[0]
    client = get_qdrant_client()
    acl_filter = _build_acl_filter(allowed_telegram_id, is_admin=is_admin)
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=acl_filter,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=acl_filter,
        )
    formatted = [
        {
            "score": hit.score,
            "id": str(hit.id),
            "text": (hit.payload or {}).get("text", ""),
            "source": (hit.payload or {}).get("source", ""),
            "chunk_index": (hit.payload or {}).get("chunk_index"),
            "payload": hit.payload or {},
        }
        for hit in hits
    ]
    logger.info("search_done question_chars=%s top_k=%s results=%s", len(question), top_k, len(formatted))
    if settings.search_cache_enabled and not bypass_cache:
        search_cache_store(question, formatted, top_k=top_k)
    return formatted


def collection_stats() -> dict[str, Any]:
    ensure_collection()
    info = get_qdrant_client().get_collection(settings.qdrant_collection)
    stats = {
        "collection": settings.qdrant_collection,
        "points_count": getattr(info, "points_count", None),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "status": str(getattr(info, "status", "unknown")),
    }
    logger.info("collection_stats %s", stats)
    return stats


# ---------------------------------------------------------------------------
# Access-control admin tools (issue #2)
# ---------------------------------------------------------------------------

def _source_filter(source: str) -> models.Filter:
    """Build a Qdrant filter that matches every point with ``source=<source>``."""
    return models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
    )


def grant_access(source: str, telegram_id: int | str) -> dict[str, Any]:
    """Append ``telegram_id`` to the ACL of every point with ``source=<source>``.

    - If the field is missing on a point, it is set to ``[telegram_id]``.
    - If ``telegram_id == "*"``, the field is set/replaced to ``["*"]``.
    - Otherwise the id is appended (deduped). The new list is then
      written back via :func:`QdrantClient.set_payload`.

    Returns a small dict ``{"source": ..., "telegram_id": ..., "updated": N}``
    that the LLM can relay to the admin.
    """
    ensure_collection()
    norm = _normalize_telegram_id(telegram_id)
    logger.info("grant_access_start source=%s telegram_id=%s", source, norm)
    client = get_qdrant_client()
    flt = _source_filter(source)
    updated = 0
    next_offset = None
    while True:
        scroll_result = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        points, next_offset = scroll_result
        if not points:
            break
        for pt in points:
            payload = dict(pt.payload or {})
            existing = payload.get(ALLOWED_TELEGRAM_IDS_FIELD)
            if norm == WILDCARD_TELEGRAM_ID:
                new_value: list[str] = [WILDCARD_TELEGRAM_ID]
            else:
                if existing is None:
                    new_value = [norm]
                else:
                    new_value = [str(x) for x in existing if x is not None]
                    if norm not in new_value:
                        new_value.append(norm)
            payload[ALLOWED_TELEGRAM_IDS_FIELD] = new_value
            client.set_payload(
                collection_name=settings.qdrant_collection,
                payload={ALLOWED_TELEGRAM_IDS_FIELD: new_value},
                points=[pt.id],
            )
            updated += 1
        if next_offset is None:
            break
    search_cache_invalidate()
    logger.info("grant_access_done source=%s telegram_id=%s updated=%s", source, norm, updated)
    return {"source": source, "telegram_id": norm, "updated": updated}


def revoke_access(source: str, telegram_id: int | str) -> dict[str, Any]:
    """Remove ``telegram_id`` from the ACL of every point with ``source=<source>``.

    If the list becomes empty, the field is *deleted* (not left as
    ``[]``) so the point reverts to public. This keeps Qdrant's
    ``is_empty`` predicate aligned with "public" rather than as a
    no-access marker. Returns ``{"source": ..., "telegram_id": ...,
    "updated": N, "removed": N}`` for the LLM to relay.
    """
    ensure_collection()
    norm = _normalize_telegram_id(telegram_id)
    logger.info("revoke_access_start source=%s telegram_id=%s", source, norm)
    client = get_qdrant_client()
    flt = _source_filter(source)
    updated = 0
    removed = 0
    next_offset = None
    while True:
        scroll_result = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        points, next_offset = scroll_result
        if not points:
            break
        for pt in points:
            payload = pt.payload or {}
            existing = payload.get(ALLOWED_TELEGRAM_IDS_FIELD)
            if existing is None:
                continue
            current = [str(x) for x in existing if x is not None]
            if norm not in current:
                continue
            new_value = [x for x in current if x != norm]
            if new_value:
                client.set_payload(
                    collection_name=settings.qdrant_collection,
                    payload={ALLOWED_TELEGRAM_IDS_FIELD: new_value},
                    points=[pt.id],
                )
            else:
                # List is now empty → delete the field so the point
                # reverts to public. This keeps is_empty aligned with
                # "public" and avoids the no-access semantic.
                client.delete_payload(
                    collection_name=settings.qdrant_collection,
                    keys=[ALLOWED_TELEGRAM_IDS_FIELD],
                    points=[pt.id],
                )
            updated += 1
            removed += 1
        if next_offset is None:
            break
    search_cache_invalidate()
    logger.info("revoke_access_done source=%s telegram_id=%s updated=%s removed=%s", source, norm, updated, removed)
    return {"source": source, "telegram_id": norm, "updated": updated, "removed": removed}


def show_access(source: str) -> dict[str, Any]:
    """Return the current ACL and chunk count for a given ``source``.

    Reads via Qdrant scroll with a ``source=<source>`` filter; does not
    write. The returned ``allowed_telegram_ids`` is the value stored on
    the first chunk (or ``[]`` when no chunk has the field). When no
    chunk exists for that source, returns
    ``{"source": ..., "allowed_telegram_ids": [], "chunk_count": 0}``.
    """
    ensure_collection()
    client = get_qdrant_client()
    flt = _source_filter(source)
    chunk_count = 0
    first_value: list[str] = []
    next_offset = None
    while True:
        scroll_result = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        points, next_offset = scroll_result
        if not points:
            break
        for pt in points:
            chunk_count += 1
            if chunk_count == 1:
                existing = (pt.payload or {}).get(ALLOWED_TELEGRAM_IDS_FIELD)
                if existing is not None:
                    first_value = [str(x) for x in existing if x is not None]
        if next_offset is None:
            break
    result = {
        "source": source,
        "allowed_telegram_ids": first_value,
        "chunk_count": chunk_count,
    }
    logger.info("show_access source=%s chunk_count=%s ids=%s", source, chunk_count, first_value)
    return result
