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

_embedding_model: TextEmbedding | None = None
_client: QdrantClient | None = None


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


def ensure_payload_indexes() -> None:
    client = get_qdrant_client()
    field_schemas = {
        "source": models.PayloadSchemaType.KEYWORD,
        "file_name": models.PayloadSchemaType.KEYWORD,
        "file_type": models.PayloadSchemaType.KEYWORD,
        "telegram_user_id": models.PayloadSchemaType.INTEGER,
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


def ingest_text(text: str, *, source: str, metadata: dict[str, Any] | None = None) -> int:
    ensure_collection()
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("ingest_text_empty source=%s", source)
        return 0

    vectors = embed_texts(chunks, query=False)
    points: list[models.PointStruct] = []
    metadata = metadata or {}
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            models.PointStruct(
                id=_point_id(source, index, chunk),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": source,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    **metadata,
                },
            )
        )

    get_qdrant_client().upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
    logger.info("ingest_text_done source=%s chunks=%s", source, len(points))
    return len(points)


def ingest_file(path: Path, *, source: str | None = None, metadata: dict[str, Any] | None = None) -> int:
    text = extract_text(path)
    source_name = source or path.name
    payload_metadata = {"file_name": path.name, "file_type": path.suffix.lower(), **(metadata or {})}
    logger.info("ingest_file_start path=%s source=%s", path, source_name)
    count = ingest_text(text, source=source_name, metadata=payload_metadata)
    logger.info("ingest_file_done path=%s source=%s chunks=%s", path, source_name, count)
    return count


def search(question: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
    ensure_collection()
    top_k = top_k or settings.top_k
    query_vector = embed_texts([question], query=True)[0]
    client = get_qdrant_client()
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
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
