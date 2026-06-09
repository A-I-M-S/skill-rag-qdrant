"""rag-qdrant: local RAG skill for ingesting text/PDF/MD/photos into Qdrant and answering questions with an OpenAI-compatible chat endpoint.

Public API:
    - Flat functions: ingest_text, ingest_file, ingest_photo, ask,
      search, stats, ensure_collection, extract_photos
    - Cache helpers: semantic_cache_stats, semantic_cache_clear,
      search_cache_stats, search_cache_clear
    - Thin RAG class that delegates to the flat functions with a
      custom Settings
    - Agent-mode message handler: handle_message, AgentMessage,
      AgentReply, Attachment, Photo
    - settings, __version__
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_handler import AgentMessage, AgentReply, Attachment, Photo, handle_message
from .cache import (
    search_cache_clear,
    search_cache_stats,
    semantic_cache_clear,
    semantic_cache_stats,
)
from .config import Settings, settings
from .inference import answer_question, ask, build_prompt
from .photo_matching import extract_photos
from .qdrant_store import (
    collection_stats,
    ensure_collection,
    ingest_file,
    ingest_photo,
    ingest_text,
    search,
)
from .text_processing import chunk_text, extract_text, normalize_text

__version__ = "0.1.0"

# Alias: stats is the public short name for collection_stats.
stats = collection_stats

__all__ = [
    "AgentMessage",
    "AgentReply",
    "Attachment",
    "Photo",
    "RAG",
    "Settings",
    "ask",
    "answer_question",
    "build_prompt",
    "chunk_text",
    "collection_stats",
    "ensure_collection",
    "extract_photos",
    "extract_text",
    "handle_message",
    "ingest_file",
    "ingest_photo",
    "ingest_text",
    "normalize_text",
    "search",
    "search_cache_clear",
    "search_cache_stats",
    "semantic_cache_clear",
    "semantic_cache_stats",
    "settings",
    "stats",
    "__version__",
]


class RAG:
    """Thin convenience class. Every method delegates to the corresponding flat function.

    Example:
        from rag_qdrant import RAG

        rag = RAG()                              # uses default settings
        rag.ingest_text("hello world", source="note")
        result = rag.ask("what was said?")
    """

    def __init__(self, custom_settings: Settings | None = None) -> None:
        self._settings = custom_settings or settings

    @property
    def settings(self) -> Settings:
        return self._settings

    def ensure_collection(self) -> None:
        ensure_collection()

    def ingest_text(self, text: str, *, source: str, metadata: dict[str, Any] | None = None) -> int:
        return ingest_text(text, source=source, metadata=metadata)

    def ingest_file(self, path: Path, *, source: str | None = None, metadata: dict[str, Any] | None = None) -> int:
        return ingest_file(path, source=source, metadata=metadata)

    def ingest_photo(self, photo: Photo) -> int:
        """Save a photo to disk and embed its description in the corpus.

        Equivalent to :func:`rag_qdrant.photo_store.save_photo` followed
        by :func:`rag_qdrant.qdrant_store.ingest_photo`. Returns the
        chunk count (1 for a valid description).
        """
        from .photo_store import save_photo as _save_photo

        path, sha256_hex, source = _save_photo(photo)
        return ingest_photo(
            path,
            description=photo.description,
            source=source,
            photo_filename=photo.filename,
            sha256_hex=sha256_hex,
            file_type=Path(photo.filename).suffix.lower(),
        )

    def search(self, question: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        return search(question, top_k=top_k)

    def ask(self, question: str) -> dict[str, Any]:
        return ask(question)

    def stats(self) -> dict[str, Any]:
        return stats()

    def semantic_cache_stats(self) -> dict[str, Any]:
        return semantic_cache_stats()

    def semantic_cache_clear(self) -> int:
        return semantic_cache_clear()

    def search_cache_stats(self) -> dict[str, Any]:
        return search_cache_stats()

    def search_cache_clear(self) -> int:
        return search_cache_clear()
