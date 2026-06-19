"""rag-qdrant: local RAG skill for ingesting text/PDF/MD/photos into Qdrant and answering questions with an OpenAI-compatible chat endpoint.

Public API:
    - Flat functions: ingest_text, ingest_file, ingest_photo, ask,
      search, stats, ensure_collection, extract_photos, grant_access,
      revoke_access, show_access
    - Cache helpers: semantic_cache_stats, semantic_cache_clear,
      search_cache_stats, search_cache_clear
    - Admin gate: is_admin, admin_telegram_ids
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
from .config import Settings, admin_telegram_ids, is_admin, settings
from .inference import answer_question, ask, build_prompt
from .photo_matching import extract_photos
from .qdrant_store import (
    ALLOWED_TELEGRAM_IDS_FIELD,
    WILDCARD_TELEGRAM_ID,
    collection_stats,
    ensure_collection,
    grant_access,
    ingest_file,
    ingest_photo,
    ingest_text,
    revoke_access,
    search,
    show_access,
)
from .text_processing import chunk_text, extract_text, normalize_text
from .user_cache import record_seen, resolve_id, resolve_username

__version__ = "0.1.0"

# Alias: stats is the public short name for collection_stats.
stats = collection_stats

__all__ = [
    "AgentMessage",
    "AgentReply",
    "Attachment",
    "ALLOWED_TELEGRAM_IDS_FIELD",
    "Photo",
    "RAG",
    "Settings",
    "WILDCARD_TELEGRAM_ID",
    "admin_telegram_ids",
    "ask",
    "answer_question",
    "build_prompt",
    "chunk_text",
    "collection_stats",
    "ensure_collection",
    "extract_photos",
    "extract_text",
    "grant_access",
    "handle_message",
    "ingest_file",
    "ingest_photo",
    "ingest_text",
    "is_admin",
    "normalize_text",
    "record_seen",
    "resolve_id",
    "resolve_username",
    "revoke_access",
    "search",
    "search_cache_clear",
    "search_cache_stats",
    "semantic_cache_clear",
    "semantic_cache_stats",
    "settings",
    "show_access",
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

    def ingest_text(
        self,
        text: str,
        *,
        source: str,
        metadata: dict[str, Any] | None = None,
        allowed_telegram_ids: list[int | str] | None = None,
    ) -> int:
        return ingest_text(text, source=source, metadata=metadata, allowed_telegram_ids=allowed_telegram_ids)

    def ingest_file(
        self,
        path: Path,
        *,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        allowed_telegram_ids: list[int | str] | None = None,
    ) -> int:
        return ingest_file(path, source=source, metadata=metadata, allowed_telegram_ids=allowed_telegram_ids)

    def ingest_photo(
        self,
        photo: Photo,
        *,
        allowed_telegram_ids: list[int | str] | None = None,
    ) -> int:
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
            allowed_telegram_ids=allowed_telegram_ids,
        )

    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        allowed_telegram_id: int | str | None = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        return search(
            question,
            top_k=top_k,
            allowed_telegram_id=allowed_telegram_id,
            is_admin=is_admin,
        )

    def ask(
        self,
        question: str,
        *,
        current_telegram_id: int | str | None = None,
    ) -> dict[str, Any]:
        return ask(question, current_telegram_id=current_telegram_id)

    def grant_access(self, source: str, telegram_id: int | str) -> dict[str, Any]:
        return grant_access(source, telegram_id)

    def revoke_access(self, source: str, telegram_id: int | str) -> dict[str, Any]:
        return revoke_access(source, telegram_id)

    def show_access(self, source: str) -> dict[str, Any]:
        return show_access(source)

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
