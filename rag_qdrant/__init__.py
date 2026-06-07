"""rag-qdrant: local RAG skill for ingesting text/PDF/MD into Qdrant and answering questions with an OpenAI-compatible chat endpoint.

Public API:
    - Flat functions: ingest_text, ingest_file, ask, search, stats, ensure_collection
    - Cache helpers: semantic_cache_stats, semantic_cache_clear,
      search_cache_stats, search_cache_clear
    - Thin RAG class that delegates to the flat functions with a custom Settings
    - Agent-mode message handler: handle_message, AgentMessage, Attachment
    - settings, __version__
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_handler import AgentMessage, Attachment, handle_message
from .cache import (
    search_cache_clear,
    search_cache_stats,
    semantic_cache_clear,
    semantic_cache_stats,
)
from .config import Settings, settings
from .inference import answer_question, ask, build_prompt
from .qdrant_store import (
    collection_stats,
    ensure_collection,
    ingest_file,
    ingest_text,
    search,
)
from .text_processing import chunk_text, extract_text, normalize_text

__version__ = "0.1.0"

# Alias: stats is the public short name for collection_stats.
stats = collection_stats

__all__ = [
    "AgentMessage",
    "Attachment",
    "RAG",
    "Settings",
    "ask",
    "answer_question",
    "build_prompt",
    "chunk_text",
    "collection_stats",
    "ensure_collection",
    "extract_text",
    "handle_message",
    "ingest_file",
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
