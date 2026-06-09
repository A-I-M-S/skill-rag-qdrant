"""Caching layer for rag-qdrant.

Two opt-in caches, both backed by per-cache SQLite files in the project's
``logs/`` directory by default:

1. **Semantic cache** (``SemanticCache``)
   Keyed by cosine similarity between the new question's embedding and
   the stored questions' embeddings. Stores the LLM answer plus the
   contexts that produced it. Hit when a stored question is similar
   enough (default threshold 0.88) to the incoming question.

2. **Search cache** (``SearchCache``)
   Keyed by a deterministic hash of the normalized question text plus
   the settings that affect the result (``top_k``,
   ``min_relevance_score``, ``qdrant_collection``, ``fastembed_model``).
   Stores the raw Qdrant search contexts. Hit when the *exact* same
   search is repeated. A small in-process LRU sits in front of the
   SQLite file for hot-path repeats.

Design notes / tradeoffs
------------------------

- Both caches are **disabled by default**. Set ``SEMANTIC_CACHE_ENABLED=1``
  and/or ``SEARCH_CACHE_ENABLED=1`` in the environment to opt in. When
  disabled, the wrappers short-circuit to one boolean read and never
  touch SQLite.

- The semantic cache is **not invalidated on ingest**, by design.
  Cached answers may become slightly stale after new content is
  ingested; the next miss-after-TTL will pick up the new content.
  Wiping the semantic cache on every ingest would defeat the point
  of caching in any non-static corpus. The search cache, by contrast,
  *is* invalidated on every successful ``ingest_text`` / ``ingest_file``
  call (see :func:`search_cache_invalidate`).

- "No relevant information found" answers are cached with a separate,
  shorter TTL (``SEMANTIC_CACHE_MISS_TTL_SECONDS``) so that an empty
  corpus doesn't pin a stale miss forever. Disable with
  ``SEMANTIC_CACHE_CACHE_MISSES=0``.

- Concurrency: all SQLite operations are wrapped in
  ``try/except sqlite3.OperationalError`` and degrade gracefully to
  the non-cached path with a warning log. The cache wrappers never
  raise.

- The in-process LRU size is hard-coded at 64 entries (see
  ``_SEARCH_LRU_SIZE``). It is not configurable; it just absorbs
  per-process hot repeats.

Log levels
----------

- INFO: hit, store, clear, invalidate
- DEBUG: miss, expire, eviction
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .config import settings
from .logging_setup import logger
from .photo_matching import extract_photos
from .text_processing import normalize_text

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: In-process LRU size for the search cache. Hard-coded, not configurable.
#: A small dict that absorbs per-process hot repeats before falling
#: through to the on-disk SQLite cache.
_SEARCH_LRU_SIZE: int = 64

#: Fraction of the cap to evict at once when the table is full.
#: Amortizes the cost of eviction across many inserts.
_EVICT_FRACTION: int = 10


# ---------------------------------------------------------------------------
# Lazy singletons (mirrors _embedding_model / _client in qdrant_store.py)
# ---------------------------------------------------------------------------

_semantic: "SemanticCache | None" = None
_search: "SearchCache | None" = None


def _get_semantic() -> "SemanticCache | None":
    global _semantic
    if not settings.semantic_cache_enabled:
        return None
    if _semantic is None:
        _semantic = SemanticCache(
            path=settings.semantic_cache_path,
            ttl_seconds=settings.semantic_cache_ttl_seconds,
            miss_ttl_seconds=settings.semantic_cache_miss_ttl_seconds,
            max_entries=settings.semantic_cache_max_entries,
            similarity_threshold=settings.semantic_cache_similarity_threshold,
        )
    return _semantic


def _get_search() -> "SearchCache | None":
    global _search
    if not settings.search_cache_enabled:
        return None
    if _search is None:
        _search = SearchCache(
            path=settings.search_cache_path,
            ttl_seconds=settings.search_cache_ttl_seconds,
            max_entries=settings.search_cache_max_entries,
        )
    return _search


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """SQLite-backed semantic cache keyed by cosine similarity.

    Stores the question text, its embedding, the LLM answer, the
    contexts that produced it, and a flag indicating whether the
    answer was a "No relevant information found" miss (which uses the
    shorter miss TTL).

    Lookup is a pure-Python scan over all non-expired rows, computing
    cosine similarity between the incoming question's embedding and
    each stored embedding. Cheap because the row count is bounded by
    ``max_entries`` (default 1000).
    """

    def __init__(
        self,
        *,
        path: Path,
        ttl_seconds: int,
        miss_ttl_seconds: int,
        max_entries: int,
        similarity_threshold: float,
    ) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.miss_ttl_seconds = miss_ttl_seconds
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self._expires = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        contexts_json TEXT NOT NULL,
                        ts REAL NOT NULL,
                        last_accessed REAL NOT NULL,
                        is_miss INTEGER NOT NULL
                    )
                    """
                )
                conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("semantic_cache_init_failed path=%s error=%s", self.path, exc)

    def _row_ttl_seconds(self, is_miss: int) -> int:
        return self.miss_ttl_seconds if is_miss else self.ttl_seconds

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        try:
            cur = conn.execute("SELECT COUNT(*) FROM semantic_cache")
            count = cur.fetchone()[0]
            if count >= self.max_entries:
                n = max(1, self.max_entries // _EVICT_FRACTION)
                conn.execute(
                    "DELETE FROM semantic_cache ORDER BY last_accessed ASC LIMIT ?",
                    (n,),
                )
                self._evictions += n
                logger.debug(
                    "semantic_cache_evicted count=%s cap=%s deleted=%s",
                    count,
                    self.max_entries,
                    n,
                )
        except sqlite3.OperationalError as exc:
            logger.warning("semantic_cache_evict_failed error=%s", exc)

    def lookup(
        self, question: str, query_embedding: list[float]
    ) -> dict[str, Any] | None:
        """Return a cached ``{"answer": str, "contexts": [...]}`` dict on hit, else ``None``.

        Updates ``last_accessed`` and increments hit/miss counters.
        Expired rows are deleted lazily on each lookup.
        """
        now = time.time()
        try:
            with sqlite3.connect(self.path) as conn:
                # Lazy expiry sweep.
                cur = conn.execute(
                    "SELECT id, is_miss, ts FROM semantic_cache"
                )
                expired_ids: list[int] = []
                for row_id, is_miss, ts in cur.fetchall():
                    if ts + self._row_ttl_seconds(is_miss) < now:
                        expired_ids.append(row_id)
                if expired_ids:
                    placeholders = ",".join("?" for _ in expired_ids)
                    conn.execute(
                        f"DELETE FROM semantic_cache WHERE id IN ({placeholders})",
                        expired_ids,
                    )
                    self._expires += len(expired_ids)
                    logger.debug(
                        "semantic_cache_expired deleted=%s", len(expired_ids)
                    )

                # Find the best match above threshold.
                cur = conn.execute(
                    "SELECT id, question, embedding, answer, contexts_json, is_miss "
                    "FROM semantic_cache"
                )
                best_id: int | None = None
                best_sim: float = -1.0
                best_answer: str | None = None
                best_contexts_json: str | None = None
                best_is_miss: int = 0
                for row_id, q, emb_json, answer, contexts_json, is_miss in cur.fetchall():
                    stored_emb = json.loads(emb_json)
                    sim = _cosine_similarity(query_embedding, stored_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = row_id
                        best_answer = answer
                        best_contexts_json = contexts_json
                        best_is_miss = int(is_miss)

                if best_id is None or best_sim < self.similarity_threshold:
                    self._misses += 1
                    logger.debug(
                        "semantic_cache_miss best_similarity=%s threshold=%s",
                        best_sim if best_id is not None else "n/a",
                        self.similarity_threshold,
                    )
                    return None

                conn.execute(
                    "UPDATE semantic_cache SET last_accessed = ? WHERE id = ?",
                    (now, best_id),
                )
                self._hits += 1
                logger.info(
                    "semantic_cache_hit similarity=%s threshold=%s is_miss=%s",
                    best_sim,
                    self.similarity_threshold,
                    bool(best_is_miss),
                )
                return {
                    "answer": best_answer,
                    "contexts": json.loads(best_contexts_json) if best_contexts_json else [],
                    "photos": extract_photos(json.loads(best_contexts_json) if best_contexts_json else []),
                }
        except sqlite3.OperationalError as exc:
            logger.warning("semantic_cache_lookup_failed error=%s", exc)
            return None

    def store(
        self,
        question: str,
        query_embedding: list[float],
        result: dict[str, Any],
        *,
        is_miss: bool,
    ) -> None:
        """Store a question+result pair. ``is_miss=True`` uses the miss TTL."""
        now = time.time()
        try:
            with sqlite3.connect(self.path) as conn:
                self._evict_if_needed(conn)
                conn.execute(
                    """
                    INSERT INTO semantic_cache
                        (question, embedding, answer, contexts_json, ts, last_accessed, is_miss)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        question,
                        json.dumps(query_embedding),
                        result.get("answer", ""),
                        json.dumps(result.get("contexts", [])),
                        now,
                        now,
                        1 if is_miss else 0,
                    ),
                )
                conn.commit()
                self._stores += 1
                logger.info(
                    "semantic_cache_store question_chars=%s is_miss=%s entries=%s",
                    len(question),
                    is_miss,
                    self.count(),
                )
        except sqlite3.OperationalError as exc:
            logger.warning("semantic_cache_store_failed error=%s", exc)

    def count(self) -> int:
        try:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM semantic_cache")
                return int(cur.fetchone()[0])
        except sqlite3.OperationalError as exc:
            logger.warning("semantic_cache_count_failed error=%s", exc)
            return 0

    def clear(self) -> int:
        try:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM semantic_cache")
                count = int(cur.fetchone()[0])
                conn.execute("DELETE FROM semantic_cache")
                conn.commit()
                self._hits = 0
                self._misses = 0
                self._stores = 0
                self._evictions = 0
                self._expires = 0
                logger.info("semantic_cache_clear deleted=%s", count)
                return count
        except sqlite3.OperationalError as exc:
            logger.warning("semantic_cache_clear_failed error=%s", exc)
            return 0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "path": str(self.path),
            "entries": self.count(),
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "evictions": self._evictions,
            "expires": self._expires,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "miss_ttl_seconds": self.miss_ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
        }


# ---------------------------------------------------------------------------
# SearchCache
# ---------------------------------------------------------------------------


class SearchCache:
    """SQLite-backed exact-match search result cache with an in-process LRU.

    Keyed by a hash of the normalized question text plus the settings
    that affect the result. On lookup, checks the in-process LRU first,
    then the SQLite file, then misses through to the caller.

    Invalidated on every successful ingest (see
    :func:`search_cache_invalidate`).
    """

    def __init__(
        self,
        *,
        path: Path,
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self._expires = 0
        self._lru: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS search_cache (
                        key TEXT PRIMARY KEY,
                        question TEXT NOT NULL,
                        contexts_json TEXT NOT NULL,
                        ts REAL NOT NULL
                    )
                    """
                )
                conn.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("search_cache_init_failed path=%s error=%s", self.path, exc)

    @staticmethod
    def make_key(
        question: str, *, top_k: int, collection: str, fastembed_model: str
    ) -> str:
        normalized = normalize_text(question)
        payload = f"{collection}|{fastembed_model}|{top_k}|{normalized}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        try:
            cur = conn.execute("SELECT COUNT(*) FROM search_cache")
            count = cur.fetchone()[0]
            if count >= self.max_entries:
                n = max(1, self.max_entries // _EVICT_FRACTION)
                conn.execute(
                    "DELETE FROM search_cache ORDER BY ts ASC LIMIT ?", (n,)
                )
                self._evictions += n
                logger.debug(
                    "search_cache_evicted count=%s cap=%s deleted=%s",
                    count,
                    self.max_entries,
                    n,
                )
        except sqlite3.OperationalError as exc:
            logger.warning("search_cache_evict_failed error=%s", exc)

    def lookup(
        self,
        question: str,
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Return ``(contexts, source)`` where source is ``"inprocess"``, ``"disk"``, or ``"miss"``."""
        key = self.make_key(
            question,
            top_k=top_k,
            collection=settings.qdrant_collection,
            fastembed_model=settings.fastembed_model,
        )
        # In-process LRU
        if key in self._lru:
            self._lru.move_to_end(key)
            self._hits += 1
            logger.info("search_cache_hit source=inprocess question_chars=%s", len(question))
            return self._lru[key], "inprocess"

        now = time.time()
        try:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute(
                    "SELECT contexts_json, ts FROM search_cache WHERE key = ?",
                    (key,),
                )
                row = cur.fetchone()
                if row is None:
                    self._misses += 1
                    logger.debug("search_cache_miss question_chars=%s", len(question))
                    return None, "miss"
                contexts_json, ts = row
                if ts + self.ttl_seconds < now:
                    conn.execute("DELETE FROM search_cache WHERE key = ?", (key,))
                    self._expires += 1
                    logger.debug("search_cache_expired key=%s", key[:12])
                    return None, "miss"
                contexts = json.loads(contexts_json)
                # Write back to LRU.
                self._lru[key] = contexts
                self._lru.move_to_end(key)
                while len(self._lru) > _SEARCH_LRU_SIZE:
                    self._lru.popitem(last=False)
                self._hits += 1
                logger.info("search_cache_hit source=disk question_chars=%s", len(question))
                return contexts, "disk"
        except sqlite3.OperationalError as exc:
            logger.warning("search_cache_lookup_failed error=%s", exc)
            return None, "miss"

    def store(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> None:
        key = self.make_key(
            question,
            top_k=top_k,
            collection=settings.qdrant_collection,
            fastembed_model=settings.fastembed_model,
        )
        now = time.time()
        # Update in-process LRU first (always).
        self._lru[key] = contexts
        self._lru.move_to_end(key)
        while len(self._lru) > _SEARCH_LRU_SIZE:
            self._lru.popitem(last=False)
        try:
            with sqlite3.connect(self.path) as conn:
                self._evict_if_needed(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO search_cache
                        (key, question, contexts_json, ts)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        key,
                        normalize_text(question),
                        json.dumps(contexts),
                        now,
                    ),
                )
                conn.commit()
                self._stores += 1
                logger.info(
                    "search_cache_store question_chars=%s top_k=%s entries=%s",
                    len(question),
                    top_k,
                    self.count(),
                )
        except sqlite3.OperationalError as exc:
            logger.warning("search_cache_store_failed error=%s", exc)

    def count(self) -> int:
        try:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM search_cache")
                return int(cur.fetchone()[0])
        except sqlite3.OperationalError as exc:
            logger.warning("search_cache_count_failed error=%s", exc)
            return 0

    def clear(self) -> int:
        try:
            with sqlite3.connect(self.path) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM search_cache")
                count = int(cur.fetchone()[0])
                conn.execute("DELETE FROM search_cache")
                conn.commit()
                self._lru.clear()
                self._hits = 0
                self._misses = 0
                self._stores = 0
                self._evictions = 0
                self._expires = 0
                logger.info("search_cache_clear deleted=%s", count)
                return count
        except sqlite3.OperationalError as exc:
            logger.warning("search_cache_clear_failed error=%s", exc)
            return 0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "path": str(self.path),
            "entries": self.count(),
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "evictions": self._evictions,
            "expires": self._expires,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "inprocess_lru_size": len(self._lru),
        }


# ---------------------------------------------------------------------------
# Module-level helpers used by inference.py and qdrant_store.py
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return -1.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def semantic_cache_lookup(
    question: str, query_embedding: list[float]
) -> dict[str, Any] | None:
    """Look up a cached answer by question similarity. Returns ``None`` if disabled or miss."""
    cache = _get_semantic()
    if cache is None:
        return None
    return cache.lookup(question, query_embedding)


def semantic_cache_store(
    question: str,
    query_embedding: list[float],
    result: dict[str, Any],
    *,
    is_miss: bool,
) -> None:
    """Store a question+result pair. No-op if disabled or misses are disabled."""
    if is_miss and not settings.semantic_cache_cache_misses:
        return
    cache = _get_semantic()
    if cache is None:
        return
    cache.store(question, query_embedding, result, is_miss=is_miss)


def semantic_cache_stats() -> dict[str, Any]:
    """Return stats about the semantic cache. Returns ``enabled=False`` when off."""
    if not settings.semantic_cache_enabled:
        return {
            "enabled": False,
            "path": str(settings.semantic_cache_path),
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "expires": 0,
            "max_entries": settings.semantic_cache_max_entries,
            "ttl_seconds": settings.semantic_cache_ttl_seconds,
            "miss_ttl_seconds": settings.semantic_cache_miss_ttl_seconds,
            "similarity_threshold": settings.semantic_cache_similarity_threshold,
        }
    cache = _get_semantic()
    if cache is None:
        return {
            "enabled": False,
            "path": str(settings.semantic_cache_path),
            "entries": 0,
            "max_entries": settings.semantic_cache_max_entries,
        }
    return cache.stats()


def semantic_cache_clear() -> int:
    """Drop all rows from the semantic cache. No-op when disabled. Returns rows deleted."""
    if not settings.semantic_cache_enabled:
        return 0
    cache = _get_semantic()
    if cache is None:
        return 0
    return cache.clear()


def search_cache_lookup(
    question: str, *, top_k: int
) -> list[dict[str, Any]] | None:
    """Look up cached search contexts. Returns ``None`` if disabled or miss."""
    cache = _get_search()
    if cache is None:
        return None
    contexts, _source = cache.lookup(question, top_k=top_k)
    return contexts


def search_cache_store(
    question: str, contexts: list[dict[str, Any]], *, top_k: int
) -> None:
    """Store search contexts. No-op if disabled."""
    cache = _get_search()
    if cache is None:
        return
    cache.store(question, contexts, top_k=top_k)


def search_cache_invalidate() -> int:
    """Drop all rows from the search cache. Called from ``ingest_text`` / ``ingest_file``."""
    if not settings.search_cache_enabled:
        logger.info("search_cache_invalidate reason=ingest skipped=disabled")
        return 0
    cache = _get_search()
    if cache is None:
        return 0
    deleted = cache.clear()
    logger.info("search_cache_invalidate reason=ingest deleted=%s", deleted)
    return deleted


def search_cache_stats() -> dict[str, Any]:
    """Return stats about the search cache. Returns ``enabled=False`` when off."""
    if not settings.search_cache_enabled:
        return {
            "enabled": False,
            "path": str(settings.search_cache_path),
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "expires": 0,
            "max_entries": settings.search_cache_max_entries,
            "ttl_seconds": settings.search_cache_ttl_seconds,
            "inprocess_lru_size": 0,
        }
    cache = _get_search()
    if cache is None:
        return {
            "enabled": False,
            "path": str(settings.search_cache_path),
            "entries": 0,
            "max_entries": settings.search_cache_max_entries,
        }
    return cache.stats()


def search_cache_clear() -> int:
    """Drop all rows from the search cache and clear the in-process LRU. Returns rows deleted."""
    if not settings.search_cache_enabled:
        return 0
    cache = _get_search()
    if cache is None:
        return 0
    return cache.clear()
