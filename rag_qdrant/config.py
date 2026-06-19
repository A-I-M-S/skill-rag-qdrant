from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SKILL_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SKILL_ROOT / ".env")

ENV_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    match = ENV_PLACEHOLDER_RE.match(value or "")
    if match:
        return os.getenv(match.group(1), "")
    return value or default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Admin role gate (issue #3)
# ---------------------------------------------------------------------------

def _load_admin_telegram_ids() -> tuple[int, ...]:
    """Load admin telegram ids from ``ADMIN_TELEGRAM_IDS`` at import time.

    Fail-closed: empty / unset / whitespace-only values raise
    :class:`RuntimeError` so a missing or misconfigured environment
    surfaces immediately, not silently. The returned tuple is frozen
    and subsequent ``os.environ`` mutations have no effect.
    """
    raw = os.getenv("ADMIN_TELEGRAM_IDS", "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise RuntimeError(
            "ADMIN_TELEGRAM_IDS is required and must contain at least one "
            "comma-separated Telegram user id. Set it in .env."
        )
    ids: list[int] = []
    for part in parts:
        try:
            ids.append(int(part))
        except ValueError as exc:
            raise RuntimeError(
                f"ADMIN_TELEGRAM_IDS entry {part!r} is not a valid integer: {exc}"
            ) from exc
    return tuple(ids)


admin_telegram_ids: tuple[int, ...] = _load_admin_telegram_ids()
ADMIN_TELEGRAM_IDS: tuple[int, ...] = admin_telegram_ids


def is_admin(telegram_id: int | str | None) -> bool:
    """Server-side check: is ``telegram_id`` in the admin set?

    Accepts ``int`` or stringified int. ``None`` is always non-admin.
    String ids are normalized to ``int`` to match the format stored
    in the admin set, so callers can pass either shape.
    """
    if telegram_id is None:
        return False
    try:
        normalized = int(telegram_id)
    except (TypeError, ValueError):
        return False
    return normalized in admin_telegram_ids


@dataclass(frozen=True)
class Settings:
    skill_root: Path = SKILL_ROOT
    qdrant_url: str = _env("QDRANT_URL")
    qdrant_api_key: str = _env("QDRANT_API_KEY")
    qdrant_collection: str = _env("QDRANT_COLLECTION", "system_rag")
    fastembed_model: str = _env("FASTEMBED_MODEL", "intfloat/multilingual-e5-small")
    embedding_dim: int = _int_env("EMBEDDING_DIM", 384)
    chunk_size: int = _int_env("CHUNK_SIZE", 900)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 150)
    top_k: int = _int_env("TOP_K", 6)
    min_relevance_score: float = _float_env("MIN_RELEVANCE_SCORE", 0.78)
    inference_base_url: str = _env("INFERENCE_BASE_URL").rstrip("/")
    inference_api_key: str = _env("INFERENCE_API_KEY")
    inference_model: str = _env("INFERENCE_MODEL", "")
    inference_temperature: float = _float_env("INFERENCE_TEMPERATURE", 0.2)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Path = SKILL_ROOT / os.getenv("LOG_FILE", "logs/rag-qdrant.log")
    photos_dir: Path = Path(_env("RAG_PHOTOS_DIR", "/root/rag-photos"))

    semantic_cache_enabled: bool = _bool_env("SEMANTIC_CACHE_ENABLED", False)
    semantic_cache_path: Path = SKILL_ROOT / os.getenv("SEMANTIC_CACHE_PATH", "logs/semantic_cache.sqlite")
    semantic_cache_ttl_seconds: int = _int_env("SEMANTIC_CACHE_TTL_SECONDS", 86400)
    semantic_cache_miss_ttl_seconds: int = _int_env("SEMANTIC_CACHE_MISS_TTL_SECONDS", 3600)
    semantic_cache_max_entries: int = _int_env("SEMANTIC_CACHE_MAX_ENTRIES", 1000)
    semantic_cache_similarity_threshold: float = _float_env("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.88)
    semantic_cache_cache_misses: bool = _bool_env("SEMANTIC_CACHE_CACHE_MISSES", True)

    search_cache_enabled: bool = _bool_env("SEARCH_CACHE_ENABLED", False)
    search_cache_path: Path = SKILL_ROOT / os.getenv("SEARCH_CACHE_PATH", "logs/search_cache.sqlite")
    search_cache_ttl_seconds: int = _int_env("SEARCH_CACHE_TTL_SECONDS", 86400)
    search_cache_max_entries: int = _int_env("SEARCH_CACHE_MAX_ENTRIES", 5000)

    def require_qdrant(self) -> None:
        missing = [name for name, value in {
            "QDRANT_URL": self.qdrant_url,
            "QDRANT_API_KEY": self.qdrant_api_key,
        }.items() if not value]
        if missing:
            raise RuntimeError(f"Missing required Qdrant setting(s): {', '.join(missing)}")

    def require_inference(self) -> None:
        missing = [name for name, value in {
            "INFERENCE_BASE_URL": self.inference_base_url,
            "INFERENCE_API_KEY": self.inference_api_key,
            "INFERENCE_MODEL": self.inference_model,
        }.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing inference setting(s): "
                + ", ".join(missing)
                + ". Configure an OpenAI-compatible chat completion endpoint in .env."
            )


settings = Settings()
