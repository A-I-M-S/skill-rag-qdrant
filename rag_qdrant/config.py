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
