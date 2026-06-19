"""Telegram ``@username`` ↔ ``telegram_id`` resolver cache.

Telegram only lets a bot look up a user by username when the user has
already messaged the bot at least once. The skill works around this by
recording every inbound sender's ``telegram_id`` / ``username`` /
``first_name`` the moment a message arrives and then resolving the
``@username`` references that admins later type in chat
(``"let @alice see the Q3 note"``).

Storage
-------

A single JSON file at ``RAG_USER_CACHE_PATH`` (env, default
``logs/user_cache.json``) holds:

.. code-block:: json

    {
        "by_username": {"alice": 428765901, "bob": 920000001},
        "by_id": {
            "428765901": {
                "username": "alice",
                "first_name": "Alice",
                "last_seen": "2026-06-19T16:00:00+00:00"
            }
        }
    }

``by_username`` maps a lowercased Telegram ``@username`` (no leading
``@``) to the user's numeric ``telegram_id`` (``int`` after
normalization). ``by_id`` maps the stringified ``telegram_id`` to a
metadata record. Both views stay in sync: every ``record_seen`` call
upserts both, and the inverse mapping is kept consistent when a user
changes their username (the stale ``by_username`` entry is removed).

Concurrency
-----------

A module-level :class:`threading.Lock` serializes every
read-modify-write of the cache. All public functions acquire it
through :func:`_with_cache_lock`. Concurrent ``record_seen`` calls
cannot race because each holds the lock for the duration of the
read-modify-write. ``asyncio`` is not used here — the handler runs
LLM calls in worker threads, but the Telegram transport is
single-threaded and the cache is tiny, so a thread lock is enough.

Atomic writes
-------------

Every write goes through :func:`_atomic_write_json`, which writes to a
temp file in the same directory and ``os.replace``s it onto the target
path. A crash mid-write leaves the previous good copy on disk; the
temp file may be orphaned and is cleaned up on the next write or
:func:`clear_cache`.

Permissions
------------

The file is created with mode ``0o600`` (owner read/write only). On
POSIX this means the bot account and the admin account can read it but
other system users cannot. Existing files are not re-chmod'd on every
write — only the initial create sets the mode, so the operator can
adjust permissions afterwards if they need to share the file.

API
---

- :func:`record_seen(telegram_id, username=None, first_name=None)`
  Upsert one user.
- :func:`resolve_username(username) -> int | None`
  Case-insensitive, strip leading ``@``. Returns ``None`` when the
  user has not been seen yet (the LLM should surface a clear error).
- :func:`resolve_id(telegram_id) -> dict | None`
  Return the metadata record (``username``, ``first_name``,
  ``last_seen``) or ``None``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from .config import settings
from .logging_setup import logger


CACHE_FILE_MODE = 0o600

_lock = threading.Lock()


def _default_cache_path() -> Path:
    """Return the default on-disk cache path.

    Honors ``RAG_USER_CACHE_PATH`` (absolute or relative to the
    project root); falls back to ``logs/user_cache.json``.
    """
    raw = os.getenv("RAG_USER_CACHE_PATH", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = settings.skill_root / path
        return path
    return settings.skill_root / "logs" / "user_cache.json"


def cache_path() -> Path:
    """Return the current cache file path. Resolved lazily per call."""
    return _default_cache_path()


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _normalize_username(value: str | None) -> str | None:
    """Strip ``@`` and lowercase. ``None``/empty → ``None``."""
    if value is None:
        return None
    cleaned = value.strip().lstrip("@").strip()
    if not cleaned:
        return None
    return cleaned.lower()


def _coerce_telegram_id(value: int | str) -> str:
    """Stringify a telegram id for storage. ``"*"`` is preserved as the wildcard."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if not text:
        raise ValueError("telegram_id must be non-empty")
    return text


def _empty_cache() -> dict[str, Any]:
    return {"by_username": {}, "by_id": {}}


def _load_unlocked(path: Path) -> dict[str, Any]:
    """Read and parse the cache JSON. Missing/corrupt → empty cache."""
    if not path.exists():
        return _empty_cache()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("user_cache_load_failed path=%s error=%s", path, exc)
        return _empty_cache()
    if not isinstance(data, dict):
        return _empty_cache()
    by_username = data.get("by_username")
    by_id = data.get("by_id")
    if not isinstance(by_username, dict):
        by_username = {}
    if not isinstance(by_id, dict):
        by_id = {}
    return {"by_username": by_username, "by_id": by_id}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically with mode ``0o600``.

    Strategy: write to a temp file in the same directory (so the
    ``os.replace`` is atomic on POSIX), fsync, then rename. The temp
    file's mode is set to ``0o600`` so the rename preserves it on
    filesystems that carry mode across rename (POSIX does).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=".user_cache.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_str)
    try:
        os.chmod(tmp_path, CACHE_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
        try:
            os.chmod(path, CACHE_FILE_MODE)
        except OSError:
            pass
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _with_cache_lock(path: Path, mutate):
    """Run ``mutate(data)`` under the module lock and persist the result.

    ``mutate`` is a callable ``(data: dict) -> None`` that mutates
    ``data`` in place. The result is written atomically before the
    lock is released.
    """
    with _lock:
        data = _load_unlocked(path)
        mutate(data)
        _atomic_write_json(path, data)


def _read_locked(path: Path):
    """Run a read-only callable under the lock. Returns whatever the callable returns."""
    with _lock:
        data = _load_unlocked(path)
        return data


def record_seen(
    telegram_id: int | str,
    *,
    username: str | None = None,
    first_name: str | None = None,
) -> None:
    """Upsert one user into the cache.

    - ``telegram_id`` is required (numeric id from the Telegram update).
    - ``username`` is the raw ``@username`` (with or without leading
      ``@``); stored lowercased, no ``@``.
    - ``first_name`` is the display name from the Telegram user
      object; truncated to 256 chars defensively.

    If ``telegram_id`` was previously bound to a different
    ``username``, the stale ``by_username`` entry is deleted so
    ``resolve_username`` cannot return a stale id. ``last_seen`` is
    always bumped to the current UTC ISO timestamp.
    """
    tid_key = _coerce_telegram_id(telegram_id)
    user_norm = _normalize_username(username)
    fname = (first_name or "").strip()[:256] or None

    def _mutate(data: dict[str, Any]) -> None:
        by_username: dict[str, str] = data["by_username"]
        by_id: dict[str, dict[str, Any]] = data["by_id"]
        prev = by_id.get(tid_key)
        prev_username = None
        if isinstance(prev, dict):
            prev_username = _normalize_username(prev.get("username"))
        if prev_username and prev_username != user_norm:
            by_username.pop(prev_username, None)
        record = {
            "username": user_norm,
            "first_name": fname,
            "last_seen": _now_iso(),
        }
        if user_norm:
            by_username[user_norm] = tid_key
        by_id[tid_key] = record

    try:
        _with_cache_lock(cache_path(), _mutate)
        logger.info(
            "user_cache_record_seen telegram_id=%s username=%s",
            tid_key,
            user_norm,
        )
    except OSError as exc:
        logger.warning("user_cache_record_seen_failed error=%s", exc)


def resolve_username(username: str) -> int | None:
    """Return the ``telegram_id`` for ``username`` or ``None``.

    Strips a leading ``@``, lowercases, and looks up
    ``by_username``. Returns ``None`` (not raises) when the user has
    never DM'd the bot — the agent handler turns that into a clear
    error for the LLM.
    """
    key = _normalize_username(username)
    if not key:
        return None
    data = _read_locked(cache_path())
    raw = data["by_username"].get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def resolve_id(telegram_id: int | str) -> dict[str, Any] | None:
    """Return the metadata record for ``telegram_id`` or ``None``."""
    try:
        tid_key = _coerce_telegram_id(telegram_id)
    except ValueError:
        return None
    data = _read_locked(cache_path())
    rec = data["by_id"].get(tid_key)
    if not isinstance(rec, dict):
        return None
    return dict(rec)


def cache_stats() -> dict[str, Any]:
    """Return ``{"path": ..., "by_username": N, "by_id": N}``."""
    path = cache_path()
    data = _read_locked(path)
    return {
        "path": str(path),
        "by_username": len(data["by_username"]),
        "by_id": len(data["by_id"]),
    }


def clear_cache() -> int:
    """Drop the cache file. Returns the number of users removed (best-effort)."""
    path = cache_path()
    with _lock:
        prev = len(_load_unlocked(path)["by_id"])
        try:
            if path.exists():
                path.unlink()
                logger.info("user_cache_clear path=%s removed=%s", path, prev)
            return prev
        except OSError as exc:
            logger.warning("user_cache_clear_failed path=%s error=%s", path, exc)
            return 0


__all__ = [
    "cache_path",
    "cache_stats",
    "clear_cache",
    "record_seen",
    "resolve_id",
    "resolve_username",
]
