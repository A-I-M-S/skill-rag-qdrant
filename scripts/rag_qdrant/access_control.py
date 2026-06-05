from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .logging_setup import logger

SCHEMA_VERSION = 1


class AccessStore:
    def __init__(self, path: Path, seed: set[int] | None = None) -> None:
        self._path = path
        self._seed: set[int] = set(seed or ())
        self._lock = asyncio.Lock()
        self._owner_id: int | None = None
        self._allowed: set[int] = set()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def seed(self) -> set[int]:
        return set(self._seed)

    @property
    def owner_id(self) -> int | None:
        return self._owner_id

    def is_owner_set(self) -> bool:
        return self._owner_id is not None

    def is_owner(self, user_id: int) -> bool:
        return self._owner_id is not None and self._owner_id == user_id

    def is_allowed(self, user_id: int) -> bool:
        if self._owner_id is not None and self._owner_id == user_id:
            return True
        return user_id in self._allowed

    def get_allowed(self) -> list[int]:
        return sorted(self._allowed)

    async def load(self) -> None:
        async with self._lock:
            await self._load_locked()

    async def _load_locked(self) -> None:
        if not self._path.exists():
            await self._bootstrap_locked()
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("telegram_access_file_read_failed path=%s error=%s", self._path, exc)
            await self._reset_to_empty_locked()
            return
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            logger.warning(
                "telegram_access_file_corrupt path=%s error=%s treating_as_empty",
                self._path,
                exc,
            )
            await self._reset_to_empty_locked()
            return
        owner = data.get("owner_id")
        allowed = data.get("allowed_user_ids") or []
        self._owner_id = int(owner) if owner is not None else None
        self._allowed = {int(item) for item in allowed if isinstance(item, int)}
        if self._owner_id is not None:
            self._allowed.discard(self._owner_id)
        logger.info(
            "telegram_access_loaded path=%s owner_set=%s allowed_count=%s",
            self._path,
            self._owner_id is not None,
            len(self._allowed),
        )

    async def _reset_to_empty_locked(self) -> None:
        self._owner_id = None
        self._allowed = set()
        await self._save_locked()

    async def _bootstrap_locked(self) -> None:
        seed_list = sorted(self._seed)
        self._owner_id = None
        self._allowed = set(seed_list)
        if seed_list:
            logger.info("telegram_seed_consumed ids=%s", seed_list)
        await self._save_locked()
        logger.info(
            "telegram_access_bootstrapped path=%s owner_set=%s allowed_count=%s",
            self._path,
            False,
            len(self._allowed),
        )

    async def _save_locked(self) -> None:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "owner_id": self._owner_id,
            "allowed_user_ids": sorted(self._allowed),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=self._path.name + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    async def claim_owner(self, user_id: int) -> bool:
        async with self._lock:
            if self._owner_id is not None:
                return False
            self._owner_id = int(user_id)
            self._allowed.discard(self._owner_id)
            await self._save_locked()
            logger.info("telegram_owner_claimed user_id=%s", self._owner_id)
            return True

    async def allow(self, user_id: int) -> bool:
        async with self._lock:
            uid = int(user_id)
            if uid == self._owner_id:
                return False
            if uid in self._allowed:
                return False
            self._allowed.add(uid)
            await self._save_locked()
            logger.info("telegram_allow_added user_id=%s by_owner=%s", uid, self._owner_id)
            return True

    async def disallow(self, user_id: int) -> bool:
        async with self._lock:
            uid = int(user_id)
            if uid == self._owner_id:
                return False
            if uid not in self._allowed:
                return False
            self._allowed.discard(uid)
            await self._save_locked()
            logger.info("telegram_allow_removed user_id=%s by_owner=%s", uid, self._owner_id)
            return True


_singleton: AccessStore | None = None


def configure_access_store(path: Path, seed: set[int] | None = None) -> AccessStore:
    global _singleton
    _singleton = AccessStore(path, seed)
    return _singleton


def get_access_store() -> AccessStore:
    if _singleton is None:
        raise RuntimeError("AccessStore is not configured. Call configure_access_store() during setup.")
    return _singleton
