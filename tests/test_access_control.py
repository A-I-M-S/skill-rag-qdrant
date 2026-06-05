import asyncio
import json
from pathlib import Path

import pytest

from scripts.rag_qdrant.access_control import SCHEMA_VERSION, AccessStore


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "telegram_access.json"


@pytest.mark.asyncio
async def test_bootstrap_empty_when_no_seed_and_no_file(store_path: Path):
    store = AccessStore(store_path, seed=set())
    await store.load()
    assert store.owner_id is None
    assert store.get_allowed() == []
    assert store_path.exists()


@pytest.mark.asyncio
async def test_bootstrap_with_seed(store_path: Path):
    store = AccessStore(store_path, seed={111, 222})
    await store.load()
    assert store.owner_id is None
    assert store.get_allowed() == [111, 222]


@pytest.mark.asyncio
async def test_claim_owner_first_succeeds(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    assert await store.claim_owner(42) is True
    assert store.is_owner(42) is True
    assert store.is_owner_set() is True
    assert store.is_allowed(42) is True


@pytest.mark.asyncio
async def test_claim_owner_second_fails(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    assert await store.claim_owner(1) is True
    assert await store.claim_owner(2) is False
    assert store.is_owner(1) is True
    assert store.is_owner(2) is False


@pytest.mark.asyncio
async def test_allow_then_is_allowed(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    await store.claim_owner(1)
    assert await store.allow(99) is True
    assert store.is_allowed(99) is True
    assert store.is_allowed(100) is False


@pytest.mark.asyncio
async def test_allow_idempotent(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    await store.claim_owner(1)
    assert await store.allow(50) is True
    assert await store.allow(50) is False
    assert store.get_allowed() == [50]


@pytest.mark.asyncio
async def test_disallow_removes(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    await store.claim_owner(1)
    await store.allow(50)
    assert await store.disallow(50) is True
    assert store.is_allowed(50) is False


@pytest.mark.asyncio
async def test_disallow_owner_refused(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    await store.claim_owner(1)
    assert await store.disallow(1) is False
    assert store.is_owner(1) is True


@pytest.mark.asyncio
async def test_disallow_not_present(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    await store.claim_owner(1)
    assert await store.disallow(999) is False


@pytest.mark.asyncio
async def test_allow_owner_id_is_rejected(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    await store.claim_owner(1)
    assert await store.allow(1) is False
    assert 1 not in store.get_allowed()


@pytest.mark.asyncio
async def test_persistence_round_trip(store_path: Path):
    s1 = AccessStore(store_path)
    await s1.load()
    await s1.claim_owner(7)
    await s1.allow(100)
    await s1.allow(50)

    s2 = AccessStore(store_path)
    await s2.load()
    assert s2.is_owner(7)
    assert s2.get_allowed() == [50, 100]


@pytest.mark.asyncio
async def test_corrupt_file_recovers(store_path: Path):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text("not valid json {{{", encoding="utf-8")
    store = AccessStore(store_path)
    await store.load()
    assert store.owner_id is None
    assert store.get_allowed() == []
    assert store_path.exists()


@pytest.mark.asyncio
async def test_seed_only_consumed_when_file_missing(store_path: Path):
    s1 = AccessStore(store_path, seed={10, 20})
    await s1.load()
    assert s1.get_allowed() == [10, 20]

    s2 = AccessStore(store_path, seed={999})
    await s2.load()
    assert s2.get_allowed() == [10, 20]


@pytest.mark.asyncio
async def test_claim_owner_race(store_path: Path):
    store = AccessStore(store_path)
    await store.load()
    results = await asyncio.gather(*(store.claim_owner(i) for i in range(50)))
    true_count = sum(1 for r in results if r)
    assert true_count == 1
    persisted = json.loads(store_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == SCHEMA_VERSION
    assert persisted["owner_id"] in range(50)
    assert persisted["allowed_user_ids"] == []


@pytest.mark.asyncio
async def test_schema_version_in_file(store_path: Path):
    store = AccessStore(store_path, seed={5})
    await store.load()
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["owner_id"] is None
    assert raw["allowed_user_ids"] == [5]


@pytest.mark.asyncio
async def test_owner_excluded_from_allowlist_on_load(store_path: Path):
    s1 = AccessStore(store_path)
    await s1.load()
    await s1.claim_owner(7)
    await s1.allow(7)

    s2 = AccessStore(store_path)
    await s2.load()
    assert 7 not in s2.get_allowed()
    assert s2.is_owner(7)
