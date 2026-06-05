"""Self-contained test runner. Works without pytest.

Usage: python3 tests/run_tests.py

The pytest-style tests in this directory are kept for when pytest is available.
This script runs the same assertions via the standard library + asyncio.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import types


def _ensure_stub(name: str) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    sys.modules[name] = mod


for _missing in ("dotenv",):
    _ensure_stub(_missing)
    setattr(sys.modules[_missing], "load_dotenv", lambda *a, **k: None)

from scripts.rag_qdrant.access_control import SCHEMA_VERSION, AccessStore  # noqa: E402
from scripts.rag_qdrant.prefix import parse_prefix  # noqa: E402

passed: list[str] = []
failed: list[tuple[str, str]] = []


def expect(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        passed.append(label)
        print(f"PASS {label}")
    else:
        failed.append((label, detail))
        print(f"FAIL {label}  {detail}")


def run_prefix_tests() -> None:
    cases = [
        ("embed_with_space", parse_prefix("embed hello world"), ("embed", "hello world")),
        ("embed_with_multiple_spaces", parse_prefix("embed   hello world"), ("embed", "hello world")),
        ("embed_with_tab", parse_prefix("embed\thello world"), ("embed", "hello world")),
        ("embed_with_newline", parse_prefix("embed\nhello world"), ("embed", "hello world")),
        ("embed_capitalized", parse_prefix("Embed hello world"), ("embed", "hello world")),
        ("embed_uppercase", parse_prefix("EMBED hello world"), ("embed", "hello world")),
        ("embed_mixed_case", parse_prefix("eMbEd hello world"), ("embed", "hello world")),
        ("embed_alone", parse_prefix("embed"), ("embed", "")),
        ("embed_alone_with_space", parse_prefix("embed "), ("embed", "")),
        ("embed_at_end_is_not_embed", parse_prefix("hello embed world"), ("query", "hello embed world")),
        ("embedding_prefix_is_not_embed", parse_prefix("embedding hello"), ("query", "embedding hello")),
        ("embedded_word_is_not_embed", parse_prefix("I am embedded in text"), ("query", "I am embedded in text")),
        ("empty_string", parse_prefix(""), ("query", "")),
        ("plain_question", parse_prefix("What is the capital of France?"), ("query", "What is the capital of France?")),
        ("embed_word_boundary_punctuation", parse_prefix("embed! hello world"), ("query", "embed! hello world")),
        ("embed_then_body", parse_prefix("embed\n\nThe quick brown fox."), ("embed", "The quick brown fox.")),
    ]
    for label, actual, expected in cases:
        expect(label, actual == expected, f"actual={actual!r} expected={expected!r}")


async def run_access_tests() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        counter = [0]

        def new_path(name: str) -> Path:
            counter[0] += 1
            return tmp_path / f"{counter[0]:03d}_{name}.json"

        # bootstrap empty
        p = new_path("empty")
        store = AccessStore(p, seed=set())
        await store.load()
        expect("bootstrap_empty_no_seed", store.owner_id is None and store.get_allowed() == [])

        # bootstrap with seed (fresh file)
        p = new_path("seed")
        store = AccessStore(p, seed={111, 222})
        await store.load()
        expect("bootstrap_with_seed", store.get_allowed() == [111, 222])

        # claim owner
        p = new_path("claim")
        store = AccessStore(p)
        await store.load()
        expect("claim_owner_first", await store.claim_owner(42) is True)
        expect("is_owner_after_claim", store.is_owner(42) is True and store.is_allowed(42) is True)

        # second claim fails
        p = new_path("second_claim")
        store = AccessStore(p)
        await store.load()
        await store.claim_owner(1)
        expect("claim_owner_second_fails", await store.claim_owner(2) is False)
        expect("owner_unchanged", store.is_owner(1) is True and store.is_owner(2) is False)

        # allow then is_allowed
        p = new_path("allow")
        store = AccessStore(p)
        await store.load()
        await store.claim_owner(1)
        await store.allow(99)
        expect("allow_then_is_allowed", store.is_allowed(99) is True and store.is_allowed(100) is False)

        # allow idempotent
        p = new_path("allow_idem")
        store = AccessStore(p)
        await store.load()
        await store.claim_owner(1)
        await store.allow(50)
        expect("allow_idempotent", await store.allow(50) is False and store.get_allowed() == [50])

        # disallow removes
        p = new_path("disallow")
        store = AccessStore(p)
        await store.load()
        await store.claim_owner(1)
        await store.allow(50)
        await store.disallow(50)
        expect("disallow_removes", store.is_allowed(50) is False)

        # disallow owner refused
        p = new_path("disallow_owner")
        store = AccessStore(p)
        await store.load()
        await store.claim_owner(1)
        expect("disallow_owner_refused", await store.disallow(1) is False and store.is_owner(1) is True)

        # disallow not present
        p = new_path("disallow_absent")
        store = AccessStore(p)
        await store.load()
        await store.claim_owner(1)
        expect("disallow_not_present", await store.disallow(999) is False)

        # allow owner_id rejected
        p = new_path("allow_owner")
        store = AccessStore(p)
        await store.load()
        await store.claim_owner(1)
        expect("allow_owner_id_rejected", await store.allow(1) is False and 1 not in store.get_allowed())

        # persistence round-trip (same path, two stores)
        p = new_path("persist")
        s1 = AccessStore(p)
        await s1.load()
        await s1.claim_owner(7)
        await s1.allow(100)
        await s1.allow(50)
        s2 = AccessStore(p)
        await s2.load()
        expect("persistence_round_trip", s2.is_owner(7) and s2.get_allowed() == [50, 100])

        # corrupt file
        p = new_path("corrupt")
        p.write_text("not valid json {{{", encoding="utf-8")
        store = AccessStore(p)
        await store.load()
        expect("corrupt_file_recovers", store.owner_id is None and store.get_allowed() == [])

        # seed only when file missing (use a different seed the second time, but expect the first)
        p = new_path("seed_only")
        s1 = AccessStore(p, seed={10, 20})
        await s1.load()
        s2 = AccessStore(p, seed={999})
        await s2.load()
        expect("seed_only_consumed_when_missing", s2.get_allowed() == [10, 20])

        # claim race
        p = new_path("race")
        store = AccessStore(p)
        await store.load()
        results = await asyncio.gather(*(store.claim_owner(i) for i in range(50)))
        expect("claim_owner_race_exactly_one", sum(1 for r in results if r) == 1)
        persisted = json.loads(p.read_text(encoding="utf-8"))
        expect("claim_owner_race_schema", persisted["schema_version"] == SCHEMA_VERSION)
        expect("claim_owner_race_owner_in_range", persisted["owner_id"] in range(50))

        # owner excluded from allowlist on load
        p = new_path("exclude")
        s1 = AccessStore(p)
        await s1.load()
        await s1.claim_owner(7)
        await s1.allow(7)
        s2 = AccessStore(p)
        await s2.load()
        expect("owner_excluded_from_allowlist_on_load", 7 not in s2.get_allowed() and s2.is_owner(7))


def main() -> int:
    print("== prefix tests ==")
    run_prefix_tests()
    print("\n== access control tests ==")
    asyncio.run(run_access_tests())
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        for label, detail in failed:
            print(f"  - {label}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
