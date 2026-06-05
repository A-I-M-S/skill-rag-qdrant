"""Self-contained test runner. Works without pytest.

Usage: python3 tests/run_tests.py

The repository no longer ships pytest-based tests. This script runs the
behavioral assertions via the standard library + asyncio.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import types  # noqa: E402


def _ensure_stub(name: str) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    sys.modules[name] = mod


for _missing in ("dotenv",):
    _ensure_stub(_missing)
    setattr(sys.modules[_missing], "load_dotenv", lambda *a, **k: None)


from scripts.rag_qdrant import access_control as _access_module  # noqa: E402
from scripts.rag_qdrant import config as config_module  # noqa: E402
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

        # set_owner first time
        p = new_path("set_owner")
        store = AccessStore(p)
        await store.load()
        expect("set_owner_first", await store.set_owner(42) is True)
        expect("is_owner_after_set", store.is_owner(42) is True and store.is_allowed(42) is True)

        # set_owner replaces existing
        p = new_path("set_owner_replace")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(1)
        expect("set_owner_replaces_returns_true", await store.set_owner(2) is True)
        expect("set_owner_replaces_owner", store.is_owner(2) is True and store.is_owner(1) is False)

        # set_owner same value is a no-op
        p = new_path("set_owner_same")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(7)
        expect("set_owner_same_returns_false", await store.set_owner(7) is False)
        expect("set_owner_same_keeps_owner", store.is_owner(7) is True)

        # auto-claim fallback: no env, no JSON, claim_owner succeeds
        p = new_path("auto_claim")
        store = AccessStore(p)
        await store.load()
        expect("auto_claim_fallback_returns_true", await store.claim_owner(99) is True)
        expect("auto_claim_fallback_owner_set", store.is_owner(99) is True)

        # env set blocks auto-claim (env-authoritative owner is already 1)
        p = new_path("env_blocks_claim")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(1)
        expect("env_set_blocks_claim_returns_false", await store.claim_owner(99) is False)
        expect("env_set_blocks_claim_owner_unchanged", store.is_owner(1) is True and store.is_owner(99) is False)

        # env wins over JSON: write JSON with owner=2, then set_owner(1) (env path)
        p = new_path("env_wins")
        p.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "owner_id": 2, "allowed_user_ids": []}),
            encoding="utf-8",
        )
        store = AccessStore(p)
        await store.load()
        expect("env_wins_loads_json_first", store.owner_id == 2)
        expect("env_wins_set_owner_overrides", await store.set_owner(1) is True)
        expect("env_wins_final_owner_is_env", store.is_owner(1) is True and store.is_owner(2) is False)
        persisted = json.loads(p.read_text(encoding="utf-8"))
        expect("env_wins_persisted_owner_is_env", persisted["owner_id"] == 1)

        # allow then is_allowed
        p = new_path("allow")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(1)
        await store.allow(99)
        expect("allow_then_is_allowed", store.is_allowed(99) is True and store.is_allowed(100) is False)

        # allow idempotent
        p = new_path("allow_idem")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(1)
        await store.allow(50)
        expect("allow_idempotent", await store.allow(50) is False and store.get_allowed() == [50])

        # disallow removes
        p = new_path("disallow")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(1)
        await store.allow(50)
        await store.disallow(50)
        expect("disallow_removes", store.is_allowed(50) is False)

        # disallow owner refused
        p = new_path("disallow_owner")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(1)
        expect("disallow_owner_refused", await store.disallow(1) is False and store.is_owner(1) is True)

        # disallow not present
        p = new_path("disallow_absent")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(1)
        expect("disallow_not_present", await store.disallow(999) is False)

        # allow owner_id rejected
        p = new_path("allow_owner")
        store = AccessStore(p)
        await store.load()
        await store.set_owner(1)
        expect("allow_owner_id_rejected", await store.allow(1) is False and 1 not in store.get_allowed())

        # persistence round-trip (same path, two stores)
        p = new_path("persist")
        s1 = AccessStore(p)
        await s1.load()
        await s1.set_owner(7)
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

        # owner excluded from allowlist on load
        p = new_path("exclude")
        s1 = AccessStore(p)
        await s1.load()
        await s1.set_owner(7)
        await s1.allow(7)
        s2 = AccessStore(p)
        await s2.load()
        expect("owner_excluded_from_allowlist_on_load", 7 not in s2.get_allowed() and s2.is_owner(7))


def run_config_tests() -> None:
    # inference_provider and OpenRouter fields are gone from the dataclass
    expect(
        "inference_provider_field_removed",
        not hasattr(config_module.Settings, "inference_provider"),
    )
    expect(
        "openrouter_url_field_removed",
        not hasattr(config_module.Settings, "openrouter_url"),
    )
    expect(
        "openrouter_api_key_field_removed",
        not hasattr(config_module.Settings, "openrouter_api_key"),
    )
    expect(
        "openrouter_model_field_removed",
        not hasattr(config_module.Settings, "openrouter_model"),
    )
    expect(
        "openrouter_provider_field_removed",
        not hasattr(config_module.Settings, "openrouter_provider"),
    )

    # telegram_owner_id defaults to None when env is unset
    cleared_env = {k: v for k, v in os.environ.items() if k != "TELEGRAM_OWNER_ID"}
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.devnull, "w") as _devnull:
            old_stdout = sys.stdout
            sys.stdout = _devnull
            try:
                import importlib

                old_env = os.environ.copy()
                os.environ.clear()
                os.environ.update(cleared_env)
                try:
                    importlib.reload(config_module)
                    cleared_settings = config_module.Settings()
                    expect("telegram_owner_id_unset_is_none", cleared_settings.telegram_owner_id is None)
                finally:
                    os.environ.clear()
                    os.environ.update(old_env)
                    importlib.reload(config_module)
            finally:
                sys.stdout = old_stdout

    # telegram_owner_id=0 is treated as unset
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.devnull, "w") as _devnull:
            old_stdout = sys.stdout
            sys.stdout = _devnull
            try:
                import importlib

                old_env = os.environ.copy()
                os.environ["TELEGRAM_OWNER_ID"] = "0"
                try:
                    importlib.reload(config_module)
                    zero_settings = config_module.Settings()
                    expect("telegram_owner_id_zero_is_none", zero_settings.telegram_owner_id is None)
                finally:
                    os.environ.clear()
                    os.environ.update(old_env)
                    importlib.reload(config_module)
            finally:
                sys.stdout = old_stdout

    # inference_base_url is populated when env is set
    expect("inference_base_url_attribute_exists", hasattr(config_module.Settings, "inference_base_url"))
    expect("inference_api_key_attribute_exists", hasattr(config_module.Settings, "inference_api_key"))
    expect("inference_model_attribute_exists", hasattr(config_module.Settings, "inference_model"))
    expect(
        "inference_temperature_attribute_exists",
        hasattr(config_module.Settings, "inference_temperature"),
    )

    s = config_module.Settings(
        inference_base_url="https://example.com/v1/",
        inference_api_key="k",
        inference_model="m",
    )
    expect("inference_base_url_value_preserved", s.inference_base_url == "https://example.com/v1/")

    # require_inference only checks the single INFERENCE_* triple
    expect("require_inference_method_exists", hasattr(config_module.Settings, "require_inference"))
    s_full = config_module.Settings(
        inference_base_url="https://example.com/v1",
        inference_api_key="k",
        inference_model="m",
    )
    try:
        s_full.require_inference()
        expect("require_inference_satisfied_with_inference_triple", True)
    except RuntimeError as exc:
        expect("require_inference_satisfied_with_inference_triple", False, str(exc))

    s_empty = config_module.Settings(
        inference_base_url="",
        inference_api_key="",
        inference_model="",
    )
    try:
        s_empty.require_inference()
        expect("require_inference_raises_when_inference_empty", False, "expected RuntimeError")
    except RuntimeError:
        expect("require_inference_raises_when_inference_empty", True)

    # inference_base_url field default strips trailing slash when sourced from env.
    # The default is computed at class-definition time, so we reload the module.
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.devnull, "w") as _devnull:
            old_stdout = sys.stdout
            sys.stdout = _devnull
            try:
                import importlib

                old_env = os.environ.copy()
                os.environ["INFERENCE_BASE_URL"] = "https://example.com/v1/"
                os.environ["INFERENCE_API_KEY"] = "k"
                os.environ["INFERENCE_MODEL"] = "m"
                try:
                    importlib.reload(config_module)
                    stripped = config_module.Settings()
                    expect(
                        "inference_base_url_default_strips_trailing_slash",
                        stripped.inference_base_url == "https://example.com/v1",
                    )
                finally:
                    os.environ.clear()
                    os.environ.update(old_env)
                    importlib.reload(config_module)
            finally:
                sys.stdout = old_stdout


def run_inference_module_tests() -> None:
    import inspect

    src = inspect.getsource(config_module)
    expect("inference_module_no_openrouter_url_reference", "OPENROUTER_URL" not in src)
    expect("inference_module_no_openrouter_ak_reference", "OPENROUTER_AK" not in src)
    expect("inference_module_no_openrouter_model_reference", "OPENROUTER_MODEL" not in src)
    expect("inference_module_no_openrouter_provider_reference", "OPENROUTER_PROVIDER" not in src)
    expect("inference_module_no_inference_provider_env_reference", "INFERENCE_PROVIDER" not in src)
    expect("inference_module_no_zo_client_identity_token_reference", "ZO_CLIENT_IDENTITY_TOKEN" not in src)

    inf_path = ROOT / "scripts" / "rag_qdrant" / "inference.py"
    inf_src = inf_path.read_text(encoding="utf-8")
    expect("inference_src_no_zo_ask_function", "_answer_with_zo_ask" not in inf_src)
    expect("inference_src_no_zo_ask_branch", "zo_ask" not in inf_src)
    expect("inference_src_no_zo_computer_url", "api.zo.computer" not in inf_src)
    expect("inference_src_no_zo_client_identity_token", "ZO_CLIENT_IDENTITY_TOKEN" not in inf_src)
    expect("inference_src_no_openrouter_function", "_answer_with_openrouter" not in inf_src)
    expect("inference_src_no_openrouter_url", "OPENROUTER_URL" not in inf_src)
    expect("inference_src_has_answer_helper", "def _answer(" in inf_src)
    expect("inference_module_uses_single_settings", "settings.inference_base_url" in inf_src)
    expect("inference_module_uses_inference_model", "settings.inference_model" in inf_src)


def main() -> int:
    print("== prefix tests ==")
    run_prefix_tests()
    print("\n== access control tests ==")
    asyncio.run(run_access_tests())
    print("\n== config tests ==")
    run_config_tests()
    print("\n== inference module tests ==")
    run_inference_module_tests()
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        for label, detail in failed:
            print(f"  - {label}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
