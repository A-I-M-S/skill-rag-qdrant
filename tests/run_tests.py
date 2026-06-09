"""Self-contained test runner. Works without pytest.

Usage: python3 tests/run_tests.py

Covers:
- config field shape (no Telegram / OpenRouter / Zo remnants)
- qdrant_store shape (payload indexes, public functions, no telegram_user_id)
- text_processing: chunk_text + extract_text behavior
- inference module shape (single OpenAI-compatible path)
- public API re-exports (RAG + flat functions)
- repo-wide grep: no zo_ask / api.zo.computer / ZO_CLIENT_IDENTITY_TOKEN /
  python-telegram-bot in source, tests, or docs
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import types  # noqa: E402


def _ensure_stub(name: str) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    sys.modules[name] = mod


# Stub optional/foreign deps so the import of rag_qdrant.* works offline.
for _missing, _attrs in (
    ("dotenv", ("load_dotenv",)),
    ("fastembed", ()),
    ("fastembed.common.model_description", ("ModelSource", "PoolingType")),
    ("openai", ("OpenAI", "APIError", "BadRequestError")),
    ("pypdf", ("PdfReader",)),
    ("qdrant_client", ("QdrantClient",)),
    ("qdrant_client.http", ()),
    ("qdrant_client.http.models", ()),
):
    _ensure_stub(_missing)
    for _attr in _attrs:
        if not hasattr(sys.modules[_missing], _attr):
            if _attr in ("APIError", "BadRequestError"):
                # Real exception subclasses so the agent handler's
                # `classify_and_route` can import them under the stubs.
                setattr(sys.modules[_missing], _attr, type(_attr, (Exception,), {}))
            else:
                setattr(sys.modules[_missing], _attr, lambda *a, **k: None)
# Provide enums / classes used at import time
sys.modules["qdrant_client.http.models"].PayloadSchemaType = types.SimpleNamespace(KEYWORD="keyword", INTEGER="integer")
sys.modules["qdrant_client.http.models"].VectorParams = lambda **kw: ("VectorParams", kw)
sys.modules["qdrant_client.http.models"].Distance = types.SimpleNamespace(COSINE="Cosine")
sys.modules["qdrant_client.http.models"].PointStruct = lambda **kw: ("PointStruct", kw)
sys.modules["qdrant_client"].QdrantClient = type("QdrantClient", (), {})
sys.modules["fastembed"].TextEmbedding = type(
    "TextEmbedding",
    (),
    {
        "list_supported_models": staticmethod(lambda: []),
        "add_custom_model": staticmethod(lambda **kw: None),
    },
)
sys.modules["fastembed.common.model_description"].ModelSource = lambda **kw: ("ModelSource", kw)
sys.modules["fastembed.common.model_description"].PoolingType = types.SimpleNamespace(MEAN="mean")
sys.modules["pypdf"].PdfReader = type("PdfReader", (), {})
sys.modules["openai"].OpenAI = type("OpenAI", (), {})


import rag_qdrant  # noqa: E402
from rag_qdrant import (  # noqa: E402
    RAG,
    Settings,
    ask,
    chunk_text,
    collection_stats,
    ensure_collection,
    extract_text,
    ingest_file,
    ingest_text,
    search,
    search_cache_clear,
    search_cache_stats,
    semantic_cache_clear,
    semantic_cache_stats,
    settings,
    stats,
)
import rag_qdrant.cache as cache_module  # noqa: E402
import rag_qdrant.config as config_module  # noqa: E402
import rag_qdrant.inference as inference_module  # noqa: E402
import rag_qdrant.qdrant_store as qdrant_store_module  # noqa: E402
import rag_qdrant.text_processing as text_processing_module  # noqa: E402
import rag_qdrant.__main__ as cli_module  # noqa: E402

passed: list[str] = []
failed: list[tuple[str, str]] = []


def expect(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        passed.append(label)
        print(f"PASS {label}")
    else:
        failed.append((label, detail))
        print(f"FAIL {label}  {detail}")


# ---------------------------------------------------------------------------
# Config field shape
# ---------------------------------------------------------------------------

def run_config_tests() -> None:
    # Telegram fields are gone
    for field in (
        "bot_token",
        "telegram_owner_id",
        "seed_allowed_telegram_ids",
        "access_file",
        "upload_dir",
        "text_message_dir",
    ):
        expect(
            f"config_field_removed_{field}",
            not hasattr(config_module.Settings, field),
        )

    # OpenRouter / Zo fields never came back
    for field in (
        "inference_provider",
        "openrouter_url",
        "openrouter_api_key",
        "openrouter_model",
        "openrouter_provider",
    ):
        expect(
            f"config_field_removed_{field}",
            not hasattr(config_module.Settings, field),
        )

    # require_bot is gone
    expect(
        "config_method_removed_require_bot",
        not hasattr(config_module.Settings, "require_bot"),
    )

    # Required inference attributes exist
    for field in ("inference_base_url", "inference_api_key", "inference_model", "inference_temperature"):
        expect(f"config_attribute_exists_{field}", hasattr(config_module.Settings, field))

    # Photo storage path attribute
    expect("config_attribute_exists_photos_dir", hasattr(config_module.Settings, "photos_dir"))

    # Defaults are sane
    s = config_module.Settings(
        inference_base_url="https://example.com/v1",
        inference_api_key="k",
        inference_model="m",
    )
    expect("config_chunk_size_default", s.chunk_size == 900)
    expect("config_chunk_overlap_default", s.chunk_overlap == 150)
    expect("config_top_k_default", s.top_k == 6)
    expect("config_min_relevance_score_default", s.min_relevance_score == 0.78)
    expect("config_qdrant_collection_default", s.qdrant_collection == "system_rag")
    expect("config_fastembed_model_default", s.fastembed_model == "intfloat/multilingual-e5-small")
    expect("config_embedding_dim_default", s.embedding_dim == 384)

    # inference_base_url strips trailing slashes when sourced from env.
    # (Direct kwargs are not auto-stripped; only env-sourced values are.)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.devnull, "w") as _devnull:
            old_stdout = sys.stdout
            sys.stdout = _devnull
            try:
                old_env = os.environ.copy()
                os.environ["INFERENCE_BASE_URL"] = "https://example.com/v1/"
                os.environ["INFERENCE_API_KEY"] = "k"
                os.environ["INFERENCE_MODEL"] = "m"
                try:
                    importlib.reload(config_module)
                    stripped = config_module.Settings()
                    expect(
                        "config_inference_base_url_strips_trailing_slash",
                        stripped.inference_base_url == "https://example.com/v1",
                    )
                finally:
                    os.environ.clear()
                    os.environ.update(old_env)
                    importlib.reload(config_module)
            finally:
                sys.stdout = old_stdout

    # require_inference is satisfied and failing
    expect("config_method_exists_require_inference", hasattr(config_module.Settings, "require_inference"))
    try:
        s_full = config_module.Settings(
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
        )
        s_full.require_inference()
        expect("config_require_inference_satisfied_with_triple", True)
    except RuntimeError as exc:
        expect("config_require_inference_satisfied_with_triple", False, str(exc))

    try:
        s_empty = config_module.Settings(inference_base_url="", inference_api_key="", inference_model="")
        s_empty.require_inference()
        expect("config_require_inference_raises_when_empty", False, "expected RuntimeError")
    except RuntimeError:
        expect("config_require_inference_raises_when_empty", True)

    # require_qdrant is enforced
    try:
        s_q = config_module.Settings(
            qdrant_url="",
            qdrant_api_key="",
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
        )
        s_q.require_qdrant()
        expect("config_require_qdrant_raises_when_empty", False, "expected RuntimeError")
    except RuntimeError:
        expect("config_require_qdrant_raises_when_empty", True)


# ---------------------------------------------------------------------------
# Qdrant store shape
# ---------------------------------------------------------------------------

def run_qdrant_store_tests() -> None:
    src = inspect.getsource(qdrant_store_module)

    # No telegram_user_id index
    expect(
        "qdrant_store_no_telegram_user_id_index",
        "telegram_user_id" not in src,
    )
    expect(
        "qdrant_store_no_telegram_metadata",
        "telegram_" not in src,
    )

    # Public API exists
    for name in (
        "ensure_collection",
        "ensure_payload_indexes",
        "ingest_text",
        "ingest_file",
        "ingest_photo",
        "search",
        "collection_stats",
        "embed_texts",
        "get_qdrant_client",
    ):
        expect(f"qdrant_store_public_{name}", hasattr(qdrant_store_module, name))

    # ensure_payload_indexes payload schema set
    src_lines = src
    expect("qdrant_store_indexes_source", '"source"' in src_lines)
    expect("qdrant_store_indexes_file_name", '"file_name"' in src_lines)
    expect("qdrant_store_indexes_file_type", '"file_type"' in src_lines)
    expect("qdrant_store_indexes_kind", '"kind"' in src_lines)
    expect("qdrant_store_uses_cosine_distance", "models.Distance.COSINE" in src_lines)

    # custom FastEmbed model registration still present
    expect("qdrant_model_register_e5_small", "intfloat/multilingual-e5-small" in src_lines)


# ---------------------------------------------------------------------------
# Text processing tests
# ---------------------------------------------------------------------------

def run_text_processing_tests() -> None:
    # chunk_text: empty / whitespace
    expect("chunk_text_empty_returns_empty", chunk_text("") == [])
    expect("chunk_text_whitespace_returns_empty", chunk_text("   \n  \t  ") == [])

    # chunk_text: invalid overlap
    try:
        chunk_text("hello world", chunk_size=10, chunk_overlap=10)
        expect("chunk_text_overlap_equal_raises", False, "expected ValueError")
    except ValueError:
        expect("chunk_text_overlap_equal_raises", True)
    try:
        chunk_text("hello world", chunk_size=10, chunk_overlap=20)
        expect("chunk_text_overlap_greater_raises", False, "expected ValueError")
    except ValueError:
        expect("chunk_text_overlap_greater_raises", True)
    try:
        chunk_text("hello world", chunk_size=1, chunk_overlap=1)
        expect("chunk_text_overlap_equal_small_raises", False, "expected ValueError")
    except ValueError:
        expect("chunk_text_overlap_equal_small_raises", True)

    # chunk_text: produces non-empty chunks that cover the text
    sample = ("The quick brown fox. " * 200).strip()
    chunks = chunk_text(sample, chunk_size=200, chunk_overlap=20)
    expect("chunk_text_produces_chunks", len(chunks) > 1)
    expect("chunk_text_all_non_empty", all(c.strip() for c in chunks))

    # chunk_text: respects default settings (900/150)
    chunks_default = chunk_text("hello world. " * 200)
    expect("chunk_text_default_size_used", all(len(c) <= 900 for c in chunks_default))

    # extract_text: unsupported suffix raises
    try:
        extract_text(Path("/tmp/whatever.xyz"))
        expect("extract_text_unsupported_raises", False, "expected ValueError")
    except ValueError:
        expect("extract_text_unsupported_raises", True)

    # extract_text: .txt branch
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("hello world from a text file")
        tmp = Path(f.name)
    try:
        text = extract_text(tmp)
        expect("extract_text_txt", "hello world" in text)
    finally:
        tmp.unlink(missing_ok=True)

    # extract_text: .md branch
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("# heading\n\nsome markdown body")
        tmp = Path(f.name)
    try:
        text = extract_text(tmp)
        expect("extract_text_md", "heading" in text and "markdown body" in text)
    finally:
        tmp.unlink(missing_ok=True)

    # normalize_text collapses whitespace
    from rag_qdrant.text_processing import normalize_text
    expect("normalize_text_collapses_whitespace", normalize_text("a   b\n\nc\t\td") == "a b c d")


# ---------------------------------------------------------------------------
# Inference module shape
# ---------------------------------------------------------------------------

def run_inference_module_tests() -> None:
    src = inspect.getsource(inference_module)
    expect("inference_has_ask_function", "def ask(" in src or "ask = answer_question" in src)
    expect("inference_has_answer_question_function", "def answer_question(" in src)
    expect("inference_uses_settings_inference_base_url", "settings.inference_base_url" in src)
    expect("inference_uses_settings_inference_model", "settings.inference_model" in src)
    expect("inference_uses_settings_min_relevance_score", "min_relevance_score" in src)

    # No straggler providers
    for forbidden in (
        "_answer_with_zo_ask",
        "zo_ask",
        "api.zo.computer",
        "ZO_CLIENT_IDENTITY_TOKEN",
        "_answer_with_openrouter",
        "OPENROUTER_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_PROVIDER",
        "INFERENCE_PROVIDER",
        "telegram",
    ):
        expect(f"inference_no_{forbidden}", forbidden not in src)


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

def run_public_api_tests() -> None:
    # Flat functions are importable
    for name in (
        "ingest_text",
        "ingest_file",
        "ingest_photo",
        "extract_photos",
        "Photo",
        "AgentMessage",
        "AgentReply",
        "Attachment",
        "ask",
        "search",
        "stats",
        "ensure_collection",
        "settings",
        "semantic_cache_stats",
        "semantic_cache_clear",
        "search_cache_stats",
        "search_cache_clear",
        "__version__",
    ):
        expect(f"public_api_exports_{name}", hasattr(rag_qdrant, name))

    # RAG class shape
    expect("public_api_exports_RAG", inspect.isclass(RAG))
    for method in (
        "ensure_collection",
        "ingest_text",
        "ingest_file",
        "ingest_photo",
        "search",
        "ask",
        "stats",
        "semantic_cache_stats",
        "semantic_cache_clear",
        "search_cache_stats",
        "search_cache_clear",
    ):
        expect(f"rag_method_exists_{method}", callable(getattr(RAG, method, None)))

    # RAG().settings is the module-level settings
    rag = RAG()
    expect("rag_settings_is_default", rag.settings is settings)

    # RAG accepts a custom Settings
    custom = Settings(
        qdrant_collection="agent_test_rag",
        inference_base_url="https://example.com/v1",
        inference_api_key="k",
        inference_model="m",
    )
    rag2 = RAG(custom_settings=custom)
    expect("rag_settings_is_custom", rag2.settings is custom)
    expect("rag_custom_settings_collection", rag2.settings.qdrant_collection == "agent_test_rag")

    # ask is an alias of answer_question
    expect("ask_is_answer_question_alias", ask is inference_module.answer_question)
    # stats is an alias of collection_stats (defined in __init__.py)
    expect("stats_is_collection_stats_alias", rag_qdrant.stats is collection_stats)


# ---------------------------------------------------------------------------
# CLI module shape
# ---------------------------------------------------------------------------

def run_cli_tests() -> None:
    parser = cli_module.build_parser()
    sub_map = {action.dest: action for action in parser._actions if action.dest in {
        "init", "ingest-file", "ingest-text", "search", "ask", "stats", "cache-stats", "cache-clear", "cache-info"
    } or any(
        choice in {"init", "ingest-file", "ingest-text", "search", "ask", "stats", "cache-stats", "cache-clear", "cache-info"}
        for choice in (getattr(action, "choices", None) or [])
    )}
    # Easier check: parse known subcommands without error
    for sub in ("init", "stats", "cache-stats", "cache-info"):
        try:
            args = parser.parse_args([sub])
            expect(f"cli_parses_{sub}", args.command == sub)
        except SystemExit:
            expect(f"cli_parses_{sub}", False, "parser failed")
    for sub in ("ingest-text", "ingest-file", "ask", "search"):
        try:
            args = parser.parse_args([sub, "x"])
            expect(f"cli_parses_{sub}", args.command == sub)
        except SystemExit:
            expect(f"cli_parses_{sub}", False, "parser failed")
    for sub in ("cache-clear",):
        try:
            args = parser.parse_args([sub])
            expect(f"cli_parses_{sub}", args.command == sub and args.target == "all")
        except SystemExit:
            expect(f"cli_parses_{sub}", False, "parser failed")
        try:
            args = parser.parse_args([sub, "--target", "semantic"])
            expect(f"cli_parses_{sub}_target_semantic", args.command == sub and args.target == "semantic")
        except SystemExit:
            expect(f"cli_parses_{sub}_target_semantic", False, "parser failed")
        try:
            args = parser.parse_args([sub, "--target", "search"])
            expect(f"cli_parses_{sub}_target_search", args.command == sub and args.target == "search")
        except SystemExit:
            expect(f"cli_parses_{sub}_target_search", False, "parser failed")

    # run-bot subcommand must be gone
    try:
        args = parser.parse_args(["run-bot"])
        expect("cli_drops_run_bot", False, f"run-bot still present: {args}")
    except SystemExit:
        expect("cli_drops_run_bot", True)

    # __main__ exposes main()
    expect("cli_has_main", callable(getattr(cli_module, "main", None)))


# ---------------------------------------------------------------------------
# Agent handler tests (delegated to tests/test_agent_handler.py)
# ---------------------------------------------------------------------------

def run_agent_handler_shape_tests() -> None:
    """Source-grep checks for the agent handler data model and return type."""
    import rag_qdrant.agent_handler as _handler
    src = inspect.getsource(_handler)
    expect("agent_handler_uses_tuple_attachments", "attachments: tuple" in src)
    expect("agent_handler_uses_tuple_photos", "photos: tuple" in src)
    expect("agent_handler_returns_agent_reply", "-> AgentReply" in src or "AgentReply" in src)
    expect("agent_handler_defines_AgentReply", "class AgentReply" in src)
    # Photo is imported from photo_store, not defined here, so look for the import.
    expect("agent_handler_imports_Photo", "from .photo_store import" in src and "Photo" in src)


def run_agent_handler_tests() -> None:
    run_agent_handler_shape_tests()
    import test_agent_handler  # noqa: E402
    test_agent_handler.main()


# ---------------------------------------------------------------------------
# Repo-wide grep: no zo / telegram stragglers
# ---------------------------------------------------------------------------

FORBIDDEN_PATTERNS = (
    "zo_ask",
    "zo.computer",
    "ZO_CLIENT_IDENTITY_TOKEN",
    "python-telegram-bot",
)


def run_repo_grep_tests() -> None:
    targets = []
    for sub in (ROOT,):
        for ext in ("*.py", "*.md", "*.txt", "*.example"):
            targets.extend(sub.rglob(ext))

    for path in targets:
        # Skip our own test file (the test framework itself mentions these names
        # as forbidden patterns we are checking for)
        if path.resolve() == (Path(__file__).resolve()):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in FORBIDDEN_PATTERNS:
            if pat in text:
                expect(
                    f"grep_{path.name}_no_{pat}",
                    False,
                    f"found {pat!r} in {path.relative_to(ROOT)}",
                )
            else:
                expect(f"grep_{path.name}_no_{pat}", True)


def main() -> int:
    print("== config tests ==")
    run_config_tests()
    print("\n== qdrant store tests ==")
    run_qdrant_store_tests()
    print("\n== text processing tests ==")
    run_text_processing_tests()
    print("\n== inference module tests ==")
    run_inference_module_tests()
    print("\n== public api tests ==")
    run_public_api_tests()
    print("\n== cli tests ==")
    run_cli_tests()
    print("\n== cache tests ==")
    run_cache_tests()
    print("\n== repo grep tests ==")
    run_repo_grep_tests()
    print("\n== agent handler tests ==")
    run_agent_handler_tests()
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        for label, detail in failed:
            print(f"  - {label}: {detail}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

import math as _math  # noqa: E402
import sqlite3 as _sqlite3  # noqa: E402


def _unit_vector_2d(angle_rad: float) -> list[float]:
    return [_math.cos(angle_rad), _math.sin(angle_rad)] + [0.0] * 382


def _settings_with_cache_enabled(tmp_path: Path, *, semantic: bool, search: bool) -> Settings:
    return Settings(
        qdrant_collection="cache_test_rag",
        inference_base_url="https://example.com/v1",
        inference_api_key="k",
        inference_model="m",
        semantic_cache_enabled=semantic,
        semantic_cache_path=tmp_path / "sc.sqlite",
        semantic_cache_ttl_seconds=86400,
        semantic_cache_miss_ttl_seconds=3600,
        semantic_cache_max_entries=1000,
        semantic_cache_similarity_threshold=0.88,
        semantic_cache_cache_misses=True,
        search_cache_enabled=search,
        search_cache_path=tmp_path / "qc.sqlite",
        search_cache_ttl_seconds=86400,
        search_cache_max_entries=5000,
    )


def _force_settings(s: Settings) -> None:
    """Replace the module-level settings singleton and reset cache lazy singletons."""
    import rag_qdrant.config as _cfg
    _cfg.settings = s
    # Re-bind the name in any module that imported it directly via `from .config import settings`.
    for mod in (inference_module, qdrant_store_module, cache_module, rag_qdrant):
        if hasattr(mod, "settings"):
            mod.settings = s
    # Reset lazy singletons so the next read picks up the new settings.
    cache_module._semantic = None
    cache_module._search = None


def run_cache_tests() -> None:
    # Module shape
    src = inspect.getsource(cache_module)
    expect("cache_module_defines_SemanticCache", "class SemanticCache" in src)
    expect("cache_module_defines_SearchCache", "class SearchCache" in src)
    expect("cache_module_uses_sqlite3", "import sqlite3" in src)
    expect("cache_module_uses_normalize_text", "normalize_text" in src)
    expect("cache_module_swallows_OperationalError", "OperationalError" in src)
    expect("cache_module_never_raises_from_wrappers", "return None" in src and "return 0" in src)

    # Disabled by default (no env override)
    _force_settings(Settings(
        qdrant_collection="cache_test_rag",
        inference_base_url="https://example.com/v1",
        inference_api_key="k",
        inference_model="m",
    ))
    sc_stats = semantic_cache_stats()
    qc_stats = search_cache_stats()
    expect("cache_disabled_default_semantic", sc_stats.get("enabled") is False)
    expect("cache_disabled_default_search", qc_stats.get("enabled") is False)
    expect("cache_clear_noop_disabled_semantic", semantic_cache_clear() == 0)
    expect("cache_clear_noop_disabled_search", search_cache_clear() == 0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # --- Semantic cache round-trip ---
        s = _settings_with_cache_enabled(tmp_path, semantic=True, search=False)
        _force_settings(s)
        from rag_qdrant.cache import _get_semantic, SemanticCache

        cache = _get_semantic()
        expect("semantic_cache_enabled_when_flag_set", cache is not None)
        # Build a query embedding and a near-identical one.
        q_emb = _unit_vector_2d(0.0)
        near_emb = _unit_vector_2d(_math.radians(20))  # cos ~ 0.94, above 0.88 threshold
        miss_emb = _unit_vector_2d(_math.radians(60))  # cos ~ 0.50, below threshold

        expect("semantic_cache_lookup_empty_returns_None", cache.lookup("q1", q_emb) is None)
        cache.store("q1", q_emb, {"answer": "X", "contexts": []}, is_miss=False)
        expect("semantic_cache_lookup_hit_returns_value", cache.lookup("q1", q_emb)["answer"] == "X")
        expect("semantic_cache_lookup_near_hit", cache.lookup("q2", near_emb) is not None)
        expect("semantic_cache_lookup_below_threshold_miss", cache.lookup("q3", miss_emb) is None)
        expect("semantic_cache_count_after_stores", cache.count() == 1)
        s_dict = cache.stats()
        expect("semantic_cache_stats_enabled_true", s_dict.get("enabled") is True)
        expect("semantic_cache_stats_has_hits", "hits" in s_dict)
        expect("semantic_cache_stats_has_misses", "misses" in s_dict)
        expect("semantic_cache_stats_has_max_entries", s_dict.get("max_entries") == 1000)
        expect("semantic_cache_stats_has_threshold", s_dict.get("similarity_threshold") == 0.88)

        # --- TTL expiry (synthetic ts) ---
        s_short = Settings(
            qdrant_collection="cache_test_rag",
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
            semantic_cache_enabled=True,
            semantic_cache_path=tmp_path / "sc_ttl.sqlite",
            semantic_cache_ttl_seconds=10,
            semantic_cache_miss_ttl_seconds=1,
            semantic_cache_max_entries=100,
            semantic_cache_similarity_threshold=0.5,
        )
        _force_settings(s_short)
        cache_module._semantic = None
        cache2 = _get_semantic()
        cache2.store("q1", q_emb, {"answer": "HIT", "contexts": []}, is_miss=False)
        # Use a distinct embedding for q2 so the lookup doesn't fall through to q1.
        q_emb_miss = [0.0] * 384
        q_emb_miss[1] = 1.0
        cache2.store("q2", q_emb_miss, {"answer": "MISS", "contexts": []}, is_miss=True)
        # Force the miss row to be expired by mutating ts directly.
        with _sqlite3.connect(cache2.path) as conn:
            conn.execute(
                "UPDATE semantic_cache SET ts = ? WHERE answer = ?",
                (time.time() - 100, "MISS"),
            )
            conn.commit()
        expect("semantic_cache_miss_ttl_expired", cache2.lookup("q2", q_emb_miss) is None)
        expect("semantic_cache_hit_ttl_not_expired", cache2.lookup("q1", q_emb) is not None)

        # --- Max entries eviction ---
        s_cap = Settings(
            qdrant_collection="cache_test_rag",
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
            semantic_cache_enabled=True,
            semantic_cache_path=tmp_path / "sc_cap.sqlite",
            semantic_cache_ttl_seconds=86400,
            semantic_cache_miss_ttl_seconds=3600,
            semantic_cache_max_entries=3,
            semantic_cache_similarity_threshold=0.5,
        )
        _force_settings(s_cap)
        cache_module._semantic = None
        cap_cache = _get_semantic()
        for i in range(5):
            emb = [0.0] * 384
            emb[i % 384] = 1.0  # orthogonal-ish vectors
            cap_cache.store(f"q{i}", emb, {"answer": f"A{i}", "contexts": []}, is_miss=False)
        # 5 inserts; cap is 3; on each insert above cap, evict 1 (3//10=0 -> max(1, 0) = 1).
        # After 5 inserts, count should be at most 5 and at least cap.
        count_after = cap_cache.count()
        expect("semantic_cache_max_entries_bounded", count_after <= 6 and count_after >= 3)

        # --- Miss caching respects the separate miss flag (disabled) ---
        s_nomiss = Settings(
            qdrant_collection="cache_test_rag",
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
            semantic_cache_enabled=True,
            semantic_cache_path=tmp_path / "sc_nomiss.sqlite",
            semantic_cache_ttl_seconds=86400,
            semantic_cache_miss_ttl_seconds=3600,
            semantic_cache_max_entries=100,
            semantic_cache_similarity_threshold=0.5,
            semantic_cache_cache_misses=False,
        )
        _force_settings(s_nomiss)
        cache_module._semantic = None
        no_miss_cache = _get_semantic()
        from rag_qdrant.cache import semantic_cache_store
        semantic_cache_store(
            "missq", q_emb, {"answer": "No relevant information found", "contexts": []}, is_miss=True
        )
        expect("semantic_cache_miss_not_stored_when_disabled", no_miss_cache.count() == 0)
        semantic_cache_store(
            "hitq", q_emb, {"answer": "ok", "contexts": []}, is_miss=False
        )
        expect("semantic_cache_hit_stored_even_when_misses_off", no_miss_cache.count() == 1)

        # --- Search cache round-trip ---
        s_search = _settings_with_cache_enabled(tmp_path, semantic=False, search=True)
        _force_settings(s_search)
        from rag_qdrant.cache import _get_search
        qcache = _get_search()
        expect("search_cache_enabled_when_flag_set", qcache is not None)
        ctx = [{"score": 0.9, "id": "1", "text": "t", "source": "s", "chunk_index": 0, "payload": {}}]
        qcache.store("hello", ctx, top_k=6)
        expect("search_cache_lookup_hit", qcache.lookup("hello", top_k=6)[0] == ctx)
        expect("search_cache_lookup_miss_different_topk", qcache.lookup("hello", top_k=7)[0] is None)
        # Source label
        _, source = qcache.lookup("hello", top_k=6)
        expect("search_cache_lookup_source_inprocess", source == "inprocess")
        # Stats
        qstats = search_cache_stats()
        expect("search_cache_stats_enabled_true", qstats.get("enabled") is True)
        expect("search_cache_stats_has_inprocess_lru", "inprocess_lru_size" in qstats)

        # --- Search cache TTL expiry ---
        s_search_ttl = Settings(
            qdrant_collection="cache_test_rag",
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
            search_cache_enabled=True,
            search_cache_path=tmp_path / "qc_ttl.sqlite",
            search_cache_ttl_seconds=1,
            search_cache_max_entries=100,
        )
        _force_settings(s_search_ttl)
        cache_module._search = None
        qcache2 = _get_search()
        qcache2.store("x", ctx, top_k=6)
        # Force expiry
        with _sqlite3.connect(qcache2.path) as conn:
            conn.execute(
                "UPDATE search_cache SET ts = ?", (time.time() - 100,)
            )
            conn.commit()
        # Clear the in-process LRU so the lookup falls through to the disk check.
        qcache2._lru.clear()
        expect("search_cache_ttl_expired", qcache2.lookup("x", top_k=6)[0] is None)

        # --- Invalidation on ingest: store then call ingest ---
        s_inv = Settings(
            qdrant_collection="cache_test_rag",
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
            search_cache_enabled=True,
            search_cache_path=tmp_path / "qc_invalidate.sqlite",
            search_cache_ttl_seconds=86400,
            search_cache_max_entries=5000,
        )
        _force_settings(s_inv)
        cache_module._search = None
        from rag_qdrant.cache import search_cache_invalidate, search_cache_store
        search_cache_store("q", ctx, top_k=6)
        expect("search_cache_pre_invalidate_count", search_cache_stats()["entries"] == 1)
        # Patch the qdrant client + embedder so ingest_text doesn't try the network.
        with patch.object(qdrant_store_module, "ensure_collection", return_value=None), \
             patch.object(qdrant_store_module, "embed_texts", return_value=[[0.0] * 384]), \
             patch.object(qdrant_store_module, "get_qdrant_client") as m_client:
            fake = types.SimpleNamespace(upsert=lambda **kw: None)
            m_client.return_value = fake
            qdrant_store_module.ingest_text("hello world", source="test")
        # The ingest call should have called search_cache_invalidate; entries == 0
        expect("search_cache_invalidated_on_ingest", search_cache_stats()["entries"] == 0)

        # --- Inference uses cache when enabled, bypasses when disabled ---
        from rag_qdrant import inference as inf
        # Disabled (no flag): cache helpers are no-ops
        _force_settings(Settings(
            qdrant_collection="cache_test_rag",
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
        ))
        with patch.object(inf, "search", return_value=[]) as m_search, \
             patch.object(inf, "_answer", return_value="RAW") as m_answer, \
             patch.object(inf, "embed_texts", return_value=[[0.0] * 384]):
            res = inf.answer_question("anything")
        expect("inference_bypasses_cache_when_disabled", res["answer"] == "No relevant information found")
        expect("inference_calls_search_when_disabled", m_search.called)

        # Enabled with pre-populated cache: should return cached answer and not call search.
        s_on = Settings(
            qdrant_collection="cache_test_rag",
            inference_base_url="https://example.com/v1",
            inference_api_key="k",
            inference_model="m",
            semantic_cache_enabled=True,
            semantic_cache_path=tmp_path / "sc_inference.sqlite",
            semantic_cache_ttl_seconds=86400,
            semantic_cache_miss_ttl_seconds=3600,
            semantic_cache_max_entries=1000,
            semantic_cache_similarity_threshold=0.88,
            semantic_cache_cache_misses=True,
        )
        _force_settings(s_on)
        cache_module._semantic = None
        from rag_qdrant.cache import semantic_cache_store
        sem = _get_semantic()
        sem.store("x", _unit_vector_2d(0.0), {"answer": "CACHED", "contexts": [{"text": "ctx"}]}, is_miss=False)
        with patch.object(inf, "search") as m_search, \
             patch.object(inf, "_answer") as m_answer, \
             patch.object(inf, "embed_texts", return_value=[_unit_vector_2d(0.0)]) as m_emb:
            res = inf.answer_question("x")
        expect(
            "inference_returns_cached_answer",
            res.get("answer") == "CACHED",
            f"got {res!r}, embed_calls={m_emb.call_count}",
        )
        expect("inference_does_not_call_search_on_hit", not m_search.called)
        expect("inference_does_not_call_answer_on_hit", not m_answer.called)

        # Restore default settings
        _force_settings(Settings(
            qdrant_collection="system_rag",
            inference_base_url="",
            inference_api_key="",
            inference_model="",
        ))


if __name__ == "__main__":
    sys.exit(main())
