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
from pathlib import Path

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
    ("openai", ("OpenAI",)),
    ("pypdf", ("PdfReader",)),
    ("qdrant_client", ("QdrantClient",)),
    ("qdrant_client.http", ()),
    ("qdrant_client.http.models", ()),
):
    _ensure_stub(_missing)
    for _attr in _attrs:
        if not hasattr(sys.modules[_missing], _attr):
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
    settings,
    stats,
)
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
        "ask",
        "search",
        "stats",
        "ensure_collection",
        "settings",
        "__version__",
    ):
        expect(f"public_api_exports_{name}", hasattr(rag_qdrant, name))

    # RAG class shape
    expect("public_api_exports_RAG", inspect.isclass(RAG))
    for method in (
        "ensure_collection",
        "ingest_text",
        "ingest_file",
        "search",
        "ask",
        "stats",
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
        "init", "ingest-file", "ingest-text", "search", "ask", "stats"
    } or any(
        choice in {"init", "ingest-file", "ingest-text", "search", "ask", "stats"}
        for choice in (getattr(action, "choices", None) or [])
    )}
    # Easier check: parse known subcommands without error
    for sub in ("init", "stats"):
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

def run_agent_handler_tests() -> None:
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


if __name__ == "__main__":
    sys.exit(main())
