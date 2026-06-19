"""CLI entry point: python -m rag_qdrant"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import (
    search_cache_clear,
    search_cache_stats,
    semantic_cache_clear,
    semantic_cache_stats,
)
from .config import is_admin, settings
from .inference import answer_question
from .logging_setup import logger
from .qdrant_store import (
    collection_stats,
    ensure_collection,
    grant_access,
    ingest_file,
    ingest_text,
    revoke_access,
    search,
    show_access,
)


def _parse_acl_ids(raw: str | None) -> list[str]:
    """Parse a comma-separated id list. Empty string → empty list (explicit no-access)."""
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _resolve_acl(args: argparse.Namespace) -> list[str] | None:
    """Convert the CLI's --allowed-telegram-ids / --no-access flags into a payload value.

    Returns ``None`` for public (default), ``[]`` for explicit no-access,
    or a list of ids for restricted.
    """
    if getattr(args, "no_access", False):
        return []
    raw = getattr(args, "allowed_telegram_ids", None)
    if raw is None:
        return None
    return _parse_acl_ids(raw)


def cmd_init(_args: argparse.Namespace) -> None:
    ensure_collection()
    print(json.dumps(collection_stats(), indent=2))


def cmd_ingest_file(args: argparse.Namespace) -> None:
    count = ingest_file(
        Path(args.path),
        source=args.source,
        allowed_telegram_ids=_resolve_acl(args),
    )
    print(json.dumps({"ingested_chunks": count}, indent=2))


def cmd_ingest_text(args: argparse.Namespace) -> None:
    count = ingest_text(
        args.text,
        source=args.source,
        allowed_telegram_ids=_resolve_acl(args),
    )
    print(json.dumps({"ingested_chunks": count}, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    results = search(
        args.question,
        top_k=args.top_k,
        allowed_telegram_id=args.telegram_id,
        is_admin=is_admin(args.telegram_id) if args.telegram_id is not None else False,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_ask(args: argparse.Namespace) -> None:
    result = answer_question(
        args.question,
        current_telegram_id=args.telegram_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_grant(args: argparse.Namespace) -> None:
    result = grant_access(args.source, args.telegram_id)
    print(json.dumps(result, indent=2))


def cmd_revoke(args: argparse.Namespace) -> None:
    result = revoke_access(args.source, args.telegram_id)
    print(json.dumps(result, indent=2))


def cmd_show_access(args: argparse.Namespace) -> None:
    result = show_access(args.source)
    print(json.dumps(result, indent=2))


def cmd_stats(_args: argparse.Namespace) -> None:
    print(json.dumps(collection_stats(), indent=2))


def cmd_cache_stats(_args: argparse.Namespace) -> None:
    payload = {
        "semantic": semantic_cache_stats(),
        "search": search_cache_stats(),
    }
    print(json.dumps(payload, indent=2))


def cmd_cache_clear(args: argparse.Namespace) -> None:
    target = args.target
    semantic_cleared = 0
    search_cleared = 0
    if target in {"semantic", "all"}:
        semantic_cleared = semantic_cache_clear()
    if target in {"search", "all"}:
        search_cleared = search_cache_clear()
    print(json.dumps({"target": target, "semantic_cleared": semantic_cleared, "search_cleared": search_cleared}, indent=2))


def cmd_cache_info(_args: argparse.Namespace) -> None:
    payload = {
        "semantic": {
            "enabled": settings.semantic_cache_enabled,
            "path": str(settings.semantic_cache_path),
            "ttl_seconds": settings.semantic_cache_ttl_seconds,
            "miss_ttl_seconds": settings.semantic_cache_miss_ttl_seconds,
            "max_entries": settings.semantic_cache_max_entries,
            "similarity_threshold": settings.semantic_cache_similarity_threshold,
            "cache_misses": settings.semantic_cache_cache_misses,
        },
        "search": {
            "enabled": settings.search_cache_enabled,
            "path": str(settings.search_cache_path),
            "ttl_seconds": settings.search_cache_ttl_seconds,
            "max_entries": settings.search_cache_max_entries,
        },
    }
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag_qdrant", description="rag-qdrant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create the Qdrant collection and payload indexes if missing")
    init_p.set_defaults(func=cmd_init)

    ingest_file_p = sub.add_parser("ingest-file", help="Ingest a PDF/TXT/MD file")
    ingest_file_p.add_argument("path")
    ingest_file_p.add_argument("--source")
    ingest_file_p.add_argument(
        "--allowed-telegram-ids",
        default=None,
        help="Comma-separated Telegram ids that may read this content. Omit for public.",
    )
    ingest_file_p.add_argument(
        "--no-access",
        action="store_true",
        help="Set allowed_telegram_ids to an empty list (admin-only). Overrides --allowed-telegram-ids.",
    )
    ingest_file_p.set_defaults(func=cmd_ingest_file)

    ingest_text_p = sub.add_parser("ingest-text", help="Ingest raw text")
    ingest_text_p.add_argument("text")
    ingest_text_p.add_argument("--source", default="manual-text")
    ingest_text_p.add_argument(
        "--allowed-telegram-ids",
        default=None,
        help="Comma-separated Telegram ids that may read this content. Omit for public.",
    )
    ingest_text_p.add_argument(
        "--no-access",
        action="store_true",
        help="Set allowed_telegram_ids to an empty list (admin-only). Overrides --allowed-telegram-ids.",
    )
    ingest_text_p.set_defaults(func=cmd_ingest_text)

    search_p = sub.add_parser("search", help="Vector search Qdrant (raw contexts, no LLM answer)")
    search_p.add_argument("question")
    search_p.add_argument("--top-k", type=int)
    search_p.add_argument(
        "--telegram-id",
        default=None,
        help="Telegram id of the searcher. Admins see everything; others see only ACL-matching chunks.",
    )
    search_p.set_defaults(func=cmd_search)

    ask_p = sub.add_parser("ask", help="Search Qdrant and answer through the configured inference model")
    ask_p.add_argument("question")
    ask_p.add_argument(
        "--telegram-id",
        default=None,
        help="Telegram id of the asker. Admins see everything; others see only ACL-matching chunks.",
    )
    ask_p.set_defaults(func=cmd_ask)

    grant_p = sub.add_parser("grant-access", help="Add a Telegram id to the ACL of every chunk in a source")
    grant_p.add_argument("source")
    grant_p.add_argument("telegram_id")
    grant_p.set_defaults(func=cmd_grant)

    revoke_p = sub.add_parser("revoke-access", help="Remove a Telegram id from the ACL of every chunk in a source")
    revoke_p.add_argument("source")
    revoke_p.add_argument("telegram_id")
    revoke_p.set_defaults(func=cmd_revoke)

    show_p = sub.add_parser("show-access", help="Show the current ACL and chunk count for a source")
    show_p.add_argument("source")
    show_p.set_defaults(func=cmd_show_access)

    stats_p = sub.add_parser("stats", help="Show Qdrant collection stats")
    stats_p.set_defaults(func=cmd_stats)

    cache_stats_p = sub.add_parser("cache-stats", help="Show semantic and search cache stats")
    cache_stats_p.set_defaults(func=cmd_cache_stats)

    cache_clear_p = sub.add_parser("cache-clear", help="Clear one or both caches")
    cache_clear_p.add_argument("--target", choices=["semantic", "search", "all"], default="all")
    cache_clear_p.set_defaults(func=cmd_cache_clear)

    cache_info_p = sub.add_parser("cache-info", help="Show effective cache configuration (paths, TTLs, caps)")
    cache_info_p.set_defaults(func=cmd_cache_info)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logger.info("cli_command command=%s", args.command)
    args.func(args)


if __name__ == "__main__":
    main()
