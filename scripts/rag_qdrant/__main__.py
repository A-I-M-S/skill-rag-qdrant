from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .access_control import configure_access_store
from .bots import run_bot
from .inference import answer_question
from .logging_setup import logger
from .qdrant_store import collection_stats, ensure_collection, ingest_file, ingest_text, search
from .config import settings


def cmd_init(args: argparse.Namespace) -> None:
    ensure_collection()
    store = configure_access_store(settings.access_file, seed=settings.seed_allowed_telegram_ids)
    asyncio.run(store.load())
    print(json.dumps(collection_stats(), indent=2))
    print(
        json.dumps(
            {
                "access_file": str(store.path),
                "owner_id": store.owner_id,
                "allowed_user_ids": store.get_allowed(),
            },
            indent=2,
        )
    )


def cmd_ingest_file(args: argparse.Namespace) -> None:
    count = ingest_file(Path(args.path), source=args.source)
    print(json.dumps({"ingested_chunks": count}, indent=2))


def cmd_ingest_text(args: argparse.Namespace) -> None:
    count = ingest_text(args.text, source=args.source)
    print(json.dumps({"ingested_chunks": count}, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    results = search(args.question, top_k=args.top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_ask(args: argparse.Namespace) -> None:
    result = answer_question(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    print(json.dumps(collection_stats(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.rag_qdrant")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create the Qdrant collection and bootstrap the access file if needed")
    init.set_defaults(func=cmd_init)

    ingest_file_parser = sub.add_parser("ingest-file", help="Ingest a PDF/TXT/MD file")
    ingest_file_parser.add_argument("path")
    ingest_file_parser.add_argument("--source")
    ingest_file_parser.set_defaults(func=cmd_ingest_file)

    ingest_text_parser = sub.add_parser("ingest-text", help="Ingest raw text")
    ingest_text_parser.add_argument("text")
    ingest_text_parser.add_argument("--source", default="manual-text")
    ingest_text_parser.set_defaults(func=cmd_ingest_text)

    search_parser = sub.add_parser("search", help="Vector search Qdrant")
    search_parser.add_argument("question")
    search_parser.add_argument("--top-k", type=int)
    search_parser.set_defaults(func=cmd_search)

    ask_parser = sub.add_parser("ask", help="Search Qdrant and answer through the inference model")
    ask_parser.add_argument("question")
    ask_parser.set_defaults(func=cmd_ask)

    stats = sub.add_parser("stats", help="Show Qdrant collection stats")
    stats.set_defaults(func=cmd_stats)

    run_bot_parser = sub.add_parser("run-bot", help="Run the unified Telegram bot (ingest + query)")
    run_bot_parser.set_defaults(func=lambda args: run_bot())

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logger.info("cli_command command=%s", args.command)
    args.func(args)


if __name__ == "__main__":
    main()
