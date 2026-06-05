"""CLI entry point: python -m rag_qdrant"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import answer_question
from .logging_setup import logger
from .qdrant_store import collection_stats, ensure_collection, ingest_file, ingest_text, search


def cmd_init(_args: argparse.Namespace) -> None:
    ensure_collection()
    print(json.dumps(collection_stats(), indent=2))


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


def cmd_stats(_args: argparse.Namespace) -> None:
    print(json.dumps(collection_stats(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag_qdrant", description="rag-qdrant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create the Qdrant collection and payload indexes if missing")
    init_p.set_defaults(func=cmd_init)

    ingest_file_p = sub.add_parser("ingest-file", help="Ingest a PDF/TXT/MD file")
    ingest_file_p.add_argument("path")
    ingest_file_p.add_argument("--source")
    ingest_file_p.set_defaults(func=cmd_ingest_file)

    ingest_text_p = sub.add_parser("ingest-text", help="Ingest raw text")
    ingest_text_p.add_argument("text")
    ingest_text_p.add_argument("--source", default="manual-text")
    ingest_text_p.set_defaults(func=cmd_ingest_text)

    search_p = sub.add_parser("search", help="Vector search Qdrant (raw contexts, no LLM answer)")
    search_p.add_argument("question")
    search_p.add_argument("--top-k", type=int)
    search_p.set_defaults(func=cmd_search)

    ask_p = sub.add_parser("ask", help="Search Qdrant and answer through the configured inference model")
    ask_p.add_argument("question")
    ask_p.set_defaults(func=cmd_ask)

    stats_p = sub.add_parser("stats", help="Show Qdrant collection stats")
    stats_p.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logger.info("cli_command command=%s", args.command)
    args.func(args)


if __name__ == "__main__":
    main()
