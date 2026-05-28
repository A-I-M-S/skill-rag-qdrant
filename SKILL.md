---
name: skill-rag-qdrant
description: System RAG skill using Telegram ingestion and query bots, Qdrant Cloud storage, FastEmbed multilingual E5 embeddings, and an inference model endpoint for grounded answers.
compatibility: Created for Zo Computer
metadata:
  author: aloy.zo.computer
---
# skill-rag-qdrant

Use this skill to run a two-bot Telegram RAG system:

- **Bot A / ingestion** receives PDFs or text, extracts text, chunks it, embeds chunks with Qdrant FastEmbed `intfloat/multilingual-e5-small`, and stores them in Qdrant.
- **Bot B / query** receives questions, embeds the question, searches Qdrant, builds a grounded prompt, sends it to the configured inference model, and returns the answer.

## Setup

1. Copy `.env.example` to `.env`.
2. Fill in Telegram bot tokens, Qdrant URL/API key, and inference provider settings.
3. Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m scripts.rag_qdrant init
python -m scripts.rag_qdrant run-ingest-bot
python -m scripts.rag_qdrant run-query-bot
```

Or run both bots in one process:

```bash
python -m scripts.rag_qdrant run-all
```

## CLI

```bash
python -m scripts.rag_qdrant --help
python -m scripts.rag_qdrant ingest-file /path/to/file.pdf
python -m scripts.rag_qdrant ingest-text "your text here" --source manual-note
python -m scripts.rag_qdrant ask "What does the document say about ...?"
python -m scripts.rag_qdrant stats
```

## Logs and storage

- Logs: `logs/rag-qdrant.log`
- Telegram uploads: `storage/uploads/`
- Telegram text message snapshots: `storage/text_messages/`

Sensitive values belong only in `.env`; `.env`, logs, local uploads, and Python caches are ignored by git.