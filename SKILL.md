---
name: rag-qdrant
description: Local RAG skill that ingests text/PDF/MD into Qdrant and answers questions with a single OpenAI-compatible chat endpoint. Uses FastEmbed multilingual E5 embeddings and a configurable Qdrant collection.
user-invocable: true
metadata:
  openclaw:
    emoji: "📚"
    requires:
      bins: ["python"]
      anyBins: []
      env: ["QDRANT_URL", "QDRANT_API_KEY", "INFERENCE_BASE_URL", "INFERENCE_API_KEY", "INFERENCE_MODEL"]
    primaryEnv: "INFERENCE_API_KEY"
    install:
      - id: pip
        kind: pip
        package: "-r requirements.txt"
        label: "Install Python dependencies (qdrant-client, fastembed, openai, pypdf, python-dotenv)"
---

# rag-qdrant

A single Qdrant-backed RAG skill. Ingest text, PDF, or Markdown into a configurable Qdrant collection with FastEmbed multilingual E5 embeddings, then ask grounded questions answered by one OpenAI-compatible chat endpoint.

The skill has no UI of its own. Use the CLI or import the Python API from an openclaw agent.

## Setup

1. Copy `.env.example` to `.env` and fill in your Qdrant connection and inference endpoint.
2. Install dependencies:

   ```bash
   python -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Initialize the Qdrant collection (creates it and payload indexes if missing):

   ```bash
   python -m rag_qdrant init
   ```

See `references/setup.md` for full environment variable documentation, Qdrant Cloud and self-hosted options, FastEmbed model selection, and OpenAI-compatible endpoint configuration.

## Inspect

Show the current collection stats (vector count, indexed vectors, status):

```bash
python -m rag_qdrant stats
```

## Ingest

Ingest a PDF, TXT, or MD file. The `source` is the stable identifier used for point-id hashing and metadata filtering; if omitted, the filename is used.

```bash
python -m rag_qdrant ingest-file /path/to/notes.pdf
python -m rag_qdrant ingest-file /path/to/notes.pdf --source meeting-2026-06-05
```

Ingest a raw text string:

```bash
python -m rag_qdrant ingest-text "The cat sat on the mat." --source manual-note
```

Supported file types: `.pdf`, `.txt`, `.md`, `.text`. Anything else raises a clear error.

## Query

Run a raw vector search (returns the top-K contexts as JSON, no LLM call):

```bash
python -m rag_qdrant search "what does the document say about chunking?" --top-k 8
```

Ask a question end-to-end: search Qdrant, filter hits below `MIN_RELEVANCE_SCORE`, build a grounded prompt, and call the configured inference model:

```bash
python -m rag_qdrant ask "What does the document say about chunking?"
```

Output is JSON: `{"answer": "...", "contexts": [{"score": ..., "text": ..., "source": ..., "chunk_index": ..., "payload": {...}}, ...]}`. If no context hits pass the relevance threshold, the answer is exactly `No relevant information found` and `contexts` is empty.

## Programmatic use

```python
from rag_qdrant import RAG, ingest_text, ask, search, stats, ensure_collection

# Flat function style
ensure_collection()
ingest_text("The cat sat on the mat.", source="manual-note")
result = ask("Where did the cat sit?")
print(result["answer"])

# Or with the thin RAG class
from rag_qdrant import RAG
rag = RAG()
rag.ingest_file("/path/to/notes.pdf", source="meeting-2026-06-05")
print(rag.ask("summarize the meeting").answer if hasattr(rag.ask, "answer") else rag.ask("summarize the meeting")["answer"])
```

See `examples/agent_usage.md` for a complete openclaw agent pattern.

## References

- `references/setup.md` — environment variables, Qdrant Cloud vs. local, FastEmbed model selection, OpenAI-compatible endpoint config
- `examples/ingest_cli.md` — worked examples of `init`, `ingest-text`, `ingest-file`, `ask`
- `examples/agent_usage.md` — how an openclaw agent imports and calls the skill programmatically
