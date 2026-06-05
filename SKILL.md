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

## Agent message handler

A small pure-library adapter that lets an openclaw agent (or a Telegram bot, webhook, REPL, etc.) treat the skill as a chat-style command surface. The agent layer turns inbound traffic into an `AgentMessage` and sends the returned string back to the user. The handler does **not** import any chat-transport package, does **not** perform network I/O, and does **not** read `.env` / config — it is pure library code that delegates to the existing flat functions.

Public types: `AgentMessage`, `Attachment`, `handle_message`.

Supported commands (case-insensitive prefix match):

| User input | Action | Reply |
| --- | --- | --- |
| `Embed <text>` | `ingest_text(text, source="telegram-<sha1(text[:40])[:12]>")` | `Ingested N chunks from telegram-<sha1[:12]>` |
| `Embed` + attached `.pdf`/`.txt`/`.md`/`.text` file | save to a temp path (cleaned up after the call), `ingest_file(path, source=<filename>)` | `Ingested N chunks from <filename>` |
| `Query <question>` | `ask(question)` | ONLY `result["answer"]` — no score, no source, no chunk_index, no payload, no `contexts` list |

Default source naming for `Embed <text>`: `telegram-<sha1[:12]>` of the first 40 characters of the stripped text. When the text is empty, the hash input falls back to the current UTC timestamp (ISO 8601, seconds) so each ingest still gets a unique source.

Negative paths (these **raise** `ValueError`; the handler produces no graceful reply):

- `Embed` with no text after the prefix **and** no attachment.
- `Query` with no text after the prefix.
- Any message that does not start with `Embed` or `Query`.

Example:

```python
from rag_qdrant import AgentMessage, Attachment, handle_message

# Embed text
reply = handle_message(AgentMessage(text="Embed The cat sat on the mat."))
# -> 'Ingested 1 chunks from telegram-3b4f0e1a9c2d'

# Embed attached file
reply = handle_message(
    AgentMessage(
        text="Embed",
        attachment=Attachment("notes.pdf", open("notes.pdf", "rb").read()),
    )
)
# -> 'Ingested 14 chunks from notes.pdf'

# Query
reply = handle_message(AgentMessage(text="Query Where did the cat sit?"))
# -> 'The cat sat on the mat.'   (only the answer, no contexts)
```

See `examples/agent_usage.md` for the full integration pattern.

## References

- `references/setup.md` — environment variables, Qdrant Cloud vs. local, FastEmbed model selection, OpenAI-compatible endpoint config
- `examples/ingest_cli.md` — worked examples of `init`, `ingest-text`, `ingest-file`, `ask`
- `examples/agent_usage.md` — how an openclaw agent imports and calls the skill programmatically
- `rag_qdrant/agent_handler.py` — `AgentMessage`, `Attachment`, `handle_message` (the chat-style adapter described above)
