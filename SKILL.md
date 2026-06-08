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

## Caching (opt-in)

Two opt-in caches, both backed by SQLite in `logs/` and disabled by default. Enable with `SEMANTIC_CACHE_ENABLED=1` and/or `SEARCH_CACHE_ENABLED=1` in `.env`.

- **Semantic cache** — caches LLM answers keyed by question similarity (cosine, default threshold `0.88`). On a hit, both the Qdrant search and the LLM call are skipped. "No relevant information found" answers use a separate, shorter TTL (`SEMANTIC_CACHE_MISS_TTL_SECONDS`, default 1h).
- **Search cache** — caches raw Qdrant contexts keyed by exact question hash. On a hit, the Qdrant round-trip is skipped. Invalidated on every successful `ingest-text` / `ingest-file`.

Inspect and manage via the CLI:

```bash
python -m rag_qdrant cache-info          # show effective config
python -m rag_qdrant cache-stats         # entries / hits / misses / evictions
python -m rag_qdrant cache-clear         # clear both (default)
python -m rag_qdrant cache-clear --target semantic
python -m rag_qdrant cache-clear --target search
```

Or programmatically:

```python
from rag_qdrant import (
    semantic_cache_stats, semantic_cache_clear,
    search_cache_stats, search_cache_clear,
)
```

The semantic cache is **not** invalidated on ingest by design (clearing on every ingest would defeat the cache in any non-static corpus). All cache wrappers catch `sqlite3.OperationalError` and fall through to the non-cached path; the cache never raises.

## Agent message handler

A small pure-library adapter that lets an openclaw agent (or a Telegram bot, webhook, REPL, etc.) treat the skill as a chat-style surface. The agent layer turns inbound traffic into an `AgentMessage` and sends the returned string back to the user. The handler does **not** import any chat-transport package and does **not** read `.env` / config — it is pure library code that delegates to the existing flat functions.

Public types: `AgentMessage`, `Attachment`, `handle_message`. There are no command prefixes, no `/raw` escape hatches, and no override switches. The configured inference model is the sole decision-maker.

### What the LLM sees

Every inbound `AgentMessage` (text or attachment) is sent to the inference model with this system prompt (excerpt — full text in `rag_qdrant/prompts.py`):

> You are the routing layer for a small RAG skill. You have two tools and one chat path. You are the only decision-maker.
>
> 1. `store_text(text, source="")` — save `text` into the knowledge base.
> 2. `ask_corpus(question)` — search the knowledge base and answer `question` grounded in what is found.
>
> Chat path (no tool call): greetings, meta-questions, small talk, and clarifications. If intent is ambiguous, prefer a one-line clarification question over a forced tool call.
>
> When you call `ask_corpus`, your visible reply must be the grounded answer only — no `contexts`, scores, sources, chunk indices, or payloads. The system drops those automatically.

The two tool schemas (OpenAI-format):

```python
{
  "name": "store_text",
  "parameters": {
    "type": "object",
    "properties": {
      "text":    {"type": "string"},
      "source":  {"type": "string", "default": ""}
    },
    "required": ["text"]
  }
}
{
  "name": "ask_corpus",
  "parameters": {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"]
  }
}
```

### What the user sees

| LLM decision | Handler action | Reply to the user |
| --- | --- | --- |
| `store_text(text)` | `ingest_text(text, source="auto-<sha1(text[:40])[:12]>")` (or the explicit `source` the LLM passed) | `Ingested N chunks from <source>` |
| `ask_corpus(question)` | `ask(question)` (Qdrant search + grounded LLM call) | ONLY `result["answer"]` — no score, no source, no chunk_index, no payload, no `contexts` list |
| No tool call (plain chat) | pass through | the LLM's reply, verbatim |

If the configured inference endpoint does not support tool calls (or any other API error happens), the wrapper returns a clear error string and the handler returns that string. The handler itself does not raise for routing failures.

### Clarification behavior

The handler is **stateless**. The original message is dropped after classification; the next inbound message is classified fresh. There are no per-chat pending slots, no session memory, and no follow-up prompt. If the LLM replies with a short clarification question (no tool call), the handler returns that string — that's the entire interaction. The user simply answers, and the next message is classified independently.

### Attachments

If `message.attachment` is present and the suffix is `.pdf` / `.txt` / `.md` / `.text`, the handler ingests the file unconditionally **before** the LLM step and builds an `Ingested N chunks from <filename>` notice. The LLM cannot veto an attachment; once sent, it's stored. The notice is prepended to the LLM's view of the user message, so the LLM knows the file is already in the corpus and can call `ask_corpus` if the caption is a question about it. An unsupported attachment suffix (anything other than `.pdf` / `.txt` / `.md` / `.text`) raises `ValueError`.

### Example

```python
from rag_qdrant import AgentMessage, Attachment, handle_message

# Text: LLM routes to store_text
reply = handle_message(AgentMessage(text="The cat sat on the mat."))
# -> 'Ingested 1 chunks from auto-3b4f0e1a9c2d'

# Text: LLM routes to ask_corpus
reply = handle_message(AgentMessage(text="Where did the cat sit?"))
# -> 'The cat sat on the mat.'   (only the answer, no contexts)

# Text: LLM replies with a clarification question
reply = handle_message(AgentMessage(text="the cat thing"))
# -> 'Do you want me to save that, or search the corpus for cat notes?'

# Attached file (auto-store, then LLM gets the notice + caption)
reply = handle_message(
    AgentMessage(
        text="summarize this",
        attachment=Attachment("notes.pdf", open("notes.pdf", "rb").read()),
    )
)
# -> '<grounded summary>'  (only the answer)
```

See `examples/agent_usage.md` for the full integration pattern.

## References

- `references/setup.md` — environment variables, Qdrant Cloud vs. local, FastEmbed model selection, OpenAI-compatible endpoint config
- `examples/ingest_cli.md` — worked examples of `init`, `ingest-text`, `ingest-file`, `ask`
- `examples/agent_usage.md` — how an openclaw agent imports and calls the skill programmatically
- `rag_qdrant/agent_handler.py` — `AgentMessage`, `Attachment`, `handle_message` (the chat-style adapter described above)
- `rag_qdrant/cache.py` — `SemanticCache` + `SearchCache` (the caching layer described above)
