# Agent usage

How an openclaw agent imports the skill and calls it programmatically.

## Public API surface

The skill re-exports flat functions and a thin `RAG` class from the top-level `rag_qdrant` package:

```python
from rag_qdrant import (
    RAG,                          # thin class (sugar over the flat functions)
    ingest_text, ingest_file,     # flat ingest
    ask, search, stats,           # flat query
    ensure_collection,            # idempotent collection + index setup
    settings, __version__,        # configuration + version
)
```

`ask` and `answer_question` are aliases. `stats` and `collection_stats` are aliases.

## The flat-function style

Use this when you want to be explicit and avoid holding any per-agent state.

```python
from rag_qdrant import ensure_collection, ingest_text, ask, search, stats

ensure_collection()

# Ingest
chunks = ingest_text(
    "OpenClaw is an agent runtime. Skills are Python packages that expose a CLI and a Python API.",
    source="openclaw-overview",
)
print(f"indexed {chunks} chunks")

# Retrieve raw contexts
contexts = search("What is OpenClaw?", top_k=4)
for hit in contexts:
    print(hit["score"], hit["source"], hit["text"][:80])

# Ask end-to-end
result = ask("What is OpenClaw?")
print(result["answer"])
print(f"  (grounded on {len(result['contexts'])} contexts)")
```

`result` is always a dict shaped like:

```python
{
    "answer": "<model answer or 'No relevant information found'>",
    "contexts": [
        {
            "score": 0.91,
            "id": "...",
            "text": "...",
            "source": "...",
            "chunk_index": 3,
            "payload": {...},     # full Qdrant payload
        },
        ...
    ],
}
```

If the retriever finds no context above `MIN_RELEVANCE_SCORE`, `answer` is exactly `No relevant information found` and `contexts` is `[]`. The skill never invents context.

## The RAG class style

Use this when you want to keep a single object around, or when you want to inject a custom `Settings`.

```python
from rag_qdrant import RAG, Settings

# Default: uses .env-loaded settings
rag = RAG()
rag.ensure_collection()
rag.ingest_text("The cat sat on the mat.", source="manual-note")
result = rag.ask("Where did the cat sit?")
print(result["answer"])

# Custom settings (e.g. an isolated test collection)
custom = Settings(
    qdrant_url=...,
    qdrant_api_key=...,
    qdrant_collection="agent_test_rag",
    inference_base_url=...,
    inference_api_key=...,
    inference_model=...,
)
test_rag = RAG(custom_settings=custom)
test_rag.ensure_collection()
test_rag.ingest_file("/tmp/notes.pdf")
print(test_rag.ask("summarize")["answer"])
```

`RAG` is intentionally thin. Every method delegates to the corresponding flat function with `self._settings` as the default. There is no caching, no connection pool, and no per-instance state beyond the settings reference.

## File ingestion with custom metadata

Both `ingest_text` and `ingest_file` take an optional `metadata` dict. Whatever you put there is merged into the Qdrant payload. Use this to scope future searches by user, session, or document version.

```python
from rag_qdrant import ingest_text, search

ingest_text(
    "Project Atlas kickoff notes ...",
    source="project-atlas-2026-06-05",
    metadata={"user_id": "agent-007", "project": "atlas"},
)

# Later: filter by metadata
# (search() does not currently expose a Qdrant filter, but the metadata
# is stored on every point and can be used by any external query.)
```

## From an openclaw tool / agent

A typical openclaw tool wraps the flat function in a single async function and surfaces it to the model:

```python
from rag_qdrant import ask, ingest_text, search, stats

def tool_ask(question: str, top_k: int | None = None) -> dict:
    return ask(question) if top_k is None else {"answer": "...", "contexts": search(question, top_k=top_k)}

def tool_ingest_text(text: str, source: str) -> int:
    return ingest_text(text, source=source)

def tool_stats() -> dict:
    return stats()
```

Return the raw dicts to the model. The `contexts` list is useful for agentic loops that want to inspect the retrieved evidence before answering.

## Error handling

- If the inference endpoint is not configured (`INFERENCE_BASE_URL` / `INFERENCE_API_KEY` / `INFERENCE_MODEL` missing), `ask()` raises `RuntimeError` with a clear message naming the missing variable. Configure your `.env` and retry.
- If Qdrant is not configured (`QDRANT_URL` / `QDRANT_API_KEY` missing), any function that touches Qdrant raises `RuntimeError`. The same fix.
- If the collection does not exist, `ensure_collection()` creates it. The other functions call `ensure_collection()` internally before doing real work, so the very first call after a fresh setup will create the collection lazily.
- File ingestion of an unsupported suffix (anything other than `.pdf`, `.txt`, `.md`, `.text`) raises `ValueError` with a clear message.
