# Agent usage

How an openclaw agent imports the skill and calls it programmatically.

## Public API surface

The skill re-exports flat functions, a thin `RAG` class, and the agent-mode message handler from the top-level `rag_qdrant` package:

```python
from rag_qdrant import (
    RAG,                          # thin class (sugar over the flat functions)
    ingest_text, ingest_file,     # flat ingest
    ask, search, stats,           # flat query
    ensure_collection,            # idempotent collection + index setup
    AgentMessage, Attachment,     # agent-mode message types
    handle_message,               # chat-style dispatcher
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

## Agent-mode message handler

The `AgentMessage` / `AgentReply` / `Attachment` / `Photo` / `handle_message` surface adapts the skill to a chat-style transport. The handler is pure library code: it does not import any chat-transport package, does not perform network I/O of its own, and does not read `.env`. The agent layer is responsible for turning inbound traffic into an `AgentMessage` and for sending the returned `AgentReply` back to the user (text + optional photo paths to attach to the message).

### How routing works

The configured inference model is the **sole** decision-maker. There are no command prefixes, no `/raw` escape hatches, and no override switches — the user does not have to phrase their request in any particular way. Every inbound turn is sent to the LLM with the system prompt and two tools (`store_text` and `ask_corpus`, see `rag_qdrant/prompts.py`); the LLM picks exactly one action or replies directly.

The handler returns an `AgentReply(text, photo_paths)` from every branch:

| LLM decision | Handler action | `reply.text` | `reply.photo_paths` |
| --- | --- | --- | --- |
| `store_text(text)` (with optional `source`) | `ingest_text(text, source="auto-<sha1(text[:40])[:12]>")` (or the explicit `source` the LLM passed) | `Ingested N chunks from <source>` | `()` |
| `ask_corpus(question)` | `ask(question)` (Qdrant search + grounded LLM call) | ONLY `result["answer"]` — no score, no source, no chunk_index, no payload, no `contexts` list | `()` (or matched photo paths) |
| No tool call (greeting, meta-question, clarification) | pass through | the LLM's reply, verbatim | `()` |

Default source naming for `store_text`: `auto-<sha1[:12]>` of the first 40 characters of the stripped text. When the text is empty, the hash input falls back to the current UTC timestamp so each ingest still gets a unique source.

### Clarification behavior

The handler is **stateless**. The original message is dropped after classification; the next inbound message is classified fresh. There are no per-chat pending slots, no session memory, and no follow-up prompt. If the LLM replies with a short clarification question instead of a tool call (e.g. "Do you want me to save that, or search the corpus?"), the handler returns that string and that's the entire interaction. The user simply answers in a new turn.

### Attachments

If the inbound `AgentMessage` carries any supported attachments (`.pdf` / `.txt` / `.md` / `.text`), the handler ingests each one unconditionally **before** the LLM step and builds one `Ingested N chunks from <filename>` notice line per file. The LLM cannot veto an attachment — once sent, it's stored. The combined notice is prepended to the LLM's view of the user message, so the LLM knows the files are already in the corpus and can call `ask_corpus` if the caption is a question about them. An unsupported attachment suffix raises `ValueError` (the only `ValueError` left in the text/attachment path).

### End-to-end examples

```python
from rag_qdrant import AgentMessage, AgentReply, Attachment, handle_message

# Text: LLM routes to store_text
reply: AgentReply = handle_message(AgentMessage(text="The cat sat on the mat."))
# reply.text     == 'Ingested 1 chunks from auto-3b4f0e1a9c2d'
# reply.photo_paths == ()

# Text: LLM routes to ask_corpus (reply.text is ONLY the answer string)
reply = handle_message(AgentMessage(text="Where did the cat sit?"))
# reply.text     == 'The cat sat on the mat.'
# reply.photo_paths == ()

# Text: LLM replies with a clarification question
reply = handle_message(AgentMessage(text="the cat thing"))
# reply.text     == 'Do you want me to save that, or search the corpus for cat notes?'

# Attached file + caption (auto-store, then LLM gets the notice + caption)
with open("notes.pdf", "rb") as f:
    reply = handle_message(AgentMessage(
        text="summarize this",
        attachments=(Attachment("notes.pdf", f.read()),),
    ))
# reply.text     == '<grounded summary>'
# reply.photo_paths == ()
```

The LLM is the routing layer. You do not need to know in advance whether the user wants to save content or ask a question — just hand the message to `handle_message` and the LLM decides. If the configured inference endpoint does not support tool calls, the handler returns a clear error string in `reply.text` instead of raising, so the calling code does not need a `try/except`.

### Wiring it to a transport (sketch)

```python
from rag_qdrant import AgentMessage, AgentReply, Attachment, handle_message


def on_user_text(text: str) -> AgentReply:
    return handle_message(AgentMessage(text=text))


def on_user_file(text: str, filename: str, content: bytes) -> AgentReply:
    return handle_message(
        AgentMessage(text=text, attachments=(Attachment(filename, content),))
    )
```

The agent / bot framework is responsible for collecting `(text, file_name, file_bytes)` from the transport, calling the helpers above, and rendering the returned `AgentReply` to the user (text + any photos in `reply.photo_paths`).

## Photo support

Photos uploaded with a description are stored on disk and embedded by description. A future query that matches the description automatically surfaces the photo on the agent's reply — no opt-in flag, no extra tool call, no LLM awareness of paths.

- **Storage**: `<RAG_PHOTOS_DIR>/<sha256(bytes)[:16]>.<ext>`. Default `RAG_PHOTOS_DIR=/root/rag-photos`. Directory is created on first write. Identical bytes dedupe on disk.
- **Description required**: the description is the only searchable signal and becomes the chunk text verbatim. Empty / whitespace descriptions raise `ValueError`. Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.bmp`, `.tiff`, `.heic`, `.heif`. Missing extension is accepted.
- **Unconditional ingest**: like attachments, the LLM cannot veto a photo.
- **`ask_corpus` propagation**: matched photos from `result["photos"]` flow into `AgentReply.photo_paths`. The LLM never sees the paths.
- **Multiple photos in one message**: each is saved and ingested separately, each gets its own source (`photo-<sha12>`), each gets its own notice line.

### Examples

```python
from rag_qdrant import AgentMessage, AgentReply, Photo, handle_message

# Photo only — saved to disk, description embedded, ack lists the photo.
with open("sunset.jpg", "rb") as f:
    reply: AgentReply = handle_message(AgentMessage(
        text="",
        photos=(Photo("sunset.jpg", f.read(), "a sunset over the bay"),),
    ))
# reply.text     == 'Ingested 1 chunk from photo-5c6fb3dfe09f (sunset.jpg)'
# reply.photo_paths == ('/root/rag-photos/5c6fb3dfe09f.jpg',)

# Photo + caption — file saved, then LLM is asked about the photo.
with open("sunset.jpg", "rb") as f:
    reply = handle_message(AgentMessage(
        text="what does this look like?",
        photos=(Photo("sunset.jpg", f.read(), "a sunset over the bay"),),
    ))
# reply.text     == '<grounded answer from the LLM>'
# reply.photo_paths == ('/root/rag-photos/5c6fb3dfe09f.jpg',)  (or empty)

# Multiple photos + text in one turn — each photo saved, ack lists each.
with open("a.jpg", "rb") as fa, open("b.jpg", "rb") as fb:
    reply = handle_message(AgentMessage(
        text="any matches?",
        photos=(
            Photo("a.jpg", fa.read(), "first photo"),
            Photo("b.jpg", fb.read(), "second photo"),
        ),
    ))
# reply.text     == '<multi-line ack + LLM reply, separated by a blank line>'
# reply.photo_paths == (...)  (matched photos from the LLM's ask_corpus call)
```

### Telegram-style flow

```python
from rag_qdrant import AgentMessage, AgentReply, Photo, handle_message


def on_user_photo_with_caption(caption: str, photo_bytes: bytes, filename: str) -> AgentReply:
    return handle_message(AgentMessage(
        text=caption,
        photos=(Photo(filename, photo_bytes, caption or "user-uploaded photo"),),
    ))


# On a photo with no caption: use a generated description or fall back to a
# prompt that asks the LLM to invent one. The skill requires a non-empty
# description — the transport layer is responsible for producing one.
def on_user_photo_no_caption(photo_bytes: bytes, filename: str) -> AgentReply:
    return handle_message(AgentMessage(
        text="",
        photos=(Photo(filename, photo_bytes, "user-uploaded photo"),),
    ))


def render(reply: AgentReply) -> None:
    # Send reply.text to the chat. Then, for each path in reply.photo_paths,
    # send the corresponding photo back to the user. The transport layer
    # reads the file from disk and attaches it to the outbound message.
    send_text(reply.text)
    for path in reply.photo_paths:
        send_photo_file(path)
```
