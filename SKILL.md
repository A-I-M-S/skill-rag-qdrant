---
name: rag-qdrant
description: Local RAG skill that ingests text/PDF/MD/photos into Qdrant and answers questions with a single OpenAI-compatible chat endpoint. Per-chunk Telegram user-id ACL, admin tools (grant / revoke / show access), and `@username` resolution via a local user cache. FastEmbed multilingual E5 embeddings.
user-invocable: true
metadata:
  openclaw:
    emoji: "📚"
    requires:
      bins: ["python"]
      anyBins: []
      env: ["QDRANT_URL", "QDRANT_API_KEY", "INFERENCE_BASE_URL", "INFERENCE_API_KEY", "INFERENCE_MODEL", "RAG_PHOTOS_DIR", "ADMIN_TELEGRAM_IDS"]
    primaryEnv: "INFERENCE_API_KEY"
    install:
      - id: pip
        kind: pip
        package: "-r requirements.txt"
        label: "Install Python dependencies (qdrant-client, fastembed, openai, pypdf, python-dotenv)"
---

# rag-qdrant

A single Qdrant-backed RAG skill with per-chunk Telegram user-id ACL. Ingest text, PDF, Markdown, or photos into a configurable Qdrant collection with FastEmbed multilingual E5 embeddings, then ask grounded questions answered by one OpenAI-compatible chat endpoint. Admins can grant / revoke / show access by `@username` or numeric Telegram id (resolved through a local user cache).

The skill has no UI of its own. Use the CLI or import the Python API from an openclaw agent.

> **Install on a dedicated OpenClaw instance.** This skill is intended to run as the user-facing Telegram bot for its intended operator. The dev / test instance is a separate OpenClaw install.

## Setup

1. Copy `.env.example` to `.env` and fill in your Qdrant connection and inference endpoint. **Set `ADMIN_TELEGRAM_IDS`** to the comma-separated Telegram user ids of the people who can use the admin tools (required at import time; fail-closed if missing).
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

> You're an office worker. Slightly grumpy, slightly clipped. Short sentences. No exclamation marks. No "as an AI". No "I'd be happy to help". No "Great question". No "Sure," or "Certainly" or "Of course". No apology, no preamble, no closing line.
>
> You have two tools and one chat path. Pick at most one per turn.
>
> 1. `store_text(text, source="")` — file something in the cabinet.
> 2. `ask_corpus(question)` — search the cabinet and answer the question from what you find.
>
> Chat path (no tool call): greetings, meta-questions, small talk, thanks, follow-up clarifications. One short grumpy sentence. No list. No warmth. "Mm." is a complete answer. "Noted." is a complete answer.
>
> When you call `ask_corpus`, your visible reply is the grounded answer only — no contexts, scores, source ids, chunk indices, payloads, or "Based on the context". The system drops all of that automatically.

The two public tool schemas (OpenAI-format):

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
| `ask_corpus(question)` | `ask(question)` (Qdrant search + grounded LLM call, ACL-filtered by caller's Telegram id) | ONLY `result["answer"]` — no score, no source, no chunk_index, no payload, no `contexts` list. If the search returned nothing (ACL-filtered or genuinely absent), the bot replies **"Nothing on that."** — same line for both cases; the filter is never explained. |
| `grant_access(source, telegram_id)` (admin only) | append `telegram_id` (numeric or `@username`) to every chunk under `source` | `Granted access to <id> for source <source> (N chunk(s) updated).` |
| `revoke_access(source, telegram_id)` (admin only) | remove `telegram_id` from every chunk under `source` | `Revoked access for <id> from source <source> (N chunk(s) updated).` |
| `show_access(source)` (admin only) | read ACL + chunk count for `source` | list of ids + chunk count, or `No chunks found for source <source>.` |
| `resolve_username(username)` (admin only) | look up `@username` in the local user cache | `@<username> (Telegram id <id>, "<first_name>").` or `That user hasn't messaged this bot yet…` if the user has never DM'd the bot. |
| No tool call (plain chat) | pass through | the LLM's reply, verbatim. Greetings like "hi" get one short grumpy sentence ("Mm."), not "Hello! How can I help you today?" |

If the configured inference endpoint does not support tool calls (or any other API error happens), the wrapper returns a clear error string and the handler returns that string. The handler itself does not raise for routing failures.

### Admin tools

`store_text`, `grant_access`, `revoke_access`, `show_access`, and `resolve_username` are admin-only. A non-admin caller asking for any of them gets `This action is only available to admins. Ask an admin to grant you access or perform the action for you.` — no Qdrant write happens. The LLM cannot bypass the gate; the handler re-checks the caller's role against `ADMIN_TELEGRAM_IDS` after the model decides which tool to call.

`grant_access` and `revoke_access` accept either a numeric Telegram id (`"123"`) or a `@username` (`"@alice"`, `"alice"`). `@username` values are resolved through the local user cache at `RAG_USER_CACHE_PATH` (default `logs/user_cache.json`); the user must have sent the bot at least one direct message before they can be looked up. The cache is auto-populated on every inbound message and stored with mode `0600`. When the username can't be resolved, the bot says `That user hasn't messaged this bot yet. Ask them to send any message to the bot first, then try again.` and skips the Qdrant write.

### Clarification behavior

The handler is **stateless**. The original message is dropped after classification; the next inbound message is classified fresh. There are no per-chat pending slots, no session memory, and no follow-up prompt. If the LLM replies with a short clarification question (no tool call), the handler returns that string — that's the entire interaction. The user simply answers, and the next message is classified independently.

### Attachments

If `message.attachments` contains any files whose suffix is `.pdf` / `.txt` / `.md` / `.text`, the handler ingests each one unconditionally **before** the LLM step and builds one `Ingested N chunks from <filename>` notice line per file. The LLM cannot veto an attachment; once sent, it's stored. The combined notice is prepended to the LLM's view of the user message, so the LLM knows the files are already in the corpus and can call `ask_corpus` if the caption is a question about them. An unsupported attachment suffix (anything other than `.pdf` / `.txt` / `.md` / `.text`) raises `ValueError`.

### Example

```python
from rag_qdrant import AgentMessage, AgentReply, Attachment, handle_message

# Text: LLM routes to store_text
reply: AgentReply = handle_message(AgentMessage(text="The cat sat on the mat."))
# reply.text     == 'Ingested 1 chunks from auto-3b4f0e1a9c2d'
# reply.photo_paths == ()

# Text: LLM routes to ask_corpus
reply = handle_message(AgentMessage(text="Where did the cat sit?"))
# reply.text     == 'The cat sat on the mat.'   (only the answer, no contexts)
# reply.photo_paths == ()  (no matched photos)

# Text: LLM replies with a clarification question
reply = handle_message(AgentMessage(text="the cat thing"))
# reply.text     == 'Do you want me to save that, or search the corpus for cat notes?'

# Attached file (auto-store, then LLM gets the notice + caption)
reply = handle_message(
    AgentMessage(
        text="summarize this",
        attachments=(Attachment("notes.pdf", open("notes.pdf", "rb").read()),),
    )
)
# reply.text     == '<grounded summary>'  (only the answer)
# reply.photo_paths == ()
```

See `examples/agent_usage.md` for the full integration pattern.

## Photo support

Photos uploaded with a description are stored on disk and embedded by description. A future query that matches the description automatically surfaces the photo on the agent's reply — no opt-in flag, no extra tool call, no LLM awareness of paths.

Public types: `Photo(filename, content, description)`, `AgentReply(text, photo_paths)`. Configuration: env var `RAG_PHOTOS_DIR`, default `/root/rag-photos`.

### Storage layout

Photos are saved to `<RAG_PHOTOS_DIR>/<sha256(bytes)[:16]>.<ext>` where `<ext>` is the lowercase original extension (with the dot, or empty if the original has none). The directory is created on the first write. Identical bytes + same filename suffix dedupe automatically: only one file is ever written for a given content + filename, and a second call returns the same path. The skill does not decode the bytes — it stores them verbatim.

### Description is required

The user-supplied description is the **only** signal that gets embedded into the vector index, and it becomes the chunk text verbatim (no templating like `"Photo: <desc>"`). Empty or whitespace-only descriptions raise `ValueError` before any disk or Qdrant work. Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.bmp`, `.tiff`, `.heic`, `.heif`. A missing extension is accepted; a present-but-unrecognized extension raises `ValueError`.

### Qdrant payload

A photo point in the corpus carries: `text` = the description, `source` = `photo-<sha256[:12]>`, `chunk_index` = 0, `chunk_count` = 1, `photo_path` = absolute path on disk, `photo_filename` = original filename, `file_type` = lowercase extension with the dot, `kind` = `"photo"`, `sha256` = full hex digest. The same point ID is reused when re-ingesting the same photo with the same description (idempotent); a different description produces a second point under the same source.

### Ingest is unconditional

The LLM cannot veto a photo. Once `Photo` is on the `AgentMessage`, it is saved to disk and its description is embedded — mirroring the existing attachment path. The handler builds an `Ingested 1 chunk from photo-<hash> (<filename>)` notice line per photo; for a multi-photo turn the lines are joined with `\n`.

### `ask_corpus` → matched photos

When the LLM routes to `ask_corpus`, `rag_qdrant.ask` returns the matched contexts plus a `photos` list (deduped by `photo_path`, first-seen order). The handler copies those paths into `AgentReply.photo_paths` and strips the LLM-facing metadata (no scores, sources, payloads, or paths in the visible answer). The LLM itself never sees the photo paths — only the system enriches the reply.

### Example

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
# reply.photo_paths == ('/root/rag-photos/5c6fb3dfe09f.jpg',)  (or empty if no match)
```

Programmatic use (no agent transport):

```python
from rag_qdrant import Photo, RAG

rag = RAG()
with open("sunset.jpg", "rb") as f:
    rag.ingest_photo(Photo("sunset.jpg", f.read(), "a sunset over the bay"))

result = rag.ask("Show me sunsets.")
# result["answer"] == '<grounded answer>'
# result["photos"] == [{'path': '/root/rag-photos/5c6fb3dfe09f.jpg',
#                        'filename': 'sunset.jpg',
#                        'source': 'photo-5c6fb3dfe09f',
#                        'score': 0.91}]
```

## References

- `references/setup.md` — environment variables, Qdrant Cloud vs. local, FastEmbed model selection, OpenAI-compatible endpoint config
- `examples/ingest_cli.md` — worked examples of `init`, `ingest-text`, `ingest-file`, `ask`
- `examples/agent_usage.md` — how an openclaw agent imports and calls the skill programmatically
- `rag_qdrant/agent_handler.py` — `AgentMessage`, `AgentReply`, `Attachment`, `Photo`, `handle_message` (the chat-style adapter described above)
- `rag_qdrant/photo_store.py` — `Photo` dataclass + `save_photo` (content-addressed on-disk dedupe)
- `rag_qdrant/photo_matching.py` — `extract_photos(contexts)` for the `ask_corpus` photo propagation
- `rag_qdrant/qdrant_store.py` — `ingest_photo(...)` (description-as-chunk)
- `rag_qdrant/cache.py` — `SemanticCache` + `SearchCache` (the caching layer described above)
