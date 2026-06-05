# Ingest + query CLI examples

All examples assume:

- `.env` is filled in with `QDRANT_URL`, `QDRANT_API_KEY`, `INFERENCE_BASE_URL`, `INFERENCE_API_KEY`, `INFERENCE_MODEL`.
- The current working directory is the skill root.
- You have a virtualenv activated with `pip install -r requirements.txt`.

## 1. `init`

Create the Qdrant collection and payload indexes if they don't exist. Idempotent: running it twice is safe. The second run logs `qdrant_collection_exists` and re-asserts the indexes (idempotent at the Qdrant level).

```bash
python -m rag_qdrant init
```

Sample output:

```json
{
  "collection": "system_rag",
  "points_count": 0,
  "indexed_vectors_count": 0,
  "status": "green"
}
```

If the collection already has vectors, `points_count` will be non-zero.

## 2. `ingest-text`

Ingest a raw string. The `--source` is the stable identifier used in the point-id hash and in the `source` payload field; it is also what you would filter on if you later wanted to scope a search.

```bash
python -m rag_qdrant ingest-text "The cat sat on the mat. The cat was orange." --source manual-note
```

Sample output:

```json
{
  "ingested_chunks": 1
}
```

If the text is empty or whitespace-only, `ingested_chunks` is `0` and a warning is logged.

## 3. `ingest-file`

Ingest a file. Supported suffixes: `.pdf`, `.txt`, `.md`, `.text`. Anything else raises a clear `ValueError`.

```bash
python -m rag_qdrant ingest-file /path/to/notes.pdf
python -m rag_qdrant ingest-file /path/to/notes.md --source meeting-2026-06-05
```

When `--source` is omitted, the filename is used. The skill always adds `file_name` and `file_type` to the payload metadata; anything you put in `--source` shows up in the `source` field used for filtering.

Sample output:

```json
{
  "ingested_chunks": 14
}
```

Chunk count depends on `CHUNK_SIZE` / `CHUNK_OVERLAP` and the text density. With the defaults (900 / 150), a typical 10-page PDF lands at roughly 10–30 chunks.

### Verifying the ingest

```bash
python -m rag_qdrant stats
```

Look for `points_count` going up by the number of chunks you ingested.

## 4. `search`

Raw vector search. Returns the top-K contexts that match the question, with their cosine-similarity scores. No LLM is called.

```bash
python -m rag_qdrant search "what does the document say about chunking?" --top-k 8
```

`--top-k` is optional and defaults to `TOP_K` from `.env` (6 by default).

Sample output (truncated):

```json
[
  {
    "score": 0.91,
    "id": "8c2c3a5b-...",
    "text": "Chunks are produced by sliding a 900-character window...",
    "source": "meeting-2026-06-05",
    "chunk_index": 3,
    "payload": {
      "text": "Chunks are produced by sliding a 900-character window...",
      "source": "meeting-2026-06-05",
      "chunk_index": 3,
      "chunk_count": 14,
      "file_name": "notes.md",
      "file_type": ".md"
    }
  }
]
```

Use `search` when you want to inspect what the retriever is finding, or when you want to feed the contexts into a different downstream system.

## 5. `ask`

Search + grounded answer. The skill:

1. Embeds the question.
2. Pulls the top `TOP_K` contexts from Qdrant.
3. Drops any context whose cosine similarity is below `MIN_RELEVANCE_SCORE` (default `0.78`).
4. If at least one context survives, builds a prompt and calls the configured inference model.
5. If no context survives, returns exactly `No relevant information found` and an empty `contexts` list. No LLM call is made in that case.

```bash
python -m rag_qdrant ask "What does the document say about chunking?"
```

Sample output:

```json
{
  "answer": "Chunks are produced by sliding a 900-character window with 150 characters of overlap, and the chunker prefers to break on sentence boundaries when possible.",
  "contexts": [
    {
      "score": 0.91,
      "text": "Chunks are produced by sliding a 900-character window...",
      "source": "meeting-2026-06-05",
      "chunk_index": 3,
      "payload": { "...": "..." }
    }
  ]
}
```

The system prompt instructs the model to answer only from the provided context and to reply `No relevant information found` if the context is insufficient. The skill does not post-process the answer for you — the model is the source of truth, subject to the relevance floor.

## End-to-end smoke test

```bash
python -m rag_qdrant init
python -m rag_qdrant ingest-text "OpenClaw is an agent runtime for skills. Skills are Python packages." --source smoke-test
python -m rag_qdrant ask "What is OpenClaw?"
```

Expected: a non-empty `answer` whose content reflects the ingested sentence, and at least one context with `score >= 0.78`.
