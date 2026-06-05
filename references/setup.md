# Setup

This skill is configured entirely through `.env` at the skill root. Copy `.env.example` to `.env` and fill in the values for your environment.

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `QDRANT_URL` | yes | — | URL of your Qdrant instance. Cloud or self-hosted. |
| `QDRANT_API_KEY` | yes | — | API key for the Qdrant instance. |
| `QDRANT_COLLECTION` | no | `system_rag` | Name of the Qdrant collection. Created on first `init`. |
| `FASTEMBED_MODEL` | no | `intfloat/multilingual-e5-small` | Any model in `TextEmbedding.list_supported_models()`. The default has a custom registration that injects the required `query:` / `passage:` prefixes. |
| `EMBEDDING_DIM` | no | `384` | Vector dimension. Must match the chosen FastEmbed model. |
| `CHUNK_SIZE` | no | `900` | Target chunk size in characters. |
| `CHUNK_OVERLAP` | no | `150` | Chunk overlap in characters. Must be `>= 0` and `< CHUNK_SIZE`. |
| `TOP_K` | no | `6` | Number of contexts retrieved per query. |
| `MIN_RELEVANCE_SCORE` | no | `0.78` | Cosine-similarity floor. Contexts below it are dropped before the LLM call; if no contexts remain, the answer is exactly `No relevant information found`. |
| `INFERENCE_BASE_URL` | yes | — | Any OpenAI-compatible chat-completion base URL. Trailing slashes are stripped. |
| `INFERENCE_API_KEY` | yes | — | API key for the inference endpoint. |
| `INFERENCE_MODEL` | yes | — | Model name to pass to `chat.completions.create`. |
| `INFERENCE_TEMPERATURE` | no | `0.2` | Sampling temperature. |
| `LOG_LEVEL` | no | `INFO` | Standard level name. |
| `LOG_FILE` | no | `logs/rag-qdrant.log` | Path is resolved relative to the skill root. Rotates at 5 MB, keeps 5 backups. |

`QDRANT_URL` may also be a placeholder like `${SOMETHING}`; in that case the value of `SOMETHING` is used. This is mostly useful for secret manager indirection.

## Qdrant Cloud

1. Create a free cluster at <https://cloud.qdrant.io>.
2. Copy the cluster URL and API key from the cluster dashboard.
3. Set:

   ```env
   QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io
   QDRANT_API_KEY=<your-api-key>
   QDRANT_COLLECTION=system_rag
   ```

4. Run `python -m rag_qdrant init` once to create the collection and payload indexes.

## Local Qdrant

Either run Qdrant via Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

or install the Qdrant server binary from <https://qdrant.tech/documentation/guides/installation/>.

Then point at it without an API key:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

The client always sends `api_key` if it is set, but the `QdrantClient` constructor accepts an empty string for unauthenticated local instances.

## FastEmbed model selection

The default is `intfloat/multilingual-e5-small` (384-dim, ~0.47 GB, MIT). The model is multilingual and requires `query:` / `passage:` prefixes; the skill adds those automatically based on whether you are embedding a query or a chunk.

To switch models:

1. Pick a model from the FastEmbed catalog:

   ```python
   from fastembed import TextEmbedding
   for m in TextEmbedding.list_supported_models():
       print(m["model"], m["dim"])
   ```

2. Set `FASTEMBED_MODEL` and `EMBEDDING_DIM` to match. For example, `intfloat/multilingual-e5-base` is 768-dim.

3. Run `python -m rag_qdrant init` to recreate the collection with the new dimension. Note: this will NOT delete the old collection; create a new collection name and re-ingest if you need a clean break.

If you select a non-default model whose name happens to be `intfloat/multilingual-e5-small`, the skill registers a custom model entry pointing at the Hugging Face mirror that FastEmbed expects. For any other custom model, register it yourself in code before calling `get_embedding_model()`.

## OpenAI-compatible endpoint config

Any service that speaks the OpenAI `chat.completions` schema works. Examples:

| Provider | `INFERENCE_BASE_URL` | `INFERENCE_MODEL` |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4.1`, etc. |
| vLLM (local) | `http://localhost:8000/v1` | whatever you launched with `--served-model-name` |
| Ollama (with OpenAI compat shim) | `http://localhost:11434/v1` | the model tag, e.g. `llama3.1` |
| LM Studio (OpenAI-compat server) | `http://localhost:1234/v1` | the loaded model id |
| Any OpenRouter-style or proxy | `<base>/v1` | the model name the proxy exposes |

The skill instantiates a fresh `OpenAI(api_key=..., base_url=...)` client per call. There is no connection pooling, no retries, and no provider dispatch — one URL, one key, one model.

## Verifying the install

```bash
python -m rag_qdrant init     # creates the collection
python -m rag_qdrant stats    # reports points_count, indexed_vectors_count, status
```

If `stats` reports `points_count: 0` and `status: green`, you are ready to ingest.
