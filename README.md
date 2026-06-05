# skill-rag-qdrant

Python system RAG skill with a single Telegram bot and Qdrant Cloud.

## Architecture

```mermaid
flowchart LR
  U[Telegram User] --> B[Telegram Bot: ingest + query]
  B -- "embed prefix text or caption" --> X[PDF/TXT extraction or plain text]
  X --> C[Chunk text]
  C --> D[FastEmbed E5 Small Multilingual]
  D --> E[Qdrant Cloud]
  B -- "any other text" --> G[Embed question]
  G --> H[Vector search Qdrant]
  H --> I[Build grounded prompt]
  I --> J[Inference model]
  J --> K[Telegram answer]
```

## Setup

```bash
cd /home/workspace/Skills/skill-rag-qdrant
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env`. Secrets must stay in `.env`; it is git-ignored.

`TELEGRAM_SEED_ALLOWLIST` is optional and is consumed only once: on first run, when `data/telegram_access.json` does not exist yet. After that, edit the JSON file or use the bot's `/allow` and `/disallow` commands.

## Commands

```bash
python -m scripts.rag_qdrant init
python -m scripts.rag_qdrant ingest-file /path/to/file.pdf
python -m scripts.rag_qdrant ingest-text "Text to index" --source manual
python -m scripts.rag_qdrant search "question"
python -m scripts.rag_qdrant ask "question"
python -m scripts.rag_qdrant stats
python -m scripts.rag_qdrant run-bot
```

## Telegram flow

The bot handles both ingestion and querying in a single conversation:

- Prefix any text with `embed ` (case-insensitive) to store it in Qdrant. Example: `embed The cat sat on the mat.`
- Send a PDF, TXT, or MD document with the caption `embed <optional note>` to store its content.
- Send any other text to ask a question. The bot searches Qdrant, builds a grounded prompt, calls the configured inference provider, and replies with an answer.

## Bot commands

| Command | Who | What it does |
|---|---|---|
| `/start` | anyone | Welcome message. |
| `/help` | anyone | List all commands and usage. |
| `/whoami` | anyone | Show your Telegram numeric user ID and your role (owner / allowed / unauthorized). |
| `/allow <user_id>` | owner only | Add a user ID to the embed allowlist. |
| `/disallow <user_id>` | owner only | Remove a user ID from the embed allowlist. The owner cannot disallow themselves. |
| `/allowlist` | owner only | Show the current owner and the allowlist. |

## Access control

- **Owner:** the first Telegram user to ever send a message to the bot is auto-promoted to owner. This is a one-time event. The owner is told on their first reply: "You are now the owner of this bot."
- **Allowlist:** the owner manages a list of additional Telegram user IDs that are allowed to embed.
- **Queries:** open to everyone, no gate.
- **Embeds:** only the owner and allowlisted users can run an `embed ` command.
- **Persistence:** owner ID and allowlist are stored in `data/telegram_access.json` and survive bot restarts. Writes are atomic (`os.replace` of a temp file). Concurrent updates are serialized with an `asyncio.Lock` inside the bot process.
- **Seeding:** `TELEGRAM_SEED_ALLOWLIST` is read once, on first run, to bootstrap the allowlist before any message is received. The first user to message the bot afterwards still becomes the owner.

## Inference providers

Default provider is OpenRouter pinned to Cloudflare with fallback disabled:

```env
INFERENCE_PROVIDER=openrouter
OPENROUTER_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_AK=...
OPENROUTER_MODEL=z-ai/glm-4.7-flash
OPENROUTER_PROVIDER=cloudflare
INFERENCE_TEMPERATURE=0.2
```

`OPENROUTER_PROVIDER=cloudflare` sends `provider.order=["cloudflare"]` and `allow_fallbacks=false`.

Legacy `zo_ask` and generic `openai_compatible` providers are still available by changing `INFERENCE_PROVIDER` and filling the matching `INFERENCE_*` settings.

## Logs

All major steps log to `logs/rag-qdrant.log`: Telegram receipt, owner claim, allowlist mutation, extraction, chunking, embedding, Qdrant collection creation/upsert/search, prompt inference, and errors.


[![Watch the video](https://img.youtube.com/vi/ib9c2kH0_Pk/0.jpg)](https://www.youtube.com/watch?v=ib9c2kH0_Pk)

[![Watch the video](https://img.youtube.com/vi/arghqsaLy8Q/0.jpg)](https://www.youtube.com/watch?v=arghqsaLy8Q)
