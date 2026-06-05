---
name: skill-rag-qdrant
description: System RAG skill using a single Telegram bot for both ingest and query, Qdrant Cloud storage, FastEmbed multilingual E5 embeddings, and a single OpenAI-compatible inference endpoint for grounded answers. Embed access is owner-managed and persistent.
compatibility: Created for Zo Computer
metadata:
  author: aloy.zo.computer
---
# skill-rag-qdrant

Use this skill to run a single Telegram bot that handles both ingestion and RAG queries:

- One Telegram bot receives text or documents. Prefix text with `embed ` (or start a document caption with `embed `) to store in Qdrant; any other text is treated as a question and answered from the retrieved context.
- The bot owner is **env-authoritative**: setting `TELEGRAM_OWNER_ID` in `.env` makes that Telegram user ID the owner on every bot start. The env var overrides any owner already in `data/telegram_access.json` and disables the first-claim fallback.
- If `TELEGRAM_OWNER_ID` is unset, the first Telegram user to ever message the bot is auto-promoted to **OWNER** (one-time event, persisted to JSON).
- The owner manages the embed allowlist via bot commands (`/allow`, `/disallow`, `/allowlist`).
- Embed access is gated to the owner and allowlisted users. Queries are open to everyone.
- Owner and allowlist state is persisted in `data/telegram_access.json` and survives restarts. The JSON file is the source of truth for the allowlist; the env var is the source of truth for the owner.

## Setup

1. Copy `.env.example` to `.env`.
2. Fill in the Telegram bot token, Qdrant URL/API key, and the single inference endpoint (`INFERENCE_BASE_URL`, `INFERENCE_API_KEY`, `INFERENCE_MODEL`).
3. **Recommended:** set `TELEGRAM_OWNER_ID` to your numeric Telegram user ID. This is the authoritative owner.
4. (Optional) Fill `TELEGRAM_SEED_ALLOWLIST` with a comma-separated list of Telegram user IDs. This is consumed **only on first run** when `data/telegram_access.json` does not exist yet; after that, edit the JSON file or use the bot commands.
5. Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m scripts.rag_qdrant init
python -m scripts.rag_qdrant run-bot
```

`init` enforces the env-authoritative owner: if `TELEGRAM_OWNER_ID` is set, it overwrites any owner already in the JSON file.

## Telegram usage

- Prefix text with `embed ` to store it in Qdrant. Example: `embed The cat sat on the mat.`
- Any other text is treated as a question and answered from the RAG collection.
- Documents (PDF/TXT/MD) are stored only if their caption starts with `embed `.

### Bot commands

| Command | Who | What it does |
|---|---|---|
| `/start` | anyone | Welcome message. |
| `/help` | anyone | Show commands and usage. |
| `/whoami` | anyone | Show your Telegram user ID and role (owner / allowed / unauthorized). |
| `/allow <user_id>` | owner only | Add a user to the embed allowlist. |
| `/disallow <user_id>` | owner only | Remove a user from the embed allowlist. The owner cannot disallow themselves. |
| `/allowlist` | owner only | Show the current owner and allowlist. |

If `TELEGRAM_OWNER_ID` is unset and no owner exists in the JSON file, the first user to send a message to the bot becomes the **owner** automatically and is told so on their first reply.

## CLI

```bash
python -m scripts.rag_qdrant --help
python -m scripts.rag_qdrant init
python -m scripts.rag_qdrant ingest-file /path/to/file.pdf
python -m scripts.rag_qdrant ingest-text "your text here" --source manual-note
python -m scripts.rag_qdrant ask "What does the document say about ...?"
python -m scripts.rag_qdrant stats
```

## Logs and storage

- Logs: `logs/rag-qdrant.log`
- Telegram uploads: `storage/uploads/`
- Telegram text message snapshots: `storage/text_messages/`
- Owner and allowlist: `data/telegram_access.json`

Sensitive values belong only in `.env`; `.env`, `data/`, logs, local uploads, and Python caches are ignored by git.
