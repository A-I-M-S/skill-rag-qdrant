---
name: skill-rag-qdrant
description: System RAG skill using a single Telegram bot for both ingest and query, Qdrant Cloud storage, FastEmbed multilingual E5 embeddings, and an inference model endpoint for grounded answers. Embed access is owner-managed and persistent.
compatibility: Created for Zo Computer
metadata:
  author: aloy.zo.computer
---
# skill-rag-qdrant

Use this skill to run a single Telegram bot that handles both ingestion and RAG queries:

- One Telegram bot receives text or documents. Prefix text with `embed ` (or start a document caption with `embed `) to store in Qdrant; any other text is treated as a question and answered from the retrieved context.
- The first Telegram user to ever message the bot is promoted to **OWNER**. The owner manages the embed allowlist via bot commands.
- Embed access is gated to the owner and allowlisted users. Queries are open to everyone.
- Owner and allowlist state is persisted in `data/telegram_access.json` and survives restarts.

## Setup

1. Copy `.env.example` to `.env`.
2. Fill in the Telegram bot token, Qdrant URL/API key, and inference provider settings.
3. (Optional) Fill `TELEGRAM_SEED_ALLOWLIST` with a comma-separated list of Telegram user IDs. This is consumed **only on first run** when `data/telegram_access.json` does not exist yet; after that, edit the JSON file or use the bot commands.
4. Install dependencies:

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

The first user to ever send a message to the bot becomes the **owner** automatically and is told so on their first reply. Owner state is persistent.

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
