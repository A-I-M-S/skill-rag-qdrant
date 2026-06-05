from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .access_control import get_access_store
from .config import settings
from .inference import answer_question
from .logging_setup import logger
from .prefix import parse_prefix
from .qdrant_store import ingest_file, ingest_text


def _embed_allowed(user_id: int | None) -> bool:
    if user_id is None:
        return False
    return get_access_store().is_allowed(user_id)


def _query_allowed(_user_id: int | None) -> bool:
    return True


async def _ensure_owner(update: Update) -> bool:
    user = update.effective_user
    if not user:
        return False
    if settings.telegram_owner_id is not None:
        return False
    store = get_access_store()
    if store.is_owner_set():
        return False
    claimed = await store.claim_owner(user.id)
    return claimed


def _owner_notice(claimed: bool) -> str:
    if not claimed:
        return ""
    return "\n\nYou are now the owner of this bot. Use /help to see owner commands."


HELP_TEXT = (
    "Commands:\n"
    "/start - welcome message\n"
    "/help - this help\n"
    "/whoami - show your Telegram user ID and role\n"
    "/allow <user_id> - (owner only) add a user to the embed allowlist\n"
    "/disallow <user_id> - (owner only) remove a user from the embed allowlist\n"
    "/allowlist - (owner only) show the owner and current allowlist\n"
    "\n"
    "Usage:\n"
    "Prefix text with 'embed ' to store it in Qdrant. Example: embed The cat sat on the mat.\n"
    "Any other text is treated as a question and answered from the RAG collection.\n"
    "Documents (PDF, TXT, MD) are stored only if their caption starts with 'embed '."
)


def _start_text(claimed_owner: bool) -> str:
    return (
        "Hello! I am a single RAG bot. I ingest and query.\n"
        "Prefix text with 'embed ' to store it in Qdrant. Any other text is a question.\n"
        "Documents (PDF/TXT/MD) need an 'embed ' caption to be stored."
    ) + _owner_notice(claimed_owner)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    claimed = await _ensure_owner(update)
    await update.message.reply_text(_start_text(claimed))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await _ensure_owner(update)
    await update.message.reply_text(HELP_TEXT)


def _role_for(user_id: int) -> str:
    store = get_access_store()
    if store.is_owner(user_id):
        return "owner"
    if store.is_allowed(user_id):
        return "allowed"
    return "unauthorized"


async def cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    await _ensure_owner(update)
    user = update.effective_user
    role = _role_for(user.id)
    await update.message.reply_text(
        f"Your Telegram user ID: `{user.id}`\nRole: {role}",
        parse_mode="Markdown",
    )


def _parse_user_id_arg(context: ContextTypes.DEFAULT_TYPE) -> int | None:
    args = context.args or []
    if len(args) != 1:
        return None
    raw = args[0].strip()
    if not raw.lstrip("-").isdigit():
        return None
    value = int(raw)
    if value <= 0:
        return None
    return value


async def cmd_allow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    claimed = await _ensure_owner(update)
    store = get_access_store()
    if not store.is_owner(update.effective_user.id):
        await update.message.reply_text("Only the owner can run /allow." + _owner_notice(claimed))
        return
    target = _parse_user_id_arg(context)
    if target is None:
        await update.message.reply_text("Usage: /allow <user_id>  (positive integer)")
        return
    if target == store.owner_id:
        await update.message.reply_text("That user ID is the owner and is already allowed.")
        return
    added = await store.allow(target)
    if added:
        await update.message.reply_text(f"Added user `{target}` to the embed allowlist.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"User `{target}` is already on the embed allowlist.", parse_mode="Markdown")


async def cmd_disallow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    claimed = await _ensure_owner(update)
    store = get_access_store()
    if not store.is_owner(update.effective_user.id):
        await update.message.reply_text("Only the owner can run /disallow." + _owner_notice(claimed))
        return
    target = _parse_user_id_arg(context)
    if target is None:
        await update.message.reply_text("Usage: /disallow <user_id>  (positive integer)")
        return
    if target == store.owner_id:
        await update.message.reply_text("The owner cannot disallow themselves.")
        return
    removed = await store.disallow(target)
    if removed:
        await update.message.reply_text(f"Removed user `{target}` from the embed allowlist.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"User `{target}` was not on the embed allowlist.", parse_mode="Markdown")


async def cmd_allowlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    claimed = await _ensure_owner(update)
    store = get_access_store()
    if not store.is_owner(update.effective_user.id):
        await update.message.reply_text("Only the owner can run /allowlist." + _owner_notice(claimed))
        return
    owner = store.owner_id
    allowed = store.get_allowed()
    lines = [f"Owner: `{owner}`" if owner is not None else "Owner: (unset)", "Allowlist:"]
    if allowed:
        lines.extend(f"- `{uid}`" for uid in allowed)
    else:
        lines.append("(empty)")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


EMBED_REJECT_QUERY_HINT = " Send a question (no 'embed' prefix) to query."

EMBED_USAGE_HINT = "Send `embed` followed by text, or attach a file with `embed` as its caption."

DOC_GUIDANCE = (
    "To embed this file, add 'embed' as its caption. Otherwise send a question "
    "(no 'embed' prefix) to query."
)


async def unified_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document or not update.effective_user:
        return

    claimed = await _ensure_owner(update)
    user_id = update.effective_user.id
    doc = update.message.document
    filename = doc.file_name or f"telegram-file-{doc.file_unique_id}"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".text"}:
        await update.message.reply_text("Unsupported file type. Send PDF, TXT, or MD files." + _owner_notice(claimed))
        return

    caption = update.message.caption or ""
    mode, body = parse_prefix(caption)

    if mode != "embed":
        await update.message.reply_text(DOC_GUIDANCE + _owner_notice(claimed))
        return

    if not _embed_allowed(user_id):
        await update.message.reply_text(
            "Only the owner or allowed users can embed." + EMBED_REJECT_QUERY_HINT + _owner_notice(claimed)
        )
        return

    stripped_caption = body.strip()

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    safe_name = "".join(char if char.isalnum() or char in ".-_" else "_" for char in filename)
    path = settings.upload_dir / f"{timestamp}-{doc.file_unique_id}-{safe_name}"

    await update.message.chat.send_action(ChatAction.TYPING)
    telegram_file = await doc.get_file()
    await telegram_file.download_to_drive(path)
    logger.info(
        "telegram_ingest_file_received user_id=%s file=%s path=%s size=%s",
        user_id, filename, path, doc.file_size,
    )

    try:
        count = await asyncio.to_thread(
            ingest_file,
            path,
            source=f"telegram:embed:{filename}",
            metadata={
                "telegram_user_id": user_id,
                "telegram_chat_id": update.effective_chat.id if update.effective_chat else None,
                "telegram_file_id": doc.file_id,
                "telegram_caption": stripped_caption,
            },
        )
    except Exception as exc:
        logger.exception("telegram_ingest_file_failed path=%s", path)
        await update.message.reply_text(f"Ingestion failed: {exc}" + _owner_notice(claimed))
        return

    await update.message.reply_text(
        f"Ingested `{filename}` into Qdrant: {count} chunks." + _owner_notice(claimed),
        parse_mode="Markdown",
    )


async def unified_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text or not update.effective_user:
        return

    claimed = await _ensure_owner(update)
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        return

    mode, body = parse_prefix(text)

    if mode == "embed":
        if not body.strip():
            await update.message.reply_text(EMBED_USAGE_HINT + _owner_notice(claimed))
            return
        if not _embed_allowed(user_id):
            await update.message.reply_text(
                "Only the owner or allowed users can embed." + EMBED_REJECT_QUERY_HINT + _owner_notice(claimed)
            )
            return

        settings.text_message_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        source = f"telegram:embed:{user_id}:{timestamp}"
        snapshot = settings.text_message_dir / f"{source.replace(':', '-')}.json"
        snapshot.write_text(
            json.dumps(
                {"source": source, "text": body, "user_id": user_id, "timestamp": timestamp},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("telegram_ingest_text_received source=%s chars=%s", source, len(body))

        try:
            count = await asyncio.to_thread(
                ingest_text,
                body,
                source=source,
                metadata={
                    "telegram_user_id": user_id,
                    "telegram_chat_id": update.effective_chat.id if update.effective_chat else None,
                    "input_type": "telegram_embed_text",
                },
            )
        except Exception as exc:
            logger.exception("telegram_ingest_text_failed source=%s", source)
            await update.message.reply_text(f"Ingestion failed: {exc}" + _owner_notice(claimed))
            return

        await update.message.reply_text(f"Stored text in Qdrant: {count} chunks." + _owner_notice(claimed))
        return

    question = text
    await update.message.chat.send_action(ChatAction.TYPING)
    logger.info(
        "telegram_query_received user_id=%s question_chars=%s",
        user_id, len(question),
    )

    try:
        result = await asyncio.to_thread(answer_question, question)
    except Exception as exc:
        logger.exception("telegram_query_failed")
        await update.message.reply_text(f"Query failed: {exc}" + _owner_notice(claimed))
        return

    reply = result["answer"]
    await update.message.reply_text(reply[:4000] + _owner_notice(claimed))


async def post_init(application: Application) -> None:
    store = get_access_store()
    await store.load()


def build_unified_application() -> Application:
    settings.require_bot()
    app = Application.builder().token(settings.bot_token).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(CommandHandler("allow", cmd_allow))
    app.add_handler(CommandHandler("disallow", cmd_disallow))
    app.add_handler(CommandHandler("allowlist", cmd_allowlist))
    app.add_handler(MessageHandler(filters.Document.ALL, unified_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unified_text))
    return app


def run_bot() -> None:
    logger.info("telegram_unified_bot_start")
    build_unified_application().run_polling(allowed_updates=Update.ALL_TYPES)
