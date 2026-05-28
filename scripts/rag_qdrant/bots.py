from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .config import settings
from .inference import answer_question
from .logging_setup import logger
from .qdrant_store import ingest_file, ingest_text


def _user_allowed(update: Update) -> bool:
    if not settings.allowed_telegram_user_ids:
        return True
    user = update.effective_user
    return bool(user and user.id in settings.allowed_telegram_user_ids)


async def _reject_if_unauthorized(update: Update) -> bool:
    if _user_allowed(update):
        return False
    user_id = update.effective_user.id if update.effective_user else "unknown"
    logger.warning("telegram_unauthorized user_id=%s", user_id)
    if update.message:
        await update.message.reply_text("Unauthorized Telegram user.")
    return True


async def ingest_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    await update.message.reply_text("Send me a PDF, TXT/MD file, or plain text. I will extract, chunk, embed, and store it in Qdrant.")


async def query_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    await update.message.reply_text("Ask a question. I will search Qdrant and answer from the retrieved context.")


async def ingest_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if not update.message or not update.message.document:
        return

    doc = update.message.document
    filename = doc.file_name or f"telegram-file-{doc.file_unique_id}"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".text"}:
        await update.message.reply_text("Unsupported file type. Send PDF, TXT, or MD files.")
        return

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    safe_name = "".join(char if char.isalnum() or char in ".-_" else "_" for char in filename)
    path = settings.upload_dir / f"{timestamp}-{doc.file_unique_id}-{safe_name}"

    await update.message.chat.send_action(ChatAction.TYPING)
    telegram_file = await doc.get_file()
    await telegram_file.download_to_drive(path)
    logger.info("telegram_ingest_file_received user_id=%s file=%s path=%s size=%s", update.effective_user.id, filename, path, doc.file_size)

    try:
        count = await asyncio.to_thread(
            ingest_file,
            path,
            source=f"telegram:{filename}",
            metadata={
                "telegram_user_id": update.effective_user.id if update.effective_user else None,
                "telegram_chat_id": update.effective_chat.id if update.effective_chat else None,
                "telegram_file_id": doc.file_id,
            },
        )
    except Exception as exc:
        logger.exception("telegram_ingest_file_failed path=%s", path)
        await update.message.reply_text(f"Ingestion failed: {exc}")
        return

    await update.message.reply_text(f"Ingested `{filename}` into Qdrant: {count} chunks.", parse_mode="Markdown")


async def ingest_message_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    if not text or text.startswith("/"):
        return

    settings.text_message_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    user_id = update.effective_user.id if update.effective_user else "unknown"
    source = f"telegram:text:{user_id}:{timestamp}"
    snapshot = settings.text_message_dir / f"{source.replace(':', '-')}.json"
    snapshot.write_text(
        json.dumps({"source": source, "text": text, "user_id": user_id, "timestamp": timestamp}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("telegram_ingest_text_received source=%s chars=%s", source, len(text))

    try:
        count = await asyncio.to_thread(
            ingest_text,
            text,
            source=source,
            metadata={
                "telegram_user_id": user_id,
                "telegram_chat_id": update.effective_chat.id if update.effective_chat else None,
                "input_type": "telegram_text",
            },
        )
    except Exception as exc:
        logger.exception("telegram_ingest_text_failed source=%s", source)
        await update.message.reply_text(f"Ingestion failed: {exc}")
        return

    await update.message.reply_text(f"Stored text in Qdrant: {count} chunks.")


async def answer_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()
    if not question or question.startswith("/"):
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    logger.info("telegram_query_received user_id=%s question_chars=%s", update.effective_user.id if update.effective_user else None, len(question))

    try:
        result = await asyncio.to_thread(answer_question, question)
    except Exception as exc:
        logger.exception("telegram_query_failed")
        await update.message.reply_text(f"Query failed: {exc}")
        return

    sources = []
    for context_item in result["contexts"]:
        source = context_item.get("source", "unknown")
        chunk_index = context_item.get("chunk_index", "?")
        score = context_item.get("score", 0)
        sources.append(f"- {source}:{chunk_index} ({score:.3f})")
    sources_text = "\n".join(sources) if sources else "No sources found."
    reply = f"{result['answer']}\n\nSources:\n{sources_text}"
    await update.message.reply_text(reply[:4000])


def build_ingest_application() -> Application:
    settings.require_ingest_bot()
    app = Application.builder().token(settings.ingest_bot_token).build()
    app.add_handler(CommandHandler("start", ingest_start))
    app.add_handler(MessageHandler(filters.Document.ALL, ingest_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ingest_message_text))
    return app


def build_query_application() -> Application:
    settings.require_query_bot()
    app = Application.builder().token(settings.query_bot_token).build()
    app.add_handler(CommandHandler("start", query_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_message))
    return app


def run_ingest_bot() -> None:
    logger.info("telegram_ingest_bot_start")
    build_ingest_application().run_polling(allowed_updates=Update.ALL_TYPES)


def run_query_bot() -> None:
    logger.info("telegram_query_bot_start")
    build_query_application().run_polling(allowed_updates=Update.ALL_TYPES)


async def run_all_bots_async() -> None:
    settings.require_ingest_bot()
    settings.require_query_bot()
    ingest_app = build_ingest_application()
    query_app = build_query_application()
    await ingest_app.initialize()
    await query_app.initialize()
    await ingest_app.start()
    await query_app.start()
    await ingest_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    await query_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("telegram_all_bots_started")
    try:
        await asyncio.Event().wait()
    finally:
        await ingest_app.updater.stop()
        await query_app.updater.stop()
        await ingest_app.stop()
        await query_app.stop()
        await ingest_app.shutdown()
        await query_app.shutdown()


def run_all_bots() -> None:
    asyncio.run(run_all_bots_async())
