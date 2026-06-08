"""Agent-mode message handler.

Adapts the rag-qdrant skill to a chat-style transport (Telegram, webhook,
REPL, openclaw agent, etc.) by handing every inbound
:class:`AgentMessage` to the configured inference model and letting the
LLM decide what to do. The handler is pure library code: it does not
import any transport package, does not perform network I/O of its own,
and does not touch ``.env`` / config. The agent layer is responsible
for turning inbound traffic into an :class:`AgentMessage` and for
sending the returned string back to the user.

Flow (executed in order):

1. If the message has a supported attachment (``.pdf`` / ``.txt`` /
   ``.md`` / ``.text``), the handler ingests the file unconditionally
   and builds an ``Ingested N chunks from <source>`` notice. The LLM
   cannot veto an attachment — once sent, it's stored. An unsupported
   attachment suffix raises :class:`ValueError`.

2. If after step 1 there is no non-empty text, the handler returns the
   attachment notice (or the empty string when there is no attachment).
   No LLM call is made.

3. Otherwise the handler calls
   :func:`rag_qdrant.inference.classify_and_route` with the system
   prompt, the two tool schemas, and the user text (with the
   attachment notice prepended when present). The LLM is the sole
   decision-maker. The handler then dispatches on the LLM's choice:

   - ``store_text`` → :func:`rag_qdrant.ingest_text` with a default
     ``auto-<sha1[:12]>`` source (or the explicit ``source`` the LLM
     passed). Reply: ``Ingested N chunks from <source>``.
   - ``ask_corpus`` → :func:`rag_qdrant.ask` (Qdrant search + grounded
     LLM call). Reply: **only** ``result["answer"]``; the
     ``contexts`` list, scores, sources, chunk indices, and payloads
     are deliberately not included.
   - plain chat    → the LLM's reply, verbatim. Used for greetings,
     meta-questions, clarifications, and the case where the LLM
     replies with a question instead of a tool call.

The handler is **stateless**. The original message is dropped after
classification; the next inbound message is classified fresh. There
are no per-chat pending slots, no session memory, and no carryover
between calls.

If the configured inference endpoint does not support tool calls (or
any other API error happens), :func:`classify_and_route` returns
``("chat", "<error string>")`` and the handler returns that string to
the user. The handler itself does not raise for routing failures.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .inference import ask, classify_and_route
from .prompts import SYSTEM_PROMPT, TOOLS
from .qdrant_store import ingest_file, ingest_text

SUPPORTED_ATTACHMENT_SUFFIXES = frozenset({'.pdf', '.txt', '.md', '.text'})

TEXT_PREFIX_LEN = 40
SOURCE_HASH_LEN = 12
SOURCE_NAMESPACE = 'auto'


@dataclass(frozen=True)
class Attachment:
    """A single file attached to an agent message.

    Attributes:
        filename: Original filename (e.g. ``"notes.pdf"``). Used to
            detect the file type and, for the auto-store step, as the
            default ``source`` passed to :func:`rag_qdrant.ingest_file`.
        content: Raw file bytes.
    """

    filename: str
    content: bytes


@dataclass(frozen=True)
class AgentMessage:
    """A single inbound message from an agent transport.

    Attributes:
        text: Plain-text body. May be empty when only an attachment is
            present.
        attachment: Optional attached file.
    """

    text: str
    attachment: Attachment | None = None


def _default_text_source(text: str) -> str:
    """Build a stable default source name for text ingest.

    Returns ``f"{SOURCE_NAMESPACE}-{sha1[:SOURCE_HASH_LEN]}"`` where the
    hash input is the first ``TEXT_PREFIX_LEN`` characters of ``text``.
    Falls back to hashing the current UTC timestamp (ISO 8601, seconds)
    when ``text`` is empty or whitespace-only, so the source is still
    unique per call.
    """
    prefix = (text or '').strip()[:TEXT_PREFIX_LEN]
    if prefix:
        seed = prefix.encode('utf-8')
    else:
        seed = datetime.now(timezone.utc).isoformat(timespec='seconds').encode('utf-8')
    digest = hashlib.sha1(seed).hexdigest()[:SOURCE_HASH_LEN]
    return f'{SOURCE_NAMESPACE}-{digest}'


def _save_and_ingest_attachment(attachment: Attachment) -> tuple[int, str]:
    """Write ``attachment`` to a temp file, ingest it, clean up.

    Returns ``(chunk_count, source)`` where ``source`` is the original
    filename. Raises :class:`ValueError` when the file suffix is not
    in :data:`SUPPORTED_ATTACHMENT_SUFFIXES`.
    """
    suffix = Path(attachment.filename).suffix.lower()
    if suffix not in SUPPORTED_ATTACHMENT_SUFFIXES:
        raise ValueError(
            f'Unsupported attachment type: {suffix or "<no suffix>"}. '
            f'Send one of {sorted(SUPPORTED_ATTACHMENT_SUFFIXES)}.'
        )

    source = attachment.filename
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix='rag_agent_'
        ) as tmp:
            tmp.write(attachment.content)
            tmp_path = Path(tmp.name)
        count = ingest_file(tmp_path, source=source)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return count, source


def handle_message(message: AgentMessage) -> str:
    """Dispatch one :class:`AgentMessage` via the LLM-routed agent flow.

    Returns the user-facing reply string. Never raises for routing
    decisions or for missing tool support from the inference endpoint
    (in those cases the LLM is asked to fall back to a chat reply, and
    a clear error string is returned). The only :class:`ValueError`
    that can escape is the attachment-suffix check inside
    :func:`_save_and_ingest_attachment`.

    The ``ask_corpus`` branch returns **only** ``result["answer"]``
    from :func:`rag_qdrant.ask`. Score, source, chunk_index, payload,
    and the contexts list are deliberately not included in the reply.
    """
    attachment_notice = ''
    if message.attachment is not None:
        count, source = _save_and_ingest_attachment(message.attachment)
        attachment_notice = f'Ingested {count} chunks from {source}'

    body = (message.text or '').strip()
    if not body:
        return attachment_notice

    llm_user_text = f"{attachment_notice}\n\n{body}" if attachment_notice else body
    action, payload = classify_and_route(
        llm_user_text,
        attachment_notice='',
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
    )

    if action == 'store_text':
        try:
            parsed = json.loads(payload)
            text = parsed.get('text') or ''
            explicit_source = (parsed.get('source') or '').strip()
        except (TypeError, ValueError):
            return 'Error: malformed store_text payload from the routing LLM.'
        if not text:
            return 'Error: store_text was called with empty text.'
        source = explicit_source or _default_text_source(text)
        count = ingest_text(text, source=source)
        return f'Ingested {count} chunks from {source}'

    if action == 'ask_corpus':
        question = (payload or '').strip()
        if not question:
            return 'Error: ask_corpus was called with an empty question.'
        result = ask(question)
        return result['answer']

    return payload or ''


__all__ = [
    'AgentMessage',
    'Attachment',
    'SUPPORTED_ATTACHMENT_SUFFIXES',
    'handle_message',
]
