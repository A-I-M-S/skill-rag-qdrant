"""Agent-mode message handler.

Adapts the rag-qdrant skill to a chat-style transport (Telegram, webhook,
REPL, etc.) by routing a single inbound :class:`AgentMessage` to one of
the existing flat functions (:func:`rag_qdrant.ingest_text`,
:func:`rag_qdrant.ingest_file`, :func:`rag_qdrant.ask`) and producing a
short user-facing reply string.

The module is pure library code. It does not import any transport
package (no chat-transport dependency), does not perform network I/O, and
does not touch ``.env`` / config. The agent layer is responsible for
turning inbound traffic into an :class:`AgentMessage` and for sending the
reply back to the user.

Rules (executed in order):

1. ``message.text`` starts with "Embed" (case-insensitive) and
   ``message.attachment`` is a supported PDF/TXT/MD file -> save the
   attachment to a temp path (deleted in a ``finally``), call
   :func:`rag_qdrant.ingest_file` with ``source=attachment.filename``,
   reply ``f"Ingested {n} chunks from {source}"``.

2. ``message.text`` starts with "Embed" (case-insensitive) and has
   non-empty body text after the prefix -> call
   :func:`rag_qdrant.ingest_text` with
   ``source=_default_text_source(body)``, reply with the same ack
   format.

3. ``message.text`` starts with "Query" (case-insensitive) and has
   non-empty body text -> call :func:`rag_qdrant.ask` and return
   **only** ``result["answer"]``. Score, source, chunk_index, payload,
   and the contexts list are deliberately dropped. When the semantic
   cache is enabled (``SEMANTIC_CACHE_ENABLED=1``), the answer may
   come from the cache; the contract (only the answer string) is
   unchanged.

4. "Embed" with no text and no attachment, or "Query" with no body,
   raises :class:`ValueError`. The handler does not produce a graceful
   reply for these cases.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .inference import ask
from .qdrant_store import ingest_file, ingest_text

EMBED_COMMAND = 'embed'
QUERY_COMMAND = 'query'

SUPPORTED_ATTACHMENT_SUFFIXES = frozenset({'.pdf', '.txt', '.md', '.text'})

TEXT_PREFIX_LEN = 40
SOURCE_HASH_LEN = 12
SOURCE_NAMESPACE = 'telegram'

_EMBED_RE = re.compile(r'^\s*embed\b', re.IGNORECASE)
_QUERY_RE = re.compile(r'^\s*query\b', re.IGNORECASE)
_STRIP_EMBED_RE = re.compile(r'^\s*embed\b\s*', re.IGNORECASE)
_STRIP_QUERY_RE = re.compile(r'^\s*query\b\s*', re.IGNORECASE)


@dataclass(frozen=True)
class Attachment:
    """A single file attached to an agent message.

    Attributes:
        filename: Original filename (e.g. ``"notes.pdf"``). Used to detect
            the file type and, for the "Embed" + attachment rule, as the
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


def _is_embed(message: AgentMessage) -> bool:
    return bool(_EMBED_RE.match(message.text or ''))


def _is_query(message: AgentMessage) -> bool:
    return bool(_QUERY_RE.match(message.text or ''))


def _strip_command(text: str, command: str) -> str:
    """Drop the command word + following whitespace from the start of ``text``."""
    if command == EMBED_COMMAND:
        return _STRIP_EMBED_RE.sub('', text or '').strip()
    if command == QUERY_COMMAND:
        return _STRIP_QUERY_RE.sub('', text or '').strip()
    raise ValueError(f'Unknown command: {command!r}')


def _handle_embed(message: AgentMessage) -> str:
    if message.attachment is not None:
        suffix = Path(message.attachment.filename).suffix.lower()
        if suffix in SUPPORTED_ATTACHMENT_SUFFIXES:
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix='rag_embed_'
                ) as tmp:
                    tmp.write(message.attachment.content)
                    tmp_path = Path(tmp.name)
                source = message.attachment.filename
                count = ingest_file(tmp_path, source=source)
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
            return f'Ingested {count} chunks from {source}'

    body = _strip_command(message.text or '', EMBED_COMMAND)
    if not body:
        raise ValueError(
            'Embed command requires either non-empty text after the '
            'prefix or a supported PDF/TXT/MD attachment.'
        )
    source = _default_text_source(body)
    count = ingest_text(body, source=source)
    return f'Ingested {count} chunks from {source}'


def _handle_query(message: AgentMessage) -> str:
    body = _strip_command(message.text or '', QUERY_COMMAND)
    if not body:
        raise ValueError("Query command requires a non-empty question after the prefix.")
    result = ask(body)
    return result['answer']


def handle_message(message: AgentMessage) -> str:
    """Dispatch one :class:`AgentMessage` to the right skill action.

    Returns the user-facing reply string. Raises :class:`ValueError` when
    the message looks like a command but is missing required content
    (Embed with no text and no attachment, or Query with no body).

    The Query branch returns **only** ``result["answer"]`` from
    :func:`rag_qdrant.ask`. Score, source, chunk_index, payload, and the
    contexts list are deliberately not included in the reply.
    """
    if _is_embed(message):
        return _handle_embed(message)
    if _is_query(message):
        return _handle_query(message)
    raise ValueError(
        'Unknown command. Send "Embed <text>" or "Query <question>", '
        'or attach a PDF/TXT/MD file with an "Embed" caption.'
    )
