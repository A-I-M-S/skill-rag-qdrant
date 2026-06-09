"""Agent-mode message handler.

Adapts the rag-qdrant skill to a chat-style transport (Telegram, webhook,
REPL, openclaw agent, etc.) by handing every inbound
:class:`AgentMessage` to the configured inference model and letting the
LLM decide what to do. The handler is pure library code: it does not
import any transport package, does not perform network I/O of its own,
and does not touch ``.env`` / config. The agent layer is responsible
for turning inbound traffic into an :class:`AgentMessage` and for
turning the returned :class:`AgentReply` back into a transport-level
response (e.g. sending the photo bytes to the user).

Flow (executed in order):

1. If the message carries any supported attachments
   (``.pdf`` / ``.txt`` / ``.md`` / ``.text``), the handler ingests
   each one unconditionally and collects one ``Ingested N chunks from
   <source>`` notice line per file. The LLM cannot veto an attachment
   — once sent, it's stored. An unsupported attachment suffix raises
   :class:`ValueError`.

2. If the message carries any photos (``Photo``), the handler validates
   each description is non-empty, writes each photo to
   :data:`settings.photos_dir` (content-addressed by sha256, deduped on
   disk), then ingests each description as a chunk with the
   photo-specific Qdrant payload (``kind="photo"``, ``photo_path``,
   ``photo_filename``, ``file_type``, ``sha256``). Each photo gets its
   own ``Ingested 1 chunk from photo-<hash> (<filename>)`` notice line.
   An unsupported photo suffix (when one is present) raises
   :class:`ValueError`. Empty / whitespace descriptions raise
   :class:`ValueError` before any disk or Qdrant work.

3. If after steps 1 and 2 there is no non-empty text, the handler
   returns an :class:`AgentReply` whose ``text`` is the combined
   multi-line notice and whose ``photo_paths`` lists every just-saved
   photo path. No LLM call is made.

4. Otherwise the handler calls
   :func:`rag_qdrant.inference.classify_and_route` with the system
   prompt, the two tool schemas, and the user text (with the combined
   ingest notice prepended when present). The LLM is the sole
   decision-maker. The handler then dispatches on the LLM's choice:

   - ``store_text`` → :func:`rag_qdrant.ingest_text` with a default
     ``auto-<sha1[:12]>`` source (or the explicit ``source`` the LLM
     passed). Reply: ``AgentReply(text=f"Ingested {count} chunks from
     {source}", photo_paths=())``.
   - ``ask_corpus`` → :func:`rag_qdrant.ask` (Qdrant search + grounded
     LLM call). Reply: ``AgentReply(text=result["answer"],
     photo_paths=tuple(p["path"] for p in result["photos"]))``. The
     LLM never sees the photo paths; the handler enriches the reply
     from the matched contexts.
   - plain chat → ``AgentReply(text=llm_reply, photo_paths=())``.

The handler is **stateless**. The original message is dropped after
classification; the next inbound message is classified fresh. There
are no per-chat pending slots, no session memory, and no carryover
between calls.

If the configured inference endpoint does not support tool calls (or
any other API error happens), :func:`classify_and_route` returns
``("chat", "<error string>")`` and the handler returns an
``AgentReply(text=error_string, photo_paths=())``. The handler itself
does not raise for routing failures.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .inference import ask, classify_and_route
from .photo_store import Photo, save_photo
from .prompts import SYSTEM_PROMPT, TOOLS
from .qdrant_store import ingest_file, ingest_photo, ingest_text

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
        text: Plain-text body. May be empty when only attachments
            and/or photos are present.
        attachments: Zero or more text-extractable file attachments
            (``.pdf`` / ``.txt`` / ``.md`` / ``.text``). Each one is
            ingested unconditionally before any LLM call.
        photos: Zero or more photos with required descriptions. Each
            photo's bytes are saved to
            :data:`rag_qdrant.config.settings.photos_dir` and its
            description is embedded in the corpus.
    """

    text: str
    attachments: tuple[Attachment, ...] = ()
    photos: tuple[Photo, ...] = ()


@dataclass(frozen=True)
class AgentReply:
    """The handler's user-facing reply.

    Attributes:
        text: The reply string the transport should display (the
            LLM's answer, a clarification, the multi-line ingest
            notice, or an error string).
        photo_paths: Absolute paths to photos that should accompany
            the reply. Populated in two situations:

            - When the handler just saved one or more photos (no
              text in the inbound turn, or the LLM didn't
              ``ask_corpus`` about them), the just-saved paths are
              listed so the transport can confirm the save.
            - When the LLM routed to ``ask_corpus`` and the matched
              contexts include photo points, the matched photo paths
              are listed so the transport can surface them as part
              of the answer.

            Empty tuple when no photos apply. The LLM never sees
            these paths; the handler enriches the reply from the
            matched contexts.
    """

    text: str
    photo_paths: tuple[str, ...] = ()


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


def _ingest_attachments(attachments: tuple[Attachment, ...]) -> list[str]:
    """Ingest every attachment; return one notice line per file."""
    lines: list[str] = []
    for att in attachments:
        count, source = _save_and_ingest_attachment(att)
        lines.append(f'Ingested {count} chunks from {source}')
    return lines


def _ingest_photos(photos: tuple[Photo, ...]) -> tuple[list[str], list[str]]:
    """Save and ingest every photo. Returns ``(notice_lines, saved_paths)``."""
    notice_lines: list[str] = []
    saved_paths: list[str] = []
    for photo in photos:
        path, sha256_hex, source = save_photo(photo)
        count = ingest_photo(
            path,
            description=photo.description,
            source=source,
            photo_filename=photo.filename,
            sha256_hex=sha256_hex,
            file_type=Path(photo.filename).suffix.lower(),
        )
        notice_lines.append(f'Ingested {count} chunk from {source} ({photo.filename})')
        saved_paths.append(str(path))
    return notice_lines, saved_paths


def _combine_notice(attachment_lines: list[str], photo_lines: list[str]) -> str:
    """Join the per-file notice lines into a single multi-line notice."""
    return "\n".join(attachment_lines + photo_lines)


def handle_message(message: AgentMessage) -> AgentReply:
    """Dispatch one :class:`AgentMessage` via the LLM-routed agent flow.

    Returns an :class:`AgentReply` (never raises for routing decisions
    or for missing tool support from the inference endpoint). The only
    :class:`ValueError` that can escape is the attachment-suffix check
    inside :func:`_save_and_ingest_attachment` or the photo
    validation inside :func:`save_photo` (empty description, bad
    suffix).

    The ``ask_corpus`` branch returns ``AgentReply(text=answer,
    photo_paths=tuple(p["path"] for p in result["photos"]))``. Score,
    source, chunk_index, payload, and the contexts list are
    deliberately not included in the reply text — only the answer
    string and the matched photo paths.
    """
    attachment_notice_lines: list[str] = _ingest_attachments(message.attachments)
    photo_notice_lines, saved_photo_paths = _ingest_photos(message.photos)
    ingest_notice = _combine_notice(attachment_notice_lines, photo_notice_lines)

    body = (message.text or '').strip()
    if not body:
        return AgentReply(text=ingest_notice, photo_paths=tuple(saved_photo_paths))

    llm_user_text = f"{ingest_notice}\n\n{body}" if ingest_notice else body
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
            return AgentReply(
                text='Error: malformed store_text payload from the routing LLM.',
                photo_paths=(),
            )
        if not text:
            return AgentReply(
                text='Error: store_text was called with empty text.',
                photo_paths=(),
            )
        source = explicit_source or _default_text_source(text)
        count = ingest_text(text, source=source)
        return AgentReply(
            text=f'Ingested {count} chunks from {source}',
            photo_paths=(),
        )

    if action == 'ask_corpus':
        question = (payload or '').strip()
        if not question:
            return AgentReply(
                text='Error: ask_corpus was called with an empty question.',
                photo_paths=(),
            )
        result = ask(question)
        matched = [p.get('path') for p in result.get('photos', []) if p.get('path')]
        return AgentReply(
            text=result['answer'],
            photo_paths=tuple(matched),
        )

    return AgentReply(text=payload or '', photo_paths=())


__all__ = [
    'AgentMessage',
    'AgentReply',
    'Attachment',
    'Photo',
    'SUPPORTED_ATTACHMENT_SUFFIXES',
    'handle_message',
]
