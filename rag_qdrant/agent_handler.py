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

from .config import is_admin
from .inference import ask, classify_and_route
from .photo_store import Photo, save_photo
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_ADMIN, TOOLS, TOOLS_WITH_ADMIN
from .qdrant_store import (
    grant_access as qdrant_grant_access,
    ingest_file,
    ingest_photo,
    ingest_text,
    revoke_access as qdrant_revoke_access,
    show_access as qdrant_show_access,
)
from .user_cache import record_seen, resolve_id as user_cache_resolve_id
from .user_cache import resolve_username as user_cache_resolve_username

SUPPORTED_ATTACHMENT_SUFFIXES = frozenset({'.pdf', '.txt', '.md', '.text'})

TEXT_PREFIX_LEN = 40
SOURCE_HASH_LEN = 12
SOURCE_NAMESPACE = 'auto'

ADMIN_ONLY_MESSAGE = (
    "This action is only available to admins. Ask an admin to grant you access "
    "or perform the action for you."
)

TOOL_NAMES_REQUIRING_ADMIN = frozenset({"store_text", "grant_access", "revoke_access", "resolve_username", "show_access"})

USERNAME_NOT_SEEN_MESSAGE = (
    "That user hasn't messaged this bot yet. Ask them to send any message to "
    "the bot first, then try again."
)


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
        current_telegram_id: Optional Telegram user id of the sender.
            When set, the handler uses it to decide admin status (via
            :func:`rag_qdrant.config.is_admin`) and to apply the
            access-control filter to ``ask_corpus`` lookups. ``None``
            is treated as a non-admin public caller.
        current_username: Optional Telegram ``@username`` of the
            sender, with or without a leading ``@``. Recorded in the
            local user cache on every inbound message so admins can
            later reference this user by ``@username`` in chat (e.g.
            "let @alice see the Q3 note"). ``None`` when the user
            has no public Telegram username.
        current_first_name: Optional Telegram display name of the
            sender. Recorded in the user cache for richer replies
            (``resolve_username`` returns it). ``None`` when
            unavailable.
    """

    text: str
    attachments: tuple[Attachment, ...] = ()
    photos: tuple[Photo, ...] = ()
    current_telegram_id: int | str | None = None
    current_username: str | None = None
    current_first_name: str | None = None


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


def _resolve_admin_telegram_id(value: int | str | None) -> int | str | None:
    """Resolve a tool-call ``telegram_id`` argument.

    Numeric / numeric-string values pass through unchanged (Qdrant
    normalizes them). String values that begin with ``@`` (or look
    like a username, i.e. non-numeric) are resolved through the
    local user cache. ``None`` and empty strings return ``None``.

    Returns the numeric id (``int``) on a successful username
    lookup, the original value when no resolution is needed, or
    ``None`` when the username is unknown (the caller turns that
    into a clear error).
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text == "*":
        return text
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    tid = user_cache_resolve_username(text)
    return tid


def _record_inbound_user(message: AgentMessage) -> None:
    """Cache the sender's id / username / first_name from an inbound message.

    Best-effort: any error (e.g. read-only filesystem) is logged at
    WARNING and swallowed. We only call :func:`record_seen` when
    ``current_telegram_id`` is set; usernames without a numeric id
    cannot be reliably resolved later.
    """
    if message.current_telegram_id is None:
        return
    record_seen(
        message.current_telegram_id,
        username=message.current_username,
        first_name=message.current_first_name,
    )


def handle_message(message: AgentMessage) -> AgentReply:
    """Dispatch one :class:`AgentMessage` via the LLM-routed agent flow.

    Returns an :class:`AgentReply` (never raises for routing decisions
    or for missing tool support from the inference endpoint). The only
    :class:`ValueError` that can escape is the attachment-suffix check
    inside :func:`_save_and_ingest_attachment` or the photo
    validation inside :func:`save_photo` (empty description, bad
    suffix).

    Admin gating: when the LLM is called with the standard
    :data:`TOOLS` schema (admins only), the LLM may invoke
    ``store_text``, ``grant_access``, ``revoke_access``, or
    ``show_access``. When the LLM is called with the read-only
    :data:`TOOLS_PUBLIC` schema (everyone), the LLM only sees
    ``ask_corpus``. The gate is enforced server-side: the LLM's tool
    choice is re-checked against the caller's admin status and
    non-admin calls return :data:`ADMIN_ONLY_MESSAGE` without
    touching Qdrant.

    The ``ask_corpus`` branch forwards the caller's
    ``current_telegram_id`` into the search filter so non-admins see
    only the chunks their ACL allows. Admins see every chunk.

    On every inbound message, the handler caches the sender's
    Telegram id / username / first_name in the local user cache
    (``logs/user_cache.json``) so admins can later reference the
    user by ``@username`` in chat. The LLM-facing ``resolve_username``
    tool exposes the same lookup back to admins.
    """
    _record_inbound_user(message)
    attachment_notice_lines: list[str] = _ingest_attachments(message.attachments)
    photo_notice_lines, saved_photo_paths = _ingest_photos(message.photos)
    ingest_notice = _combine_notice(attachment_notice_lines, photo_notice_lines)

    body = (message.text or '').strip()
    if not body:
        return AgentReply(text=ingest_notice, photo_paths=tuple(saved_photo_paths))

    llm_user_text = f"{ingest_notice}\n\n{body}" if ingest_notice else body
    caller_is_admin = is_admin(message.current_telegram_id)
    tool_schema = TOOLS_WITH_ADMIN if caller_is_admin else TOOLS
    system_prompt = SYSTEM_PROMPT_WITH_ADMIN if caller_is_admin else SYSTEM_PROMPT

    action, payload = classify_and_route(
        llm_user_text,
        attachment_notice='',
        system_prompt=system_prompt,
        tools=tool_schema,
    )

    if action in TOOL_NAMES_REQUIRING_ADMIN:
        if not caller_is_admin:
            return AgentReply(text=ADMIN_ONLY_MESSAGE, photo_paths=())
    else:
        # Public tools (ask_corpus) and the chat path: still permitted.
        pass

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

    if action == 'grant_access':
        try:
            parsed = json.loads(payload)
            source = (parsed.get('source') or '').strip()
            tid = parsed.get('telegram_id')
        except (TypeError, ValueError):
            return AgentReply(
                text='Error: malformed grant_access payload from the routing LLM.',
                photo_paths=(),
            )
        if not source or tid is None:
            return AgentReply(
                text='Error: grant_access requires non-empty source and telegram_id.',
                photo_paths=(),
            )
        resolved = _resolve_admin_telegram_id(tid)
        if resolved is None:
            return AgentReply(text=USERNAME_NOT_SEEN_MESSAGE, photo_paths=())
        result = qdrant_grant_access(source, resolved)
        return AgentReply(
            text=f'Granted access to {result["telegram_id"]} for source {result["source"]} '
                 f'({result["updated"]} chunk(s) updated).',
            photo_paths=(),
        )

    if action == 'revoke_access':
        try:
            parsed = json.loads(payload)
            source = (parsed.get('source') or '').strip()
            tid = parsed.get('telegram_id')
        except (TypeError, ValueError):
            return AgentReply(
                text='Error: malformed revoke_access payload from the routing LLM.',
                photo_paths=(),
            )
        if not source or tid is None:
            return AgentReply(
                text='Error: revoke_access requires non-empty source and telegram_id.',
                photo_paths=(),
            )
        resolved = _resolve_admin_telegram_id(tid)
        if resolved is None:
            return AgentReply(text=USERNAME_NOT_SEEN_MESSAGE, photo_paths=())
        result = qdrant_revoke_access(source, resolved)
        return AgentReply(
            text=f'Revoked access for {result["telegram_id"]} from source {result["source"]} '
                 f'({result["removed"]} chunk(s) updated).',
            photo_paths=(),
        )

    if action == 'resolve_username':
        try:
            parsed = json.loads(payload)
            username = (parsed.get('username') or '').strip()
        except (TypeError, ValueError):
            return AgentReply(
                text='Error: malformed resolve_username payload from the routing LLM.',
                photo_paths=(),
            )
        if not username:
            return AgentReply(
                text='Error: resolve_username requires a non-empty username.',
                photo_paths=(),
            )
        tid = user_cache_resolve_username(username)
        if tid is None:
            return AgentReply(text=USERNAME_NOT_SEEN_MESSAGE, photo_paths=())
        meta = user_cache_resolve_id(tid) or {}
        first = (meta.get('first_name') or '').strip()
        if first:
            return AgentReply(
                text=f'@{meta.get("username") or username.lstrip("@")} '
                     f'(Telegram id {tid}, "{first}").',
                photo_paths=(),
            )
        return AgentReply(
            text=f'@{meta.get("username") or username.lstrip("@")} '
                 f'(Telegram id {tid}).',
            photo_paths=(),
        )

    if action == 'show_access':
        try:
            parsed = json.loads(payload)
            source = (parsed.get('source') or '').strip()
        except (TypeError, ValueError):
            return AgentReply(
                text='Error: malformed show_access payload from the routing LLM.',
                photo_paths=(),
            )
        if not source:
            return AgentReply(
                text='Error: show_access requires a non-empty source.',
                photo_paths=(),
            )
        result = qdrant_show_access(source)
        if result['chunk_count'] == 0:
            return AgentReply(
                text=f'No chunks found for source {result["source"]}.',
                photo_paths=(),
            )
        ids = result['allowed_telegram_ids'] or []
        if not ids:
            return AgentReply(
                text=f'Source {result["source"]} has {result["chunk_count"]} chunk(s); '
                     f'no Telegram id is granted access (admin-only).',
                photo_paths=(),
            )
        return AgentReply(
            text=f'Source {result["source"]} has {result["chunk_count"]} chunk(s); '
                 f'allowed Telegram ids: {", ".join(ids)}.',
            photo_paths=(),
        )

    if action == 'ask_corpus':
        question = (payload or '').strip()
        if not question:
            return AgentReply(
                text='Error: ask_corpus was called with an empty question.',
                photo_paths=(),
            )
        result = ask(question, current_telegram_id=message.current_telegram_id)
        matched = [p.get('path') for p in result.get('photos', []) if p.get('path')]
        answer = result.get('answer') or ''
        # Override the LLM-facing "No relevant information found" with the
        # grumpy voice: never explain the filter, never leak that a
        # restricted chunk exists — both ACL-filtered and genuinely absent
        # surface as the same curt line.
        from .inference import NO_RELEVANT_ANSWER
        if answer.strip() == NO_RELEVANT_ANSWER:
            return AgentReply(text='Nothing on that.', photo_paths=())
        return AgentReply(text=answer, photo_paths=tuple(matched))

    return AgentReply(text=payload or '', photo_paths=())


__all__ = [
    'ADMIN_ONLY_MESSAGE',
    'AgentMessage',
    'AgentReply',
    'Attachment',
    'Photo',
    'SUPPORTED_ATTACHMENT_SUFFIXES',
    'USERNAME_NOT_SEEN_MESSAGE',
    'handle_message',
    'is_admin',
]
