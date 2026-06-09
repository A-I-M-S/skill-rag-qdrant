"""Self-contained tests for ``rag_qdrant.agent_handler`` and the
``classify_and_route`` wrapper in ``rag_qdrant.inference``.

Usage: ``python3 tests/test_agent_handler.py``

Mirrors the offline-stubbing style of ``tests/run_tests.py`` so the
file runs without Qdrant / FastEmbed / OpenAI / pypdf / dotenv.

Covers (all behavioral — the old "Embed/Query prefix" rules are gone):

- LLM routes to ``store_text`` (default source) → ``ingest_text`` is
  called with ``auto-<sha1[:12]>`` and the ack format is preserved.
- LLM routes to ``store_text`` with an explicit ``source`` → handler
  uses that source verbatim.
- LLM routes to ``ask_corpus`` → handler calls ``ask`` and returns
  ONLY ``result["answer"]``; no contexts/score/source/payload leak;
  ``result["photos"]`` paths are propagated into ``AgentReply.photo_paths``.
- LLM returns a plain chat reply → handler returns an ``AgentReply``
  with that string as ``text`` and ``photo_paths=()``.
- LLM replies with a clarification question → handler returns the
  clarification string. Stateless — no per-call carryover.
- ``classify_and_route`` returns a tool-unsupported error string →
  handler returns it in an ``AgentReply`` with empty ``photo_paths``,
  no exception.
- Attachments only (no text) → each file is ingested, the ack lists
  each on its own line, ``photo_paths=()``.
- Attachment + text → file is ingested first, then the LLM is called
  with the multi-line notice + caption in its context. The stubbed
  LLM routes to ``ask_corpus``; the handler returns only the answer.
- Attachment with an unsupported suffix (``.jpg`` is text-unsupported;
  use a real unsupported one like ``.bin``) → ``ValueError``.
- Photo only (no text) → photo is saved to disk and its description
  is embedded; ack lists the photo on its own line;
  ``photo_paths=(abs_path,)``; no LLM call.
- Multiple photos in one message → each is saved and ingested; the
  ack lists each on its own line; ``photo_paths`` has every saved path.
- Multiple attachments in one message → each is ingested; the ack
  lists each on its own line.
- Mixed attachments + photos in one turn → combined multi-line notice
  lists everything; LLM still gets the combined notice + body.
- Photo + caption "what does this look like?" → photo saved, LLM
  routes to ``ask_corpus`` with the photo notice + caption;
  matched photo paths flow into ``AgentReply.photo_paths``.
- Photo ingest dedupes on disk: same bytes + same filename → only
  one file is written, both calls return the same path.
- Photo empty / whitespace description → ``ValueError``.
- Photo unsupported suffix (``.xyz``) → ``ValueError``; missing
  suffix accepted.
- Photo ingest payload shape: stub ``qdrant_store.ingest_photo``,
  capture kwargs, assert ``description`` / ``source`` / ``photo_filename`` /
  ``sha256_hex`` / ``file_type``.
- Cache invalidation on photo ingest: ``ingest_photo`` calls
  ``search_cache_invalidate`` (via the underlying ``ingest_text``).
- Public exports: ``AgentMessage``, ``AgentReply``, ``Attachment``,
  ``Photo``, ``handle_message`` re-exported from the top-level
  ``rag_qdrant`` package and listed in ``__all__``.

Plus unit tests for ``classify_and_route`` itself, stubbing the
``openai.OpenAI`` client:

- Tool call ``store_text`` → returns ``("store_text", <json>)``.
- Tool call ``ask_corpus`` → returns ``("ask_corpus", <question>)``.
- Plain assistant content → returns ``("chat", <content>)``.
- Fake client raises ``openai.BadRequestError`` → returns
  ``("chat", "<error string naming tool support>")``.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Stub foreign deps so the package imports offline (same trick as run_tests.py)
# ---------------------------------------------------------------------------

def _ensure_stub(name: str) -> None:
    if name in sys.modules:
        return
    sys.modules[name] = types.ModuleType(name)


for _missing, _attrs in (
    ('dotenv', ('load_dotenv',)),
    ('fastembed', ()),
    ('fastembed.common.model_description', ('ModelSource', 'PoolingType')),
    ('openai', ('OpenAI', 'APIError', 'BadRequestError')),
    ('pypdf', ('PdfReader',)),
    ('qdrant_client', ('QdrantClient',)),
    ('qdrant_client.http', ()),
    ('qdrant_client.http.models', ()),
):
    _ensure_stub(_missing)
    for _attr in _attrs:
        if not hasattr(sys.modules[_missing], _attr):
            if _attr in ('APIError', 'BadRequestError'):
                setattr(sys.modules[_missing], _attr, type(_attr, (Exception,), {}))
            else:
                setattr(sys.modules[_missing], _attr, lambda *a, **k: None)

sys.modules['qdrant_client.http.models'].PayloadSchemaType = types.SimpleNamespace(
    KEYWORD='keyword', INTEGER='integer'
)
sys.modules['qdrant_client.http.models'].VectorParams = lambda **kw: ('VectorParams', kw)
sys.modules['qdrant_client.http.models'].Distance = types.SimpleNamespace(COSINE='Cosine')
sys.modules['qdrant_client.http.models'].PointStruct = lambda **kw: ('PointStruct', kw)
sys.modules['qdrant_client'].QdrantClient = type('QdrantClient', (), {})
sys.modules['fastembed'].TextEmbedding = type(
    'TextEmbedding',
    (),
    {
        'list_supported_models': staticmethod(lambda: []),
        'add_custom_model': staticmethod(lambda **kw: None),
    },
)
sys.modules['fastembed.common.model_description'].ModelSource = lambda **kw: ('ModelSource', kw)
sys.modules['fastembed.common.model_description'].PoolingType = types.SimpleNamespace(MEAN='mean')
sys.modules['pypdf'].PdfReader = type('PdfReader', (), {})
sys.modules['openai'].OpenAI = type('OpenAI', (), {})

# Re-import to make sure the package picks up the stubs cleanly.
importlib.invalidate_caches()
if 'rag_qdrant' in sys.modules:
    importlib.reload(sys.modules['rag_qdrant'])

from rag_qdrant import (  # noqa: E402
    AgentMessage,
    AgentReply,
    Attachment,
    Photo,
    handle_message,
)
import rag_qdrant.agent_handler as handler_module  # noqa: E402
import rag_qdrant.inference as inference_module  # noqa: E402
import rag_qdrant.photo_store as photo_store_module  # noqa: E402
from rag_qdrant.agent_handler import _default_text_source  # noqa: E402
from rag_qdrant.config import Settings  # noqa: E402
from rag_qdrant.prompts import SYSTEM_PROMPT, TOOLS  # noqa: E402

passed: list[str] = []
failed: list[tuple[str, str]] = []


def expect(label: str, condition: bool, detail: str = '') -> None:
    if condition:
        passed.append(label)
        print(f'PASS {label}')
    else:
        failed.append((label, detail))
        print(f'FAIL {label}  {detail}')


# ---------------------------------------------------------------------------
# Helpers: stub classify_and_route / settings.photos_dir / qdrant_store
# ---------------------------------------------------------------------------

def _stub_classify(return_value):
    return patch.object(handler_module, 'classify_and_route', return_value=return_value)


def _force_photos_dir(tmp: Path):
    """Patch the settings singleton's photos_dir to ``tmp`` and propagate
    the patched settings to every module that imported it by name."""
    s = Settings(
        qdrant_url='x', qdrant_api_key='y',
        inference_base_url='https://example.com/v1',
        inference_api_key='k', inference_model='m',
        photos_dir=tmp,
    )
    import rag_qdrant.config as _cfg
    import rag_qdrant as _pkg
    _cfg.settings = s
    for mod in (handler_module, photo_store_module, inference_module, _pkg):
        if hasattr(mod, 'settings'):
            mod.settings = s
    return s


# ---------------------------------------------------------------------------
# Handler behavioral tests
# ---------------------------------------------------------------------------

def run_handler_behavioral_tests() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _force_photos_dir(tmp_path)

        # 1) LLM routes to store_text, no explicit source → auto-<hash>
        with _stub_classify(('store_text', json.dumps({'text': 'The cat sat on the mat.', 'source': ''}))) as m_classify, \
             patch.object(handler_module, 'ingest_text', return_value=3) as m_text, \
             patch.object(handler_module, 'ingest_file') as m_file, \
             patch.object(handler_module, 'ingest_photo') as m_photo, \
             patch.object(handler_module, 'ask') as m_ask:
            reply = handle_message(AgentMessage(text='save this for later'))
        expect('agent_reply_is_dataclass_store_text', isinstance(reply, AgentReply))
        expect('handler_store_text_calls_classify', m_classify.called)
        expect('handler_store_text_calls_ingest_text', m_text.called)
        expect('handler_store_text_does_not_call_ingest_file', not m_file.called)
        expect('handler_store_text_does_not_call_ask', not m_ask.called)
        expect('handler_store_text_does_not_call_ingest_photo', not m_photo.called)
        expect('handler_store_text_reply_starts_with_ingested', reply.text.startswith('Ingested 3 chunks from '))
        expect('handler_store_text_reply_uses_auto_namespace', ' auto-' in reply.text)
        expect('handler_store_text_photo_paths_empty', reply.photo_paths == ())
        suffix = reply.text.split(' from ', 1)[1]
        expect('handler_store_text_auto_source_has_12_hex', len(suffix) == len('auto-') + 12 and suffix.startswith('auto-'))
        _, kwargs = m_text.call_args
        expect('handler_store_text_source_kwarg_is_auto', kwargs.get('source', '').startswith('auto-'))

        # 2) LLM routes to store_text with an explicit source → handler uses it
        with _stub_classify(('store_text', json.dumps({'text': 'meeting notes', 'source': 'meeting-2026-06-05'}))) as m_classify, \
             patch.object(handler_module, 'ingest_text', return_value=1) as m_text:
            reply = handle_message(AgentMessage(text='remember these notes'))
        expect('handler_store_text_explicit_source_reply', reply.text == 'Ingested 1 chunks from meeting-2026-06-05')
        expect('handler_store_text_explicit_source_kwarg', m_text.call_args[1].get('source') == 'meeting-2026-06-05')
        expect('handler_store_text_explicit_source_photo_paths_empty', reply.photo_paths == ())

        # 3) LLM routes to ask_corpus → handler returns ONLY result["answer"]; photos propagated
        ask_result = {
            'answer': 'the cat sat on the mat',
            'contexts': [
                {
                    'score': 0.91,
                    'source': 'meeting-2026-06-05',
                    'chunk_index': 3,
                    'text': 'the cat sat',
                    'payload': {'leak': 'do-not-leak'},
                }
            ],
            'photos': [],
        }
        with _stub_classify(('ask_corpus', 'Where did the cat sit?')) as m_classify, \
             patch.object(handler_module, 'ingest_text') as m_text, \
             patch.object(handler_module, 'ingest_file') as m_file, \
             patch.object(handler_module, 'ingest_photo') as m_photo, \
             patch.object(handler_module, 'ask', return_value=ask_result) as m_ask:
            reply = handle_message(AgentMessage(text='where did the cat sit?'))
        expect('handler_ask_corpus_calls_classify', m_classify.called)
        expect('handler_ask_corpus_calls_ask', m_ask.called)
        expect('handler_ask_corpus_does_not_call_ingest', not (m_text.called or m_file.called or m_photo.called))
        expect('handler_ask_corpus_answer_only', reply.text == 'the cat sat on the mat')
        for forbidden in ('score', 'source', 'chunk_index', 'payload', 'contexts', 'meeting-2026-06-05', 'do-not-leak'):
            expect(f'handler_ask_corpus_no_{forbidden}_in_reply', forbidden not in reply.text)
        expect('handler_ask_corpus_photo_paths_empty_when_no_photos', reply.photo_paths == ())

        # 4) LLM returns a plain chat reply → verbatim
        with _stub_classify(('chat', "Hey, what's up?")) as m_classify, \
             patch.object(handler_module, 'ingest_text') as m_text, \
             patch.object(handler_module, 'ingest_file') as m_file, \
             patch.object(handler_module, 'ingest_photo') as m_photo, \
             patch.object(handler_module, 'ask') as m_ask:
            reply = handle_message(AgentMessage(text='hello there'))
        expect('handler_chat_calls_classify', m_classify.called)
        expect('handler_chat_does_not_call_actions', not (m_text.called or m_file.called or m_ask.called or m_photo.called))
        expect('handler_chat_reply_verbatim', reply.text == "Hey, what's up?")
        expect('handler_chat_photo_paths_empty', reply.photo_paths == ())

        # 5) LLM clarifies with a question → handler returns the clarification string
        clarification = 'Do you want me to store that, or search the corpus for related notes?'
        with _stub_classify(('chat', clarification)) as m_classify, \
             patch.object(handler_module, 'ingest_text') as m_text, \
             patch.object(handler_module, 'ingest_file') as m_file, \
             patch.object(handler_module, 'ingest_photo') as m_photo, \
             patch.object(handler_module, 'ask') as m_ask:
            reply = handle_message(AgentMessage(text='something ambiguous'))
        expect('handler_clarify_reply_verbatim', reply.text == clarification)
        expect('handler_clarify_does_not_call_actions', not (m_text.called or m_file.called or m_ask.called or m_photo.called))
        with _stub_classify(('chat', 'fresh classification')) as m_classify2:
            reply2 = handle_message(AgentMessage(text='different message'))
        expect('handler_stateless_second_call', reply2.text == 'fresh classification' and m_classify2.called)

        # 6) classify_and_route returns a tool-unsupported error string → AgentReply
        err = 'Error: the configured inference endpoint rejected tool calls. The agent handler requires tool support.'
        with _stub_classify(('chat', err)) as m_classify, \
             patch.object(handler_module, 'ingest_text') as m_text, \
             patch.object(handler_module, 'ingest_file') as m_file, \
             patch.object(handler_module, 'ingest_photo') as m_photo, \
             patch.object(handler_module, 'ask') as m_ask:
            try:
                reply = handle_message(AgentMessage(text='hello'))
                expect('handler_tool_unsupported_no_exception', True)
            except Exception as exc:
                expect('handler_tool_unsupported_no_exception', False, f'raised {exc!r}')
                reply = AgentReply(text='', photo_paths=())
        expect('handler_tool_unsupported_returns_string', reply.text == err)
        expect('handler_tool_unsupported_photo_paths_empty', reply.photo_paths == ())
        expect('handler_tool_unsupported_does_not_call_actions', not (m_text.called or m_file.called or m_ask.called or m_photo.called))

        # 7) Attachment only (no text) → each ingested, ack per line, photo_paths empty
        with _stub_classify(('chat', 'should not be called')) as m_classify, \
             patch.object(handler_module, 'ingest_text') as m_text, \
             patch.object(handler_module, 'ingest_photo') as m_photo, \
             patch.object(handler_module, 'ingest_file', return_value=7) as m_file, \
             patch.object(handler_module, 'ask') as m_ask:
            reply = handle_message(
                AgentMessage(text='', attachments=(Attachment('notes.txt', b'hello world'),))
            )
        expect('handler_attachment_only_calls_ingest_file', m_file.called)
        expect('handler_attachment_only_does_not_call_classify', not m_classify.called)
        expect('handler_attachment_only_does_not_call_ingest_text', not m_text.called)
        expect('handler_attachment_only_does_not_call_ask', not m_ask.called)
        expect('handler_attachment_only_does_not_call_ingest_photo', not m_photo.called)
        expect('handler_attachment_only_ack_format', reply.text == 'Ingested 7 chunks from notes.txt')
        expect('handler_attachment_only_photo_paths_empty', reply.photo_paths == ())

        # 8) Attachment + caption → file ingested first, LLM called with notice + caption,
        #    stubbed LLM routes to ask_corpus, handler returns only the answer.
        captured_user_text = {}

        def _capture_classify(message_text, **_kwargs):
            captured_user_text['value'] = message_text
            return ('ask_corpus', 'summarize this file')

        with patch.object(handler_module, 'classify_and_route', side_effect=_capture_classify), \
             patch.object(handler_module, 'ingest_text') as m_text, \
             patch.object(handler_module, 'ingest_file', return_value=12) as m_file, \
             patch.object(handler_module, 'ingest_photo') as m_photo, \
             patch.object(handler_module, 'ask', return_value={'answer': 'summary answer', 'contexts': [], 'photos': []}) as m_ask:
            reply = handle_message(
                AgentMessage(text='summarize this', attachments=(Attachment('notes.pdf', b'%PDF-1.4 fake'),))
            )
        expect('handler_attachment_plus_text_ingests_file', m_file.called)
        expect('handler_attachment_plus_text_calls_ask', m_ask.called)
        expect('handler_attachment_plus_text_does_not_ingest_text', not m_text.called)
        expect('handler_attachment_plus_text_does_not_ingest_photo', not m_photo.called)
        expect('handler_attachment_plus_text_reply_is_answer_only', reply.text == 'summary answer')
        expect('handler_attachment_plus_text_calls_ask_with_question', m_ask.call_args[0][0] == 'summarize this file')
        expect('handler_attachment_plus_text_llm_sees_notice', 'Ingested 12 chunks from notes.pdf' in captured_user_text['value'])
        expect('handler_attachment_plus_text_llm_sees_caption', 'summarize this' in captured_user_text['value'])
        expect('handler_attachment_plus_text_photo_paths_empty', reply.photo_paths == ())

        # 9) Attachment with unsupported suffix → ValueError
        try:
            handle_message(AgentMessage(text='hi', attachments=(Attachment('photo.bin', b'binary'),)))
            expect('handler_unsupported_attachment_suffix_raises', False, 'expected ValueError')
        except ValueError:
            expect('handler_unsupported_attachment_suffix_raises', True)

        # 10) Empty text + no attachments + no photos → empty reply
        with _stub_classify(('chat', 'nope')) as m_classify, \
             patch.object(handler_module, 'ingest_file') as m_file, \
             patch.object(handler_module, 'ingest_photo') as m_photo:
            reply = handle_message(AgentMessage(text='', attachments=(), photos=()))
        expect('handler_empty_no_attachment_returns_empty', reply.text == '')
        expect('handler_empty_no_attachment_no_classify', not m_classify.called)
        expect('handler_empty_no_attachment_no_ingest_file', not m_file.called)
        expect('handler_empty_no_attachment_no_ingest_photo', not m_photo.called)
        expect('handler_empty_photo_paths_empty', reply.photo_paths == ())

        # 11) Whitespace-only text → empty reply, no LLM call
        with _stub_classify(('chat', 'nope')) as m_classify:
            reply = handle_message(AgentMessage(text='   \n  ', attachments=(), photos=()))
        expect('handler_whitespace_only_returns_empty', reply.text == '')
        expect('handler_whitespace_only_no_classify', not m_classify.called)
        expect('handler_whitespace_only_photo_paths_empty', reply.photo_paths == ())

        # ----- Photo tests -----

        # 12) Photo only (no text) → saved to disk, description embedded, ack single line, no LLM call
        photo_bytes = b'fake-jpg-bytes-sunset'
        photo = Photo('sunset.jpg', photo_bytes, 'a sunset over the bay')
        with _stub_classify(('chat', 'should not be called')) as m_classify, \
             patch.object(handler_module, 'ingest_text') as m_text, \
             patch.object(handler_module, 'ingest_file') as m_file, \
             patch.object(handler_module, 'ingest_photo', return_value=1) as m_photo, \
             patch.object(handler_module, 'ask') as m_ask:
            reply = handle_message(AgentMessage(text='', photos=(photo,)))
        expect('photo_only_calls_ingest_photo', m_photo.called)
        expect('photo_only_does_not_call_classify', not m_classify.called)
        expect('photo_only_does_not_call_ingest_text', not m_text.called)
        expect('photo_only_does_not_call_ingest_file', not m_file.called)
        expect('photo_only_does_not_call_ask', not m_ask.called)
        # File exists on disk under photos_dir, content-addressed
        from hashlib import sha256 as _sha
        expected_stem = _sha(photo_bytes).hexdigest()[:16]
        expected_path = tmp_path / f'{expected_stem}.jpg'
        expect('photo_only_file_written_to_disk', expected_path.exists())
        expect('photo_only_path_resolves_under_photos_dir', expected_path.parent == tmp_path)
        expect('photo_only_ack_starts_with_ingested', reply.text.startswith('Ingested 1 chunk from photo-'))
        expect('photo_only_ack_mentions_filename', 'sunset.jpg' in reply.text)
        expect('photo_only_photo_paths_has_saved_path', expected_path.as_posix() in reply.photo_paths or str(expected_path) in reply.photo_paths)

        # 13) Photo ingest dedupes on disk: same bytes → only one file written, same path returned
        photo_a = Photo('same.jpg', photo_bytes, 'first description')
        photo_b = Photo('same.jpg', photo_bytes, 'second description')
        with patch.object(handler_module, 'ingest_photo', return_value=1):
            reply_a = handle_message(AgentMessage(text='', photos=(photo_a,)))
            reply_b = handle_message(AgentMessage(text='', photos=(photo_b,)))
        # Same content + same filename + same photos_dir → same on-disk path
        expect('photo_dedupe_same_path', reply_a.photo_paths[0] == reply_b.photo_paths[0])
        # File exists exactly once
        matches = list(tmp_path.glob('*.jpg'))
        expect('photo_dedupe_one_file_on_disk', len(matches) == 1)

        # 14) Photo empty description → ValueError
        with patch.object(handler_module, 'ingest_photo') as m_photo:
            try:
                handle_message(AgentMessage(text='', photos=(Photo('a.jpg', b'x', ''),)))
                expect('photo_empty_description_raises', False, 'expected ValueError')
            except ValueError:
                expect('photo_empty_description_raises', True)
        # We never reached the ingest step
        # (the patch is fresh, so m_photo is reset by the with block)

        # 15) Photo whitespace-only description → ValueError
        with patch.object(handler_module, 'ingest_photo') as m_photo:
            try:
                handle_message(AgentMessage(text='', photos=(Photo('a.jpg', b'x', '   \n  '),)))
                expect('photo_whitespace_description_raises', False, 'expected ValueError')
            except ValueError:
                expect('photo_whitespace_description_raises', True)

        # 16) Photo unsupported suffix → ValueError
        with patch.object(handler_module, 'ingest_photo') as m_photo:
            try:
                handle_message(AgentMessage(text='', photos=(Photo('a.xyz', b'x', 'desc'),)))
                expect('photo_unsupported_suffix_raises', False, 'expected ValueError')
            except ValueError:
                expect('photo_unsupported_suffix_raises', True)

        # 17) Photo missing suffix → accepted
        with patch.object(handler_module, 'ingest_photo', return_value=1):
            try:
                reply = handle_message(AgentMessage(text='', photos=(Photo('noext', b'noext-bytes-2', 'a thing'),)))
                expect('photo_missing_suffix_accepted', True)
            except ValueError as exc:
                expect('photo_missing_suffix_accepted', False, f'raised {exc!r}')
                return
        # File written without extension
        from hashlib import sha256 as _sha2
        noext_stem = _sha2(b'noext-bytes-2').hexdigest()[:16]
        expect('photo_missing_suffix_file_no_extension', (tmp_path / noext_stem).exists())

        # 18) Photo ingest payload shape (stub ingest_photo, capture kwargs)
        photo_bytes_18 = b'payload-shape-bytes'
        photo_18 = Photo('cap.jpg', photo_bytes_18, 'a cat on a windowsill')
        from hashlib import sha256 as _sha3
        expected_sha = _sha3(photo_bytes_18).hexdigest()
        captured_18 = {}

        def _capture_ingest_photo(*args, **kwargs):
            captured_18.update(kwargs)
            captured_18.setdefault('args', args)
            return 1

        with patch.object(handler_module, 'ingest_photo', side_effect=_capture_ingest_photo):
            handle_message(AgentMessage(text='', photos=(photo_18,)))
        expect('photo_ingest_payload_description', captured_18.get('description') == 'a cat on a windowsill')
        expect('photo_ingest_payload_source_photo_ns', (captured_18.get('source') or '').startswith('photo-'))
        expect('photo_ingest_payload_source_sha12', len(captured_18.get('source', '')) == len('photo-') + 12)
        expect('photo_ingest_payload_filename', captured_18.get('photo_filename') == 'cap.jpg')
        expect('photo_ingest_payload_sha256_hex_64', len(captured_18.get('sha256_hex') or '') == 64)
        expect('photo_ingest_payload_sha256_full_hex', captured_18.get('sha256_hex') == expected_sha)
        expect('photo_ingest_payload_file_type', captured_18.get('file_type') == '.jpg')

        # 19) Photo + caption → photo saved, LLM routes to ask_corpus, photos propagated
        ask_with_photo = {
            'answer': 'it is a sunset over the bay',
            'contexts': [],
            'photos': [{'path': '/tmp/photo-already-on-disk.jpg', 'filename': 'sunset.jpg', 'source': 'photo-abc', 'score': 0.92}],
        }
        captured_user_text_19 = {}

        def _capture_classify_19(message_text, **_kwargs):
            captured_user_text_19['value'] = message_text
            return ('ask_corpus', 'what does this look like?')

        with patch.object(handler_module, 'classify_and_route', side_effect=_capture_classify_19), \
             patch.object(handler_module, 'ingest_photo', return_value=1) as m_photo, \
             patch.object(handler_module, 'ask', return_value=ask_with_photo) as m_ask:
            reply = handle_message(
                AgentMessage(
                    text='what does this look like?',
                    photos=(Photo('sunset.jpg', b'photo-19-bytes', 'a sunset over the bay'),),
                )
            )
        expect('photo_plus_caption_saves_photo', m_photo.called)
        expect('photo_plus_caption_calls_ask', m_ask.called)
        expect('photo_plus_caption_calls_ask_with_caption', m_ask.call_args[0][0] == 'what does this look like?')
        expect('photo_plus_caption_reply_text', reply.text == 'it is a sunset over the bay')
        expect('photo_plus_caption_photo_paths_propagated', reply.photo_paths == ('/tmp/photo-already-on-disk.jpg',))
        # LLM-bound text contains the photo notice and the caption
        expect('photo_plus_caption_llm_sees_photo_notice', 'photo-' in captured_user_text_19['value'])
        expect('photo_plus_caption_llm_sees_caption', 'what does this look like?' in captured_user_text_19['value'])

        # 20) ask_corpus returns multiple photos → reply.photo_paths lists all, first-seen, deduped
        ask_multi_photo = {
            'answer': 'two photos',
            'contexts': [],
            'photos': [
                {'path': '/p1', 'filename': 'a.jpg', 'source': 'photo-a', 'score': 0.9},
                {'path': '/p2', 'filename': 'b.jpg', 'source': 'photo-b', 'score': 0.8},
            ],
        }
        with _stub_classify(('ask_corpus', 'q')), \
             patch.object(handler_module, 'ask', return_value=ask_multi_photo):
            reply = handle_message(AgentMessage(text='q'))
        expect('ask_corpus_propagates_multiple_photos', reply.photo_paths == ('/p1', '/p2'))
        expect('ask_corpus_preserves_first_seen_order', list(reply.photo_paths) == ['/p1', '/p2'])

        # 21) ask_corpus with no photos → empty photo_paths
        with _stub_classify(('ask_corpus', 'q')), \
             patch.object(handler_module, 'ask', return_value={'answer': 'a', 'contexts': [], 'photos': []}):
            reply = handle_message(AgentMessage(text='q'))
        expect('ask_corpus_no_photos_empty_paths', reply.photo_paths == ())

        # 22) ask_corpus with a duplicate photo path → propagated as-is from result["photos"]
        #     (the dedupe is in extract_photos on the inference side, not the handler).
        ask_dup_photo = {
            'answer': 'a',
            'contexts': [],
            'photos': [
                {'path': '/p1', 'filename': 'a.jpg', 'source': 'photo-a', 'score': 0.9},
                {'path': '/p1', 'filename': 'a.jpg', 'source': 'photo-a', 'score': 0.85},
            ],
        }
        with _stub_classify(('ask_corpus', 'q')), \
             patch.object(handler_module, 'ask', return_value=ask_dup_photo):
            reply = handle_message(AgentMessage(text='q'))
        expect('ask_corpus_propagates_photo_paths_in_order', list(reply.photo_paths) == ['/p1', '/p1'])

        # 23) Multiple photos in one message → each saved, ack multi-line, two paths
        with patch.object(handler_module, 'ingest_photo', return_value=1) as m_photo:
            reply = handle_message(AgentMessage(
                text='',
                photos=(
                    Photo('a.jpg', b'bytes-a', 'first photo'),
                    Photo('b.jpg', b'bytes-b', 'second photo'),
                ),
            ))
        expect('multi_photo_calls_ingest_photo_twice', m_photo.call_count == 2)
        expect('multi_photo_ack_two_lines', '\n' in reply.text and reply.text.count('photo-') >= 2)
        expect('multi_photo_paths_count', len(reply.photo_paths) == 2)
        # All paths are absolute and under photos_dir
        for p in reply.photo_paths:
            expect(f'multi_photo_path_under_photos_dir:{p}', Path(p).is_absolute() and Path(p).parent == tmp_path)

        # 24) Multiple attachments in one message → each ingested, ack multi-line
        with patch.object(handler_module, 'ingest_file', return_value=2) as m_file, \
             patch.object(handler_module, 'ingest_photo') as m_photo:
            reply = handle_message(AgentMessage(
                text='',
                attachments=(
                    Attachment('a.txt', b'hello'),
                    Attachment('b.md', b'world'),
                ),
            ))
        expect('multi_attach_calls_ingest_file_twice', m_file.call_count == 2)
        expect('multi_attach_does_not_ingest_photo', not m_photo.called)
        expect('multi_photo_paths_empty', reply.photo_paths == ())
        expect('multi_attach_ack_two_lines', reply.text.count('Ingested') == 2 and 'a.txt' in reply.text and 'b.md' in reply.text)

        # 25) Mixed attachments + photos in one turn → combined multi-line notice
        with patch.object(handler_module, 'ingest_file', return_value=3) as m_file, \
             patch.object(handler_module, 'ingest_photo', return_value=1) as m_photo, \
             _stub_classify(('chat', 'chat reply')) as m_classify:
            reply = handle_message(AgentMessage(
                text='what do you see?',
                attachments=(Attachment('note.txt', b'txt-bytes'),),
                photos=(Photo('p.jpg', b'photo-bytes', 'a bird in a tree'),),
            ))
        expect('mixed_calls_ingest_file', m_file.called)
        expect('mixed_calls_ingest_photo', m_photo.called)
        expect('mixed_calls_classify', m_classify.called)
        # When the LLM takes the chat branch, photo_paths is empty —
        # the chat branch does not surface the just-saved photos.
        expect('mixed_chat_branch_photo_paths_empty', reply.photo_paths == ())
        expect('mixed_reply_text_is_chat', reply.text == 'chat reply')
        # The LLM saw the combined notice
        captured_25 = {}

        def _capture_classify_25(message_text, **_kwargs):
            captured_25['text'] = message_text
            return ('chat', 'chat reply')

        with patch.object(handler_module, 'classify_and_route', side_effect=_capture_classify_25), \
             patch.object(handler_module, 'ingest_file', return_value=1), \
             patch.object(handler_module, 'ingest_photo', return_value=1):
            handle_message(AgentMessage(
                text='what do you see?',
                attachments=(Attachment('note.txt', b'txt-bytes'),),
                photos=(Photo('p.jpg', b'photo-bytes', 'a bird in a tree'),),
            ))
        expect('mixed_llm_sees_attachment_line', 'note.txt' in captured_25.get('text', ''))
        expect('mixed_llm_sees_photo_line', 'photo-' in captured_25.get('text', ''))
        expect('mixed_llm_sees_caption', 'what do you see?' in captured_25.get('text', ''))

        # 26) Cache invalidation on photo ingest — ingest_photo → ingest_text → search_cache_invalidate
        # We let the real handler_module.ingest_photo run, but stub the
        # downstream Qdrant calls so the test stays offline.
        with patch('rag_qdrant.qdrant_store.embed_texts', return_value=[[0.0] * 384]), \
             patch('rag_qdrant.qdrant_store.ensure_collection', return_value=None), \
             patch('rag_qdrant.qdrant_store.get_qdrant_client') as m_client, \
             patch('rag_qdrant.qdrant_store.search_cache_invalidate') as m_inv:
            fake = types.SimpleNamespace(upsert=lambda **kw: None)
            m_client.return_value = fake
            handle_message(AgentMessage(text='', photos=(Photo('inv.jpg', b'inv-bytes', 'desc'),)))
        expect('photo_ingest_invalidates_search_cache', m_inv.called)

        # 27) Photo ingest creates photos_dir if it doesn't exist
        nested = tmp_path / 'does' / 'not' / 'exist' / 'yet'
        expect('photo_ingest_creates_dir_starts_missing', not nested.exists())
        _force_photos_dir(nested)
        with patch.object(handler_module, 'ingest_photo', return_value=1):
            handle_message(AgentMessage(text='', photos=(Photo('cd.jpg', b'cd-bytes', 'create dir'),)))
        expect('photo_ingest_creates_dir_now_exists', nested.exists())
        # And the file is inside the nested dir
        from hashlib import sha256 as _sha4
        cd_stem = _sha4(b'cd-bytes').hexdigest()[:16]
        expect('photo_ingest_creates_dir_file_inside', (nested / f'{cd_stem}.jpg').exists())

        # 28) Public exports
        from rag_qdrant import AgentMessage as ExportedAgentMessage
        from rag_qdrant import AgentReply as ExportedAgentReply
        from rag_qdrant import Attachment as ExportedAttachment
        from rag_qdrant import Photo as ExportedPhoto
        from rag_qdrant import handle_message as ExportedHandle
        expect('handler_public_export_AgentMessage', ExportedAgentMessage is AgentMessage)
        expect('handler_public_export_AgentReply', ExportedAgentReply is AgentReply)
        expect('handler_public_export_Attachment', ExportedAttachment is Attachment)
        expect('handler_public_export_Photo', ExportedPhoto is Photo)
        expect('handler_public_export_handle_message', ExportedHandle is handle_message)
        import rag_qdrant
        for name in ('AgentMessage', 'AgentReply', 'Attachment', 'Photo', 'handle_message'):
            expect(f'public_dunder_all_has_{name}', name in rag_qdrant.__all__)

        # 29) photo_paths is always a tuple, even when empty
        expect('agent_reply_photo_paths_is_tuple_empty', isinstance(reply.photo_paths, tuple))
        expect('agent_reply_photo_paths_is_tuple_nonempty', isinstance(handle_message(AgentMessage(text='x', attachments=(), photos=())).photo_paths, tuple))

        # 30) prompts module: system prompt + two tool schemas
        expect('prompts_module_SYSTEM_PROMPT_is_string', isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 50)
        expect('prompts_module_TOOLS_has_two_entries', isinstance(TOOLS, list) and len(TOOLS) == 2)
        tool_names = [t['function']['name'] for t in TOOLS]
        expect('prompts_module_TOOLS_contains_store_text', 'store_text' in tool_names)
        expect('prompts_module_TOOLS_contains_ask_corpus', 'ask_corpus' in tool_names)
        store_text = next(t for t in TOOLS if t['function']['name'] == 'store_text')
        expect('prompts_module_store_text_text_required', 'text' in store_text['function']['parameters']['required'])
        ask_corpus = next(t for t in TOOLS if t['function']['name'] == 'ask_corpus')
        expect('prompts_module_ask_corpus_question_required', 'question' in ask_corpus['function']['parameters']['required'])


# ---------------------------------------------------------------------------
# _default_text_source unit tests (now uses the 'auto' namespace)
# ---------------------------------------------------------------------------

def run_default_source_tests() -> None:
    src1 = _default_text_source('hello world')
    src2 = _default_text_source('hello world')
    expect('default_source_is_stable', src1 == src2)
    expect('default_source_namespace_is_auto', src1.startswith('auto-'))
    expect('default_source_len_12', len(src1.split('-', 1)[1]) == 12)

    expect(
        'default_source_different_for_different_text',
        _default_text_source('alpha') != _default_text_source('beta'),
    )

    src_empty = _default_text_source('')
    src_ws = _default_text_source('   \n  ')
    expect('default_source_empty_namespaced', src_empty.startswith('auto-'))
    expect('default_source_whitespace_namespaced', src_ws.startswith('auto-'))
    expect('default_source_empty_len_12', len(src_empty.split('-', 1)[1]) == 12)

    long_text = 'a' * 200
    long_text_trimmed = 'a' * 40
    expect(
        'default_source_uses_prefix_only',
        _default_text_source(long_text) == _default_text_source(long_text_trimmed),
    )


# ---------------------------------------------------------------------------
# classify_and_route wrapper unit tests
# ---------------------------------------------------------------------------

class _BadRequestError(Exception):
    pass


class _APIError(Exception):
    pass


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeMessage:
    def __init__(self, content: str = '', tool_calls=None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeResponse:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    def __init__(self, response_or_exc) -> None:
        self._response_or_exc = response_or_exc

    def create(self, **_kwargs):
        obj = self._response_or_exc
        if isinstance(obj, Exception):
            raise obj
        return obj


class _FakeChat:
    def __init__(self, response_or_exc) -> None:
        self.completions = _FakeCompletions(response_or_exc)


class _FakeOpenAI:
    def __init__(self, response_or_exc) -> None:
        self.chat = _FakeChat(response_or_exc)


def _patch_settings(**overrides):
    s = inference_module.settings
    return patch.object(
        inference_module,
        'settings',
        types.SimpleNamespace(
            inference_api_key=getattr(s, 'inference_api_key', 'k'),
            inference_base_url=getattr(s, 'inference_base_url', 'https://example.com/v1'),
            inference_model=getattr(s, 'inference_model', 'm'),
            inference_temperature=getattr(s, 'inference_temperature', 0.2),
            **overrides,
        ),
    )


def run_classify_and_route_tests() -> None:
    # 1) Tool call store_text
    msg = _FakeMessage(
        tool_calls=[type('T', (), {'function': _FakeFunction('store_text', json.dumps({'text': 'hello', 'source': ''}))})()],
    )
    with patch.object(inference_module, 'OpenAI', return_value=_FakeOpenAI(_FakeResponse(msg))), \
         _patch_settings():
        action, payload = inference_module.classify_and_route('user text')
    expect('classify_store_text_action', action == 'store_text')
    expect('classify_store_text_payload_is_json', isinstance(payload, str))
    parsed = json.loads(payload)
    expect('classify_store_text_text_field', parsed.get('text') == 'hello')
    expect('classify_store_text_source_default_empty', parsed.get('source') == '')

    # 2) Tool call ask_corpus
    msg = _FakeMessage(
        tool_calls=[type('T', (), {'function': _FakeFunction('ask_corpus', json.dumps({'question': 'What is X?'}))})()],
    )
    with patch.object(inference_module, 'OpenAI', return_value=_FakeOpenAI(_FakeResponse(msg))), \
         _patch_settings():
        action, payload = inference_module.classify_and_route('user text')
    expect('classify_ask_corpus_action', action == 'ask_corpus')
    expect('classify_ask_corpus_payload', payload == 'What is X?')

    # 3) Plain content, no tool calls
    msg = _FakeMessage(content='hello there', tool_calls=[])
    with patch.object(inference_module, 'OpenAI', return_value=_FakeOpenAI(_FakeResponse(msg))), \
         _patch_settings():
        action, payload = inference_module.classify_and_route('hi')
    expect('classify_chat_action', action == 'chat')
    expect('classify_chat_payload', payload == 'hello there')

    # 4) Tool call BadRequestError → graceful chat fallback
    with patch.object(inference_module, 'OpenAI', return_value=_FakeOpenAI(_BadRequestError('tools not supported'))), \
         _patch_settings():
        action, payload = inference_module.classify_and_route('hi')
    expect('classify_bad_request_action', action == 'chat')
    expect('classify_bad_request_payload_mentions_tool', 'tool' in payload.lower())
    expect('classify_bad_request_does_not_raise', True)

    # 5) APIError → graceful chat fallback
    with patch.object(inference_module, 'OpenAI', return_value=_FakeOpenAI(_APIError('server down'))), \
         _patch_settings():
        action, payload = inference_module.classify_and_route('hi')
    expect('classify_api_error_action', action == 'chat')
    expect('classify_api_error_does_not_raise', True)

    # 6) Malformed JSON in tool args → graceful chat fallback
    msg = _FakeMessage(
        tool_calls=[type('T', (), {'function': _FakeFunction('store_text', '{not json')})()],
    )
    with patch.object(inference_module, 'OpenAI', return_value=_FakeOpenAI(_FakeResponse(msg))), \
         _patch_settings():
        action, payload = inference_module.classify_and_route('hi')
    expect('classify_bad_args_action', action == 'chat')
    expect('classify_bad_args_payload_mentions_malformed', 'malformed' in payload.lower() or 'json' in payload.lower())

    # 7) Unsupported tool name → graceful chat fallback
    msg = _FakeMessage(
        tool_calls=[type('T', (), {'function': _FakeFunction('some_other_tool', '{}')})()],
    )
    with patch.object(inference_module, 'OpenAI', return_value=_FakeOpenAI(_FakeResponse(msg))), \
         _patch_settings():
        action, payload = inference_module.classify_and_route('hi')
    expect('classify_unsupported_tool_action', action == 'chat')
    expect('classify_unsupported_tool_payload_mentions_tool', 'tool' in payload.lower())

    # 8) attachment_notice is prepended to the user message
    captured_kwargs = {}

    class _CapturingCompletions(_FakeCompletions):
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeResponse(_FakeMessage(content='ok'))

    class _CapturingOpenAI(_FakeOpenAI):
        def __init__(self) -> None:
            self.chat = type('C', (), {'completions': _CapturingCompletions(None)})()

    with patch.object(inference_module, 'OpenAI', return_value=_CapturingOpenAI()), \
         _patch_settings():
        inference_module.classify_and_route('caption text', attachment_notice='Ingested 5 chunks from a.pdf')
    user_msg = captured_kwargs['messages'][-1]
    expect('classify_attachment_notice_in_user_msg', 'Ingested 5 chunks from a.pdf' in user_msg['content'])
    expect('classify_caption_in_user_msg', 'caption text' in user_msg['content'])


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    print('== agent handler behavioral tests ==')
    run_handler_behavioral_tests()
    print('\n== agent handler default-source tests ==')
    run_default_source_tests()
    print('\n== classify_and_route unit tests ==')
    run_classify_and_route_tests()
    print(f'\n{len(passed)} passed, {len(failed)} failed')
    if failed:
        for label, detail in failed:
            print(f'  - {label}: {detail}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
