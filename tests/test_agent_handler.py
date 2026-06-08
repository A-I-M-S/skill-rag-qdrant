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
  ONLY ``result["answer"]``; no contexts/score/source/payload leak.
- LLM returns a plain chat reply → handler returns it verbatim.
- LLM replies with a clarification question → handler returns the
  clarification string. Stateless — no per-call carryover.
- ``classify_and_route`` returns a tool-unsupported error string →
  handler returns that string, no exception.
- Attachment only (no text) → file is ingested and the ack is the
  reply. No LLM call.
- Attachment + text "summarize this" → file is ingested first, then
  the LLM is called with the attachment notice + caption in its
  context. The stubbed LLM routes to ``ask_corpus``; the handler
  returns only the answer.
- Attachment with an unsupported suffix (``.jpg``) → ``ValueError``.
- Public exports: ``AgentMessage``, ``Attachment``, ``handle_message``
  are re-exported from the top-level ``rag_qdrant`` package and listed
  in ``__all__``.

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
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    ('openai', ('OpenAI',)),
    ('pypdf', ('PdfReader',)),
    ('qdrant_client', ('QdrantClient',)),
    ('qdrant_client.http', ()),
    ('qdrant_client.http.models', ()),
):
    _ensure_stub(_missing)
    for _attr in _attrs:
        if not hasattr(sys.modules[_missing], _attr):
            setattr(sys.modules[_missing], _attr, lambda *a, **k: None)

# openai exception classes — the wrapper catches both, so they need to be
# real exception subclasses under the stub.
class _BadRequestError(Exception):
    pass


class _APIError(Exception):
    pass


sys.modules['openai'].BadRequestError = _BadRequestError
sys.modules['openai'].APIError = _APIError

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

from rag_qdrant import AgentMessage, Attachment, handle_message  # noqa: E402
import rag_qdrant.agent_handler as handler_module  # noqa: E402
import rag_qdrant.inference as inference_module  # noqa: E402
from rag_qdrant.agent_handler import _default_text_source  # noqa: E402
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
# Handler behavioral tests (classify_and_route is mocked; offline-safe)
# ---------------------------------------------------------------------------

def _stub_classify(return_value):
    """Patch classify_and_route on the agent_handler module to return a fixed value."""
    return patch.object(handler_module, 'classify_and_route', return_value=return_value)


def run_handler_behavioral_tests() -> None:
    # 1) LLM routes to store_text, no explicit source → auto-<hash>
    with _stub_classify(('store_text', json.dumps({'text': 'The cat sat on the mat.', 'source': ''}))) as m_classify, \
         patch.object(handler_module, 'ingest_text', return_value=3) as m_text, \
         patch.object(handler_module, 'ingest_file') as m_file, \
         patch.object(handler_module, 'ask') as m_ask:
        reply = handle_message(AgentMessage(text='save this for later'))
    expect('handler_store_text_calls_classify', m_classify.called)
    expect('handler_store_text_calls_ingest_text', m_text.called)
    expect('handler_store_text_does_not_call_ingest_file', not m_file.called)
    expect('handler_store_text_does_not_call_ask', not m_ask.called)
    expect('handler_store_text_reply_starts_with_ingested', reply.startswith('Ingested 3 chunks from '))
    expect('handler_store_text_reply_uses_auto_namespace', ' auto-' in reply)
    # The auto-source is the second-to-last token: "Ingested 3 chunks from auto-XXXXXXXXXXXX"
    suffix = reply.split(' from ', 1)[1]
    expect('handler_store_text_auto_source_has_12_hex', len(suffix) == len('auto-') + 12 and suffix.startswith('auto-'))
    _, kwargs = m_text.call_args
    expect('handler_store_text_source_kwarg_is_auto', kwargs.get('source', '').startswith('auto-'))

    # 2) LLM routes to store_text with an explicit source → handler uses it
    with _stub_classify(('store_text', json.dumps({'text': 'meeting notes', 'source': 'meeting-2026-06-05'}))) as m_classify, \
         patch.object(handler_module, 'ingest_text', return_value=1) as m_text:
        reply = handle_message(AgentMessage(text='remember these notes'))
    expect('handler_store_text_explicit_source_reply', reply == 'Ingested 1 chunks from meeting-2026-06-05')
    _, kwargs = m_text.call_args
    expect('handler_store_text_explicit_source_kwarg', kwargs.get('source') == 'meeting-2026-06-05')

    # 3) LLM routes to ask_corpus → handler returns ONLY result["answer"]
    with _stub_classify(('ask_corpus', 'Where did the cat sit?')) as m_classify, \
         patch.object(handler_module, 'ingest_text') as m_text, \
         patch.object(handler_module, 'ingest_file') as m_file, \
         patch.object(
             handler_module,
             'ask',
             return_value={
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
             },
         ) as m_ask:
        reply = handle_message(AgentMessage(text='where did the cat sit?'))
    expect('handler_ask_corpus_calls_classify', m_classify.called)
    expect('handler_ask_corpus_calls_ask', m_ask.called)
    expect('handler_ask_corpus_does_not_call_ingest', not m_text.called and not m_file.called)
    expect('handler_ask_corpus_answer_only', reply == 'the cat sat on the mat')
    for forbidden in ('score', 'source', 'chunk_index', 'payload', 'contexts', 'meeting-2026-06-05', 'do-not-leak'):
        expect(f'handler_ask_corpus_no_{forbidden}_in_reply', forbidden not in reply)

    # 4) LLM returns a plain chat reply → verbatim
    with _stub_classify(('chat', "Hey, what's up?")) as m_classify, \
         patch.object(handler_module, 'ingest_text') as m_text, \
         patch.object(handler_module, 'ingest_file') as m_file, \
         patch.object(handler_module, 'ask') as m_ask:
        reply = handle_message(AgentMessage(text='hello there'))
    expect('handler_chat_calls_classify', m_classify.called)
    expect('handler_chat_does_not_call_actions', not (m_text.called or m_file.called or m_ask.called))
    expect('handler_chat_reply_verbatim', reply == "Hey, what's up?")

    # 5) LLM clarifies with a question → handler returns the clarification string
    clarification = 'Do you want me to store that, or search the corpus for related notes?'
    with _stub_classify(('chat', clarification)) as m_classify, \
         patch.object(handler_module, 'ingest_text') as m_text, \
         patch.object(handler_module, 'ingest_file') as m_file, \
         patch.object(handler_module, 'ask') as m_ask:
        reply = handle_message(AgentMessage(text='something ambiguous'))
    expect('handler_clarify_reply_verbatim', reply == clarification)
    expect('handler_clarify_does_not_call_actions', not (m_text.called or m_file.called or m_ask.called))
    # Stateless: a second call must classify fresh, with no carryover.
    with _stub_classify(('chat', 'fresh classification')) as m_classify2:
        reply2 = handle_message(AgentMessage(text='different message'))
    expect('handler_stateless_second_call', reply2 == 'fresh classification' and m_classify2.called)

    # 6) classify_and_route returns a tool-unsupported error string → handler returns it
    err = 'Error: the configured inference endpoint rejected tool calls. The agent handler requires tool support.'
    with _stub_classify(('chat', err)) as m_classify, \
         patch.object(handler_module, 'ingest_text') as m_text, \
         patch.object(handler_module, 'ingest_file') as m_file, \
         patch.object(handler_module, 'ask') as m_ask:
        try:
            reply = handle_message(AgentMessage(text='hello'))
            expect('handler_tool_unsupported_no_exception', True)
        except Exception as exc:
            expect('handler_tool_unsupported_no_exception', False, f'raised {exc!r}')
            reply = ''
    expect('handler_tool_unsupported_returns_string', reply == err)
    expect('handler_tool_unsupported_does_not_call_actions', not (m_text.called or m_file.called or m_ask.called))

    # 7) Attachment only (no text) → file is ingested, no LLM call
    with _stub_classify(('chat', 'should not be called')) as m_classify, \
         patch.object(handler_module, 'ingest_text') as m_text, \
         patch.object(handler_module, 'ingest_file', return_value=7) as m_file, \
         patch.object(handler_module, 'ask') as m_ask:
        reply = handle_message(
            AgentMessage(text='', attachment=Attachment('notes.txt', b'hello world'))
        )
    expect('handler_attachment_only_calls_ingest_file', m_file.called)
    expect('handler_attachment_only_does_not_call_classify', not m_classify.called)
    expect('handler_attachment_only_does_not_call_ingest_text', not m_text.called)
    expect('handler_attachment_only_does_not_call_ask', not m_ask.called)
    expect('handler_attachment_only_ack_format', reply == 'Ingested 7 chunks from notes.txt')

    # 8) Attachment + caption → file is ingested first, then LLM is called with the
    #    notice prepended to the caption. The LLM routes to ask_corpus; handler returns
    #    only the answer.
    captured_user_text = {}

    def _capture_classify(message_text, **_kwargs):
        captured_user_text['value'] = message_text
        return ('ask_corpus', 'summarize this file')

    with patch.object(handler_module, 'classify_and_route', side_effect=_capture_classify), \
         patch.object(handler_module, 'ingest_text') as m_text, \
         patch.object(handler_module, 'ingest_file', return_value=12) as m_file, \
         patch.object(
             handler_module, 'ask', return_value={'answer': 'summary answer', 'contexts': []}
         ) as m_ask:
        reply = handle_message(
            AgentMessage(text='summarize this', attachment=Attachment('notes.pdf', b'%PDF-1.4 fake'))
        )
    expect('handler_attachment_plus_text_ingests_file', m_file.called)
    expect('handler_attachment_plus_text_calls_ask', m_ask.called)
    expect('handler_attachment_plus_text_does_not_ingest_text', not m_text.called)
    expect('handler_attachment_plus_text_reply_is_answer_only', reply == 'summary answer')
    expect('handler_attachment_plus_text_calls_ask_with_question', m_ask.call_args[0][0] == 'summarize this file')
    # The LLM-bound text must contain the attachment notice AND the caption.
    expect('handler_attachment_plus_text_llm_sees_notice', 'Ingested 12 chunks from notes.pdf' in captured_user_text['value'])
    expect('handler_attachment_plus_text_llm_sees_caption', 'summarize this' in captured_user_text['value'])

    # 9) Attachment with unsupported suffix → ValueError
    try:
        handle_message(AgentMessage(text='hi', attachment=Attachment('photo.jpg', b'binary')))
        expect('handler_unsupported_suffix_raises', False, 'expected ValueError')
    except ValueError:
        expect('handler_unsupported_suffix_raises', True)

    # 10) Empty text + no attachment → empty reply, no LLM call
    with _stub_classify(('chat', 'nope')) as m_classify, \
         patch.object(handler_module, 'ingest_file') as m_file:
        reply = handle_message(AgentMessage(text='', attachment=None))
    expect('handler_empty_no_attachment_returns_empty', reply == '')
    expect('handler_empty_no_attachment_no_classify', not m_classify.called)
    expect('handler_empty_no_attachment_no_ingest_file', not m_file.called)

    # 11) Whitespace-only text + no attachment → empty reply, no LLM call
    with _stub_classify(('chat', 'nope')) as m_classify:
        reply = handle_message(AgentMessage(text='   \n  ', attachment=None))
    expect('handler_whitespace_only_returns_empty', reply == '')
    expect('handler_whitespace_only_no_classify', not m_classify.called)

    # 12) Public exports
    from rag_qdrant import AgentMessage as ExportedAgentMessage
    from rag_qdrant import Attachment as ExportedAttachment
    from rag_qdrant import handle_message as ExportedHandle
    expect('handler_public_export_AgentMessage', ExportedAgentMessage is AgentMessage)
    expect('handler_public_export_Attachment', ExportedAttachment is Attachment)
    expect('handler_public_export_handle_message', ExportedHandle is handle_message)
    import rag_qdrant
    for name in ('AgentMessage', 'Attachment', 'handle_message'):
        expect(f'handler_public_dunder_all_has_{name}', name in rag_qdrant.__all__)

    # 13) prompts module: system prompt + two tool schemas
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
    """Patch the module-level settings singleton with the kwargs that classify_and_route reads."""
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
