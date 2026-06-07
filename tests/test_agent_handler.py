"""Self-contained tests for ``rag_qdrant.agent_handler``.

Usage: ``python3 tests/test_agent_handler.py``

Mirrors the offline-stubbing style of ``tests/run_tests.py`` so the
file runs without Qdrant / FastEmbed / OpenAI / pypdf / dotenv.

Covers (source-grep + behavioral):

- Rule 1: ``Embed <text>`` -> ``ingest_text`` with default source
  ``telegram-<sha1(text[:40])[:12]>`` (timestamp fallback when empty).
- Rule 2: ``Embed`` + PDF/TXT/MD attachment -> ``ingest_file`` with
  ``source=<filename>``; temp file is written and cleaned up.
- Rule 3: ``Query <text>`` -> ``ask``; return **only**
  ``result["answer"]``; score/source/chunk_index/payload/contexts are
  never included in the reply.
- Rule 4: ack format ``"Ingested N chunks from <source>"``.

Plus negative-path checks:

- ``Embed`` with no text and no attachment -> ``ValueError``.
- ``Query`` with no body -> ``ValueError``.
- Unknown command -> ``ValueError``.
"""

from __future__ import annotations

import importlib
import re
import sys
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
from rag_qdrant.agent_handler import _default_text_source  # noqa: E402

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
# Helpers
# ---------------------------------------------------------------------------

HANDLER_PATH = ROOT / 'rag_qdrant' / 'agent_handler.py'


def _handler_source() -> str:
    return HANDLER_PATH.read_text(encoding='utf-8')


def _query_branch_block(src: str) -> str:
    """Extract the body of ``_handle_query`` (or the inline Query branch).

    Used to scope the "no leaked metadata" assertions to the Query path
    so they don't false-positive on the Embed branch's ``source=`` arg.
    """
    match = re.search(r'def _handle_query\(.*?\):\n((?:\s.*\n)+)', src)
    if match is None:
        return ''
    return match.group(1)


# ---------------------------------------------------------------------------
# Source-grep assertions (the four rules)
# ---------------------------------------------------------------------------

def run_source_grep_tests() -> None:
    src = _handler_source()

    # Rule 1: Embed <text> -> ingest_text with default source
    expect('rule1_calls_ingest_text', 'ingest_text(' in src)
    expect('rule1_uses_default_text_source', '_default_text_source' in src)
    expect('rule1_source_namespace', '\'telegram\'' in src or '"telegram"' in src)
    expect('rule1_hashes_prefix_or_ts', 'sha1' in src and 'datetime' in src)
    expect('rule1_strips_embed_prefix', '_STRIP_EMBED_RE' in src)

    # Rule 2: Embed + attachment -> ingest_file with source=filename
    expect('rule2_calls_ingest_file', 'ingest_file(' in src)
    expect('rule2_writes_tempfile', 'tempfile' in src and 'NamedTemporaryFile' in src)
    expect('rule2_source_is_filename', 'message.attachment.filename' in src)
    expect(
        'rule2_supported_suffix_set',
        '.pdf' in src and '.txt' in src and '.md' in src and '.text' in src,
    )
    expect('rule2_unlinks_tempfile', 'unlink' in src)

    # Rule 3: Query <text> -> ask, return ONLY result["answer"]
    expect('rule3_calls_ask', 'ask(' in src)
    expect(
        'rule3_returns_answer_key',
        "result['answer']" in src or 'result["answer"]' in src,
    )
    query_block = _query_branch_block(src)
    for forbidden in ('score', 'source', 'chunk_index', 'payload', 'contexts'):
        expect(f'rule3_no_{forbidden}_in_query_block', forbidden not in query_block)
    expect('rule3_strips_query_prefix', '_STRIP_QUERY_RE' in src)

    # Rule 4: ack format
    expect('rule4_uses_ingested_prefix', '\'Ingested' in src or '"Ingested' in src)
    expect('rule4_includes_chunks_from', 'chunks from' in src)

    # Negative paths
    expect('neg_embed_no_body_raises', 'ValueError' in src)
    expect('neg_query_no_body_raises', 'ValueError' in src)


# ---------------------------------------------------------------------------
# _default_text_source unit tests
# ---------------------------------------------------------------------------

def run_default_source_tests() -> None:
    # Non-empty text -> stable hash of the first 40 chars
    src1 = _default_text_source('hello world')
    src2 = _default_text_source('hello world')
    expect('default_source_is_stable', src1 == src2)
    expect('default_source_has_namespace', src1.startswith('telegram-'))
    expect('default_source_len_12', len(src1.split('-', 1)[1]) == 12)

    # Different prefixes -> different source
    expect(
        'default_source_different_for_different_text',
        _default_text_source('alpha') != _default_text_source('beta'),
    )

    # Empty / whitespace -> still a telegram- prefixed source (timestamp fallback)
    src_empty = _default_text_source('')
    src_ws = _default_text_source('   \n  ')
    expect('default_source_empty_still_namespaced', src_empty.startswith('telegram-'))
    expect('default_source_whitespace_still_namespaced', src_ws.startswith('telegram-'))
    expect('default_source_empty_len_12', len(src_empty.split('-', 1)[1]) == 12)

    # Long text is truncated to 40 chars of the stripped text
    long_text = 'a' * 200
    long_text_trimmed = 'a' * 40
    expect(
        'default_source_uses_prefix_only',
        _default_text_source(long_text) == _default_text_source(long_text_trimmed),
    )


# ---------------------------------------------------------------------------
# Behavioral tests (stubbed deps; offline-safe)
# ---------------------------------------------------------------------------

def run_behavioral_tests() -> None:
    # Rule 1: Embed <text> -> ingest_text, ack format
    with patch.object(handler_module, 'ingest_text', return_value=3) as m_text, \
         patch.object(handler_module, 'ingest_file') as m_file, \
         patch.object(handler_module, 'ask') as m_ask:
        reply = handle_message(AgentMessage(text='Embed The cat sat on the mat.'))
    expect('behavior_rule1_called_ingest_text', m_text.called)
    expect('behavior_rule1_did_not_call_ingest_file', not m_file.called)
    expect('behavior_rule1_did_not_call_ask', not m_ask.called)
    expect('behavior_rule1_ack_format', reply.startswith('Ingested 3 chunks from telegram-'))

    # Case-insensitive prefix
    with patch.object(handler_module, 'ingest_text', return_value=1) as m_text:
        reply = handle_message(AgentMessage(text='embed hello'))
    expect('behavior_rule1_case_insensitive', m_text.called and reply.startswith('Ingested 1 chunks from telegram-'))

    # Rule 2: Embed + PDF attachment -> ingest_file with source=filename
    with patch.object(handler_module, 'ingest_text') as m_text, \
         patch.object(handler_module, 'ingest_file', return_value=14) as m_file, \
         patch.object(handler_module, 'ask') as m_ask:
        reply = handle_message(
            AgentMessage(
                text='Embed',
                attachment=Attachment('notes.pdf', b'%PDF-1.4 fake bytes'),
            )
        )
    expect('behavior_rule2_called_ingest_file', m_file.called)
    expect('behavior_rule2_did_not_call_ingest_text', not m_text.called)
    expect('behavior_rule2_did_not_call_ask', not m_ask.called)
    expect('behavior_rule2_ack_format', reply == 'Ingested 14 chunks from notes.pdf')
    # Source kwarg is the attachment filename
    _, kwargs = m_file.call_args
    expect('behavior_rule2_source_kwarg_is_filename', kwargs.get('source') == 'notes.pdf')
    # Positional path arg is a real Path on disk that gets cleaned up
    args, _ = m_file.call_args
    expect('behavior_rule2_path_is_Path', isinstance(args[0], Path))
    expect('behavior_rule2_path_no_suffix', args[0].suffix == '.pdf')
    expect('behavior_rule2_tempfile_cleaned_up', not args[0].exists())

    # Rule 2 also supports .txt and .md
    for fname in ('a.txt', 'b.md', 'c.text'):
        with patch.object(handler_module, 'ingest_file', return_value=2) as m_file:
            reply = handle_message(
                AgentMessage(
                    text='Embed',
                    attachment=Attachment(fname, b'x'),
                )
            )
        expect(f'behavior_rule2_supports_{fname}', reply == f'Ingested 2 chunks from {fname}')

    # Rule 3: Query <text> -> ask, reply is ONLY result['answer']
    with patch.object(handler_module, 'ingest_text') as m_text, \
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
        reply = handle_message(AgentMessage(text='Query where did the cat sit?'))
    expect('behavior_rule3_called_ask', m_ask.called)
    expect('behavior_rule3_did_not_call_ingest', not m_text.called and not m_file.called)
    expect('behavior_rule3_answer_only', reply == 'the cat sat on the mat')
    for forbidden in ('score', 'source', 'chunk_index', 'payload', 'contexts', 'meeting-2026-06-05', 'do-not-leak'):
        expect(f'behavior_rule3_no_{forbidden}_in_reply', forbidden not in reply)

    # Case-insensitive Query
    with patch.object(
        handler_module, 'ask', return_value={'answer': 'A', 'contexts': []}
    ) as m_ask:
        reply = handle_message(AgentMessage(text='  QUERY hi'))
    expect('behavior_rule3_case_insensitive', m_ask.called and reply == 'A')

    # Rule 3 with cache: handler returns ONLY the answer, even when
    # the answer comes from the cache. We patch `ask` to return a
    # result that the handler treats as a cache hit, then verify the
    # contract is preserved (no leaked metadata, no contexts).
    cached_result = {
        'answer': 'cached answer only',
        'contexts': [
            {
                'score': 0.95,
                'source': 'leaky-source',
                'chunk_index': 7,
                'text': 'should-not-appear',
                'payload': {'leak': 'do-not-leak'},
            }
        ],
    }
    with patch.object(handler_module, 'ask', return_value=cached_result) as m_ask:
        reply = handle_message(AgentMessage(text='Query anything'))
    expect('behavior_rule3_cache_path_called_ask', m_ask.called)
    expect('behavior_rule3_cache_path_answer_only', reply == 'cached answer only')
    for forbidden in ('score', 'source', 'chunk_index', 'payload', 'contexts', 'leaky-source', 'do-not-leak', 'should-not-appear'):
        expect(f'behavior_rule3_cache_path_no_{forbidden}', forbidden not in reply)

    # Negative: Embed with no text and no attachment -> ValueError
    try:
        handle_message(AgentMessage(text='Embed', attachment=None))
        expect('neg_embed_no_body_no_attachment_raises', False, 'expected ValueError')
    except ValueError:
        expect('neg_embed_no_body_no_attachment_raises', True)
    try:
        handle_message(AgentMessage(text='Embed '))
        expect('neg_embed_whitespace_body_raises', False, 'expected ValueError')
    except ValueError:
        expect('neg_embed_whitespace_body_raises', True)

    # Negative: Query with no body -> ValueError
    try:
        handle_message(AgentMessage(text='Query'))
        expect('neg_query_no_body_raises', False, 'expected ValueError')
    except ValueError:
        expect('neg_query_no_body_raises', True)
    try:
        handle_message(AgentMessage(text='Query   '))
        expect('neg_query_whitespace_body_raises', False, 'expected ValueError')
    except ValueError:
        expect('neg_query_whitespace_body_raises', True)

    # Negative: unknown command -> ValueError
    try:
        handle_message(AgentMessage(text='hello there'))
        expect('neg_unknown_command_raises', False, 'expected ValueError')
    except ValueError:
        expect('neg_unknown_command_raises', True)

    # Public exports
    from rag_qdrant import AgentMessage as ExportedAgentMessage
    from rag_qdrant import Attachment as ExportedAttachment
    from rag_qdrant import handle_message as ExportedHandle
    expect('public_export_AgentMessage', ExportedAgentMessage is AgentMessage)
    expect('public_export_Attachment', ExportedAttachment is Attachment)
    expect('public_export_handle_message', ExportedHandle is handle_message)
    import rag_qdrant
    for name in ('AgentMessage', 'Attachment', 'handle_message'):
        expect(f'public_dunder_all_has_{name}', name in rag_qdrant.__all__)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    print('== agent handler source-grep tests ==')
    run_source_grep_tests()
    print('\n== agent handler default-source tests ==')
    run_default_source_tests()
    print('\n== agent handler behavioral tests ==')
    run_behavioral_tests()
    print(f'\n{len(passed)} passed, {len(failed)} failed')
    if failed:
        for label, detail in failed:
            print(f'  - {label}: {detail}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
