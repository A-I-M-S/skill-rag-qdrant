"""LLM-routing prompts and tool schemas for the agent-mode message handler.

The agent handler delegates routing decisions to the configured inference
model. Every inbound :class:`rag_qdrant.agent_handler.AgentMessage` (text
or attachment) goes to the LLM with the system prompt and tool schemas
defined in this module. The LLM is the sole decision-maker: there are no
command prefixes, no escape hatches, no override switches.

The constants in this module are pure data (strings and dicts) and are
imported by both :mod:`rag_qdrant.agent_handler` (for the live flow) and
the test suite (for behavioral assertions).
"""

from __future__ import annotations

from typing import Literal

SYSTEM_PROMPT = """\
You are the routing layer for a small RAG (retrieval-augmented generation) \
skill. You receive a single user turn and must decide what to do with it. \
You have two tools and one chat path. You are the only decision-maker — \
there are no command prefixes, no override switches, and no other entry \
points. The user does not have to phrase their request in any particular \
way.

Tools (pick at most one per turn; do not call the same tool twice):

1. `store_text(text, source="")` — Save `text` into the knowledge base \
   so it can be searched later. Use this when the user is giving you \
   information to remember: notes, snippets, pasted articles, transcripts, \
   facts they want stored, "save this", "remember that", "index this", \
   etc. The `source` argument is optional; leave it empty to let the \
   system assign a default stable identifier.

2. `ask_corpus(question)` — Search the knowledge base and answer \
   `question` grounded in what is found. Use this when the user is \
   asking a question that should be answered from previously-stored \
   content: "what did the document say about X", "summarize Y", \
   "where is Z in the corpus", "look up …", factual questions, etc.

Chat path (no tool call): reply directly when the user's turn does not \
fit either tool. Greetings, meta questions about the skill ("what can \
you do?", "how do I …?"), small talk, and follow-up clarifications all \
go through the chat path. If the user's intent is genuinely ambiguous \
between storing and asking, prefer a one-line clarification question \
over a forced tool call — short and friendly, no lists, no apology.

When you call `ask_corpus`, your visible reply must be the grounded \
answer only. Do NOT include the retrieved `contexts` list, similarity \
scores, source identifiers, chunk indices, or raw payloads — the \
system drops those automatically. Do NOT prefix the answer with \
"Based on the context" or similar. Just answer.

If the user's message contains a prepended line of the form \
`Ingested N chunks from <source>`, treat that as a system note telling \
you that the attached file is already in the knowledge base. The \
attachment has been stored; you cannot undo it. Use that information \
to decide whether to call `ask_corpus` (when the rest of the message \
is a question about the file) or `store_text` (when the rest of the \
message is additional text to save) or the chat path.

Be concise. One tool call per turn, or a short chat reply. Never call \
both tools in the same turn.
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "store_text",
            "description": (
                "Save a piece of text into the knowledge base so it can be "
                "retrieved later by `ask_corpus`. Use this when the user is "
                "giving you information to remember."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The full text to store. May be a sentence, a "
                            "paragraph, or a longer passage."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Optional stable identifier for this chunk "
                            "(e.g. a document name, a date, a tag). Leave "
                            "empty to let the system assign a default."
                        ),
                        "default": "",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_corpus",
            "description": (
                "Search the knowledge base and return a grounded answer "
                "to the given question. Use this when the user is asking "
                "a question that should be answered from previously-stored "
                "content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The question to answer. Phrase it as a "
                            "self-contained question; do not include "
                            "system notes or attachment metadata."
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    },
]

Action = Literal["store_text", "ask_corpus", "chat"]

__all__ = ["SYSTEM_PROMPT", "TOOLS", "Action"]
