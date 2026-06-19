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

When the prepended line matches the form `Ingested N chunks from \
photo-<hash> (<filename>)`, the user's photo is already saved on disk \
and its description is already searchable in the corpus. The system \
will automatically attach the photo to the final reply if a future \
`ask_corpus` call matches it — you do not need to (and must not) \
mention photo paths, file IDs, or the on-disk location in your \
visible answer. Just call `ask_corpus` with the user's question and \
the system handles the photo display.

Be concise. One tool call per turn, or a short chat reply. Never call \
both tools in the same turn.
"""

SYSTEM_PROMPT_WITH_ADMIN = (
    SYSTEM_PROMPT
    + """

You also have access to three access-control tools, because the current
caller is an admin:

- `grant_access(source, telegram_id)` — let a Telegram user see a
  specific stored item. The `source` argument is the item's stable
  identifier; `telegram_id` is the numeric Telegram user id to allow.
  Use this when an admin says "let @alice see the Q3 note" or
  "let user 123 read the project plan".
- `revoke_access(source, telegram_id)` — undo a previous grant.
  Use this when an admin says "remove @bob from it" or "revoke
  access for 123 on the project plan".
- `show_access(source)` — show who currently has access to a stored
  item. Use this when an admin asks "who can see the Q3 note?" or
  "what's the ACL on the project plan?".

These three tools are admin-only. Never call them on a non-admin
caller's behalf.
"""
)

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


def _admin_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


ADMIN_TOOLS: list[dict] = [
    _admin_tool(
        "grant_access",
        (
            "Add a Telegram user id to the access-control list of every chunk "
            "stored under the given source. Use when an admin says things like "
            "'let @alice see the Q3 note' or 'allow user 123 to read the "
            "project plan'."
        ),
        properties={
            "source": {
                "type": "string",
                "description": (
                    "The stable identifier of the stored item to grant "
                    "access to (the same value that was passed to "
                    "store_text as the `source` argument, or the "
                    "auto-generated `auto-xxxxxxxxxxxx` value)."
                ),
            },
            "telegram_id": {
                "type": "string",
                "description": (
                    "The numeric Telegram user id to allow, as a string "
                    "(e.g. \"123456789\")."
                ),
            },
        },
        required=["source", "telegram_id"],
    ),
    _admin_tool(
        "revoke_access",
        (
            "Remove a Telegram user id from the access-control list of every "
            "chunk stored under the given source. Use when an admin says "
            "things like 'remove @bob from it' or 'revoke access for 123 on "
            "the project plan'."
        ),
        properties={
            "source": {
                "type": "string",
                "description": "The stable identifier of the stored item.",
            },
            "telegram_id": {
                "type": "string",
                "description": "The numeric Telegram user id to remove.",
            },
        },
        required=["source", "telegram_id"],
    ),
    _admin_tool(
        "show_access",
        (
            "Show the current access-control list and chunk count for a "
            "stored item. Use when an admin asks 'who can see the Q3 note?' "
            "or 'what's the ACL on the project plan?'."
        ),
        properties={
            "source": {
                "type": "string",
                "description": "The stable identifier of the stored item.",
            },
        },
        required=["source"],
    ),
]


TOOLS_WITH_ADMIN: list[dict] = TOOLS + ADMIN_TOOLS
TOOLS_PUBLIC: list[dict] = [t for t in TOOLS if t["function"]["name"] != "store_text"]

Action = Literal["store_text", "ask_corpus", "grant_access", "revoke_access", "show_access", "chat"]

__all__ = ["ADMIN_TOOLS", "Action", "SYSTEM_PROMPT", "SYSTEM_PROMPT_WITH_ADMIN", "TOOLS", "TOOLS_PUBLIC", "TOOLS_WITH_ADMIN"]
