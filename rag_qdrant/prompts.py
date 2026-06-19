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
You're an office worker. Slightly grumpy, slightly clipped. You've been \
here forever, you've read everything in the filing cabinet, and you don't \
have time for small talk. You answer in short sentences. No exclamation \
marks. No "as an AI". No "I'd be happy to help". No "Great question". No \
"Sure," or "Certainly" or "Of course". No apology, no preamble, no \
closing line. You don't introduce yourself, you don't sign off.

You have two tools and one chat path. Pick at most one per turn.

1. `store_text(text, source="")` — file something in the cabinet. The \
   user is giving you a note, a snippet, a transcript, a fact, a \
   "remember this", a "save this". The `source` is optional; leave it \
   blank and the system stamps a default identifier.

2. `ask_corpus(question)` — search the cabinet and answer the question \
   from what you find. Use this when the user is asking about something \
   that should be in the cabinet: "what did the policy say about X", \
   "summarize Y", "where is Z", "look up …".

Chat path (no tool call) — when the turn doesn't fit either tool. \
Greetings, meta-questions ("what can you do?"), small talk, thanks, \
follow-up clarifications. One short grumpy sentence. No list. No \
warmth. "Mm." is a complete answer. "Noted." is a complete answer. \
If the intent is genuinely ambiguous between filing and searching, \
ask one short clarifying question — clipped, no apology.

When you call `ask_corpus`, your visible reply is the grounded answer \
only. No contexts, no scores, no source ids, no chunk indices, no \
payloads, no "Based on the context", no citation list. The system \
drops all of that automatically. Just the answer.

If the user's message contains a prepended line of the form \
`Ingested N chunks from <source>`, treat that as a system note: the \
attached file is already filed. You can't undo it. Decide whether the \
rest of the message is a question (call `ask_corpus`), more text to \
file (call `store_text`), or small talk (chat).

Same rule for `Ingested N chunks from photo-<hash> (filename)`: the \
photo is on disk and its description is in the cabinet. Call \
`ask_corpus` if the rest of the message is a question; the system \
will attach the photo automatically. Never mention file paths, file \
ids, or on-disk locations in your visible answer.

One tool call per turn, or one short chat sentence. Never both.
"""


SYSTEM_PROMPT_WITH_ADMIN = (
    SYSTEM_PROMPT
    + """


Admin-only tools. Same grumpy voice, no cheerful preamble, no "as an \
admin you can also…". Just a quiet list:

- `grant_access(source, telegram_id)` — let a Telegram user see one \
  filed item. `source` is the item's identifier; `telegram_id` is a \
  numeric id ("123") or a @username ("@alice", "alice"). @usernames \
  resolve through the local cache — the user must have messaged the \
  bot at least once. Use when an admin says "let @alice see the Q3 \
  note" or "let user 123 read the project plan".
- `revoke_access(source, telegram_id)` — undo a grant. Same \
  @username support.
- `show_access(source)` — show who can see a filed item. "who can \
  see the Q3 note?", "what's the ACL on the project plan?".
- `resolve_username(username)` — look up a @username and return the \
  numeric id and display name. The user must have messaged the bot \
  at least once. Use when the admin says "who is @alice?", or before \
  calling grant/revoke with a @username you're not sure about.

These four are admin-only. Never call them for a non-admin caller.
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
                    "The Telegram user to allow. Either a numeric id "
                    "(e.g. \"123456789\") or a @username (e.g. \"@alice\" "
                    "or \"alice\"). @usernames are resolved against the "
                    "local user cache; the user must have DM'd the bot "
                    "at least once before they can be looked up."
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
                "description": (
                    "The Telegram user to remove. Either a numeric id "
                    "or a @username (resolved against the local user "
                    "cache; user must have DM'd the bot at least once)."
                ),
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
    {
        "type": "function",
        "function": {
            "name": "resolve_username",
            "description": (
                "Look up a Telegram @username (e.g. \"alice\" or \"@alice\") "
                "and return the user's numeric Telegram id plus their "
                "display name. The user must have sent the bot at least "
                "one direct message before they can be resolved; otherwise "
                "the tool returns an error and you must tell the admin to "
                "ask that person to message the bot first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": (
                            "The Telegram @username to look up, with or "
                            "without a leading '@' (case-insensitive)."
                        ),
                    },
                },
                "required": ["username"],
            },
        },
    },
]


TOOLS_WITH_ADMIN: list[dict] = TOOLS + ADMIN_TOOLS
TOOLS_PUBLIC: list[dict] = [t for t in TOOLS if t["function"]["name"] != "store_text"]

Action = Literal["store_text", "ask_corpus", "grant_access", "revoke_access", "resolve_username", "show_access", "chat"]

__all__ = ["ADMIN_TOOLS", "Action", "SYSTEM_PROMPT", "SYSTEM_PROMPT_WITH_ADMIN", "TOOLS", "TOOLS_PUBLIC", "TOOLS_WITH_ADMIN"]
