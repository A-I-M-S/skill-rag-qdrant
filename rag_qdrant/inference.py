from __future__ import annotations

import json

from openai import APIError, BadRequestError, OpenAI

from .cache import (
    semantic_cache_lookup,
    semantic_cache_store,
)
from .config import is_admin, settings
from .logging_setup import logger
from .photo_matching import extract_photos
from .prompts import Action, SYSTEM_PROMPT as _ROUTING_SYSTEM_PROMPT
from .qdrant_store import embed_texts, search

SYSTEM_PROMPT = """You are a precise RAG assistant. Answer only from the provided context. Do not cite sources, mention source IDs, or include a Sources section. If the provided context does not contain enough relevant information to answer, reply exactly: No relevant information found."""

NO_RELEVANT_ANSWER = "No relevant information found"


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_blocks = []
    for item in contexts:
        text = item.get("text", "")
        context_blocks.append(text)
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context found."
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _answer(prompt: str) -> str:
    client = OpenAI(api_key=settings.inference_api_key, base_url=settings.inference_base_url)
    response = client.chat.completions.create(
        model=settings.inference_model,
        temperature=settings.inference_temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def answer_question(
    question: str,
    *,
    current_telegram_id: int | str | None = None,
    allowed_telegram_id: int | str | None = None,
    is_admin_override: bool | None = None,
) -> dict:
    """Search Qdrant (with the ACL filter) and answer via the configured inference model.

    The ACL parameters mirror :func:`rag_qdrant.qdrant_store.search`:

    - ``current_telegram_id`` is the user who sent the question. If
      non-``None``, admins (per :func:`rag_qdrant.config.is_admin`) see
      every chunk; non-admins see only chunks whose ACL matches the
      id. When ``current_telegram_id`` is ``None`` the behavior is the
      public default (filter requires the ``"*"`` wildcard; legacy
      un-ACLed chunks are still returned via Qdrant's missing-field
      semantics).
    - ``is_admin_override`` lets a caller (e.g. the CLI) force the
      admin path without going through :func:`is_admin`.
    - ``allowed_telegram_id`` is a low-level escape hatch for the
      same value; if it is provided, it is used directly instead of
      :func:`is_admin`.
    """
    settings.require_inference()

    if allowed_telegram_id is None and current_telegram_id is not None:
        if is_admin_override is not None:
            admin = bool(is_admin_override)
        else:
            admin = is_admin(current_telegram_id)
        effective_acl_id: int | str | None = None if admin else current_telegram_id
        effective_admin = admin
    else:
        effective_acl_id = allowed_telegram_id
        effective_admin = bool(is_admin_override) if is_admin_override is not None else False

    user_specific = effective_acl_id is not None and not effective_admin

    if settings.semantic_cache_enabled and not user_specific:
        query_embedding = embed_texts([question], query=True)[0]
        cached = semantic_cache_lookup(question, query_embedding)
        if cached is not None:
            return cached
    else:
        query_embedding = None

    contexts = search(
        question,
        top_k=settings.top_k,
        query_vector=query_embedding,
        allowed_telegram_id=effective_acl_id,
        is_admin=effective_admin,
    )
    contexts = [item for item in contexts if float(item.get("score") or 0) >= settings.min_relevance_score]
    if not contexts:
        logger.info("inference_no_relevant_context question_chars=%s min_relevance_score=%s", len(question), settings.min_relevance_score)
        result = {"answer": NO_RELEVANT_ANSWER, "contexts": [], "photos": []}
        if settings.semantic_cache_enabled and query_embedding is not None and not user_specific:
            semantic_cache_store(question, query_embedding, result, is_miss=True)
        return result
    prompt = build_prompt(question, contexts)
    logger.info(
        "inference_start model=%s question_chars=%s contexts=%s",
        settings.inference_model,
        len(question),
        len(contexts),
    )
    answer = _answer(prompt)
    logger.info(
        "inference_done model=%s answer_chars=%s",
        settings.inference_model,
        len(answer),
    )
    result = {"answer": answer, "contexts": contexts, "photos": extract_photos(contexts)}
    if settings.semantic_cache_enabled and query_embedding is not None and not user_specific:
        semantic_cache_store(question, query_embedding, result, is_miss=False)
    return result


ask = answer_question


def classify_and_route(
    message_text: str,
    *,
    attachment_notice: str = "",
    system_prompt: str = _ROUTING_SYSTEM_PROMPT,
    tools: list[dict] | None = None,
) -> tuple[Action, str]:
    """Ask the inference model to route one inbound turn to a single action.

    Thin wrapper around the OpenAI-compatible chat completions endpoint
    used by :func:`answer_question`. The model sees the system prompt,
    the two tool schemas, and a single user message; it returns either a
    tool call (``store_text`` or ``ask_corpus``) or a plain chat reply.

    Parameters
    ----------
    message_text:
        The user-facing text the LLM should classify. Already stripped
        of any transport noise; this is what the LLM "sees" as the
        user's request.
    attachment_notice:
        Optional informational line (``"Ingested N chunks from <source>"``)
        prepended to the user message when the inbound turn came with a
        supported attachment. Defaults to empty.
    system_prompt:
        Override the system prompt (used by tests). Defaults to the
        module-level routing prompt.
    tools:
        Override the tool schema list (used by tests). Defaults to
        importing :data:`rag_qdrant.prompts.TOOLS` lazily to keep this
        module's import surface stable.

    Returns
    -------
    ``(action, payload)`` where ``action`` is one of:

    - ``"store_text"`` → ``payload`` is a JSON string with keys
      ``"text"`` (required) and ``"source"`` (optional, may be empty).
    - ``"ask_corpus"`` → ``payload`` is the question string verbatim.
    - ``"chat"``       → ``payload`` is the assistant's plain text reply,
      or a clear error string when the endpoint rejected tool support
      (or any other API error) — in which case the caller treats it as
      a chat reply and surfaces it to the user.

    This function never raises. On any
    :class:`openai.BadRequestError` / :class:`openai.APIError` / JSON
    parse failure, it returns ``("chat", "<error string>")`` so the
    agent handler can return a graceful string reply without an
    exception path.
    """
    if tools is None:
        from .prompts import TOOLS as _TOOLS
        tools = _TOOLS

    user_content = message_text or ""
    if attachment_notice:
        user_content = f"{attachment_notice}\n\n{user_content}" if user_content else attachment_notice

    try:
        client = OpenAI(api_key=settings.inference_api_key, base_url=settings.inference_base_url)
        response = client.chat.completions.create(
            model=settings.inference_model,
            temperature=settings.inference_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
    except BadRequestError as exc:
        logger.warning("classify_and_route_bad_request error=%s", exc)
        return (
            "chat",
            f"Error: the configured inference endpoint rejected tool calls ({exc}). The agent handler requires an OpenAI-compatible endpoint that supports tool/function calling. Falling back to chat.",
        )
    except APIError as exc:
        logger.warning("classify_and_route_api_error error=%s", exc)
        return ("chat", f"Error: inference endpoint call failed ({exc}).")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("classify_and_route_unexpected error=%s", exc)
        return ("chat", f"Error: inference endpoint call failed ({exc}).")

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        call = tool_calls[0]
        name = getattr(call.function, "name", "") or ""
        raw_args = getattr(call.function, "arguments", "") or ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (TypeError, ValueError) as exc:
            logger.warning("classify_and_route_bad_args name=%s error=%s", name, exc)
            return ("chat", f"Error: malformed tool-call arguments from the LLM ({exc}).")
        if name == "store_text":
            text = args.get("text") or ""
            if not text:
                return ("chat", "Error: store_text called with empty text.")
            return ("store_text", json.dumps({"text": text, "source": args.get("source") or ""}))
        if name == "ask_corpus":
            question = args.get("question") or ""
            if not question:
                return ("chat", "Error: ask_corpus called with empty question.")
            return ("ask_corpus", question)
        return ("chat", f"Error: unsupported tool call from the LLM ({name!r}).")

    return ("chat", message.content or "")
