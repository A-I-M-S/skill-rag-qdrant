from __future__ import annotations

import os
import requests
from openai import OpenAI

from .config import settings
from .logging_setup import logger
from .qdrant_store import search

SYSTEM_PROMPT = """You are a precise RAG assistant. Answer only from the provided context. If the context is insufficient, say what is missing. Cite sources inline as [source:chunk_index]."""


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_blocks = []
    for item in contexts:
        source = item.get("source", "unknown")
        chunk_index = item.get("chunk_index", "?")
        score = item.get("score", 0)
        text = item.get("text", "")
        context_blocks.append(f"[source={source} chunk={chunk_index} score={score:.4f}]\n{text}")
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context found."
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _answer_with_zo_ask(prompt: str) -> str:
    token = settings.inference_api_key or os.getenv("ZO_CLIENT_IDENTITY_TOKEN", "")
    response = requests.post(
        "https://api.zo.computer/zo/ask",
        headers={
            "authorization": token,
            "content-type": "application/json",
        },
        json={
            "input": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "model_name": settings.inference_model,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("output", "") if isinstance(data, dict) else str(data)


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(message.get("reasoning") or "")


def _answer_with_openrouter(prompt: str) -> str:
    payload = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": settings.inference_temperature,
        "stream": False,
    }
    if settings.openrouter_provider:
        payload["provider"] = {
            "order": [settings.openrouter_provider],
            "allow_fallbacks": False,
        }

    response = requests.post(
        settings.openrouter_url,
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", []) if isinstance(data, dict) else []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {data}")
    message = choices[0].get("message", {})
    return _message_text(message)


def _answer_with_openai_compatible(prompt: str) -> str:
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


def answer_question(question: str) -> dict:
    settings.require_inference()
    contexts = search(question, top_k=settings.top_k)
    prompt = build_prompt(question, contexts)
    provider = settings.inference_provider.lower()
    model_name = settings.openrouter_model if provider == "openrouter" else settings.inference_model
    logger.info(
        "inference_start provider=%s model=%s question_chars=%s contexts=%s",
        provider,
        model_name,
        len(question),
        len(contexts),
    )
    if provider == "openrouter":
        answer = _answer_with_openrouter(prompt)
    elif provider == "zo_ask":
        answer = _answer_with_zo_ask(prompt)
    elif provider == "openai_compatible":
        answer = _answer_with_openai_compatible(prompt)
    else:
        raise RuntimeError("INFERENCE_PROVIDER must be openrouter, zo_ask, or openai_compatible")
    logger.info("inference_done provider=%s model=%s answer_chars=%s", provider, model_name, len(answer))
    return {"answer": answer, "contexts": contexts}
