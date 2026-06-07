from __future__ import annotations

from openai import OpenAI

from .cache import (
    semantic_cache_lookup,
    semantic_cache_store,
)
from .config import settings
from .logging_setup import logger
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


def answer_question(question: str) -> dict:
    settings.require_inference()
    if settings.semantic_cache_enabled:
        query_embedding = embed_texts([question], query=True)[0]
        cached = semantic_cache_lookup(question, query_embedding)
        if cached is not None:
            return cached
    else:
        query_embedding = None

    contexts = search(question, top_k=settings.top_k, query_vector=query_embedding)
    contexts = [item for item in contexts if float(item.get("score") or 0) >= settings.min_relevance_score]
    if not contexts:
        logger.info("inference_no_relevant_context question_chars=%s min_relevance_score=%s", len(question), settings.min_relevance_score)
        result = {"answer": NO_RELEVANT_ANSWER, "contexts": []}
        if settings.semantic_cache_enabled and query_embedding is not None:
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
    result = {"answer": answer, "contexts": contexts}
    if settings.semantic_cache_enabled and query_embedding is not None:
        semantic_cache_store(question, query_embedding, result, is_miss=False)
    return result


ask = answer_question
