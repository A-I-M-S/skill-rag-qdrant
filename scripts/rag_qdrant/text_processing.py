from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .config import settings
from .logging_setup import logger

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def extract_pdf_text(path: Path) -> str:
    logger.info("extract_pdf_start path=%s", path)
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        logger.info("extract_pdf_page path=%s page=%s chars=%s", path, index, len(page_text))
        pages.append(page_text)
    text = normalize_text("\n\n".join(pages))
    logger.info("extract_pdf_done path=%s pages=%s chars=%s", path, len(reader.pages), len(text))
    return text


def extract_text_file(path: Path) -> str:
    logger.info("extract_text_file_start path=%s", path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(text)
    logger.info("extract_text_file_done path=%s chars=%s", path, len(text))
    return text


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix in {".txt", ".md", ".text"}:
        return extract_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix}. Send PDF, TXT, or MD files.")


def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = normalize_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + int(chunk_size * 0.6):
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - chunk_overlap)
    logger.info("chunk_text_done chars=%s chunks=%s chunk_size=%s overlap=%s", text_len, len(chunks), chunk_size, chunk_overlap)
    return chunks
