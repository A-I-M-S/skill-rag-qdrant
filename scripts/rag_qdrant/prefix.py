from __future__ import annotations

import re
from typing import Literal

EMBED_PREFIX = "embed"
# Match "embed" at start, case-insensitive, followed by whitespace or end of string.
# Punctuation like "embed!" is intentionally NOT a prefix - it must be a clean word followed
# by whitespace (or just the bare word).
EMBED_PREFIX_RE = re.compile(r"^embed(?=\s|$)", re.IGNORECASE)


def parse_prefix(text: str) -> tuple[Literal["embed", "query"], str]:
    match = EMBED_PREFIX_RE.match(text)
    if not match:
        return "query", text
    body = text[match.end():]
    body = body.lstrip(" \t\n\r")
    return "embed", body
