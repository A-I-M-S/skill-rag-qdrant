"""Photo storage and on-disk dedupe.

A :class:`Photo` is a user-supplied (filename, bytes, description)
triple. The bytes go to disk at a content-addressed path so identical
uploads dedupe automatically; the description is what the inference
model embeds and searches. The actual embedding and Qdrant write
happen in :mod:`rag_qdrant.qdrant_store` (see
:func:`rag_qdrant.qdrant_store.ingest_photo`); this module is purely
the disk-side half.

The path layout is::

    <settings.photos_dir>/<sha256[:16]>.<ext>

where ``<ext>`` is the lowercase original extension (with the dot),
or empty when the original filename has no extension. The
``target.exists()`` short-circuit at write time is what makes
idempotent re-ingest a no-op on disk; the Qdrant point ID dedupe is a
separate concern handled by the existing :func:`_point_id` hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .logging_setup import logger

#: Recognized photo extensions. The skill never decodes the bytes, so
#: this is a permissive set: anything an image-typed transport might
#: send. An unknown / missing extension is still accepted (the file
#: is stored without an extension) — see :func:`save_photo`.
SUPPORTED_PHOTO_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic", ".heif"}
)

#: Length of the sha256 prefix used in the on-disk filename. Short
#: enough to be human-readable, long enough that collisions in a
#: single corpus are vanishingly unlikely.
_PHOTO_FILE_HASH_LEN: int = 16

#: Length of the sha256 prefix used in the Qdrant ``source`` field.
#: Kept in sync with the existing ``SOURCE_HASH_LEN`` for text sources
#: so all source identifiers have a similar shape.
_PHOTO_SOURCE_HASH_LEN: int = 12

#: Source-namespace prefix for photo points. Visible in Qdrant
#: payloads and in the agent-handler ingest ack.
PHOTO_SOURCE_NAMESPACE: str = "photo"


@dataclass(frozen=True)
class Photo:
    """A single photo attached to an agent message.

    Attributes:
        filename: Original filename (e.g. ``"sunset.jpg"``). Used to
            pick the on-disk extension and stored in the Qdrant
            payload as ``photo_filename``.
        content: Raw image bytes. Stored verbatim on disk; the skill
            does not decode them.
        description: Required user-supplied description of the photo.
            This is the only signal that gets embedded into the
            vector index, so a future query matches the photo by
            description similarity. Empty / whitespace-only
            descriptions are rejected by :func:`save_photo`.
    """

    filename: str
    content: bytes
    description: str


def _photo_target_path(sha256_hex: str, filename: str) -> Path:
    """Build the on-disk path for a photo, content-addressed by sha256."""
    suffix = Path(filename).suffix.lower()
    return settings.photos_dir / f"{sha256_hex[:_PHOTO_FILE_HASH_LEN]}{suffix}"


def save_photo(photo: Photo) -> tuple[Path, str, str]:
    """Write ``photo`` to disk (idempotent) and return the metadata.

    Returns ``(target_path, sha256_hex, source)`` where ``source`` is
    the Qdrant source identifier ``photo-<sha256[:12]>`` used for the
    matching point.

    Raises:
        ValueError: if the description is empty/whitespace, or if
            ``photo.filename`` has an extension that is not in
            :data:`SUPPORTED_PHOTO_SUFFIXES`. Missing extensions are
            accepted (the file is stored without one).
    """
    description = (photo.description or "").strip()
    if not description:
        raise ValueError(
            "Photo description is required and must be non-empty. "
            "Describe the photo so the description can be embedded and searched."
        )

    suffix = Path(photo.filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_PHOTO_SUFFIXES:
        raise ValueError(
            f"Unsupported photo type: {suffix!r}. "
            f"Supported: {sorted(SUPPORTED_PHOTO_SUFFIXES)} "
            f"(or no extension)."
        )

    sha256_hex = hashlib.sha256(photo.content).hexdigest()
    target = _photo_target_path(sha256_hex, photo.filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Idempotent write: identical bytes (same sha256) never overwrite
    # a file with different content, because the path is keyed on
    # the content hash. We still skip the write when the file
    # already exists to save the I/O and keep the atime/mtime quiet.
    if not target.exists():
        target.write_bytes(photo.content)
        logger.info("photo_store_wrote path=%s bytes=%s sha256=%s", target, len(photo.content), sha256_hex)
    else:
        logger.info("photo_store_dedup path=%s bytes=%s sha256=%s", target, len(photo.content), sha256_hex)

    source = f"{PHOTO_SOURCE_NAMESPACE}-{sha256_hex[:_PHOTO_SOURCE_HASH_LEN]}"
    return target, sha256_hex, source
