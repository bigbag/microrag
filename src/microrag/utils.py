"""Utility functions for MicroRAG."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from microrag.models import Document


def generate_id(content: str | None = None) -> str:
    """Generate a unique document ID.

    Args:
        content: Optional content to use for deterministic ID generation.
                 If None, generates a random UUID.

    Returns:
        A unique string ID.
    """
    if content is not None:
        # Generate deterministic ID from content
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    return uuid.uuid4().hex[:16]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks.

    Tries to split at sentence boundaries when possible.

    Args:
        text: Text to split.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    prev_start = -1

    while start < len(text):
        # Calculate end position
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break

        # Try to find a good break point (sentence boundary)
        break_point = _find_break_point(text, start, end)

        chunks.append(text[start:break_point])

        # Move start position accounting for overlap
        new_start = break_point - chunk_overlap

        # Ensure progress - if we're not moving forward, force it
        if new_start <= prev_start:
            new_start = break_point

        prev_start = start
        start = new_start

    return chunks


def _find_break_point(text: str, start: int, end: int) -> int:
    """Find a good break point near the end position.

    Tries to break at sentence boundaries (. ! ?) or paragraph breaks.

    Args:
        text: Full text.
        start: Start position of current chunk.
        end: Desired end position.

    Returns:
        Actual break point position.
    """
    # Look for sentence endings in the last 20% of the chunk
    search_start = start + int((end - start) * 0.8)
    search_region = text[search_start:end]

    # Sentence-ending patterns (with possible closing quotes/parens)
    for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n", "\n\n"]:
        pos = search_region.rfind(sep)
        if pos != -1:
            return search_start + pos + len(sep)

    # Fall back to any whitespace near the end
    for i in range(end - 1, search_start, -1):
        if text[i].isspace():
            return i + 1

    # No good break point found, just use the end
    return end


def normalize_document_input(
    doc: str | dict[str, Any] | Document,
) -> tuple[str | None, str, dict[str, Any]]:
    """Normalize various document input formats.

    Args:
        doc: Document in various formats:
             - str: Plain text content
             - dict: {"content": str, "metadata": dict, "id": str}
             - Document: Document dataclass

    Returns:
        Tuple of (id or None, content, metadata).

    Raises:
        ValueError: If input format is invalid.
    """
    # Import here to avoid circular imports
    from microrag.models import Document

    if isinstance(doc, str):
        return (None, doc, {})

    if isinstance(doc, Document):
        return (doc.id, doc.content, doc.metadata)

    if isinstance(doc, dict):
        content = doc.get("content")
        if not content or not isinstance(content, str):
            raise ValueError("Document dict must have 'content' string key")
        return (
            doc.get("id"),
            content,
            doc.get("metadata", {}),
        )

    raise ValueError(f"Invalid document type: {type(doc)}")
