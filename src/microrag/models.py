"""Data models for MicroRAG."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Document:
    """A document with content, metadata, and optional embedding."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Document content cannot be empty")


@dataclass
class SearchResult:
    """A search result containing document, score, and rank."""

    document: Document
    score: float
    rank: int

    @property
    def id(self) -> str:
        """Convenience accessor for document ID."""
        return self.document.id

    @property
    def content(self) -> str:
        """Convenience accessor for document content."""
        return self.document.content

    @property
    def metadata(self) -> dict[str, Any]:
        """Convenience accessor for document metadata."""
        return self.document.metadata
