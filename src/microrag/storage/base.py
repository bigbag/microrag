"""Abstract storage adapter interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from microrag.models import Document


class IStorageAdapter(ABC):
    """Abstract interface for storage adapters.

    This interface defines the contract for document storage and retrieval
    with support for vector similarity search and full-text search.
    """

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to storage.

        Args:
            documents: Sequence of documents with embeddings.
        """

    @abstractmethod
    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Document if found, None otherwise.
        """

    @abstractmethod
    def get_all_documents(self) -> list[Document]:
        """Retrieve all documents from storage.

        Returns:
            List of all documents.
        """

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            True if document was deleted, False if not found.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from storage."""

    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in storage.

        Returns:
            Document count.
        """

    @abstractmethod
    def build_vector_index(
        self,
        ef_construction: int = 200,
        ef_search: int = 100,
        m: int = 16,
    ) -> None:
        """Build HNSW vector index for similarity search.

        Args:
            ef_construction: HNSW build-time parameter.
            ef_search: HNSW search-time parameter.
            m: HNSW M parameter (connections per layer).
        """

    @abstractmethod
    def build_fts_index(self) -> None:
        """Build full-text search index."""

    @abstractmethod
    def vector_search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search by vector similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum number of results.

        Returns:
            List of (doc_id, score) tuples sorted by descending similarity.
        """

    @abstractmethod
    def fts_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search by full-text search.

        Args:
            query: Search query string.
            top_k: Maximum number of results.

        Returns:
            List of (doc_id, score) tuples sorted by descending relevance.
        """

    @abstractmethod
    def close(self) -> None:
        """Close storage connection and release resources."""
