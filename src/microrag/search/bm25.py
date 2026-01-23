"""BM25 search index."""

import logging
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from microrag.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 search index using rank-bm25.

    This class provides BM25 keyword search with configurable tokenization.

    Args:
        query_processor: QueryProcessor for tokenization.
    """

    def __init__(self, query_processor: QueryProcessor) -> None:
        self._query_processor = query_processor
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    @property
    def is_built(self) -> bool:
        """Check if the index has been built."""
        return self._bm25 is not None

    def build(self, doc_ids: Sequence[str], documents: Sequence[str]) -> None:
        """Build the BM25 index from documents.

        Args:
            doc_ids: Sequence of document IDs (parallel to documents).
            documents: Sequence of document texts.
        """
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must have same length")

        logger.info("Building BM25 index for %d document(s)", len(documents))
        self._doc_ids = list(doc_ids)
        tokenized_docs = self._query_processor.tokenize_documents(documents)

        # Handle empty corpus
        if not tokenized_docs:
            logger.debug("Empty corpus, skipping BM25 index build")
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(tokenized_docs)
        logger.debug("BM25 index built successfully")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the index with a query.

        Args:
            query: Search query string.
            top_k: Maximum number of results.

        Returns:
            List of (doc_id, score) tuples sorted by descending relevance.
        """
        if self._bm25 is None or not self._doc_ids:
            logger.debug("BM25 index not built, returning empty results")
            return []

        query_tokens = self._query_processor.process_for_bm25(query)
        logger.debug("BM25 query tokens: %s", query_tokens)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        scored_indices = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]

        results = [(self._doc_ids[i], score) for i, score in top_indices]
        logger.debug("BM25 search returned %d result(s)", len(results))
        return results

    def clear(self) -> None:
        """Clear the index."""
        self._bm25 = None
        self._doc_ids = []
