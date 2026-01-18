"""Hybrid search with RRF fusion."""

from collections.abc import Sequence

from microrag.embedding import EmbeddingModel
from microrag.models import Document, SearchResult
from microrag.query_processor import QueryProcessor
from microrag.search.bm25 import BM25Index
from microrag.storage.base import IStorageAdapter


class HybridSearcher:
    """Hybrid searcher combining semantic, BM25, and FTS search with RRF fusion.

    This class implements three-tier hybrid search:
    - Semantic search using vector similarity (HNSW index)
    - BM25 search for keyword matching
    - FTS search using DuckDB's full-text search

    Results are fused using Reciprocal Rank Fusion (RRF).

    Args:
        storage: Storage adapter for document retrieval and vector/FTS search.
        embedding_model: Embedding model for query encoding.
        query_processor: Query processor for text processing.
        alpha: Weight for semantic search (0-1). BM25 and FTS share remaining weight.
        rrf_k: RRF constant (default 60 per original paper).
    """

    def __init__(
        self,
        storage: IStorageAdapter,
        embedding_model: EmbeddingModel,
        query_processor: QueryProcessor,
        alpha: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        self._storage = storage
        self._embedding_model = embedding_model
        self._query_processor = query_processor
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._bm25_index = BM25Index(query_processor)

    def build_index(self, documents: Sequence[Document]) -> None:
        """Build the BM25 index from documents.

        Args:
            documents: Sequence of documents to index.
        """
        doc_ids = [doc.id for doc in documents]
        doc_contents = [doc.content for doc in documents]
        self._bm25_index.build(doc_ids, doc_contents)

    def search(
        self,
        query: str,
        top_k: int = 10,
        hybrid_enabled: bool = True,
        similarity_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Perform hybrid search.

        Args:
            query: Search query string.
            top_k: Maximum number of results.
            hybrid_enabled: If True, use hybrid search. If False, semantic only.
            similarity_threshold: Minimum score threshold for results.

        Returns:
            List of SearchResult objects sorted by descending relevance.
        """
        # Process query
        processed_query = self._query_processor.process(query)

        # Get query embedding
        query_embedding = self._embedding_model.encode_single(processed_query)

        # Semantic search
        semantic_results = self._storage.vector_search(query_embedding, top_k=top_k * 2)

        if not hybrid_enabled:
            return self._to_search_results(semantic_results, top_k, similarity_threshold)

        # BM25 search
        bm25_results = self._bm25_index.search(processed_query, top_k=top_k * 2)

        # FTS search
        fts_results = self._storage.fts_search(processed_query, top_k=top_k * 2)

        # Fuse results using RRF
        fused_scores = self._rrf_fusion(
            semantic_results,
            bm25_results,
            fts_results,
        )

        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        return self._to_search_results(sorted_results[:top_k], top_k, similarity_threshold)

    def _rrf_fusion(
        self,
        semantic_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        fts_results: list[tuple[str, float]],
    ) -> dict[str, float]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each ranking list.
        Weights are applied: semantic gets alpha, BM25 and FTS share (1-alpha).

        Args:
            semantic_results: Results from semantic search.
            bm25_results: Results from BM25 search.
            fts_results: Results from FTS search.

        Returns:
            Dictionary mapping doc_id to fused score.
        """
        fused: dict[str, float] = {}
        k = self._rrf_k

        # Weight distribution
        semantic_weight = self._alpha
        keyword_weight = (1 - self._alpha) / 2  # Split between BM25 and FTS

        # Semantic contribution
        for rank, (doc_id, _) in enumerate(semantic_results, 1):
            rrf_score = semantic_weight / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0) + rrf_score

        # BM25 contribution
        for rank, (doc_id, _) in enumerate(bm25_results, 1):
            rrf_score = keyword_weight / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0) + rrf_score

        # FTS contribution
        for rank, (doc_id, _) in enumerate(fts_results, 1):
            rrf_score = keyword_weight / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0) + rrf_score

        return fused

    def _to_search_results(
        self,
        results: list[tuple[str, float]] | Sequence[tuple[str, float]],
        top_k: int,
        threshold: float,
    ) -> list[SearchResult]:
        """Convert (doc_id, score) tuples to SearchResult objects.

        Args:
            results: List of (doc_id, score) tuples.
            top_k: Maximum number of results.
            threshold: Minimum score threshold.

        Returns:
            List of SearchResult objects.
        """
        search_results = []
        for rank, (doc_id, score) in enumerate(results, 1):
            if score < threshold:
                continue

            doc = self._storage.get_document(doc_id)
            if doc is None:
                continue

            search_results.append(
                SearchResult(
                    document=doc,
                    score=score,
                    rank=rank,
                )
            )

            if len(search_results) >= top_k:
                break

        return search_results

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Perform semantic-only search.

        Args:
            query: Search query string.
            top_k: Maximum number of results.

        Returns:
            List of (doc_id, score) tuples.
        """
        processed_query = self._query_processor.process(query)
        query_embedding = self._embedding_model.encode_single(processed_query)
        return self._storage.vector_search(query_embedding, top_k=top_k)

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Perform BM25-only search.

        Args:
            query: Search query string.
            top_k: Maximum number of results.

        Returns:
            List of (doc_id, score) tuples.
        """
        processed_query = self._query_processor.process(query)
        return self._bm25_index.search(processed_query, top_k=top_k)

    def fts_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Perform FTS-only search.

        Args:
            query: Search query string.
            top_k: Maximum number of results.

        Returns:
            List of (doc_id, score) tuples.
        """
        processed_query = self._query_processor.process(query)
        return self._storage.fts_search(processed_query, top_k=top_k)
