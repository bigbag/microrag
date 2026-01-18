"""Tests for BM25Index."""

import pytest

from microrag.query_processor import QueryProcessor
from microrag.search.bm25 import BM25Index


class TestBM25Index:
    """Tests for BM25Index class."""

    @pytest.fixture
    def bm25_index(self, query_processor: QueryProcessor) -> BM25Index:
        """Create BM25 index."""
        return BM25Index(query_processor)

    def test_search_returns_relevant_results(self, bm25_index: BM25Index):
        """Test that search returns relevant results."""
        doc_ids = ["doc1", "doc2", "doc3"]
        documents = [
            "Machine learning is powerful for data analysis",
            "Deep learning uses neural networks for recognition",
            "Natural language processing understands text",
        ]

        bm25_index.build(doc_ids, documents)
        results = bm25_index.search("machine learning", top_k=3)

        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_search_edge_cases(self, bm25_index: BM25Index):
        """Test search edge cases."""
        bm25_index.build(["doc1"], ["Machine learning"])

        # Empty query
        assert bm25_index.search("", top_k=3) == []
        # No matches
        assert bm25_index.search("xyznonexistent", top_k=3) == []
        # Search before build
        empty_index = BM25Index(bm25_index._query_processor)
        assert empty_index.search("query", top_k=3) == []

    def test_build_validation(self, bm25_index: BM25Index):
        """Test that mismatched doc_ids and documents raises error."""
        with pytest.raises(ValueError, match="same length"):
            bm25_index.build(["doc1", "doc2"], ["only one doc"])

    def test_abbreviation_expansion_in_search(self, query_processor: QueryProcessor):
        """Test that abbreviations are expanded during search."""
        bm25_index = BM25Index(query_processor)
        # Need 4+ docs for BM25 IDF to work correctly
        doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        documents = [
            "Machine learning is a field of AI",
            "Data science uses statistics",
            "Programming involves writing code",
            "Databases store structured data",
        ]

        bm25_index.build(doc_ids, documents)
        results = bm25_index.search("ML", top_k=2)

        assert len(results) > 0
        assert results[0][0] == "doc1"
