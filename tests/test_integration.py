"""Integration tests for MicroRAG."""

import pytest

from microrag import Document, MicroRAG, RAGConfig, SearchResult
from microrag.exceptions import MicroRAGError


class TestMicroRAGIntegration:
    """Integration tests for MicroRAG class."""

    def test_add_documents_various_formats(self, rag_with_mock_embeddings: MicroRAG):
        """Test adding documents in various formats."""
        doc_ids = rag_with_mock_embeddings.add_documents(
            [
                "String document",
                {"content": "Dict document", "id": "custom_id"},
                Document(id="obj_doc", content="Object document"),
            ]
        )

        assert len(doc_ids) == 3
        assert "custom_id" in doc_ids
        assert "obj_doc" in doc_ids

    def test_chunking(self, rag_with_mock_embeddings: MicroRAG):
        """Test document chunking."""
        large_content = "word " * 500

        # With chunking
        doc_ids = rag_with_mock_embeddings.add_documents([large_content])
        assert len(doc_ids) > 1

        # Without chunking
        rag_with_mock_embeddings._documents.clear()
        doc_ids = rag_with_mock_embeddings.add_documents([large_content], chunk=False)
        assert len(doc_ids) == 1

    def test_search_workflow(self, rag_with_mock_embeddings: MicroRAG):
        """Test the full search workflow."""
        # Search before index fails
        rag_with_mock_embeddings.add_documents(["Test document"])
        with pytest.raises(MicroRAGError, match="Index not built"):
            rag_with_mock_embeddings.search("query")

        # Build and search
        rag_with_mock_embeddings.add_documents(
            [
                "Machine learning is great",
                "Deep learning is powerful",
            ]
        )
        rag_with_mock_embeddings.build_index()

        results = rag_with_mock_embeddings.search("machine learning", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_options(self, rag_with_mock_embeddings: MicroRAG):
        """Test search with various options."""
        rag_with_mock_embeddings.add_documents(["Machine learning document"])
        rag_with_mock_embeddings.build_index()

        # Hybrid disabled
        results = rag_with_mock_embeddings.search("machine", hybrid=False)
        assert isinstance(results, list)

        # High threshold filters results
        results = rag_with_mock_embeddings.search("query", threshold=0.99)
        for r in results:
            assert r.score >= 0.99

    def test_document_operations(self, rag_with_mock_embeddings: MicroRAG):
        """Test document retrieval and management."""
        rag_with_mock_embeddings.add_documents(
            [
                Document(id="test1", content="Test content"),
                "Another document",
            ]
        )
        rag_with_mock_embeddings.build_index()

        assert rag_with_mock_embeddings.count() == 2
        assert rag_with_mock_embeddings.get_document("test1") is not None
        assert len(rag_with_mock_embeddings.get_all_documents()) == 2

        rag_with_mock_embeddings.clear()
        assert rag_with_mock_embeddings.count() == 0

    def test_abbreviation_expansion_in_search(self, rag_config: RAGConfig, mock_embedding_model):
        """Test that abbreviations are expanded during search."""
        config = rag_config.with_updates(abbreviations={"ML": "machine learning"})

        with MicroRAG(config) as rag:
            rag._embedding_model = mock_embedding_model
            rag.add_documents(
                [
                    "Machine learning is a field of artificial intelligence",
                    "Data science uses statistics and programming",
                    "Database systems store structured information",
                    "Web development involves creating websites",
                ]
            )
            rag.build_index()

            results = rag.search("ML techniques", threshold=0.0)
            assert len(results) > 0
            assert any("machine learning" in r.content.lower() for r in results)
