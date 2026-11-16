"""Integration tests for MicroRAG."""

import numpy as np
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
        assert len(results) == 2
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

    def test_document_operations(self, rag_with_mock_embeddings: MicroRAG, mock_embeddings):
        """Test document retrieval and management."""
        rag_with_mock_embeddings.add_documents(
            [
                Document(id="test1", content="Test content", embedding=mock_embeddings[0]),
                "Another document",
            ]
        )
        rag_with_mock_embeddings.build_index()

        assert rag_with_mock_embeddings.count() == 2
        stored = rag_with_mock_embeddings.get_document("test1")
        assert stored is not None
        np.testing.assert_array_equal(stored.embedding, mock_embeddings[0])
        assert len(rag_with_mock_embeddings.get_all_documents()) == 2

        rag_with_mock_embeddings.clear()
        assert rag_with_mock_embeddings.count() == 0

    @pytest.mark.parametrize("hybrid", [False, True])
    def test_search_stopword_only_documents(self, rag_with_mock_embeddings: MicroRAG, hybrid):
        rag_with_mock_embeddings.add_documents(["the and or"])
        rag_with_mock_embeddings.build_index()

        results = rag_with_mock_embeddings.search("the and or", hybrid=hybrid)
        assert [result.content for result in results] == ["the and or"]

    @pytest.mark.parametrize("reuse_instance", [False, True])
    def test_reopen_file_database(
        self, rag_config: RAGConfig, mock_embedding_model, tmp_path, reuse_instance: bool
    ):
        """Test that a reopened file database is searchable after build_index()."""
        config = rag_config.with_updates(db_path=str(tmp_path / "rag.duckdb"))

        with MicroRAG(config, embedding_model=mock_embedding_model) as rag:
            doc_ids = rag.add_documents(
                ["Machine learning is great", "Cooking pasta at home", "Growing garden flowers"]
            )
            rag.build_index()

        if not reuse_instance:
            rag = MicroRAG(config, embedding_model=mock_embedding_model)

        with rag:
            rag.build_index()
            results = rag.search("pasta")
            assert [doc_id for doc_id, _ in rag.searcher.bm25_search("pasta")] == [doc_ids[1]]

        assert results[0].content == "Cooking pasta at home"

    def test_abbreviation_expansion_in_search(self, rag_config: RAGConfig, mock_embedding_model):
        """Test that abbreviations are expanded during search."""
        config = rag_config.with_updates(abbreviations={"ML": "machine learning"})

        with MicroRAG(config, embedding_model=mock_embedding_model) as rag:
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
