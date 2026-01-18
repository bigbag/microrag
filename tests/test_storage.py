"""Tests for DuckDBStorage."""

import numpy as np
import pytest

from microrag.exceptions import StorageError
from microrag.models import Document
from microrag.storage import DuckDBStorage


class TestDuckDBStorage:
    """Tests for DuckDBStorage class."""

    def test_add_and_retrieve_documents(
        self, storage: DuckDBStorage, sample_documents: list[Document]
    ):
        """Test adding and retrieving documents."""
        storage.add_documents(sample_documents)
        assert storage.count() == len(sample_documents)

        doc = storage.get_document("doc1")
        assert doc is not None
        assert doc.id == "doc1"
        assert doc.content == sample_documents[0].content

        all_docs = storage.get_all_documents()
        assert len(all_docs) == len(sample_documents)

    def test_add_document_without_embedding_raises_error(self, storage: DuckDBStorage):
        """Test that adding document without embedding raises error."""
        doc = Document(id="test", content="Test content")
        with pytest.raises(StorageError, match="no embedding"):
            storage.add_documents([doc])

    def test_delete_and_clear(self, storage: DuckDBStorage, sample_documents: list[Document]):
        """Test deleting and clearing documents."""
        storage.add_documents(sample_documents)

        assert storage.delete_document("doc1") is True
        assert storage.get_document("doc1") is None
        assert storage.delete_document("nonexistent") is False

        storage.clear()
        assert storage.count() == 0

    def test_upsert_document(self, storage: DuckDBStorage, embedding_dim: int):
        """Test that adding document with same ID updates it."""
        embedding = np.random.rand(embedding_dim).astype(np.float32)
        doc1 = Document(id="test", content="Original content", embedding=embedding)
        storage.add_documents([doc1])

        doc2 = Document(id="test", content="Updated content", embedding=embedding)
        storage.add_documents([doc2])

        assert storage.count() == 1
        retrieved = storage.get_document("test")
        assert retrieved is not None
        assert retrieved.content == "Updated content"

    def test_vector_search(self, storage_with_docs: DuckDBStorage, embedding_dim: int):
        """Test vector similarity search."""
        query_embedding = np.random.rand(embedding_dim).astype(np.float32)
        results = storage_with_docs.vector_search(query_embedding, top_k=3)

        assert len(results) <= 3
        if len(results) > 1:
            scores = [r[1] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_fts_search(self, storage_with_docs: DuckDBStorage):
        """Test full-text search."""
        results = storage_with_docs.fts_search("machine learning", top_k=3)
        assert isinstance(results, list)

        # Empty query returns empty
        assert storage_with_docs.fts_search("", top_k=3) == []
