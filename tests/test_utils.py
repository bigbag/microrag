"""Tests for utility functions."""

import pytest

from microrag.models import Document
from microrag.utils import chunk_text, generate_id, normalize_document_input


class TestGenerateId:
    """Tests for generate_id function."""

    def test_generate_deterministic_id(self):
        """Test generating deterministic ID from content."""
        content = "Test content"
        id1 = generate_id(content)
        id2 = generate_id(content)
        assert id1 == id2
        assert len(id1) == 16

        # Different content produces different IDs
        id3 = generate_id("Other content")
        assert id1 != id3


class TestChunkText:
    """Tests for chunk_text function."""

    def test_short_text_no_chunking(self):
        """Test that short text is not chunked."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert chunks == [text]

    def test_empty_text(self):
        """Test empty text returns empty list."""
        chunks = chunk_text("", chunk_size=100, chunk_overlap=20)
        assert chunks == []

    def test_long_text_chunked(self):
        """Test that long text is chunked properly."""
        text = "A" * 300
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        for chunk in chunks[:-1]:
            assert len(chunk) <= 100

    def test_sentence_boundary_preference(self):
        """Test that chunking prefers sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. " * 10
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

        for chunk in chunks[:-1]:
            stripped = chunk.rstrip()
            assert stripped.endswith(".")


class TestNormalizeDocumentInput:
    """Tests for normalize_document_input function."""

    def test_various_input_formats(self):
        """Test normalizing different input formats."""
        # String input
        doc_id, content, metadata = normalize_document_input("Test content")
        assert doc_id is None
        assert content == "Test content"
        assert metadata == {}

        # Dict input
        doc_id, content, metadata = normalize_document_input(
            {
                "id": "test1",
                "content": "Test content",
                "metadata": {"source": "test"},
            }
        )
        assert doc_id == "test1"
        assert content == "Test content"
        assert metadata == {"source": "test"}

        # Document input
        doc = Document(id="test1", content="Test content", metadata={"source": "test"})
        doc_id, content, metadata = normalize_document_input(doc)
        assert doc_id == "test1"

    def test_invalid_input_raises_error(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="must have 'content'"):
            normalize_document_input({"metadata": {}})

        with pytest.raises(ValueError, match="Invalid document type"):
            normalize_document_input(123)  # type: ignore
