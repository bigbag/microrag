"""Tests for data models."""

import pytest

from microrag.models import Document, SearchResult


class TestDocument:
    """Tests for Document dataclass."""

    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Document(id="test1", content="")


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_convenience_accessors(self):
        """Test convenience property accessors."""
        metadata = {"source": "test.txt"}
        doc = Document(id="test1", content="Test content", metadata=metadata)
        result = SearchResult(document=doc, score=0.85, rank=1)

        assert result.id == "test1"
        assert result.content == "Test content"
        assert result.metadata == metadata
