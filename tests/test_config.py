"""Tests for RAGConfig."""

import pytest

from microrag.config import RAGConfig


class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_create_config(self):
        """Test creating config with various fields."""
        config = RAGConfig(
            model_path="/path/to/model",
            model_file="model.onnx",
            db_path="./test.duckdb",
            embedding_dim=768,
            abbreviations={"ML": "machine learning"},
        )
        assert config.model_path == "/path/to/model"
        assert config.model_file == "model.onnx"
        assert config.embedding_dim == 768
        assert config.abbreviations == {"ML": "machine learning"}

    def test_empty_model_path_raises_error(self):
        """Test that empty model_path raises ValueError."""
        with pytest.raises(ValueError, match="model_path is required"):
            RAGConfig(model_path="")

    def test_invalid_embedding_dim_raises_error(self):
        """Test that non-positive embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            RAGConfig(model_path="/path", embedding_dim=0)

    def test_invalid_chunk_config_raises_error(self):
        """Test chunk configuration validation."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RAGConfig(model_path="/path", chunk_size=-1)

        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            RAGConfig(model_path="/path", chunk_overlap=-1)

        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            RAGConfig(model_path="/path", chunk_size=100, chunk_overlap=100)

    def test_invalid_alpha_and_threshold_raises_error(self):
        """Test alpha and threshold validation."""
        with pytest.raises(ValueError, match="hybrid_alpha must be between 0 and 1"):
            RAGConfig(model_path="/path", hybrid_alpha=1.5)

        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            RAGConfig(model_path="/path", similarity_threshold=-0.1)

    def test_with_updates(self):
        """Test creating new config with updates."""
        config = RAGConfig(model_path="/path/to/model", embedding_dim=384)
        new_config = config.with_updates(embedding_dim=768, db_path="./new.duckdb")

        assert config.embedding_dim == 384  # Original unchanged
        assert new_config.embedding_dim == 768
        assert new_config.db_path == "./new.duckdb"
