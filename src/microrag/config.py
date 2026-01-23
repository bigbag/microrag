"""Configuration for MicroRAG."""

from dataclasses import dataclass, field
from typing import Any

from microrag.stopwords import ENGLISH_STOPWORDS


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for MicroRAG instance.

    Attributes:
        model_path: Model path (sentence-transformers) or model name (fastembed).
        embedding_backend: Embedding backend ("auto", "sentence-transformers", "fastembed").
        model_file: ONNX model filename within model_path (sentence-transformers only).
        fastembed_cache_dir: Cache directory for fastembed models.
        db_path: DuckDB database path. Use ":memory:" for in-memory database.
        embedding_dim: Dimension of embedding vectors.
        chunk_size: Maximum size of text chunks in characters.
        chunk_overlap: Overlap between consecutive chunks.
        hybrid_enabled: Enable hybrid search (semantic + BM25 + FTS).
        hybrid_alpha: Weight for semantic search in fusion (0-1).
        similarity_threshold: Minimum score threshold for results.
        abbreviations: Mapping of abbreviations to expansions for query processing.
        stopwords: Set of stopwords for BM25 tokenization.
        remove_stopwords: Enable stopword removal in BM25 tokenization.
        hnsw_ef_construction: HNSW index build-time parameter.
        hnsw_ef_search: HNSW index search-time parameter.
        hnsw_m: HNSW index M parameter (connections per layer).
        batch_size: Batch size for embedding generation.
    """

    model_path: str = ""
    embedding_backend: str = "auto"
    model_file: str | None = None
    fastembed_cache_dir: str | None = None
    db_path: str = ":memory:"
    embedding_dim: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200
    hybrid_enabled: bool = True
    hybrid_alpha: float = 0.7
    similarity_threshold: float = 0.4
    abbreviations: dict[str, str] = field(default_factory=dict)
    stopwords: frozenset[str] = field(default_factory=lambda: ENGLISH_STOPWORDS)
    remove_stopwords: bool = True
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    hnsw_m: int = 16
    batch_size: int = 32

    def __post_init__(self) -> None:
        valid_backends = ("auto", "sentence-transformers", "fastembed")
        if self.embedding_backend not in valid_backends:
            raise ValueError(f"embedding_backend must be one of {valid_backends}")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if not 0 <= self.hybrid_alpha <= 1:
            raise ValueError("hybrid_alpha must be between 0 and 1")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

    def with_updates(self, **kwargs: Any) -> "RAGConfig":
        """Create a new config with updated values."""
        current: dict[str, Any] = {
            "model_path": self.model_path,
            "embedding_backend": self.embedding_backend,
            "model_file": self.model_file,
            "fastembed_cache_dir": self.fastembed_cache_dir,
            "db_path": self.db_path,
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "hybrid_enabled": self.hybrid_enabled,
            "hybrid_alpha": self.hybrid_alpha,
            "similarity_threshold": self.similarity_threshold,
            "abbreviations": dict(self.abbreviations),
            "stopwords": self.stopwords,
            "remove_stopwords": self.remove_stopwords,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "hnsw_m": self.hnsw_m,
            "batch_size": self.batch_size,
        }
        current.update(kwargs)
        return RAGConfig(**current)
