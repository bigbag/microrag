"""MicroRAG - A feature-rich, universal RAG library for Python.

MicroRAG provides:
- ONNX-backed embeddings (CPU-only, no PyTorch at runtime)
- DuckDB storage with HNSW vector indexes
- Three-tier hybrid search (semantic + BM25 + FTS) with RRF fusion
- Query preprocessing with abbreviation expansion

Example:
    ```python
    from microrag import MicroRAG, RAGConfig

    config = RAGConfig(
        model_path="/path/to/all-MiniLM-L6-v2",
        db_path="./rag.duckdb",
    )

    with MicroRAG(config) as rag:
        rag.add_documents(["Document 1", "Document 2"])
        rag.build_index()
        results = rag.search("query")
    ```
"""

from microrag.config import RAGConfig
from microrag.core import MicroRAG
from microrag.exceptions import (
    ConfigurationError,
    DocumentError,
    EmbeddingError,
    IndexError,
    MicroRAGError,
    SearchError,
    StorageError,
)
from microrag.models import Document, SearchResult

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "MicroRAG",
    "RAGConfig",
    # Data models
    "Document",
    "SearchResult",
    # Exceptions
    "MicroRAGError",
    "ConfigurationError",
    "DocumentError",
    "EmbeddingError",
    "IndexError",
    "SearchError",
    "StorageError",
    # Version
    "__version__",
]
