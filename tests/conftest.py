"""Test fixtures for MicroRAG."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from microrag import MicroRAG, RAGConfig
from microrag.models import Document
from microrag.query_processor import QueryProcessor
from microrag.storage import DuckDBStorage

# Test constants
TEST_EMBEDDING_DIM = 384
FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def embedding_dim() -> int:
    """Return test embedding dimension."""
    return TEST_EMBEDDING_DIM


@pytest.fixture
def mock_embeddings() -> np.ndarray:
    """Generate mock embeddings for testing."""
    np.random.seed(42)
    return np.random.rand(10, TEST_EMBEDDING_DIM).astype(np.float32)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    np.random.seed(42)
    docs = [
        Document(
            id="doc1",
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "wiki"},
            embedding=np.random.rand(TEST_EMBEDDING_DIM).astype(np.float32),
        ),
        Document(
            id="doc2",
            content="Deep learning uses neural networks with many layers.",
            metadata={"source": "textbook"},
            embedding=np.random.rand(TEST_EMBEDDING_DIM).astype(np.float32),
        ),
        Document(
            id="doc3",
            content="Natural language processing enables computers to understand text.",
            metadata={"source": "article"},
            embedding=np.random.rand(TEST_EMBEDDING_DIM).astype(np.float32),
        ),
        Document(
            id="doc4",
            content="Python is a popular programming language for data science.",
            metadata={"source": "blog"},
            embedding=np.random.rand(TEST_EMBEDDING_DIM).astype(np.float32),
        ),
        Document(
            id="doc5",
            content="Transformers have revolutionized NLP and computer vision.",
            metadata={"source": "paper"},
            embedding=np.random.rand(TEST_EMBEDDING_DIM).astype(np.float32),
        ),
    ]
    return docs


@pytest.fixture
def storage() -> DuckDBStorage:
    """Create in-memory DuckDB storage."""
    return DuckDBStorage(db_path=":memory:", embedding_dim=TEST_EMBEDDING_DIM)


@pytest.fixture
def storage_with_docs(storage: DuckDBStorage, sample_documents: list[Document]) -> DuckDBStorage:
    """Create storage pre-populated with sample documents."""
    storage.add_documents(sample_documents)
    storage.build_vector_index()
    storage.build_fts_index()
    return storage


@pytest.fixture
def query_processor() -> QueryProcessor:
    """Create query processor with test abbreviations."""
    return QueryProcessor(
        abbreviations={"ML": "machine learning", "NLP": "natural language processing"},
        remove_stopwords=True,
    )


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    mock = MagicMock()
    mock.embedding_dim = TEST_EMBEDDING_DIM

    def encode_side_effect(texts, normalize=True):
        np.random.seed(hash(str(texts)) % 2**32)
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), TEST_EMBEDDING_DIM).astype(np.float32)

    def encode_single_side_effect(text, normalize=True):
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(TEST_EMBEDDING_DIM).astype(np.float32)

    mock.encode.side_effect = encode_side_effect
    mock.encode_single.side_effect = encode_single_side_effect

    return mock


@pytest.fixture
def rag_config() -> RAGConfig:
    """Create test RAG config with in-memory DB."""
    return RAGConfig(
        model_path="/fake/model/path",
        db_path=":memory:",
        embedding_dim=TEST_EMBEDDING_DIM,
        abbreviations={"ML": "machine learning"},
    )


@pytest.fixture
def rag_with_mock_embeddings(rag_config: RAGConfig, mock_embedding_model):
    """Create MicroRAG instance with mocked embedding model."""
    rag = MicroRAG(rag_config)
    rag._embedding_model = mock_embedding_model
    return rag
