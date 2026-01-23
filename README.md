# MicroRAG

[![CI](https://github.com/bigbag/microrag/workflows/CI/badge.svg)](https://github.com/bigbag/microrag/actions?query=workflow%3ACI)
[![pypi](https://img.shields.io/pypi/v/microrag.svg)](https://pypi.python.org/pypi/microrag)
[![downloads](https://img.shields.io/pypi/dm/microrag.svg)](https://pypistats.org/packages/microrag)
[![versions](https://img.shields.io/pypi/pyversions/microrag.svg)](https://github.com/bigbag/microrag)
[![license](https://img.shields.io/github/license/bigbag/microrag.svg)](https://github.com/bigbag/microrag/blob/master/LICENSE)

A feature-rich, universal RAG library for Python with ONNX-backed embeddings and DuckDB storage.

## Features

- **Flexible embedding backends** - Choose between [sentence-transformers](https://sbert.net/) (ONNX-optimized) or [FastEmbed](https://github.com/qdrant/fastembed) (lightweight)
- **[DuckDB](https://duckdb.org) storage** - Persistent vector storage with HNSW indexes for fast similarity search
- **Three-tier hybrid search** - Combines semantic, BM25, and full-text search with RRF fusion
- **Query preprocessing** - Abbreviation expansion and stopword removal for better search
- **Flexible document input** - Accept strings, dicts, or Document objects
- **Text chunking** - Automatic chunking with sentence boundary detection

### Why ONNX?

MicroRAG uses [ONNX](https://onnx.ai) (Open Neural Network Exchange) format for embedding models:

- **Faster inference** - ONNX Runtime provides optimized CPU execution, often 2-3x faster than PyTorch
- **Smaller footprint** - No need for full PyTorch/TensorFlow installation in production
- **Cross-platform** - Same model runs on any platform without framework dependencies
- **Quantization support** - Easy to use INT8/FP16 quantized models for even faster inference

## Installation

```bash
# Core (no embedding backend - bring your own)
pip install microrag

# With sentence-transformers backend (ONNX-optimized)
pip install microrag[sentence-transformers]

# With FastEmbed backend (lightweight, fast)
pip install microrag[fastembed]

# All backends
pip install microrag[all]

# For CPU-only PyTorch (with sentence-transformers)
pip install microrag[sentence-transformers,cpu]
```

## Quick Start

### With sentence-transformers (local model)

```python
from microrag import MicroRAG, RAGConfig

config = RAGConfig(
    model_path="/path/to/all-MiniLM-L6-v2",
    embedding_backend="sentence-transformers",  # or "auto"
    db_path="./rag.duckdb",
    embedding_dim=384,
)

with MicroRAG(config) as rag:
    # Add documents (strings, dicts, or Document objects)
    rag.add_documents([
        "Machine learning is a subset of artificial intelligence.",
        {"content": "Deep learning uses neural networks.", "metadata": {"source": "wiki"}},
    ])

    # Build search indexes
    rag.build_index()

    # Search
    results = rag.search("neural networks", top_k=5)
    for r in results:
        print(f"{r.score:.3f}: {r.content}")
```

### With FastEmbed (auto-download)

```python
from microrag import MicroRAG, RAGConfig

config = RAGConfig(
    model_path="BAAI/bge-small-en-v1.5",  # Model name, auto-downloaded
    embedding_backend="fastembed",
)

with MicroRAG(config) as rag:
    rag.add_documents(["Machine learning is a subset of AI."])
    rag.build_index()
    results = rag.search("neural networks")
```

## Search Pipeline

MicroRAG uses a three-tier hybrid search architecture that combines multiple retrieval methods for better results:

```
Query: "ML techniques"
         │
         ▼
┌─────────────────────────────────────┐
│      Query Preprocessing            │
│  • Normalize whitespace             │
│  • Expand abbreviations (ML→machine │
│    learning)                        │
│  • Tokenize for BM25                │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Parallel Search                │
│                                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐
│  │ Semantic │  │  BM25    │  │    FTS     │
│  │  Search  │  │  Search  │  │   Search   │
│  │ (Vector) │  │(Keywords)│  │ (Stemmed)  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘
│       │             │              │
│       ▼             ▼              ▼
│    Results       Results        Results
│   + scores      + scores       + scores
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    Reciprocal Rank Fusion (RRF)     │
│                                     │
│  score = Σ 1/(k + rank_i)           │
│                                     │
│  Combines rankings from all methods │
│  with configurable weighting        │
└─────────────────────────────────────┘
         │
         ▼
      Final ranked results
```

### Search Components

- **Semantic** - HNSW vector similarity; understands meaning and context
- **BM25** - Term frequency scoring; exact keyword matching
- **FTS** - DuckDB full-text search; stemming and linguistic matching

### Why Hybrid Search?

Each search method has different strengths:

- **Semantic search** finds conceptually similar documents even with different wording
- **BM25** excels at finding exact keyword matches
- **FTS** handles word variations through stemming

By combining all three with RRF fusion, MicroRAG achieves better recall and precision than any single method alone.

## Configuration

```python
from microrag import RAGConfig

config = RAGConfig(
    # Embedding
    model_path="/path/to/model",      # Model path or name
    embedding_backend="auto",         # "auto", "sentence-transformers", "fastembed"

    # Storage
    db_path=":memory:",               # DuckDB path (":memory:" for in-memory)
    embedding_dim=384,                # Embedding vector dimension

    # Chunking
    chunk_size=1000,                  # Max characters per chunk
    chunk_overlap=200,                # Overlap between chunks

    # Search
    hybrid_enabled=True,              # Enable hybrid search
    hybrid_alpha=0.7,                 # Semantic weight (0-1)
    similarity_threshold=0.4,         # Min score threshold

    # Query processing
    abbreviations={"ML": "machine learning"},  # Query expansion
    remove_stopwords=True,            # Remove stopwords for BM25

    # HNSW tuning
    hnsw_ef_construction=200,         # Build-time parameter
    hnsw_ef_search=100,               # Search-time parameter
    hnsw_enable_persistence=False,    # Experimental index persistence
)
```

### Configuration Options

**Embedding:**
- `model_path` (str) - Model path (sentence-transformers) or model name (fastembed)
- `embedding_backend` (str, default: "auto") - Backend: "auto", "sentence-transformers", "fastembed"
- `model_file` (str, default: None) - ONNX filename (sentence-transformers only)
- `fastembed_cache_dir` (str, default: None) - Cache directory (fastembed only)

**Storage:**
- `db_path` (str, default: `:memory:`) - DuckDB database path
- `embedding_dim` (int, default: 384) - Embedding vector dimension

**Chunking:**
- `chunk_size` (int, default: 1000) - Text chunking size in characters
- `chunk_overlap` (int, default: 200) - Overlap between chunks

**Search:**
- `hybrid_enabled` (bool, default: True) - Enable hybrid search
- `hybrid_alpha` (float, default: 0.7) - Semantic weight in fusion (0-1)
- `similarity_threshold` (float, default: 0.4) - Minimum score to return

**Query Processing:**
- `abbreviations` (dict, default: None) - Query expansion mapping
- `stopwords` (set, default: English) - Stopwords for BM25 tokenization
- `remove_stopwords` (bool, default: True) - Enable stopword removal

**HNSW Tuning:**
- `hnsw_ef_construction` (int, default: 200) - HNSW build parameter
- `hnsw_ef_search` (int, default: 100) - HNSW search parameter
- `hnsw_enable_persistence` (bool, default: False) - Enable experimental HNSW index persistence

## API Reference

### MicroRAG

Main class for RAG operations.

```python
from microrag import MicroRAG, RAGConfig

config = RAGConfig(model_path="/path/to/model")

# Use as context manager (recommended)
with MicroRAG(config) as rag:
    rag.add_documents([...])
    rag.build_index()
    results = rag.search("query")

# Or manage lifecycle manually
rag = MicroRAG(config)
try:
    # ... use rag
finally:
    rag.close()
```

**Methods:**

- `add_documents(docs, chunk=True)` - Add documents (str, dict, or Document)
- `build_index()` - Build HNSW, BM25, and FTS indexes
- `search(query, top_k=10, threshold=None, hybrid=None)` - Search documents
- `get_document(doc_id)` - Get document by ID
- `get_all_documents()` - Get all documents
- `count()` - Get document count
- `clear()` - Remove all documents
- `close()` - Close resources

### Document

Document data model.

```python
from microrag import Document

doc = Document(
    id="doc1",                    # Optional, auto-generated if not provided
    content="Document text...",   # Required
    metadata={"source": "wiki"},  # Optional metadata
)
```

### SearchResult

Search result with score and document data.

```python
results = rag.search("query")

for result in results:
    print(result.score)      # Similarity score
    print(result.content)    # Document content
    print(result.metadata)   # Document metadata
    print(result.document)   # Full Document object
```

## Adding Documents

MicroRAG accepts documents in multiple formats:

```python
# Strings
rag.add_documents([
    "First document content",
    "Second document content",
])

# Dicts with metadata
rag.add_documents([
    {"content": "Document text", "metadata": {"source": "file.txt"}},
    {"id": "custom_id", "content": "Another document"},
])

# Document objects
from microrag import Document

rag.add_documents([
    Document(id="doc1", content="Text", metadata={"key": "value"}),
])

# Disable chunking for pre-chunked content
rag.add_documents(["Already chunked text"], chunk=False)
```

## Examples

See the `examples/` directory for complete working examples:

- **[basic_usage.py](examples/basic_usage.py)** - Core workflow: adding documents, building indexes, searching
- **[advanced_config.py](examples/advanced_config.py)** - Custom abbreviations, hybrid search tuning, config variants
- **[faq_search.py](examples/faq_search.py)** - FAQ/knowledge base search with metadata filtering

Run examples with:

```bash
make example name=basic_usage
make example name=advanced_config
make example name=faq_search
```

## Development

```bash
# Clone and install
git clone https://github.com/yourname/microrag.git
cd microrag
uv sync --group dev

# Run tests
uv run pytest

# Run linting
uv run ruff check src/ tests/
uv run mypy src/

# Format code
uv run ruff format src/ tests/
```

## License

MIT License - see [LICENSE](LICENSE) file.
