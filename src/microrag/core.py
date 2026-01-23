"""Core MicroRAG class."""

import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any

from microrag.config import RAGConfig
from microrag.embedding import IEmbeddingModel, create_embedding_model
from microrag.exceptions import DocumentError, MicroRAGError
from microrag.models import Document, SearchResult
from microrag.query_processor import QueryProcessor
from microrag.search.hybrid import HybridSearcher
from microrag.storage import DuckDBStorage
from microrag.utils import chunk_text, generate_id, normalize_document_input

logger = logging.getLogger(__name__)


class MicroRAG:
    """Main MicroRAG class for document indexing and retrieval.

    MicroRAG provides a simple API for:
    - Adding documents with automatic embedding generation
    - Building HNSW, BM25, and FTS indexes
    - Hybrid search with RRF fusion

    Example:
        ```python
        config = RAGConfig(
            model_path="/path/to/model",
            db_path="./rag.duckdb",
        )

        with MicroRAG(config) as rag:
            rag.add_documents(["Doc 1", "Doc 2"])
            rag.build_index()
            results = rag.search("query")
        ```

    Args:
        config: RAGConfig instance with all settings.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._storage: DuckDBStorage | None = None
        self._embedding_model: IEmbeddingModel | None = None
        self._query_processor: QueryProcessor | None = None
        self._searcher: HybridSearcher | None = None
        self._documents: list[Document] = []
        self._index_built = False
        logger.debug("MicroRAG initialized with db_path=%s", config.db_path)

    def __enter__(self) -> "MicroRAG":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and close resources."""
        self.close()

    @property
    def config(self) -> RAGConfig:
        """Get the configuration."""
        return self._config

    @property
    def storage(self) -> DuckDBStorage:
        """Get or create the storage adapter."""
        if self._storage is None:
            self._storage = DuckDBStorage(
                db_path=self._config.db_path,
                embedding_dim=self._config.embedding_dim,
                hnsw_enable_persistence=self._config.hnsw_enable_persistence,
            )
        return self._storage

    @property
    def embedding_model(self) -> IEmbeddingModel:
        """Get or create the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = create_embedding_model(
                backend=self._config.embedding_backend,
                model_path=self._config.model_path,
                model_file=self._config.model_file,
                batch_size=self._config.batch_size,
                cache_dir=self._config.fastembed_cache_dir,
            )
        return self._embedding_model

    @property
    def query_processor(self) -> QueryProcessor:
        """Get or create the query processor."""
        if self._query_processor is None:
            self._query_processor = QueryProcessor(
                abbreviations=dict(self._config.abbreviations),
                stopwords=self._config.stopwords,
                remove_stopwords=self._config.remove_stopwords,
            )
        return self._query_processor

    @property
    def searcher(self) -> HybridSearcher:
        """Get or create the hybrid searcher."""
        if self._searcher is None:
            self._searcher = HybridSearcher(
                storage=self.storage,
                embedding_model=self.embedding_model,
                query_processor=self.query_processor,
                alpha=self._config.hybrid_alpha,
            )
        return self._searcher

    def add_documents(
        self,
        documents: Sequence[str | dict[str, Any] | Document],
        chunk: bool = True,
    ) -> list[str]:
        """Add documents to the RAG system.

        Documents can be provided in various formats:
        - str: Plain text content
        - dict: {"content": str, "metadata": dict, "id": str}
        - Document: Document dataclass

        Large documents are automatically chunked if chunk=True.

        Args:
            documents: Sequence of documents in any supported format.
            chunk: Whether to chunk large documents.

        Returns:
            List of document IDs that were added.
        """
        doc_ids = []
        logger.info("Adding %d document(s), chunk=%s", len(documents), chunk)

        for doc in documents:
            try:
                doc_id, content, metadata = normalize_document_input(doc)
            except ValueError as e:
                raise DocumentError(str(e)) from e

            # Chunk if needed
            if chunk and len(content) > self._config.chunk_size:
                chunks = chunk_text(
                    content,
                    chunk_size=self._config.chunk_size,
                    chunk_overlap=self._config.chunk_overlap,
                )

                for i, chunk_content in enumerate(chunks):
                    chunk_id = doc_id or generate_id(chunk_content)
                    if len(chunks) > 1:
                        chunk_id = f"{chunk_id}_{i}"

                    chunk_metadata = {**metadata, "_chunk_index": i, "_total_chunks": len(chunks)}

                    doc_obj = Document(
                        id=chunk_id,
                        content=chunk_content,
                        metadata=chunk_metadata,
                    )
                    self._documents.append(doc_obj)
                    doc_ids.append(chunk_id)
            else:
                final_id = doc_id or generate_id(content)
                doc_obj = Document(
                    id=final_id,
                    content=content,
                    metadata=metadata,
                )
                self._documents.append(doc_obj)
                doc_ids.append(final_id)

        self._index_built = False
        logger.debug("Added %d document(s): %s", len(doc_ids), doc_ids[:5])
        return doc_ids

    def build_index(self) -> None:
        """Build all indexes (HNSW, BM25, FTS).

        This method:
        1. Generates embeddings for all documents
        2. Stores documents in DuckDB
        3. Builds HNSW vector index
        4. Builds BM25 index
        5. Builds FTS index

        Must be called after add_documents() and before search().
        """
        if not self._documents:
            logger.debug("No documents to index")
            return

        logger.info("Building index for %d document(s)", len(self._documents))

        # Generate embeddings for documents without them
        docs_needing_embeddings = [d for d in self._documents if d.embedding is None]

        if docs_needing_embeddings:
            logger.debug("Generating embeddings for %d document(s)", len(docs_needing_embeddings))
            contents = [d.content for d in docs_needing_embeddings]
            embeddings = self.embedding_model.encode(contents)

            for doc, embedding in zip(docs_needing_embeddings, embeddings, strict=True):
                doc.embedding = embedding

        # Store documents
        self.storage.add_documents(self._documents)

        # Build HNSW index
        self.storage.build_vector_index(
            ef_construction=self._config.hnsw_ef_construction,
            ef_search=self._config.hnsw_ef_search,
            m=self._config.hnsw_m,
        )

        # Build FTS index
        self.storage.build_fts_index()

        # Build BM25 index
        self.searcher.build_index(self._documents)

        self._index_built = True
        logger.info("Index build complete")

    def search(
        self,
        query: str,
        top_k: int = 10,
        hybrid: bool | None = None,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for documents matching the query.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            hybrid: Override hybrid search setting from config.
            threshold: Override similarity threshold from config.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        if not self._index_built:
            raise MicroRAGError("Index not built. Call build_index() first.")

        logger.info("Searching for query=%r, top_k=%d, hybrid=%s", query, top_k, hybrid)
        hybrid_enabled = hybrid if hybrid is not None else self._config.hybrid_enabled
        similarity_threshold = (
            threshold if threshold is not None else self._config.similarity_threshold
        )

        results = self.searcher.search(
            query=query,
            top_k=top_k,
            hybrid_enabled=hybrid_enabled,
            similarity_threshold=similarity_threshold,
        )
        logger.debug("Search returned %d result(s)", len(results))
        return results

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Document if found, None otherwise.
        """
        return self.storage.get_document(doc_id)

    def get_all_documents(self) -> list[Document]:
        """Retrieve all documents from storage.

        Returns:
            List of all documents.
        """
        return self.storage.get_all_documents()

    def count(self) -> int:
        """Get the number of documents in storage.

        Returns:
            Document count.
        """
        return self.storage.count()

    def clear(self) -> None:
        """Remove all documents and reset indexes."""
        self.storage.clear()
        self._documents.clear()
        self._index_built = False

    def close(self) -> None:
        """Close all resources."""
        logger.debug("Closing MicroRAG resources")
        if self._storage is not None:
            self._storage.close()
            self._storage = None
        self._embedding_model = None
        self._query_processor = None
        self._searcher = None
