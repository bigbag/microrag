"""DuckDB storage adapter implementation."""

import contextlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
from numpy.typing import NDArray

from microrag.exceptions import StorageError
from microrag.models import Document
from microrag.storage.base import IStorageAdapter

logger = logging.getLogger(__name__)


class DuckDBStorage(IStorageAdapter):
    """DuckDB-based storage adapter with HNSW and FTS support.

    This implementation uses DuckDB's VSS extension for HNSW vector indexing
    and the FTS extension for full-text search.

    Args:
        db_path: Path to DuckDB database file. Use ":memory:" for in-memory.
        embedding_dim: Dimension of embedding vectors.
        hnsw_enable_persistence: Enable experimental HNSW index persistence.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_dim: int = 384,
        hnsw_enable_persistence: bool = False,
    ) -> None:
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._hnsw_enable_persistence = hnsw_enable_persistence
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._vector_index_built = False
        self._fts_index_built = False
        logger.debug(
            "DuckDBStorage initialized: db_path=%s, embedding_dim=%d",
            db_path,
            embedding_dim,
        )

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is not None:
            return self._conn

        try:
            # Create directory if needed for file-based DB
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

            logger.debug("Connecting to DuckDB: %s", self._db_path)
            self._conn = duckdb.connect(self._db_path)
            self._init_schema()
            logger.info("DuckDB connection established: %s", self._db_path)
            return self._conn
        except Exception as e:
            raise StorageError(f"Failed to connect to DuckDB: {e}") from e

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the database connection."""
        return self._get_connection()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._conn
        if conn is None:
            return

        logger.debug("Initializing DuckDB schema and extensions")
        # Install and load required extensions
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")

        # Enable experimental HNSW persistence if requested
        if self._hnsw_enable_persistence:
            conn.execute("SET hnsw_enable_experimental_persistence = true")

        # Create documents table
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSON,
                embedding FLOAT[{self._embedding_dim}]
            )
        """)

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to storage."""
        if not documents:
            return

        logger.debug("Storing %d document(s) in DuckDB", len(documents))
        conn = self.conn
        try:
            for doc in documents:
                if doc.embedding is None:
                    raise StorageError(f"Document {doc.id} has no embedding")

                embedding_list = doc.embedding.tolist()
                metadata_json = json.dumps(doc.metadata)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents (id, content, metadata, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    [doc.id, doc.content, metadata_json, embedding_list],
                )

            # Invalidate indexes after adding documents
            self._vector_index_built = False
            self._fts_index_built = False
            logger.debug("Documents stored successfully")

        except Exception as e:
            raise StorageError(f"Failed to add documents: {e}") from e

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID."""
        try:
            result = self.conn.execute(
                "SELECT id, content, metadata, embedding FROM documents WHERE id = ?",
                [doc_id],
            ).fetchone()

            if result is None:
                return None

            doc_id, content, metadata_json, embedding = result
            metadata = json.loads(metadata_json) if metadata_json else {}
            embedding_array = np.array(embedding, dtype=np.float32) if embedding else None

            return Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                embedding=embedding_array,
            )
        except Exception as e:
            raise StorageError(f"Failed to get document: {e}") from e

    def get_all_documents(self) -> list[Document]:
        """Retrieve all documents from storage."""
        try:
            results = self.conn.execute(
                "SELECT id, content, metadata, embedding FROM documents"
            ).fetchall()

            documents = []
            for doc_id, content, metadata_json, embedding in results:
                metadata = json.loads(metadata_json) if metadata_json else {}
                embedding_array = np.array(embedding, dtype=np.float32) if embedding else None
                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        embedding=embedding_array,
                    )
                )
            return documents
        except Exception as e:
            raise StorageError(f"Failed to get all documents: {e}") from e

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            result = self.conn.execute(
                "DELETE FROM documents WHERE id = ? RETURNING id",
                [doc_id],
            ).fetchone()
            return result is not None
        except Exception as e:
            raise StorageError(f"Failed to delete document: {e}") from e

    def clear(self) -> None:
        """Remove all documents from storage."""
        try:
            self.conn.execute("DELETE FROM documents")
            self._vector_index_built = False
            self._fts_index_built = False
        except Exception as e:
            raise StorageError(f"Failed to clear documents: {e}") from e

    def count(self) -> int:
        """Get the number of documents in storage."""
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return result[0] if result else 0
        except Exception as e:
            raise StorageError(f"Failed to count documents: {e}") from e

    def build_vector_index(
        self,
        ef_construction: int = 200,
        ef_search: int = 100,
        m: int = 16,
    ) -> None:
        """Build HNSW vector index for similarity search."""
        try:
            logger.info(
                "Building HNSW vector index: ef_construction=%d, ef_search=%d, m=%d",
                ef_construction,
                ef_search,
                m,
            )
            conn = self.conn

            # Drop existing index if any
            conn.execute("DROP INDEX IF EXISTS documents_embedding_idx")

            # Create HNSW index
            conn.execute(f"""
                CREATE INDEX documents_embedding_idx ON documents
                USING HNSW (embedding)
                WITH (
                    metric = 'cosine',
                    ef_construction = {ef_construction},
                    ef_search = {ef_search},
                    m = {m}
                )
            """)

            self._vector_index_built = True
            logger.debug("HNSW vector index built")
        except Exception as e:
            raise StorageError(f"Failed to build vector index: {e}") from e

    def build_fts_index(self) -> None:
        """Build full-text search index."""
        try:
            logger.info("Building FTS index")
            conn = self.conn

            # Create FTS index using PRAGMA
            conn.execute("""
                PRAGMA create_fts_index(
                    'documents',
                    'id',
                    'content',
                    stemmer = 'porter',
                    stopwords = 'english',
                    ignore = '(\\.|[^a-z]+)',
                    strip_accents = 1,
                    lower = 1
                )
            """)

            self._fts_index_built = True
            logger.debug("FTS index built")
        except Exception as e:
            # FTS index might already exist - that's ok
            if "already exists" not in str(e).lower():
                raise StorageError(f"Failed to build FTS index: {e}") from e
            logger.debug("FTS index already exists")
            self._fts_index_built = True

    def vector_search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search by vector similarity using HNSW index."""
        try:
            logger.debug("Performing vector search, top_k=%d", top_k)
            embedding_list = query_embedding.tolist()

            # Use array_cosine_similarity for similarity search
            # Note: embedding dimension must be in SQL directly, not parameterized
            results = self.conn.execute(
                f"""
                SELECT id, array_cosine_similarity(embedding, ?::FLOAT[{self._embedding_dim}]) as score
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY score DESC
                LIMIT ?
                """,
                [embedding_list, top_k],
            ).fetchall()

            result_list = [(row[0], float(row[1])) for row in results]
            logger.debug("Vector search returned %d result(s)", len(result_list))
            return result_list
        except Exception as e:
            raise StorageError(f"Failed to perform vector search: {e}") from e

    def fts_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search by full-text search."""
        if not query.strip():
            return []

        try:
            logger.debug("Performing FTS search: query=%r, top_k=%d", query, top_k)
            results = self.conn.execute(
                """
                SELECT id, fts_main_documents.match_bm25(id, ?) as score
                FROM documents
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT ?
                """,
                [query, top_k],
            ).fetchall()

            result_list = [(row[0], float(row[1])) for row in results]
            logger.debug("FTS search returned %d result(s)", len(result_list))
            return result_list
        except Exception as e:
            # FTS might not be built or query might be empty after stemming
            if "no such function" in str(e).lower() or "fts" in str(e).lower():
                logger.debug("FTS search unavailable or returned no results")
                return []
            raise StorageError(f"Failed to perform FTS search: {e}") from e

    def close(self) -> None:
        """Close storage connection and release resources."""
        logger.debug("Closing DuckDB connection")
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
