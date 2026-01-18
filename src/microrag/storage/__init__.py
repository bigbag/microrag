"""Storage module for MicroRAG."""

from microrag.storage.base import IStorageAdapter
from microrag.storage.duckdb import DuckDBStorage

__all__ = ["IStorageAdapter", "DuckDBStorage"]
