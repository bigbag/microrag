"""MicroRAG exception hierarchy."""


class MicroRAGError(Exception):
    """Base exception for MicroRAG errors."""


class ConfigurationError(MicroRAGError):
    """Raised when configuration is invalid."""


class EmbeddingError(MicroRAGError):
    """Raised when embedding generation fails."""


class StorageError(MicroRAGError):
    """Raised when storage operations fail."""


class IndexError(MicroRAGError):
    """Raised when index operations fail."""


class SearchError(MicroRAGError):
    """Raised when search operations fail."""


class DocumentError(MicroRAGError):
    """Raised when document operations fail."""
