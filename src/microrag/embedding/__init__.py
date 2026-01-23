"""Embedding module for MicroRAG."""

from microrag.embedding.base import IEmbeddingModel
from microrag.embedding.factory import create_embedding_model, get_available_backends

# Backward compatibility alias
EmbeddingModel = IEmbeddingModel

__all__ = [
    "IEmbeddingModel",
    "EmbeddingModel",
    "create_embedding_model",
    "get_available_backends",
]
