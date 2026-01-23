"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class IEmbeddingModel(ABC):
    """Abstract interface for embedding models.

    All embedding backends must implement this interface to be compatible
    with MicroRAG.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        ...

    @abstractmethod
    def encode(
        self,
        texts: str | Sequence[str],
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """Encode texts into embeddings.

        Args:
            texts: Single text or sequence of texts to encode.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            Array of shape (n_texts, embedding_dim) with float32 embeddings.
        """
        ...

    @abstractmethod
    def encode_single(
        self,
        text: str,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """Encode a single text into an embedding.

        Args:
            text: Text to encode.
            normalize: Whether to L2-normalize the embedding.

        Returns:
            Array of shape (embedding_dim,) with float32 embedding.
        """
        ...
