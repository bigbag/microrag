"""FastEmbed embedding backend."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from microrag.embedding.base import IEmbeddingModel
from microrag.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastembed import TextEmbedding


def _import_fastembed() -> Any:
    """Import fastembed with helpful error message."""
    try:
        from fastembed import TextEmbedding

        return TextEmbedding
    except ImportError as e:
        raise ImportError(
            "fastembed is not installed. Install it with: pip install microrag[fastembed]"
        ) from e


class FastEmbedModel(IEmbeddingModel):
    """Embedding model using FastEmbed.

    FastEmbed is a lightweight, fast embedding library from Qdrant.
    Models are automatically downloaded and cached.

    Args:
        model_name: Model name (e.g., "BAAI/bge-small-en-v1.5").
        batch_size: Batch size for encoding.
        cache_dir: Optional cache directory for models.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None
        self._embedding_dim: int | None = None

    def _load_model(self) -> "TextEmbedding":
        """Load the FastEmbed model."""
        if self._model is not None:
            return self._model

        TextEmbedding = _import_fastembed()

        try:
            logger.info("Loading FastEmbed model: %s", self._model_name)
            kwargs: dict[str, Any] = {}
            if self._cache_dir:
                logger.debug("Using cache directory: %s", self._cache_dir)
                kwargs["cache_dir"] = self._cache_dir

            self._model = TextEmbedding(
                model_name=self._model_name,
                **kwargs,
            )
            logger.info("FastEmbed model loaded successfully")
            return self._model
        except Exception as e:
            raise EmbeddingError(f"Failed to load FastEmbed model {self._model_name}: {e}") from e

    @property
    def model(self) -> "TextEmbedding":
        """Get the loaded model, loading it if necessary."""
        return self._load_model()

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            test_embedding = list(self.model.embed(["test"]))[0]
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim

    def _normalize(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2-normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        result: NDArray[np.float32] = (embeddings / norms).astype(np.float32)
        return result

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
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        try:
            logger.debug("Encoding %d text(s) with batch_size=%d", len(texts), self._batch_size)
            embeddings_gen = self.model.embed(
                list(texts),
                batch_size=self._batch_size,
            )
            embeddings = np.array(list(embeddings_gen), dtype=np.float32)

            if normalize:
                embeddings = self._normalize(embeddings)

            logger.debug("Encoding complete, shape=%s", embeddings.shape)
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to encode texts: {e}") from e

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
        embeddings = self.encode([text], normalize=normalize)
        result: NDArray[np.float32] = embeddings[0]
        return result
