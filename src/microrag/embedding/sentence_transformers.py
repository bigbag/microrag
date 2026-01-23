"""Sentence-transformers embedding backend with ONNX support."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from microrag.embedding.base import IEmbeddingModel
from microrag.exceptions import EmbeddingError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _import_sentence_transformers() -> Any:
    """Import sentence-transformers with helpful error message."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install it with: pip install microrag[sentence-transformers]"
        ) from e


class SentenceTransformerModel(IEmbeddingModel):
    """Embedding model using sentence-transformers with ONNX backend.

    This class wraps sentence-transformers to provide ONNX-optimized
    embeddings for CPU inference.

    Args:
        model_path: Path to the sentence-transformer model directory.
        model_file: Optional ONNX model filename for quantized models.
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        model_path: str,
        model_file: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not model_path:
            raise ValueError("model_path is required for sentence-transformers backend")
        self._model_path = Path(model_path)
        self._model_file = model_file
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> "SentenceTransformer":
        """Load the sentence-transformer model with ONNX backend."""
        if self._model is not None:
            return self._model

        SentenceTransformer = _import_sentence_transformers()
        model_path = str(self._model_path)

        try:
            model_kwargs: dict[str, dict[str, str]] = {}

            if self._model_file:
                model_kwargs["model_kwargs"] = {"file_name": self._model_file}

            self._model = SentenceTransformer(
                model_path,
                backend="onnx",
                **model_kwargs,
            )
            return self._model
        except Exception as e:
            raise EmbeddingError(f"Failed to load model from {model_path}: {e}") from e

    @property
    def model(self) -> "SentenceTransformer":
        """Get the loaded model, loading it if necessary."""
        return self._load_model()

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension() or 384

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
            embeddings = self.model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            result: NDArray[np.float32] = np.asarray(embeddings, dtype=np.float32)
            return result
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
