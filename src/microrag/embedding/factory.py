"""Factory for creating embedding models."""

from microrag.embedding.base import IEmbeddingModel
from microrag.exceptions import ConfigurationError


def _is_sentence_transformers_available() -> bool:
    """Check if sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _is_fastembed_available() -> bool:
    """Check if fastembed is installed."""
    try:
        import fastembed  # noqa: F401

        return True
    except ImportError:
        return False


def get_available_backends() -> list[str]:
    """Get list of available embedding backends.

    Returns:
        List of available backend names.
    """
    backends = []
    if _is_sentence_transformers_available():
        backends.append("sentence-transformers")
    if _is_fastembed_available():
        backends.append("fastembed")
    return backends


def _detect_backend() -> str:
    """Auto-detect the best available backend.

    Priority: sentence-transformers > fastembed

    Returns:
        Name of the detected backend.

    Raises:
        ConfigurationError: If no backend is available.
    """
    if _is_sentence_transformers_available():
        return "sentence-transformers"
    if _is_fastembed_available():
        return "fastembed"

    raise ConfigurationError(
        "No embedding backend available. Install one with:\n"
        "  pip install microrag[sentence-transformers]  # ONNX-optimized\n"
        "  pip install microrag[fastembed]              # Lightweight"
    )


def create_embedding_model(
    backend: str,
    model_path: str,
    model_file: str | None = None,
    batch_size: int = 32,
    cache_dir: str | None = None,
) -> IEmbeddingModel:
    """Create an embedding model with the specified backend.

    Args:
        backend: Backend to use ("auto", "sentence-transformers", "fastembed").
        model_path: Model path (for sentence-transformers) or model name (for fastembed).
        model_file: ONNX model filename (sentence-transformers only).
        batch_size: Batch size for encoding.
        cache_dir: Cache directory (fastembed only).

    Returns:
        Configured embedding model.

    Raises:
        ConfigurationError: If backend is invalid or unavailable.
    """
    if backend == "auto":
        backend = _detect_backend()

    if backend == "sentence-transformers":
        if not _is_sentence_transformers_available():
            raise ConfigurationError(
                "sentence-transformers backend requested but not installed. "
                "Install with: pip install microrag[sentence-transformers]"
            )
        from microrag.embedding.sentence_transformers import SentenceTransformerModel

        return SentenceTransformerModel(
            model_path=model_path,
            model_file=model_file,
            batch_size=batch_size,
        )

    if backend == "fastembed":
        if not _is_fastembed_available():
            raise ConfigurationError(
                "fastembed backend requested but not installed. "
                "Install with: pip install microrag[fastembed]"
            )
        from microrag.embedding.fastembed import FastEmbedModel

        return FastEmbedModel(
            model_name=model_path,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )

    raise ConfigurationError(
        f"Unknown embedding backend: {backend}. "
        f"Available backends: {', '.join(get_available_backends()) or 'none installed'}"
    )
