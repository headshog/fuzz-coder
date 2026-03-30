from __future__ import annotations

from .base import EmbeddingBackend
from .sentence_transformers_backend import SentenceTransformersBackend


def get_embedding_backend(model_name: str, backend: str = "sentence_transformers") -> EmbeddingBackend:
    """Factory for embedding backends. Keeps default behavior unchanged."""
    if backend == "sentence_transformers":
        return SentenceTransformersBackend(model_name)
    raise ValueError(f"Unsupported embedding backend: {backend}")
