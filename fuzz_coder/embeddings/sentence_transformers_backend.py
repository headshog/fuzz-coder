from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingBackend


class SentenceTransformersBackend(EmbeddingBackend):
    """Default embedding backend used by existing scripts."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"
