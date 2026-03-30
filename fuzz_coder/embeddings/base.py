from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np


class EmbeddingBackend(ABC):
    """Abstract embedding backend for pluggable vectorization."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
