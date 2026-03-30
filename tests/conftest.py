import sys
import types

import numpy as np


# Provide lightweight stubs so tests can run without heavy ML deps installed.
if "sentence_transformers" not in sys.modules:
    st = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, convert_to_numpy=True, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            # deterministic tiny embeddings
            arr = np.array([[float(len(t) % 7 + 1), 1.0, 0.5] for t in texts], dtype="float32")
            return arr

    class _CrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return np.array([0.1 for _ in pairs], dtype="float32")

    st.SentenceTransformer = _SentenceTransformer
    st.CrossEncoder = _CrossEncoder
    sys.modules["sentence_transformers"] = st

if "faiss" not in sys.modules:
    faiss_mod = types.ModuleType("faiss")
    _INDEX_REGISTRY = {}

    class _FakeIndexFlatIP:
        def __init__(self, dim):
            self.dim = dim
            self.d = dim
            self.vecs = None

        def add(self, vecs):
            self.vecs = np.array(vecs, dtype="float32")

        def search(self, emb, k):
            q = np.array(emb, dtype="float32")
            if self.vecs is None or len(self.vecs) == 0:
                scores = np.zeros((q.shape[0], k), dtype="float32")
                ids = -np.ones((q.shape[0], k), dtype="int64")
                return scores, ids

            sim = q @ self.vecs.T
            top_idx = np.argsort(-sim, axis=1)[:, :k]
            top_scores = np.take_along_axis(sim, top_idx, axis=1).astype("float32")
            return top_scores, top_idx.astype("int64")

    def _read_index(path):
        if path not in _INDEX_REGISTRY:
            raise FileNotFoundError(path)
        return _INDEX_REGISTRY[path]

    def _write_index(index, path):
        _INDEX_REGISTRY[path] = index
        with open(path, "wb") as f:
            f.write(b"FAKE_FAISS_INDEX")

    faiss_mod.IndexFlatIP = _FakeIndexFlatIP
    faiss_mod.read_index = _read_index
    faiss_mod.write_index = _write_index
    sys.modules["faiss"] = faiss_mod
