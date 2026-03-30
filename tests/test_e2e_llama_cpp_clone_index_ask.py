import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from fuzz_coder.ask import app as ask_app
from fuzz_coder.index import app as index_app


class _DeterministicEmbeddingBackend:
    def __init__(self, dim=24):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            b = t.encode("utf-8", errors="ignore")
            for j in range(self.dim):
                out[i, j] = float(b[j % len(b)] if b else 0) / 255.0
        return out


class _DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return np.array([0.2 for _ in pairs], dtype="float32")


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.network
def test_e2e_llama_cpp_clone_index_and_ask(tmp_path, monkeypatch, capsys):
    if os.getenv("RUN_LLAMA_CPP_E2E") != "1":
        pytest.skip("Set RUN_LLAMA_CPP_E2E=1 to run slow network e2e test")

    if shutil.which("git") is None:
        pytest.skip("git is not available")

    repo = tmp_path / "llama.cpp"
    clone_cmd = [
        "git", "clone", "--depth", "1",
        "https://github.com/ggml-org/llama.cpp.git",
        str(repo),
    ]
    proc = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        pytest.skip(f"git clone failed: {proc.stderr[:300]}")

    out = tmp_path / "index_data"

    # Index llama.cpp with deterministic backend for repeatability and no external model downloads.
    monkeypatch.setattr(index_app, "get_embedding_backend", lambda *a, **k: _DeterministicEmbeddingBackend(dim=24))
    monkeypatch.setattr(sys, "argv", [
        "index_fuzz_coder.py",
        "--src", str(repo),
        "--out", str(out),
        "--language", "c_cpp",
        "--embedding_backend", "sentence_transformers",
    ])
    index_app.main()

    meta_path = out / "meta.jsonl"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_count = sum(1 for _ in f)
    assert meta_count > 300

    idx = ask_app.faiss.read_index(str(out / "semantic.faiss"))
    dim = getattr(idx, "d", getattr(idx, "dim", 24))

    # Run ask-cycle with mocked LLM but real indexed data.
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DeterministicEmbeddingBackend(dim=dim))
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)

    captured_prompt = {}

    def _fake_post(_url, json=None, timeout=None):
        captured_prompt["prompt"] = json["prompt"]
        return _Resp({"response": "LLM_E2E_OK"})

    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Напиши список функций пригодных для фаззинга из директории src",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
        "--index_dir", str(out),
        "--model", "dummy-model",
        "--verbose",
    ])

    ask_app.main()

    output = capsys.readouterr().out
    assert "LLM_E2E_OK" in output
    assert "Path filters: ['src']" in output

    prompt = captured_prompt["prompt"].replace("\\", "/")
    assert "/src/" in prompt
    assert "/examples/" not in prompt
