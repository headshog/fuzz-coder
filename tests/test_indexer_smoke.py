import json
import sys

import numpy as np

from fuzz_coder.index import app as index_app


class _DeterministicEmbeddingBackend:
    def __init__(self, dim=16):
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


def test_indexer_smoke_local_project(tmp_path, monkeypatch):
    src = tmp_path / "proj"
    (src / "src" / "parsers").mkdir(parents=True, exist_ok=True)
    (src / "src" / "io").mkdir(parents=True, exist_ok=True)

    (src / "src" / "parsers" / "parser.cpp").write_text(
        """
        #include <string>
        bool parse_header(const std::string & s) { return !s.empty(); }
        int decode_buf(const unsigned char * data, size_t n) { return (int)n; }
        """,
        encoding="utf-8",
    )
    (src / "src" / "io" / "writer.cpp").write_text(
        """
        #include <cstdio>
        void write_log(const char * s) { printf("%s", s); }
        """,
        encoding="utf-8",
    )

    out = tmp_path / "index_data"

    monkeypatch.setattr(index_app, "get_embedding_backend", lambda *a, **k: _DeterministicEmbeddingBackend(dim=8))
    monkeypatch.setattr(sys, "argv", [
        "index_fuzz_coder.py",
        "--src", str(src),
        "--out", str(out),
        "--language", "c_cpp",
        "--embedding_backend", "sentence_transformers",
    ])

    index_app.main()

    expected = [
        out / "semantic.faiss",
        out / "meta.jsonl",
        out / "lexical_index.json",
        out / "special_indices.json",
        out / "symbols.json",
        out / "call_graph.json",
        out / "called_by.json",
    ]
    for p in expected:
        assert p.exists(), f"missing artifact: {p}"

    with open(out / "meta.jsonl", "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert rows
    assert any("src/parsers" in r["file"].replace("\\", "/") for r in rows)

    special = json.loads((out / "special_indices.json").read_text(encoding="utf-8"))
    for key in ["stdin", "file_input", "api_calls", "output", "memory_management", "error_handling", "by_type"]:
        assert key in special


def test_indexer_smoke_java_project(tmp_path, monkeypatch):
    src = tmp_path / "proj_java"
    (src / "src" / "parser").mkdir(parents=True, exist_ok=True)

    (src / "src" / "parser" / "HeaderParser.java").write_text(
        """
        package parser;
        import java.io.BufferedReader;
        import java.io.FileReader;

        public class HeaderParser {
            public boolean parseHeader(String line) {
                return line != null && !line.isEmpty();
            }

            public static byte[] readBytes(String path) throws Exception {
                try (BufferedReader br = new BufferedReader(new FileReader(path))) {
                    String line = br.readLine();
                    return line == null ? new byte[0] : line.getBytes();
                }
            }
        }
        """,
        encoding="utf-8",
    )

    out = tmp_path / "index_data_java"

    monkeypatch.setattr(index_app, "get_embedding_backend", lambda *a, **k: _DeterministicEmbeddingBackend(dim=8))
    monkeypatch.setattr(sys, "argv", [
        "index_fuzz_coder.py",
        "--src", str(src),
        "--out", str(out),
        "--language", "java",
        "--embedding_backend", "sentence_transformers",
    ])

    index_app.main()

    with open(out / "meta.jsonl", "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert rows
    assert any(r["file"].replace("\\", "/").endswith(".java") for r in rows)
    assert any(r["name"] == "parseHeader" for r in rows)

    special = json.loads((out / "special_indices.json").read_text(encoding="utf-8"))
    assert "by_type" in special
    assert "string" in special["by_type"]
