import json
import sys

import numpy as np

from hybrid_code.ask import app as ask_app


class _DummyEmbeddingBackend:
    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[1.0, 0.5, 0.25] for _ in texts], dtype="float32")


class _DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return np.array([0.2 for _ in pairs], dtype="float32")


class _FakeFaissIndex:
    def __init__(self, n_docs):
        self.n_docs = max(1, n_docs)

    def search(self, emb, k):
        ids = np.array([[i % self.n_docs for i in range(k)]], dtype="int64")
        scores = np.ones((1, k), dtype="float32")
        return scores, ids


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_ask_cycle_with_mocked_requests_and_path_filter(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "parse_json_payload",
            "file": "/repo/src/parsers/json_parser.cpp",
            "start_line": 10,
            "end_line": 40,
            "signature": "parse_json_payload(const std::string & s)",
            "parameters": [{"name": "s", "type": "const std::string &", "raw": "const std::string & s"}],
            "code": "bool parse_json_payload(const std::string & s){ parse(s); return true; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": True,
            "has_error_handling": False,
        },
        {
            "id": 1,
            "name": "decode_binary_blob",
            "file": "/repo/src/parsers/binary_decoder.cpp",
            "start_line": 5,
            "end_line": 50,
            "signature": "decode_binary_blob(const uint8_t * buf, size_t n)",
            "parameters": [
                {"name": "buf", "type": "const uint8_t *", "raw": "const uint8_t * buf"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int decode_binary_blob(const uint8_t * buf, size_t n){ memcpy(dst, buf, n); return parse(buf,n); }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": True,
            "has_error_handling": False,
        },
        {
            "id": 2,
            "name": "write_log",
            "file": "/repo/src/io/logger.cpp",
            "start_line": 1,
            "end_line": 20,
            "signature": "write_log(const std::string & s)",
            "parameters": [{"name": "s", "type": "const std::string &", "raw": "const std::string & s"}],
            "code": "void write_log(const std::string & s){ printf(\"%s\", s.c_str()); }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": True,
            "uses_memory_management": False,
            "has_error_handling": False,
        },
    ]

    with open(index_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row) + "\n")

    _write_json(index_dir / "lexical_index.json", {})
    _write_json(index_dir / "special_indices.json", {
        "stdin": [],
        "file_input": [0],
        "api_calls": [],
        "output": [2],
        "memory_management": [0, 1],
        "error_handling": [],
        "by_type": {"string": [0, 2], "byte_array": [1], "integer": [1]},
    })
    _write_json(index_dir / "symbols.json", {"parse_json_payload": [0], "decode_binary_blob": [1], "write_log": [2]})
    _write_json(index_dir / "call_graph.json", {})
    _write_json(index_dir / "called_by.json", {})

    captured_prompt = {}

    def _fake_post(_url, json=None, timeout=None):
        captured_prompt["prompt"] = json["prompt"]
        return _Resp({"response": "MOCKED_LLM_ANSWER"})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("hybrid_code.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Напиши список функций пригодных для фаззинга из директории src/parsers",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_hybrid_code.py",
        "--index_dir", str(index_dir),
        "--model", "dummy-model",
        "--verbose",
    ])

    ask_app.main()

    out = capsys.readouterr().out
    assert "MOCKED_LLM_ANSWER" in out
    assert "Path filters: ['src/parsers']" in out

    prompt = captured_prompt["prompt"]
    assert "/repo/src/parsers/json_parser.cpp" in prompt
    assert "/repo/src/parsers/binary_decoder.cpp" in prompt
    assert "/repo/src/io/logger.cpp" not in prompt


def test_ask_cycle_other_functions_excludes_previously_listed(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "parse_json_payload",
            "file": "/repo/src/parsers/json_parser.cpp",
            "start_line": 10,
            "end_line": 40,
            "signature": "parse_json_payload(const std::string & s)",
            "parameters": [{"name": "s", "type": "const std::string &", "raw": "const std::string & s"}],
            "code": "bool parse_json_payload(const std::string & s){ parse(s); return true; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": True,
            "has_error_handling": False,
        },
        {
            "id": 1,
            "name": "decode_binary_blob",
            "file": "/repo/src/parsers/binary_decoder.cpp",
            "start_line": 5,
            "end_line": 50,
            "signature": "decode_binary_blob(const uint8_t * buf, size_t n)",
            "parameters": [
                {"name": "buf", "type": "const uint8_t *", "raw": "const uint8_t * buf"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int decode_binary_blob(const uint8_t * buf, size_t n){ memcpy(dst, buf, n); return parse(buf,n); }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": True,
            "has_error_handling": False,
        },
        {
            "id": 2,
            "name": "validate_header",
            "file": "/repo/src/parsers/header.cpp",
            "start_line": 1,
            "end_line": 30,
            "signature": "validate_header(const uint8_t * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const uint8_t *", "raw": "const uint8_t * data"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "bool validate_header(const uint8_t * data, size_t n){ if(n < 4) return false; return true; }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": True,
            "has_error_handling": True,
        },
    ]

    with open(index_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row) + "\n")

    _write_json(index_dir / "lexical_index.json", {})
    _write_json(index_dir / "special_indices.json", {
        "stdin": [],
        "file_input": [0],
        "api_calls": [],
        "output": [],
        "memory_management": [0, 1, 2],
        "error_handling": [2],
        "by_type": {"string": [0], "byte_array": [1, 2], "integer": [1, 2]},
    })
    _write_json(index_dir / "symbols.json", {
        "parse_json_payload": [0],
        "decode_binary_blob": [1],
        "validate_header": [2],
    })
    _write_json(index_dir / "call_graph.json", {})
    _write_json(index_dir / "called_by.json", {})

    prompts = []
    llm_responses = iter([
        {"response": "1. parse_json_payload - good fuzz target"},
        {"response": "1. decode_binary_blob\n2. validate_header"},
    ])

    def _fake_post(_url, json=None, timeout=None):
        prompts.append(json["prompt"])
        return _Resp(next(llm_responses))

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("hybrid_code.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Напиши список функций пригодных для фаззинга из директории src/parsers",
        "Дай список других функций для фаззинга из директории src/parsers",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_hybrid_code.py",
        "--index_dir", str(index_dir),
        "--model", "dummy-model",
        "--verbose",
    ])

    ask_app.main()
    out = capsys.readouterr().out

    assert "parse_json_payload - good fuzz target" in out
    assert "decode_binary_blob" in out
    assert len(prompts) == 2

    # First prompt includes all candidates from parsers directory.
    assert "/repo/src/parsers/json_parser.cpp" in prompts[0]
    # Second prompt excludes already-listed function parse_json_payload.
    assert "/repo/src/parsers/json_parser.cpp" not in prompts[1]
    assert "/repo/src/parsers/binary_decoder.cpp" in prompts[1]
