import json
import sys

import numpy as np

from fuzz_coder.ask import app as ask_app


class _DummyEmbeddingBackend:
    def encode(self, texts, _convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[1.0, 0.5, 0.25] for _ in texts], dtype="float32")


class _DummyCrossEncoder:
    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, pairs):
        return np.array([0.2 for _ in pairs], dtype="float32")


class _FakeFaissIndex:
    def __init__(self, n_docs):
        self.n_docs = max(1, n_docs)

    def search(self, _emb, k):
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

    def _fake_post(_url, **kwargs):
        payload = kwargs.get("json") or {}
        captured_prompt["prompt"] = payload["prompt"]
        return _Resp({"response": "MOCKED_LLM_ANSWER"})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Напиши список функций пригодных для фаззинга из директории src/parsers",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
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

    def _fake_post(_url, **kwargs):
        payload = kwargs.get("json") or {}
        prompts.append(payload["prompt"])
        return _Resp(next(llm_responses))

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Напиши список функций пригодных для фаззинга из директории src/parsers",
        "Дай список других функций для фаззинга из директории src/parsers",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
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
    assert "### Conversation History:" not in prompts[0]
    assert "### Conversation History:" in prompts[1]


def test_help_command_prints_capabilities_without_llm_call(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Minimal index artifacts required at startup.
    with open(index_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        f.write("")
    _write_json(index_dir / "lexical_index.json", {})
    _write_json(index_dir / "special_indices.json", {})
    _write_json(index_dir / "symbols.json", {})
    _write_json(index_dir / "call_graph.json", {})
    _write_json(index_dir / "called_by.json", {})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(1))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)

    def _should_not_be_called(*_a, **_k):
        raise AssertionError("LLM must not be called for help command")

    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _should_not_be_called)

    inputs = iter([
        "help",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
        "--index_dir", str(index_dir),
        "--model", "dummy-model",
    ])

    ask_app.main()
    out = capsys.readouterr().out

    assert "or 'help' to see examples" in out
    assert "What I can do:" in out
    assert "Example queries:" in out


def test_example_generation_does_not_emit_listing_verification_warning(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "llama_sampler_init_grammar_lazy_patterns",
            "file": "/repo/src/llama.cpp",
            "start_line": 100,
            "end_line": 180,
            "signature": "llama_sampler_init_grammar_lazy_patterns(const struct llama_vocab * vocab, const char * grammar_str)",
            "parameters": [
                {"name": "vocab", "type": "const struct llama_vocab *", "raw": "const struct llama_vocab * vocab"},
                {"name": "grammar_str", "type": "const char *", "raw": "const char * grammar_str"},
            ],
            "code": "llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str);",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        }
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
        "memory_management": [],
        "error_handling": [0],
        "by_type": {},
    })
    _write_json(index_dir / "symbols.json", {"llama_sampler_init_grammar_lazy_patterns": [0], "main": [0]})
    _write_json(index_dir / "call_graph.json", {"0": {"called_by": [], "resolved_calls": []}})
    _write_json(index_dir / "called_by.json", {})

    def _fake_post(_url, **_kwargs):
        return _Resp({"response": "fprintf(stderr, \"demo\");"})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Give an example calling llama_sampler_init_grammar_lazy_patterns from main function that is best for fuzzing",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
        "--index_dir", str(index_dir),
        "--model", "dummy-model",
        "--verbose",
    ])

    ask_app.main()
    out = capsys.readouterr().out

    assert "Type: example_generation" in out
    assert "[⚠️  VERIFICATION WARNING]" not in out
    assert "Unknown functions (not found in index)" not in out


def test_example_generation_missing_target_function_fails_fast_without_llm(tmp_path, monkeypatch, capsys):
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
            "code": "bool parse_json_payload(const std::string & s){ return !s.empty(); }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": False,
        }
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
        "memory_management": [],
        "error_handling": [],
        "by_type": {},
    })
    _write_json(index_dir / "symbols.json", {"parse_json_payload": [0]})
    _write_json(index_dir / "call_graph.json", {"0": {"called_by": [], "resolved_calls": []}})
    _write_json(index_dir / "called_by.json", {})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)

    def _should_not_be_called(*_a, **_k):
        raise AssertionError("LLM must not be called when explicit target function is absent")

    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _should_not_be_called)

    inputs = iter([
        "Write an example of llama_sampler_init_grammar_lazy_patterns function called from main",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
        "--index_dir", str(index_dir),
        "--model", "dummy-model",
    ])

    ask_app.main()
    out = capsys.readouterr().out
    assert "Target function(s) not found in index" in out


def test_listing_hallucinated_output_replaced_with_context_listing(tmp_path, monkeypatch, capsys):
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
            "name": "DEPRECATED",
            "file": "/repo/include/llama.h",
            "start_line": 1360,
            "end_line": 1514,
            "signature": "DEPRECATED(LLAMA_API struct llama_sampler * foo(...)); /// comment blob",
            "parameters": [],
            "code": "DEPRECATED(...); /// docs",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": True,
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
        "output": [],
        "memory_management": [0, 1, 2],
        "error_handling": [],
        "by_type": {"string": [0], "byte_array": [1], "integer": [1]},
    })
    _write_json(index_dir / "symbols.json", {"parse_json_payload": [0], "decode_binary_blob": [1], "DEPRECATED": [2]})
    _write_json(index_dir / "call_graph.json", {})
    _write_json(index_dir / "called_by.json", {})

    def _fake_post(_url, **_kwargs):
        return _Resp({"response": "1. **`llama_sampler_init_grammar_lazy_patterns`**\n- Signature: `...`"})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write a list of functions that can be used for fuzzing",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
        "--index_dir", str(index_dir),
        "--model", "dummy-model",
        "--verbose",
    ])

    ask_app.main()
    out = capsys.readouterr().out

    assert "[⚠️  VERIFICATION WARNING]" in out
    assert "parse_json_payload" in out
    assert "decode_binary_blob" in out
    assert "DEPRECATED" not in out
