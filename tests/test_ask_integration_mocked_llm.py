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


def test_ask_reports_missing_index_directory_and_exits_early(tmp_path, monkeypatch, capsys):
    missing_dir = tmp_path / "no_such_index_dir"

    def _should_not_read_index(_path):
        raise AssertionError("faiss.read_index must not be called when index_dir is missing")

    monkeypatch.setattr(ask_app.faiss, "read_index", _should_not_read_index)
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
        "--index_dir", str(missing_dir),
        "--model", "dummy-model",
    ])

    ask_app.main()
    out = capsys.readouterr().out

    assert "Error: index directory does not exist:" in out
    assert "Run index_fuzz_coder.py first to build the project index." in out


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

    out_lower = out.lower()
    assert "quit" in out_lower and "help" in out_lower
    assert "what i can do" in out_lower
    assert "example quer" in out_lower
    assert "list functions" in out_lower
    assert "chat aliases" in out_lower
    assert "fuzz ->" in out_lower
    assert "fuzz wide ->" in out_lower
    assert "more fuzz ->" in out_lower
    assert "more fuzz wide ->" in out_lower
    assert "example function_name ->" in out_lower
    assert "explain symbol ->" in out_lower


def test_alias_fuzz_variants_expand_to_canonical_queries(tmp_path, monkeypatch, capsys):
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
        "memory_management": [0],
        "error_handling": [],
        "by_type": {"string": [0], "byte_array": [], "integer": []},
    })
    _write_json(index_dir / "symbols.json", {"parse_json_payload": [0]})
    _write_json(index_dir / "call_graph.json", {})
    _write_json(index_dir / "called_by.json", {})

    prompts = []

    def _fake_post(_url, **kwargs):
        payload = kwargs.get("json") or {}
        prompts.append(payload.get("prompt", ""))
        return _Resp({"response": "alias ok"})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "fuzz",
        "fuzz wide",
        "more fuzz",
        "more fuzz wide",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _=None: next(inputs))
    monkeypatch.setattr(sys, "argv", [
        "ask_fuzz_coder.py",
        "--index_dir", str(index_dir),
        "--model", "dummy-model",
    ])

    ask_app.main()
    _ = capsys.readouterr().out

    assert len(prompts) == 4
    assert "### Current Question: Write a large list (20-30) of functions that can be used for fuzzing." in prompts[0]
    assert "at most 4 parameters" in prompts[0]
    assert "### Current Question: Write a list of functions that can be used for fuzzing without parameter count limit." in prompts[1]
    assert "at most 4 parameters" not in prompts[1]
    assert "### Current Question: Write other functions in a large list (20-30) that can be used for fuzzing." in prompts[2]
    assert "Exclude functions already listed previously." in prompts[2]
    assert "### Current Question: Write other functions that are good for fuzzing without parameter count limit." in prompts[3]


def test_alias_example_function_name_expands_to_example_query(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "decode_binary_blob",
            "file": "/repo/src/parsers/binary_decoder.cpp",
            "start_line": 5,
            "end_line": 50,
            "signature": "decode_binary_blob(const uint8_t * buf, size_t n)",
            "parameters": [
                {"name": "buf", "type": "const uint8_t *", "raw": "const uint8_t * buf"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int decode_binary_blob(const uint8_t * buf, size_t n){ return (int)n; }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": True,
            "has_error_handling": False,
        },
        {
            "id": 1,
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 30,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int", "raw": "int argc"},
                {"name": "argv", "type": "char **", "raw": "char ** argv"},
            ],
            "code": "int main(int argc, char ** argv){ decode_binary_blob(nullptr, 0); return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
    ]

    with open(index_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row) + "\n")

    _write_json(index_dir / "lexical_index.json", {})
    _write_json(index_dir / "special_indices.json", {
        "stdin": [],
        "file_input": [1],
        "api_calls": [],
        "output": [],
        "memory_management": [0],
        "error_handling": [1],
        "by_type": {"string": [], "byte_array": [0], "integer": [0]},
    })
    _write_json(index_dir / "symbols.json", {"decode_binary_blob": [0], "main": [1]})
    _write_json(index_dir / "call_graph.json", {"1": {"called_by": [], "resolved_calls": [0]}, "0": {"called_by": [1], "resolved_calls": []}})
    _write_json(index_dir / "called_by.json", {"0": [1], "1": [], "decode_binary_blob": [1], "main": []})

    captured_prompt = {}

    def _fake_post(_url, **kwargs):
        payload = kwargs.get("json") or {}
        captured_prompt["prompt"] = payload.get("prompt", "")
        return _Resp({
            "response": (
                "```cpp\nint main(){return 0;}\n```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parsers/binary_decoder.cpp:5-50`\n"
                "- Signature: `decode_binary_blob(const uint8_t * buf, size_t n)`\n"
            )
        })

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "example decode_binary_blob",
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

    assert "Target function(s) not found in index" not in out
    assert "### Current Question: Write an example of decode_binary_blob function." in captured_prompt["prompt"]
    assert "define a standalone main()" in captured_prompt["prompt"]
    assert "data given from file in argv[1]" in captured_prompt["prompt"]


def test_alias_example_with_module_tail_preserves_module_constraint():
    expanded, used = ask_app.expand_chat_alias("example split from module vendor/cpp-httplib")
    assert used is True
    assert expanded.startswith("Write an example of split function")
    assert "from module vendor/cpp-httplib" in expanded
    assert "split from module vendor/cpp-httplib function" not in expanded


def test_alias_example_with_path_then_module_is_normalized():
    expanded, used = ask_app.expand_chat_alias("example split from vendor/cpp-httplib/httplib.cpp module")
    assert used is True
    assert "from module vendor/cpp-httplib/httplib.cpp" in expanded


def test_alias_explain_function_name_expands_to_parameter_semantics_query():
    expanded, used = ask_app.expand_chat_alias("explain llama_params_fit")
    assert used is True
    assert expanded.startswith("Analyze function parameter semantics for llama_params_fit:")
    assert "for each parameter, explain its role" in expanded
    assert "unknown from provided context" in expanded


def test_alias_explain_with_module_tail_preserves_module_constraint():
    expanded, used = ask_app.expand_chat_alias("explain split from module vendor/cpp-httplib")
    assert used is True
    assert expanded.startswith("Analyze function parameter semantics for split:")
    assert "from module vendor/cpp-httplib" in expanded


def test_alias_fuzz_wide_expands_to_large_broad_fuzz_query():
    expanded, used = ask_app.expand_chat_alias("fuzz wide")
    assert used is True
    assert expanded.startswith("Write a list of functions that can be used for fuzzing")
    assert "without parameter count limit" in expanded
    assert "at most 4 parameters" not in expanded


def test_alias_more_fuzz_wide_expands_to_large_novelty_fuzz_query():
    expanded, used = ask_app.expand_chat_alias("more fuzz wide")
    assert used is True
    assert expanded.startswith("Write other functions that are good for fuzzing")
    assert "without parameter count limit" in expanded
    assert "at most 4 parameters" not in expanded


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
        return _Resp({
            "response": (
                "```cpp\n"
                "int main() {\n"
                "    // demo call\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/llama.cpp:100-180`\n"
                "- Signature: `llama_sampler_init_grammar_lazy_patterns(const struct llama_vocab * vocab, const char * grammar_str)`\n"
            )
        })

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
    assert "[⚠️  EXAMPLE VERIFICATION WARNING]" not in out
    assert "Unknown functions (not found in index)" not in out


def test_example_generation_prompt_contains_real_target_and_caller_locations(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/player/main.cpp",
            "start_line": 10,
            "end_line": 80,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int", "raw": "int argc"},
                {"name": "argv", "type": "char **", "raw": "char ** argv"},
            ],
            "code": "int main(int argc, char ** argv) { ma_device_init__dsound(&device, &cfg, &pb, &cp); return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
        {
            "id": 1,
            "name": "ma_device_init__dsound",
            "file": "/repo/vendor/miniaudio/miniaudio.h",
            "start_line": 26135,
            "end_line": 26407,
            "signature": "ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture)",
            "parameters": [
                {"name": "pDevice", "type": "ma_device*", "raw": "ma_device* pDevice"},
                {"name": "pConfig", "type": "const ma_device_config*", "raw": "const ma_device_config* pConfig"},
            ],
            "code": "ma_result ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture) { return MA_SUCCESS; }",
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
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"byte_array": [], "string": [], "integer": []},
    })
    _write_json(index_dir / "symbols.json", {"main": [0], "ma_device_init__dsound": [1]})
    _write_json(index_dir / "call_graph.json", {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    })
    _write_json(index_dir / "called_by.json", {"0": [], "1": [0], "main": [], "ma_device_init__dsound": [0]})

    captured_prompt = {}

    def _fake_post(_url, **kwargs):
        payload = kwargs.get("json") or {}
        captured_prompt["prompt"] = payload["prompt"]
        return _Resp({"response": "OK"})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of ma_device_init__dsound function that is called from main function and its parameters are constructed from file bytes given in argv[1]",
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

    prompt = captured_prompt["prompt"]
    assert "Include a short \"Evidence from codebase\" note" in prompt
    assert "### Structured Example Facts" in prompt
    assert "File: /repo/tools/player/main.cpp:10-80" in prompt
    assert "File: /repo/vendor/miniaudio/miniaudio.h:26135-26407" in prompt
    assert "main(int argc, char ** argv)" in prompt
    assert "ma_device_init__dsound(ma_device* pDevice" in prompt


def test_example_generation_invalid_response_is_replaced_with_context_grounded_fallback(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/player/main.cpp",
            "start_line": 10,
            "end_line": 80,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int", "raw": "int argc"},
                {"name": "argv", "type": "char **", "raw": "char ** argv"},
            ],
            "code": "int main(int argc, char ** argv) { ma_device_init__dsound(&device, &cfg, &pb, &cp); return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
        {
            "id": 1,
            "name": "ma_device_init__dsound",
            "file": "/repo/vendor/miniaudio/miniaudio.h",
            "start_line": 26135,
            "end_line": 26407,
            "signature": "ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture)",
            "parameters": [
                {"name": "pDevice", "type": "ma_device*", "raw": "ma_device* pDevice"},
                {"name": "pConfig", "type": "const ma_device_config*", "raw": "const ma_device_config* pConfig"},
                {"name": "pDescriptorPlayback", "type": "ma_device_descriptor*", "raw": "ma_device_descriptor* pDescriptorPlayback"},
                {"name": "pDescriptorCapture", "type": "ma_device_descriptor*", "raw": "ma_device_descriptor* pDescriptorCapture"},
            ],
            "code": "ma_result ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture) { return MA_SUCCESS; }",
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
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"byte_array": [0], "string": [], "integer": []},
    })
    _write_json(index_dir / "symbols.json", {"main": [0], "ma_device_init__dsound": [1]})
    _write_json(index_dir / "call_graph.json", {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    })
    _write_json(index_dir / "called_by.json", {"0": [], "1": [0], "main": [], "ma_device_init__dsound": [0]})

    def _fake_post(_url, **_kwargs):
        # Deliberately ungrounded response: no File/Signature evidence.
        return _Resp({"response": "Use some fake function style example without evidence."})

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of ma_device_init__dsound function that is called from main function and its parameters are constructed from file bytes given in argv[1]",
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

    assert "[⚠️  EXAMPLE VERIFICATION WARNING]" in out
    assert "Replaced model output with deterministic context-based example" in out
    assert "Example Code (deterministic context-grounded fallback)" in out
    assert "Evidence from codebase:" in out
    assert "File: `/repo/vendor/miniaudio/miniaudio.h:26135-26407`" in out
    assert "Signature: `ma_device_init__dsound(" in out


def test_example_generation_second_pass_repairs_invalid_answer_without_fallback(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/player/main.cpp",
            "start_line": 10,
            "end_line": 80,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int", "raw": "int argc"},
                {"name": "argv", "type": "char **", "raw": "char ** argv"},
            ],
            "code": "int main(int argc, char ** argv) { ma_device_init__dsound(&device, &cfg, &pb, &cp); return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
        {
            "id": 1,
            "name": "ma_device_init__dsound",
            "file": "/repo/vendor/miniaudio/miniaudio.h",
            "start_line": 26135,
            "end_line": 26407,
            "signature": "ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture)",
            "parameters": [
                {"name": "pDevice", "type": "ma_device*", "raw": "ma_device* pDevice"},
                {"name": "pConfig", "type": "const ma_device_config*", "raw": "const ma_device_config* pConfig"},
                {"name": "pDescriptorPlayback", "type": "ma_device_descriptor*", "raw": "ma_device_descriptor* pDescriptorPlayback"},
                {"name": "pDescriptorCapture", "type": "ma_device_descriptor*", "raw": "ma_device_descriptor* pDescriptorCapture"},
            ],
            "code": "ma_result ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture) { return MA_SUCCESS; }",
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
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"byte_array": [0], "string": [], "integer": []},
    })
    _write_json(index_dir / "symbols.json", {"main": [0], "ma_device_init__dsound": [1]})
    _write_json(index_dir / "call_graph.json", {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    })
    _write_json(index_dir / "called_by.json", {"0": [], "1": [0], "main": [], "ma_device_init__dsound": [0]})

    llm_payloads = iter([
        {"response": "invalid answer without evidence"},
        {"response": "still invalid candidate without grounded evidence"},
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> input_bytes;\n"
                "    input_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    ma_device device{};\n"
                "    ma_device_descriptor pb{};\n"
                "    ma_device_descriptor cp{};\n"
                "    auto rc = ma_device_init__dsound(&device, reinterpret_cast<const ma_device_config *>(input_bytes.data()), &pb, &cp);\n"
                "    (void)rc;\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/vendor/miniaudio/miniaudio.h:26135-26407`\n"
                "- Signature: `ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig, ma_device_descriptor* pDescriptorPlayback, ma_device_descriptor* pDescriptorCapture)`\n"
                "- File: `/repo/tools/player/main.cpp:10-80`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `ma_device_init__dsound(&device, &cfg, &pb, &cp)`\n"
            )
        },
    ])
    call_count = {"n": 0}

    def _fake_post(_url, **_kwargs):
        call_count["n"] += 1
        return _Resp(next(llm_payloads))

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of ma_device_init__dsound function that is called from main function and its parameters are constructed from file bytes given in argv[1]",
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

    assert call_count["n"] == 3
    assert "[Example Second Pass]" in out
    assert "Accepted second-pass repaired answer" in out
    assert "Replaced model output with deterministic context-based example" not in out
    assert "ma_device_init__dsound(&device, &cfg, &pb, &cp)" in out
    assert "Evidence from codebase:" in out


def test_example_generation_second_pass_repairs_target_call_arity_mismatch(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 70,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int", "raw": "int argc"},
                {"name": "argv", "type": "char **", "raw": "char ** argv"},
            ],
            "code": "int main(int argc, char ** argv) { return parse_payload(buf, n); }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
        {
            "id": 1,
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const uint8_t *", "raw": "const uint8_t * data"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int parse_payload(const uint8_t * data, size_t n) { return (int)n; }",
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
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"byte_array": [1], "string": [], "integer": [1]},
    })
    _write_json(index_dir / "symbols.json", {"main": [0], "parse_payload": [1]})
    _write_json(index_dir / "call_graph.json", {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    })
    _write_json(index_dir / "called_by.json", {"0": [], "1": [0], "main": [], "parse_payload": [0]})

    llm_payloads = iter([
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> buf;\n"
                "    buf.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    parse_payload(buf.data());\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> buf;\n"
                "    buf.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    parse_payload(buf.data());\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> buf;\n"
                "    buf.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    parse_payload(buf.data(), buf.size());\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
    ])
    call_count = {"n": 0}

    def _fake_post(_url, **_kwargs):
        call_count["n"] += 1
        return _Resp(next(llm_payloads))

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of parse_payload function that is called from main function and its parameters are constructed from data given from file in argv[1]",
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

    assert call_count["n"] == 3
    assert "[Example Second Pass]" in out
    assert "target_call_arity_mismatch" in out
    assert "Accepted second-pass repaired answer" in out
    assert "Replaced model output with deterministic context-based example" not in out


def test_example_generation_rejects_when_file_bytes_not_used_in_target_call(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 70,
            "signature": "main(int argc, char ** argv)",
            "parameters": [],
            "code": "int main(int argc, char ** argv) { return parse_payload(buf, n); }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
        {
            "id": 1,
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const uint8_t *", "raw": "const uint8_t * data"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int parse_payload(const uint8_t * data, size_t n) { return (int)n; }",
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
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"byte_array": [1], "string": [], "integer": [1]},
    })
    _write_json(index_dir / "symbols.json", {"main": [0], "parse_payload": [1]})
    _write_json(index_dir / "call_graph.json", {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    })
    _write_json(index_dir / "called_by.json", {"0": [], "1": [0], "main": [], "parse_payload": [0]})

    def _fake_post(_url, **_kwargs):
        return _Resp({
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream file(argv[1]);\n"
                "    std::string line;\n"
                "    std::getline(file, line);\n"
                "    parse_payload(nullptr, 0);\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        })

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of parse_payload function that is called from main function and its parameters are constructed from data given from file in argv[1]",
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

    assert "[⚠️  EXAMPLE VERIFICATION WARNING]" in out
    assert "file_data_flow_to_target" in out
    assert "Replaced model output with deterministic context-based example" in out


def test_example_generation_second_pass_repairs_missing_argv1_to_target_data_flow(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 70,
            "signature": "main(int argc, char ** argv)",
            "parameters": [],
            "code": "int main(int argc, char ** argv) { return parse_payload(buf, n); }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
        {
            "id": 1,
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const uint8_t *", "raw": "const uint8_t * data"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int parse_payload(const uint8_t * data, size_t n) { return (int)n; }",
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
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"byte_array": [1], "string": [], "integer": [1]},
    })
    _write_json(index_dir / "symbols.json", {"main": [0], "parse_payload": [1]})
    _write_json(index_dir / "call_graph.json", {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    })
    _write_json(index_dir / "called_by.json", {"0": [], "1": [0], "main": [], "parse_payload": [0]})

    llm_payloads = iter([
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream file(argv[1]);\n"
                "    std::string line;\n"
                "    std::getline(file, line);\n"
                "    parse_payload(nullptr, 0);\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream file(argv[1]);\n"
                "    std::string line;\n"
                "    std::getline(file, line);\n"
                "    parse_payload(nullptr, 0);\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> input_bytes;\n"
                "    input_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    parse_payload(input_bytes.data(), input_bytes.size());\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
    ])
    call_count = {"n": 0}

    def _fake_post(_url, **_kwargs):
        call_count["n"] += 1
        return _Resp(next(llm_payloads))

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of parse_payload function that is called from main function and its parameters are constructed from data given from file in argv[1]",
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

    assert call_count["n"] == 3
    assert "[Example Second Pass]" in out
    assert "file_data_not_used_in_target_call" in out
    assert "Accepted second-pass repaired answer" in out
    assert "Replaced model output with deterministic context-based example" not in out


def test_example_generation_rejects_wrong_caller_evidence_and_falls_back(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)

    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 70,
            "signature": "main(int argc, char ** argv)",
            "parameters": [],
            "code": "int main(int argc, char ** argv) { return parse_payload(buf, n); }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
        {
            "id": 1,
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const uint8_t *", "raw": "const uint8_t * data"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int parse_payload(const uint8_t * data, size_t n) { return (int)n; }",
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
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"byte_array": [1], "string": [], "integer": [1]},
    })
    _write_json(index_dir / "symbols.json", {"main": [0], "parse_payload": [1]})
    _write_json(index_dir / "call_graph.json", {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    })
    _write_json(index_dir / "called_by.json", {"0": [], "1": [0], "main": [], "parse_payload": [0]})

    def _fake_post(_url, **_kwargs):
        return _Resp({
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> buf;\n"
                "    buf.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    parse_payload(buf.data(), buf.size());\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/other/not_main.cpp:1-70`\n"
                "- Signature: `worker_main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        })

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(len(meta)))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of parse_payload function that is called from main function and its parameters are constructed from data given from file in argv[1]",
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

    assert "[⚠️  EXAMPLE VERIFICATION WARNING]" in out
    assert "caller_file_reference" in out or "caller_signature_reference" in out
    assert "Replaced model output with deterministic context-based example" in out


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


def _prepare_parse_payload_example_index(index_dir):
    meta = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 70,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int", "raw": "int argc"},
                {"name": "argv", "type": "char **", "raw": "char ** argv"},
            ],
            "code": "int main(int argc, char ** argv) { return parse_payload(buf, n); }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
        {
            "id": 1,
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const uint8_t *", "raw": "const uint8_t * data"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int parse_payload(const uint8_t * data, size_t n) { return (int)n; }",
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
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"byte_array": [1], "string": [], "integer": [1]},
    })
    _write_json(index_dir / "symbols.json", {"main": [0], "parse_payload": [1]})
    _write_json(index_dir / "call_graph.json", {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    })
    _write_json(index_dir / "called_by.json", {"0": [], "1": [0], "main": [], "parse_payload": [0]})


def test_example_dual_pass_both_candidates_conflicting_forces_fallback(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)
    _prepare_parse_payload_example_index(index_dir)

    llm_payloads = iter([
        {"response": "invalid answer without evidence"},
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> buf;\n"
                "    buf.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    parse_payload(buf.data());\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> buf;\n"
                "    buf.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    parse_payload(nullptr, 0);\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
    ])
    call_count = {"n": 0}

    def _fake_post(_url, **_kwargs):
        call_count["n"] += 1
        return _Resp(next(llm_payloads))

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(2))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of parse_payload function that is called from main function and its parameters are constructed from data given from file in argv[1]",
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

    assert call_count["n"] == 3
    assert "[Example Second Pass]" in out
    assert "No consistent candidate from dual-pass runner; fallback will be used" in out
    assert "Replaced model output with deterministic context-based example" in out


def test_example_dual_pass_prefers_more_grounded_candidate_deterministically(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "index_data"
    index_dir.mkdir(parents=True, exist_ok=True)
    _prepare_parse_payload_example_index(index_dir)

    llm_payloads = iter([
        {"response": "invalid answer without evidence"},
        {
            "response": (
                "int main(int argc, char ** argv) {\n"
                "    parse_payload(reinterpret_cast<const uint8_t *>(argv[1]), 1);\n"
                "    return 0;\n"
                "}\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
        {
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    std::vector<uint8_t> input_bytes;\n"
                "    input_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    parse_payload(input_bytes.data(), input_bytes.size());\n"
                "    return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            )
        },
    ])
    call_count = {"n": 0}

    def _fake_post(_url, **_kwargs):
        call_count["n"] += 1
        return _Resp(next(llm_payloads))

    monkeypatch.setattr(ask_app.faiss, "read_index", lambda _p: _FakeFaissIndex(2))
    monkeypatch.setattr(ask_app, "get_embedding_backend", lambda *a, **k: _DummyEmbeddingBackend())
    monkeypatch.setattr(ask_app, "CrossEncoder", _DummyCrossEncoder)
    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _fake_post)

    inputs = iter([
        "Write an example of parse_payload function that is called from main function and its parameters are constructed from data given from file in argv[1]",
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

    assert call_count["n"] == 3
    assert "[Example Second Pass]" in out
    assert "Accepted second-pass repaired answer" in out
    assert "Winner: candidate_2" in out
    assert "std::vector<uint8_t> input_bytes;" in out
    assert "parse_payload(reinterpret_cast<const uint8_t *>(argv[1]), 1);" not in out
