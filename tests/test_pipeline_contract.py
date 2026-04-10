import numpy as np

from fuzz_coder.ask import core as ask_core
from fuzz_coder.ask.pipeline import PipelineConfig, PipelineResult, QueryPipeline


class _DummyEmbeddingBackend:
    def encode(self, texts, _convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[1.0, 0.5, 0.25] for _ in texts], dtype="float32")


class _FakeFaissIndex:
    def __init__(self, n_docs):
        self.n_docs = max(1, n_docs)

    def search(self, _emb, k):
        ids = np.array([[i % self.n_docs for i in range(k)]], dtype="int64")
        scores = np.ones((1, k), dtype="float32")
        return scores, ids


def _base_meta_listing():
    return [
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
            "code": "int decode_binary_blob(const uint8_t * buf, size_t n){ return (int)n; }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": True,
            "has_error_handling": False,
        },
    ]


def _build_pipeline(monkeypatch, meta, call_llm_stub, *, shadow_mode=False, shadow_runner=None):
    special_indices = {
        "stdin": [],
        "file_input": [i for i, m in enumerate(meta) if m.get("has_file_input")],
        "api_calls": [],
        "output": [i for i, m in enumerate(meta) if m.get("has_output")],
        "memory_management": [i for i, m in enumerate(meta) if m.get("uses_memory_management")],
        "error_handling": [i for i, m in enumerate(meta) if m.get("has_error_handling")],
        "by_type": {"string": [], "byte_array": [], "integer": []},
    }
    symbols = {}
    for i, m in enumerate(meta):
        symbols.setdefault(m["name"], []).append(i)

    call_graph = {}
    for i in range(len(meta)):
        call_graph[str(i)] = {"called_by": [], "resolved_calls": []}

    planner = ask_core.QueryPlanner(special_indices, symbols, call_graph, {}, meta=meta)
    monkeypatch.setattr(ask_core, "call_llm", call_llm_stub)

    return QueryPipeline(
        core_module=ask_core,
        planner=planner,
        idx=_FakeFaissIndex(len(meta)),
        lex={},
        meta=meta,
        embed_model=_DummyEmbeddingBackend(),
        reranker=None,
        symbols=symbols,
        call_graph=call_graph,
        called_by={},
        config=PipelineConfig(
            top_k=10,
            rerank_top_k=5,
            max_prompt_chars=20000,
            model="dummy-model",
            verbose=False,
            shadow_mode=shadow_mode,
        ),
        shadow_runner=shadow_runner,
    )


def test_pipeline_shadow_mode_contract_matches_non_shadow(monkeypatch):
    meta = _base_meta_listing()

    def _stub_call_llm(_prompt, _model, temperature=0.1):  # noqa: ARG001
        return {
            "ok": True,
            "response": (
                "1. **`parse_json_payload`**\n"
                "- File: `/repo/src/parsers/json_parser.cpp:10-40`\n"
                "- Signature: `parse_json_payload(const std::string & s)`\n"
                "- Why it matches: parse path"
            ),
            "error": None,
        }

    p1 = _build_pipeline(monkeypatch, meta, _stub_call_llm, shadow_mode=False)
    r1 = p1.run("Write a list of functions that can be used for fuzzing", [])

    p2 = _build_pipeline(monkeypatch, meta, _stub_call_llm, shadow_mode=True)
    r2 = p2.run("Write a list of functions that can be used for fuzzing", [])

    assert r1.answer == r2.answer
    assert r1.used_fallback == r2.used_fallback
    assert r1.used_second_pass == r2.used_second_pass
    assert abs(r1.confidence - r2.confidence) < 1e-9
    assert "_pipeline_shadow_diff" not in r2.verification_flags


def test_pipeline_shadow_mode_reports_contract_diff(monkeypatch):
    meta = _base_meta_listing()

    def _stub_call_llm(_prompt, _model, temperature=0.1):  # noqa: ARG001
        return {"ok": True, "response": "fixed answer", "error": None}

    def _shadow_runner(_q, _history):
        return PipelineResult(
            answer="shadow changed answer",
            history_answer="shadow changed answer",
            elapsed=0.01,
            confidence=0.2,
            verification_flags={"is_valid": False},
            used_fallback=True,
            used_second_pass=False,
            analysis={},
            prompt="",
            llm_ok=True,
            show_response_time=True,
        )

    pipeline = _build_pipeline(
        monkeypatch,
        meta,
        _stub_call_llm,
        shadow_mode=True,
        shadow_runner=_shadow_runner,
    )
    result = pipeline.run("Write a list of functions that can be used for fuzzing", [])

    assert "_pipeline_shadow_diff" in result.verification_flags
    diff = result.verification_flags["_pipeline_shadow_diff"]
    assert "answer" in diff
    assert "confidence" in diff
    assert "used_fallback" in diff


def test_pipeline_result_contract_fields_present(monkeypatch):
    meta = _base_meta_listing()

    def _stub_call_llm(_prompt, _model, temperature=0.1):  # noqa: ARG001
        return {"ok": True, "response": "simple answer", "error": None}

    pipeline = _build_pipeline(monkeypatch, meta, _stub_call_llm, shadow_mode=False)
    result = pipeline.run("Explain parse_json_payload", [])

    assert isinstance(result.confidence, float)
    assert isinstance(result.verification_flags, dict)
    assert isinstance(result.used_fallback, bool)
    assert isinstance(result.used_second_pass, bool)
    assert isinstance(result.reasoning_path, str)
    assert result.reasoning_path in {"single_pass", "dual_pass_consistent", "dual_pass_fallback"}


def test_pipeline_golden_listing_falls_back_to_context_when_model_hallucinates(monkeypatch):
    meta = _base_meta_listing()

    def _stub_call_llm(_prompt, _model, temperature=0.1):  # noqa: ARG001
        return {
            "ok": True,
            "response": "1. **`unknown_hallucinated_fn`**\n- Signature: `int unknown_hallucinated_fn()`",
            "error": None,
        }

    pipeline = _build_pipeline(monkeypatch, meta, _stub_call_llm, shadow_mode=False)
    result = pipeline.run("Write a list of functions that can be used for fuzzing", [])

    assert result.used_fallback is True
    assert "parse_json_payload" in result.answer
    assert "decode_binary_blob" in result.answer
    assert "unknown_hallucinated_fn" not in result.answer


def test_pipeline_golden_example_fallback_when_invalid_example_answer(monkeypatch):
    meta = [
        {
            "id": 0,
            "name": "split",
            "file": "/repo/vendor/cpp-httplib/httplib.cpp",
            "start_line": 1309,
            "end_line": 1329,
            "signature": "split(const char *b, const char *e, char d, size_t m, std::function<void(const char *, const char *)> fn)",
            "parameters": [
                {"name": "b", "type": "const char *", "raw": "const char * b"},
                {"name": "e", "type": "const char *", "raw": "const char * e"},
                {"name": "d", "type": "char", "raw": "char d"},
                {"name": "m", "type": "size_t", "raw": "size_t m"},
                {"name": "fn", "type": "std::function<void(const char *, const char *)>", "raw": "std::function<void(const char *, const char *)> fn"},
            ],
            "code": "void split(const char *b, const char *e, char d, size_t m, std::function<void(const char *, const char *)> fn) {}",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": False,
        },
    ]

    def _stub_call_llm(_prompt, _model, temperature=0.1):  # noqa: ARG001
        return {"ok": True, "response": "bad example without evidence", "error": None}

    pipeline = _build_pipeline(monkeypatch, meta, _stub_call_llm, shadow_mode=False)
    result = pipeline.run(
        (
            "Write an example of split function. In the generated snippet, define a standalone main() and call split "
            "from it. Construct its parameters from data given from file in argv[1] from module vendor/cpp-httplib/httplib.cpp"
        ),
        [],
    )

    assert result.used_fallback is True
    assert result.used_second_pass is True
    assert result.reasoning_path == "dual_pass_fallback"
    assert "Example Code (deterministic context-grounded fallback)" in result.answer
    assert "Target function: `split`" in result.answer


def test_pipeline_golden_example_dual_pass_selects_grounded_candidate(monkeypatch):
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

    llm_responses = iter([
        {"ok": True, "response": "invalid answer without evidence", "error": None},
        {
            "ok": True,
            "response": (
                "int main(int argc, char ** argv) {\n"
                "  parse_payload(reinterpret_cast<const uint8_t *>(argv[1]), 1);\n"
                "  return 0;\n"
                "}\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            ),
            "error": None,
        },
        {
            "ok": True,
            "response": (
                "```cpp\n"
                "int main(int argc, char ** argv) {\n"
                "  std::ifstream in(argv[1], std::ios::binary);\n"
                "  std::vector<uint8_t> input_bytes;\n"
                "  input_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "  parse_payload(input_bytes.data(), input_bytes.size());\n"
                "  return 0;\n"
                "}\n"
                "```\n"
                "Evidence from codebase:\n"
                "- File: `/repo/src/parser.cpp:20-60`\n"
                "- Signature: `parse_payload(const uint8_t * data, size_t n)`\n"
                "- File: `/repo/tools/main.cpp:1-70`\n"
                "- Signature: `main(int argc, char ** argv)`\n"
                "- Observed call: `parse_payload(buf, n)`\n"
            ),
            "error": None,
        },
    ])

    def _stub_call_llm(_prompt, _model, temperature=0.1):  # noqa: ARG001
        return next(llm_responses)

    special_indices = {
        "stdin": [],
        "file_input": [0],
        "api_calls": [],
        "output": [],
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"string": [], "byte_array": [1], "integer": [1]},
    }
    symbols = {"main": [0], "parse_payload": [1]}
    call_graph = {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    }
    called_by = {"0": [], "1": [0], "main": [], "parse_payload": [0]}
    planner = ask_core.QueryPlanner(special_indices, symbols, call_graph, called_by, meta=meta)
    monkeypatch.setattr(ask_core, "call_llm", _stub_call_llm)

    pipeline = QueryPipeline(
        core_module=ask_core,
        planner=planner,
        idx=_FakeFaissIndex(len(meta)),
        lex={},
        meta=meta,
        embed_model=_DummyEmbeddingBackend(),
        reranker=None,
        symbols=symbols,
        call_graph=call_graph,
        called_by=called_by,
        config=PipelineConfig(
            top_k=10,
            rerank_top_k=5,
            max_prompt_chars=20000,
            model="dummy-model",
            verbose=False,
            shadow_mode=False,
        ),
    )

    result = pipeline.run(
        "Write an example of parse_payload function that is called from main function and its parameters are constructed from data given from file in argv[1]",
        [],
    )

    assert result.used_second_pass is True
    assert result.used_fallback is False
    assert result.reasoning_path == "dual_pass_consistent"
    assert "std::vector<uint8_t> input_bytes;" in result.answer
    assert "parse_payload(reinterpret_cast<const uint8_t *>(argv[1]), 1);" not in result.answer


def test_pipeline_parameter_analysis_enforces_function_signature_header_and_falls_back(monkeypatch):
    meta = [
        {
            "id": 0,
            "name": "llama_params_fit",
            "file": "/repo/src/llama.cpp",
            "start_line": 100,
            "end_line": 180,
            "signature": "llama_params_fit(const char * path_model, struct llama_model_params * mparams, uint32_t n_ctx_min)",
            "parameters": [
                {"name": "path_model", "type": "const char *", "raw": "const char * path_model"},
                {"name": "mparams", "type": "struct llama_model_params *", "raw": "struct llama_model_params * mparams"},
                {"name": "n_ctx_min", "type": "uint32_t", "raw": "uint32_t n_ctx_min"},
            ],
            "code": "bool llama_params_fit(const char * path_model, struct llama_model_params * mparams, uint32_t n_ctx_min) { return true; }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
            "has_output": False,
            "uses_memory_management": False,
            "has_error_handling": True,
        },
    ]

    def _stub_call_llm(_prompt, _model, temperature=0.1):  # noqa: ARG001
        return {
            "ok": True,
            "response": (
                "1. path_model: const char *\n"
                "2. mparams: struct llama_model_params *\n"
                "3. n_ctx_min: uint32_t\n"
            ),
            "error": None,
        }

    pipeline = _build_pipeline(monkeypatch, meta, _stub_call_llm, shadow_mode=False)
    result = pipeline.run(
        "Analyze function parameter semantics for llama_params_fit: for each parameter explain role and format",
        [],
    )

    assert result.used_fallback is True
    assert "Function: `llama_params_fit`" in result.answer
    assert "Signature: `llama_params_fit(" in result.answer
    assert "Parameters:" in result.answer


def test_pipeline_example_grounding_includes_real_caller_and_observed_call(monkeypatch):
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
            "code": (
                "int main(int argc, char ** argv) {\n"
                "  ma_device device{};\n"
                "  ma_device_config cfg{};\n"
                "  ma_device_descriptor pb{};\n"
                "  ma_device_descriptor cp{};\n"
                "  ma_device_init__dsound(&device, &cfg, &pb, &cp);\n"
                "  return 0;\n"
                "}\n"
            ),
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

    def _stub_call_llm(_prompt, _model, temperature=0.1):  # noqa: ARG001
        return {"ok": True, "response": "invalid answer without evidence", "error": None}

    call_graph = {
        "0": {"called_by": [], "resolved_calls": [1]},
        "1": {"called_by": [0], "resolved_calls": []},
    }
    called_by = {"0": [], "1": [0], "main": [], "ma_device_init__dsound": [0]}

    special_indices = {
        "stdin": [],
        "file_input": [0],
        "api_calls": [],
        "output": [],
        "memory_management": [1],
        "error_handling": [0, 1],
        "by_type": {"string": [], "byte_array": [0], "integer": []},
    }
    symbols = {"main": [0], "ma_device_init__dsound": [1]}
    planner = ask_core.QueryPlanner(special_indices, symbols, call_graph, called_by, meta=meta)
    monkeypatch.setattr(ask_core, "call_llm", _stub_call_llm)

    pipeline = QueryPipeline(
        core_module=ask_core,
        planner=planner,
        idx=_FakeFaissIndex(len(meta)),
        lex={},
        meta=meta,
        embed_model=_DummyEmbeddingBackend(),
        reranker=None,
        symbols=symbols,
        call_graph=call_graph,
        called_by=called_by,
        config=PipelineConfig(
            top_k=10,
            rerank_top_k=5,
            max_prompt_chars=20000,
            model="dummy-model",
            verbose=False,
            shadow_mode=False,
        ),
    )
    result = pipeline.run(
        "Write an example of ma_device_init__dsound function that is called from main function and its parameters are constructed from file bytes given in argv[1]",
        [],
    )

    assert result.used_fallback is True
    assert "/repo/vendor/miniaudio/miniaudio.h:26135-26407" in result.answer
    assert "/repo/tools/player/main.cpp:10-80" in result.answer
    assert "Observed call: `ma_device_init__dsound(&device, &cfg, &pb, &cp)`" in result.answer
