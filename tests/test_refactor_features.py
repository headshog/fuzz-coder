import pytest

from fuzz_coder.ask import core as ask_core
from fuzz_coder.index import app as index_app
from fuzz_coder.index import core as index_core
from fuzz_coder.languages.registry import get_language_profile


def test_call_llm_returns_structured_status(monkeypatch):
    class _Resp:
        def json(self):
            return {"response": "ok"}

    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", lambda *a, **k: _Resp())
    result = ask_core.call_llm("prompt", "model")
    assert result["ok"] is True
    assert result["response"] == "ok"
    assert result["error"] is None


def test_call_llm_error_returns_structured_status(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("fuzz_coder.ask.llm.requests.post", _boom)
    result = ask_core.call_llm("prompt", "model")
    assert result["ok"] is False
    assert result["response"] == ""
    assert "network down" in result["error"]


def test_build_prompt_respects_global_char_budget():
    frags = []
    for i in range(20):
        frags.append({
            "name": f"fn_{i}",
            "file": f"/repo/src/module/file_{i}.cpp",
            "start_line": 1,
            "end_line": 300,
            "signature": f"fn_{i}(const char * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const char *", "raw": "const char * data"},
                {"name": "n", "type": "size_t", "raw": "size_t n"},
            ],
            "code": "int x = 0;\n" + ("parse(data);\n" * 400),
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        })

    analysis = {
        "query_type": "listing",
        "needs_stdin": False,
        "needs_file": False,
        "needs_api": False,
        "needs_params": False,
        "requested_types": [],
        "needs_parse_like": True,
        "exclude_output": False,
        "needs_fuzz_targets": True,
        "path_filters": ["src/module"],
        "constraint_mode": "all",
    }

    prompt = ask_core.build_prompt(frags, "List fuzz functions", analysis=analysis, conversation_history=None, max_prompt_chars=2000)
    assert len(prompt) <= 2000


def test_prompt_listing_uses_layered_policy_and_single_listing_system_block():
    frags = [{
        "name": "parse_json_payload",
        "file": "/repo/src/parsers/json.cpp",
        "start_line": 10,
        "end_line": 50,
        "signature": "parse_json_payload(const std::string & s)",
        "parameters": [{"name": "s", "type": "const std::string &", "raw": "const std::string & s"}],
        "code": "bool parse_json_payload(const std::string & s){ return !s.empty(); }",
        "has_stdin": False,
        "has_file_input": True,
        "has_api_call": False,
    }]
    analysis = {
        "query_type": "listing",
        "needs_fuzz_targets": True,
        "constraint_mode": "all",
        "path_filters": ["src/parsers"],
    }

    prompt = ask_core.build_prompt(frags, "Write a list of functions that can be used for fuzzing", analysis=analysis)
    assert "### POLICY LAYERS" in prompt
    assert "MUST:" in prompt
    assert "SHOULD:" in prompt
    assert "NICE TO HAVE:" in prompt
    assert "### SYSTEM BLOCK (LISTING)" in prompt
    assert "### SYSTEM BLOCK (EXAMPLE_GENERATION)" not in prompt
    assert "### SYSTEM BLOCK (IMPLEMENTATION_EXPLANATION)" not in prompt
    assert "### OUTPUT FORMAT (STRICT)" in prompt
    assert prompt.count("### SYSTEM BLOCK") == 1


def test_prompt_example_uses_short_example_system_block_and_strict_template():
    frags = [{
        "name": "ma_device_init__dsound",
        "file": "/repo/vendor/miniaudio/miniaudio.h",
        "start_line": 26135,
        "end_line": 26407,
        "signature": "ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig)",
        "parameters": [
            {"name": "pDevice", "type": "ma_device*", "raw": "ma_device* pDevice"},
            {"name": "pConfig", "type": "const ma_device_config*", "raw": "const ma_device_config* pConfig"},
        ],
        "code": "ma_result ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig){return 0;}",
        "has_stdin": False,
        "has_file_input": False,
        "has_api_call": False,
    }]
    analysis = {"query_type": "example_generation"}

    prompt = ask_core.build_prompt(
        frags,
        "Write an example of ma_device_init__dsound from main with argv[1] bytes",
        analysis=analysis,
    )
    assert "### SYSTEM BLOCK (EXAMPLE_GENERATION)" in prompt
    assert "### SYSTEM BLOCK (LISTING)" not in prompt
    assert "### SYSTEM BLOCK (IMPLEMENTATION_EXPLANATION)" not in prompt
    assert 'Include a short "Evidence from codebase" note' in prompt
    assert "### OUTPUT FORMAT (STRICT)" in prompt
    assert "```cpp" in prompt
    assert prompt.count("### SYSTEM BLOCK") == 1


def test_prompt_explanation_uses_short_explanation_system_block():
    frags = [{
        "name": "decode_binary_blob",
        "file": "/repo/src/parsers/blob.cpp",
        "start_line": 1,
        "end_line": 40,
        "signature": "decode_binary_blob(const uint8_t * data, size_t n)",
        "parameters": [
            {"name": "data", "type": "const uint8_t *", "raw": "const uint8_t * data"},
            {"name": "n", "type": "size_t", "raw": "size_t n"},
        ],
        "code": "int decode_binary_blob(const uint8_t * data, size_t n){ if(n < 4) return -1; return 0; }",
        "has_stdin": False,
        "has_file_input": False,
        "has_api_call": False,
    }]
    analysis = {"query_type": "implementation_explanation"}

    prompt = ask_core.build_prompt(frags, "Explain how decode_binary_blob works", analysis=analysis)
    assert "### SYSTEM BLOCK (IMPLEMENTATION_EXPLANATION)" in prompt
    assert "### SYSTEM BLOCK (LISTING)" not in prompt
    assert "### SYSTEM BLOCK (EXAMPLE_GENERATION)" not in prompt
    assert "### OUTPUT FORMAT (STRICT)" in prompt
    assert prompt.count("### SYSTEM BLOCK") == 1


def test_build_call_graph_prefers_same_file_and_avoids_ambiguous_cross_file():
    chunks = [
        {
            "id": 0,
            "name": "foo",
            "signature": "foo(int x)",
            "file": "/repo/a.cpp",
            "body": "return x;",
        },
        {
            "id": 1,
            "name": "foo",
            "signature": "foo(const char * s)",
            "file": "/repo/b.cpp",
            "body": "return 0;",
        },
        {
            "id": 2,
            "name": "caller_same_file",
            "signature": "caller_same_file()",
            "file": "/repo/a.cpp",
            "body": "int y = foo(1); return y;",
        },
        {
            "id": 3,
            "name": "caller_ambiguous",
            "signature": "caller_ambiguous()",
            "file": "/repo/c.cpp",
            "body": "int y = foo(1); return y;",
        },
    ]

    call_graph, called_by = index_core.build_call_graph(chunks)

    assert call_graph[2]["resolved_calls"] == [0]
    assert call_graph[3]["resolved_calls"] == []
    assert call_graph[0]["called_by"] == [2]
    assert call_graph[1]["called_by"] == []
    assert called_by["0"] == [2]
    assert 2 in called_by["foo"]


def test_build_call_graph_skips_ambiguous_cross_file_even_if_signature_matches():
    chunks = [
        {
            "id": 0,
            "name": "foo",
            "signature": "foo(int x)",
            "file": "/repo/a.cpp",
            "body": "return x;",
        },
        {
            "id": 1,
            "name": "foo",
            "signature": "foo(int x)",
            "file": "/repo/b.cpp",
            "body": "return x + 1;",
        },
        {
            "id": 2,
            "name": "caller",
            "signature": "caller()",
            "file": "/repo/c.cpp",
            "body": "return foo(1);",
        },
    ]

    call_graph, called_by = index_core.build_call_graph(chunks)

    assert call_graph[2]["resolved_calls"] == []
    assert call_graph[0]["called_by"] == []
    assert call_graph[1]["called_by"] == []
    assert called_by["0"] == []
    assert called_by["1"] == []


def test_build_call_graph_uses_namespace_and_arity_for_overload_resolution():
    chunks = [
        {
            "id": 0,
            "name": "target",
            "signature": "target(int x)",
            "file": "/repo/mod/a.cpp",
            "body": "return x;",
        },
        {
            "id": 1,
            "name": "target",
            "signature": "target(const uint8_t * data, size_t n)",
            "file": "/repo/mod/a.cpp",
            "body": "return (int)n;",
        },
        {
            "id": 2,
            "name": "caller_one",
            "signature": "caller_one()",
            "file": "/repo/mod/a.cpp",
            "body": "return ns::target(123);",
        },
        {
            "id": 3,
            "name": "caller_two",
            "signature": "caller_two(const uint8_t * data, size_t n)",
            "file": "/repo/mod/a.cpp",
            "body": "return ns::target(data, n);",
        },
    ]

    call_graph, called_by = index_core.build_call_graph(chunks)

    assert call_graph[2]["resolved_calls"] == [0]
    assert call_graph[3]["resolved_calls"] == [1]
    assert any(d["qualified"] == "ns::target" and d["arity"] == 1 for d in call_graph[2]["call_details"])
    assert any(d["qualified"] == "ns::target" and d["arity"] == 2 for d in call_graph[3]["call_details"])
    assert 2 in called_by["0"]
    assert 3 in called_by["1"]


def test_build_call_graph_called_by_is_correct_for_same_name_same_arity_in_different_files():
    chunks = [
        {
            "id": 0,
            "name": "foo",
            "signature": "foo(int x)",
            "file": "/repo/a.cpp",
            "body": "return x;",
        },
        {
            "id": 1,
            "name": "foo",
            "signature": "foo(int x)",
            "file": "/repo/b.cpp",
            "body": "return x + 1;",
        },
        {
            "id": 2,
            "name": "caller_a",
            "signature": "caller_a()",
            "file": "/repo/a.cpp",
            "body": "return foo(1);",
        },
        {
            "id": 3,
            "name": "caller_b",
            "signature": "caller_b()",
            "file": "/repo/b.cpp",
            "body": "return foo(2);",
        },
        {
            "id": 4,
            "name": "caller_ambiguous",
            "signature": "caller_ambiguous()",
            "file": "/repo/c.cpp",
            "body": "return foo(3);",
        },
    ]

    call_graph, called_by = index_core.build_call_graph(chunks)

    assert call_graph[2]["resolved_calls"] == [0]
    assert call_graph[3]["resolved_calls"] == [1]
    assert call_graph[4]["resolved_calls"] == []

    assert call_graph[0]["called_by"] == [2]
    assert call_graph[1]["called_by"] == [3]

    # Name-based reverse index remains legacy aggregate, id-based data stays precise.
    assert called_by["foo"] == [2, 3]
    assert called_by["0"] == [2]
    assert called_by["1"] == [3]


def test_regex_extractor_skips_prototypes_and_uses_local_body_brace():
    text = """
    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(int32_t penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);
    LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);

    struct not_a_function_block {
        int field;
    };

    int real_fuzz_target(const char * data, int n) {
        if (!data) { return 0; }
        return n;
    }
    """

    funcs = index_core.extract_functions_regex("llama.h", text)
    names = {f["name"] for f in funcs}

    assert "real_fuzz_target" in names
    assert "llama_sampler_init_penalties" not in names
    assert "llama_sampler_init_infill" not in names


def test_regex_extractor_skips_deprecated_macro_block():
    text = """
    DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const llama_token * trigger_tokens,
                                size_t num_trigger_tokens), "use llama_sampler_init_grammar_lazy_patterns instead");

    LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab, const char * grammar_str, const char * grammar_root, const char ** trigger_patterns, size_t num_trigger_patterns, const llama_token * trigger_tokens, size_t num_trigger_tokens);

    struct llama_perf_context_data {
        int x;
    };
    """

    funcs = index_core.extract_functions_regex("llama.h", text)
    names = {f["name"] for f in funcs}

    assert "DEPRECATED" not in names
    assert "llama_sampler_init_grammar_lazy_patterns" not in names


def test_is_valid_function_chunk_rejects_macro_like_noise():
    bad = {
        "name": "DEPRECATED",
        "signature": "DEPRECATED(LLAMA_API struct llama_sampler * foo(...)); /// comment",
    }
    good = {
        "name": "decode_binary_blob",
        "signature": "decode_binary_blob(const uint8_t * buf, size_t n)",
    }
    assert ask_core.is_valid_function_chunk(bad) is False
    assert ask_core.is_valid_function_chunk(good) is True


def test_example_generation_candidates_focus_on_named_function_and_neighbors():
    planner = ask_core.QueryPlanner(
        special_indices={},
        symbols={"target_fn": [1]},
        call_graph={"1": {"called_by": [0], "resolved_calls": [2]}},
        called_by={},
        meta=[],
    )

    analysis = {
        "query_type": "example_generation",
        "function_names": ["target_fn"],
        "needs_stdin": False,
        "needs_file": False,
        "needs_api": False,
        "needs_output": False,
        "needs_memory_mgmt": False,
        "needs_error_handling": False,
        "path_filters": [],
        "needs_fuzz_targets": False,
        "requested_types": [],
        "expand_callers": False,
        "expand_callees": False,
        "exclude_output": False,
    }

    candidates = planner.get_search_candidates(analysis, k=10)
    assert 1 in candidates
    assert 0 in candidates
    assert 2 in candidates


def test_example_generation_primary_function_has_priority_over_secondary_mentions():
    planner = ask_core.QueryPlanner(
        special_indices={},
        symbols={"target_fn": [10], "main": [1, 2, 3]},
        call_graph={"10": {"called_by": [20], "resolved_calls": [30]}},
        called_by={},
        meta=[],
    )
    analysis = {
        "query_type": "example_generation",
        "function_names": ["target_fn", "main"],
        "primary_function_name": "target_fn",
        "needs_stdin": False,
        "needs_file": False,
        "needs_api": False,
        "needs_output": False,
        "needs_memory_mgmt": False,
        "needs_error_handling": False,
        "path_filters": [],
        "needs_fuzz_targets": False,
        "requested_types": [],
        "expand_callers": False,
        "expand_callees": False,
        "exclude_output": False,
    }

    candidates = planner.get_search_candidates(analysis, k=10)
    assert candidates[0] == 10
    assert 20 in candidates
    assert 30 in candidates


def test_regex_extractor_line_range_uses_real_matching_brace():
    text = (
        "int foo(int x) {\n"
        "    if (x) {\n"
        "        while (x--) {\n"
        "            x += 1;\n"
        "        }\n"
        "    }\n"
        "    return x;\n"
        "}\n"
        "\n"
        "int bar() {\n"
        "    return 1;\n"
        "}\n"
    )

    funcs = index_core.extract_functions_regex("sample.cpp", text)
    by_name = {f["name"]: f for f in funcs}

    assert "foo" in by_name
    assert "bar" in by_name
    assert by_name["foo"]["start_line"] == 1
    assert by_name["foo"]["end_line"] == 8
    assert by_name["bar"]["start_line"] == 10
    assert by_name["bar"]["end_line"] == 12


def test_regex_extractor_handles_long_multiline_signature():
    params = ",\n".join([f"    int p{i}" for i in range(1, 31)])
    text = (
        "int very_long_signature(\n"
        f"{params}\n"
        ") {\n"
        "    return p1;\n"
        "}\n"
    )

    funcs = index_core.extract_functions_regex("sample.cpp", text)
    by_name = {f["name"]: f for f in funcs}

    assert "very_long_signature" in by_name
    assert len(by_name["very_long_signature"]["parameters"]) == 30


def test_regex_parameter_parser_handles_templates_function_pointers_and_defaults():
    text = (
        "int complex_params(\n"
        "    std::map<std::string, std::vector<int>> data,\n"
        "    void (*cb)(int, int),\n"
        "    std::array<int, 3> arr = {1, 2, 3}\n"
        ") {\n"
        "    return 0;\n"
        "}\n"
    )

    funcs = index_core.extract_functions_regex("sample.cpp", text)
    by_name = {f["name"]: f for f in funcs}
    assert "complex_params" in by_name
    params = by_name["complex_params"]["parameters"]
    assert len(params) == 3
    assert params[0]["name"] == "data"
    assert params[1]["name"] == "cb"
    assert params[2]["name"] == "arr"


@pytest.mark.parametrize(
    "params_text, expected_count, expected_names",
    [
        (
            "const std::function<void(int, int)> & cb, std::vector<std::pair<int, int>> xs = {{1, 2}}",
            2,
            ["cb", "xs"],
        ),
        (
            "int (*fn)(int, int), std::array<int, 3> arr = {1, 2, 3}, const char * label = \"x,y\"",
            3,
            ["fn", "arr", "label"],
        ),
        (
            "const char * s = \"a,b\", int values[4], std::vector<int> v = std::vector<int>{1,2,3}",
            3,
            ["s", "values", "v"],
        ),
    ],
)
def test_split_top_level_params_handles_edge_cases(params_text, expected_count, expected_names):
    parts = index_core.split_top_level_params(params_text)
    assert len(parts) == expected_count
    parsed = index_core.parse_parameters_simple(params_text)
    assert [p["name"] for p in parsed] == expected_names


def test_apply_language_profile_java_changes_detection_behavior():
    try:
        index_app.apply_language_profile("java")

        assert index_core.SUPPORTED_EXT == {".java"}
        input_info = index_core.detect_input_type(
            "byte[] data = Files.readAllBytes(Path.of(argv[0]));"
        )
        features = index_core.detect_code_features(
            "ByteBuffer bb = ByteBuffer.allocateDirect(16); try { System.out.println(bb); } catch (Exception e) {}"
        )
        assert input_info["has_file_input"] is True
        assert "byte_array" in features["types_used"]
        assert features["uses_memory_management"] is True
        assert features["has_error_handling"] is True
    finally:
        # Keep global parser/search patterns deterministic for other tests.
        index_app.apply_language_profile("c_cpp")


def test_language_registry_returns_java_profile_with_expected_surface():
    java_profile = get_language_profile("java")
    assert ".java" in java_profile.supported_ext
    assert "class" in java_profile.control_keywords
    assert any("readAllBytes" in p for p in java_profile.file_input_patterns)


def test_deterministic_example_fallback_does_not_cast_file_bytes_to_std_function():
    frags = [
        {
            "id": 1,
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
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "split",
        "function_names": ["split"],
    }

    ans = ask_core.build_example_answer_from_context(frags, analysis=analysis)

    assert "reinterpret_cast<std::function" not in ans
    assert "std::function<void(const char *, const char *)>" in ans
    assert "input_bytes.data() + input_bytes.size()" in ans
    assert "static_cast<size_t>(input_bytes.size())" in ans
