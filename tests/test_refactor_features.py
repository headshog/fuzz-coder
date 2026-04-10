import pytest

from fuzz_coder.ask import core as ask_core
from fuzz_coder.index import app as index_app
from fuzz_coder.index import core as index_core
from fuzz_coder.languages.registry import get_language_frontend, get_language_profile


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

    # Name-based reverse index remains shared aggregate, id-based data stays precise.
    assert called_by["foo"] == [2, 3]
    assert called_by["0"] == [2]
    assert called_by["1"] == [3]


def test_build_call_graph_java_uses_member_qualifier_and_arity_for_overload_resolution():
    chunks = [
        {
            "id": 0,
            "name": "parsePayload",
            "signature": "parsePayload(byte[] data)",
            "file": "/repo/src/parser/Parser.java",
            "body": "return data.length;",
        },
        {
            "id": 1,
            "name": "parsePayload",
            "signature": "parsePayload(byte[] data, int n)",
            "file": "/repo/src/parser/Parser.java",
            "body": "return n;",
        },
        {
            "id": 2,
            "name": "callerOne",
            "signature": "callerOne(byte[] data)",
            "file": "/repo/src/parser/Parser.java",
            "body": "return Parser.parsePayload(data);",
        },
        {
            "id": 3,
            "name": "callerTwo",
            "signature": "callerTwo(byte[] data, int n)",
            "file": "/repo/src/parser/Parser.java",
            "body": "return Parser.parsePayload(data, n);",
        },
    ]

    call_graph, called_by = index_core.build_call_graph(chunks, language_name="java")

    assert call_graph[2]["resolved_calls"] == [0]
    assert call_graph[3]["resolved_calls"] == [1]
    assert any(d["qualified"] == "Parser.parsePayload" and d["arity"] == 1 for d in call_graph[2]["call_details"])
    assert any(d["qualified"] == "Parser.parsePayload" and d["arity"] == 2 for d in call_graph[3]["call_details"])
    assert 2 in called_by["0"]
    assert 3 in called_by["1"]


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


def test_language_registry_returns_frontends_with_expected_names():
    cpp_frontend = get_language_frontend("c_cpp")
    java_frontend = get_language_frontend("java")
    assert cpp_frontend.name == "c_cpp"
    assert java_frontend.name == "java"


def test_extract_functions_regex_java_strips_annotation_via_frontend_hook():
    text = (
        "@Deprecated\n"
        "public static int parseHeader(String line) {\n"
        "    return 0;\n"
        "}\n"
    )
    funcs = index_core.extract_functions_regex("Sample.java", text, language_name="java")
    names = [f["name"] for f in funcs]
    assert "parseHeader" in names


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


def test_regex_parameter_parser_treats_void_parameter_list_as_empty():
    parsed = index_core.parse_parameters_simple("void")
    assert parsed == []


def test_regex_parameter_parser_handles_attributes_and_top_level_defaults():
    params_text = (
        "const std::vector<int> & xs [[maybe_unused]], "
        "int limit = std::max(1, 2), "
        "const char * label = \"x,y\""
    )
    parsed = index_core.parse_parameters_simple(params_text)
    assert [p["name"] for p in parsed] == ["xs", "limit", "label"]


def test_deterministic_example_fallback_preserves_reference_inside_std_function_template():
    frags = [
        {
            "id": 0,
            "name": "process_request",
            "file": "/repo/vendor/cpp-httplib/httplib.cpp",
            "start_line": 7990,
            "end_line": 8242,
            "signature": "process_request(Stream &strm, const std::function<void(Request &)> &setup_request)",
            "parameters": [
                {"name": "strm", "type": "Stream &", "raw": "Stream & strm"},
                {
                    "name": "setup_request",
                    "type": "const std::function<void(Request &)> &",
                    "raw": "const std::function<void(Request &)> & setup_request",
                },
            ],
            "code": "bool process_request(Stream &strm, const std::function<void(Request &)> &setup_request) { return true; }",
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
        "primary_function_name": "process_request",
        "function_names": ["process_request"],
    }

    ans = ask_core.build_example_answer_from_context(frags, analysis=analysis)
    assert "Request )" not in ans
    assert "std::function<void(Request &)>" in ans


def test_deterministic_example_fallback_sanitizes_multiline_observed_call_comment():
    frags = [
        {
            "id": 0,
            "name": "open_output_file",
            "file": "/repo/src/ffmpeg/doc/examples/transcode_aac.c",
            "start_line": 146,
            "end_line": 247,
            "signature": "open_output_file(const char *filename, AVCodecContext *input_codec_context, AVFormatContext **output_format_context, AVCodecContext **output_codec_context)",
            "parameters": [
                {"name": "filename", "type": "const char *", "raw": "const char * filename"},
                {"name": "input_codec_context", "type": "AVCodecContext *", "raw": "AVCodecContext * input_codec_context"},
                {"name": "output_format_context", "type": "AVFormatContext **", "raw": "AVFormatContext ** output_format_context"},
                {"name": "output_codec_context", "type": "AVCodecContext **", "raw": "AVCodecContext ** output_codec_context"},
            ],
            "code": "int open_output_file(...) { return 0; }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
        },
        {
            "id": 1,
            "name": "main",
            "file": "/repo/src/ffmpeg/doc/examples/transcode_aac.c",
            "start_line": 778,
            "end_line": 883,
            "signature": "main(int argc, char **argv)",
            "parameters": [],
            "code": "int main(int argc, char **argv) { return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
    ]
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "open_output_file",
        "function_names": ["open_output_file", "main"],
    }
    observed = (
        "open_output_file(argv[2], input_codec_context,\n"
        "                         &output_format_context, &output_codec_context)"
    )
    example_context = {
        "target": frags[0],
        "caller": frags[1],
        "observed_call": {
            "expr": observed,
            "args": ["argv[2]", "input_codec_context", "&output_format_context", "&output_codec_context"],
        },
    }

    ans = ask_core.build_example_answer_from_context(frags, analysis=analysis, example_context=example_context)

    assert (
        "// Observed call pattern in codebase: open_output_file(argv[2], input_codec_context, "
        "&output_format_context, &output_codec_context)"
    ) in ans
    assert "\n                         &output_format_context" not in ans
    assert "- Observed call: `open_output_file(argv[2], input_codec_context, &output_format_context, &output_codec_context)`" in ans


def test_deterministic_example_fallback_reuses_caller_decls_and_prefers_argv1_for_filename():
    frags = [
        {
            "id": 0,
            "name": "open_output_file",
            "file": "/repo/src/ffmpeg/doc/examples/transcode_aac.c",
            "start_line": 146,
            "end_line": 247,
            "signature": "open_output_file(const char *filename, AVCodecContext *input_codec_context, AVFormatContext **output_format_context, AVCodecContext **output_codec_context)",
            "parameters": [
                {"name": "filename", "type": "const char *", "raw": "const char * filename"},
                {"name": "input_codec_context", "type": "AVCodecContext *", "raw": "AVCodecContext * input_codec_context"},
                {"name": "output_format_context", "type": "AVFormatContext **", "raw": "AVFormatContext ** output_format_context"},
                {"name": "output_codec_context", "type": "AVCodecContext **", "raw": "AVCodecContext ** output_codec_context"},
            ],
            "code": "int open_output_file(...) { return 0; }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
        },
        {
            "id": 1,
            "name": "main",
            "file": "/repo/src/ffmpeg/doc/examples/transcode_aac.c",
            "start_line": 778,
            "end_line": 883,
            "signature": "main(int argc, char **argv)",
            "parameters": [],
            "code": (
                "int main(int argc, char **argv) {\n"
                "    AVCodecContext *input_codec_context = NULL;\n"
                "    AVFormatContext *output_format_context = NULL;\n"
                "    AVCodecContext *output_codec_context = NULL;\n"
                "    open_output_file(argv[2], input_codec_context,\n"
                "                     &output_format_context, &output_codec_context);\n"
                "    return 0;\n"
                "}\n"
            ),
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
    ]
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "open_output_file",
        "function_names": ["open_output_file", "main"],
    }
    example_context = {
        "target": frags[0],
        "caller": frags[1],
        "observed_call": {
            "expr": "open_output_file(argv[2], input_codec_context, &output_format_context, &output_codec_context)",
            "args": ["argv[2]", "input_codec_context", "&output_format_context", "&output_codec_context"],
        },
    }

    ans = ask_core.build_example_answer_from_context(frags, analysis=analysis, example_context=example_context)

    assert "AVCodecContext * input_codec_context = NULL;" in ans
    assert "AVFormatContext * output_format_context = NULL;" in ans
    assert "AVCodecContext * output_codec_context = NULL;" in ans
    assert "open_output_file(argv[1], input_codec_context, &output_format_context, &output_codec_context);" in ans
    assert "AVFormatContext ** output_format_context" not in ans
    assert "AVCodecContext ** output_codec_context" not in ans


def test_deterministic_example_fallback_uses_allocator_for_context_pointer_when_available():
    frags = [
        {
            "id": 0,
            "name": "avi_read_header",
            "file": "/repo/src/ffmpeg/libavformat/avidec.c",
            "start_line": 508,
            "end_line": 1114,
            "signature": "avi_read_header(AVFormatContext *s)",
            "parameters": [
                {"name": "s", "type": "AVFormatContext *", "raw": "AVFormatContext * s"},
            ],
            "code": "int avi_read_header(AVFormatContext *s) { return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
    ]
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "avi_read_header",
        "function_names": ["avi_read_header"],
    }
    symbols = {
        "avi_read_header": [0],
        "avformat_alloc_context": [1],
    }

    ans = ask_core.build_example_answer_from_context(frags, analysis=analysis, symbols=symbols)

    assert "#include <libavformat/avformat.h>" in ans
    assert "AVFormatContext * s = avformat_alloc_context();" in ans
    assert "if (!s) {" in ans
    assert "Failed to initialize s" in ans
    assert "AVFormatContext s_obj{};" not in ans


def test_deterministic_example_fallback_uses_java_generator_when_language_is_java():
    frags = [
        {
            "id": 0,
            "name": "parsePayload",
            "file": "/repo/src/parser/Parser.java",
            "start_line": 10,
            "end_line": 60,
            "signature": "parsePayload(byte[] data, int n)",
            "parameters": [
                {"name": "data", "type": "byte[]", "raw": "byte[] data"},
                {"name": "n", "type": "int", "raw": "int n"},
            ],
            "code": "static int parsePayload(byte[] data, int n) { return n; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
    ]
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "parsePayload",
        "function_names": ["parsePayload"],
    }
    example_context = {
        "target": frags[0],
        "caller": None,
        "observed_call": None,
        "language": "java",
    }

    ans = ask_core.build_example_answer_from_context(
        frags,
        analysis=analysis,
        example_context=example_context,
    )

    assert "```java" in ans
    assert "import java.nio.file.Files;" in ans
    assert "Path.of(args[0])" in ans
    assert "std::ifstream" not in ans
    assert "parsePayload(" in ans


def test_deterministic_example_fallback_auto_detects_java_from_fragment_paths():
    frags = [
        {
            "id": 0,
            "name": "decodeHeader",
            "file": "/repo/src/parser/HeaderDecoder.java",
            "start_line": 10,
            "end_line": 40,
            "signature": "decodeHeader(String line)",
            "parameters": [
                {"name": "line", "type": "String", "raw": "String line"},
            ],
            "code": "static int decodeHeader(String line) { return line.length(); }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
    ]
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "decodeHeader",
        "function_names": ["decodeHeader"],
    }

    ans = ask_core.build_example_answer_from_context(frags, analysis=analysis)

    assert "```java" in ans
    assert "public static void main(String[] args) throws Exception" in ans
    assert "decodeHeader(" in ans


def test_build_type_init_index_collects_struct_init_patterns():
    chunks = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/src/demo.c",
            "start_line": 10,
            "code": (
                "int main() {\n"
                "    AVFormatContext *s = avformat_alloc_context();\n"
                "    FooConfig cfg = foo_config_default();\n"
                "    return 0;\n"
                "}\n"
            ),
        }
    ]

    idx = index_core.build_type_init_index(chunks)
    assert idx.get("version") == 4
    types = idx.get("types", {})
    assert isinstance(idx.get("struct_field_writes", {}), dict)
    assert isinstance(idx.get("function_effects", {}), dict)
    assert isinstance(idx.get("callsite_arg_flow", {}), dict)
    assert isinstance(idx.get("init_recipes_by_type", {}), dict)
    assert "AVFormatContext" in types
    assert any(
        e.get("kind") == "pointer_call" and "avformat_alloc_context(" in e.get("expr", "")
        for e in types["AVFormatContext"]
    )
    assert "FooConfig" in types
    assert any(
        e.get("kind") == "value_call" and "foo_config_default(" in e.get("expr", "")
        for e in types["FooConfig"]
    )


def test_deterministic_example_fallback_uses_type_init_index_for_non_pointer_struct_param():
    frags = [
        {
            "id": 0,
            "name": "foo_open",
            "file": "/repo/src/foo.c",
            "start_line": 20,
            "end_line": 80,
            "signature": "foo_open(FooConfig cfg)",
            "parameters": [
                {"name": "cfg", "type": "FooConfig", "raw": "FooConfig cfg"},
            ],
            "code": "int foo_open(FooConfig cfg) { return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
    ]
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "foo_open",
        "function_names": ["foo_open"],
    }
    type_init_index = {
        "version": 2,
        "types": {
            "FooConfig": [
                {
                    "kind": "value_call",
                    "expr": "foo_config_default()",
                    "function": "foo_config_default",
                    "self_contained": True,
                    "count": 3,
                    "score": 18,
                }
            ]
        },
        "required_fields_by_function": {},
    }

    ans = ask_core.build_example_answer_from_context(
        frags,
        analysis=analysis,
        type_init_index=type_init_index,
    )

    assert "FooConfig cfg = foo_config_default();" in ans
    assert "foo_open(cfg);" in ans


def test_build_type_init_index_collects_required_field_chains_before_target_call():
    chunks = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/src/demo.c",
            "start_line": 1,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int"},
                {"name": "argv", "type": "char **"},
            ],
            "code": (
                "int main(int argc, char ** argv) {\n"
                "    FooConfig cfg = foo_config_default();\n"
                "    cfg.path = argv[1];\n"
                "    cfg.enable = 1;\n"
                "    return foo_open(cfg);\n"
                "}\n"
            ),
        },
        {
            "id": 1,
            "name": "foo_open",
            "file": "/repo/src/foo.c",
            "start_line": 20,
            "signature": "foo_open(FooConfig cfg)",
            "parameters": [
                {"name": "cfg", "type": "FooConfig"},
            ],
            "code": "int foo_open(FooConfig cfg) { return 0; }",
        },
    ]

    idx = index_core.build_type_init_index(chunks)
    req = idx.get("required_fields_by_function", {})
    assert "foo_open" in req
    assert req["foo_open"]
    first = req["foo_open"][0]
    fields = list(first.get("fields", []))
    assert any(f.get("path") == "path" for f in fields)
    assert any(f.get("path") == "enable" for f in fields)
    assert any(bool(f.get("required")) for f in fields)


def test_build_type_init_index_collects_init_recipes_by_type():
    chunks = [
        {
            "id": 0,
            "name": "main",
            "file": "/repo/src/demo.c",
            "start_line": 1,
            "signature": "main(int argc, char ** argv)",
            "parameters": [
                {"name": "argc", "type": "int"},
                {"name": "argv", "type": "char **"},
            ],
            "code": (
                "int main(int argc, char ** argv) {\n"
                "    FooCtx *ctx = foo_ctx_create();\n"
                "    ctx->path = argv[1];\n"
                "    return foo_run(ctx);\n"
                "}\n"
            ),
        },
        {
            "id": 1,
            "name": "foo_run",
            "file": "/repo/src/foo.c",
            "start_line": 30,
            "signature": "foo_run(FooCtx * ctx)",
            "parameters": [
                {"name": "ctx", "type": "FooCtx *"},
            ],
            "code": "int foo_run(FooCtx * ctx) { return 0; }",
        },
    ]

    idx = index_core.build_type_init_index(chunks)
    recipes = idx.get("init_recipes_by_type", {})
    assert "FooCtx" in recipes
    assert recipes["FooCtx"]
    rec = recipes["FooCtx"][0]
    assert rec.get("target_function") == "foo_run"
    assert int(rec.get("arg_index", -1)) == 0
    assert "foo_ctx_create(" in str(rec.get("allocator_expr", ""))
    assert any(str(f.get("path", "")) == "path" for f in list(rec.get("fields") or []))


def test_build_type_init_index_collects_init_recipe_when_call_arg_is_function_parameter():
    chunks = [
        {
            "id": 0,
            "name": "caller",
            "file": "/repo/src/demo.c",
            "start_line": 1,
            "signature": "caller(FooCtx * ctx)",
            "parameters": [
                {"name": "ctx", "type": "FooCtx *", "raw": "FooCtx * ctx"},
            ],
            "code": (
                "int caller(FooCtx * ctx) {\n"
                "    ctx->path = argv[1];\n"
                "    return foo_run(ctx);\n"
                "}\n"
            ),
        },
        {
            "id": 1,
            "name": "foo_run",
            "file": "/repo/src/foo.c",
            "start_line": 30,
            "signature": "foo_run(FooCtx * ctx)",
            "parameters": [
                {"name": "ctx", "type": "FooCtx *", "raw": "FooCtx * ctx"},
            ],
            "code": "int foo_run(FooCtx * ctx) { return 0; }",
        },
    ]

    idx = index_core.build_type_init_index(chunks)
    recipes = idx.get("init_recipes_by_type", {})
    assert "FooCtx" in recipes
    assert recipes["FooCtx"]
    rec = recipes["FooCtx"][0]
    assert rec.get("target_function") == "foo_run"
    assert int(rec.get("arg_index", -1)) == 0
    assert any(str(f.get("path", "")) == "path" for f in list(rec.get("fields") or []))


def test_build_type_init_index_synthesizes_recipe_from_required_fields_when_direct_recipe_missing():
    chunks = [
        {
            "id": 0,
            "name": "foo_run",
            "file": "/repo/src/foo.c",
            "start_line": 1,
            "signature": "foo_run(FooCtx * ctx)",
            "parameters": [
                {"name": "ctx", "type": "FooCtx *", "raw": "FooCtx * ctx"},
            ],
            "code": (
                "int foo_run(FooCtx * ctx) {\n"
                "    if (ctx->path) return 0;\n"
                "    return -1;\n"
                "}\n"
            ),
        },
    ]

    idx = index_core.build_type_init_index(chunks)
    recipes = idx.get("init_recipes_by_type", {})
    assert "FooCtx" in recipes
    assert recipes["FooCtx"]
    rec = recipes["FooCtx"][0]
    assert rec.get("target_function") == "foo_run"
    assert int(rec.get("arg_index", -1)) == 0
    assert any(str(f.get("path", "")) == "path" for f in list(rec.get("fields") or []))


def test_build_type_init_index_synthesizes_recipe_from_struct_field_writes_when_other_sources_missing():
    chunks = [
        {
            "id": 0,
            "name": "configure_ctx",
            "file": "/repo/src/foo.c",
            "start_line": 1,
            "signature": "configure_ctx(FooCtx * ctx)",
            "parameters": [
                {"name": "ctx", "type": "FooCtx *", "raw": "FooCtx * ctx"},
            ],
            "code": (
                "void configure_ctx(FooCtx * ctx) {\n"
                "    ctx->path = argv[1];\n"
                "    ctx->enabled = 0;\n"
                "}\n"
            ),
        },
    ]

    idx = index_core.build_type_init_index(chunks)
    recipes = idx.get("init_recipes_by_type", {})
    assert "FooCtx" in recipes
    assert recipes["FooCtx"]
    rec = recipes["FooCtx"][0]
    assert rec.get("target_function", None) in {"", "configure_ctx"}
    assert any(str(f.get("path", "")) == "path" for f in list(rec.get("fields") or []))


def test_deterministic_example_fallback_prefers_init_recipe_allocator_and_fields():
    frags = [
        {
            "id": 0,
            "name": "foo_run",
            "file": "/repo/src/foo.c",
            "start_line": 20,
            "end_line": 80,
            "signature": "foo_run(FooCtx * ctx)",
            "parameters": [
                {"name": "ctx", "type": "FooCtx *", "raw": "FooCtx * ctx"},
            ],
            "code": "int foo_run(FooCtx * ctx) { return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
    ]
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "foo_run",
        "function_names": ["foo_run"],
    }
    type_init_index = {
        "version": 4,
        "types": {},
        "required_fields_by_function": {},
        "init_recipes_by_type": {
            "FooCtx": [
                {
                    "target_function": "foo_run",
                    "arg_index": 0,
                    "arg_name": "ctx",
                    "score": 9,
                    "call_sites": 3,
                    "allocator_expr": "foo_ctx_create()",
                    "fields": [
                        {"path": "path", "access": "arrow", "support": 1.0, "sample_expr": "argv[1]"},
                    ],
                }
            ]
        },
    }

    ans = ask_core.build_example_answer_from_context(
        frags,
        analysis=analysis,
        type_init_index=type_init_index,
    )

    assert "FooCtx * ctx = foo_ctx_create();" in ans
    assert "ctx->path = argv[1];" in ans
    assert "foo_run(ctx);" in ans


def test_deterministic_example_fallback_applies_required_field_hints():
    frags = [
        {
            "id": 0,
            "name": "foo_open",
            "file": "/repo/src/foo.c",
            "start_line": 20,
            "end_line": 80,
            "signature": "foo_open(FooConfig cfg)",
            "parameters": [
                {"name": "cfg", "type": "FooConfig", "raw": "FooConfig cfg"},
            ],
            "code": "int foo_open(FooConfig cfg) { return 0; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
    ]
    analysis = {
        "query_type": "example_generation",
        "needs_file": True,
        "primary_function_name": "foo_open",
        "function_names": ["foo_open"],
    }
    type_init_index = {
        "version": 2,
        "types": {
            "FooConfig": [
                {
                    "kind": "value_default",
                    "expr": "",
                    "function": None,
                    "self_contained": True,
                    "count": 1,
                    "score": 1,
                }
            ]
        },
        "required_fields_by_function": {
            "foo_open": [
                {
                    "arg_index": 0,
                    "arg_name": "cfg",
                    "type": "FooConfig",
                    "call_sites": 3,
                    "fields": [
                        {
                            "path": "path",
                            "access": "dot",
                            "count": 3,
                            "support": 1.0,
                            "required": True,
                            "sample_expr": "argv[1]",
                            "self_contained": True,
                            "evidence": [],
                        },
                        {
                            "path": "enable",
                            "access": "dot",
                            "count": 3,
                            "support": 1.0,
                            "required": True,
                            "sample_expr": "1",
                            "self_contained": True,
                            "evidence": [],
                        },
                    ],
                }
            ]
        },
    }

    ans = ask_core.build_example_answer_from_context(
        frags,
        analysis=analysis,
        type_init_index=type_init_index,
    )

    assert "FooConfig cfg{};" in ans
    assert "cfg.path = argv[1];" in ans
    assert "cfg.enable = 1;" in ans


def test_regex_parameter_parser_handles_qualified_function_pointer_and_parameter_pack():
    params_text = "void (* const cb)(int), Args&&... args"
    parsed = index_core.parse_parameters_simple(params_text)
    assert [p["name"] for p in parsed] == ["cb", "args"]


def test_regex_extractor_handles_very_long_signature_over_100_lines():
    params = ",\n".join([f"    int p{i}" for i in range(1, 121)])
    text = (
        "int huge_signature(\n"
        f"{params}\n"
        ") {\n"
        "    return p1;\n"
        "}\n"
    )
    funcs = index_core.extract_functions_regex("sample.cpp", text)
    by_name = {f["name"]: f for f in funcs}
    assert "huge_signature" in by_name
    assert len(by_name["huge_signature"]["parameters"]) == 120
