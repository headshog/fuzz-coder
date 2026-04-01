from fuzz_coder.ask import core as ask_core
from fuzz_coder.index import core as index_core
from fuzz_coder.languages.registry import get_language_profile, get_supported_language_names


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
    assert 2 in called_by["foo"]


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

def test_language_registry_includes_java_profile():
    names = get_supported_language_names()
    assert "c_cpp" in names
    assert "java" in names

    java_profile = get_language_profile("java")
    assert ".java" in java_profile.supported_ext
