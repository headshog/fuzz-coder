from fuzz_coder.ask import core as ask_core
from fuzz_coder.index import core as index_core


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
    def _boom(*a, **k):
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


def test_build_prompt_enforces_strict_fuzz_listing_format():
    frags = [
        {
            "name": "parse_one",
            "file": "/repo/src/a.cpp",
            "start_line": 10,
            "end_line": 40,
            "signature": "parse_one(const char * data, size_t n)",
            "parameters": [{"name": "data", "type": "const char *", "raw": "const char * data"}],
            "code": "int parse_one(const char * data, size_t n) { return (int)n; }",
            "has_stdin": False,
            "has_file_input": True,
            "has_api_call": False,
        },
        {
            "name": "parse_two",
            "file": "/repo/src/b.cpp",
            "start_line": 20,
            "end_line": 50,
            "signature": "parse_two(std::string s)",
            "parameters": [{"name": "s", "type": "std::string", "raw": "std::string s"}],
            "code": "int parse_two(std::string s) { return (int)s.size(); }",
            "has_stdin": False,
            "has_file_input": False,
            "has_api_call": False,
        },
    ]

    analysis = {
        "query_type": "listing",
        "is_listing": True,
        "needs_stdin": False,
        "needs_file": False,
        "needs_api": False,
        "needs_params": False,
        "requested_types": [],
        "needs_parse_like": True,
        "exclude_output": False,
        "needs_fuzz_targets": True,
        "path_filters": [],
        "constraint_mode": "all",
        "exclude_previously_listed": False,
    }

    prompt = ask_core.build_prompt(
        frags,
        "Write a list of functions that can be used for fuzzing",
        analysis=analysis,
        conversation_history=None,
        max_prompt_chars=20000,
    )

    assert "Output format is STRICT." in prompt
    assert "Fuzzable: `High|Medium|Low`" in prompt
    assert "Return at most 2 functions" in prompt
    assert "ranked highest" in prompt


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
