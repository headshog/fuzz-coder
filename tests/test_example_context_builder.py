from fuzz_coder.ask.example_context import build_example_context


def test_example_context_builder_extracts_target_caller_observed_call_and_shapes():
    frags = [
        {
            "id": 10,
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 80,
            "signature": "main(int argc, char ** argv)",
            "parameters": [],
            "code": (
                "int main(int argc, char ** argv) {\n"
                "    std::vector<uint8_t> input_bytes;\n"
                "    std::ifstream in(argv[1], std::ios::binary);\n"
                "    input_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());\n"
                "    return parse_payload(input_bytes.data(), input_bytes.size());\n"
                "}\n"
            ),
        },
        {
            "id": 11,
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const uint8_t *"},
                {"name": "n", "type": "size_t"},
            ],
            "code": "int parse_payload(const uint8_t * data, size_t n) { return (int)n; }",
        },
    ]
    analysis = {
        "primary_function_name": "parse_payload",
        "function_names": ["parse_payload", "main"],
        "needs_file": True,
    }

    ctx = build_example_context(frags, analysis=analysis)
    assert ctx["target"]["name"] == "parse_payload"
    assert ctx["caller"]["name"] == "main"
    assert "parse_payload(" in ctx["observed_call"]["expr"]
    assert ctx["observed_call"]["arity"] == 2
    assert len(ctx["arg_shapes"]) == 2
    assert ctx["arg_shapes"][0]["shape"] == "byte_buffer"
    assert ctx["arg_shapes"][1]["shape"] in {"size_or_length", "integer"}
    assert ctx["file_data_flow_hints"]["requires_file_data"] is True
    assert ctx["file_data_flow_hints"]["caller_reads_argv1"] is True
    assert ctx["file_data_flow_hints"]["target_uses_file_data"] is True


def test_example_context_builder_marks_file_data_not_used_by_target_call():
    frags = [
        {
            "id": 20,
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 80,
            "signature": "main(int argc, char ** argv)",
            "parameters": [],
            "code": (
                "int main(int argc, char ** argv) {\n"
                "    std::ifstream file(argv[1]);\n"
                "    std::string line;\n"
                "    std::getline(file, line);\n"
                "    return parse_payload(nullptr, 0);\n"
                "}\n"
            ),
        },
        {
            "id": 21,
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
            "parameters": [
                {"name": "data", "type": "const uint8_t *"},
                {"name": "n", "type": "size_t"},
            ],
            "code": "int parse_payload(const uint8_t * data, size_t n) { return (int)n; }",
        },
    ]
    analysis = {
        "primary_function_name": "parse_payload",
        "function_names": ["parse_payload", "main"],
        "needs_file": True,
    }

    ctx = build_example_context(frags, analysis=analysis)
    assert ctx["target"]["name"] == "parse_payload"
    assert ctx["caller"]["name"] == "main"
    assert ctx["file_data_flow_hints"]["requires_file_data"] is True
    assert ctx["file_data_flow_hints"]["caller_reads_argv1"] is True
    assert "line" in ctx["file_data_flow_hints"]["source_vars"]
    assert ctx["file_data_flow_hints"]["target_uses_file_data"] is False


def test_example_context_builder_prefers_higher_quality_target_under_path_filter():
    frags = [
        {
            "id": 30,
            "name": "split",
            "file": "/repo/vendor/cpp-httplib/httplib.cpp",
            "start_line": 1304,
            "end_line": 1307,
            "signature": "split(const char *b, const char *e, char d, std::function<void(const char *, const char *)> fn)",
            "parameters": [
                {"name": "b", "type": "const char *"},
                {"name": "e", "type": "const char *"},
                {"name": "d", "type": "char"},
                {"name": "fn", "type": "std::function<void(const char *, const char *)>"},
            ],
            "code": "split(b, e, d, fn);",
        },
        {
            "id": 31,
            "name": "split",
            "file": "/repo/vendor/cpp-httplib/httplib.cpp",
            "start_line": 1309,
            "end_line": 1329,
            "signature": "split(const char *b, const char *e, char d, size_t m, std::function<void(const char *, const char *)> fn)",
            "parameters": [
                {"name": "b", "type": "const char *"},
                {"name": "e", "type": "const char *"},
                {"name": "d", "type": "char"},
                {"name": "m", "type": "size_t"},
                {"name": "fn", "type": "std::function<void(const char *, const char *)>"},
            ],
            "code": "void split(...) { if (m) { fn(b, e); } }",
        },
    ]
    analysis = {
        "primary_function_name": "split",
        "function_names": ["split"],
        "path_filters": ["vendor/cpp-httplib"],
    }

    ctx = build_example_context(frags, analysis=analysis)
    assert ctx["target"]["start_line"] == 1309
    assert "size_t m" in ctx["target"]["signature"]
