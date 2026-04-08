from fuzz_coder.ask.verification import verify_answer_with_context, verify_example_answer_with_context


def test_verification_ignores_parameter_name_noise_in_backticks():
    context = [
        {"name": "llama_sampler_init_adaptive_p"},
    ]
    answer = """
1. **`llama_sampler_init_adaptive_p`**
   - File: `../llama.cpp/include/llama.h:10-20`
   - Signature: `LLAMA_API struct llama_sampler * llama_sampler_init_adaptive_p(float target, float decay, uint32_t seed);`
   - Why it matches: Handles adaptive probability parameters.
   - Fuzzable: Medium - Numeric edge cases.
"""

    v = verify_answer_with_context(answer, context)
    assert v["is_valid"] is True
    assert v["hallucinated"] == set()
    assert "target" not in v["hallucinated"]
    assert "decay" not in v["hallucinated"]
    assert "void" not in v["hallucinated"]


def test_verification_detects_unknown_function_from_structured_list():
    context = [
        {"name": "known_fn"},
    ]
    answer = """
1. **`known_fn`**
   - Signature: `int known_fn(const char * data)`
2. **`unknown_fn`**
   - Signature: `int unknown_fn(int x)`
"""

    v = verify_answer_with_context(answer, context)
    assert v["is_valid"] is False
    assert v["hallucinated"] == {"unknown_fn"}
    assert v["out_of_context"] == set()


def test_verification_marks_known_but_out_of_context_separately():
    context = [
        {"name": "known_fn"},
    ]
    answer = """
1. **`known_fn`**
   - Signature: `int known_fn(const char * data)`
2. **`other_known_fn`**
   - Signature: `int other_known_fn(int x)`
"""

    v = verify_answer_with_context(
        answer,
        context,
        known_functions={"known_fn", "other_known_fn", "third_fn"},
    )
    assert v["is_valid"] is False
    assert v["hallucinated"] == set()
    assert v["out_of_context"] == {"other_known_fn"}


def test_verification_fallback_call_like_when_unstructured():
    context = [
        {"name": "parse_config"},
    ]
    answer = "Use parse_config(data) and unknown_parse(data) for this task."

    v = verify_answer_with_context(answer, context)
    assert v["is_valid"] is False
    assert "unknown_parse" in v["hallucinated"]
    assert "parse_config" in v["mentioned"]


def test_verification_detects_file_mismatch_for_structured_entry():
    context = [
        {
            "name": "parse_config",
            "file": "/repo/src/config/parser.cpp",
            "start_line": 10,
            "end_line": 40,
            "signature": "parse_config(const char * data, size_t n)",
        },
    ]
    answer = """
1. **`parse_config`**
   - File: `/repo/src/other/file.cpp:10-40`
   - Signature: `parse_config(const char * data, size_t n)`
"""

    v = verify_answer_with_context(answer, context, known_functions={"parse_config"})
    assert v["is_valid"] is False
    assert v["hallucinated"] == set()
    assert v["out_of_context"] == set()
    assert v["file_mismatches"] == {"parse_config"}


def test_verification_detects_signature_mismatch_for_structured_entry():
    context = [
        {
            "name": "parse_config",
            "file": "/repo/src/config/parser.cpp",
            "start_line": 10,
            "end_line": 40,
            "signature": "parse_config(const char * data, size_t n)",
        },
    ]
    answer = """
1. **`parse_config`**
   - File: `/repo/src/config/parser.cpp:10-40`
   - Signature: `parse_config(std::string data)`
"""

    v = verify_answer_with_context(answer, context, known_functions={"parse_config"})
    assert v["is_valid"] is False
    assert v["hallucinated"] == set()
    assert v["out_of_context"] == set()
    assert v["signature_mismatches"] == {"parse_config"}


def test_example_verification_requires_file_signature_and_line_evidence():
    context = [
        {
            "name": "ma_device_init__dsound",
            "file": "/repo/vendor/miniaudio/miniaudio.h",
            "start_line": 26135,
            "end_line": 26407,
            "signature": "ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig)",
        },
        {
            "name": "main",
            "file": "/repo/tools/player/main.cpp",
            "start_line": 10,
            "end_line": 80,
            "signature": "main(int argc, char ** argv)",
        },
    ]

    answer = """
```cpp
int main(int argc, char ** argv) {
    return 0;
}
```
Evidence from codebase:
- File: `/repo/vendor/miniaudio/miniaudio.h:26135-26407`
- Signature: `ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig)`
"""
    v = verify_example_answer_with_context(
        answer,
        context,
        target_function="ma_device_init__dsound",
        known_functions={"ma_device_init__dsound", "main"},
    )
    assert v["is_valid"] is True
    assert v["missing_requirements"] == set()
    assert v["file_matches"] >= 1
    assert v["signature_matches"] >= 1
    assert v["has_line_reference"] is True


def test_example_verification_fails_when_evidence_missing():
    context = [
        {
            "name": "ma_device_init__dsound",
            "file": "/repo/vendor/miniaudio/miniaudio.h",
            "start_line": 26135,
            "end_line": 26407,
            "signature": "ma_device_init__dsound(ma_device* pDevice, const ma_device_config* pConfig)",
        },
    ]
    answer = """
```cpp
int main() {
    ma_device_init__dsound(nullptr, nullptr);
    return 0;
}
```
"""
    v = verify_example_answer_with_context(
        answer,
        context,
        target_function="ma_device_init__dsound",
        known_functions={"ma_device_init__dsound"},
    )
    assert v["is_valid"] is False
    assert "file_reference" in v["missing_requirements"]
    assert "signature_reference" in v["missing_requirements"]
    assert "line_reference" in v["missing_requirements"]


def test_example_verification_detects_target_call_arity_mismatch_as_consistency_issue():
    context = [
        {
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
        },
    ]
    answer = """
```cpp
int main() {
    parse_payload(nullptr);
    return 0;
}
```
Evidence from codebase:
- File: `/repo/src/parser.cpp:20-60`
- Signature: `parse_payload(const uint8_t * data, size_t n)`
"""
    v = verify_example_answer_with_context(
        answer,
        context,
        target_function="parse_payload",
        known_functions={"parse_payload"},
    )
    assert v["is_valid"] is True
    assert "target_call_arity_mismatch" in v["consistency_issues"]
    assert v["confidence_level"] in {"medium", "low"}


def test_example_verification_requires_caller_and_observed_call_when_context_has_caller():
    context = [
        {
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 10,
            "end_line": 50,
            "signature": "main(int argc, char ** argv)",
        },
        {
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
        },
    ]
    example_context = {
        "target": context[1],
        "caller": context[0],
        "observed_call": {"expr": "parse_payload(buf, n)", "args": ["buf", "n"], "arity": 2},
        "arg_shapes": [],
        "file_data_flow_hints": {"requires_file_data": False},
    }
    answer = """
```cpp
int main() {
    parse_payload(nullptr, 0);
    return 0;
}
```
Evidence from codebase:
- File: `/repo/src/parser.cpp:20-60`
- Signature: `parse_payload(const uint8_t * data, size_t n)`
"""
    v = verify_example_answer_with_context(
        answer,
        context,
        target_function="parse_payload",
        known_functions={"main", "parse_payload"},
        example_context=example_context,
    )
    assert v["is_valid"] is False
    assert "caller_file_reference" in v["missing_requirements"]
    assert "caller_signature_reference" in v["missing_requirements"]
    assert "observed_call_reference" in v["missing_requirements"]


def test_example_verification_rejects_file_based_example_without_argv1_to_target_flow():
    context = [
        {
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 10,
            "end_line": 50,
            "signature": "main(int argc, char ** argv)",
        },
        {
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 20,
            "end_line": 60,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
        },
    ]
    example_context = {
        "target": context[1],
        "caller": context[0],
        "observed_call": {"expr": "parse_payload(buf, n)", "args": ["buf", "n"], "arity": 2},
        "arg_shapes": [],
        "file_data_flow_hints": {
            "requires_file_data": True,
            "caller_reads_argv1": True,
            "source_vars": ["line"],
        },
    }
    answer = """
```cpp
int main(int argc, char ** argv) {
    std::ifstream file(argv[1]);
    std::string line;
    std::getline(file, line);
    parse_payload(nullptr, 0);
    return 0;
}
```
Evidence from codebase:
- File: `/repo/src/parser.cpp:20-60`
- Signature: `parse_payload(const uint8_t * data, size_t n)`
- File: `/repo/tools/main.cpp:10-50`
- Signature: `main(int argc, char ** argv)`
- Observed call: `parse_payload(buf, n)`
"""
    v = verify_example_answer_with_context(
        answer,
        context,
        target_function="parse_payload",
        known_functions={"main", "parse_payload"},
        example_context=example_context,
    )
    assert v["is_valid"] is False
    assert "file_data_flow_to_target" in v["missing_requirements"]
    assert "file_data_not_used_in_target_call" in v["consistency_issues"]
    assert v["confidence_level"] == "low"


def test_example_verification_ignores_observed_call_noise_when_no_expected_caller_context():
    context = [
        {
            "name": "split",
            "file": "/repo/vendor/cpp-httplib/httplib.cpp",
            "start_line": 1309,
            "end_line": 1329,
            "signature": "split(const char *b, const char *e, char d, size_t m, std::function<void(const char *, const char *)> fn)",
        },
    ]
    answer = """
```cpp
int main(int argc, char ** argv) {
    return 0;
}
```
Evidence from codebase:
- File: `/repo/vendor/cpp-httplib/httplib.cpp:1309-1329`
- Signature: `split(const char *b, const char *e, char d, size_t m, std::function<void(const char *, const char *)> fn)`
- Observed call: `not_the_target(foo, bar)`
"""
    v = verify_example_answer_with_context(
        answer,
        context,
        target_function="split",
        known_functions={"split"},
        example_context={"target": context[0], "caller": None, "observed_call": None, "file_data_flow_hints": {"requires_file_data": False}},
    )
    assert "observed_call_not_target" not in v["consistency_issues"]


def test_example_verification_accepts_transitive_file_flow_chain_to_target_args():
    context = [
        {
            "name": "main",
            "file": "/repo/tools/main.cpp",
            "start_line": 1,
            "end_line": 120,
            "signature": "main(int argc, char ** argv)",
        },
        {
            "name": "parse_payload",
            "file": "/repo/src/parser.cpp",
            "start_line": 10,
            "end_line": 40,
            "signature": "parse_payload(const uint8_t * data, size_t n)",
        },
    ]
    example_context = {
        "target": context[1],
        "caller": context[0],
        "observed_call": {"expr": "parse_payload(buf, n)", "args": ["buf", "n"], "arity": 2},
        "arg_shapes": [],
        "file_data_flow_hints": {
            "requires_file_data": True,
            "caller_reads_argv1": True,
            "source_vars": ["line"],
        },
    }
    answer = """
```cpp
int main(int argc, char ** argv) {
    std::ifstream file(argv[1]);
    std::string line;
    std::getline(file, line);
    std::string tmp = line;
    Config cfg{};
    cfg.path = tmp;
    parse_payload(reinterpret_cast<const uint8_t *>(cfg.path.data()), cfg.path.size());
    return 0;
}
```
Evidence from codebase:
- File: `/repo/src/parser.cpp:10-40`
- Signature: `parse_payload(const uint8_t * data, size_t n)`
- File: `/repo/tools/main.cpp:1-120`
- Signature: `main(int argc, char ** argv)`
- Observed call: `parse_payload(buf, n)`
"""
    v = verify_example_answer_with_context(
        answer,
        context,
        target_function="parse_payload",
        known_functions={"main", "parse_payload"},
        example_context=example_context,
    )
    assert v["is_valid"] is True
    assert v["target_uses_file_data"] is True
    assert "file_data_not_used_in_target_call" not in v["consistency_issues"]


def test_example_verification_penalizes_default_init_when_strong_recipe_exists():
    context = [
        {
            "name": "foo_run",
            "file": "/repo/src/foo.c",
            "start_line": 10,
            "end_line": 40,
            "signature": "foo_run(FooCtx * ctx)",
            "parameters": [
                {"name": "ctx", "type": "FooCtx *"},
            ],
        },
    ]
    answer = """
```cpp
int main() {
    FooCtx ctx_obj{};
    FooCtx * ctx = &ctx_obj;
    foo_run(ctx);
    return 0;
}
```
Evidence from codebase:
- File: `/repo/src/foo.c:10-40`
- Signature: `foo_run(FooCtx * ctx)`
"""
    type_init_index = {
        "version": 4,
        "init_recipes_by_type": {
            "FooCtx": [
                {
                    "target_function": "foo_run",
                    "arg_index": 0,
                    "arg_name": "ctx",
                    "score": 10,
                    "call_sites": 4,
                    "allocator_expr": "foo_ctx_create()",
                    "fields": [
                        {"path": "path", "access": "arrow", "support": 1.0, "sample_expr": "argv[1]"},
                    ],
                }
            ]
        },
    }

    v = verify_example_answer_with_context(
        answer,
        context,
        target_function="foo_run",
        known_functions={"foo_run"},
        type_init_index=type_init_index,
    )
    assert v["is_valid"] is True
    assert "default_init_used_despite_recipe" in v["consistency_issues"]
    assert "FooCtx" in v["default_init_penalized_nominals"]
    assert v["confidence_level"] in {"medium", "low"}
