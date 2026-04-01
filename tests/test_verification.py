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
