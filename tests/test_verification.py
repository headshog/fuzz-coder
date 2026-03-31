from fuzz_coder.ask.verification import verify_answer_with_context


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
