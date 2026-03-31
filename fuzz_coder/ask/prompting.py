from __future__ import annotations

from typing import Any, Dict, List, Optional


DEFAULT_MAX_PROMPT_CHARS = 20000


def _build_thinking_prompt_with_limit(frags, q, analysis=None, conversation_history=None, code_char_limit=1500):
    max_items = len(frags)
    ctx = ""
    for i, f in enumerate(frags, 1):
        param_info = ""
        if f.get("parameters"):
            params = f["parameters"]
            param_strs = [f"{p['name']}: {p['type']}" for p in params]
            param_info = f"\n  Parameters: {', '.join(param_strs)}"

        input_info = ""
        if f.get("has_stdin"):
            input_info = " [STDIN]"
        elif f.get("has_file_input"):
            input_info = " [FILE]"
        elif f.get("has_api_call"):
            input_info = " [API]"

        ctx += f"""
[Function {i}] {f["name"]}{input_info}
  File: {f["file"]}:{f["start_line"]}-{f["end_line"]}
  Signature: {f["signature"]}{param_info}
  Code:
```cpp
{f["code"][:code_char_limit]}
```
"""

    history_ctx = ""
    if conversation_history:
        history_ctx = "\n### Conversation History:\n"
        for idx, (prev_q, prev_a) in enumerate(conversation_history[-3:], 1):
            history_ctx += f"User: {prev_q}\nAssistant: {prev_a[:300]}...\n\n"

    instructions = ""
    if analysis:
        listing_constraints = []
        if analysis.get("needs_stdin"):
            listing_constraints.append("must read from stdin")
        if analysis.get("needs_file"):
            listing_constraints.append("must read from files")
        if analysis.get("needs_api"):
            listing_constraints.append("must make API/network calls")
        if analysis.get("needs_params"):
            listing_constraints.append("must have input parameters")
        if analysis.get("requested_types"):
            listing_constraints.append(f"must match requested types: {', '.join(analysis['requested_types'])}")
        if analysis.get("needs_parse_like"):
            listing_constraints.append("should be parse/input-processing related")
        if analysis.get("exclude_output"):
            listing_constraints.append("must NOT be write/output-like functions")
        if analysis.get("needs_fuzz_targets"):
            listing_constraints.append("should be good fuzzing targets (parsing/decoding, memory handling, complex input processing)")
        if analysis.get("path_filters"):
            listing_constraints.append(f"must be located in path/module: {', '.join(analysis['path_filters'])}")
        if analysis.get("exclude_previously_listed"):
            listing_constraints.append("must exclude functions already listed in previous answers")

        listing_constraints_text = ""
        if listing_constraints:
            mode_text = "all must hold" if analysis.get("constraint_mode") == "all" else "any of positive constraints may match"
            listing_constraints_text = f"\nConstraint mode: {mode_text}. Constraints: " + "; ".join(listing_constraints)

        if analysis["query_type"] == "listing":
            if analysis.get("needs_fuzz_targets"):
                step3_text = "Create a numbered list of the BEST fuzzing-target candidates (rank strongest to weaker)"
                step4_extra = "\n   - Why it is fuzzable (input surface, parser/state complexity, memory/bounds risk)"
                step7_text = "If strict matches are unclear, still return best candidates by fuzzing potential; avoid empty output unless context is empty"
                strict_output_format = f"""
10. Output format is STRICT. Use this exact per-item schema:
   1. **`function_name`**
      - File: `path`
      - Signature: `full signature`
      - Why it matches: one concise reason tied to the query
      - Fuzzable: `High|Medium|Low` - short risk rationale
11. Return at most {max_items} functions (never more than available context functions)
12. End with a short ranking summary paragraph:
   - "Functions like A, B are ranked highest due to ..."
   - "Others such as C are lower due to ..."
"""
            else:
                step3_text = "Create a numbered list of matching functions"
                step4_extra = ""
                step7_text = 'If no functions match, explicitly state "No matching functions found in the codebase"'
                strict_output_format = f"""
10. Output format is STRICT. Use this exact per-item schema:
   1. **`function_name`**
      - File: `path`
      - Signature: `full signature`
      - Why it matches: one concise reason tied to the query
11. Return at most {max_items} functions (never more than available context functions)
"""

            instructions = f"""
### Instructions for Listing Queries:
1. Carefully analyze EACH function in the context above
2. Check if it matches the criteria from the question (respect the declared constraint mode)
3. {step3_text}
4. For each function include:
   - Name and file location
   - Relevant parameters with types
   - Brief explanation why it matches{step4_extra}
5. CRITICAL: Only include functions that ACTUALLY exist in the provided context
6. DO NOT invent or assume functions beyond what is shown
7. {step7_text}
8. After listing, perform SELF-VERIFICATION:
   - Re-check each listed function against the original criteria
   - Confirm the function signature matches the requirements
   - Mark any uncertain entries with [NEEDS REVIEW]
9. Respect negative constraints (e.g. "not write-like") strictly{listing_constraints_text}
{strict_output_format}"""

        elif analysis["query_type"] == "example_generation":
            instructions = """
### Instructions for Example Generation:
1. Identify the specific function(s) mentioned or implied
2. Write a complete, compilable code example showing how to call this function
3. Include:
   - Necessary #include statements
   - Proper variable declarations with correct types
   - The function call with appropriate arguments
   - Error handling if relevant
4. Add comments explaining key parts
5. Make sure the example is realistic and follows the codebase patterns
6. CRITICAL: Use ONLY the parameter types and names from the actual function signature
7. DO NOT invent parameters or change types
8. After generating, perform SELF-VERIFICATION:
   - Check that all parameter types match the function signature exactly
   - Verify the function name is correct
   - Ensure the example would compile with the given signature"""

        elif analysis["query_type"] == "implementation_explanation":
            instructions = """
### Instructions for Implementation Explanation:
1. Explain the algorithm/logic step by step
2. Reference specific code sections from the context with line numbers
3. Describe:
   - What the function does
   - How it processes inputs
   - Key operations and their purpose
   - Return value meaning
4. Use simple language but be technically accurate
5. CRITICAL: Base explanations ONLY on the provided code
6. DO NOT speculate about implementation details not visible in the code
7. If something is unclear from the code, state "Implementation detail not visible in provided code"
8. After explaining, perform SELF-VERIFICATION:
   - Cross-reference each claim with actual code lines
   - Remove any assumptions not supported by the code"""

        elif analysis["needs_type_info"]:
            instructions = """
### Instructions for Type-Specific Queries:
1. Focus on parameter types mentioned in the question
2. Check each function's signature carefully
3. List only functions whose parameters match the requested type
4. Include the exact type signature for verification
5. If Russian terms used (e.g., "массив байтов"), match to C++ types like:
   - byte array → uint8_t*, char*, std::vector<uint8_t>, QByteArray
   - string → std::string, char*, const char*
   - integer → int, int32_t, int64_t, size_t
6. CRITICAL: Verify the type match before including in the answer
7. DO NOT include functions where you're unsure about the type
8. After listing, perform SELF-VERIFICATION:
   - For each function, quote the exact parameter type from the signature
   - Explain why this type matches (or doesn't match) the query
   - Mark uncertain matches with [TYPE UNCERTAIN]"""

        elif analysis["follow_up"]:
            instructions = """
### Instructions for Follow-up Questions:
1. This is a follow-up question - consider the conversation context
2. If user references "these functions" or "from the list", use previous answer
3. Maintain consistency with earlier responses
4. Build upon previous information rather than repeating
5. CRITICAL: If referencing functions from previous answer, verify they exist in current context
6. If the current context doesn't contain previously mentioned functions, state this explicitly"""

    thinking_instructions = """
Think through the task step-by-step internally.
Do NOT output the phase-by-phase reasoning.
Return only the final answer with concise evidence (function name, file, signature, why it matches)."""

    return f"""You are an expert code analyst with deep understanding of codebases.

{history_ctx}
### Current Question: {q}

### Available Functions from Codebase:
{ctx}
{instructions}
{thinking_instructions}
"""


def _build_simple_prompt_with_limit(frags, q, code_char_limit=2000):
    ctx = ""
    for i, f in enumerate(frags, 1):
        param_info = ""
        if f.get("parameters"):
            params = f["parameters"]
            param_strs = [f"{p['name']}: {p['type']}" for p in params]
            param_info = f"\nParameters: {', '.join(param_strs)}"

        input_info = ""
        if f.get("has_stdin"):
            input_info = "\n[Reads from stdin]"
        elif f.get("has_file_input"):
            input_info = "\n[Reads from files]"
        elif f.get("has_api_call"):
            input_info = "\n[Makes API calls]"

        ctx += f"""
--- Function {i} ---
File: {f["file"]}
Function: {f["name"]}{param_info}{input_info}
Signature: {f["signature"]}
Code:
{f["code"][:code_char_limit]}

"""

    return f"""You are an expert code analyst. Answer the question based on the provided code context.

{ctx}

Question: {q}

Answer:"""


def _build_prompt_with_limits(frags, q, analysis=None, conversation_history=None, code_char_limit_thinking=1500, code_char_limit_simple=2000):
    complex_types = ["listing", "example_generation", "implementation_explanation", "type_specific", "input_specific"]
    use_thinking = (analysis and analysis["query_type"] in complex_types) or (conversation_history and len(conversation_history) > 0)

    if use_thinking:
        return _build_thinking_prompt_with_limit(
            frags,
            q,
            analysis=analysis,
            conversation_history=conversation_history,
            code_char_limit=code_char_limit_thinking,
        )

    return _build_simple_prompt_with_limit(frags, q, code_char_limit=code_char_limit_simple)


def build_prompt(frags, q, analysis=None, conversation_history=None, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS):
    """Build prompt with a total character budget.

    Strategy:
    1) Try full candidate set with decreasing per-function code snippets.
    2) If still too large, reduce function count from the tail.
    3) Last resort: hard truncate.
    """
    if max_prompt_chars is None or max_prompt_chars <= 0:
        max_prompt_chars = DEFAULT_MAX_PROMPT_CHARS

    code_limits = [1500, 1200, 900, 700, 500, 350, 250, 150]
    last_prompt = ""

    for limit in code_limits:
        prompt = _build_prompt_with_limits(
            frags,
            q,
            analysis=analysis,
            conversation_history=conversation_history,
            code_char_limit_thinking=limit,
            code_char_limit_simple=max(250, int(limit * 1.2)),
        )
        last_prompt = prompt
        if len(prompt) <= max_prompt_chars:
            return prompt

    trimmed = list(frags)
    min_limit = code_limits[-1]
    while len(trimmed) > 1:
        trimmed = trimmed[:-1]
        prompt = _build_prompt_with_limits(
            trimmed,
            q,
            analysis=analysis,
            conversation_history=conversation_history,
            code_char_limit_thinking=min_limit,
            code_char_limit_simple=max(250, int(min_limit * 1.2)),
        )
        last_prompt = prompt
        if len(prompt) <= max_prompt_chars:
            return prompt

    if len(last_prompt) <= max_prompt_chars:
        return last_prompt

    return last_prompt[:max_prompt_chars]
