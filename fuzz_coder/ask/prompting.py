from __future__ import annotations

DEFAULT_MAX_PROMPT_CHARS = 20000


def _render_function_context(frags, code_char_limit):
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
    return ctx


def _render_history_context(conversation_history):
    if not conversation_history:
        return ""

    history_ctx = "\n### Conversation History:\n"
    for prev_q, prev_a in conversation_history[-3:]:
        history_ctx += f"User: {prev_q}\nAssistant: {prev_a[:300]}...\n\n"
    return history_ctx


def _format_loc(chunk):
    if not chunk:
        return "not found"
    return f"{chunk.get('file', '')}:{chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}"


def _render_example_context_facts(example_context):
    if not example_context:
        return ""

    target = example_context.get("target")
    caller = example_context.get("caller")
    observed_call = example_context.get("observed_call")
    arg_shapes = list(example_context.get("arg_shapes") or [])
    flow = example_context.get("file_data_flow_hints") or {}

    if not target:
        return ""

    lines = [
        "### Structured Example Facts",
        "Target:",
        f"- Name: {target.get('name', 'unknown')}",
        f"- File: {_format_loc(target)}",
        f"- Signature: {target.get('signature', '')}",
        "Caller:",
    ]

    if caller:
        lines.extend([
            f"- Name: {caller.get('name', 'unknown')}",
            f"- File: {_format_loc(caller)}",
            f"- Signature: {caller.get('signature', '')}",
        ])
    else:
        lines.append("- not found in selected context")

    lines.append("Observed call-site:")
    if observed_call:
        lines.extend([
            f"- Expr: {observed_call.get('expr', '')}",
            f"- Arity: {observed_call.get('arity', 'unknown')}",
        ])
    else:
        lines.append("- not found in selected context")

    lines.append("Arg shapes:")
    if arg_shapes:
        for a in arg_shapes[:12]:
            observed = a.get("observed_arg")
            obs_suffix = f" <- observed `{observed}`" if observed else ""
            lines.append(f"- [{a.get('index')}] {a.get('name')}: {a.get('type')} ({a.get('shape')}){obs_suffix}")
    else:
        lines.append("- no parameter metadata")

    lines.append("File-data flow hints:")
    lines.append(f"- requires_file_data: {bool(flow.get('requires_file_data'))}")
    lines.append(f"- caller_reads_argv1: {bool(flow.get('caller_reads_argv1'))}")
    lines.append(f"- target_uses_file_data: {bool(flow.get('target_uses_file_data'))}")
    lines.append(f"- argv1_direct_to_target: {bool(flow.get('argv1_direct_to_target'))}")

    src_vars = list(flow.get("source_vars") or [])
    if src_vars:
        lines.append(f"- source_vars: {', '.join(src_vars[:8])}")

    return "\n".join(lines)


def _render_function_hints(function_hints):
    if not function_hints:
        return ""

    lines = ["### Function Docs/Usage Hints"]
    for fn, hints in (function_hints or {}).items():
        lines.append(f"Function: `{fn}`")
        if not hints:
            lines.append("- no external hints")
            continue
        for h in hints[:8]:
            file_loc = f"{h.get('file', '')}:{h.get('line', '?')}"
            snippet = " ".join(str(h.get("snippet", "")).split())
            snippet = snippet[:320]
            source = h.get("source", "docs")
            lines.append(f"- [{source}] `{file_loc}`: {snippet}")
    return "\n".join(lines)


def _build_constraint_text(analysis):
    constraints = []
    if analysis.get("needs_stdin"):
        constraints.append("must read from stdin")
    if analysis.get("needs_file"):
        constraints.append("must read from files")
    if analysis.get("needs_api"):
        constraints.append("must make API/network calls")
    if analysis.get("needs_output"):
        constraints.append("must perform output/logging operations")
    if analysis.get("needs_memory_mgmt"):
        constraints.append("must involve memory/buffer management")
    if analysis.get("needs_error_handling"):
        constraints.append("must include error-handling paths")
    if analysis.get("needs_params"):
        constraints.append("must have input parameters")
    if isinstance(analysis.get("max_param_count"), int):
        constraints.append(f"must have at most {analysis['max_param_count']} parameters")
    if analysis.get("requested_types"):
        constraints.append(f"must match requested types: {', '.join(analysis['requested_types'])}")
    if analysis.get("needs_parse_like"):
        constraints.append("should be parse/input-processing related")
    if analysis.get("needs_broad_fuzz_surface"):
        constraints.append(
            "must match at least one input surface: parse-like OR stdin/file input OR "
            "path/file-handle/simple-pointer/vector-like parameters"
        )
    if analysis.get("exclude_output"):
        constraints.append("must NOT be write/output-like functions")
    if analysis.get("needs_fuzz_targets"):
        constraints.append("should be good fuzzing targets (parsing/decoding, memory handling, complex inputs)")
    if analysis.get("path_filters"):
        constraints.append(f"must be located in path/module: {', '.join(analysis['path_filters'])}")
    if analysis.get("exclude_previously_listed"):
        constraints.append("must exclude functions already listed in previous answers")

    if not constraints:
        return "No explicit hard constraints in query."

    mode_text = (
        "all positive constraints must hold"
        if analysis.get("constraint_mode") == "all"
        else "match strongest subset of positive constraints"
    )
    return f"Constraint mode: {mode_text}. Constraints: " + "; ".join(constraints)


def _build_policy_layers_block():
    return """
### POLICY LAYERS
MUST:
- Use only functions from "Available Functions from Codebase".
- Keep function name, file, line range, and signature exact.
- Never invent functions, signatures, parameters, files, or line ranges.
- If context is missing required evidence, state that explicitly.
SHOULD:
- Prefer real call-site evidence and real argument shapes from context.
- Keep each item concise and tied to concrete evidence.
NICE TO HAVE:
- Briefly rank confidence/strength when returning multiple candidates.
""".strip()


def _build_listing_system_block(analysis, max_items):
    desired_count = analysis.get("listing_target_count")
    fuzz_line = ""
    summary_line = ""
    if analysis.get("needs_fuzz_targets"):
        fuzz_line = "\n- Fuzzable: `High|Medium|Low` - short risk rationale"
        summary_line = "\nSummary: 1-2 sentences on strongest vs weaker candidates."

    count_rule = f"Return at most {max_items} functions from context."
    if isinstance(desired_count, int) and desired_count > 0:
        desired = min(max_items, desired_count)
        count_rule = (
            f"Return {desired} functions if at least {desired} are available; "
            f"otherwise return all available (up to {max_items})."
        )

    return f"""
### SYSTEM BLOCK (LISTING)
MUST:
- Apply query constraints exactly.
- {count_rule}
- Use OUTPUT FORMAT exactly.
SHOULD:
- Rank strongest matches first.
NICE TO HAVE:
- Mark uncertain matches as `[NEEDS REVIEW]`.

Constraints:
- {_build_constraint_text(analysis)}

### OUTPUT FORMAT (STRICT)
1. **`function_name`**
- File: `path:start-end`
- Signature: `full signature`
- Why it matches: one concise reason tied to query{fuzz_line}{summary_line}
""".strip()


def _build_example_system_block():
    return """
### SYSTEM BLOCK (EXAMPLE_GENERATION)
MUST:
- Use the exact target function signature from context (name, arity, types).
- Ground example in real caller/callee usage from context when available.
- Include a short "Evidence from codebase" note.
- Evidence must include at least one `File: path:start-end` and one `Signature: ...`.
- If query requests argv/file bytes, construct arguments from `argv[1]` bytes.
- If caller evidence is missing, state that explicitly and provide minimal safe scaffold.
SHOULD:
- Make the snippet compilable and realistic for the shown signature.
NICE TO HAVE:
- Reuse naming/order patterns from observed call-sites.

### OUTPUT FORMAT (STRICT)
```cpp
<compilable example>
```
Evidence from codebase:
- File: `path:start-end`
- Signature: `full signature`
- File: `path:start-end` or `not found in provided context`
- Signature: `full signature` or `not found in provided context`
- Observed call: `callee(arg1, arg2)` or `not found in provided context`
""".strip()


def _build_explanation_system_block():
    return """
### SYSTEM BLOCK (IMPLEMENTATION_EXPLANATION)
MUST:
- Explain only behavior visible in provided code.
- Quote concrete evidence via file/line references.
- State unknowns explicitly instead of guessing.
SHOULD:
- Keep explanation step-wise and compact.
NICE TO HAVE:
- Mention notable edge cases and tradeoffs.

### OUTPUT FORMAT (STRICT)
- Purpose: <what function does>
- Input handling: <how inputs are validated/parsed>
- Core flow: <main steps>
- Error handling: <checks/fail paths>
- Return value: <what is returned and when>
- Evidence: `path:start-end`
""".strip()


def _build_parameter_analysis_system_block():
    return """
### SYSTEM BLOCK (PARAMETER_ANALYSIS)
MUST:
- Focus on the target function from context.
- For each parameter, explain expected data format/type and likely semantic role.
- Ground every claim in one of: signature, observed call-site usage, docs/guide hints.
- If meaning is unknown, explicitly mark as "unknown from provided context".
SHOULD:
- Distinguish input buffer/size/pointer/output/config/path parameters.
- Mention where parameter values usually come from in code.
NICE TO HAVE:
- Mention boundary/validation implications useful for fuzzing.

### OUTPUT FORMAT (STRICT)
Function: `name`
Signature: `full signature`
Parameters:
1. `param` (`type`)
- Role: ...
- Expected format: ...
- Typical source: ...
- Evidence: `path:start-end` (and/or docs hint)
Unknowns:
- ...
""".strip()


def _build_generic_system_block(analysis, max_items):
    if analysis.get("needs_type_info"):
        return f"""
### SYSTEM BLOCK (TYPE_SPECIFIC)
MUST:
- Return at most {max_items} functions.
- Match requested parameter types exactly from signature.
- Use OUTPUT FORMAT exactly.
SHOULD:
- Prefer direct type matches over loose semantic matches.
NICE TO HAVE:
- Add one-line type mapping note when query uses natural language type names.

### OUTPUT FORMAT (STRICT)
1. **`function_name`**
- File: `path:start-end`
- Signature: `full signature`
- Why it matches: concise type-based reason
""".strip()

    return f"""
### SYSTEM BLOCK (GENERAL)
MUST:
- Use only current context functions.
- Return at most {max_items} items if listing-like response is needed.
SHOULD:
- Keep answer focused on the question.
NICE TO HAVE:
- Add concise evidence references.
""".strip()


def _build_intent_system_block(analysis, max_items):
    if not analysis:
        return _build_generic_system_block({}, max_items)

    query_type = analysis.get("query_type")
    if query_type == "listing":
        return _build_listing_system_block(analysis, max_items)
    if query_type == "example_generation":
        return _build_example_system_block()
    if query_type == "parameter_analysis":
        return _build_parameter_analysis_system_block()
    if query_type == "implementation_explanation":
        return _build_explanation_system_block()
    return _build_generic_system_block(analysis, max_items)


def _build_thinking_prompt_with_limit(
    frags,
    q,
    analysis=None,
    conversation_history=None,
    code_char_limit=1500,
    example_context=None,
    function_hints=None,
):
    max_items = len(frags)
    ctx = _render_function_context(frags, code_char_limit=code_char_limit)
    history_ctx = _render_history_context(conversation_history)
    example_ctx_block = ""
    if (analysis or {}).get("query_type") in {"example_generation", "parameter_analysis"}:
        example_ctx_block = _render_example_context_facts(example_context)
    function_hints_block = ""
    if (analysis or {}).get("query_type") == "parameter_analysis":
        function_hints_block = _render_function_hints(function_hints)
    policy_block = _build_policy_layers_block()
    intent_block = _build_intent_system_block(analysis or {}, max_items=max_items)
    thinking_instructions = """
### RESPONSE MODE
Think silently. Do not output chain-of-thought.
Return only final answer in the required format.
""".strip()

    return f"""You are an expert code analyst with deep understanding of codebases.

{history_ctx}
### Current Question: {q}

### Available Functions from Codebase:
{ctx}
{example_ctx_block}
{function_hints_block}
{policy_block}

{intent_block}

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


def _build_prompt_with_limits(
    frags,
    q,
    analysis=None,
    conversation_history=None,
    code_char_limit_thinking=1500,
    code_char_limit_simple=2000,
    example_context=None,
    function_hints=None,
):
    complex_types = ["listing", "example_generation", "parameter_analysis", "implementation_explanation", "type_specific", "input_specific"]
    use_thinking = (analysis and analysis["query_type"] in complex_types) or (conversation_history and len(conversation_history) > 0)

    if use_thinking:
        return _build_thinking_prompt_with_limit(
            frags,
            q,
            analysis=analysis,
            conversation_history=conversation_history,
            code_char_limit=code_char_limit_thinking,
            example_context=example_context,
            function_hints=function_hints,
        )

    return _build_simple_prompt_with_limit(frags, q, code_char_limit=code_char_limit_simple)


def build_prompt(
    frags,
    q,
    analysis=None,
    conversation_history=None,
    max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS,
    example_context=None,
    function_hints=None,
):
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
            example_context=example_context,
            function_hints=function_hints,
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
            example_context=example_context,
            function_hints=function_hints,
        )
        last_prompt = prompt
        if len(prompt) <= max_prompt_chars:
            return prompt

    if len(last_prompt) <= max_prompt_chars:
        return last_prompt

    return last_prompt[:max_prompt_chars]
