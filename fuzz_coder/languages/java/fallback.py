#!/usr/bin/env python3
"""Java deterministic fallback example generator."""

from __future__ import annotations

import re

from fuzz_coder.ask.example_context import build_example_context as _build_example_context_impl

from ..c_cpp.fallback import (
    _format_file_loc,
    _next_unique_name,
    _normalize_inline_snippet,
    is_valid_function_chunk,
)


def _infer_java_nominal_type(type_text):
    t = str(type_text or "").strip()
    if not t:
        return None
    t = re.sub(r"<[^>]*>", "", t)
    t = t.replace("[]", "").strip()
    tokens = re.findall(r"[A-Za-z_]\w*", t)
    if not tokens:
        return None
    return tokens[-1]


def _java_is_primitive_type(type_text):
    t = str(type_text or "").strip().lower()
    return t in {"byte", "short", "int", "long", "float", "double", "boolean", "char"}


def _java_default_value_expr(type_text, param_name, needs_file):
    t = str(type_text or "").strip()
    t_lower = t.lower()
    pname = str(param_name or "").lower()

    if "byte[]" in t_lower:
        return "inputBytes" if needs_file else "new byte[0]"
    if "string" in t_lower:
        return "new String(inputBytes)" if needs_file else "\"\""
    if t_lower in {"boolean", "bool"}:
        return "false"
    if t_lower in {"char"}:
        return "'\\0'"
    if any(x in t_lower for x in ["int", "long", "short", "integer", "size"]):
        if needs_file and any(k in pname for k in ["size", "len", "length", "count", "bytes", "n"]):
            return "inputBytes.length"
        return "0"
    if any(x in t_lower for x in ["float", "double"]):
        return "0.0"
    if "list<" in t_lower:
        return "new java.util.ArrayList<>()"
    if "set<" in t_lower:
        return "new java.util.HashSet<>()"
    if "map<" in t_lower:
        return "new java.util.HashMap<>()"
    if t_lower.endswith("[]"):
        nominal = _infer_java_nominal_type(t) or "Object"
        return f"new {nominal}[0]"

    nominal = _infer_java_nominal_type(t)
    if nominal in {"Byte", "Short", "Integer", "Long"}:
        return "0"
    if nominal in {"Float", "Double"}:
        return "0.0"
    if nominal in {"Boolean"}:
        return "false"
    if nominal in {"Character"}:
        return "'\\0'"
    if nominal and nominal not in {"Object", "Void"}:
        return f"new {nominal}()"
    return "null"


def _java_pick_callee_expr(target_name, observed_call_expr):
    observed = str(observed_call_expr or "").strip()
    if observed:
        m = re.match(r"\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", observed)
        if m:
            callee = str(m.group(1) or "").strip()
            if callee.split(".")[-1] == target_name:
                return callee
    return target_name


def _build_java_param_binding(param, analysis, used_names):
    pname = str(param.get("name", "")).strip() or "arg"
    ptype = str(param.get("type", "")).strip() or "Object"
    var = _next_unique_name(pname, used_names)
    expr = _java_default_value_expr(ptype, pname, bool((analysis or {}).get("needs_file")))

    if expr == "null" and _java_is_primitive_type(ptype):
        expr = _java_default_value_expr(ptype, pname, False)

    return [f"{ptype} {var} = {expr};"], var


def build_example_answer_from_context_java(frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
    _ = symbols
    _ = type_init_index
    analysis = analysis or {}
    clean_frags = [f for f in frags if is_valid_function_chunk(f)]
    if not clean_frags:
        return "Not enough verified context to build a reliable example."

    if example_context is None:
        example_context = _build_example_context_impl(
            clean_frags,
            analysis=analysis,
            language_name="java",
        )

    target = example_context.get("target")
    if target is None or target not in clean_frags:
        target = clean_frags[0]
    target_name = str(target.get("name", "")).strip() or "targetFunction"
    target_sig = target.get("signature") or f"{target_name}()"

    caller = example_context.get("caller")
    if caller is not None and caller not in clean_frags:
        caller = None
    observed_call_obj = example_context.get("observed_call") or {}
    observed_call = observed_call_obj.get("expr")
    observed_call_inline = _normalize_inline_snippet(observed_call)
    params = list(target.get("parameters") or [])

    include_lines = [
        "import java.nio.file.Files;",
        "import java.nio.file.Path;",
    ]

    needs_file_bytes = bool(analysis.get("needs_file"))
    body_lines = []
    if needs_file_bytes:
        body_lines.extend([
            "if (args.length < 1) {",
            "    System.err.println(\"Usage: ExampleHarness <input_file>\");",
            "    return;",
            "}",
            "",
            "byte[] inputBytes = Files.readAllBytes(Path.of(args[0]));",
            "",
        ])
    else:
        body_lines.extend([
            "byte[] inputBytes = new byte[0];",
            "",
        ])

    used_names = set()
    arg_exprs = []
    for p in params:
        decl_lines, arg_expr = _build_java_param_binding(p, analysis, used_names)
        body_lines.extend(decl_lines)
        arg_exprs.append(arg_expr)

    if params:
        body_lines.append("")

    callee_expr = _java_pick_callee_expr(target_name, observed_call_inline)
    args_joined = ", ".join(arg_exprs)
    sig_head = target_sig.split(target_name, 1)[0]
    is_void_return = re.search(r"\bvoid\s*$", sig_head.strip()) is not None
    if is_void_return:
        body_lines.append(f"{callee_expr}({args_joined});")
    else:
        body_lines.append(f"Object result = {callee_expr}({args_joined});")
        body_lines.append("if (result != null) {")
        body_lines.append("    result.toString();")
        body_lines.append("}")

    code_lines = []
    code_lines.extend(include_lines)
    code_lines.append("")
    code_lines.append(f"// Target signature (from indexed code): {_normalize_inline_snippet(target_sig)}")
    if observed_call_inline:
        code_lines.append(f"// Observed call pattern in codebase: {observed_call_inline}")
    code_lines.append("public class ExampleHarness {")
    code_lines.append("    public static void main(String[] args) throws Exception {")
    code_lines.extend([f"        {ln}" if ln else "" for ln in body_lines])
    code_lines.append("    }")
    code_lines.append("}")

    lines = [
        "### Example Code (deterministic context-grounded fallback)",
        "```java",
        "\n".join(code_lines),
        "```",
        "",
        f"Target function: `{target_name}`",
        "Evidence from codebase:",
        f"- File: `{_format_file_loc(target)}`",
        f"- Signature: `{target_sig}`",
    ]
    if caller is not None:
        caller_sig = caller.get("signature") or f"{caller.get('name', 'caller')}()"
        lines.append(f"- File: `{_format_file_loc(caller)}`")
        lines.append(f"- Signature: `{caller_sig}`")
        if observed_call_inline:
            lines.append(f"- Observed call: `{observed_call_inline}`")
    else:
        lines.append("- Note: direct caller context for target function was not found in selected fragments.")

    return "\n".join(lines)
