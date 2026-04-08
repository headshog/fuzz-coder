#!/usr/bin/env python3
"""Shared deterministic fallback APIs for ask pipeline.

Language-specific fallback generation lives in `fuzz_coder.languages`.
"""

from __future__ import annotations

import re

from fuzz_coder.languages.example_fallback_generators import (
    build_example_answer_from_context as _build_example_answer_from_context_impl,
)
from fuzz_coder.languages.registry import infer_ask_language_from_fragments

from .example_context import build_example_context as _build_example_context_impl
from .example_context import _classify_param_shape as _classify_param_shape_impl


def is_valid_function_chunk(chunk):
    if not isinstance(chunk, dict):
        return False
    required = ["name", "signature", "file", "start_line", "end_line"]
    if any(k not in chunk for k in required):
        return False

    name = str(chunk.get("name", "")).strip()
    signature = str(chunk.get("signature", "")).strip()
    if not name or not signature:
        return False

    bad_name_tokens = {
        "if", "for", "while", "switch", "catch", "class", "struct", "enum",
        "namespace", "return", "sizeof", "typedef", "using", "case", "default",
    }
    if name in bad_name_tokens:
        return False

    if re.search(r"\b(if|for|while|switch|catch)\s*\(", signature):
        return False

    try:
        s = int(chunk.get("start_line", 0) or 0)
        e = int(chunk.get("end_line", 0) or 0)
        if s <= 0 or e < s:
            return False
    except Exception:
        return False

    return True


def _format_file_loc(chunk):
    return f"{chunk.get('file', '')}:{chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}"


def build_example_answer_from_context(frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
    """Dispatch deterministic example fallback to language-specific generator."""
    return _build_example_answer_from_context_impl(
        frags,
        analysis=analysis,
        example_context=example_context,
        symbols=symbols,
        type_init_index=type_init_index,
    )


def _infer_param_role_from_shape(pname, ptype, language_name="c_cpp"):
    shape = _classify_param_shape_impl(ptype, pname, language_name=language_name)
    n = (pname or "").lower()
    if "path" in n or "file" in n:
        return shape, "path/file locator"
    if "ctx" in n:
        return shape, "context/config value"
    if "param" in n or "config" in n:
        return shape, "configuration structure/value"
    if shape == "byte_buffer":
        return shape, "raw input buffer / payload bytes"
    if shape == "size_or_length":
        return shape, "size/length/count for related payload"
    if shape == "out_pointer":
        return shape, "output pointer populated by callee"
    if shape == "pointer":
        return shape, "pointer to object/array for in/out behavior"
    if shape == "reference":
        return shape, "by-reference in/out value"
    if shape == "integer":
        return shape, "numeric control parameter"
    if shape == "float":
        return shape, "floating-point control parameter"
    if shape == "bool":
        return shape, "boolean feature/behavior flag"
    if shape == "string":
        return shape, "text/string input"
    return shape, "unknown from provided context"


def build_parameter_analysis_from_context(frags, analysis=None, example_context=None, function_hints=None):
    """Deterministic fallback for parameter-analysis answers."""
    analysis = analysis or {}
    clean_frags = [f for f in frags if is_valid_function_chunk(f)]
    if not clean_frags:
        return "Not enough verified context to build parameter analysis."

    inferred_language = infer_ask_language_from_fragments(clean_frags, default="c_cpp")
    if example_context is None:
        example_context = _build_example_context_impl(
            clean_frags,
            analysis=analysis,
            language_name=inferred_language,
        )

    target = example_context.get("target")
    if target is None or target not in clean_frags:
        primary = analysis.get("primary_function_name")
        by_name = {f.get("name"): f for f in clean_frags}
        target = by_name.get(primary) if primary else None
    if target is None:
        target = clean_frags[0]

    target_name = str(target.get("name", "")).strip() or "target_function"
    target_sig = target.get("signature") or f"{target_name}()"
    params = list(target.get("parameters") or [])
    caller = example_context.get("caller")
    observed = example_context.get("observed_call") or {}
    observed_args = list(observed.get("args") or [])
    language_name = str(example_context.get("language") or inferred_language or "c_cpp")

    hints_for_target = list((function_hints or {}).get(target_name, []))

    lines = [
        f"Function: `{target_name}`",
        f"Signature: `{target_sig}`",
        "Parameters:",
    ]

    unknowns = []
    for idx, p in enumerate(params, 1):
        pname = str(p.get("name", "")).strip() or f"arg{idx - 1}"
        ptype = str(p.get("type", "")).strip() or "unknown"
        shape, role = _infer_param_role_from_shape(pname, ptype, language_name=language_name)

        expected_fmt = f"{shape}; declared type `{ptype}`"
        src_bits = []
        if idx - 1 < len(observed_args):
            src_bits.append(f"observed caller arg `{observed_args[idx - 1]}`")
        if "path" in pname.lower() or "file" in pname.lower():
            src_bits.append("usually comes from file path/CLI/config")
        if shape in {"byte_buffer", "size_or_length"}:
            src_bits.append("often paired with buffer+size data flow")
        typical_source = "; ".join(src_bits) if src_bits else "not explicit in selected fragments"

        evidence_items = [f"`{_format_file_loc(target)}`"]
        if caller is not None:
            evidence_items.append(f"`{_format_file_loc(caller)}`")
        if hints_for_target:
            h0 = hints_for_target[0]
            evidence_items.append(f"`{h0.get('file', '')}:{h0.get('line', '?')}` (docs/guide)")

        lines.extend([
            f"{idx}. `{pname}` (`{ptype}`)",
            f"- Role: {role}",
            f"- Expected format: {expected_fmt}",
            f"- Typical source: {typical_source}",
            f"- Evidence: {', '.join(evidence_items)}",
        ])

        if role == "unknown from provided context":
            unknowns.append(f"`{pname}` semantic meaning is not explicit in current context")

    lines.append("Unknowns:")
    if unknowns:
        for u in unknowns:
            lines.append(f"- {u}")
    else:
        lines.append("- No critical unknowns from selected context; domain-specific constraints may still exist.")

    if hints_for_target:
        lines.append("Docs/guide hints:")
        for h in hints_for_target[:4]:
            snippet = " ".join(str(h.get("snippet", "")).split())[:260]
            lines.append(
                f"- `{h.get('file', '')}:{h.get('line', '?')}`: {snippet}"
            )

    return "\n".join(lines)
