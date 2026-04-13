#!/usr/bin/env python3
"""Shared deterministic fallback APIs for ask pipeline.

Language-specific fallback generation lives in `fuzz_coder.languages`.
"""

from __future__ import annotations

import re
from collections import defaultdict

from fuzz_coder.languages.registry import get_ask_language_adapter

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


def _resolve_type_name_case_insensitive(type_init_index, requested_name: str) -> str:
    req = str(requested_name or "").strip()
    if not req:
        return ""
    candidates = set()
    raw = type_init_index or {}
    if isinstance(raw, dict):
        for key in ["types", "struct_field_writes", "init_recipes_by_type"]:
            d = raw.get(key)
            if isinstance(d, dict):
                candidates.update(str(k) for k in d.keys())
    if req in candidates:
        return req
    req_l = req.lower()
    for c in sorted(candidates):
        if c.lower() == req_l:
            return c
    for c in sorted(candidates):
        if c.lower().endswith(req_l):
            return c
    return req


def _pick_sample_expr(*values: str) -> str:
    for v in values:
        s = str(v or "").strip()
        if s:
            return s
    return ""


def _format_evidence_refs(evidence_rows, max_refs=3) -> str:
    refs = []
    seen = set()
    for e in list(evidence_rows or []):
        file_path = str((e or {}).get("file", "")).strip()
        line = (e or {}).get("line")
        if not file_path:
            continue
        if isinstance(line, int) and line > 0:
            ref = f"{file_path}:{line}"
        else:
            ref = file_path
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
        if len(refs) >= max_refs:
            break
    if not refs:
        return "unknown from provided context"
    return ", ".join(f"`{r}`" for r in refs)


def _infer_field_role(field_path: str) -> str:
    p = str(field_path or "").lower()
    if not p:
        return "unknown from provided context"
    if any(x in p for x in ["size", "len", "length", "count", "num", "n_"]):
        return "size/length/count metadata"
    if any(x in p for x in ["path", "file", "name", "uri"]):
        return "path/file/name metadata"
    if any(x in p for x in ["buf", "data", "ptr", "bytes", "payload"]):
        return "buffer/data pointer or payload holder"
    if any(x in p for x in ["err", "error", "status", "code"]):
        return "error/status code"
    if any(x in p for x in ["flag", "enable", "valid", "ready", "done"]):
        return "boolean/feature-state flag"
    if any(x in p for x in ["id", "index", "idx"]):
        return "identifier/index metadata"
    return "state/config field used by related functions"


def _infer_value_format(expr: str) -> str:
    e = str(expr or "").strip()
    if not e:
        return "unknown from provided context"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", e):
        return "numeric literal/value"
    if e in {"NULL", "null", "nullptr"}:
        return "nullable pointer/reference"
    if e in {"true", "false", "TRUE", "FALSE"}:
        return "boolean value"
    if re.search(r"\".*\"", e):
        return "string/text value"
    if re.search(r"[A-Za-z_]\w*\s*\(", e):
        return "value produced by helper/API call"
    if e.startswith("{") and e.endswith("}"):
        return "aggregate/default-initializer value"
    return "derived expression/value"


def build_type_analysis_context(frags, analysis=None, type_init_index=None):
    analysis = analysis or {}
    raw = type_init_index or {}
    type_names = list(analysis.get("type_names") or [])
    primary = str(analysis.get("primary_type_name") or (type_names[0] if type_names else "")).strip()
    if not primary:
        return {"target_type": "", "resolved_type": "", "fields": [], "init_patterns": [], "usage": []}

    resolved = _resolve_type_name_case_insensitive(raw, primary)
    types_map = raw.get("types") if isinstance(raw.get("types"), dict) else {}
    struct_writes_map = raw.get("struct_field_writes") if isinstance(raw.get("struct_field_writes"), dict) else {}
    recipes_map = raw.get("init_recipes_by_type") if isinstance(raw.get("init_recipes_by_type"), dict) else {}
    function_effects = raw.get("function_effects") if isinstance(raw.get("function_effects"), dict) else {}
    required_fields = raw.get("required_fields_by_function") if isinstance(raw.get("required_fields_by_function"), dict) else {}

    init_patterns = list(types_map.get(resolved, []))
    struct_writes = list(struct_writes_map.get(resolved, []))
    recipe_rows = list(recipes_map.get(resolved, []))

    field_acc = defaultdict(lambda: {
        "path": "",
        "writes": 0,
        "reads": 0,
        "required_hits": 0,
        "support": 0.0,
        "sample_expr": "",
        "evidence": [],
        "sources": defaultdict(int),
    })

    def _touch_field(path: str):
        key = str(path or "").strip()
        if not key:
            return None
        rec = field_acc[key]
        rec["path"] = key
        return rec

    for row in struct_writes:
        rec = _touch_field(row.get("path", ""))
        if rec is None:
            continue
        rec["writes"] += int(row.get("count", 0) or 0)
        rec["sample_expr"] = _pick_sample_expr(rec.get("sample_expr"), row.get("sample_expr", ""))
        rec["evidence"].extend(list(row.get("evidence") or []))
        rec["sources"]["struct_field_writes"] += int(row.get("count", 0) or 0)

    for recipe in recipe_rows:
        for f in list(recipe.get("fields") or []):
            rec = _touch_field(f.get("path", ""))
            if rec is None:
                continue
            rec["writes"] += int(f.get("count", 0) or 0)
            rec["support"] = max(float(rec.get("support", 0.0)), float(f.get("support", 0.0) or 0.0))
            rec["sample_expr"] = _pick_sample_expr(rec.get("sample_expr"), f.get("sample_expr", ""))
            rec["evidence"].extend(list(f.get("evidence") or []))
            src_total = 0
            for _, sv in dict(f.get("sources") or {}).items():
                try:
                    src_total += int(sv)
                except Exception:
                    pass
            rec["sources"]["init_recipes"] += max(1, src_total)

    usage = []
    for fn, entries in function_effects.items():
        for e in list(entries or []):
            if str(e.get("type", "")).strip().lower() != resolved.lower():
                continue
            writes = list(e.get("writes") or [])
            reads = list(e.get("reads") or [])
            usage.append({
                "function": fn,
                "signature": str(e.get("signature", "")),
                "arg_index": int(e.get("arg_index", 0) or 0),
                "arg_name": str(e.get("arg_name", "")),
                "writes_count": len(writes),
                "reads_count": len(reads),
            })
            for w in writes:
                rec = _touch_field(w.get("path", ""))
                if rec is None:
                    continue
                rec["writes"] += int(w.get("count", 0) or 0)
                rec["sample_expr"] = _pick_sample_expr(rec.get("sample_expr"), w.get("sample_expr", ""))
                rec["evidence"].extend(list(w.get("evidence") or []))
                rec["sources"]["function_effects_write"] += int(w.get("count", 0) or 0)
            for r in reads:
                rec = _touch_field(r.get("path", ""))
                if rec is None:
                    continue
                rec["reads"] += int(r.get("count", 0) or 0)
                rec["evidence"].extend(list(r.get("evidence") or []))
                rec["sources"]["function_effects_read"] += int(r.get("count", 0) or 0)

    for fn, entries in required_fields.items():
        for e in list(entries or []):
            if str(e.get("type", "")).strip().lower() != resolved.lower():
                continue
            for f in list(e.get("fields") or []):
                rec = _touch_field(f.get("path", ""))
                if rec is None:
                    continue
                if bool(f.get("required")):
                    rec["required_hits"] += 1
                rec["support"] = max(float(rec.get("support", 0.0)), float(f.get("support", 0.0) or 0.0))
                rec["sample_expr"] = _pick_sample_expr(rec.get("sample_expr"), f.get("sample_expr", ""))
                rec["evidence"].extend(list(f.get("evidence") or []))
                rec["sources"]["required_fields"] += int(f.get("count", 0) or 0)

    # Additional weak evidence from retrieved fragments signatures.
    type_pat = re.compile(rf"\b{re.escape(resolved)}\b")
    for f in list(frags or []):
        if not isinstance(f, dict):
            continue
        sig = str(f.get("signature", ""))
        if type_pat.search(sig):
            usage.append({
                "function": str(f.get("name", "")),
                "signature": sig,
                "arg_index": -1,
                "arg_name": "",
                "writes_count": 0,
                "reads_count": 0,
            })

    fields = list(field_acc.values())
    fields.sort(
        key=lambda x: (
            int(x.get("required_hits", 0)),
            float(x.get("support", 0.0)),
            int(x.get("writes", 0)) + int(x.get("reads", 0)),
            len(str(x.get("path", ""))),
        ),
        reverse=True,
    )
    usage.sort(
        key=lambda x: (
            int(x.get("writes_count", 0)) + int(x.get("reads_count", 0)),
            str(x.get("function", "")),
        ),
        reverse=True,
    )
    return {
        "target_type": primary,
        "resolved_type": resolved,
        "fields": fields[:32],
        "init_patterns": list(init_patterns)[:12],
        "usage": usage[:24],
        "struct_writes_count": len(struct_writes),
        "recipe_count": len(recipe_rows),
    }


def build_type_analysis_from_context(frags, analysis=None, type_init_index=None):
    analysis = analysis or {}
    inferred_language = infer_ask_language_from_fragments(frags or [], default="c_cpp")
    adapter = get_ask_language_adapter(inferred_language)

    ctx = build_type_analysis_context(
        frags,
        analysis=analysis,
        type_init_index=type_init_index,
    )
    target = str(ctx.get("target_type", "")).strip()
    resolved = str(ctx.get("resolved_type", "")).strip()
    if not target:
        return "Not enough verified context to build type/class analysis."

    fields = list(ctx.get("fields") or [])
    init_patterns = list(ctx.get("init_patterns") or [])
    usage = list(ctx.get("usage") or [])

    lines = [
        f"Type: `{target}`",
        f"Resolved type key: `{resolved or target}`",
        "Field Semantics:",
    ]

    unknowns = []
    if fields:
        for i, f in enumerate(fields[:16], 1):
            path = str(f.get("path", "")).strip()
            if not path:
                continue
            sample_expr = str(f.get("sample_expr", "")).strip()
            role = _infer_field_role(path)
            fmt = _infer_value_format(sample_expr)
            writes = int(f.get("writes", 0) or 0)
            reads = int(f.get("reads", 0) or 0)
            required_hits = int(f.get("required_hits", 0) or 0)
            if writes > 0 and reads > 0:
                direction = "inout"
            elif writes > 0:
                direction = "output/state-mutation"
            elif reads > 0:
                direction = "input/read-mostly"
            else:
                direction = "unknown from provided context"
            if required_hits > 0:
                direction += "; often required before some calls"
            evidence = _format_evidence_refs(f.get("evidence", []), max_refs=3)
            lines.extend([
                f"{i}. `{path}`",
                f"- Role: {role}",
                f"- Expected format/range: {fmt}",
                f"- Direction: {direction}",
                f"- Sample value/expression: `{sample_expr or 'unknown from provided context'}`",
                f"- Evidence: {evidence}",
            ])
    else:
        lines.append("- unknown from provided context")
        unknowns.append("No field-write/read evidence for this type in the current index.")

    lines.append("Observed Initialization Patterns:")
    if init_patterns:
        for p in init_patterns[:8]:
            expr = str(p.get("expr", "")).strip() or "unknown"
            kind = str(p.get("kind", "")).strip() or "unknown"
            ev = _format_evidence_refs(p.get("evidence", []), max_refs=2)
            lines.append(f"- {kind}: `{expr}`; evidence: {ev}")
    else:
        lines.append("- unknown from provided context")
        unknowns.append("No initialization patterns were recorded for this type.")

    lines.append("Typical Usage in Codebase:")
    if usage:
        seen = set()
        for u in usage[:10]:
            fn = str(u.get("function", "")).strip()
            if not fn or fn in seen:
                continue
            seen.add(fn)
            sig = str(u.get("signature", "")).strip()
            if sig:
                lines.append(f"- `{fn}`: `{sig}`")
            else:
                lines.append(f"- `{fn}`")
    else:
        lines.append("- unknown from provided context")
        unknowns.append("No usage sites with this type were identified in retrieved context.")

    lines.append("Unknowns:")
    if unknowns:
        for u in unknowns:
            lines.append(f"- {u}")
    else:
        lines.append("- No critical unknowns from provided context; semantic intent is still inferred heuristically.")

    # Keep deterministic mention of inferred language adapter for traceability.
    lines.append(f"Language adapter: `{adapter.name}`")
    return "\n".join(lines)
