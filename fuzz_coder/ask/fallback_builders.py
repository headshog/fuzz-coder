#!/usr/bin/env python3
"""Deterministic fallback builders for example and parameter-analysis intents."""

import os
import re

from .example_context import build_example_context as _build_example_context_impl
from .example_context import _classify_param_shape as _classify_param_shape_impl


PATH_PARAM_NAME_HINTS = [
    "path", "file", "filename", "fname", "filepath", "dir", "directory",
]


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

def _find_matching_paren_text(text, open_idx):
    """Find matching ')' for '(' with nested bracket and literal awareness."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "(":
        return -1

    depth = 0
    i = open_idx
    n = len(text)
    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_str:
            if not escape and ch == '"':
                in_str = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if in_char:
            if not escape and ch == "'":
                in_char = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_str = True
            escape = False
            i += 1
            continue
        if ch == "'":
            in_char = True
            escape = False
            i += 1
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


def _split_top_level_arguments(args_text):
    out = []
    cur = []
    depth_angle = 0
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    in_str = False
    in_char = False
    escape = False

    for ch in args_text:
        if in_str:
            cur.append(ch)
            if not escape and ch == '"':
                in_str = False
            escape = (ch == "\\" and not escape)
            continue

        if in_char:
            cur.append(ch)
            if not escape and ch == "'":
                in_char = False
            escape = (ch == "\\" and not escape)
            continue

        if ch == '"':
            in_str = True
            escape = False
            cur.append(ch)
            continue
        if ch == "'":
            in_char = True
            escape = False
            cur.append(ch)
            continue

        if ch == "<":
            depth_angle += 1
            cur.append(ch)
            continue
        if ch == ">":
            depth_angle = max(0, depth_angle - 1)
            cur.append(ch)
            continue
        if ch == "(":
            depth_paren += 1
            cur.append(ch)
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            cur.append(ch)
            continue
        if ch == "{":
            depth_brace += 1
            cur.append(ch)
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            cur.append(ch)
            continue
        if ch == "[":
            depth_bracket += 1
            cur.append(ch)
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            cur.append(ch)
            continue

        if ch == "," and depth_angle == 0 and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
            part = "".join(cur).strip()
            if part:
                out.append(part)
            cur = []
            continue

        cur.append(ch)

    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _extract_call_argument_lists(code, target_name, limit=3):
    if not code or not target_name:
        return []

    pattern = re.compile(
        rf"(?<![A-Za-z0-9_~])(?:[A-Za-z_]\w*::)*{re.escape(target_name)}\s*\("
    )
    out = []
    seen = set()
    for m in pattern.finditer(code):
        open_idx = code.find("(", m.start())
        if open_idx == -1:
            continue
        close_idx = _find_matching_paren_text(code, open_idx)
        if close_idx == -1:
            continue

        args_text = code[open_idx + 1:close_idx]
        args = _split_top_level_arguments(args_text)
        expr = code[m.start():close_idx + 1].strip()

        key = (expr, tuple(args))
        if key in seen:
            continue
        seen.add(key)
        out.append({"expr": expr, "args": args})
        if len(out) >= limit:
            break

    return out


def _base_decl_type(type_text):
    t = re.sub(r"\s+", " ", str(type_text or "")).strip()
    if not t:
        return "int"

    # Remove top-level cv and ref qualifiers without touching template internals.
    t = re.sub(r"^\s*const\s+", "", t)
    t = re.sub(r"\s+const\s*$", "", t)
    t = re.sub(r"\s*&&\s*$", "", t)
    t = re.sub(r"\s*&\s*$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or "int"


def _pointee_decl_type(type_text):
    t = _base_decl_type(type_text)
    while True:
        stripped = re.sub(r"\s+", " ", t).strip()
        if not stripped.endswith("*"):
            return stripped or "int"
        t = stripped[:-1].strip()


def _value_decl_type(type_text):
    return _base_decl_type(type_text)


def _strip_angle_template_content(type_text):
    s = type_text or ""
    out = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def _infer_effective_param_type(param, pname):
    ptype = re.sub(r"\s+", " ", str(param.get("type", "")).strip())
    raw = re.sub(r"\s+", " ", str(param.get("raw", "")).strip())
    if not raw:
        return ptype

    raw = raw.split("=", 1)[0].strip()
    if pname:
        raw = re.sub(rf"\b{re.escape(pname)}\b\s*$", "", raw).strip()

    if not ptype:
        return raw
    if ("*" in raw and "*" not in ptype) or ("&" in raw and "&" not in ptype):
        return raw
    if len(raw) > len(ptype):
        return raw
    return ptype


def _next_unique_name(base, used):
    name = re.sub(r"\W+", "_", base or "").strip("_")
    if not name:
        name = "arg"
    if name[0].isdigit():
        name = f"v_{name}"
    candidate = name
    idx = 2
    while candidate in used:
        candidate = f"{name}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def _is_simple_literal(expr):
    if not expr:
        return False
    return re.fullmatch(
        r"(?:[-+]?\d+(?:\.\d+)?(?:[uUlLfF]*)|nullptr|NULL|true|false|'.*'|\".*\")",
        expr.strip(),
    ) is not None


def _is_simple_identifier(expr):
    return re.fullmatch(r"[A-Za-z_]\w*", (expr or "").strip()) is not None


def _is_file_size_param(type_text, name):
    t = (type_text or "").lower()
    n = (name or "").lower()
    if any(k in n for k in ["size", "len", "length", "count", "bytes", "n"]):
        return any(k in t for k in [
            "size_t", "ssize_t", "int", "long", "uint", "int32", "int64", "uint32", "uint64",
        ])
    return False


def _is_size_type(type_text):
    t = (type_text or "").lower()
    return "size_t" in t or "ssize_t" in t


def _is_path_like_param_name(name):
    n = str(name or "").lower()
    return any(k in n for k in ["path", "file", "filename", "fname", "filepath", "model"])


def _pointer_depth(type_text):
    return str(type_text or "").count("*")


def _extract_nominal_type_name(type_text):
    t = _pointee_decl_type(type_text)
    tokens = re.findall(r"[A-Za-z_]\w*", t)
    if not tokens:
        return None
    skip = {
        "const", "volatile", "restrict", "__restrict__", "struct", "class", "enum",
        "unsigned", "signed", "long", "short", "int", "float", "double", "bool", "void",
        "size_t", "ssize_t", "auto", "typename",
    }
    for tok in reversed(tokens):
        if tok.lower() not in skip:
            return tok
    return None


def _split_camel_type_tokens(type_name):
    return re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", str(type_name or ""))


def _candidate_allocators_for_type(type_name):
    nominal = str(type_name or "").strip()
    if not nominal:
        return []

    bases = []
    seen = set()

    def _add_base(x):
        b = str(x or "").strip().lower()
        if not b or b in seen:
            return
        seen.add(b)
        bases.append(b)

    lowered = nominal.lower()
    if lowered.endswith("context") and len(lowered) > len("context"):
        _add_base(lowered[: -len("context")])

    parts = [p.lower() for p in _split_camel_type_tokens(nominal) if p]
    if parts and parts[-1] in {"context", "ctx", "state", "config", "params", "param"}:
        parts = parts[:-1]
    if parts:
        _add_base("".join(parts))

    out = []
    seen_fn = set()
    for base in bases:
        fn = f"{base}_alloc_context"
        if fn not in seen_fn:
            seen_fn.add(fn)
            out.append(fn)
    return out


def _build_symbol_lookup(symbols):
    lookup = {}
    for fn in (symbols or {}).keys():
        if isinstance(fn, str) and fn:
            lookup.setdefault(fn.lower(), fn)
    return lookup


def _build_type_init_lookup(type_init_index):
    lookup = {}
    raw = type_init_index or {}
    if isinstance(raw, dict) and isinstance(raw.get("types"), dict):
        raw = raw.get("types", {})

    for k, v in (raw or {}).items():
        if not isinstance(k, str) or not k:
            continue
        if not isinstance(v, list):
            continue
        lookup.setdefault(k.lower(), [])
        for item in v:
            if isinstance(item, dict):
                lookup[k.lower()].append(item)
    return lookup


def _build_required_field_lookup(type_init_index):
    raw = type_init_index or {}
    if not isinstance(raw, dict):
        return {}
    by_function = raw.get("required_fields_by_function")
    if not isinstance(by_function, dict):
        return {}

    out = {}
    for fn, entries in by_function.items():
        if not isinstance(fn, str) or not fn:
            continue
        if not isinstance(entries, list):
            continue
        clean_entries = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            if not isinstance(e.get("fields"), list):
                continue
            clean_entries.append(e)
        if clean_entries:
            out[fn] = clean_entries
    return out


def _build_function_effects_lookup(type_init_index):
    raw = type_init_index or {}
    if not isinstance(raw, dict):
        return {}
    by_function = raw.get("function_effects")
    if not isinstance(by_function, dict):
        return {}
    out = {}
    for fn, entries in by_function.items():
        if not isinstance(fn, str) or not fn:
            continue
        if not isinstance(entries, list):
            continue
        clean = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            clean.append(e)
        if clean:
            out[fn] = clean
    return out


def _extract_bound_base_for_field_init(arg_expr, ptype):
    expr = str(arg_expr or "").strip()
    if not expr:
        return None, None

    is_pointer = "*" in str(ptype or "")
    op = "->" if is_pointer else "."

    m = re.fullmatch(r"&\s*([A-Za-z_]\w*)", expr)
    if m:
        return m.group(1), "."
    m = re.fullmatch(r"([A-Za-z_]\w*)", expr)
    if m:
        return m.group(1), op
    m = re.fullmatch(r"\*+\s*([A-Za-z_]\w*)", expr)
    if m:
        return m.group(1), "->"
    return None, None


def _default_rhs_for_field_path(path, analysis):
    p = str(path or "").lower()
    needs_file = bool((analysis or {}).get("needs_file"))
    if needs_file and any(k in p for k in ["path", "file", "filename", "model"]):
        return "argv[1]"
    if needs_file and any(k in p for k in ["size", "length", "len", "count", "nbytes", "bytes"]):
        return "static_cast<size_t>(input_bytes.size())"
    if needs_file and any(k in p for k in ["data", "buf", "buffer", "ptr", "bytes"]):
        return "reinterpret_cast<const uint8_t *>(input_bytes.data())"
    if any(k in p for k in ["enable", "enabled", "use_", "has_", "flag"]):
        return "false"
    return "{}"


def _is_self_contained_field_expr(expr):
    e = str(expr or "").strip()
    if not e:
        return False
    if re.search(r"\bargv\b|\bargc\b|\binput_bytes\b", e):
        return True
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", e):
        return True
    if re.fullmatch(r"(?:[A-Za-z_]\w*::)*[A-Z_][A-Z0-9_]*", e):
        return True
    if re.fullmatch(r"[-+*/(){}\[\],\sA-Za-z0-9_:.><=&|^!~'\"`]+", e):
        tokens = re.findall(r"[A-Za-z_]\w*", e)
        if not tokens:
            return True
        blacklist = {
            "this", "ctx", "state", "config", "cfg", "tmp", "it", "iter", "result",
            "request", "response", "res", "req", "self",
        }
        return all(t.lower() not in blacklist for t in tokens)
    return False


def _infer_required_field_hints(params, target_name, type_init_index=None):
    by_fn = _build_required_field_lookup(type_init_index)
    effects_by_fn = _build_function_effects_lookup(type_init_index)
    entries = list(by_fn.get(str(target_name or "").strip(), []))

    out = {}
    for i, p in enumerate(params or []):
        ptype = _infer_effective_param_type(p, str(p.get("name", "")).strip() or f"arg{i}")
        nominal = _extract_nominal_type_name(ptype)
        if not nominal:
            continue

        candidates = []
        for e in entries:
            if int(e.get("arg_index", -1)) != i:
                continue
            etype = str(e.get("type", "")).strip()
            if etype and etype.lower() != nominal.lower():
                continue
            candidates.append(e)

        fields = []
        if candidates:
            best = sorted(
                candidates,
                key=lambda x: int(x.get("call_sites", 0)),
                reverse=True,
            )[0]
            for f in list(best.get("fields") or []):
                if not isinstance(f, dict):
                    continue
                path = str(f.get("path", "")).strip()
                if not path:
                    continue
                fields.append(f)

        if not fields:
            for e in list(effects_by_fn.get(str(target_name or "").strip(), [])):
                if int(e.get("arg_index", -1)) != i:
                    continue
                etype = str(e.get("type", "")).strip()
                if etype and etype.lower() != nominal.lower():
                    continue
                for w in list(e.get("writes") or []):
                    path = str((w or {}).get("path", "")).strip()
                    if not path:
                        continue
                    fields.append({
                        "path": path,
                        "access": str((w or {}).get("access", "dot")),
                        "count": int((w or {}).get("count", 1)),
                        "support": 0.35,
                        "required": False,
                        "sample_expr": str((w or {}).get("sample_expr", "")).strip(),
                        "self_contained": bool((w or {}).get("sample_expr")),
                        "sources": {"function_effect_write": 1},
                        "evidence": list((w or {}).get("evidence", [])),
                    })
                for r in list(e.get("reads") or []):
                    path = str((r or {}).get("path", "")).strip()
                    if not path:
                        continue
                    fields.append({
                        "path": path,
                        "access": str((r or {}).get("access", "dot")),
                        "count": int((r or {}).get("count", 1)),
                        "support": 0.25,
                        "required": False,
                        "sample_expr": "",
                        "self_contained": False,
                        "sources": {"function_effect_read": 1},
                        "evidence": list((r or {}).get("evidence", [])),
                    })

        if fields:
            dedup = {}
            for f in fields:
                k = (str(f.get("path", "")), str(f.get("access", "dot")))
                if not k[0]:
                    continue
                cur = dedup.get(k)
                if cur is None:
                    dedup[k] = f
                    continue
                if float(f.get("support", 0.0)) > float(cur.get("support", 0.0)):
                    dedup[k] = f
            out[i] = sorted(
                dedup.values(),
                key=lambda x: (bool(x.get("required")), float(x.get("support", 0.0)), int(x.get("count", 0))),
                reverse=True,
            )
    return out


def _build_required_field_init_lines(required_field_hints, params, arg_exprs, analysis):
    lines = []
    seen = set()
    for i, fields in (required_field_hints or {}).items():
        if i >= len(params) or i >= len(arg_exprs):
            continue
        p = params[i]
        pname = str(p.get("name", "")).strip() or f"arg{i}"
        ptype = _infer_effective_param_type(p, pname)
        base, op = _extract_bound_base_for_field_init(arg_exprs[i], ptype)
        if not base or not op:
            continue

        picked = [f for f in fields if bool(f.get("required"))]
        if not picked:
            picked = sorted(
                fields,
                key=lambda x: (float(x.get("support", 0.0)), int(x.get("count", 0))),
                reverse=True,
            )[:2]
        else:
            picked = sorted(
                picked,
                key=lambda x: (float(x.get("support", 0.0)), int(x.get("count", 0))),
                reverse=True,
            )[:4]

        for f in picked:
            path = str(f.get("path", "")).strip()
            if not path:
                continue
            lhs = f"{base}{op}{path}"
            rhs = str(f.get("sample_expr", "")).strip()
            if not _is_self_contained_field_expr(rhs):
                rhs = _default_rhs_for_field_path(path, analysis)
            stmt = f"{lhs} = {rhs};"
            if stmt in seen:
                continue
            seen.add(stmt)
            lines.append(stmt)
    return lines


def _pick_type_init_candidate(entries, want_pointer):
    if not entries:
        return None

    if want_pointer:
        prioritized_kinds = ["pointer_call", "value_call"]
    else:
        prioritized_kinds = ["value_call", "value_brace", "value_default"]

    candidates = []
    for e in entries:
        kind = str(e.get("kind") or "")
        if kind not in prioritized_kinds:
            continue
        expr = str(e.get("expr") or "").strip()
        if not expr and kind != "value_default":
            continue
        if kind in {"pointer_call", "value_call"} and not bool(e.get("self_contained")):
            continue
        score = int(e.get("score", 0)) + int(e.get("count", 0))
        candidates.append((prioritized_kinds.index(kind), -score, len(expr), e))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def _infer_type_init_hints(params, symbols, type_init_index=None):
    lookup = _build_symbol_lookup(symbols)
    type_lookup = _build_type_init_lookup(type_init_index)
    if not lookup:
        lookup = {}

    hints = {}
    for idx, p in enumerate(params or []):
        pname = str(p.get("name", "")).strip() or f"arg{idx}"
        ptype = _infer_effective_param_type(p, pname)
        nominal = _extract_nominal_type_name(ptype)
        if not nominal:
            continue

        is_pointer = _pointer_depth(ptype) == 1

        from_type_index = _pick_type_init_candidate(type_lookup.get(nominal.lower(), []), want_pointer=is_pointer)
        if from_type_index:
            hints[idx] = {
                "kind": str(from_type_index.get("kind") or ""),
                "expr": str(from_type_index.get("expr") or "").strip(),
                "function": str(from_type_index.get("function") or "").strip() or None,
                "nominal_type": nominal,
                "source": "type_init_index",
            }
            continue

        if not is_pointer:
            continue

        for cand in _candidate_allocators_for_type(nominal):
            resolved = lookup.get(cand.lower())
            if resolved:
                hints[idx] = {
                    "kind": "allocator_call",
                    "expr": f"{resolved}()",
                    "function": resolved,
                    "nominal_type": nominal,
                    "source": "symbol_heuristic",
                }
                break
    return hints


def _headers_from_type_init_hints(type_init_hints):
    headers = []
    for hint in (type_init_hints or {}).values():
        fn = str((hint or {}).get("function") or "").lower()
        if fn.startswith("avformat_"):
            headers.append("#include <libavformat/avformat.h>")
        elif fn.startswith("avcodec_"):
            headers.append("#include <libavcodec/avcodec.h>")
        elif fn.startswith("avutil_"):
            headers.append("#include <libavutil/avutil.h>")
        elif fn.startswith("sws_"):
            headers.append("#include <libswscale/swscale.h>")
        elif fn.startswith("swr_"):
            headers.append("#include <libswresample/swresample.h>")

    out = []
    seen = set()
    for h in headers:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _format_file_loc(chunk):
    return f"{chunk.get('file', '')}:{chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}"


def _normalize_inline_snippet(text, max_len=260):
    one_line = " ".join(str(text or "").split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3].rstrip() + "..."


def _extract_hint_var_name(arg_hint):
    hint = (arg_hint or "").strip()
    if not hint:
        return None, False
    if hint.startswith("&") and _is_simple_identifier(hint[1:]):
        return hint[1:], True
    if _is_simple_identifier(hint):
        return hint, False
    return None, False


def _extract_caller_decl_hints(caller_code, observed_args):
    if not caller_code or not observed_args:
        return {}

    names = []
    for arg in observed_args:
        name, _ = _extract_hint_var_name(arg)
        if name and name not in names:
            names.append(name)
    if not names:
        return {}

    lines = str(caller_code).splitlines()
    hints = {}

    def _is_decl_type_like(type_text):
        t = str(type_text or "").strip()
        if not t:
            return False
        lowered = t.lower()
        if any(k in lowered for k in ["return ", "if ", "for ", "while ", "switch ", ","]):
            return False
        if "=" in t or "(" in t or ")" in t:
            return False
        return bool(re.search(r"[A-Za-z_][\w:<>*&\s]*$", t))

    for name in names:
        found = None
        pat_basic = re.compile(
            rf"^\s*(?P<type>.+?)\b{re.escape(name)}\b\s*(?P<init>=\s*.+|\{{.*\}})?\s*;\s*$"
        )
        pat_ctor = re.compile(
            rf"^\s*(?P<type>.+?)\b{re.escape(name)}\b\s*\((?P<ctor>.*)\)\s*;\s*$"
        )
        for line in lines:
            m = pat_basic.match(line)
            if m:
                type_text = (m.group("type") or "").strip()
                if not _is_decl_type_like(type_text):
                    continue
                init = (m.group("init") or "").strip()
                if init.startswith("="):
                    found = {"type": type_text, "init_kind": "equals", "init_expr": init[1:].strip()}
                elif init.startswith("{"):
                    found = {"type": type_text, "init_kind": "brace", "init_expr": init}
                else:
                    found = {"type": type_text, "init_kind": "none", "init_expr": ""}
                continue

            m = pat_ctor.match(line)
            if not m:
                continue
            type_text = (m.group("type") or "").strip()
            if not _is_decl_type_like(type_text):
                continue
            ctor_expr = (m.group("ctor") or "").strip()
            found = {"type": type_text, "init_kind": "paren", "init_expr": ctor_expr}
        if found:
            hints[name] = found

    return hints


def _build_decl_statement(type_text, var_name, init_hint=None):
    t = re.sub(r"\s+", " ", str(type_text or "").strip()) or "int"
    v = str(var_name or "").strip() or "arg"
    if not init_hint:
        return f"{t} {v}{{}};"

    kind = str(init_hint.get("init_kind") or "none")
    expr = str(init_hint.get("init_expr") or "").strip()
    if kind == "equals" and expr:
        return f"{t} {v} = {expr};"
    if kind == "brace" and expr:
        return f"{t} {v}{expr};"
    if kind == "paren":
        return f"{t} {v}({expr});"
    return f"{t} {v}{{}};"


def _extract_observed_arg_base_vars(observed_args):
    bases = []
    seen = set()
    for arg in observed_args or []:
        expr = str(arg or "").strip()
        if not expr:
            continue
        for token in re.findall(r"[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*", expr):
            token = re.sub(r"\s+", "", token)
            if not token:
                continue
            base = token.split("->", 1)[0].split(".", 1)[0]
            for name in [token, base]:
                if name and name not in seen:
                    seen.add(name)
                    bases.append(name)
    return bases


def _extract_caller_flow_statements(caller_code, observed_args, max_lines=8):
    code = str(caller_code or "")
    if not code:
        return []

    relevant = _extract_observed_arg_base_vars(observed_args)
    if not relevant:
        return []

    out = []
    seen = set()
    assign_re = re.compile(
        r"^\s*([A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*)\s*=\s*([^;]+);\s*$"
    )
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line or "==" in line or "+=" in line or "-=" in line or "*=" in line or "/=" in line:
            continue
        m = assign_re.match(line)
        if not m:
            continue
        lhs = re.sub(r"\s+", "", (m.group(1) or "").strip())
        rhs = (m.group(2) or "").strip()
        lhs_base = lhs.split("->", 1)[0].split(".", 1)[0]
        if lhs not in relevant and lhs_base not in relevant:
            continue
        stmt = f"{lhs} = {rhs};"
        if stmt in seen:
            continue
        seen.add(stmt)
        out.append(stmt)
        if len(out) >= max_lines:
            break
    return out


def _build_param_binding(param, analysis, arg_hint, used_names, caller_decl_hints=None, type_init_hint=None):
    pname = str(param.get("name", "")).strip() or "arg"
    ptype = _infer_effective_param_type(param, pname)
    ptype_lower = ptype.lower()
    needs_file = bool((analysis or {}).get("needs_file"))
    ptype_top = _strip_angle_template_content(ptype)
    ptype_top_lower = ptype_top.lower()

    is_pointer = "*" in ptype_top
    is_ref = "&" in ptype_top
    is_std_string = ("std::string" in ptype) or ("string" in ptype_lower)
    is_std_function = "std::function" in ptype_lower
    is_char_or_byte_ptr = is_pointer and not is_std_function and any(x in ptype_top_lower for x in [
        "char *", "char*", "uint8_t *", "uint8_t*", "unsigned char *", "unsigned char*",
        "std::byte *", "std::byte*", "void *", "void*",
    ])

    value_type = _value_decl_type(ptype)
    decls = []

    hint = (arg_hint or "").strip()
    hint_var, hint_is_addr = _extract_hint_var_name(hint)
    hint_decl = (caller_decl_hints or {}).get(hint_var) if hint_var else None
    if hint:
        if hint_is_addr:
            var = _next_unique_name(hint_var, used_names)
            if hint_decl:
                decls.append(_build_decl_statement(hint_decl.get("type"), var, hint_decl))
            else:
                decls.append(f"{_pointee_decl_type(ptype)} {var}{{}};")
            return decls, f"&{var}"

        if _is_simple_literal(hint):
            return decls, hint

        if _is_simple_identifier(hint):
            if hint_decl:
                var = _next_unique_name(hint, used_names)
                decls.append(_build_decl_statement(hint_decl.get("type"), var, hint_decl))
                return decls, var
            if is_pointer:
                storage = _next_unique_name(f"{hint}_obj", used_names)
                ptr_name = _next_unique_name(hint, used_names)
                decls.append(f"{_pointee_decl_type(ptype)} {storage}{{}};")
                decls.append(f"{ptype} {ptr_name} = &{storage};")
                return decls, ptr_name
            if is_std_string and needs_file:
                var = _next_unique_name(hint, used_names)
                decls.append(f"std::string {var}(input_bytes.begin(), input_bytes.end());")
                return decls, var
            var = _next_unique_name(hint, used_names)
            decls.append(f"{value_type} {var}{{}};")
            return decls, var

    if needs_file and is_char_or_byte_ptr and is_pointer and _is_path_like_param_name(pname):
        if any(x in ptype_top_lower for x in ["char *", "char*"]):
            return decls, "argv[1]"

    if needs_file and is_char_or_byte_ptr and is_pointer:
        n = (pname or "").lower()
        if (
            n in {"e", "end", "last", "finish", "to"}
            or n.endswith("_end")
            or n.endswith("end_ptr")
            or n.endswith("last")
        ):
            return decls, f"reinterpret_cast<{ptype}>(input_bytes.data() + input_bytes.size())"
        return decls, f"reinterpret_cast<{ptype}>(input_bytes.data())"

    if needs_file and _is_file_size_param(ptype, pname):
        cast_type = value_type or "size_t"
        return decls, f"static_cast<{cast_type}>(input_bytes.size())"
    if needs_file and _is_size_type(ptype):
        cast_type = value_type or "size_t"
        return decls, f"static_cast<{cast_type}>(input_bytes.size())"

    if is_std_string and needs_file and _is_path_like_param_name(pname):
        var = _next_unique_name(f"{pname}_str", used_names)
        decls.append(f"std::string {var}(argv[1]);")
        return decls, var

    if is_std_string and needs_file:
        var = _next_unique_name(f"{pname}_str", used_names)
        decls.append(f"std::string {var}(input_bytes.begin(), input_bytes.end());")
        return decls, var

    if needs_file and not is_pointer and not is_ref and "char" in ptype_lower:
        n = (pname or "").lower()
        if any(k in n for k in ["d", "delim", "delimiter", "sep", "separator"]):
            var = _next_unique_name(pname, used_names)
            decls.append(f"{value_type} {var} = '\\n';")
            return decls, var

    hint_kind = str((type_init_hint or {}).get("kind") or "")
    hint_expr = str((type_init_hint or {}).get("expr") or "").strip()
    if is_pointer and _pointer_depth(ptype) == 1 and hint_kind in {"allocator_call", "pointer_call", "value_call"} and hint_expr:
        ptr_name = _next_unique_name(pname, used_names)
        decls.append(f"{ptype} {ptr_name} = {hint_expr};")
        decls.append(f"if (!{ptr_name}) {{")
        decls.append(f"    std::cerr << \"Failed to initialize {pname}\\n\";")
        decls.append("    return 1;")
        decls.append("}")
        return decls, ptr_name

    if (not is_pointer) and hint_kind in {"value_call", "value_brace", "value_default"}:
        var = _next_unique_name(pname, used_names)
        if hint_kind == "value_call" and hint_expr:
            decls.append(f"{value_type} {var} = {hint_expr};")
        elif hint_kind == "value_brace":
            expr = hint_expr or "{}"
            if not expr.startswith("{"):
                expr = "{" + expr + "}"
            decls.append(f"{value_type} {var}{expr};")
        else:
            decls.append(f"{value_type} {var}{{}};")
        return decls, var

    if is_ref and hint_kind in {"value_call", "value_brace", "value_default"}:
        var = _next_unique_name(pname, used_names)
        if hint_kind == "value_call" and hint_expr:
            decls.append(f"{value_type} {var} = {hint_expr};")
        elif hint_kind == "value_brace":
            expr = hint_expr or "{}"
            if not expr.startswith("{"):
                expr = "{" + expr + "}"
            decls.append(f"{value_type} {var}{expr};")
        else:
            decls.append(f"{value_type} {var}{{}};")
        return decls, var

    if is_pointer:
        storage = _next_unique_name(f"{pname}_obj", used_names)
        ptr_name = _next_unique_name(pname, used_names)
        decls.append(f"{_pointee_decl_type(ptype)} {storage}{{}};")
        decls.append(f"{ptype} {ptr_name} = &{storage};")
        return decls, ptr_name

    if is_ref:
        var = _next_unique_name(pname, used_names)
        decls.append(f"{value_type} {var}{{}};")
        return decls, var

    if "bool" in ptype_lower:
        var = _next_unique_name(pname, used_names)
        decls.append(f"{value_type} {var} = false;")
        return decls, var

    if any(t in ptype_lower for t in ["float", "double"]):
        var = _next_unique_name(pname, used_names)
        decls.append(f"{value_type} {var} = 0;")
        return decls, var

    if any(t in ptype_lower for t in ["int", "long", "short", "size_t", "uint"]):
        var = _next_unique_name(pname, used_names)
        decls.append(f"{value_type} {var} = 0;")
        return decls, var

    var = _next_unique_name(pname, used_names)
    decls.append(f"{value_type} {var}{{}};")
    return decls, var


def build_example_answer_from_context(frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
    """Deterministic fallback for grounded example-generation answers."""
    analysis = analysis or {}
    clean_frags = [f for f in frags if is_valid_function_chunk(f)]
    if not clean_frags:
        return "Not enough verified context to build a reliable example."

    if example_context is None:
        example_context = _build_example_context_impl(clean_frags, analysis=analysis)

    target = example_context.get("target")
    if target is None or target not in clean_frags:
        target = clean_frags[0]
    target_name = str(target.get("name", "")).strip() or "target_function"
    target_sig = target.get("signature") or f"{target_name}()"
    params = list(target.get("parameters") or [])
    type_init_hints = _infer_type_init_hints(params, symbols, type_init_index=type_init_index)
    required_field_hints = _infer_required_field_hints(
        params,
        target_name=target_name,
        type_init_index=type_init_index,
    )

    call_hint_args = []
    caller = example_context.get("caller")
    if caller is not None and caller not in clean_frags:
        caller = None
    observed_call_obj = example_context.get("observed_call") or {}
    observed_call = observed_call_obj.get("expr")
    if isinstance(observed_call_obj.get("args"), list):
        call_hint_args = observed_call_obj.get("args", [])
    observed_call_inline = _normalize_inline_snippet(observed_call)
    caller_decl_hints = _extract_caller_decl_hints(
        str((caller or {}).get("code", "")),
        call_hint_args,
    )
    caller_flow_lines = _extract_caller_flow_statements(
        str((caller or {}).get("code", "")),
        call_hint_args,
    )

    header_name = os.path.basename(str(target.get("file", "")))
    include_target_header = bool(re.search(r"\.(h|hpp|hh|hxx)$", header_name))
    needs_file_bytes = bool(analysis.get("needs_file"))

    include_lines = [
        "#include <cstdint>",
        "#include <cstddef>",
        "#include <vector>",
        "#include <string>",
        "#include <fstream>",
        "#include <iterator>",
        "#include <iostream>",
    ]
    include_lines.extend(_headers_from_type_init_hints(type_init_hints))
    if include_target_header:
        include_lines.append(f"#include \"{header_name}\"")

    body_lines = []
    if needs_file_bytes:
        body_lines.extend([
            "if (argc < 2) {",
            "    std::cerr << \"Usage: \" << argv[0] << \" <input_file>\\n\";",
            "    return 1;",
            "}",
            "",
            "std::vector<uint8_t> input_bytes;",
            "std::ifstream in(argv[1], std::ios::binary);",
            "if (!in) {",
            "    std::cerr << \"Failed to open input file\\n\";",
            "    return 1;",
            "}",
            "input_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());",
            "",
        ])
    else:
        body_lines.append("std::vector<uint8_t> input_bytes;")
        body_lines.append("")

    used_names = set()
    decl_lines = []
    arg_exprs = []
    for i, p in enumerate(params):
        hint = call_hint_args[i] if i < len(call_hint_args) else None
        d, arg = _build_param_binding(
            p,
            analysis,
            hint,
            used_names,
            caller_decl_hints=caller_decl_hints,
            type_init_hint=type_init_hints.get(i),
        )
        decl_lines.extend(d)
        arg_exprs.append(arg)
    body_lines.extend(decl_lines)
    required_field_lines = _build_required_field_init_lines(
        required_field_hints=required_field_hints,
        params=params,
        arg_exprs=arg_exprs,
        analysis=analysis,
    )
    if required_field_lines:
        body_lines.extend(required_field_lines)
    if caller_flow_lines:
        body_lines.extend(caller_flow_lines)
    if decl_lines or required_field_lines or caller_flow_lines:
        body_lines.append("")

    args_joined = ", ".join(arg_exprs)
    sig_head = target_sig.split(target_name, 1)[0]
    is_void_return = re.search(r"\bvoid\s*$", sig_head.strip()) is not None
    if is_void_return:
        body_lines.append(f"{target_name}({args_joined});")
    else:
        body_lines.append(f"auto result = {target_name}({args_joined});")
        body_lines.append("(void)result;")
    body_lines.append("return 0;")

    code_lines = []
    code_lines.extend(include_lines)
    code_lines.append("")
    code_lines.append(f"// Target signature (from indexed code): {_normalize_inline_snippet(target_sig)}")
    if observed_call_inline:
        code_lines.append(f"// Observed call pattern in codebase: {observed_call_inline}")
    code_lines.append("int main(int argc, char ** argv) {")
    code_lines.extend([f"    {ln}" if ln else "" for ln in body_lines])
    code_lines.append("}")

    lines = [
        "### Example Code (deterministic context-grounded fallback)",
        "```cpp",
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


def _infer_param_role_from_shape(pname, ptype):
    shape = _classify_param_shape_impl(ptype, pname)
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

    if example_context is None:
        example_context = _build_example_context_impl(clean_frags, analysis=analysis)

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
        shape, role = _infer_param_role_from_shape(pname, ptype)

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

