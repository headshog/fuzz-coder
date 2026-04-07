from __future__ import annotations

import re
from typing import Dict, List, Optional


def _find_matching_paren_text(text: str, open_idx: int) -> int:
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "(":
        return -1

    depth = 0
    in_str = False
    in_char = False
    escape = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_str:
            if not escape and ch == '"':
                in_str = False
            escape = (ch == "\\" and not escape)
            continue
        if in_char:
            if not escape and ch == "'":
                in_char = False
            escape = (ch == "\\" and not escape)
            continue

        if ch == '"':
            in_str = True
            escape = False
            continue
        if ch == "'":
            in_char = True
            escape = False
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_level_arguments(args_text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
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


def _extract_call_argument_lists(code: str, target_name: str, limit: int = 5) -> List[Dict]:
    if not code or not target_name:
        return []

    pattern = re.compile(
        rf"(?<![A-Za-z0-9_~])(?:[A-Za-z_]\w*::)*{re.escape(target_name)}\s*\("
    )
    out: List[Dict] = []
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
        out.append({
            "expr": expr,
            "args": args,
            "arity": len(args),
        })
        if len(out) >= limit:
            break
    return out


def _classify_param_shape(param_type: str, param_name: str) -> str:
    t = (param_type or "").lower()
    n = (param_name or "").lower()
    if any(x in t for x in ["char *", "char*", "uint8_t *", "uint8_t*", "unsigned char *", "unsigned char*"]):
        return "byte_buffer"
    if any(x in t for x in ["std::string", "string"]):
        return "string"
    if "*" in t:
        if any(k in n for k in ["out", "dst", "dest", "output"]):
            return "out_pointer"
        return "pointer"
    if "&" in t:
        return "reference"
    if any(x in t for x in ["size_t", "ssize_t", "int", "long", "short", "uint"]):
        if any(k in n for k in ["size", "len", "length", "count", "bytes", "n"]):
            return "size_or_length"
        return "integer"
    if any(x in t for x in ["float", "double"]):
        return "float"
    if "bool" in t:
        return "bool"
    return "value"


def _extract_file_source_vars(code: str) -> List[str]:
    if not code:
        return []

    vars_found: List[str] = []
    patterns = [
        r"\bstd::ifstream\s+([A-Za-z_]\w*)\s*\(\s*argv\s*\[\s*1\s*\]",
        r"\bauto\s+([A-Za-z_]\w*)\s*=\s*[^;\n]*argv\s*\[\s*1\s*\]",
        r"\b(?:std::string|std::vector<[^>]+>|std::vector<\s*uint8_t\s*>)\s+([A-Za-z_]\w*)\s*\([^;\n]*argv\s*\[\s*1\s*\]",
        r"\bstd::getline\s*\([^,\n]+,\s*([A-Za-z_]\w+)\s*\)",
        r"\b([A-Za-z_]\w+)\s*\.assign\s*\(\s*std::istreambuf_iterator<",
        r"\bfread\s*\(\s*([A-Za-z_]\w+)\s*,",
    ]
    for pat in patterns:
        for m in re.finditer(pat, code):
            name = (m.group(1) or "").strip()
            if name and name not in vars_found:
                vars_found.append(name)
    return vars_found


def _normalize_chain_token(token: str) -> str:
    t = str(token or "").strip()
    t = re.sub(r"\s+", "", t)
    return t


def _lhs_base_name(lhs_expr: str) -> str:
    lhs = _normalize_chain_token(lhs_expr)
    if "->" in lhs:
        return lhs.split("->", 1)[0]
    if "." in lhs:
        return lhs.split(".", 1)[0]
    return lhs


def _extract_simple_assignment_edges(code: str) -> List[tuple[str, str]]:
    """Extract simple assignment edges like lhs = rhs; from caller code.

    This is intentionally lightweight and heuristic-driven. It ignores compound
    assignments and equality checks by matching only top-level '=' tokens.
    """
    if not code:
        return []

    pat = re.compile(
        r"([A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*)\s*"
        r"(?<![=!<>+\-*/%&|^])=(?!=)\s*"
        r"([^;]+);"
    )
    edges: List[tuple[str, str]] = []
    for m in pat.finditer(code):
        lhs_raw = (m.group(1) or "").strip()
        rhs_raw = (m.group(2) or "").strip()
        if not lhs_raw or not rhs_raw:
            continue

        # Skip very obvious non-dataflow cases.
        if lhs_raw.startswith(("return ", "if ", "while ", "for ", "switch ")):
            continue

        lhs = _normalize_chain_token(lhs_raw)
        edges.append((lhs, rhs_raw))
    return edges


def _extract_file_data_flow_symbols(code: str, source_vars: List[str]) -> List[str]:
    """Propagate file-derived vars through simple assignment chains.

    Example: line -> tmp -> cfg.path -> target(cfg.path)
    """
    derived = {_normalize_chain_token(v) for v in (source_vars or []) if v}
    if not code:
        return sorted({v for v in derived if v})

    # Add base-object aliases for member variables.
    for v in list(derived):
        base = _lhs_base_name(v)
        if base:
            derived.add(base)

    edges = _extract_simple_assignment_edges(code)
    if not edges:
        return sorted({v for v in derived if v})

    for _ in range(6):
        changed = False
        names = [v for v in derived if v]
        for lhs, rhs in edges:
            if not names:
                break
            if _arg_uses_any_var(rhs, names):
                if lhs not in derived:
                    derived.add(lhs)
                    changed = True
                base = _lhs_base_name(lhs)
                if base and base not in derived:
                    derived.add(base)
                    changed = True
        if not changed:
            break

    return sorted({v for v in derived if v})


def _arg_uses_any_var(arg_expr: str, names: List[str]) -> bool:
    expr = arg_expr or ""
    for name in names:
        n = _normalize_chain_token(name)
        if not n:
            continue
        if "." in n or "->" in n:
            pat = rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])"
        else:
            pat = rf"\b{re.escape(n)}\b"
        if re.search(pat, expr):
            return True
    return False


def _format_file_loc(chunk: Dict) -> str:
    return f"{chunk.get('file', '')}:{chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}"


def _normalize_path(path: str) -> str:
    p = (path or "").replace("\\", "/").lower().strip()
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def _file_matches_any_filter(file_path: str, path_filters: List[str]) -> bool:
    if not path_filters:
        return True
    p = _normalize_path(file_path)
    if not p:
        return False
    for flt in path_filters:
        f = _normalize_path(flt)
        if not f:
            continue
        if f in p:
            return True
    return False


def _target_quality_score(chunk: Dict) -> tuple:
    span = 0
    try:
        span = int(chunk.get("end_line", 0)) - int(chunk.get("start_line", 0))
    except Exception:
        span = 0
    code = str(chunk.get("code", "") or "")
    has_body = ("{" in code and "}" in code) or span >= 5
    params_count = len(list(chunk.get("parameters") or []))
    sig_len = len(str(chunk.get("signature", "") or ""))
    code_len = len(code)
    return (
        1 if has_body else 0,
        params_count,
        span,
        sig_len,
        code_len,
    )


def _choose_target(frags: List[Dict], analysis: Optional[Dict]) -> Optional[Dict]:
    if not frags:
        return None
    primary = (analysis or {}).get("primary_function_name")
    candidates = list(frags)
    if primary:
        primary_candidates = [f for f in frags if f.get("name") == primary]
        if primary_candidates:
            candidates = primary_candidates

    path_filters = list((analysis or {}).get("path_filters") or [])
    filtered = [c for c in candidates if _file_matches_any_filter(str(c.get("file", "")), path_filters)]
    if filtered:
        candidates = filtered

    if not candidates:
        return frags[0]
    return sorted(candidates, key=_target_quality_score, reverse=True)[0]


def _choose_caller(
    frags: List[Dict],
    target: Dict,
    analysis: Optional[Dict],
) -> tuple[Optional[Dict], List[Dict]]:
    target_name = str(target.get("name", "")).strip()
    if not target_name:
        return None, []

    caller_candidates: List[tuple[Dict, List[Dict]]] = []
    for f in frags:
        if f.get("name") == target_name:
            continue
        calls = _extract_call_argument_lists(str(f.get("code", "")), target_name, limit=5)
        if calls:
            caller_candidates.append((f, calls))

    if not caller_candidates:
        return None, []

    secondary_names = [
        n for n in (analysis or {}).get("function_names", [])
        if n and n != target_name
    ]
    for wanted in secondary_names:
        for c, calls in caller_candidates:
            if c.get("name") == wanted:
                return c, calls

    for c, calls in caller_candidates:
        if str(c.get("name", "")).lower() == "main":
            return c, calls

    return caller_candidates[0]


class ExampleContextBuilder:
    def __init__(self, frags: List[Dict], analysis: Optional[Dict] = None):
        self.frags = frags or []
        self.analysis = analysis or {}

    def build(self) -> Dict:
        target = _choose_target(self.frags, self.analysis)
        if target is None:
            return {
                "target": None,
                "caller": None,
                "observed_call": None,
                "arg_shapes": [],
                "file_data_flow_hints": {
                    "requires_file_data": False,
                    "caller_reads_argv1": False,
                    "source_vars": [],
                    "target_uses_file_data": False,
                    "argv1_direct_to_target": False,
                    "evidence": [],
                },
            }

        caller, caller_calls = _choose_caller(self.frags, target, self.analysis)
        observed_call = None
        target_arity = len(list(target.get("parameters") or []))
        if caller_calls:
            preferred = None
            for c in caller_calls:
                if c.get("arity") == target_arity:
                    preferred = c
                    break
            if preferred is None:
                preferred = caller_calls[0]
            observed_call = preferred

        params = list(target.get("parameters") or [])
        observed_args = list((observed_call or {}).get("args", []))
        arg_shapes = []
        for i, p in enumerate(params):
            p_name = str(p.get("name", "")).strip() or f"arg{i}"
            p_type = str(p.get("type", "")).strip()
            arg_shapes.append({
                "index": i,
                "name": p_name,
                "type": p_type,
                "shape": _classify_param_shape(p_type, p_name),
                "observed_arg": observed_args[i] if i < len(observed_args) else None,
            })

        requires_file_data = bool(self.analysis.get("needs_file"))
        caller_code = str((caller or {}).get("code", ""))
        source_vars = _extract_file_source_vars(caller_code)
        flow_vars = _extract_file_data_flow_symbols(caller_code, source_vars)
        caller_reads_argv1 = "argv[1]" in caller_code and bool(re.search(
            r"\b(ifstream|fopen|open|read|getline|istreambuf_iterator)\b", caller_code
        ))

        target_uses_file_data = False
        argv1_direct_to_target = False
        for a in observed_args:
            if "argv[1]" in (a or ""):
                argv1_direct_to_target = True
                target_uses_file_data = True
                break
            if _arg_uses_any_var(a, flow_vars):
                target_uses_file_data = True
                break

        evidence = [
            f"target={target.get('name')} @ {_format_file_loc(target)}",
        ]
        if caller is not None:
            evidence.append(f"caller={caller.get('name')} @ {_format_file_loc(caller)}")
        if observed_call is not None:
            evidence.append(f"observed_call={observed_call.get('expr')}")
        if source_vars:
            evidence.append(f"file_source_vars={', '.join(source_vars)}")
        if flow_vars:
            evidence.append(f"file_flow_vars={', '.join(flow_vars)}")

        return {
            "target": target,
            "caller": caller,
            "observed_call": observed_call,
            "arg_shapes": arg_shapes,
            "file_data_flow_hints": {
                "requires_file_data": requires_file_data,
                "caller_reads_argv1": caller_reads_argv1,
                "source_vars": source_vars,
                "flow_vars": flow_vars,
                "target_uses_file_data": target_uses_file_data,
                "argv1_direct_to_target": argv1_direct_to_target,
                "evidence": evidence,
            },
        }


def build_example_context(frags: List[Dict], analysis: Optional[Dict] = None) -> Dict:
    return ExampleContextBuilder(frags, analysis=analysis).build()
