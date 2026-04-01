from __future__ import annotations

import re
from collections import defaultdict


CALL_RE = re.compile(
    r"(?<![A-Za-z0-9_~])("
    r"(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*"
    r"|"
    r"[A-Za-z_]\w*\s*(?:->|\.)\s*[A-Za-z_]\w*"
    r")\s*\("
)


HARD_CALL_SKIP = {
    "if", "for", "while", "switch", "return", "sizeof", "alignof", "decltype",
    "catch", "throw", "new", "delete",
}


def _mask_non_code_regions(text):
    """Replace strings/chars/comments with spaces while preserving indices."""
    chars = list(text)
    n = len(chars)
    i = 0

    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape = False

    while i < n:
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch != "\n":
                chars[i] = " "
            else:
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                chars[i] = " "
                chars[i + 1] = " "
                in_block_comment = False
                i += 2
                continue
            chars[i] = " "
            i += 1
            continue

        if in_str:
            chars[i] = " "
            if not escape and ch == '"':
                in_str = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if in_char:
            chars[i] = " "
            if not escape and ch == "'":
                in_char = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            chars[i] = " "
            chars[i + 1] = " "
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            chars[i] = " "
            chars[i + 1] = " "
            in_block_comment = True
            i += 2
            continue

        if ch == '"':
            chars[i] = " "
            in_str = True
            escape = False
            i += 1
            continue

        if ch == "'":
            chars[i] = " "
            in_char = True
            escape = False
            i += 1
            continue

        i += 1

    return "".join(chars)


def _count_call_arity(code, open_paren_idx):
    """Count top-level arguments for a call starting at open_paren_idx."""
    n = len(code)
    i = open_paren_idx + 1
    paren_depth = 1
    brace_depth = 0
    bracket_depth = 0

    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape = False

    saw_token = False
    comma_count = 0

    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""

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
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            paren_depth -= 1
            if paren_depth == 0:
                return comma_count + 1 if saw_token else 0
            i += 1
            continue

        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth = max(0, brace_depth - 1)
            i += 1
            continue
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            i += 1
            continue

        if paren_depth == 1 and brace_depth == 0 and bracket_depth == 0:
            if ch == ",":
                comma_count += 1
                i += 1
                continue
            if not ch.isspace():
                saw_token = True

        i += 1

    return None


def _signature_arity(signature):
    if not signature:
        return None
    sig = str(signature)
    lp = sig.find("(")
    rp = sig.rfind(")")
    if lp == -1 or rp == -1 or rp < lp:
        return None
    params = sig[lp + 1:rp].strip()
    if not params or params == "void":
        return 0

    depth_paren = 0
    depth_angle = 0
    depth_brace = 0
    depth_bracket = 0
    arity = 1
    i = 0
    n = len(params)
    while i < n:
        ch = params[i]
        if ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle = max(0, depth_angle - 1)
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == "," and depth_paren == 0 and depth_angle == 0 and depth_brace == 0 and depth_bracket == 0:
            arity += 1
        i += 1
    return arity


def _parse_call_candidate(expr):
    expr = re.sub(r"\s+", "", expr or "")
    if not expr:
        return None

    is_member = False
    qualified = None

    if "->" in expr:
        is_member = True
        parts = expr.split("->")
        call_name = parts[-1]
        qualified = expr
    elif "." in expr:
        is_member = True
        parts = expr.split(".")
        call_name = parts[-1]
        qualified = expr
    elif "::" in expr:
        qualified = expr
        call_name = expr.split("::")[-1]
    else:
        call_name = expr

    if not call_name:
        return None
    if call_name in HARD_CALL_SKIP:
        return None
    if re.fullmatch(r"[A-Z_][A-Z0-9_]*", call_name):
        return None

    return {
        "name": call_name,
        "qualified": qualified,
        "is_member": is_member,
    }


def detect_call_details(body, control_keywords):
    """Detect call-like expressions with optional namespace/member and arity."""
    masked = _mask_non_code_regions(body)
    details = []
    seen = set()

    for m in CALL_RE.finditer(masked):
        raw_expr = m.group(1)
        open_paren_idx = m.end() - 1
        parsed = _parse_call_candidate(raw_expr)
        if not parsed:
            continue
        if parsed["name"] in control_keywords:
            continue

        arity = _count_call_arity(body, open_paren_idx)
        entry = {
            "name": parsed["name"],
            "qualified": parsed["qualified"],
            "is_member": parsed["is_member"],
            "arity": arity,
        }
        key = (entry["name"], entry["qualified"], entry["is_member"], entry["arity"])
        if key in seen:
            continue
        seen.add(key)
        details.append(entry)
    return details


def detect_calls(body, control_keywords):
    """Backward-compatible call detector returning only call names."""
    return [d["name"] for d in detect_call_details(body, control_keywords)]


def _resolve_call_ids(call_info, caller_chunk, name_to_ids, chunks_by_id):
    call_name = call_info.get("name")
    call_arity = call_info.get("arity")
    if not call_name:
        return []

    candidates = name_to_ids.get(call_name, [])
    if not candidates:
        return []

    if call_arity is not None:
        arity_filtered = []
        for cid in candidates:
            target_arity = _signature_arity(chunks_by_id[cid].get("signature", ""))
            if target_arity is None or target_arity == call_arity:
                arity_filtered.append(cid)
        if arity_filtered:
            candidates = arity_filtered

    if len(candidates) == 1:
        return list(candidates)

    caller_file = caller_chunk.get("file")
    same_file = [cid for cid in candidates if chunks_by_id[cid].get("file") == caller_file]

    # Prefer unique same-file target (helps with namespaces/modules split across files).
    if len(same_file) == 1:
        return same_file

    # If same-file candidates share one signature, keep them (template instantiations/duplicates).
    if len(same_file) > 1:
        sigs = {chunks_by_id[cid].get("signature", "") for cid in same_file}
        if len(sigs) == 1:
            return sorted(set(same_file))
        return []

    # Ambiguous cross-file call: skip to reduce false positives for overloads/common names.
    return []


def build_call_graph(chunks, control_keywords):
    """Build a call graph with conservative name+file+signature resolution."""
    name_to_ids = defaultdict(list)
    chunks_by_id = {}

    for c in chunks:
        name_to_ids[c["name"]].append(c["id"])
        chunks_by_id[c["id"]] = c

    call_graph = {}
    reverse_by_id = defaultdict(set)
    calls_by_id = {}
    call_details_by_id = {}
    resolved_calls_by_id = {}

    for c in chunks:
        caller_id = c["id"]
        call_details = detect_call_details(c.get("body", ""), control_keywords)
        calls = list(dict.fromkeys(d["name"] for d in call_details))
        resolved_calls = []

        for call in call_details:
            target_ids = _resolve_call_ids(call, c, name_to_ids, chunks_by_id)
            resolved_calls.extend(target_ids)

        resolved_calls = sorted(set(resolved_calls))
        calls_by_id[caller_id] = calls
        call_details_by_id[caller_id] = call_details
        resolved_calls_by_id[caller_id] = resolved_calls
        for tid in resolved_calls:
            reverse_by_id[tid].add(caller_id)

    # Build deterministic per-function entries using reverse edges by function ID.
    for c in chunks:
        cid = c["id"]
        call_graph[cid] = {
            "calls": calls_by_id.get(cid, []),
            "call_details": call_details_by_id.get(cid, []),
            "resolved_calls": resolved_calls_by_id.get(cid, []),
            "called_by": sorted(reverse_by_id.get(cid, set())),
        }

    # Keep "called_by.json" useful for both new id-based and legacy name-based lookups.
    called_by_out = {}
    for c in chunks:
        cid = c["id"]
        callers = sorted(reverse_by_id.get(cid, set()))
        called_by_out[str(cid)] = callers

    called_by_name = defaultdict(set)
    for c in chunks:
        cid = c["id"]
        called_by_name[c["name"]].update(reverse_by_id.get(cid, set()))
    for name, callers in called_by_name.items():
        called_by_out[name] = sorted(callers)

    return call_graph, called_by_out
