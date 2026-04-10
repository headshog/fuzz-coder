from __future__ import annotations

import re
from typing import Dict, List, Set


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


def _mask_non_code_regions(text: str) -> str:
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


def _count_call_arity(code: str, open_paren_idx: int):
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


def _parse_call_candidate(expr: str):
    expr = re.sub(r"\s+", "", expr or "")
    if not expr:
        return None

    is_member = False
    qualified = None

    if "->" in expr:
        is_member = True
        call_name = expr.split("->")[-1]
        qualified = expr
    elif "." in expr:
        is_member = True
        call_name = expr.split(".")[-1]
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


def detect_call_details(body: str, control_keywords: Set[str]) -> List[Dict]:
    """Detect call-like expressions with optional namespace/member and arity."""
    masked = _mask_non_code_regions(body or "")
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

        arity = _count_call_arity(body or "", open_paren_idx)
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
