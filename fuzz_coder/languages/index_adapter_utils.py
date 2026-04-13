from __future__ import annotations

import re
from typing import List, Optional, Tuple


def strip_default_initializer_top_level(param_text: str) -> str:
    s = str(param_text or "")
    depth_angle = 0
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    in_str = False
    in_char = False
    escape = False

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
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

        if ch == "<":
            depth_angle += 1
            i += 1
            continue
        if ch == ">":
            depth_angle = max(0, depth_angle - 1)
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            i += 1
            continue
        if ch == "{":
            depth_brace += 1
            i += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            i += 1
            continue
        if ch == "[":
            depth_bracket += 1
            i += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            i += 1
            continue

        if ch == "=" and depth_angle == 0 and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
            return s[:i].rstrip()

        i += 1
    return s.strip()


def extract_trailing_identifier_top_level(param_text: str) -> Optional[Tuple[str, int]]:
    s = str(param_text or "").rstrip()
    if not s:
        return None

    depth_angle = 0
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    in_str = False
    in_char = False
    escape = False

    i = len(s) - 1
    while i >= 0:
        ch = s[i]
        if in_str:
            if not escape and ch == '"':
                in_str = False
            escape = (ch == "\\" and not escape)
            i -= 1
            continue
        if in_char:
            if not escape and ch == "'":
                in_char = False
            escape = (ch == "\\" and not escape)
            i -= 1
            continue

        if ch == '"':
            in_str = True
            escape = False
            i -= 1
            continue
        if ch == "'":
            in_char = True
            escape = False
            i -= 1
            continue

        if ch == ">":
            depth_angle += 1
            i -= 1
            continue
        if ch == "<":
            depth_angle = max(0, depth_angle - 1)
            i -= 1
            continue
        if ch == ")":
            depth_paren += 1
            i -= 1
            continue
        if ch == "(":
            depth_paren = max(0, depth_paren - 1)
            i -= 1
            continue
        if ch == "}":
            depth_brace += 1
            i -= 1
            continue
        if ch == "{":
            depth_brace = max(0, depth_brace - 1)
            i -= 1
            continue
        if ch == "]":
            depth_bracket += 1
            i -= 1
            continue
        if ch == "[":
            depth_bracket = max(0, depth_bracket - 1)
            i -= 1
            continue

        if depth_angle == 0 and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
            if ch.isalnum() or ch == "_":
                end = i
                start = i
                while start >= 0 and (s[start].isalnum() or s[start] == "_"):
                    start -= 1
                name = s[start + 1:end + 1]
                if name and name not in {
                    "const", "volatile", "unsigned", "signed", "short", "long",
                    "int", "char", "float", "double", "bool", "void", "size_t",
                    "ssize_t", "struct", "class", "enum", "typename", "auto", "final",
                    "public", "private", "protected", "static",
                }:
                    return name, start + 1
                return None
            if ch.isspace():
                i -= 1
                continue
            if ch in "*&])":
                i -= 1
                continue
            return None

        i -= 1
    return None


def normalize_expr(expr: str, max_len: int = 240) -> str:
    e = " ".join(str(expr or "").split()).strip()
    if len(e) <= max_len:
        return e
    return e[: max_len - 3].rstrip() + "..."


def extract_simple_identifiers(expr: str) -> List[str]:
    e = str(expr or "").strip()
    if not e:
        return []
    return re.findall(r"[A-Za-z_]\w*", e)

