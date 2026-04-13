from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _scan_signature_for_body_impl(signature_text: str) -> Tuple[int, bool]:
    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape = False

    seen_lparen = False
    paren_depth = 0

    i = 0
    n = len(signature_text)
    while i < n:
        ch = signature_text[i]
        nxt = signature_text[i + 1] if i + 1 < n else ""

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
            seen_lparen = True
            paren_depth += 1
            i += 1
            continue

        if ch == ")" and paren_depth > 0:
            paren_depth -= 1
            i += 1
            continue

        if seen_lparen and paren_depth == 0:
            if ch == "{":
                return i, False
            if ch == ";":
                return -1, True

        i += 1

    return -1, False


def _find_matching_paren_impl(text: str, pos: int) -> int:
    if pos < 0 or pos >= len(text) or text[pos] != "(":
        return -1

    depth = 0
    i = pos
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
            i += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return i
            i += 1
            continue

        i += 1

    return -1


def _split_top_level_params_impl(params_text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []

    depth_angle = 0
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0

    in_str = False
    in_char = False
    escape = False

    i = 0
    n = len(params_text)
    while i < n:
        ch = params_text[i]

        if in_str:
            cur.append(ch)
            if not escape and ch == '"':
                in_str = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if in_char:
            cur.append(ch)
            if not escape and ch == "'":
                in_char = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if ch == '"':
            in_str = True
            escape = False
            cur.append(ch)
            i += 1
            continue

        if ch == "'":
            in_char = True
            escape = False
            cur.append(ch)
            i += 1
            continue

        if ch == "<":
            depth_angle += 1
            cur.append(ch)
            i += 1
            continue
        if ch == ">":
            depth_angle = max(0, depth_angle - 1)
            cur.append(ch)
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
            cur.append(ch)
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            cur.append(ch)
            i += 1
            continue
        if ch == "{":
            depth_brace += 1
            cur.append(ch)
            i += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            cur.append(ch)
            i += 1
            continue
        if ch == "[":
            depth_bracket += 1
            cur.append(ch)
            i += 1
            continue
        if ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            cur.append(ch)
            i += 1
            continue

        if ch == "," and depth_angle == 0 and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
            part = "".join(cur).strip()
            if part:
                out.append(part)
            cur = []
            i += 1
            continue

        cur.append(ch)
        i += 1

    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


@dataclass(frozen=True)
class IndexLanguageAdapter:
    name: str

    def scan_signature_for_body(self, signature_text: str) -> Tuple[int, bool]:
        return _scan_signature_for_body_impl(signature_text)

    def find_matching_paren(self, text: str, pos: int) -> int:
        return _find_matching_paren_impl(text, pos)

    def split_top_level_params(self, params_text: str) -> List[str]:
        return _split_top_level_params_impl(params_text)

    def qualified_symbol_pattern(self) -> str:
        return r"[A-Za-z_]\w*"

    def qualifier_separators(self) -> Tuple[str, ...]:
        return (".",)

    def qualified_constant_pattern(self) -> str:
        return r"[A-Z_][A-Z0-9_]*"

    def extract_call_name(self, expr: str) -> Optional[str]:
        text = str(expr or "").strip()
        if not text:
            return None
        m = re.match(rf"\s*(?P<name>{self.qualified_symbol_pattern()})\s*\(", text)
        if not m:
            return None
        name = str(m.group("name") or "").strip()
        for sep in self.qualifier_separators():
            if sep:
                name = name.split(sep)[-1]
        return name or None

    def is_literal_like_token(self, token: str) -> bool:
        t = str(token or "").strip()
        if not t:
            return False
        if re.fullmatch(r"(?:[-+]?\d+(?:\.\d+)?(?:[uUlLfF]*)|nullptr|NULL|true|false)", t):
            return True
        if re.fullmatch(r"'(?:\\.|[^'])+'", t):
            return True
        if re.fullmatch(r"\"(?:\\.|[^\"])*\"", t):
            return True
        return re.fullmatch(self.qualified_constant_pattern(), t) is not None

    def is_self_contained_call_expr(self, expr: str) -> bool:
        e = str(expr or "").strip()
        if not e:
            return False
        if self.extract_call_name(e) is None:
            return False
        open_idx = e.find("(")
        if open_idx == -1:
            return False
        close_idx = self.find_matching_paren(e, open_idx)
        if close_idx == -1 or e[close_idx + 1:].strip():
            return False
        args = self.split_top_level_params(e[open_idx + 1:close_idx])
        if not args:
            return True
        return all(self.is_literal_like_token(a) for a in args)

    def parse_single_parameter_simple(self, param_text: str) -> Tuple[str, str]:
        raise NotImplementedError

    def parse_parameters_simple(self, params_text: str) -> List[Dict]:
        params: List[Dict] = []
        text = str(params_text or "").strip()
        if not text or text == "void":
            return params

        for idx, param in enumerate(self.split_top_level_params(text)):
            item = param.strip()
            if not item:
                continue
            name, param_type = self.parse_single_parameter_simple(item)
            if not name or name == "unknown":
                name = f"arg{idx}"
            params.append({
                "name": name,
                "type": param_type,
                "is_reference": "&" in item,
                "is_pointer": "*" in item,
                "is_const": "const" in item.lower() or "final" in item.lower(),
                "raw": item,
            })
        return params

    def extract_init_patterns(self, code: str) -> List[Dict]:
        raise NotImplementedError

    def extract_variable_type_hints(self, code: str) -> Dict[str, Dict]:
        raise NotImplementedError

    def extract_field_assignments(self, code: str) -> List[Dict]:
        raise NotImplementedError

    def extract_field_reads(self, code: str) -> List[Dict]:
        raise NotImplementedError

    def extract_call_sites(self, code: str, control_keywords: Sequence[str]) -> List[Dict]:
        raise NotImplementedError

    def extract_dependency_chain(self, code: str, base_var: str, before_line: int, max_steps: int = 8) -> List[Dict]:
        raise NotImplementedError

    def extract_arg_base_var(self, arg_expr: str) -> Optional[str]:
        raise NotImplementedError

    def extract_nominal_type_name_for_init(self, type_text: str) -> Optional[str]:
        raise NotImplementedError


from .c_cpp.index_adapter import CCppIndexLanguageAdapter
from .java.index_adapter import JavaIndexLanguageAdapter


_INDEX_ADAPTERS = {
    "c_cpp": CCppIndexLanguageAdapter(name="c_cpp"),
    "java": JavaIndexLanguageAdapter(name="java"),
}


def get_index_language_adapter(language_name: str) -> IndexLanguageAdapter:
    return _INDEX_ADAPTERS.get(language_name or "", _INDEX_ADAPTERS["c_cpp"])


def infer_index_language_from_path(file_path: str, default: str = "c_cpp") -> str:
    suffix = Path(str(file_path or "")).suffix.lower()
    if suffix == ".java":
        return "java"
    return default
