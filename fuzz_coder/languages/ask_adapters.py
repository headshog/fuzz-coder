from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


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


def _extract_nominal_type_name_default(type_text: str) -> str:
    t = re.sub(r"\b(const|volatile|restrict|__restrict__|struct|class|enum|final)\b", " ", str(type_text or ""))
    t = t.replace("*", " ").replace("&", " ").replace("[]", " ")
    t = re.sub(r"<[^>]*>", " ", t)
    tokens = re.findall(r"[A-Za-z_]\w*", t)
    if not tokens:
        return ""
    skip = {
        "unsigned", "signed", "long", "short", "int", "float", "double", "bool", "void",
        "size_t", "ssize_t", "auto", "typename", "byte", "boolean", "char", "var",
        "public", "private", "protected", "static", "extends", "super",
    }
    for tok in reversed(tokens):
        if tok.lower() not in skip:
            return tok
    return ""


@dataclass(frozen=True)
class AskLanguageAdapter:
    name: str

    def non_function_tokens(self) -> set[str]:
        return set()

    def call_pattern(self, target_name: str) -> re.Pattern:
        raise NotImplementedError

    def find_matching_paren(self, text: str, open_idx: int) -> int:
        return _find_matching_paren_text(text, open_idx)

    def split_top_level_arguments(self, args_text: str) -> List[str]:
        return _split_top_level_arguments(args_text)

    def extract_call_argument_lists(self, code: str, target_name: str, limit: int = 5) -> List[Dict]:
        if not code or not target_name:
            return []

        pattern = self.call_pattern(target_name)
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

    def extract_call_arities(self, code: str, target_name: str, limit: int = 64) -> List[int]:
        calls = self.extract_call_argument_lists(code, target_name, limit=limit)
        return [int(c.get("arity", 0)) for c in calls]

    def classify_param_shape(self, param_type: str, param_name: str) -> str:
        raise NotImplementedError

    def extract_file_source_vars(self, code: str) -> List[str]:
        raise NotImplementedError

    def normalize_chain_token(self, token: str) -> str:
        t = str(token or "").strip()
        t = re.sub(r"\s+", "", t)
        return t

    def lhs_base_name(self, lhs_expr: str) -> str:
        lhs = self.normalize_chain_token(lhs_expr)
        if "." in lhs:
            return lhs.split(".", 1)[0]
        return lhs

    def extract_simple_assignment_edges(self, code: str) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def arg_uses_any_var(self, arg_expr: str, names: Sequence[str]) -> bool:
        expr = arg_expr or ""
        for name in names:
            n = self.normalize_chain_token(name)
            if not n:
                continue
            if "." in n:
                pat = rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])"
            else:
                pat = rf"\b{re.escape(n)}\b"
            if re.search(pat, expr):
                return True
        return False

    def extract_file_data_flow_symbols(self, code: str, source_vars: Sequence[str]) -> List[str]:
        derived = {self.normalize_chain_token(v) for v in (source_vars or []) if v}
        if not code:
            return sorted({v for v in derived if v})

        for v in list(derived):
            base = self.lhs_base_name(v)
            if base:
                derived.add(base)

        edges = self.extract_simple_assignment_edges(code)
        if not edges:
            return sorted({v for v in derived if v})

        for _ in range(6):
            changed = False
            names = [v for v in derived if v]
            for lhs, rhs in edges:
                if not names:
                    break
                if self.arg_uses_any_var(rhs, names):
                    if lhs not in derived:
                        derived.add(lhs)
                        changed = True
                    base = self.lhs_base_name(lhs)
                    if base and base not in derived:
                        derived.add(base)
                        changed = True
            if not changed:
                break

        return sorted({v for v in derived if v})

    def caller_reads_cli_file_data(self, code: str) -> bool:
        return self.is_cli_file_expr(code) and bool(re.search(r"\b(read|open|getline|ifstream|fopen)\b", code or ""))

    def is_cli_file_expr(self, expr: str) -> bool:
        raise NotImplementedError

    def extract_nominal_type_name(self, type_text: str) -> str:
        return _extract_nominal_type_name_default(type_text)

    def supports_default_init_recipe_penalty(self) -> bool:
        return False


from .c_cpp.ask_adapter import CCppAskLanguageAdapter
from .java.ask_adapter import JavaAskLanguageAdapter


_ASK_ADAPTERS: Dict[str, AskLanguageAdapter] = {
    "c_cpp": CCppAskLanguageAdapter(name="c_cpp"),
    "java": JavaAskLanguageAdapter(name="java"),
}


def get_ask_language_adapter(language_name: str) -> AskLanguageAdapter:
    return _ASK_ADAPTERS.get(language_name or "", _ASK_ADAPTERS["c_cpp"])


def infer_ask_language_from_fragments(frags: Sequence[Dict], default: str = "c_cpp") -> str:
    counts = {"c_cpp": 0, "java": 0}
    for f in frags or []:
        file_path = str((f or {}).get("file", "")).strip()
        if not file_path:
            continue
        suffix = Path(file_path).suffix.lower()
        if suffix == ".java":
            counts["java"] += 1
        elif suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"}:
            counts["c_cpp"] += 1

    if counts["java"] > counts["c_cpp"]:
        return "java"
    return default
