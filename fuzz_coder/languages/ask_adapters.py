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


@dataclass(frozen=True)
class AskLanguageAdapter:
    name: str

    def call_pattern(self, target_name: str) -> re.Pattern:
        raise NotImplementedError

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


class CCppAskLanguageAdapter(AskLanguageAdapter):
    def call_pattern(self, target_name: str) -> re.Pattern:
        return re.compile(
            rf"(?<![A-Za-z0-9_~])(?:[A-Za-z_]\w*::)*{re.escape(target_name)}\s*\("
        )

    def classify_param_shape(self, param_type: str, param_name: str) -> str:
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

    def extract_file_source_vars(self, code: str) -> List[str]:
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

    def lhs_base_name(self, lhs_expr: str) -> str:
        lhs = self.normalize_chain_token(lhs_expr)
        if "->" in lhs:
            return lhs.split("->", 1)[0]
        if "." in lhs:
            return lhs.split(".", 1)[0]
        return lhs

    def extract_simple_assignment_edges(self, code: str) -> List[Tuple[str, str]]:
        if not code:
            return []
        pat = re.compile(
            r"([A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*)\s*"
            r"(?<![=!<>+\-*/%&|^])=(?!=)\s*"
            r"([^;]+);"
        )
        edges: List[Tuple[str, str]] = []
        for m in pat.finditer(code):
            lhs_raw = (m.group(1) or "").strip()
            rhs_raw = (m.group(2) or "").strip()
            if not lhs_raw or not rhs_raw:
                continue
            if lhs_raw.startswith(("return ", "if ", "while ", "for ", "switch ")):
                continue
            lhs = self.normalize_chain_token(lhs_raw)
            edges.append((lhs, rhs_raw))
        return edges

    def arg_uses_any_var(self, arg_expr: str, names: Sequence[str]) -> bool:
        expr = arg_expr or ""
        for name in names:
            n = self.normalize_chain_token(name)
            if not n:
                continue
            if "." in n or "->" in n:
                pat = rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])"
            else:
                pat = rf"\b{re.escape(n)}\b"
            if re.search(pat, expr):
                return True
        return False

    def is_cli_file_expr(self, expr: str) -> bool:
        return re.search(r"\bargv\s*\[\s*1\s*\]", expr or "") is not None


class JavaAskLanguageAdapter(AskLanguageAdapter):
    def call_pattern(self, target_name: str) -> re.Pattern:
        return re.compile(
            rf"(?<![A-Za-z0-9_])(?:[A-Za-z_]\w*\s*\.)*{re.escape(target_name)}\s*\("
        )

    def classify_param_shape(self, param_type: str, param_name: str) -> str:
        t = (param_type or "").lower()
        n = (param_name or "").lower()
        if "byte[]" in t or "bytebuffer" in t:
            return "byte_buffer"
        if "string" in t or "charsequence" in t:
            return "string"
        if t.endswith("[]"):
            return "array"
        if any(x in t for x in ["int", "long", "short", "integer", "size"]):
            if any(k in n for k in ["size", "len", "length", "count", "bytes", "n"]):
                return "size_or_length"
            return "integer"
        if any(x in t for x in ["float", "double", "bigdecimal"]):
            return "float"
        if "boolean" in t or "bool" in t:
            return "bool"
        if any(x in t for x in ["path", "file", "inputstream", "reader"]):
            return "file_or_path"
        return "value"

    def extract_file_source_vars(self, code: str) -> List[str]:
        if not code:
            return []
        vars_found: List[str] = []
        patterns = [
            r"\b(?:byte\[\]|String|Path|File|InputStream|BufferedReader)\s+([A-Za-z_]\w*)\s*=\s*[^;\n]*args\s*\[\s*[01]\s*\]",
            r"\b(?:byte\[\]|String)\s+([A-Za-z_]\w*)\s*=\s*Files\.(?:readAllBytes|readString)\s*\([^;\n]*args\s*\[\s*[01]\s*\]",
            r"\b([A-Za-z_]\w+)\s*=\s*Files\.(?:readAllBytes|readString)\s*\(",
        ]
        for pat in patterns:
            for m in re.finditer(pat, code):
                name = (m.group(1) or "").strip()
                if name and name not in vars_found:
                    vars_found.append(name)
        return vars_found

    def extract_simple_assignment_edges(self, code: str) -> List[Tuple[str, str]]:
        if not code:
            return []
        pat = re.compile(
            r"([A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)\s*"
            r"(?<![=!<>+\-*/%&|^])=(?!=)\s*"
            r"([^;]+);"
        )
        edges: List[Tuple[str, str]] = []
        for m in pat.finditer(code):
            lhs_raw = (m.group(1) or "").strip()
            rhs_raw = (m.group(2) or "").strip()
            if not lhs_raw or not rhs_raw:
                continue
            if lhs_raw.startswith(("return ", "if ", "while ", "for ", "switch ")):
                continue
            lhs = self.normalize_chain_token(lhs_raw)
            edges.append((lhs, rhs_raw))
        return edges

    def caller_reads_cli_file_data(self, code: str) -> bool:
        expr = code or ""
        if not self.is_cli_file_expr(expr):
            return False
        return bool(
            re.search(
                r"\b(Files\.readAllBytes|Files\.readString|new\s+FileInputStream|new\s+FileReader|new\s+BufferedReader|Paths\.get|Path\.of)\b",
                expr,
            )
        )

    def is_cli_file_expr(self, expr: str) -> bool:
        return re.search(r"\bargs\s*\[\s*[01]\s*\]", expr or "") is not None or re.search(
            r"\bargv\s*\[\s*1\s*\]", expr or ""
        ) is not None


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
