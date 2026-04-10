from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ..ask_adapters import AskLanguageAdapter, _extract_nominal_type_name_default


class CCppAskLanguageAdapter(AskLanguageAdapter):
    def non_function_tokens(self) -> set[str]:
        return {
            "if", "for", "while", "switch", "return", "sizeof", "catch",
            "new", "delete", "throw", "else", "do", "class", "struct",
            "namespace", "template", "typedef", "using", "enum", "union",
            "printf", "scanf", "malloc", "free", "memset", "memcpy",
            "std", "vector", "string", "map", "set",
            "phase", "criteria", "console", "input", "output", "function",
            "void", "int", "float", "double", "char", "bool", "const", "size_t",
        }

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

    def extract_nominal_type_name(self, type_text: str) -> str:
        return _extract_nominal_type_name_default(type_text)

    def supports_default_init_recipe_penalty(self) -> bool:
        return True

