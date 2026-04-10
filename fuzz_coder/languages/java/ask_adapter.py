from __future__ import annotations

import re
from typing import List, Tuple

from ..ask_adapters import AskLanguageAdapter, _extract_nominal_type_name_default


class JavaAskLanguageAdapter(AskLanguageAdapter):
    def non_function_tokens(self) -> set[str]:
        return {
            "if", "for", "while", "switch", "return", "catch",
            "new", "throw", "else", "do", "class", "interface",
            "package", "import", "enum", "record",
            "system", "out", "println", "print", "logger",
            "phase", "criteria", "console", "input", "output", "function",
            "void", "int", "float", "double", "char", "boolean", "const", "final",
        }

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

    def extract_nominal_type_name(self, type_text: str) -> str:
        return _extract_nominal_type_name_default(type_text)

