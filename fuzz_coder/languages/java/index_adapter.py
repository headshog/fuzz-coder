from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..index_adapters import (
    IndexLanguageAdapter,
)
from ..index_adapter_utils import (
    extract_simple_identifiers as _extract_simple_identifiers,
    extract_trailing_identifier_top_level as _extract_trailing_identifier_top_level,
    normalize_expr as _normalize_expr,
    strip_default_initializer_top_level as _strip_default_initializer_top_level,
)


class JavaIndexLanguageAdapter(IndexLanguageAdapter):
    def qualified_symbol_pattern(self) -> str:
        return r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"

    def qualifier_separators(self) -> Tuple[str, ...]:
        return (".",)

    def qualified_constant_pattern(self) -> str:
        return r"(?:[A-Za-z_]\w*\.)*[A-Z_][A-Z0-9_]*"

    def extract_init_patterns(self, code: str) -> List[Dict]:
        out: List[Dict] = []
        val_eq = re.compile(
            r"^\s*(?:final\s+)?(?P<type>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:<[^;=(){}]+>)?(?:\s*\[\])*)\s+"
            r"(?P<var>[A-Za-z_]\w*)\s*=\s*(?P<expr>[^;]+)\s*;\s*$"
        )
        val_brace = re.compile(
            r"^\s*(?:final\s+)?(?P<type>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:<[^;=(){}]+>)?(?:\s*\[\])*)\s+"
            r"(?P<var>[A-Za-z_]\w*)\s*=\s*(?P<expr>\{[^;]*\})\s*;\s*$"
        )
        val_default = re.compile(
            r"^\s*(?:final\s+)?(?P<type>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:<[^;=(){}]+>)?(?:\s*\[\])*)\s+"
            r"(?P<var>[A-Za-z_]\w*)\s*;\s*$"
        )

        for li, raw in enumerate(str(code or "").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith(("if ", "for ", "while ", "switch ", "return ", "throw ")):
                continue

            match = val_brace.match(line)
            if match:
                out.append({
                    "line": li,
                    "kind": "value_brace",
                    "type_text": str(match.group("type") or "").strip(),
                    "expr": str(match.group("expr") or "").strip(),
                })
                continue

            match = val_eq.match(line)
            if match:
                out.append({
                    "line": li,
                    "kind": "value_call",
                    "type_text": str(match.group("type") or "").strip(),
                    "expr": str(match.group("expr") or "").strip(),
                })
                continue

            match = val_default.match(line)
            if match:
                out.append({
                    "line": li,
                    "kind": "value_default",
                    "type_text": str(match.group("type") or "").strip(),
                    "expr": "",
                })
        return out

    def parse_single_parameter_simple(self, param_text: str) -> Tuple[str, str]:
        raw = param_text.strip()
        if not raw:
            return "unknown", "unknown"

        no_default = _strip_default_initializer_top_level(raw)
        no_default = re.sub(r"^(@[A-Za-z_]\w*(?:\([^)]*\))?\s+)+", "", no_default)
        no_default = re.sub(r"\s+", " ", no_default).strip()

        vararg = re.search(r"\.\.\.\s*([A-Za-z_]\w*)\s*$", no_default)
        if vararg:
            name = vararg.group(1)
            ptype = no_default[:vararg.start(1)].strip()
            return name, ptype or "unknown"

        arr = re.search(r"([A-Za-z_]\w*)\s*(\[\])\s*$", no_default)
        if arr:
            name = arr.group(1)
            ptype = no_default[:arr.start(1)].strip()
            return name, ptype or "unknown"

        trailing = _extract_trailing_identifier_top_level(no_default)
        if trailing:
            name, start_idx = trailing
            ptype = no_default[:start_idx].strip()
            if ptype:
                return name, ptype
            return name, "unknown"

        return raw, "unknown"

    def extract_variable_type_hints(self, code: str) -> Dict[str, Dict]:
        hints: Dict[str, Dict] = {}
        decl = re.compile(
            r"^\s*(?:final\s+)?(?P<type>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:<[^;=()]+>)?(?:\s*\[\])*)\s+"
            r"(?P<var>[A-Za-z_]\w*)\b"
        )
        for raw in str(code or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith(("if ", "for ", "while ", "switch ", "return ")):
                continue
            m = decl.match(line)
            if not m:
                continue
            var = str(m.group("var") or "").strip()
            nominal = self.extract_nominal_type_name_for_init(m.group("type"))
            if not var or not nominal:
                continue
            hints[var] = {"nominal_type": nominal, "is_pointer": False}
        return hints

    def extract_field_assignments(self, code: str) -> List[Dict]:
        out: List[Dict] = []
        pat = re.compile(
            r"^\s*(?P<base>[A-Za-z_]\w*)\s*(?P<tail>(?:\.\s*[A-Za-z_]\w+)+)\s*=\s*(?P<rhs>[^;]+)\s*;\s*$"
        )
        for li, raw in enumerate(str(code or "").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            if "==" in line or "+=" in line or "-=" in line or "*=" in line or "/=" in line:
                continue
            m = pat.match(line)
            if not m:
                continue
            base = str(m.group("base") or "").strip()
            tail = str(m.group("tail") or "")
            rhs = str(m.group("rhs") or "").strip()
            pieces = re.findall(r"\.\s*([A-Za-z_]\w+)", tail)
            if not base or not pieces or not rhs:
                continue
            path = ".".join(pieces)
            out.append({
                "line": li,
                "base": base,
                "access": "dot",
                "path": path,
                "rhs": _normalize_expr(rhs),
            })
        return out

    def extract_field_reads(self, code: str) -> List[Dict]:
        out: List[Dict] = []
        access_pat = re.compile(r"(?P<base>[A-Za-z_]\w*)\s*(?P<tail>(?:\.\s*[A-Za-z_]\w+)+)")
        assign_pat = re.compile(r"^\s*(?P<base>[A-Za-z_]\w*)\s*(?P<tail>(?:\.\s*[A-Za-z_]\w+)+)\s*=\s*[^;]+;\s*$")

        for li, raw in enumerate(str(code or "").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue

            write_key = None
            m_assign = assign_pat.match(line)
            if m_assign:
                base = str(m_assign.group("base") or "").strip()
                tail = str(m_assign.group("tail") or "")
                pieces = re.findall(r"\.\s*([A-Za-z_]\w+)", tail)
                if pieces:
                    write_key = (base, ".".join(pieces))

            for m in access_pat.finditer(line):
                base = str(m.group("base") or "").strip()
                tail = str(m.group("tail") or "")
                pieces = re.findall(r"\.\s*([A-Za-z_]\w+)", tail)
                if not base or not pieces:
                    continue
                path = ".".join(pieces)
                key = (base, path)
                if write_key is not None and key == write_key:
                    continue
                out.append({"line": li, "base": base, "access": "dot", "path": path})
        return out

    def extract_call_sites(self, code: str, control_keywords: Sequence[str]) -> List[Dict]:
        out: List[Dict] = []
        control_set = set(control_keywords or [])
        call_re = re.compile(r"(?<![A-Za-z0-9_])(?P<expr>(?:[A-Za-z_]\w*\s*\.\s*)*[A-Za-z_]\w*)\s*\(")

        for m in call_re.finditer(code or ""):
            expr_raw = str(m.group("expr") or "")
            expr = re.sub(r"\s+", "", expr_raw)
            if not expr:
                continue
            name = expr.split(".")[-1]
            if not name or name in control_set:
                continue

            open_idx = m.end() - 1
            close_idx = self.find_matching_paren(code, open_idx)
            if close_idx == -1:
                continue
            args_text = code[open_idx + 1:close_idx]
            args = self.split_top_level_params(args_text)
            line_no = code.count("\n", 0, m.start()) + 1
            out.append({"name": name, "arity": len(args), "args": args, "line": line_no})
        return out

    def extract_dependency_chain(self, code: str, base_var: str, before_line: int, max_steps: int = 8) -> List[Dict]:
        lines = str(code or "").splitlines()
        upto = max(0, int(before_line) - 1)
        if not lines or not base_var:
            return []

        decl_init_pat = re.compile(
            r"^\s*(?:final\s+)?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:<[^;=()]+>)?(?:\s*\[\])?\s+"
            r"(?P<lhs>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;]+)\s*;\s*$"
        )
        assign_pat = re.compile(r"^\s*(?P<lhs>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;]+)\s*;\s*$")

        tracked = {str(base_var)}
        chain: List[Dict] = []
        seen_stmt = set()

        for li in range(min(upto, len(lines)), 0, -1):
            if len(chain) >= max_steps:
                break
            line = lines[li - 1].strip()
            if not line or line.startswith("//"):
                continue
            if "==" in line or "+=" in line or "-=" in line or "*=" in line or "/=" in line:
                continue

            m = assign_pat.match(line) or decl_init_pat.match(line)
            if not m:
                continue
            lhs = str(m.group("lhs") or "").strip()
            rhs = str(m.group("rhs") or "").strip()
            if lhs not in tracked:
                continue

            stmt = f"{lhs} = {rhs};"
            if stmt in seen_stmt:
                continue
            seen_stmt.add(stmt)
            chain.append({"line": li, "statement": stmt})

            for tok in _extract_simple_identifiers(rhs):
                if tok != lhs:
                    tracked.add(tok)

        chain.sort(key=lambda x: int(x.get("line", 0)))
        return chain

    def extract_arg_base_var(self, arg_expr: str) -> Optional[str]:
        expr = str(arg_expr or "").strip()
        if not expr:
            return None

        cast = re.match(r"^\([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:<[^>]+>)?(?:\[\])?\)\s*(.+)$", expr)
        if cast:
            expr = cast.group(1).strip()

        if re.fullmatch(r"this\.([A-Za-z_]\w*)", expr):
            return expr.split(".", 1)[1]
        if re.fullmatch(r"[A-Za-z_]\w*(?:\[[^\]]+\])?", expr):
            return re.sub(r"\[[^\]]+\]$", "", expr)
        if re.fullmatch(r"[A-Za-z_]\w*", expr):
            return expr
        return None

    def extract_nominal_type_name_for_init(self, type_text: str) -> Optional[str]:
        t = str(type_text or "")
        t = re.sub(r"^(@[A-Za-z_]\w*(?:\([^)]*\))?\s+)+", "", t)
        t = re.sub(r"\b(final|volatile|transient|public|private|protected|static)\b", " ", t)
        t = re.sub(r"<[^>]*>", " ", t)
        t = t.replace("[]", " ").replace("*", " ").replace("&", " ")
        t = re.sub(r"\s+", " ", t).strip()
        tokens = re.findall(r"[A-Za-z_]\w*", t)
        if not tokens:
            return None
        skip = {
            "byte", "short", "int", "long", "float", "double", "boolean", "char", "void", "var",
            "extends", "super",
        }
        for tok in reversed(tokens):
            if tok.lower() not in skip:
                return tok
        return None
