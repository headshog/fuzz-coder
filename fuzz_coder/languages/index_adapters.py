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


def _strip_default_initializer_top_level(param_text: str) -> str:
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


def _extract_trailing_identifier_top_level(param_text: str) -> Optional[Tuple[str, int]]:
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


def _normalize_expr(expr: str, max_len: int = 240) -> str:
    e = " ".join(str(expr or "").split()).strip()
    if len(e) <= max_len:
        return e
    return e[: max_len - 3].rstrip() + "..."


def _extract_simple_identifiers(expr: str) -> List[str]:
    e = str(expr or "").strip()
    if not e:
        return []
    return re.findall(r"[A-Za-z_]\w*", e)


@dataclass(frozen=True)
class IndexLanguageAdapter:
    name: str

    def scan_signature_for_body(self, signature_text: str) -> Tuple[int, bool]:
        return _scan_signature_for_body_impl(signature_text)

    def find_matching_paren(self, text: str, pos: int) -> int:
        return _find_matching_paren_impl(text, pos)

    def split_top_level_params(self, params_text: str) -> List[str]:
        return _split_top_level_params_impl(params_text)

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


class CCppIndexLanguageAdapter(IndexLanguageAdapter):
    def parse_single_parameter_simple(self, param_text: str) -> Tuple[str, str]:
        raw = param_text.strip()
        if not raw:
            return "unknown", "unknown"

        no_default = _strip_default_initializer_top_level(raw)
        no_default = re.sub(r"\b__attribute__\s*\(\(.*?\)\)", " ", no_default)
        no_default = re.sub(r"\[\[.*?\]\]", " ", no_default)
        no_default = re.sub(r"\s+", " ", no_default).strip()
        if no_default == "...":
            return "varargs", "..."

        pack = re.search(r"\.\.\.\s*([A-Za-z_]\w*)\s*$", no_default)
        if pack:
            name = pack.group(1)
            ptype = no_default[:pack.start(1)].strip()
            return name, ptype or "unknown"

        fp = re.search(
            r"\(\s*[*&]\s*(?:(?:const|volatile|restrict|__restrict__)\s+)*([A-Za-z_]\w*)\s*\)",
            no_default,
        )
        if fp:
            name = fp.group(1)
            return name, no_default

        arr = re.search(r"([A-Za-z_]\w*)\s*(\[[^\]]*\])\s*$", no_default)
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
        ptr_decl = re.compile(
            r"^\s*(?P<type>(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\*\s*"
            r"(?P<var>[A-Za-z_]\w*)\b"
        )
        val_decl = re.compile(
            r"^\s*(?P<type>(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+"
            r"(?P<var>[A-Za-z_]\w*)\b"
        )

        for raw in str(code or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            if "(" in line and ")" in line and line.endswith("{"):
                continue

            match = ptr_decl.match(line)
            is_ptr = True
            if not match:
                match = val_decl.match(line)
                is_ptr = False
            if not match:
                continue

            var = str(match.group("var") or "").strip()
            nominal = self.extract_nominal_type_name_for_init(match.group("type"))
            if not var or not nominal:
                continue
            hints[var] = {"nominal_type": nominal, "is_pointer": is_ptr}
        return hints

    def extract_field_assignments(self, code: str) -> List[Dict]:
        out: List[Dict] = []
        pat = re.compile(
            r"^\s*(?P<base>[A-Za-z_]\w*)\s*"
            r"(?P<tail>(?:(?:\.|->)\s*[A-Za-z_]\w+)+)\s*=\s*"
            r"(?P<rhs>[^;]+)\s*;\s*$"
        )

        for li, raw in enumerate(str(code or "").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            if "==" in line or "+=" in line or "-=" in line or "*=" in line or "/=" in line:
                continue
            m = pat.match(line)
            if not m:
                continue

            base = str(m.group("base") or "").strip()
            tail = str(m.group("tail") or "")
            rhs = str(m.group("rhs") or "").strip()
            if not base or not tail or not rhs:
                continue

            pieces = re.findall(r"(\.|->)\s*([A-Za-z_]\w+)", tail)
            if not pieces:
                continue
            access = "arrow" if pieces[0][0] == "->" else "dot"
            path = ".".join(name for _, name in pieces)
            out.append({
                "line": li,
                "base": base,
                "access": access,
                "path": path,
                "rhs": _normalize_expr(rhs),
            })
        return out

    def extract_field_reads(self, code: str) -> List[Dict]:
        out: List[Dict] = []
        access_pat = re.compile(r"(?P<base>[A-Za-z_]\w*)\s*(?P<tail>(?:(?:\.|->)\s*[A-Za-z_]\w+)+)")
        assign_pat = re.compile(
            r"^\s*(?P<base>[A-Za-z_]\w*)\s*(?P<tail>(?:(?:\.|->)\s*[A-Za-z_]\w+)+)\s*=\s*[^;]+;\s*$"
        )

        for li, raw in enumerate(str(code or "").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue

            write_key = None
            m_assign = assign_pat.match(line)
            if m_assign:
                base = str(m_assign.group("base") or "").strip()
                tail = str(m_assign.group("tail") or "")
                pieces = re.findall(r"(\.|->)\s*([A-Za-z_]\w+)", tail)
                if pieces:
                    access = "arrow" if pieces[0][0] == "->" else "dot"
                    path = ".".join(name for _, name in pieces)
                    write_key = (base, access, path)

            for m in access_pat.finditer(line):
                base = str(m.group("base") or "").strip()
                tail = str(m.group("tail") or "")
                if not base or not tail:
                    continue
                pieces = re.findall(r"(\.|->)\s*([A-Za-z_]\w+)", tail)
                if not pieces:
                    continue
                access = "arrow" if pieces[0][0] == "->" else "dot"
                path = ".".join(name for _, name in pieces)
                key = (base, access, path)
                if write_key is not None and key == write_key:
                    continue
                out.append({"line": li, "base": base, "access": access, "path": path})
        return out

    def extract_call_sites(self, code: str, control_keywords: Sequence[str]) -> List[Dict]:
        out: List[Dict] = []
        control_set = set(control_keywords or [])
        call_re = re.compile(
            r"(?<![A-Za-z0-9_~])(?P<expr>(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*|[A-Za-z_]\w*\s*(?:->|\.)\s*[A-Za-z_]\w*)\s*\("
        )

        for m in call_re.finditer(code or ""):
            expr = re.sub(r"\s+", "", (m.group("expr") or ""))
            if not expr:
                continue
            if "->" in expr:
                name = expr.split("->")[-1]
            elif "." in expr:
                name = expr.split(".")[-1]
            else:
                name = expr.split("::")[-1]
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
            r"^\s*(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*(?:\s*[*&]\s*)?\s*"
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
            if not line or line.startswith("//") or line.startswith("#"):
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
        if re.fullmatch(r"&\s*([A-Za-z_]\w*)", expr):
            return re.sub(r"^&\s*", "", expr).strip()
        if re.fullmatch(r"\*+\s*([A-Za-z_]\w*)", expr):
            return re.sub(r"^\*+\s*", "", expr).strip()
        if re.fullmatch(r"[A-Za-z_]\w*", expr):
            return expr
        return None

    def extract_nominal_type_name_for_init(self, type_text: str) -> Optional[str]:
        t = re.sub(r"\b(const|volatile|restrict|__restrict__)\b", " ", str(type_text or ""))
        t = re.sub(r"\s+", " ", t).strip()
        t = t.replace("*", " ").replace("&", " ")
        tokens = re.findall(r"[A-Za-z_]\w*", t)
        if not tokens:
            return None
        skip = {
            "struct", "class", "enum",
            "unsigned", "signed", "long", "short",
            "int", "float", "double", "bool", "void",
            "size_t", "ssize_t", "auto", "typename",
        }
        for tok in reversed(tokens):
            if tok.lower() not in skip:
                return tok
        return None


class JavaIndexLanguageAdapter(IndexLanguageAdapter):
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
