#!/usr/bin/env python3
"""
Hybrid Code Indexer with Tree-sitter AST, Call Graph, and Semantic Search
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from pathlib import Path
import zipfile
import re
import os
from bisect import bisect_right
from .call_graph import detect_calls as _detect_calls_impl
from .call_graph import build_call_graph as _build_call_graph_impl
from fuzz_coder.languages.registry import get_language_frontend

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

SUPPORTED_EXT = {
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"
}

CONTROL_KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "catch",
    "new", "delete", "throw", "else", "do", "class", "struct",
    "namespace", "template", "typedef", "using", "enum", "union"
}

# Patterns for detecting input sources
STDIN_PATTERNS = [
    r"\b(cin)\s*>>",
    r"\b(scanf|getchar|getc|gets|gets_s)\s*\(",
    r"\b(fgets|fscanf)\s*\([^)]*(stdin)\b",
    r"\bstd::getline\s*\(\s*(std::)?cin\b",
    r"\bgetline\s*\(\s*(std::)?cin\b",
    r"\bread\s*\(\s*(0|STDIN_FILENO)\b",
    r"\b(std::)?cin\b",
    r"\b(System\.Console\.Read)\b",
    r"\b(Console\.Read|ReadLine|ReadKey)\b",
    r"\b(input|raw_input)\s*\(",
    r"\b(sys\.stdin|process\.stdin)\b",
]

FILE_INPUT_PATTERNS = [
    r"\bf(open|fopen|ifstream|fstream)\s*\(",
    r"\b(fread|fgets|fscanf|fgetc|getc)\s*\(",
    r"\b(std::)?(ifstream|fstream|ofstream)\b",
    r"\b(File\.Open|File\.Read|StreamReader)\b",
    r"\b(fs\.readFileSync?|fs\.createReadStream)\b",
    r"\b(Path\.OpenText|File\.ReadAllText)\b",
]

API_CALL_PATTERNS = [
    r"\b(http_client|HttpClient|curl_easy|wget)\b",
    r"\b(requests\.(get|post|put|delete|patch))\b",
    r"\b(fetch|axios|XMLHttpRequest)\b",
    r"\b(urllib\.(request|urlopen))\b",
    r"\b(httplib::Client|boost::beast)\b",
]

# Additional patterns for better code understanding
OUTPUT_PATTERNS = [
    r"\b(cout|printf|fprintf|sprintf)\b",
    r"\b(std::)?(cout|cerr|clog|print|writeln)\b",
    r"\b(Console\.Write|System\.out)\b",
]

MEMORY_MANAGEMENT_PATTERNS = [
    r"\b(malloc|calloc|realloc|free)\b",
    r"\b(new|delete)\b",
    r"\b(shared_ptr|unique_ptr|weak_ptr)\b",
]

ERROR_HANDLING_PATTERNS = [
    r"\b(throw|try|catch|finally)\b",
    r"\b(errno|perror|strerror)\b",
    r"\b(assert|static_assert)\b",
]

# Type system patterns for better type-based queries
TYPE_PATTERNS = {
    "byte_array": [r"\b(uint8_t|unsigned\s+char|char\s*\*|std::vector<uint8_t>|QByteArray|ByteBuffer)\b"],
    "string": [r"\b(std::string|char\s*\*|const\s+char\s*\*|QString|std::wstring)\b"],
    "integer": [r"\b(int|long|short|int32_t|int64_t|size_t|ssize_t)\b"],
    "float": [r"\b(float|double|long\s+double)\b"],
    "pointer": [r"\w+\s*\*\s*\w+"],
    "reference": [r"\w+\s*&\s*\w+"],
    "template": [r"\b(std::vector|std::map|std::set|std::unordered_map|std::array)\b"],
}


def unpack_if_zip(src, dst):
    src = Path(src)
    if src.suffix == ".zip":
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(dst)
        return dst
    return src


def collect_files(src):
    files = []
    for root, dirs, names in os.walk(src):
        # Skip hidden directories and common build directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                   {'build', 'dist', 'venv', '.venv', 'node_modules', '__pycache__',
                    'obj', 'bin', 'out', 'target'}]
        for n in names:
            p = Path(root)/n
            if p.suffix.lower() in SUPPORTED_EXT:
                files.append(p)
    return files


def read_text(path):
    try:
        return path.read_text(encoding="utf8")
    except:
        return path.read_text(errors="ignore")


def detect_input_type(code):
    """Detect if function uses stdin, file input, or API calls"""
    result = {
        "has_stdin": False,
        "has_file_input": False,
        "has_api_call": False,
        "input_sources": []
    }

    for pattern in STDIN_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            result["has_stdin"] = True
            result["input_sources"].append("stdin")
            break

    for pattern in FILE_INPUT_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            result["has_file_input"] = True
            result["input_sources"].append("file")
            break

    for pattern in API_CALL_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            result["has_api_call"] = True
            result["input_sources"].append("api")
            break

    return result


def detect_code_features(code):
    """Detect additional code features for better indexing"""
    result = {
        "has_output": False,
        "uses_memory_management": False,
        "has_error_handling": False,
        "types_used": [],
        "features": []
    }

    # Check for output operations
    for pattern in OUTPUT_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            result["has_output"] = True
            result["features"].append("output")
            break

    # Check for memory management
    for pattern in MEMORY_MANAGEMENT_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            result["uses_memory_management"] = True
            result["features"].append("memory_management")
            break

    # Check for error handling
    for pattern in ERROR_HANDLING_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            result["has_error_handling"] = True
            result["features"].append("error_handling")
            break

    # Detect types used
    for type_name, patterns in TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                result["types_used"].append(type_name)
                break

    return result


class TreeSitterParser:
    """Parse code using tree-sitter for accurate AST extraction"""

    def __init__(self, language_name="c_cpp"):
        self.language_name = language_name
        self.frontend = get_language_frontend(language_name)
        if not HAS_TREE_SITTER:
            self.parser = None
            return

        try:
            raw_language = self._get_raw_language(language_name)
            if raw_language is None:
                print(f"Warning: Tree-sitter grammar for language '{language_name}' is not available")
                self.parser = None
                return

            try:
                # tree-sitter API variant where Language(...) accepts capsule/ptr.
                self.language = Language(raw_language)
            except Exception:
                # tree-sitter API variant where language() already returns Language.
                self.language = raw_language

            self.parser = Parser()
            if hasattr(self.parser, "set_language"):
                # Older API
                self.parser.set_language(self.language)
            else:
                # Newer API
                self.parser.language = self.language
        except Exception as e:
            print(f"Warning: Could not initialize tree-sitter: {e}")
            self.parser = None

    def parse_functions(self, filepath, code):
        """Extract functions with parameters using tree-sitter"""
        if not self.parser:
            return extract_functions_regex(filepath, code, language_name=self.language_name)

        functions = []
        try:
            source_bytes = code.encode("utf-8", errors="ignore")
            tree = self.parser.parse(source_bytes)
            root = tree.root_node
            functions = self.frontend.parse_tree_sitter_functions(
                parser_utils=self,
                source_bytes=source_bytes,
                root=root,
                control_keywords=CONTROL_KEYWORDS,
            )

        except Exception as e:
            print(f"Tree-sitter error for {filepath}: {e}")
            # Fallback to regex
            functions = extract_functions_regex(filepath, code, language_name=self.language_name)

        if not functions:
            functions = extract_functions_regex(filepath, code, language_name=self.language_name)

        return functions

    def _get_raw_language(self, language_name):
        _ = language_name
        return self.frontend.get_tree_sitter_raw_language()

    def _parse_parameters(self, params_text):
        """Parse parameter list into structured format"""
        params = []
        # Remove parentheses
        params_text = params_text.strip("()")
        if not params_text.strip():
            return params

        # Split by comma (handling nested templates)
        depth = 0
        current = ""
        for char in params_text:
            if char in "<(":
                depth += 1
            elif char in ">)":
                depth -= 1
            elif char == "," and depth == 0:
                if current.strip():
                    params.append(self._parse_single_param(current.strip()))
                current = ""
                continue
            current += char

        if current.strip():
            params.append(self._parse_single_param(current.strip()))

        return params

    def _parse_single_param(self, param_text):
        """Parse a single parameter"""
        parts = param_text.split()
        if len(parts) >= 2:
            # Try to identify type and name
            name = parts[-1].split("&")[-1].split("*")[-1].strip()
            param_type = " ".join(parts[:-1])

            # Check if it's a reference or pointer
            is_reference = "&" in param_text
            is_pointer = "*" in param_text
            is_const = "const" in param_text.lower()

            return {
                "name": name,
                "type": param_type,
                "is_reference": is_reference,
                "is_pointer": is_pointer,
                "is_const": is_const,
                "raw": param_text
            }
        else:
            return {
                "name": param_text,
                "type": "unknown",
                "is_reference": False,
                "is_pointer": False,
                "is_const": False,
                "raw": param_text
            }

    def _iter_nodes_by_type(self, root_node, node_type):
        """Yield nodes of a specific type via DFS traversal."""
        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.type == node_type:
                yield node
            # Reverse children to keep left-to-right traversal order.
            stack.extend(reversed(node.children))

    def _extract_function_name_node(self, func_node):
        """Extract function name node from function_definition declarator chain."""
        declarator = func_node.child_by_field_name("declarator")
        if declarator is None:
            return None

        current = declarator
        for _ in range(32):
            if current.type in {"identifier", "field_identifier"}:
                return current

            if current.type == "qualified_identifier":
                name_node = current.child_by_field_name("name")
                if name_node is not None:
                    if name_node.type in {"identifier", "field_identifier"}:
                        return name_node
                    current = name_node
                    continue

            next_decl = current.child_by_field_name("declarator")
            if next_decl is None:
                break
            current = next_decl

        return None

    def _extract_function_params_node(self, func_node):
        """Extract parameter list node from function_definition declarator chain."""
        declarator = func_node.child_by_field_name("declarator")
        if declarator is None:
            return None

        current = declarator
        for _ in range(32):
            if current.type == "function_declarator":
                params = current.child_by_field_name("parameters")
                if params is not None:
                    return params

            next_decl = current.child_by_field_name("declarator")
            if next_decl is None:
                break
            current = next_decl

        return None

    def _build_signature(self, name, params):
        """Build function signature string"""
        param_strs = [p["raw"] for p in params]
        return f"{name}({', '.join(param_strs)})"


def extract_functions_regex(_filepath, text, language_name="c_cpp"):
    """Fallback regex-based function extraction"""
    res = []
    frontend = get_language_frontend(language_name)
    lines_with_end = text.splitlines(keepends=True)
    lines = [ln.rstrip("\r\n") for ln in lines_with_end]
    # Keep fallback conservative but resilient for heavy template/macros signatures.
    max_signature_scan_lines = 160
    max_signature_scan_chars = 40000

    offsets = []
    cur = 0
    for line in lines_with_end:
        offsets.append(cur)
        cur += len(line)

    i = 0
    n = len(lines)

    while i < n:
        if "(" not in lines[i]:
            i += 1
            continue

        start_i = i
        sig_lines = []
        found_body = False
        body_brace_pos = -1
        j = i
        scanned_chars = 0

        while j < n and (j - i) < max_signature_scan_lines and scanned_chars < max_signature_scan_chars:
            line = lines[j]
            sig_lines.append(line)
            scanned_chars += len(line) + 1
            joined = "\n".join(sig_lines)

            brace_pos, terminated_decl = scan_signature_for_body(joined)

            if brace_pos != -1:
                found_body = True
                body_brace_pos = brace_pos
                break

            if terminated_decl:
                break

            j += 1

        if not found_body:
            i += 1
            continue

        joined = "\n".join(sig_lines)
        brace_pos = body_brace_pos
        signature = joined[:brace_pos].strip()

        if not signature or ")" not in signature:
            i += 1
            continue

        compact = " ".join(signature.split())

        # Regex fallback must only index function definitions, not declaration/macro blobs.
        if ";" in compact:
            i += 1
            continue
        if any(tok in compact for tok in ["///", "/*", "*/", "DEPRECATED("]):
            i += 1
            continue

        compact = frontend.preprocess_regex_signature(compact)

        bad_prefixes = ("if ", "for ", "while ", "switch ",
                        "catch ", "#", "typedef ", "return ", "class ", "struct ")
        bad_prefixes = frontend.merge_regex_bad_prefixes(bad_prefixes)
        if compact.startswith(bad_prefixes):
            i += 1
            continue

        lp = compact.find("(")
        head = compact[:lp].strip()
        if not head:
            i += 1
            continue

        m = re.search(r"([A-Za-z_]\w*)\s*$", head)
        if not m:
            i += 1
            continue

        name = m.group(1)
        # Guard against macro invocations accidentally captured as "functions",
        # e.g. DEPRECATED(...), likely when fallback parser spans declaration blocks.
        if head == name and re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            i += 1
            continue
        if name in CONTROL_KEYWORDS:
            i += 1
            continue

        # Extract parameters
        rp = find_matching_paren(compact, lp)
        if rp != -1:
            params_text = compact[lp+1:rp]
            params = parse_parameters_simple(params_text)
        else:
            params = []

        start_offset = offsets[start_i]
        # Important: bind to brace detected inside this exact signature window.
        open_brace = start_offset + brace_pos
        if open_brace >= len(text) or text[open_brace] != "{":
            # Fallback for uncommon newline encodings.
            window_end = min(len(text), start_offset + len(joined) + 4)
            open_brace = text.find("{", start_offset, window_end)
        if open_brace == -1:
            i += 1
            continue

        end = find_matching_brace(text, open_brace)
        if end == -1:
            i += 1
            continue

        code = text[start_offset:end + 1]
        start_line = bisect_right(offsets, start_offset)
        end_line_idx = max(0, bisect_right(offsets, end) - 1)
        end_line = end_line_idx + 1
        res.append({
            "name": name,
            "signature": compact,
            "parameters": params,
            "code": code,
            "body": code,
            "start_line": start_line,
            "end_line": end_line,
            "parser": "regex"
        })
        i = end_line_idx + 1

    return res


def scan_signature_for_body(signature_text):
    """Find top-level body brace in a function signature window.

    Returns (brace_pos, terminated_decl):
      - brace_pos >= 0: opening "{" of function body found
      - terminated_decl True: declaration ended with ";" before any body
    """
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


def parse_parameters_simple(params_text):
    """Simple parameter parsing for regex fallback"""
    params = []
    params_text = params_text.strip()
    if not params_text or params_text == "void":
        return params

    for idx, param in enumerate(split_top_level_params(params_text)):
        param = param.strip()
        if not param:
            continue

        name, param_type = parse_single_parameter_simple(param)
        if not name or name == "unknown":
            name = f"arg{idx}"
        params.append({
            "name": name,
            "type": param_type,
            "is_reference": "&" in param,
            "is_pointer": "*" in param,
            "is_const": "const" in param.lower(),
            "raw": param
        })

    return params


def split_top_level_params(params_text):
    """Split C/C++ parameter list by top-level commas."""
    out = []
    cur = []

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


def parse_single_parameter_simple(param_text):
    """Best-effort parsing of a single C/C++ parameter."""
    raw = param_text.strip()
    if not raw:
        return "unknown", "unknown"

    no_default = _strip_default_initializer_top_level(raw)
    no_default = re.sub(r"\b__attribute__\s*\(\(.*?\)\)", " ", no_default)
    no_default = re.sub(r"\[\[.*?\]\]", " ", no_default)
    no_default = re.sub(r"\s+", " ", no_default).strip()
    if no_default == "...":
        return "varargs", "..."

    # Named parameter packs: Args&&... args
    pack = re.search(r"\.\.\.\s*([A-Za-z_]\w*)\s*$", no_default)
    if pack:
        name = pack.group(1)
        ptype = no_default[:pack.start(1)].strip()
        return name, ptype or "unknown"

    # Function pointer param: void (*cb)(int)
    fp = re.search(
        r"\(\s*[*&]\s*(?:(?:const|volatile|restrict|__restrict__)\s+)*([A-Za-z_]\w*)\s*\)",
        no_default,
    )
    if fp:
        name = fp.group(1)
        return name, no_default

    # Array-style parameter: int data[4]
    arr = re.search(r"([A-Za-z_]\w*)\s*(\[[^\]]*\])\s*$", no_default)
    if arr:
        name = arr.group(1)
        ptype = no_default[:arr.start(1)].strip()
        return name, ptype or "unknown"

    # Generic trailing identifier at top-level.
    trailing = _extract_trailing_identifier_top_level(no_default)
    if trailing:
        name, start_idx = trailing
        ptype = no_default[:start_idx].strip()
        if ptype:
            return name, ptype
        return name, "unknown"

    return raw, "unknown"


def _strip_default_initializer_top_level(param_text):
    """Strip default initializer (`= ...`) only at top-level."""
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


def _extract_trailing_identifier_top_level(param_text):
    """Return (identifier, start_idx) for a trailing top-level parameter name."""
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
                # Ensure name is really trailing token (skip known type-only endings).
                if name and name not in {
                    "const", "volatile", "unsigned", "signed", "short", "long",
                    "int", "char", "float", "double", "bool", "void", "size_t",
                    "ssize_t", "struct", "class", "enum", "typename", "auto",
                }:
                    return name, start + 1
                return None
            if ch.isspace():
                i -= 1
                continue
            # Hit top-level non-identifier token before any trailing identifier.
            if ch in "*&])":
                i -= 1
                continue
            return None

        i -= 1
    return None


def find_matching_brace(text, pos):
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

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


def find_matching_paren(text, pos):
    """Find matching ')' for '(' at position pos, honoring nested parens and literals/comments."""
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


def detect_calls(body, language_name="c_cpp"):
    """Detect function calls in code body."""
    return _detect_calls_impl(body, CONTROL_KEYWORDS, language_name=language_name)


def build_call_graph(chunks, language_name="c_cpp"):
    """Build a call graph from function chunks."""
    return _build_call_graph_impl(chunks, CONTROL_KEYWORDS, language_name=language_name)


def build_embeddings(chunks, model_name, embedding_backend=None):
    """Build semantic embeddings with enriched context"""
    model = SentenceTransformer(model_name) if embedding_backend is None else embedding_backend

    texts = []
    for c in chunks:
        # Create enriched text for better semantic search
        param_info = ""
        if c.get("parameters"):
            param_names = [p["name"] for p in c["parameters"]]
            param_types = [p["type"] for p in c["parameters"]]
            param_info = f"parameters: {', '.join(param_names)}; types: {', '.join(param_types)}"

        input_info = ""
        if c.get("has_stdin"):
            input_info = " reads from stdin"
        elif c.get("has_file_input"):
            input_info = " reads from files"
        elif c.get("has_api_call"):
            input_info = " makes API calls"

        t = f"""function: {c["name"]}
{c["signature"]}
{param_info}
file: {c["file"]}
{input_info}
code:
{c["code"]}"""
        texts.append(t)

    if embedding_backend is None:
        emb = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    else:
        emb = model.encode(texts)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, 1e-9, None)
    return emb


def build_lexical_index(chunks):
    """Build lexical index for keyword search"""
    idx = defaultdict(list)
    for i, c in enumerate(chunks):
        # Index function name, parameters, and code
        words = set()
        words.update(re.findall(r"[A-Za-z_]\w+", c["name"].lower()))
        words.update(re.findall(r"[A-Za-z_]\w+", c["code"].lower()))

        if c.get("parameters"):
            for p in c["parameters"]:
                words.add(p["name"].lower())
                words.update(re.findall(r"[A-Za-z_]\w+", p["type"].lower()))

        if c.get("input_sources"):
            words.update(c["input_sources"])

        for w in words:
            idx[w].append(i)
    return idx


def _split_call_args_top_level(args_text):
    args_text = str(args_text or "").strip()
    if not args_text:
        return []
    return split_top_level_params(args_text)


def _is_literal_like_token(token):
    t = str(token or "").strip()
    if not t:
        return False
    if re.fullmatch(r"(?:[-+]?\d+(?:\.\d+)?(?:[uUlLfF]*)|nullptr|NULL|true|false)", t):
        return True
    if re.fullmatch(r"'(?:\\.|[^'])+'", t):
        return True
    if re.fullmatch(r"\"(?:\\.|[^\"])*\"", t):
        return True
    # allow namespaced/UPPERCASE constants
    if re.fullmatch(r"(?:[A-Za-z_]\w*::)*[A-Z_][A-Z0-9_]*", t):
        return True
    return False


def _is_self_contained_call_expr(expr):
    e = str(expr or "").strip()
    m = re.fullmatch(r"(?P<fn>(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*\((?P<args>.*)\)", e)
    if not m:
        return False
    args = _split_call_args_top_level(m.group("args") or "")
    if not args:
        return True
    for a in args:
        if not _is_literal_like_token(a):
            return False
    return True


def _extract_nominal_type_name_for_init(type_text):
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


def _normalize_expr(expr, max_len=240):
    e = " ".join(str(expr or "").split()).strip()
    if len(e) <= max_len:
        return e
    return e[: max_len - 3].rstrip() + "..."


def _extract_call_name(expr):
    e = str(expr or "").strip()
    m = re.match(r"((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*\(", e)
    if not m:
        return None
    return m.group(1).split("::")[-1]


def _init_pattern_score(kind, expr):
    score = 0
    e = str(expr or "").lower()
    if kind == "pointer_call":
        score += 6
    elif kind == "value_call":
        score += 5
    elif kind == "value_brace":
        score += 3
    elif kind == "value_default":
        score += 1

    for kw in ["alloc", "create", "init", "default", "open", "new"]:
        if kw in e:
            score += 2
    if _is_self_contained_call_expr(expr):
        score += 3
    return score


def _extract_variable_type_hints(code):
    """Extract local variable -> nominal type hints from a function body."""
    hints = {}
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
        nominal = _extract_nominal_type_name_for_init(match.group("type"))
        if not var or not nominal:
            continue
        hints[var] = {"nominal_type": nominal, "is_pointer": is_ptr}
    return hints


def _extract_field_assignments(code):
    """Extract assignments like cfg.x = ... and cfg->x = ..."""
    out = []
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


def _extract_field_reads(code):
    """Extract field reads like cfg.x / cfg->x (excluding direct LHS writes)."""
    out = []
    access_pat = re.compile(
        r"(?P<base>[A-Za-z_]\w*)\s*(?P<tail>(?:(?:\.|->)\s*[A-Za-z_]\w+)+)"
    )
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
            out.append({
                "line": li,
                "base": base,
                "access": access,
                "path": path,
            })
    return out


def _extract_call_sites(code, control_keywords):
    """Extract call sites with name/arity/args/line from function body."""
    out = []
    masked = code
    # Reuse call-regex from call graph module semantics.
    call_re = re.compile(
        r"(?<![A-Za-z0-9_~])(?P<expr>"
        r"(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*"
        r"|"
        r"[A-Za-z_]\w*\s*(?:->|\.)\s*[A-Za-z_]\w*"
        r")\s*\("
    )

    for m in call_re.finditer(masked):
        expr = re.sub(r"\s+", "", (m.group("expr") or ""))
        if not expr:
            continue
        if "->" in expr:
            name = expr.split("->")[-1]
        elif "." in expr:
            name = expr.split(".")[-1]
        else:
            name = expr.split("::")[-1]
        if not name or name in control_keywords:
            continue

        open_idx = m.end() - 1
        close_idx = find_matching_paren(code, open_idx)
        if close_idx == -1:
            continue
        args_text = code[open_idx + 1:close_idx]
        args = _split_call_args_top_level(args_text)
        line_no = code.count("\n", 0, m.start()) + 1
        out.append({
            "name": name,
            "arity": len(args),
            "args": args,
            "line": line_no,
        })
    return out


def _extract_simple_identifiers(expr):
    e = str(expr or "").strip()
    if not e:
        return []
    return re.findall(r"[A-Za-z_]\w*", e)


def _extract_dependency_chain(code, base_var, before_line, max_steps=8):
    """Build simple backward dependency chain: tmp -> cfg -> call."""
    lines = str(code or "").splitlines()
    upto = max(0, int(before_line) - 1)
    if not lines or not base_var:
        return []

    decl_init_pat = re.compile(
        r"^\s*(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*(?:\s*[*&]\s*)?\s*"
        r"(?P<lhs>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;]+)\s*;\s*$"
    )
    assign_pat = re.compile(
        r"^\s*(?P<lhs>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;]+)\s*;\s*$"
    )

    tracked = {str(base_var)}
    chain = []
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
        chain.append({
            "line": li,
            "statement": stmt,
        })

        for tok in _extract_simple_identifiers(rhs):
            if tok != lhs:
                tracked.add(tok)

    chain.sort(key=lambda x: int(x.get("line", 0)))
    return chain


def _extract_arg_base_var(arg_expr):
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


def _field_rhs_score(expr):
    e = _normalize_expr(expr)
    if _is_literal_like_token(e):
        return 6
    if _is_self_contained_call_expr(e):
        return 5
    if re.fullmatch(r"(?:[A-Za-z_]\w*::)*[A-Z_][A-Z0-9_]*", e):
        return 4
    return 1


def _recipe_rhs_score(expr):
    e = _normalize_expr(expr)
    if _is_self_contained_call_expr(e):
        return 8
    if _is_literal_like_token(e):
        return 6
    if re.fullmatch(r"(?:[A-Za-z_]\w*::)*[A-Z_][A-Z0-9_]*", e):
        return 5
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", e):
        return 4
    return 1


def build_type_init_index(chunks, max_per_type=16):
    """Build v3 structural dataflow index for better example grounding."""
    by_type = defaultdict(dict)  # type -> (kind, expr) -> aggregated record
    by_name = defaultdict(list)
    for c in chunks or []:
        name = str(c.get("name", "")).strip()
        if name:
            by_name[name].append(c)

    pat_ptr_eq = re.compile(
        r"^\s*(?P<type>(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\*\s*"
        r"(?P<var>[A-Za-z_]\w*)\s*=\s*(?P<expr>[^;]+)\s*;\s*$"
    )
    pat_val_eq = re.compile(
        r"^\s*(?P<type>(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+"
        r"(?P<var>[A-Za-z_]\w*)\s*=\s*(?P<expr>[^;]+)\s*;\s*$"
    )
    pat_val_ctor = re.compile(
        r"^\s*(?P<type>(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+"
        r"(?P<var>[A-Za-z_]\w*)\s*\((?P<expr>[^;]*)\)\s*;\s*$"
    )
    pat_val_brace = re.compile(
        r"^\s*(?P<type>(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+"
        r"(?P<var>[A-Za-z_]\w*)\s*(?P<expr>\{[^;]*\})\s*;\s*$"
    )
    pat_val_default = re.compile(
        r"^\s*(?P<type>(?:const\s+)?(?:struct\s+|class\s+|enum\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+"
        r"(?P<var>[A-Za-z_]\w*)\s*;\s*$"
    )

    struct_field_writes_acc = defaultdict(dict)  # nominal -> (path,access) -> rec
    function_effects_acc = {}  # (fn,sig,arg_i,arg_name,type) -> rec
    helper_effect_lookup = defaultdict(list)  # (helper_fn,arg_index) -> [{"type","writes"}]
    callsite_arg_flow = defaultdict(list)
    req = {}
    recipe_acc = {}

    def _ensure_req_bucket(target_name, arg_index, arg_name, nominal):
        key = (str(target_name), int(arg_index), str(arg_name), str(nominal))
        bucket = req.get(key)
        if bucket is None:
            bucket = {
                "function": key[0],
                "arg_index": key[1],
                "arg_name": key[2],
                "type": key[3],
                "call_sites": 0,
                "fields": {},  # (path, access) -> rec
            }
            req[key] = bucket
        return bucket

    def _add_req_field(bucket, *, path, access, rhs, evidence, source_tag):
        if not path:
            return
        fkey = (str(path), str(access or "dot"))
        rec = bucket["fields"].get(fkey)
        if rec is None:
            rec = {
                "path": fkey[0],
                "access": fkey[1],
                "count": 0,
                "rhs": defaultdict(int),
                "evidence": [],
                "sources": defaultdict(int),
            }
            bucket["fields"][fkey] = rec
        rec["count"] += 1
        rhs_norm = _normalize_expr(rhs) if rhs else ""
        if rhs_norm:
            rec["rhs"][rhs_norm] += 1
        rec["sources"][str(source_tag or "unknown")] += 1
        if evidence and len(rec["evidence"]) < 4:
            rec["evidence"].append(dict(evidence))

    def _ensure_recipe_bucket(nominal, target_name, arg_index, arg_name, is_pointer):
        key = (str(nominal), str(target_name), int(arg_index), str(arg_name))
        bucket = recipe_acc.get(key)
        if bucket is None:
            bucket = {
                "type": key[0],
                "target_function": key[1],
                "arg_index": key[2],
                "arg_name": key[3],
                "is_pointer": bool(is_pointer),
                "call_sites": 0,
                "var_assign": defaultdict(int),  # rhs -> count
                "fields": {},  # (path,access) -> rec
                "evidence": [],
            }
            recipe_acc[key] = bucket
        return bucket

    def _add_recipe_field(bucket, *, path, access, rhs, evidence, source_tag):
        if not path:
            return
        fkey = (str(path), str(access or "dot"))
        rec = bucket["fields"].get(fkey)
        if rec is None:
            rec = {
                "path": fkey[0],
                "access": fkey[1],
                "count": 0,
                "rhs": defaultdict(int),
                "sources": defaultdict(int),
                "evidence": [],
            }
            bucket["fields"][fkey] = rec
        rec["count"] += 1
        rhs_norm = _normalize_expr(rhs) if rhs else ""
        if rhs_norm:
            rec["rhs"][rhs_norm] += 1
        rec["sources"][str(source_tag or "unknown")] += 1
        if evidence and len(rec["evidence"]) < 4:
            rec["evidence"].append(dict(evidence))

    # Pass 1: aggregate init patterns + function effects + struct writes.
    for c in chunks or []:
        file_path = str(c.get("file", ""))
        fn_name = str(c.get("name", "")).strip()
        signature = str(c.get("signature", "")).strip() or (f"{fn_name}()")
        start_line = int(c.get("start_line", 1) or 1)
        code = str(c.get("code", "") or c.get("body", ""))
        if not code:
            continue

        # v3/A: init patterns per nominal type (existing logic)
        for li, raw in enumerate(code.splitlines(), start=1):
            line = (raw or "").strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue

            kind = None
            type_text = None
            expr = None
            match = None
            for k, pat in [
                ("pointer_call", pat_ptr_eq),
                ("value_call", pat_val_eq),
                ("value_call", pat_val_ctor),
                ("value_brace", pat_val_brace),
                ("value_default", pat_val_default),
            ]:
                m = pat.match(line)
                if m:
                    match = m
                    kind = k
                    type_text = m.group("type")
                    expr = (m.groupdict().get("expr") or "").strip()
                    break
            if not match or not kind or not type_text:
                continue
            if kind in {"pointer_call", "value_call"}:
                if not re.fullmatch(r"(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*\s*\([^;]*\)", expr or ""):
                    continue
            nominal = _extract_nominal_type_name_for_init(type_text)
            if not nominal:
                continue

            expr_norm = _normalize_expr(expr)
            key = (kind, expr_norm)
            rec = by_type[nominal].get(key)
            if rec is None:
                rec = {
                    "kind": kind,
                    "expr": expr_norm,
                    "function": _extract_call_name(expr_norm) if kind in {"pointer_call", "value_call"} else None,
                    "self_contained": _is_self_contained_call_expr(expr_norm) if kind in {"pointer_call", "value_call"} else (kind != "value_default"),
                    "count": 0,
                    "score": 0,
                    "evidence": [],
                }
                by_type[nominal][key] = rec
            rec["count"] += 1
            rec["score"] += _init_pattern_score(kind, expr_norm)
            if len(rec["evidence"]) < 3:
                rec["evidence"].append({
                    "file": file_path,
                    "line": start_line + li - 1,
                    "function": fn_name,
                })

        var_hints = _extract_variable_type_hints(code)
        field_writes = _extract_field_assignments(code)
        field_reads = _extract_field_reads(code)
        call_sites = _extract_call_sites(code, CONTROL_KEYWORDS)

        # v3/B: global struct field writes by nominal type
        for fw in field_writes:
            hint = var_hints.get(str(fw.get("base", "")).strip())
            if not hint:
                continue
            nominal = str(hint.get("nominal_type", "")).strip()
            if not nominal:
                continue
            k = (str(fw.get("path", "")), str(fw.get("access", "dot")))
            rec = struct_field_writes_acc[nominal].get(k)
            if rec is None:
                rec = {
                    "path": k[0],
                    "access": k[1],
                    "count": 0,
                    "rhs": defaultdict(int),
                    "functions": set(),
                    "evidence": [],
                }
                struct_field_writes_acc[nominal][k] = rec
            rec["count"] += 1
            rhs_norm = _normalize_expr(fw.get("rhs", ""))
            if rhs_norm:
                rec["rhs"][rhs_norm] += 1
            if fn_name:
                rec["functions"].add(fn_name)
            if len(rec["evidence"]) < 4:
                rec["evidence"].append({
                    "file": file_path,
                    "line": start_line + int(fw.get("line", 0)),
                    "function": fn_name,
                })

        # v3/C: per-function effects on parameter fields (+ helper calls)
        params = list(c.get("parameters") or [])
        for pi, param in enumerate(params):
            pname = str(param.get("name", "")).strip()
            ptype_nominal = _extract_nominal_type_name_for_init(param.get("type", ""))
            if not pname or not ptype_nominal:
                continue
            key = (fn_name, signature, int(pi), pname, ptype_nominal)
            eff = function_effects_acc.get(key)
            if eff is None:
                eff = {
                    "function": fn_name,
                    "signature": signature,
                    "arg_index": int(pi),
                    "arg_name": pname,
                    "type": ptype_nominal,
                    "writes": {},
                    "reads": {},
                    "helper_calls": defaultdict(int),
                }
                function_effects_acc[key] = eff

            for fw in field_writes:
                if str(fw.get("base", "")).strip() != pname:
                    continue
                rk = (str(fw.get("path", "")), str(fw.get("access", "dot")))
                wr = eff["writes"].get(rk)
                if wr is None:
                    wr = {
                        "path": rk[0],
                        "access": rk[1],
                        "count": 0,
                        "rhs": defaultdict(int),
                        "evidence": [],
                    }
                    eff["writes"][rk] = wr
                wr["count"] += 1
                rhs_norm = _normalize_expr(fw.get("rhs", ""))
                if rhs_norm:
                    wr["rhs"][rhs_norm] += 1
                if len(wr["evidence"]) < 3:
                    wr["evidence"].append({
                        "file": file_path,
                        "line": start_line + int(fw.get("line", 0)),
                        "function": fn_name,
                    })

            for fr in field_reads:
                if str(fr.get("base", "")).strip() != pname:
                    continue
                rk = (str(fr.get("path", "")), str(fr.get("access", "dot")))
                rr = eff["reads"].get(rk)
                if rr is None:
                    rr = {
                        "path": rk[0],
                        "access": rk[1],
                        "count": 0,
                        "evidence": [],
                    }
                    eff["reads"][rk] = rr
                rr["count"] += 1
                if len(rr["evidence"]) < 3:
                    rr["evidence"].append({
                        "file": file_path,
                        "line": start_line + int(fr.get("line", 0)),
                        "function": fn_name,
                    })

            for cs in call_sites:
                for ai, a in enumerate(list(cs.get("args") or [])):
                    if _extract_arg_base_var(a) != pname:
                        continue
                    helper_name = str(cs.get("name") or "").strip()
                    if helper_name:
                        eff["helper_calls"][(helper_name, int(ai))] += 1

    # finalize types_out
    types_out = {}
    for nominal, variants in by_type.items():
        ranked = sorted(
            variants.values(),
            key=lambda x: (x.get("score", 0), x.get("count", 0), len(str(x.get("expr", "")))),
            reverse=True,
        )
        types_out[nominal] = ranked[:max_per_type]

    # finalize struct_field_writes
    struct_field_writes_out = {}
    for nominal, fields in struct_field_writes_acc.items():
        rows = []
        for _, fr in fields.items():
            rhs_items = sorted(
                fr["rhs"].items(),
                key=lambda x: (_field_rhs_score(x[0]), x[1], len(str(x[0]))),
                reverse=True,
            )
            rows.append({
                "path": fr["path"],
                "access": fr["access"],
                "count": int(fr["count"]),
                "sample_expr": rhs_items[0][0] if rhs_items else "",
                "functions": sorted(list(fr["functions"]))[:16],
                "evidence": list(fr["evidence"]),
            })
        rows.sort(key=lambda x: (x.get("count", 0), len(str(x.get("path", "")))), reverse=True)
        struct_field_writes_out[nominal] = rows[: max(8, max_per_type * 4)]

    # finalize function_effects + helper lookup
    function_effects_by_name = defaultdict(list)
    for _, eff in function_effects_acc.items():
        writes = []
        for _, wr in eff["writes"].items():
            rhs_items = sorted(
                wr["rhs"].items(),
                key=lambda x: (_field_rhs_score(x[0]), x[1], len(str(x[0]))),
                reverse=True,
            )
            writes.append({
                "path": wr["path"],
                "access": wr["access"],
                "count": int(wr["count"]),
                "sample_expr": rhs_items[0][0] if rhs_items else "",
                "evidence": list(wr["evidence"]),
            })
        writes.sort(key=lambda x: (x.get("count", 0), len(str(x.get("path", "")))), reverse=True)

        reads = []
        for _, rr in eff["reads"].items():
            reads.append({
                "path": rr["path"],
                "access": rr["access"],
                "count": int(rr["count"]),
                "evidence": list(rr["evidence"]),
            })
        reads.sort(key=lambda x: (x.get("count", 0), len(str(x.get("path", "")))), reverse=True)

        helper_calls = []
        for (hname, hidx), cnt in sorted(eff["helper_calls"].items(), key=lambda x: (x[1], x[0]), reverse=True):
            helper_calls.append({
                "name": hname,
                "arg_index": int(hidx),
                "count": int(cnt),
            })

        row = {
            "signature": eff["signature"],
            "arg_index": int(eff["arg_index"]),
            "arg_name": eff["arg_name"],
            "type": eff["type"],
            "writes": writes[:24],
            "reads": reads[:24],
            "helper_calls": helper_calls[:16],
        }
        function_effects_by_name[eff["function"]].append(row)
        if row["writes"]:
            helper_effect_lookup[(eff["function"], int(eff["arg_index"]))].append({
                "type": eff["type"],
                "writes": row["writes"],
            })

    function_effects_out = {}
    for fn, entries in function_effects_by_name.items():
        function_effects_out[fn] = sorted(
            entries,
            key=lambda e: (int(e.get("arg_index", 0)), str(e.get("arg_name", ""))),
        )

    # Pass 2: callsite flow + required fields from direct dataflow + helper effects.
    for c in chunks or []:
        code = str(c.get("code", "") or c.get("body", ""))
        if not code:
            continue
        file_path = str(c.get("file", ""))
        caller_name = str(c.get("name", "")).strip()
        start_line = int(c.get("start_line", 1) or 1)
        var_hints = _extract_variable_type_hints(code)
        field_assignments = _extract_field_assignments(code)
        call_sites = _extract_call_sites(code, CONTROL_KEYWORDS)
        if not call_sites:
            continue

        for cs in call_sites:
            cs_name = str(cs.get("name") or "").strip()
            cs_line = int(cs.get("line") or 0)
            cs_args = list(cs.get("args") or [])

            arg_flow_rows = []
            for ai, arg in enumerate(cs_args):
                base_var = _extract_arg_base_var(arg)
                arg_row = {
                    "arg_index": int(ai),
                    "arg_expr": str(arg),
                    "base_var": base_var,
                }
                if base_var:
                    hint = var_hints.get(base_var)
                    if hint and str(hint.get("nominal_type", "")).strip():
                        arg_row["nominal_type"] = str(hint.get("nominal_type", "")).strip()

                    chain = _extract_dependency_chain(code, base_var, before_line=cs_line, max_steps=8)
                    if chain:
                        arg_row["dependency_chain"] = chain

                    f_before = []
                    seen_fb = set()
                    for fa in field_assignments:
                        if str(fa.get("base", "")) != base_var:
                            continue
                        if int(fa.get("line", 0)) >= cs_line:
                            continue
                        k = (str(fa.get("path", "")), str(fa.get("access", "dot")))
                        if k in seen_fb:
                            continue
                        seen_fb.add(k)
                        f_before.append({
                            "path": k[0],
                            "access": k[1],
                            "rhs": _normalize_expr(fa.get("rhs", "")),
                            "line": start_line + int(fa.get("line", 0)),
                        })
                    if f_before:
                        arg_row["field_writes_before_call"] = f_before[:12]

                    helper_fields = []
                    seen_hf = set()
                    for prev in call_sites:
                        if int(prev.get("line", 0)) >= cs_line:
                            continue
                        p_name = str(prev.get("name") or "").strip()
                        p_args = list(prev.get("args") or [])
                        for pj, pa in enumerate(p_args):
                            if _extract_arg_base_var(pa) != base_var:
                                continue
                            for he in helper_effect_lookup.get((p_name, int(pj)), []):
                                for wr in list(he.get("writes") or []):
                                    hk = (str(wr.get("path", "")), str(wr.get("access", "dot")), p_name)
                                    if not hk[0] or hk in seen_hf:
                                        continue
                                    seen_hf.add(hk)
                                    helper_fields.append({
                                        "path": hk[0],
                                        "access": hk[1],
                                        "helper": hk[2],
                                        "sample_expr": str(wr.get("sample_expr", "")),
                                    })
                    if helper_fields:
                        arg_row["helper_field_writes"] = helper_fields[:12]

                arg_flow_rows.append(arg_row)

            callsite_arg_flow[caller_name].append({
                "target": cs_name,
                "arity": int(cs.get("arity") or 0),
                "line": start_line + cs_line - 1 if cs_line > 0 else start_line,
                "file": file_path,
                "args": [str(a) for a in cs_args],
                "arg_flow": arg_flow_rows,
            })

            # Required fields inference against resolved target chunks.
            target_candidates = by_name.get(cs_name, [])
            if not target_candidates:
                continue
            arity = int(cs.get("arity") or 0)
            arity_matches = [tc for tc in target_candidates if len(list(tc.get("parameters") or [])) == arity]
            if arity_matches:
                target_candidates = arity_matches
            if not target_candidates:
                continue

            for target in target_candidates:
                params = list(target.get("parameters") or [])
                for i, param in enumerate(params):
                    if i >= len(cs_args):
                        continue
                    arg = cs_args[i]
                    base_var = _extract_arg_base_var(arg)
                    if not base_var:
                        continue
                    hint = var_hints.get(base_var)
                    if not hint:
                        continue
                    nominal = _extract_nominal_type_name_for_init(param.get("type", ""))
                    if not nominal:
                        continue
                    if str(hint.get("nominal_type", "")).lower() != nominal.lower():
                        continue

                    bucket = _ensure_req_bucket(
                        str(target.get("name", "")).strip(),
                        i,
                        str(param.get("name", "")).strip() or f"arg{i}",
                        nominal,
                    )
                    bucket["call_sites"] += 1

                    recipe_bucket = _ensure_recipe_bucket(
                        nominal,
                        str(target.get("name", "")).strip(),
                        i,
                        str(param.get("name", "")).strip() or f"arg{i}",
                        bool(hint.get("is_pointer")),
                    )
                    recipe_bucket["call_sites"] += 1
                    if len(recipe_bucket["evidence"]) < 4:
                        recipe_bucket["evidence"].append({
                            "file": file_path,
                            "line": start_line + cs_line - 1 if cs_line > 0 else start_line,
                            "function": caller_name,
                        })

                    chain = _extract_dependency_chain(code, base_var, before_line=cs_line, max_steps=8)
                    for ch in chain:
                        stmt = str(ch.get("statement", "")).strip()
                        m_assign = re.match(r"^\s*(?P<lhs>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;]+)\s*;\s*$", stmt)
                        if not m_assign:
                            continue
                        if str(m_assign.group("lhs") or "").strip() != base_var:
                            continue
                        rhs_norm = _normalize_expr(m_assign.group("rhs") or "")
                        if not rhs_norm:
                            continue
                        recipe_bucket["var_assign"][rhs_norm] += 1

                    seen_fields = set()
                    for fa in field_assignments:
                        if str(fa.get("base", "")) != base_var:
                            continue
                        if int(fa.get("line", 0)) >= cs_line:
                            continue
                        k = (str(fa.get("path", "")), str(fa.get("access", "dot")))
                        if k in seen_fields:
                            continue
                        seen_fields.add(k)
                        _add_req_field(
                            bucket,
                            path=k[0],
                            access=k[1],
                            rhs=fa.get("rhs", ""),
                            evidence={
                                "file": file_path,
                                "line": start_line + int(fa.get("line", 0)),
                                "function": caller_name,
                            },
                            source_tag="direct_field_write",
                        )
                        _add_recipe_field(
                            recipe_bucket,
                            path=k[0],
                            access=k[1],
                            rhs=fa.get("rhs", ""),
                            evidence={
                                "file": file_path,
                                "line": start_line + int(fa.get("line", 0)),
                                "function": caller_name,
                            },
                            source_tag="direct_field_write",
                        )

                    for prev in call_sites:
                        if int(prev.get("line", 0)) >= cs_line:
                            continue
                        p_name = str(prev.get("name") or "").strip()
                        p_args = list(prev.get("args") or [])
                        for pj, pa in enumerate(p_args):
                            if _extract_arg_base_var(pa) != base_var:
                                continue
                            for he in helper_effect_lookup.get((p_name, int(pj)), []):
                                for wr in list(he.get("writes") or []):
                                    _add_req_field(
                                        bucket,
                                        path=str(wr.get("path", "")),
                                        access=str(wr.get("access", "dot")),
                                        rhs=str(wr.get("sample_expr", "")),
                                        evidence={
                                            "file": file_path,
                                            "line": start_line + int(prev.get("line", 0)),
                                            "function": caller_name,
                                        },
                                        source_tag=f"helper:{p_name}",
                                    )
                                    _add_recipe_field(
                                        recipe_bucket,
                                        path=str(wr.get("path", "")),
                                        access=str(wr.get("access", "dot")),
                                        rhs=str(wr.get("sample_expr", "")),
                                        evidence={
                                            "file": file_path,
                                            "line": start_line + int(prev.get("line", 0)),
                                            "function": caller_name,
                                        },
                                        source_tag=f"helper:{p_name}",
                                    )

    # Pass 3: If no/weak caller dataflow, backfill from callee arg-field reads.
    for fn, entries in function_effects_out.items():
        for entry in entries:
            reads = list(entry.get("reads") or [])
            if not reads:
                continue
            bucket = _ensure_req_bucket(
                fn,
                int(entry.get("arg_index", 0)),
                str(entry.get("arg_name", "")),
                str(entry.get("type", "")),
            )
            for r in reads[:24]:
                _add_req_field(
                    bucket,
                    path=str(r.get("path", "")),
                    access=str(r.get("access", "dot")),
                    rhs="",
                    evidence=(list(r.get("evidence") or [])[:1] or [{}])[0],
                    source_tag="callee_read",
                )

    # finalize required_fields_by_function
    required_fields_by_function = defaultdict(list)
    for _, bucket in req.items():
        call_sites = int(bucket.get("call_sites", 0))
        denom = max(1, call_sites)
        fields_out = []
        for _, frec in bucket["fields"].items():
            rhs_items = sorted(
                frec["rhs"].items(),
                key=lambda x: (_field_rhs_score(x[0]), x[1], len(str(x[0]))),
                reverse=True,
            )
            sample_expr = rhs_items[0][0] if rhs_items else ""

            if call_sites > 0:
                support = float(frec["count"]) / float(denom)
                required = bool(
                    (frec["count"] == denom and denom >= 1)
                    or (support >= 0.80 and frec["count"] >= 2)
                )
            else:
                # Backfilled from callee-read path: informative, but not "required".
                support = min(0.49, 0.12 * float(frec["count"]))
                required = False

            fields_out.append({
                "path": frec["path"],
                "access": frec["access"],
                "count": int(frec["count"]),
                "support": round(support, 4),
                "required": required,
                "sample_expr": sample_expr,
                "self_contained": bool(sample_expr and (_is_literal_like_token(sample_expr) or _is_self_contained_call_expr(sample_expr))),
                "sources": {k: int(v) for k, v in sorted(frec["sources"].items(), key=lambda x: (-x[1], x[0]))},
                "evidence": list(frec["evidence"]),
            })

        fields_out.sort(key=lambda x: (x["required"], x["support"], x["count"], len(x["path"])), reverse=True)
        if not fields_out:
            continue
        required_fields_by_function[bucket["function"]].append({
            "arg_index": int(bucket["arg_index"]),
            "arg_name": bucket["arg_name"],
            "type": bucket["type"],
            "call_sites": int(call_sites),
            "fields": fields_out[:24],
        })

    req_out = {}
    for fname, entries in required_fields_by_function.items():
        req_out[fname] = sorted(
            entries,
            key=lambda e: (int(e.get("arg_index", 0)), str(e.get("arg_name", ""))),
        )

    callsite_arg_flow_out = {}
    for caller, rows in callsite_arg_flow.items():
        callsite_arg_flow_out[caller] = rows[:128]

    init_recipes_by_type = defaultdict(list)
    for _, bucket in recipe_acc.items():
        fields = []
        nontrivial_field_count = 0
        for _, frec in bucket["fields"].items():
            rhs_items = sorted(
                frec["rhs"].items(),
                key=lambda x: (_recipe_rhs_score(x[0]), x[1], len(str(x[0]))),
                reverse=True,
            )
            sample_expr = rhs_items[0][0] if rhs_items else ""
            if sample_expr and _recipe_rhs_score(sample_expr) >= 4:
                nontrivial_field_count += 1
            fields.append({
                "path": frec["path"],
                "access": frec["access"],
                "count": int(frec["count"]),
                "support": round(float(frec["count"]) / max(1, int(bucket.get("call_sites", 0))), 4),
                "sample_expr": sample_expr,
                "sources": {k: int(v) for k, v in sorted(frec["sources"].items(), key=lambda x: (-x[1], x[0]))},
                "evidence": list(frec["evidence"]),
            })
        fields.sort(key=lambda x: (x.get("support", 0.0), x.get("count", 0), len(str(x.get("path", ""))), _recipe_rhs_score(x.get("sample_expr", ""))), reverse=True)

        var_assign_items = sorted(
            bucket["var_assign"].items(),
            key=lambda x: (_recipe_rhs_score(x[0]), x[1], len(str(x[0]))),
            reverse=True,
        )
        allocator_expr = ""
        for expr, _ in var_assign_items:
            if _is_self_contained_call_expr(expr):
                allocator_expr = expr
                break
        if not allocator_expr and var_assign_items:
            allocator_expr = var_assign_items[0][0]

        step_statements = []
        if allocator_expr:
            step_statements.append({
                "statement": f"$arg = {allocator_expr};",
                "kind": "value_assign",
            })
        for f in fields[:12]:
            access_op = "->" if str(f.get("access")) == "arrow" else "."
            rhs = str(f.get("sample_expr", "")).strip() or "{}"
            step_statements.append({
                "statement": f"$arg{access_op}{f.get('path', '')} = {rhs};",
                "kind": "field_write",
            })

        recipe_score = (
            int(bucket.get("call_sites", 0)) * 3
            + (4 if allocator_expr else 0)
            + min(6, nontrivial_field_count * 2)
            + min(6, len(fields))
        )

        row = {
            "target_function": bucket["target_function"],
            "arg_index": int(bucket["arg_index"]),
            "arg_name": bucket["arg_name"],
            "is_pointer": bool(bucket.get("is_pointer")),
            "call_sites": int(bucket.get("call_sites", 0)),
            "score": int(recipe_score),
            "allocator_expr": allocator_expr,
            "fields": fields[:16],
            "steps": step_statements[:16],
            "evidence": list(bucket.get("evidence", []))[:6],
        }
        if row["allocator_expr"] or row["fields"]:
            init_recipes_by_type[bucket["type"]].append(row)

    init_recipes_out = {}
    for nominal, rows in init_recipes_by_type.items():
        rows.sort(
            key=lambda r: (
                int(r.get("score", 0)),
                int(r.get("call_sites", 0)),
                bool(r.get("allocator_expr")),
                len(list(r.get("fields") or [])),
            ),
            reverse=True,
        )
        init_recipes_out[nominal] = rows[: max(8, max_per_type * 3)]

    return {
        "version": 4,
        "types": types_out,
        "struct_field_writes": struct_field_writes_out,
        "function_effects": function_effects_out,
        "callsite_arg_flow": callsite_arg_flow_out,
        "required_fields_by_function": req_out,
        "init_recipes_by_type": init_recipes_out,
    }
