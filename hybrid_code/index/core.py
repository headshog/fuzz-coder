#!/usr/bin/env python3
"""
Hybrid Code Indexer with Tree-sitter AST, Call Graph, and Semantic Search
"""

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import numpy as np
from collections import defaultdict
from pathlib import Path
import zipfile
import re
import json
import argparse
import os
import hashlib
from .call_graph import detect_calls as _detect_calls_impl
from .call_graph import build_call_graph as _build_call_graph_impl

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp
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

    def __init__(self):
        if not HAS_TREE_SITTER:
            self.parser = None
            return

        try:
            self.language = Language(tree_sitter_cpp.language())
            self.parser = Parser()
            self.parser.set_language(self.language)
        except Exception as e:
            print(f"Warning: Could not initialize tree-sitter: {e}")
            self.parser = None

    def parse_functions(self, filepath, code):
        """Extract functions with parameters using tree-sitter"""
        if not self.parser:
            return extract_functions_regex(filepath, code)

        functions = []
        try:
            source_bytes = code.encode("utf-8", errors="ignore")
            tree = self.parser.parse(source_bytes)
            root = tree.root_node

            # Query for function definitions
            query = self.language.query("""
                (function_definition
                    declarator: (function_declarator
                        declarator: (identifier) @name
                        parameters: (parameter_list) @params)) @func

                (function_definition
                    declarator: (qualified_identifier
                        name: (identifier) @name)
                    parameters: (parameter_list) @params) @func
            """)

            captures = query.captures(root)

            for node, capture_name in captures:
                if capture_name == "func":
                    func_node = node

                    # Find function name
                    name_node = None
                    params_node = None

                    for child in func_node.children:
                        if child.type == "function_declarator":
                            for gc in child.children:
                                if gc.type == "identifier":
                                    name_node = gc
                                elif gc.type == "parameter_list":
                                    params_node = gc
                        elif child.type == "qualified_identifier":
                            name_node = child.child_by_field_name("name")
                        elif child.type == "parameter_list":
                            params_node = child

                    if name_node is None:
                        continue

                    func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")

                    if func_name in CONTROL_KEYWORDS:
                        continue

                    # Extract parameters
                    params = []
                    if params_node:
                        params_text = source_bytes[params_node.start_byte:params_node.end_byte].decode("utf-8", errors="ignore")
                        params = self._parse_parameters(params_text)

                    # Get function body
                    body_node = func_node.child_by_field_name("body")
                    if not body_node:
                        continue

                    func_code = source_bytes[func_node.start_byte:func_node.end_byte].decode("utf-8", errors="ignore")

                    # Calculate line numbers
                    start_line = source_bytes[:func_node.start_byte].count(b"\n") + 1
                    end_line = source_bytes[:func_node.end_byte].count(b"\n") + 1

                    functions.append({
                        "name": func_name,
                        "signature": self._build_signature(func_name, params),
                        "parameters": params,
                        "code": func_code,
                        "body": func_code,
                        "start_line": start_line,
                        "end_line": end_line,
                        "parser": "tree-sitter"
                    })

        except Exception as e:
            print(f"Tree-sitter error for {filepath}: {e}")
            # Fallback to regex
            functions = extract_functions_regex(filepath, code)

        if not functions:
            functions = extract_functions_regex(filepath, code)

        return functions

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

    def _build_signature(self, name, params):
        """Build function signature string"""
        param_strs = [p["raw"] for p in params]
        return f"{name}({', '.join(param_strs)})"


def extract_functions_regex(filepath, text):
    """Fallback regex-based function extraction"""
    res = []
    lines = text.splitlines()

    offsets = []
    cur = 0
    for line in lines:
        offsets.append(cur)
        cur += len(line) + 1

    i = 0
    n = len(lines)

    while i < n:
        if "(" not in lines[i]:
            i += 1
            continue

        start_i = i
        sig_lines = []
        found_body = False
        j = i

        while j < n and j < i + 20:
            sig_lines.append(lines[j])
            joined = "\n".join(sig_lines)

            semi_pos = joined.find(";")
            brace_pos = joined.find("{")

            if brace_pos != -1 and (semi_pos == -1 or brace_pos < semi_pos):
                found_body = True
                break

            if semi_pos != -1:
                break

            j += 1

        if not found_body:
            i += 1
            continue

        joined = "\n".join(sig_lines)
        brace_pos = joined.find("{")
        signature = joined[:brace_pos].strip()

        if not signature or ")" not in signature:
            i += 1
            continue

        compact = " ".join(signature.split())

        bad_prefixes = ("if ", "for ", "while ", "switch ",
                        "catch ", "#", "typedef ", "return ", "class ", "struct ")
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
        if name in CONTROL_KEYWORDS:
            i += 1
            continue

        # Extract parameters
        rp = compact.find(")", lp)
        if rp != -1:
            params_text = compact[lp+1:rp]
            params = parse_parameters_simple(params_text)
        else:
            params = []

        start_offset = offsets[start_i]
        open_brace = text.find("{", start_offset)
        if open_brace == -1:
            i += 1
            continue

        end = find_matching_brace(text, open_brace)
        if end == -1:
            i += 1
            continue

        code = text[start_offset:end + 1]
        res.append({
            "name": name,
            "signature": compact,
            "parameters": params,
            "code": code,
            "body": code,
            "start_line": start_i + 1,
            "end_line": text.count("\n", 0, end) + 1,
            "parser": "regex"
        })
        i = text.count("\n", 0, end) + 1

    return res


def parse_parameters_simple(params_text):
    """Simple parameter parsing for regex fallback"""
    params = []
    params_text = params_text.strip()
    if not params_text:
        return params

    # Split by comma
    for param in params_text.split(","):
        param = param.strip()
        if not param:
            continue

        parts = param.split()
        if len(parts) >= 2:
            name = parts[-1].split("&")[-1].split("*")[-1].strip()
            param_type = " ".join(parts[:-1])
            params.append({
                "name": name,
                "type": param_type,
                "is_reference": "&" in param,
                "is_pointer": "*" in param,
                "is_const": "const" in param.lower(),
                "raw": param
            })
        else:
            params.append({
                "name": param,
                "type": "unknown",
                "is_reference": False,
                "is_pointer": False,
                "is_const": False,
                "raw": param
            })

    return params


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


def detect_calls(body):
    """Detect function calls in code body"""
    return _detect_calls_impl(body, CONTROL_KEYWORDS)


def build_call_graph(chunks):
    """Build a call graph from function chunks"""
    return _build_call_graph_impl(chunks, CONTROL_KEYWORDS)


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
