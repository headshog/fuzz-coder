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
from fuzz_coder.languages.registry import (
    get_language_frontend,
    get_language_profile,
    get_index_language_adapter,
    infer_index_language_from_path,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

ACTIVE_INDEX_LANGUAGE = "c_cpp"
_DEFAULT_PROFILE = get_language_profile(ACTIVE_INDEX_LANGUAGE)

SUPPORTED_EXT = set(_DEFAULT_PROFILE.supported_ext)
CONTROL_KEYWORDS = set(_DEFAULT_PROFILE.control_keywords)
STDIN_PATTERNS = list(_DEFAULT_PROFILE.stdin_patterns)
FILE_INPUT_PATTERNS = list(_DEFAULT_PROFILE.file_input_patterns)
API_CALL_PATTERNS = list(_DEFAULT_PROFILE.api_call_patterns)
OUTPUT_PATTERNS = list(_DEFAULT_PROFILE.output_patterns)
MEMORY_MANAGEMENT_PATTERNS = list(_DEFAULT_PROFILE.memory_management_patterns)
ERROR_HANDLING_PATTERNS = list(_DEFAULT_PROFILE.error_handling_patterns)
TYPE_PATTERNS = dict(_DEFAULT_PROFILE.type_patterns)


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
            profile = get_language_profile(self.language_name)
            functions = self.frontend.parse_tree_sitter_functions(
                parser_utils=self,
                source_bytes=source_bytes,
                root=root,
                control_keywords=set(profile.control_keywords),
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
        return parse_parameters_simple(params_text, language_name=self.language_name)

    def _iter_nodes_by_type(self, root_node, node_type):
        """Yield nodes of a specific type via DFS traversal."""
        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.type == node_type:
                yield node
            # Reverse children to keep left-to-right traversal order.
            stack.extend(reversed(node.children))

    def _build_signature(self, name, params):
        """Build function signature string"""
        param_strs = [p["raw"] for p in params]
        return f"{name}({', '.join(param_strs)})"


def extract_functions_regex(_filepath, text, language_name="c_cpp"):
    """Fallback regex-based function extraction"""
    res = []
    frontend = get_language_frontend(language_name)
    adapter = get_index_language_adapter(language_name)
    profile = get_language_profile(language_name)
    control_keywords = set(profile.control_keywords)
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

            brace_pos, terminated_decl = adapter.scan_signature_for_body(joined)

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
        if name in control_keywords:
            i += 1
            continue

        # Extract parameters
        rp = adapter.find_matching_paren(compact, lp)
        if rp != -1:
            params_text = compact[lp+1:rp]
            params = adapter.parse_parameters_simple(params_text)
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


def scan_signature_for_body(signature_text, language_name=None):
    lang = language_name or "c_cpp"
    adapter = get_index_language_adapter(lang)
    return adapter.scan_signature_for_body(signature_text or "")


def parse_parameters_simple(params_text, language_name=None):
    lang = language_name or "c_cpp"
    adapter = get_index_language_adapter(lang)
    return adapter.parse_parameters_simple(params_text or "")


def split_top_level_params(params_text, language_name=None):
    lang = language_name or "c_cpp"
    adapter = get_index_language_adapter(lang)
    return adapter.split_top_level_params(params_text or "")


def parse_single_parameter_simple(param_text, language_name=None):
    lang = language_name or "c_cpp"
    adapter = get_index_language_adapter(lang)
    return adapter.parse_single_parameter_simple(param_text or "")


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


def find_matching_paren(text, pos, language_name=None):
    lang = language_name or "c_cpp"
    adapter = get_index_language_adapter(lang)
    return adapter.find_matching_paren(text or "", pos)


def detect_calls(body, language_name="c_cpp"):
    """Detect function calls in code body."""
    profile = get_language_profile(language_name)
    return _detect_calls_impl(body, set(profile.control_keywords), language_name=language_name)


def build_call_graph(chunks, language_name="c_cpp"):
    """Build a call graph from function chunks."""
    profile = get_language_profile(language_name)
    return _build_call_graph_impl(chunks, set(profile.control_keywords), language_name=language_name)


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


def _split_call_args_top_level(args_text, language_name=None):
    args_text = str(args_text or "").strip()
    if not args_text:
        return []
    return split_top_level_params(args_text, language_name=language_name)


def _is_literal_like_token(token, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.is_literal_like_token(token)


def _is_self_contained_call_expr(expr, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.is_self_contained_call_expr(expr)


def _extract_nominal_type_name_for_init(type_text, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_nominal_type_name_for_init(type_text)


def _normalize_expr(expr, max_len=240):
    e = " ".join(str(expr or "").split()).strip()
    if len(e) <= max_len:
        return e
    return e[: max_len - 3].rstrip() + "..."


def _extract_call_name(expr, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_call_name(expr)


def _init_pattern_score(kind, expr, language_name=None):
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
    if _is_self_contained_call_expr(expr, language_name=language_name):
        score += 3
    return score


def _extract_variable_type_hints(code, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_variable_type_hints(code or "")


def _extract_init_patterns(code, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_init_patterns(code or "")


def _extract_field_assignments(code, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_field_assignments(code or "")


def _extract_field_reads(code, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_field_reads(code or "")


def _extract_call_sites(code, control_keywords, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_call_sites(code or "", control_keywords or [])


def _extract_simple_identifiers(expr):
    e = str(expr or "").strip()
    if not e:
        return []
    return re.findall(r"[A-Za-z_]\w*", e)


def _extract_dependency_chain(code, base_var, before_line, max_steps=8, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_dependency_chain(code or "", base_var, before_line, max_steps=max_steps)


def _extract_arg_base_var(arg_expr, language_name=None):
    lang = language_name or ACTIVE_INDEX_LANGUAGE
    adapter = get_index_language_adapter(lang)
    return adapter.extract_arg_base_var(arg_expr)


def _field_rhs_score(expr, language_name=None):
    e = _normalize_expr(expr)
    if _is_literal_like_token(e, language_name=language_name):
        return 6
    if _is_self_contained_call_expr(e, language_name=language_name):
        return 5
    return 1


def _recipe_rhs_score(expr, language_name=None):
    e = _normalize_expr(expr)
    if _is_self_contained_call_expr(e, language_name=language_name):
        return 8
    if _is_literal_like_token(e, language_name=language_name):
        return 6
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", e):
        return 4
    return 1


def build_type_init_index(chunks, max_per_type=16, language_name=None):
    """Build v3 structural dataflow index for better example grounding."""
    default_language = language_name or ACTIVE_INDEX_LANGUAGE
    by_type = defaultdict(dict)  # type -> (kind, expr) -> aggregated record
    by_name = defaultdict(list)
    for c in chunks or []:
        name = str(c.get("name", "")).strip()
        if name:
            by_name[name].append(c)

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
        chunk_language = infer_index_language_from_path(file_path, default=default_language)
        chunk_profile = get_language_profile(chunk_language)
        chunk_control_keywords = set(chunk_profile.control_keywords)
        fn_name = str(c.get("name", "")).strip()
        signature = str(c.get("signature", "")).strip() or (f"{fn_name}()")
        start_line = int(c.get("start_line", 1) or 1)
        code = str(c.get("code", "") or c.get("body", ""))
        if not code:
            continue

        # v3/A: language-adapter init patterns per nominal type.
        for init_item in _extract_init_patterns(code, language_name=chunk_language):
            kind = str((init_item or {}).get("kind") or "").strip()
            type_text = str((init_item or {}).get("type_text") or "").strip()
            expr = str((init_item or {}).get("expr") or "").strip()
            li = int((init_item or {}).get("line") or 0)
            if not kind or not type_text or li <= 0:
                continue
            if kind in {"pointer_call", "value_call"}:
                if not _extract_call_name(expr or "", language_name=chunk_language):
                    continue
            nominal = _extract_nominal_type_name_for_init(type_text, language_name=chunk_language)
            if not nominal:
                continue

            expr_norm = _normalize_expr(expr)
            key = (kind, expr_norm)
            rec = by_type[nominal].get(key)
            if rec is None:
                rec = {
                    "kind": kind,
                    "expr": expr_norm,
                    "function": _extract_call_name(expr_norm, language_name=chunk_language) if kind in {"pointer_call", "value_call"} else None,
                    "self_contained": _is_self_contained_call_expr(expr_norm, language_name=chunk_language) if kind in {"pointer_call", "value_call"} else (kind != "value_default"),
                    "count": 0,
                    "score": 0,
                    "evidence": [],
                }
                by_type[nominal][key] = rec
            rec["count"] += 1
            rec["score"] += _init_pattern_score(kind, expr_norm, language_name=chunk_language)
            if len(rec["evidence"]) < 3:
                rec["evidence"].append({
                    "file": file_path,
                    "line": start_line + li - 1,
                    "function": fn_name,
                })

        var_hints = _extract_variable_type_hints(code, language_name=chunk_language)
        field_writes = _extract_field_assignments(code, language_name=chunk_language)
        field_reads = _extract_field_reads(code, language_name=chunk_language)
        call_sites = _extract_call_sites(code, chunk_control_keywords, language_name=chunk_language)

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
            ptype_nominal = _extract_nominal_type_name_for_init(param.get("type", ""), language_name=chunk_language)
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
                    if _extract_arg_base_var(a, language_name=chunk_language) != pname:
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
        chunk_language = infer_index_language_from_path(file_path, default=default_language)
        chunk_profile = get_language_profile(chunk_language)
        chunk_control_keywords = set(chunk_profile.control_keywords)
        caller_name = str(c.get("name", "")).strip()
        start_line = int(c.get("start_line", 1) or 1)
        var_hints = _extract_variable_type_hints(code, language_name=chunk_language)
        field_assignments = _extract_field_assignments(code, language_name=chunk_language)
        call_sites = _extract_call_sites(code, chunk_control_keywords, language_name=chunk_language)
        if not call_sites:
            continue

        for cs in call_sites:
            cs_name = str(cs.get("name") or "").strip()
            cs_line = int(cs.get("line") or 0)
            cs_args = list(cs.get("args") or [])

            arg_flow_rows = []
            for ai, arg in enumerate(cs_args):
                base_var = _extract_arg_base_var(arg, language_name=chunk_language)
                arg_row = {
                    "arg_index": int(ai),
                    "arg_expr": str(arg),
                    "base_var": base_var,
                }
                if base_var:
                    hint = var_hints.get(base_var)
                    if hint and str(hint.get("nominal_type", "")).strip():
                        arg_row["nominal_type"] = str(hint.get("nominal_type", "")).strip()

                    chain = _extract_dependency_chain(
                        code,
                        base_var,
                        before_line=cs_line,
                        max_steps=8,
                        language_name=chunk_language,
                    )
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
                            if _extract_arg_base_var(pa, language_name=chunk_language) != base_var:
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
                    base_var = _extract_arg_base_var(arg, language_name=chunk_language)
                    if not base_var:
                        continue
                    hint = var_hints.get(base_var)
                    if not hint:
                        continue
                    nominal = _extract_nominal_type_name_for_init(
                        param.get("type", ""),
                        language_name=chunk_language,
                    )
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

                    chain = _extract_dependency_chain(
                        code,
                        base_var,
                        before_line=cs_line,
                        max_steps=8,
                        language_name=chunk_language,
                    )
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
                            if _extract_arg_base_var(pa, language_name=chunk_language) != base_var:
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
