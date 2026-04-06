#!/usr/bin/env python3
"""
Advanced Code Query System with Thinking Mode, Context Memory, and Multi-Step Reasoning
Supports follow-up questions, code generation examples, and deep codebase understanding
"""

import numpy as np
import re
import json
import os
from collections import defaultdict
from .llm import call_llm as _call_llm_impl
from .prompting import build_prompt as _build_prompt_impl
from .prompting import _build_thinking_prompt_with_limit as _build_thinking_prompt_impl
from .query_analysis import analyze_query_v2 as _analyze_query_v2_impl
from .query_analysis import compare_analysis as _compare_analysis_impl
from .query_analysis import extract_max_param_count as _extract_max_param_count_impl
from .query_analysis import extract_explicit_function_mentions as _extract_explicit_function_mentions_impl
from .example_context import build_example_context as _build_example_context_impl
from .example_context import _classify_param_shape as _classify_param_shape_impl
from .example_grounding import build_example_context_grounded as _build_example_context_grounded_impl
from .verification import verify_answer_with_context as _verify_answer_with_context_impl
from .verification import verify_example_answer_with_context as _verify_example_answer_with_context_impl

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

OLLAMA_URL = "http://localhost:11434/api/generate"

TYPE_KEYWORDS = {
    "byte_array": ["byte", "uint8", "char*", "buffer", "массив байт", "байт", "bytes", "qbytearray"],
    "string": ["string", "str", "char[]", "строка", "строку", "cstring", "wstring"],
    "integer": ["int", "integer", "число", "int32", "int64", "цел", "size_t", "ssize_t"],
    "float": ["float", "double", "веществен", "floating", "плавающ"],
    "template": ["vector", "array", "список", "массив", "std::vector", "template", "std::array"],
    "pointer": ["pointer", "указатель"],
    "reference": ["reference", "ссылка"],
}

# Canonical names used by index_fuzz_coder.py -> special_indices["by_type"]
TYPE_ALIASES = {
    "int": "integer",
    "vector": "template",
}

TYPE_INDEX_ALIASES = {
    "integer": ["int"],
    "template": ["vector"],
}

WRITE_LIKE_KEYWORDS = [
    "write", "output", "print", "printf", "fprintf", "cout", "cerr", "clog",
    "log", "dump", "serialize", "save", "emit", "flush", "store",
    "запис", "вывод", "печат", "лог",
]

PARSE_LIKE_KEYWORDS = [
    "parse", "parser", "token", "tokenize", "split", "decode", "deserialize",
    "scan", "lex", "grammar", "peg", "readline", "from_string", "parse_",
    "парс", "разбор",
]

FUZZ_QUERY_KEYWORDS = [
    "fuzz", "fuzzer", "fuzzing", "libfuzzer", "afl", "afl++", "honggfuzz",
    "oss-fuzz", "mutation", "coverage-guided", "asan", "ubsan", "sanitizer",
    "фазз", "фаззинг", "фузз", "фуззинг",
]

FUZZ_TARGET_KEYWORDS = [
    "parse", "decode", "deserialize", "token", "grammar", "load", "read",
    "json", "xml", "yaml", "gguf", "tensor", "prompt", "chat", "template",
    "sample", "kv", "buffer", "memcpy", "memmove", "strncpy", "snprintf",
    "base64", "utf8", "utf-8", "binary", "header", "payload",
]

SIMPLE_POINTER_TYPE_HINTS = [
    "char", "signed char", "unsigned char",
    "short", "unsigned short",
    "int", "unsigned int",
    "long", "unsigned long",
    "long long", "unsigned long long",
    "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t",
    "size_t", "ssize_t",
    "float", "double", "bool",
]

FILE_HANDLE_TYPE_HINTS = [
    "file *", "file*", "std::file",
    "ifstream", "ofstream", "fstream",
    "istream", "ostream", "stream &", "stream&",
]

PATH_PARAM_NAME_HINTS = [
    "path", "file", "filename", "fname", "filepath", "dir", "directory",
]

VECTOR_LIKE_TYPE_HINTS = [
    "std::vector<", "vector<",
    "std::array<", "array<",
    "std::span<", "span<",
]

PATH_FILTER_PATTERNS = [
    # English: from/in module|directory|subdirectory|folder|path <value>
    r"\b(?:from|in)\s+(?:the\s+)?(?:module|directory|subdirectory|folder|path)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1",
    # English: from/in <value> module|directory|...
    r"\b(?:from|in)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1\s+(?:module|directory|subdirectory|folder|path)\b",
    r"\bunder\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1",
    # Russian: из/в модуля|директории|поддиректории|папки <value>
    r"\bиз\s+(?:модуля|директории|поддиректории|папки)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1",
    r"\bв\s+(?:модуле|директории|поддиректории|папке)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1",
    # Russian: из/в <value> модуле|директории|...
    r"\b(?:из|в)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1\s+(?:модуле|модуля|директории|поддиректории|папке)\b",
]

COMMON_QUERY_WORDS = {
    "write", "list", "of", "functions", "that", "can", "be", "used", "for", "fuzzing",
    "give", "an", "example", "from", "main", "function", "called", "call", "how",
    "to", "is", "in", "codebase", "show", "me", "the", "a", "and", "or", "with",
}

# Words that should not become function targets when seen as plain tokens,
# even if such symbols exist somewhere in the index.
QUERY_SYMBOL_BLACKLIST = {
    "write", "list", "show", "find", "give", "make", "need", "example",
    "function", "functions", "called", "calling", "from", "for", "with",
    "that", "this", "these", "those", "can", "used", "use", "is", "are",
    "of", "in", "on", "to", "an", "a", "the",
    "construct", "constructed", "build", "built", "define", "generated", "snippet",
    "format", "context", "range", "role",
}


def query_contains_keyword(query_lower, keyword):
    """Match keywords safely; short alpha keywords use token boundaries."""
    if len(keyword) <= 3 and keyword.isalpha():
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(keyword)}(?![A-Za-z0-9_])"
        return re.search(pattern, query_lower) is not None
    return keyword in query_lower


def query_has_any_keyword(query_lower, keywords):
    """Check if query contains any keyword using safe matching rules."""
    return any(query_contains_keyword(query_lower, kw) for kw in keywords)


def normalize_path_filter(path):
    """Normalize user-provided path/module filters for robust matching."""
    p = (path or "").strip().strip("`'\"")
    p = p.replace("\\", "/")
    p = re.sub(r"/{2,}", "/", p)
    while p.startswith("./"):
        p = p[2:]
    p = p.strip().rstrip("/")
    return p.lower()


def clean_path_candidate(raw):
    """Trim natural-language tails captured by broad path patterns."""
    p = (raw or "").strip()
    if not p:
        return p

    # Stop at common relative clauses/conjunctions after path mention.
    p = re.split(
        r"\s+(?:that|which|where|with|and|or|who|whose|function|functions|method|methods|котор(?:ый|ая|ые|ого|ому|ых)?|где|и|или|с|функц(?:ия|ии|ий|ию|иями)?)\b",
        p,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    return p.strip(" \t\r\n.,:;!?")


def extract_path_filters_from_query(query):
    """Extract module/directory constraints from query."""
    filters = []
    invalid_filters = {"main", "main()", "function", "functions", "module", "directory"}

    for pattern in PATH_FILTER_PATTERNS:
        for m in re.finditer(pattern, query, flags=re.IGNORECASE):
            raw = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
            norm = normalize_path_filter(clean_path_candidate(raw))
            if norm and norm not in invalid_filters:
                filters.append(norm)

    # Additional explicit path-like hints in backticks or quotes.
    for p in re.findall(r"`([^`]+[/\\][^`]+)`", query):
        norm = normalize_path_filter(p)
        if norm and norm not in invalid_filters:
            filters.append(norm)
    for p in re.findall(r"['\"]([^'\"]+[/\\][^'\"]+)['\"]", query):
        norm = normalize_path_filter(p)
        if norm and norm not in invalid_filters:
            filters.append(norm)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for f in filters:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def chunk_matches_path_filters(chunk, path_filters):
    """Check whether chunk file path satisfies at least one path filter."""
    if not path_filters:
        return True
    file_path = normalize_path_filter(str(chunk.get("file", "")))
    return any(f in file_path for f in path_filters)


def is_valid_function_chunk(chunk):
    """Drop obvious parser artifacts (macros/comment blobs) from retrieval/output."""
    name = str(chunk.get("name", "")).strip()
    if not name:
        return False

    # Macro-like symbols (e.g. DEPRECATED) are not callable functions.
    if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
        return False

    sig = str(chunk.get("signature", "")).strip()
    if not sig:
        return False

    if name not in sig:
        return False

    # A definition signature should not contain declaration separators or comment blobs.
    if ";" in sig:
        return False
    if any(tok in sig for tok in ["///", "/*", "*/", "DEPRECATED("]):
        return False
    if "{" in sig or "}" in sig:
        return False

    # Long signatures are usually malformed merged blocks from regex fallback.
    if len(sig) > 500:
        return False

    return True


def extract_function_names_from_text(text, symbols):
    """Extract known function names mentioned in free text."""
    if not text or not symbols:
        return set()

    found = set()
    for fn in symbols.keys():
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(fn)}(?![A-Za-z0-9_])", text):
            found.add(fn)
    return found


def extract_function_like_candidates(query, known_symbols_by_lower=None):
    """Extract function-like identifiers from user query.

    Keeps explicit symbol-like tokens and also plain identifiers that are
    present in the known symbol table (e.g. "split", "main").
    """
    candidates = []
    known_symbols_by_lower = known_symbols_by_lower or {}

    # Prefer explicit code-style references.
    for t in re.findall(r"`([^`]+)`", query):
        candidates.append((t, "explicit"))
    for t in re.findall(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*(?=\()", query):
        candidates.append((t, "call_like"))

    # Function-focused phrase patterns (captures plain names like "split" or "main").
    phrase_patterns = [
        r"\b(?:of|for|from|in|using|use|invoke|invoking|calling|call)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+(?:function|method)\b",
        r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+(?:function|method)\b",
    ]
    for pat in phrase_patterns:
        for t in re.findall(pat, query, flags=re.IGNORECASE):
            candidates.append((t, "phrase"))

    # Fallback: generic identifier scan.
    for t in re.findall(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\b", query):
        candidates.append((t, "token"))

    out = []
    seen = set()
    for raw, source in candidates:
        token = (raw or "").strip().strip("`'\".,:;!?()[]{}")
        if not token:
            continue

        token_lower = token.lower()
        known_symbol_match = token_lower in known_symbols_by_lower

        # Keep identifiers that look like actual symbols, avoid plain prose.
        looks_like_symbol = (
            "_" in token or
            "::" in token or
            (any(ch.isupper() for ch in token[1:]) and any(ch.islower() for ch in token))
        )

        if source == "token":
            if token_lower in QUERY_SYMBOL_BLACKLIST:
                continue
            if token_lower in COMMON_QUERY_WORDS and token_lower != "main":
                continue
            if len(token) < 3 and not known_symbol_match:
                continue
            if not (looks_like_symbol or known_symbol_match):
                continue
        else:
            # For explicit/call-like/phrase sources keep strong mentions,
            # but still drop obvious language words.
            if token_lower in QUERY_SYMBOL_BLACKLIST and token_lower != "main":
                continue

        if token_lower not in seen:
            seen.add(token_lower)
            out.append(token)

    return out


def choose_primary_example_function(query, resolved_function_names):
    """Pick primary target function for example-generation queries.

    Heuristics:
    - Prefer symbol after calling/invoke/use/example-of phrases.
    - Treat "from <fn> function" as context function, not primary target.
    - If ambiguous and `main` is present with others, prefer non-main.
    """
    if not resolved_function_names:
        return None
    if len(resolved_function_names) == 1:
        return resolved_function_names[0]

    q = query or ""
    patterns = [
        r"\b(?:calling|call|invoke|invoking|using|use)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\b",
        r"\b(?:example|пример)\s+(?:of\s+)?([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\b",
        r"\b(?:пример)\s+(?:вызова|использования)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\b",
    ]
    lowered_map = {fn.lower(): fn for fn in resolved_function_names}
    for pat in patterns:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            cand = (m.group(1) or "").strip().lower()
            if cand in lowered_map:
                return lowered_map[cand]

    helper_context = set()
    for m in re.finditer(
        r"\bfrom\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+function\b",
        q,
        flags=re.IGNORECASE,
    ):
        helper_context.add((m.group(1) or "").strip().lower())

    if helper_context:
        for fn in resolved_function_names:
            if fn.lower() not in helper_context:
                return fn

    if any(fn.lower() == "main" for fn in resolved_function_names):
        for fn in resolved_function_names:
            if fn.lower() != "main":
                return fn

    return resolved_function_names[0]


def normalize_type_name(type_name):
    """Normalize type names to canonical keys."""
    t = type_name.lower().strip()
    return TYPE_ALIASES.get(t, t)


def extract_requested_types(query_lower):
    """Extract canonical requested types from query."""
    matched = []
    for type_name, keywords in TYPE_KEYWORDS.items():
        if any(query_contains_keyword(query_lower, kw) for kw in keywords):
            matched.append(type_name)

    # Pointer/reference are special symbols and may be requested explicitly
    if "*" in query_lower and "pointer" not in matched:
        matched.append("pointer")
    if "&" in query_lower and "reference" not in matched:
        matched.append("reference")

    return sorted(set(normalize_type_name(t) for t in matched))


def query_excludes_output(query_lower):
    """Detect negative output/write constraints like 'not write-like'."""
    explicit_phrases = [
        "not write", "not write-like", "but not write", "without write",
        "exclude write", "except write", "not output", "exclude output",
        "without output", "не запис", "не вывод", "без вывода", "кроме write",
        "кроме вывода", "excluding write",
    ]
    if any(p in query_lower for p in explicit_phrases):
        return True

    # Pattern-based negative constraints close to output keywords
    neg_en = r"\b(not|without|exclude|except|excluding)\b[^.\n]{0,60}\b(write|output|print|printf|fprintf|cout|log|dump|serialize)\b"
    neg_ru = r"\b(не|без|кроме|исключая)\b[^.\n]{0,60}\b(запис\w*|вывод\w*|печат\w*|лог\w*)\b"
    return re.search(neg_en, query_lower) is not None or re.search(neg_ru, query_lower) is not None


def parameter_matches_type(param, type_name):
    """Check if parameter likely matches a canonical type."""
    p_raw = f"{param.get('raw', '')} {param.get('type', '')}".lower()
    type_name = normalize_type_name(type_name)

    if type_name == "pointer":
        return "*" in p_raw
    if type_name == "reference":
        return "&" in p_raw

    type_param_hints = {
        "byte_array": ["uint8", "unsigned char", "char *", "byte", "qbytearray", "vector<uint8"],
        "string": ["std::string", "string", "char *", "const char *", "qstring", "wstring", "char[]"],
        "integer": ["int", "long", "short", "int32", "int64", "size_t", "ssize_t", "uint32", "uint64"],
        "float": ["float", "double", "long double"],
        "template": ["std::vector", "vector<", "std::array", "array<", "std::map", "map<", "std::set", "set<", "unordered_map"],
    }
    return any(h in p_raw for h in type_param_hints.get(type_name, []))


def chunk_matches_requested_types(chunk, requested_types):
    """Return True if chunk has at least one parameter matching any requested type."""
    if not requested_types:
        return True
    if not chunk.get("parameters"):
        return False

    canonical = [normalize_type_name(t) for t in requested_types]
    for p in chunk["parameters"]:
        for t in canonical:
            if parameter_matches_type(p, t):
                return True
    return False


def _param_text(param):
    return f"{param.get('raw', '')} {param.get('type', '')} {param.get('name', '')}".lower()


def has_simple_pointer_param(chunk):
    for p in chunk.get("parameters") or []:
        txt = _param_text(p)
        if "*" not in txt:
            continue
        # Skip function-pointer style parameters.
        if "(*" in txt:
            continue
        if any(hint in txt for hint in SIMPLE_POINTER_TYPE_HINTS):
            return True
    return False


def has_file_handle_param(chunk):
    for p in chunk.get("parameters") or []:
        txt = _param_text(p)
        if any(hint in txt for hint in FILE_HANDLE_TYPE_HINTS):
            return True
    return False


def has_path_like_param(chunk):
    for p in chunk.get("parameters") or []:
        p_name = str(p.get("name", "")).lower()
        txt = _param_text(p)
        has_name_hint = any(h in p_name for h in PATH_PARAM_NAME_HINTS)
        has_path_type = any(t in txt for t in ["std::string", "string", "char *", "const char *", "filesystem::path", "path"])
        if has_name_hint and has_path_type:
            return True
    return False


def has_vector_like_param(chunk):
    for p in chunk.get("parameters") or []:
        txt = _param_text(p)
        if any(h in txt for h in VECTOR_LIKE_TYPE_HINTS):
            return True
    return False


def chunk_matches_broad_fuzz_surface(chunk):
    """OR-surface for broad fuzz listing queries."""
    if is_parse_like_chunk(chunk):
        return True
    if chunk.get("has_stdin") or chunk.get("has_file_input"):
        return True
    if has_path_like_param(chunk):
        return True
    if has_file_handle_param(chunk):
        return True
    if has_simple_pointer_param(chunk):
        return True
    if has_vector_like_param(chunk):
        return True
    return False


def is_write_like_chunk(chunk):
    """Heuristic detector for write/output-like functions."""
    if chunk.get("has_output"):
        return True

    fields = " ".join([
        chunk.get("name", ""),
        chunk.get("signature", ""),
        chunk.get("code", "")[:1200],
    ]).lower()

    return any(kw in fields for kw in WRITE_LIKE_KEYWORDS)


def is_parse_like_chunk(chunk):
    """Heuristic detector for parse/input-processing functions."""
    name = chunk.get("name", "").lower()
    sig = chunk.get("signature", "").lower()
    code = chunk.get("code", "")[:1200].lower()
    fields = " ".join([name, sig, code])
    return any(kw in fields for kw in PARSE_LIKE_KEYWORDS)


def fuzz_target_score(chunk):
    """Estimate how suitable a function is as a fuzzing target (0.0-1.0)."""
    score = 0.0

    if chunk.get("has_stdin"):
        score += 0.20
    if chunk.get("has_file_input"):
        score += 0.20
    if chunk.get("has_api_call"):
        score += 0.10
    if chunk.get("uses_memory_management"):
        score += 0.15
    if chunk.get("has_error_handling"):
        score += 0.05
    if is_parse_like_chunk(chunk):
        score += 0.25

    params = chunk.get("parameters") or []
    if params:
        score += min(0.20, 0.04 * len(params))

    for p in params:
        p_text = f"{p.get('raw', '')} {p.get('type', '')} {p.get('name', '')}".lower()
        if "*" in p_text or "&" in p_text:
            score += 0.05
        if any(token in p_text for token in [
            "char", "string", "buffer", "data", "byte", "uint8",
            "vector<", "array<", "span", "size_t", "int", "len", "length"
        ]):
            score += 0.04

    fields = " ".join([
        chunk.get("name", ""),
        chunk.get("signature", ""),
        chunk.get("code", "")[:1600],
    ]).lower()
    if any(kw in fields for kw in FUZZ_TARGET_KEYWORDS):
        score += 0.18

    # Deprioritize pure output sinks when they do not parse/read/process inputs.
    if is_write_like_chunk(chunk) and not (
        chunk.get("has_stdin") or
        chunk.get("has_file_input") or
        chunk.get("has_api_call") or
        is_parse_like_chunk(chunk)
    ):
        score -= 0.10

    if chunk.get("name", "").lower() in {"main"}:
        score -= 0.10

    return max(0.0, min(1.0, score))


def collect_positive_constraint_matches(chunk, analysis):
    """Collect positive input/parse matches for flexible all/any filtering."""
    matches = {}
    if analysis.get("needs_stdin"):
        matches["stdin"] = bool(chunk.get("has_stdin"))
    if analysis.get("needs_file"):
        matches["file"] = bool(chunk.get("has_file_input"))
    if analysis.get("needs_api"):
        matches["api"] = bool(chunk.get("has_api_call"))
    if analysis.get("needs_output"):
        matches["output"] = bool(chunk.get("has_output"))
    if analysis.get("needs_memory_mgmt"):
        matches["memory_mgmt"] = bool(chunk.get("uses_memory_management"))
    if analysis.get("needs_error_handling"):
        matches["error_handling"] = bool(chunk.get("has_error_handling"))
    if analysis.get("requested_types"):
        matches["types"] = chunk_matches_requested_types(chunk, analysis["requested_types"])
    if analysis.get("needs_parse_like"):
        matches["parse_like"] = is_parse_like_chunk(chunk)
    if analysis.get("needs_broad_fuzz_surface"):
        matches["broad_fuzz_surface"] = chunk_matches_broad_fuzz_surface(chunk)
    if analysis.get("needs_fuzz_targets"):
        min_fuzz_score = analysis.get("min_fuzz_score", 0.35)
        matches["fuzz_target"] = fuzz_target_score(chunk) >= min_fuzz_score
    return matches


def chunk_matches_constraints(chunk, analysis):
    """Hard constraints matcher used for strict listing pre-filter."""
    if not is_valid_function_chunk(chunk):
        return False
    if analysis.get("needs_params") and not chunk.get("parameters"):
        return False
    max_params = analysis.get("max_param_count")
    if isinstance(max_params, int) and max_params >= 0:
        if len(chunk.get("parameters") or []) > max_params:
            return False
    if analysis.get("exclude_output") and is_write_like_chunk(chunk):
        return False
    if analysis.get("path_filters") and not chunk_matches_path_filters(chunk, analysis["path_filters"]):
        return False

    positive_matches = collect_positive_constraint_matches(chunk, analysis)
    if positive_matches:
        mode = analysis.get("constraint_mode", "all")
        if mode == "any":
            if not any(positive_matches.values()):
                return False
        else:
            if not all(positive_matches.values()):
                return False

    return True


def prefilter_listing_candidates(candidate_ids, meta, analysis):
    """Strict pre-filter before LLM for listing-like queries."""
    out = []
    for cid in candidate_ids:
        chunk = meta[cid]
        if not chunk_matches_constraints(chunk, analysis):
            continue
        if analysis.get("needs_fuzz_targets"):
            min_fuzz_score = analysis.get("min_fuzz_score", 0.35)
            if fuzz_target_score(chunk) < min_fuzz_score:
                continue
        out.append(cid)

    if out or not analysis.get("needs_fuzz_targets"):
        return out

    # Fallback for broad fuzzing queries: keep best available candidates
    # instead of returning an empty set too often.
    relaxed = []
    for cid in candidate_ids:
        chunk = meta[cid]
        if chunk_matches_constraints(chunk, analysis):
            relaxed.append((cid, fuzz_target_score(chunk)))

    relaxed.sort(key=lambda x: x[1], reverse=True)
    min_fallback = analysis.get("min_fallback_fuzz_score", 0.15)
    return [cid for cid, s in relaxed if s >= min_fallback]


def reciprocal_rank_fusion(rank_lists, rrf_k=60):
    """Fuse multiple ranked lists using RRF."""
    scores = defaultdict(float)
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[int(doc_id)] += 1.0 / (rrf_k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked], scores


def load_meta(path):
    """Load metadata from index"""
    meta = []
    with open(path/"meta.jsonl") as f:
        for l in f:
            meta.append(json.loads(l))
    return meta


def load_json(path):
    """Load JSON file if exists"""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def embed(queries, model):
    """Encode queries with normalization"""
    if isinstance(queries, str):
        queries = [queries]
    try:
        v = model.encode(queries, convert_to_numpy=True)
    except TypeError:
        v = model.encode(queries)
    v = v/np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype("float32")


def semantic_search(idx, emb, k):
    """Search in FAISS index"""
    D, I = idx.search(emb, k)
    return I[0], D[0]


def lexical_search(q, lex):
    """Keyword-based search with scoring"""
    tokens = re.findall(r"[A-Za-z_]\w+", q.lower())
    scores = defaultdict(float)

    for t in tokens:
        if t in lex:
            for doc_id in lex[t]:
                scores[doc_id] += 1.0

    # Sort by score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids, scores


def keyword_match_score(query, chunk):
    """Calculate keyword overlap score"""
    query_words = set(re.findall(r"[a-z_]\w+", query.lower()))

    # Extract keywords from chunk
    chunk_words = set()
    chunk_words.update(re.findall(r"[a-z_]\w+", chunk["name"].lower()))
    chunk_words.update(re.findall(r"[a-z_]\w+", chunk["code"][:500].lower()))

    if chunk.get("parameters"):
        for p in chunk["parameters"]:
            chunk_words.add(p["name"].lower())
            chunk_words.add(p["type"].lower())

    if not query_words:
        return 0.0

    overlap = len(query_words & chunk_words)
    return overlap / max(len(query_words), 1)


def parameter_match_score(query, chunk):
    """Check if query mentions parameters and if chunk has them"""
    param_keywords = ["param", "argument", "arg", "input", "receive", "accept", "take", "байт", "массив", "array", "byte"]
    has_param_query = any(kw in query.lower() for kw in param_keywords)

    if not has_param_query:
        return 0.5  # Neutral if query doesn't care about params

    if chunk.get("parameters") and len(chunk["parameters"]) > 0:
        # Check if parameter types match query intent
        query_lower = query.lower()
        for p in chunk["parameters"]:
            p_type = p["type"].lower()
            p_name = p["name"].lower()
            if any(kw in p_type or kw in p_name for kw in ["byte", "uint8", "char", "buffer", "data", "array", "vector", "string"]):
                return 1.0
        return 0.7  # Has params but not exact match
    return 0.0


def input_type_match_score(query, chunk):
    """Score based on input type matching"""
    query_lower = query.lower()

    # Check what input types the query is asking about
    wants_stdin = any(w in query_lower for w in ["stdin", "standard input", "console input", "cin", "scanf", "getchar", "fgets"])
    wants_file = any(w in query_lower for w in ["file", "fopen", "ifstream", "fstream", "read from file"])
    wants_api = any(w in query_lower for w in ["api", "http", "request", "network", "curl", "socket"])
    wants_param = any(w in query_lower for w in ["param", "argument", "input", "receive", "accept", "pass", "переда", "вход"])

    if not (wants_stdin or wants_file or wants_api or wants_param):
        return 0.5  # No specific input type requested

    score = 0.0
    if wants_stdin and chunk.get("has_stdin"):
        score += 0.5
    if wants_file and chunk.get("has_file_input"):
        score += 0.5
    if wants_api and chunk.get("has_api_call"):
        score += 0.5
    if wants_param and chunk.get("parameters") and len(chunk["parameters"]) > 0:
        score += 0.5

    return max(0.0, min(1.0, score))


def type_match_score(query, chunk):
    """Check if query asks about specific types and chunk has them"""
    query_lower = query.lower()
    matched_types = extract_requested_types(query_lower)

    if not matched_types:
        return 0.5  # No specific type requested

    if not chunk.get("parameters"):
        return 0.0

    chunk_types = set()
    for p in chunk["parameters"]:
        for t in TYPE_KEYWORDS.keys():
            if parameter_matches_type(p, t):
                chunk_types.add(t)

    overlap = len(set(matched_types) & chunk_types)
    return min(1.0, overlap / max(len(matched_types), 1))


def rerank_chunks(query, chunks, meta, analysis=None):
    """Rerank chunks using multiple signals"""
    if not chunks:
        return []

    scored_chunks = []

    for idx in chunks:
        chunk = meta[idx]
        if not is_valid_function_chunk(chunk):
            continue

        # Multiple scoring signals
        kw_score = keyword_match_score(query, chunk)
        param_score = parameter_match_score(query, chunk)
        input_score = input_type_match_score(query, chunk)
        type_score = type_match_score(query, chunk)
        parse_score = 1.0 if is_parse_like_chunk(chunk) else 0.0
        fuzz_score = fuzz_target_score(chunk)

        # Combine scores with weights
        total_score = (
            0.3 * kw_score +
            0.25 * param_score +
            0.25 * input_score +
            0.2 * type_score
        )

        # Soft constraints to help reranking even before strict pre-filter.
        if analysis:
            if analysis.get("needs_parse_like"):
                total_score += 0.2 * parse_score
            if analysis.get("needs_fuzz_targets"):
                total_score += 0.35 * fuzz_score
                if fuzz_score < analysis.get("min_fallback_fuzz_score", 0.15):
                    total_score -= 0.1

            positive_matches = collect_positive_constraint_matches(chunk, analysis)
            if positive_matches:
                if analysis.get("constraint_mode") == "any":
                    if any(positive_matches.values()):
                        total_score += 0.1
                    else:
                        total_score -= 0.2
                else:
                    if all(positive_matches.values()):
                        total_score += 0.1
                    else:
                        total_score -= 0.2

            if analysis.get("exclude_output") and is_write_like_chunk(chunk):
                total_score -= 0.25
            if analysis.get("requested_types") and not chunk_matches_requested_types(chunk, analysis["requested_types"]):
                total_score -= 0.15
            if analysis.get("needs_params") and not chunk.get("parameters"):
                total_score -= 0.15

        scored_chunks.append({
            "idx": idx,
            "chunk": chunk,
            "score": total_score,
                "breakdown": {
                    "keyword": kw_score,
                    "parameter": param_score,
                    "input_type": input_score,
                    "type_match": type_score,
                    "parse_like": parse_score,
                    "fuzz_target": fuzz_score
                }
            })

    # Sort by total score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks


def fuzzable_level(score):
    if score >= 0.65:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def listing_match_reason(chunk, analysis):
    reasons = []
    if analysis.get("needs_parse_like") and is_parse_like_chunk(chunk):
        reasons.append("parse/input-processing logic")
    if chunk.get("has_stdin"):
        reasons.append("reads from stdin")
    if chunk.get("has_file_input"):
        reasons.append("reads from files")
    if chunk.get("has_api_call"):
        reasons.append("handles API/network input")
    if chunk.get("uses_memory_management"):
        reasons.append("memory/buffer handling")
    if chunk.get("has_error_handling"):
        reasons.append("error-handling paths")
    if not reasons:
        reasons.append("matches retrieval constraints from indexed context")
    return ", ".join(reasons[:2])


def build_listing_answer_from_context(frags, analysis=None):
    """Deterministic listing fallback when model output fails verification."""
    analysis = analysis or {}
    clean_frags = [f for f in frags if is_valid_function_chunk(f)]
    if not clean_frags:
        return "No matching functions found in the indexed codebase for the specified constraints."

    lines = []
    for i, f in enumerate(clean_frags, 1):
        score = fuzz_target_score(f)
        level = fuzzable_level(score)
        reason = listing_match_reason(f, analysis)
        sig = f.get("signature") or f"{f.get('name', '')}()"
        file_loc = f"{f.get('file', '')}:{f.get('start_line', '?')}-{f.get('end_line', '?')}"
        lines.append(
            f"{i}. **`{f.get('name', '')}`**\n"
            f"   - File: `{file_loc}`\n"
            f"   - Signature: `{sig}`\n"
            f"   - Why it matches: {reason}\n"
            f"   - Fuzzable: {level} - score={score:.2f}"
        )

    if analysis.get("needs_fuzz_targets") and len(clean_frags) >= 2:
        top_names = [f.get("name", "") for f in clean_frags[:2] if f.get("name")]
        lower_names = [f.get("name", "") for f in clean_frags[2:4] if f.get("name")]
        if top_names:
            summary = f"Functions like {', '.join(top_names)} are ranked highest due to broader input/memory surfaces."
            if lower_names:
                summary += f" Others such as {', '.join(lower_names)} are lower due to narrower input complexity."
            lines.append("")
            lines.append(summary)

    return "\n".join(lines)


def _find_matching_paren_text(text, open_idx):
    """Find matching ')' for '(' with nested bracket and literal awareness."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "(":
        return -1

    depth = 0
    i = open_idx
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
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


def _split_top_level_arguments(args_text):
    out = []
    cur = []
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


def _extract_call_argument_lists(code, target_name, limit=3):
    if not code or not target_name:
        return []

    pattern = re.compile(
        rf"(?<![A-Za-z0-9_~])(?:[A-Za-z_]\w*::)*{re.escape(target_name)}\s*\("
    )
    out = []
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
        out.append({"expr": expr, "args": args})
        if len(out) >= limit:
            break

    return out


def _base_decl_type(type_text):
    t = re.sub(r"\bconst\b", " ", type_text or "")
    t = t.replace("*", " ").replace("&", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t or "int"


def _strip_angle_template_content(type_text):
    s = type_text or ""
    out = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def _infer_effective_param_type(param, pname):
    ptype = re.sub(r"\s+", " ", str(param.get("type", "")).strip())
    raw = re.sub(r"\s+", " ", str(param.get("raw", "")).strip())
    if not raw:
        return ptype

    raw = raw.split("=", 1)[0].strip()
    if pname:
        raw = re.sub(rf"\b{re.escape(pname)}\b\s*$", "", raw).strip()

    if not ptype:
        return raw
    if ("*" in raw and "*" not in ptype) or ("&" in raw and "&" not in ptype):
        return raw
    if len(raw) > len(ptype):
        return raw
    return ptype


def _next_unique_name(base, used):
    name = re.sub(r"\W+", "_", base or "").strip("_")
    if not name:
        name = "arg"
    if name[0].isdigit():
        name = f"v_{name}"
    candidate = name
    idx = 2
    while candidate in used:
        candidate = f"{name}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def _is_simple_literal(expr):
    if not expr:
        return False
    return re.fullmatch(
        r"(?:[-+]?\d+(?:\.\d+)?(?:[uUlLfF]*)|nullptr|NULL|true|false|'.*'|\".*\")",
        expr.strip(),
    ) is not None


def _is_simple_identifier(expr):
    return re.fullmatch(r"[A-Za-z_]\w*", (expr or "").strip()) is not None


def _is_file_size_param(type_text, name):
    t = (type_text or "").lower()
    n = (name or "").lower()
    if any(k in n for k in ["size", "len", "length", "count", "bytes", "n"]):
        return any(k in t for k in [
            "size_t", "ssize_t", "int", "long", "uint", "int32", "int64", "uint32", "uint64",
        ])
    return False


def _is_size_type(type_text):
    t = (type_text or "").lower()
    return "size_t" in t or "ssize_t" in t


def _format_file_loc(chunk):
    return f"{chunk.get('file', '')}:{chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}"


def _build_param_binding(param, analysis, arg_hint, used_names):
    pname = str(param.get("name", "")).strip() or "arg"
    ptype = _infer_effective_param_type(param, pname)
    ptype_lower = ptype.lower()
    needs_file = bool((analysis or {}).get("needs_file"))
    ptype_top = _strip_angle_template_content(ptype)
    ptype_top_lower = ptype_top.lower()

    is_pointer = "*" in ptype_top
    is_ref = "&" in ptype_top
    is_std_string = ("std::string" in ptype) or ("string" in ptype_lower)
    is_std_function = "std::function" in ptype_lower
    is_char_or_byte_ptr = is_pointer and not is_std_function and any(x in ptype_top_lower for x in [
        "char *", "char*", "uint8_t *", "uint8_t*", "unsigned char *", "unsigned char*",
        "std::byte *", "std::byte*", "void *", "void*",
    ])

    value_type = ptype.replace("&", "").strip() or _base_decl_type(ptype)
    decls = []

    hint = (arg_hint or "").strip()
    if hint:
        if hint.startswith("&") and _is_simple_identifier(hint[1:]):
            var = _next_unique_name(hint[1:], used_names)
            decls.append(f"{_base_decl_type(ptype)} {var}{{}};")
            return decls, f"&{var}"

        if _is_simple_literal(hint):
            return decls, hint

        if _is_simple_identifier(hint):
            if is_pointer:
                storage = _next_unique_name(f"{hint}_obj", used_names)
                ptr_name = _next_unique_name(hint, used_names)
                decls.append(f"{_base_decl_type(ptype)} {storage}{{}};")
                decls.append(f"{ptype} {ptr_name} = &{storage};")
                return decls, ptr_name
            if is_std_string and needs_file:
                var = _next_unique_name(hint, used_names)
                decls.append(f"std::string {var}(input_bytes.begin(), input_bytes.end());")
                return decls, var
            var = _next_unique_name(hint, used_names)
            decls.append(f"{value_type} {var}{{}};")
            return decls, var

    if needs_file and is_char_or_byte_ptr and is_pointer:
        n = (pname or "").lower()
        if any(k in n for k in ["e", "end", "last", "finish", "to"]):
            return decls, f"reinterpret_cast<{ptype}>(input_bytes.data() + input_bytes.size())"
        return decls, f"reinterpret_cast<{ptype}>(input_bytes.data())"

    if needs_file and _is_file_size_param(ptype, pname):
        cast_type = value_type or "size_t"
        return decls, f"static_cast<{cast_type}>(input_bytes.size())"
    if needs_file and _is_size_type(ptype):
        cast_type = value_type or "size_t"
        return decls, f"static_cast<{cast_type}>(input_bytes.size())"

    if is_std_string and needs_file:
        var = _next_unique_name(f"{pname}_str", used_names)
        decls.append(f"std::string {var}(input_bytes.begin(), input_bytes.end());")
        return decls, var

    if needs_file and not is_pointer and not is_ref and "char" in ptype_lower:
        n = (pname or "").lower()
        if any(k in n for k in ["d", "delim", "delimiter", "sep", "separator"]):
            var = _next_unique_name(pname, used_names)
            decls.append(f"{value_type} {var} = '\\n';")
            return decls, var

    if is_pointer:
        storage = _next_unique_name(f"{pname}_obj", used_names)
        ptr_name = _next_unique_name(pname, used_names)
        decls.append(f"{_base_decl_type(ptype)} {storage}{{}};")
        decls.append(f"{ptype} {ptr_name} = &{storage};")
        return decls, ptr_name

    if is_ref:
        var = _next_unique_name(pname, used_names)
        decls.append(f"{_base_decl_type(ptype)} {var}{{}};")
        return decls, var

    if "bool" in ptype_lower:
        var = _next_unique_name(pname, used_names)
        decls.append(f"{value_type} {var} = false;")
        return decls, var

    if any(t in ptype_lower for t in ["float", "double"]):
        var = _next_unique_name(pname, used_names)
        decls.append(f"{value_type} {var} = 0;")
        return decls, var

    if any(t in ptype_lower for t in ["int", "long", "short", "size_t", "uint"]):
        var = _next_unique_name(pname, used_names)
        decls.append(f"{value_type} {var} = 0;")
        return decls, var

    var = _next_unique_name(pname, used_names)
    decls.append(f"{value_type} {var}{{}};")
    return decls, var


def build_example_answer_from_context(frags, analysis=None, example_context=None):
    """Deterministic fallback for grounded example-generation answers."""
    analysis = analysis or {}
    clean_frags = [f for f in frags if is_valid_function_chunk(f)]
    if not clean_frags:
        return "Not enough verified context to build a reliable example."

    if example_context is None:
        example_context = _build_example_context_impl(clean_frags, analysis=analysis)

    target = example_context.get("target")
    if target is None or target not in clean_frags:
        target = clean_frags[0]
    target_name = str(target.get("name", "")).strip() or "target_function"
    target_sig = target.get("signature") or f"{target_name}()"
    params = list(target.get("parameters") or [])

    call_hint_args = []
    caller = example_context.get("caller")
    if caller is not None and caller not in clean_frags:
        caller = None
    observed_call_obj = example_context.get("observed_call") or {}
    observed_call = observed_call_obj.get("expr")
    if isinstance(observed_call_obj.get("args"), list):
        call_hint_args = observed_call_obj.get("args", [])

    header_name = os.path.basename(str(target.get("file", "")))
    include_target_header = bool(re.search(r"\.(h|hpp|hh|hxx)$", header_name))
    needs_file_bytes = bool(analysis.get("needs_file"))

    include_lines = [
        "#include <cstdint>",
        "#include <cstddef>",
        "#include <vector>",
        "#include <string>",
        "#include <fstream>",
        "#include <iterator>",
        "#include <iostream>",
    ]
    if include_target_header:
        include_lines.append(f"#include \"{header_name}\"")

    body_lines = []
    if needs_file_bytes:
        body_lines.extend([
            "if (argc < 2) {",
            "    std::cerr << \"Usage: \" << argv[0] << \" <input_file>\\n\";",
            "    return 1;",
            "}",
            "",
            "std::vector<uint8_t> input_bytes;",
            "std::ifstream in(argv[1], std::ios::binary);",
            "if (!in) {",
            "    std::cerr << \"Failed to open input file\\n\";",
            "    return 1;",
            "}",
            "input_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());",
            "",
        ])
    else:
        body_lines.append("std::vector<uint8_t> input_bytes;")
        body_lines.append("")

    used_names = set()
    decl_lines = []
    arg_exprs = []
    for i, p in enumerate(params):
        hint = call_hint_args[i] if i < len(call_hint_args) else None
        d, arg = _build_param_binding(p, analysis, hint, used_names)
        decl_lines.extend(d)
        arg_exprs.append(arg)
    body_lines.extend(decl_lines)
    if decl_lines:
        body_lines.append("")

    args_joined = ", ".join(arg_exprs)
    sig_head = target_sig.split(target_name, 1)[0]
    is_void_return = re.search(r"\bvoid\s*$", sig_head.strip()) is not None
    if is_void_return:
        body_lines.append(f"{target_name}({args_joined});")
    else:
        body_lines.append(f"auto result = {target_name}({args_joined});")
        body_lines.append("(void)result;")
    body_lines.append("return 0;")

    code_lines = []
    code_lines.extend(include_lines)
    code_lines.append("")
    code_lines.append(f"// Target signature (from indexed code): {target_sig}")
    if observed_call:
        code_lines.append(f"// Observed call pattern in codebase: {observed_call}")
    code_lines.append("int main(int argc, char ** argv) {")
    code_lines.extend([f"    {ln}" if ln else "" for ln in body_lines])
    code_lines.append("}")

    lines = [
        "### Example Code (deterministic context-grounded fallback)",
        "```cpp",
        "\n".join(code_lines),
        "```",
        "",
        f"Target function: `{target_name}`",
        "Evidence from codebase:",
        f"- File: `{_format_file_loc(target)}`",
        f"- Signature: `{target_sig}`",
    ]
    if caller is not None:
        caller_sig = caller.get("signature") or f"{caller.get('name', 'caller')}()"
        lines.append(f"- File: `{_format_file_loc(caller)}`")
        lines.append(f"- Signature: `{caller_sig}`")
        if observed_call:
            lines.append(f"- Observed call: `{observed_call}`")
    else:
        lines.append("- Note: direct caller context for target function was not found in selected fragments.")

    return "\n".join(lines)


def _infer_param_role_from_shape(pname, ptype):
    shape = _classify_param_shape_impl(ptype, pname)
    n = (pname or "").lower()
    if "path" in n or "file" in n:
        return shape, "path/file locator"
    if "ctx" in n:
        return shape, "context/config value"
    if "param" in n or "config" in n:
        return shape, "configuration structure/value"
    if shape == "byte_buffer":
        return shape, "raw input buffer / payload bytes"
    if shape == "size_or_length":
        return shape, "size/length/count for related payload"
    if shape == "out_pointer":
        return shape, "output pointer populated by callee"
    if shape == "pointer":
        return shape, "pointer to object/array for in/out behavior"
    if shape == "reference":
        return shape, "by-reference in/out value"
    if shape == "integer":
        return shape, "numeric control parameter"
    if shape == "float":
        return shape, "floating-point control parameter"
    if shape == "bool":
        return shape, "boolean feature/behavior flag"
    if shape == "string":
        return shape, "text/string input"
    return shape, "unknown from provided context"


def build_parameter_analysis_from_context(frags, analysis=None, example_context=None, function_hints=None):
    """Deterministic fallback for parameter-analysis answers."""
    analysis = analysis or {}
    clean_frags = [f for f in frags if is_valid_function_chunk(f)]
    if not clean_frags:
        return "Not enough verified context to build parameter analysis."

    if example_context is None:
        example_context = _build_example_context_impl(clean_frags, analysis=analysis)

    target = example_context.get("target")
    if target is None or target not in clean_frags:
        primary = analysis.get("primary_function_name")
        by_name = {f.get("name"): f for f in clean_frags}
        target = by_name.get(primary) if primary else None
    if target is None:
        target = clean_frags[0]

    target_name = str(target.get("name", "")).strip() or "target_function"
    target_sig = target.get("signature") or f"{target_name}()"
    params = list(target.get("parameters") or [])
    caller = example_context.get("caller")
    observed = example_context.get("observed_call") or {}
    observed_args = list(observed.get("args") or [])

    hints_for_target = list((function_hints or {}).get(target_name, []))

    lines = [
        f"Function: `{target_name}`",
        f"Signature: `{target_sig}`",
        "Parameters:",
    ]

    unknowns = []
    for idx, p in enumerate(params, 1):
        pname = str(p.get("name", "")).strip() or f"arg{idx - 1}"
        ptype = str(p.get("type", "")).strip() or "unknown"
        shape, role = _infer_param_role_from_shape(pname, ptype)

        expected_fmt = f"{shape}; declared type `{ptype}`"
        src_bits = []
        if idx - 1 < len(observed_args):
            src_bits.append(f"observed caller arg `{observed_args[idx - 1]}`")
        if "path" in pname.lower() or "file" in pname.lower():
            src_bits.append("usually comes from file path/CLI/config")
        if shape in {"byte_buffer", "size_or_length"}:
            src_bits.append("often paired with buffer+size data flow")
        typical_source = "; ".join(src_bits) if src_bits else "not explicit in selected fragments"

        evidence_items = [f"`{_format_file_loc(target)}`"]
        if caller is not None:
            evidence_items.append(f"`{_format_file_loc(caller)}`")
        if hints_for_target:
            h0 = hints_for_target[0]
            evidence_items.append(f"`{h0.get('file', '')}:{h0.get('line', '?')}` (docs/guide)")

        lines.extend([
            f"{idx}. `{pname}` (`{ptype}`)",
            f"- Role: {role}",
            f"- Expected format: {expected_fmt}",
            f"- Typical source: {typical_source}",
            f"- Evidence: {', '.join(evidence_items)}",
        ])

        if role == "unknown from provided context":
            unknowns.append(f"`{pname}` semantic meaning is not explicit in current context")

    lines.append("Unknowns:")
    if unknowns:
        for u in unknowns:
            lines.append(f"- {u}")
    else:
        lines.append("- No critical unknowns from selected context; domain-specific constraints may still exist.")

    if hints_for_target:
        lines.append("Docs/guide hints:")
        for h in hints_for_target[:4]:
            snippet = " ".join(str(h.get("snippet", "")).split())[:260]
            lines.append(
                f"- `{h.get('file', '')}:{h.get('line', '?')}`: {snippet}"
            )

    return "\n".join(lines)


class QueryPlanner:
    """Advanced query planner with thinking mode support"""

    def __init__(self, special_indices, symbols, call_graph, called_by, meta=None):
        self.special_indices = special_indices
        self.symbols = symbols
        self.call_graph = call_graph
        self.called_by = called_by
        self.meta = meta or []
        self.symbols_by_lower = defaultdict(list)
        for fn in self.symbols.keys():
            self.symbols_by_lower[fn.lower()].append(fn)

    def analyze_query(self, query, context_history=None):
        """Analyze query with v2-by-default analyzer and optional shadow diff."""
        legacy_mode = os.getenv("FC_QUERY_ANALYZER_LEGACY", "0") == "1"
        use_v2 = not legacy_mode
        shadow = os.getenv("FC_QUERY_ANALYZER_SHADOW", "0") == "1"

        legacy = None
        if not use_v2 or shadow:
            legacy = self._analyze_query_legacy(query, context_history=context_history)
        if use_v2:
            v2 = _analyze_query_v2_impl(
                query,
                context_history=context_history,
                symbols_by_lower=self.symbols_by_lower,
            )
            if shadow and legacy is not None:
                diff = _compare_analysis_impl(legacy, v2)
                if diff:
                    v2["_shadow_diff"] = diff
            return v2

        if shadow and legacy is not None:
            v2 = _analyze_query_v2_impl(
                query,
                context_history=context_history,
                symbols_by_lower=self.symbols_by_lower,
            )
            diff = _compare_analysis_impl(legacy, v2)
            if diff:
                legacy["_shadow_diff"] = diff
        return legacy

    def _analyze_query_legacy(self, query, context_history=None):
        """Analyze query to determine search strategy with thinking mode"""
        query_lower = query.lower()

        analysis = {
            "query_type": "general",
            "keywords": re.findall(r"[a-z_а-яё]\w+", query_lower),
            "is_listing": False,
            "needs_stdin": False,
            "needs_file": False,
            "needs_api": False,
            "needs_output": False,
            "needs_memory_mgmt": False,
            "needs_error_handling": False,
            "needs_params": False,
            "needs_types": False,
            "needs_param_semantics": False,
            "needs_parse_like": False,
            "needs_fuzz_targets": False,
            "needs_type_info": False,
            "requested_types": [],
            "constraint_mode": "all",
            "exclude_output": False,
            "max_param_count": None,
            "needs_broad_fuzz_surface": False,
            "min_fuzz_score": 0.35,
            "min_fallback_fuzz_score": 0.15,
            "listing_target_count": None,
            "path_filters": [],
            "exclude_previously_listed": False,
            "function_names": [],
            "primary_function_name": None,
            "expand_callers": False,
            "expand_callees": False,
            "needs_example": False,
            "needs_implementation": False,
            "follow_up": False,
            "referenced_functions": [],
            "query_function_candidates": [],
        }

        # Detect input type requirements
        if any(w in query_lower for w in ["stdin", "standard input", "console", "cin", "scanf", "getchar", "fgets", "ввод"]):
            analysis["needs_stdin"] = True

        if any(w in query_lower for w in ["file", "fopen", "ifstream", "fstream", "read from file", "файл"]):
            analysis["needs_file"] = True

        if any(w in query_lower for w in ["api", "http", "request", "network", "curl", "socket", "сеть"]):
            analysis["needs_api"] = True

        # Detect output/error/memory focused queries (kept conservative to avoid
        # matching imperative phrases like "write a list ...").
        if any(w in query_lower for w in [
            "stdout", "stderr", "output", "print", "printf", "fprintf", "cout", "cerr",
            "clog", "logging", "logger", "log ", "log-", "вывод", "печать", "логг",
        ]) or re.search(r"\bwrite(s|d|ing)?\s+(to|into)\b", query_lower):
            analysis["needs_output"] = True

        if any(w in query_lower for w in [
            "memory", "buffer", "malloc", "calloc", "realloc", "free",
            "new/delete", "memcpy", "memmove", "heap", "stack",
            "памят", "буфер", "переполн",
        ]):
            analysis["needs_memory_mgmt"] = True

        if any(w in query_lower for w in [
            "error handling", "error", "errors", "exception", "exceptions",
            "throw", "catch", "assert", "errno", "validation",
            "ошиб", "исключен", "валидац",
        ]):
            analysis["needs_error_handling"] = True

        # Detect parameter-related queries
        if any(w in query_lower for w in ["param", "argument", "arg", "input", "receive", "accept", "take", "переда", "вход", "параметр"]):
            analysis["needs_params"] = True

        param_semantics_markers = [
            "parameter semantics",
            "what parameters", "which parameters", "parameter meanings", "meaning of parameter",
            "what does parameter", "parameter format", "data format of parameter",
            "куда передается", "что означает параметр", "какие параметры принимает",
            "какие аргументы принимает", "формат параметров", "формат данных параметра",
            "откуда берутся параметры", "source of parameters",
        ]
        if any(m in query_lower for m in param_semantics_markers):
            analysis["needs_param_semantics"] = True
            analysis["needs_params"] = True

        # Detect type-specific queries
        analysis["requested_types"] = extract_requested_types(query_lower)
        if analysis["requested_types"]:
            analysis["needs_type_info"] = True
            analysis["needs_types"] = True

        # Detect parse-like intent
        if any(w in query_lower for w in [
            "parse", "parser", "parsing", "tokenize", "split", "decode", "deserialize",
            "scan", "lex", "grammar", "peg", "разбор", "парс"
        ]):
            analysis["needs_parse_like"] = True

        # Detect fuzzing-target intent
        if query_has_any_keyword(query_lower, FUZZ_QUERY_KEYWORDS):
            analysis["needs_fuzz_targets"] = True
            # Fuzzing asks are effectively listing/ranking asks even without explicit "list".
            analysis["is_listing"] = True

        # Detect explicit function-like mentions from query text.
        analysis["query_function_candidates"] = extract_function_like_candidates(
            query,
            known_symbols_by_lower=self.symbols_by_lower,
        )
        for cand in analysis["query_function_candidates"]:
            for resolved in self.symbols_by_lower.get(cand.lower(), []):
                if resolved not in analysis["function_names"]:
                    analysis["function_names"].append(resolved)
                    analysis["referenced_functions"].append(resolved)

        # Alias-friendly behavior: in prompts like
        # "define a standalone main() and call X from it", "main" is snippet context,
        # not a target function to retrieve from codebase.
        if ("standalone main" in query_lower or "define a standalone main" in query_lower) and len(analysis["function_names"]) > 1:
            analysis["function_names"] = [fn for fn in analysis["function_names"] if fn.lower() != "main"]
            analysis["referenced_functions"] = [fn for fn in analysis["referenced_functions"] if fn.lower() != "main"]

        # Detect call graph expansion needs
        if any(w in query_lower for w in ["call", "invoke", "use", "caller", "callee", "called by", "вызыва", "использу"]):
            if "caller" in query_lower or "called by" in query_lower or "кто вызыва" in query_lower:
                analysis["expand_callers"] = True
            else:
                analysis["expand_callees"] = True

        # Detect listing/enumeration queries
        if any(w in query_lower for w in ["list", "enumerate", "show all", "find all", "which functions", "какие функции", "перечисли", "покажи все"]):
            analysis["is_listing"] = True
            analysis["query_type"] = "listing"

        explicit_max_params = _extract_max_param_count_impl(query_lower)
        if explicit_max_params is not None:
            analysis["max_param_count"] = explicit_max_params

        # Detect exclusion constraints
        if query_excludes_output(query_lower):
            analysis["exclude_output"] = True
            # Negative output constraint overrides positive output intent.
            analysis["needs_output"] = False

        # Detect path/module filters
        analysis["path_filters"] = extract_path_filters_from_query(query)

        # Detect novelty requests: "other/different/new functions"
        novelty_requested = any(w in query_lower for w in [
            "other", "another", "different", "new", "remaining", "else",
            "друг", "еще", "ещё", "остальн", "дополнительно"
        ])
        if novelty_requested and (
            analysis["is_listing"] or "function" in query_lower or "функц" in query_lower
        ):
            analysis["exclude_previously_listed"] = True

        # Decide whether positive constraints are all-required or any-of
        positive_signals = 0
        positive_signals += int(analysis["needs_stdin"])
        positive_signals += int(analysis["needs_file"])
        positive_signals += int(analysis["needs_api"])
        positive_signals += int(analysis["needs_output"])
        positive_signals += int(analysis["needs_memory_mgmt"])
        positive_signals += int(analysis["needs_error_handling"])
        positive_signals += int(bool(analysis["requested_types"]))
        positive_signals += int(analysis["needs_parse_like"])
        positive_signals += int(analysis["needs_fuzz_targets"])

        has_or_connector = re.search(r"\b(or|или)\b", query_lower) is not None
        has_and_connector = re.search(r"\b(and|и)\b", query_lower) is not None

        if positive_signals > 1:
            if has_or_connector:
                analysis["constraint_mode"] = "any"
            elif has_and_connector:
                analysis["constraint_mode"] = "all"
            elif analysis["is_listing"] and analysis["needs_parse_like"]:
                # Typical query style: "parse ... or stdin/string/bytes input"
                analysis["constraint_mode"] = "any"

        if analysis["needs_fuzz_targets"] and analysis["is_listing"]:
            analysis["needs_broad_fuzz_surface"] = True
            if analysis["max_param_count"] is None:
                analysis["max_param_count"] = 4
            analysis["constraint_mode"] = "any"
            if not analysis.get("listing_target_count"):
                analysis["listing_target_count"] = 25
            explicit_mentions = _extract_explicit_function_mentions_impl(
                query,
                known_symbols_by_lower=self.symbols_by_lower,
            )
            if explicit_mentions:
                explicit_set = set(explicit_mentions)
                analysis["function_names"] = [fn for fn in analysis["function_names"] if fn in explicit_set]
                if not analysis["function_names"]:
                    analysis["function_names"] = list(explicit_mentions)
                analysis["referenced_functions"] = [fn for fn in analysis["referenced_functions"] if fn in explicit_set]
                analysis["query_function_candidates"] = [fn for fn in analysis["query_function_candidates"] if fn in explicit_set]
            else:
                analysis["function_names"] = []
                analysis["referenced_functions"] = []
                analysis["query_function_candidates"] = []

        # Detect example generation requests
        if any(w in query_lower for w in ["example", "пример", "как вызвать", "как использовать", "usage", "использовани"]):
            analysis["needs_example"] = True
            analysis["query_type"] = "example_generation"
            # Example generation should not be treated as listing/ranking query.
            analysis["is_listing"] = False
            analysis["exclude_previously_listed"] = False
            # If user named concrete functions, focus retrieval on those instead of broad fuzz-target discovery.
            if analysis["function_names"]:
                analysis["needs_fuzz_targets"] = False
                analysis["primary_function_name"] = choose_primary_example_function(
                    query,
                    analysis["function_names"],
                )

        # Detect implementation questions
        if any(w in query_lower for w in ["implement", "реализ", "как работает", "how does", "algorithm", "алгоритм"]):
            analysis["needs_implementation"] = True
            analysis["query_type"] = "implementation_explanation"

        # Check for follow-up questions
        if context_history:
            # Use phrase-level references instead of raw tokens like "that":
            # "that" is common in English relative clauses ("functions that parse...").
            follow_up_patterns = [
                r"\b(these|those)\b",
                r"\b(this|that)\s+(one|ones|function|functions|list|result|results|candidate|candidates|answer)\b",
                r"\b(from|in)\s+(the\s+)?(list|previous|above|earlier)\b",
                r"\b(previous|above|earlier)\s+(list|answer|results?)\b",
                r"\bиз\s+(этого|того|предыдущего)?\s*списк",
                r"\b(эт(и|от|у)|тот)\s+(функц|спис)",
                r"\b(предыдущ|выше|ранее)\s+(ответ|спис|результ)",
                r"\b(далее|дальше|продолж)\b",
                r"\bfunction\s+[abc]\b",
            ]

            if any(re.search(p, query_lower) for p in follow_up_patterns):
                analysis["follow_up"] = True

            if analysis["exclude_previously_listed"]:
                analysis["follow_up"] = True

        # Determine query type
        if analysis["needs_example"]:
            analysis["query_type"] = "example_generation"
        elif analysis["needs_implementation"]:
            analysis["query_type"] = "implementation_explanation"
        elif analysis["needs_param_semantics"] and len(analysis["function_names"]) > 0:
            analysis["query_type"] = "parameter_analysis"
            analysis["is_listing"] = False
            analysis["needs_file"] = False
            analysis["needs_output"] = False
            analysis["needs_api"] = False
            analysis["needs_stdin"] = False
            if not analysis.get("primary_function_name"):
                analysis["primary_function_name"] = analysis["function_names"][0]
        elif analysis["is_listing"]:
            analysis["query_type"] = "listing"
        elif analysis["needs_stdin"] or analysis["needs_file"] or analysis["needs_api"]:
            analysis["query_type"] = "input_specific"
        elif analysis["needs_type_info"]:
            analysis["query_type"] = "type_specific"
        elif len(analysis["function_names"]) > 0:
            analysis["query_type"] = "function_specific"

        if analysis.get("query_type") != "listing":
            analysis["listing_target_count"] = None
            analysis["needs_broad_fuzz_surface"] = False

        return analysis

    def collect_previously_listed_ids(self, context_history):
        """Collect function ids mentioned in prior assistant responses."""
        if not context_history:
            return set()

        ids = set()
        for _q, ans in context_history:
            mentioned = extract_function_names_from_text(ans, self.symbols)
            for fn in mentioned:
                for idx in self.symbols.get(fn, []):
                    ids.add(idx)
        return ids

    def get_search_candidates(self, analysis, k=20):
        """Get candidate indices based on query analysis"""
        if analysis.get("query_type") in {"example_generation", "parameter_analysis", "function_specific", "implementation_explanation"} and analysis.get("function_names"):
            focused = []
            primary_name = analysis.get("primary_function_name")
            if analysis.get("query_type") == "example_generation" and primary_name:
                focus_names = [primary_name]
            else:
                focus_names = list(analysis["function_names"])

            for func_name in focus_names:
                for idx in self.symbols.get(func_name, []):
                    if idx not in focused:
                        focused.append(idx)

            # Bring immediate call-graph neighborhood for realistic usage examples.
            # Keep deterministic priority: target -> callers -> callees.
            ordered = []
            seen = set()

            def _add_id(x):
                if x not in seen:
                    seen.add(x)
                    ordered.append(x)

            for idx in focused:
                _add_id(idx)
            for func_name in analysis.get("function_names", []):
                if func_name in focus_names:
                    continue
                for idx in self.symbols.get(func_name, []):
                    _add_id(idx)
            for idx in focused:
                cg = self.call_graph.get(str(idx), {})
                for caller in cg.get("called_by", []):
                    _add_id(caller)
            for idx in focused:
                cg = self.call_graph.get(str(idx), {})
                for callee in cg.get("resolved_calls", []):
                    _add_id(callee)

            return ordered[: max(k * 3, 30)]

        candidates = set()

        # Use special indices for input-specific queries
        if analysis["needs_stdin"] and "stdin" in self.special_indices:
            candidates.update(self.special_indices["stdin"])

        if analysis["needs_file"] and "file_input" in self.special_indices:
            candidates.update(self.special_indices["file_input"])

        if analysis["needs_api"] and "api_calls" in self.special_indices:
            candidates.update(self.special_indices["api_calls"])

        # Use additional feature-based indices
        if analysis.get("needs_output") and "output" in self.special_indices:
            candidates.update(self.special_indices["output"])

        if analysis.get("needs_memory_mgmt") and "memory_management" in self.special_indices:
            candidates.update(self.special_indices["memory_management"])

        if analysis.get("needs_error_handling") and "error_handling" in self.special_indices:
            candidates.update(self.special_indices["error_handling"])

        # Path/module constraint candidates (deterministic, independent from vector search).
        if analysis.get("path_filters") and self.meta:
            for idx, chunk in enumerate(self.meta):
                if chunk_matches_path_filters(chunk, analysis["path_filters"]):
                    candidates.add(idx)

        # Fuzz-target discovery mode: broaden candidates to likely crash-prone surfaces.
        if analysis.get("needs_fuzz_targets"):
            for key in ["stdin", "file_input", "api_calls", "memory_management", "error_handling"]:
                if key in self.special_indices:
                    candidates.update(self.special_indices[key])

            by_type = self.special_indices.get("by_type", {})
            for type_name in ["byte_array", "string", "template", "pointer", "integer"]:
                if type_name in by_type:
                    candidates.update(by_type[type_name])
                for alias in TYPE_INDEX_ALIASES.get(type_name, []):
                    if alias in by_type:
                        candidates.update(by_type[alias])

        if analysis.get("needs_broad_fuzz_surface"):
            for key in ["stdin", "file_input"]:
                if key in self.special_indices:
                    candidates.update(self.special_indices[key])
            by_type = self.special_indices.get("by_type", {})
            for type_name in ["byte_array", "string", "template", "pointer", "integer"]:
                if type_name in by_type:
                    candidates.update(by_type[type_name])
                for alias in TYPE_INDEX_ALIASES.get(type_name, []):
                    if alias in by_type:
                        candidates.update(by_type[alias])

        # Type-based search
        if analysis.get("requested_types"):
            by_type = self.special_indices.get("by_type", {})
            for type_name in analysis["requested_types"]:
                if type_name in by_type:
                    candidates.update(by_type[type_name])
                for alias in TYPE_INDEX_ALIASES.get(type_name, []):
                    if alias in by_type:
                        candidates.update(by_type[alias])

        # Add function-specific candidates
        for func_name in analysis["function_names"]:
            if func_name in self.symbols:
                candidates.update(self.symbols[func_name])

        # Expand via call graph if needed
        if analysis["expand_callers"] or analysis["expand_callees"]:
            expanded = set(candidates)
            for idx in list(candidates):
                if str(idx) in self.call_graph:
                    if analysis["expand_callers"]:
                        expanded.update(self.call_graph[str(idx)].get("called_by", []))
                    if analysis["expand_callees"]:
                        expanded.update(self.call_graph[str(idx)].get("resolved_calls", []))
            candidates = expanded

        # Strict exclusion at retrieval level when user asked for "not write-like/output"
        if analysis.get("exclude_output") and "output" in self.special_indices:
            candidates.difference_update(self.special_indices["output"])

        return sorted(candidates)[:k*3]  # deterministic order for stable retrieval


def build_thinking_prompt(frags, q, analysis=None, conversation_history=None):
    """Compatibility wrapper for thinking prompt builder."""
    return _build_thinking_prompt_impl(frags, q, analysis=analysis, conversation_history=conversation_history)


def build_example_context(frags, analysis=None):
    """Build structured context facts for example-generation."""
    return _build_example_context_impl(frags, analysis=analysis)


def build_example_context_grounded(
    *,
    frags,
    analysis=None,
    meta=None,
    symbols=None,
    call_graph=None,
    called_by=None,
    candidate_ids=None,
    top_k=40,
):
    """Build richer example grounding using expanded retrieval/call-graph evidence."""
    return _build_example_context_grounded_impl(
        frags=frags,
        analysis=analysis,
        meta=meta or [],
        symbols=symbols or {},
        call_graph=call_graph or {},
        called_by=called_by or {},
        candidate_ids=candidate_ids or [],
        top_k=top_k,
    )


def build_prompt(
    frags,
    q,
    analysis=None,
    conversation_history=None,
    max_prompt_chars=20000,
    example_context=None,
    function_hints=None,
):
    """Main prompt builder with total-character budget."""
    return _build_prompt_impl(
        frags,
        q,
        analysis=analysis,
        conversation_history=conversation_history,
        max_prompt_chars=max_prompt_chars,
        example_context=example_context,
        function_hints=function_hints,
    )


def call_llm(prompt, model, temperature=0.1):
    """Call LLM and return structured status dict."""
    return _call_llm_impl(prompt, model, OLLAMA_URL, temperature=temperature)


def verify_answer_with_context(answer, context_frags, known_functions=None):
    """Verify that functions mentioned in answer exist in provided context."""
    return _verify_answer_with_context_impl(answer, context_frags, known_functions=known_functions)


def verify_example_answer_with_context(
    answer,
    context_frags,
    target_function=None,
    known_functions=None,
    example_context=None,
):
    """Verify example-generation answer against file/signature/line evidence and context."""
    return _verify_example_answer_with_context_impl(
        answer,
        context_frags,
        target_function=target_function,
        known_functions=known_functions,
        example_context=example_context,
    )
