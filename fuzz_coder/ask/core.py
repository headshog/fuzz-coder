#!/usr/bin/env python3
"""
Advanced Code Query System with Thinking Mode, Context Memory, and Multi-Step Reasoning
Supports follow-up questions, code generation examples, and deep codebase understanding
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from pathlib import Path
import requests
import re
import json
import argparse
import os
from collections import defaultdict
import time
from typing import List, Dict, Any, Optional
from .llm import call_llm as _call_llm_impl
from .prompting import build_prompt as _build_prompt_impl
from .prompting import _build_thinking_prompt_with_limit as _build_thinking_prompt_impl
from .verification import verify_answer_with_context as _verify_answer_with_context_impl

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

PATH_FILTER_PATTERNS = [
    # English: from/in module|directory|subdirectory|folder|path <value>
    r"\b(?:from|in)\s+(?:the\s+)?(?:module|directory|subdirectory|folder|path)\s+([`\"']?)([^`\"'\n,;]+)\1",
    r"\bunder\s+([`\"']?)([^`\"'\n,;]+)\1",
    # Russian: из/в модуля|директории|поддиректории|папки <value>
    r"\bиз\s+(?:модуля|директории|поддиректории|папки)\s+([`\"']?)([^`\"'\n,;]+)\1",
    r"\bв\s+(?:модуле|директории|поддиректории|папке)\s+([`\"']?)([^`\"'\n,;]+)\1",
]


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
        r"\s+(?:that|which|where|with|and|or|who|whose|котор(?:ый|ая|ые|ого|ому|ых)?|где|и|или|с)\b",
        p,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    return p.strip(" \t\r\n.,:;!?")


def extract_path_filters_from_query(query, query_lower):
    """Extract module/directory constraints from query."""
    filters = []

    for pattern in PATH_FILTER_PATTERNS:
        for m in re.finditer(pattern, query, flags=re.IGNORECASE):
            raw = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
            norm = normalize_path_filter(clean_path_candidate(raw))
            if norm:
                filters.append(norm)

    # Additional explicit path-like hints in backticks or quotes.
    for p in re.findall(r"`([^`]+[/\\][^`]+)`", query):
        norm = normalize_path_filter(p)
        if norm:
            filters.append(norm)
    for p in re.findall(r"['\"]([^'\"]+[/\\][^'\"]+)['\"]", query):
        norm = normalize_path_filter(p)
        if norm:
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


def extract_function_names_from_text(text, symbols):
    """Extract known function names mentioned in free text."""
    if not text or not symbols:
        return set()

    found = set()
    for fn in symbols.keys():
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(fn)}(?![A-Za-z0-9_])", text):
            found.add(fn)
    return found


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
    if analysis.get("requested_types"):
        matches["types"] = chunk_matches_requested_types(chunk, analysis["requested_types"])
    if analysis.get("needs_parse_like"):
        matches["parse_like"] = is_parse_like_chunk(chunk)
    if analysis.get("needs_fuzz_targets"):
        min_fuzz_score = analysis.get("min_fuzz_score", 0.35)
        matches["fuzz_target"] = fuzz_target_score(chunk) >= min_fuzz_score
    return matches


def chunk_matches_constraints(chunk, analysis):
    """Hard constraints matcher used for strict listing pre-filter."""
    if analysis.get("needs_params") and not chunk.get("parameters"):
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


def lexical_search(q, lex, meta):
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


def rerank_chunks(query, chunks, meta, call_graph=None, analysis=None):
    """Rerank chunks using multiple signals"""
    if not chunks:
        return []

    scored_chunks = []

    for idx in chunks:
        chunk = meta[idx]

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


class QueryPlanner:
    """Advanced query planner with thinking mode support"""

    def __init__(self, special_indices, symbols, call_graph, called_by, meta=None):
        self.special_indices = special_indices
        self.symbols = symbols
        self.call_graph = call_graph
        self.called_by = called_by
        self.meta = meta or []

    def analyze_query(self, query, context_history=None):
        """Analyze query to determine search strategy with thinking mode"""
        query_lower = query.lower()

        analysis = {
            "query_type": "general",
            "keywords": re.findall(r"[a-z_а-яё]\w+", query_lower),
            "is_listing": False,
            "needs_stdin": False,
            "needs_file": False,
            "needs_api": False,
            "needs_params": False,
            "needs_types": False,
            "needs_parse_like": False,
            "needs_fuzz_targets": False,
            "needs_type_info": False,
            "requested_types": [],
            "constraint_mode": "all",
            "exclude_output": False,
            "min_fuzz_score": 0.35,
            "min_fallback_fuzz_score": 0.15,
            "path_filters": [],
            "exclude_previously_listed": False,
            "function_names": [],
            "expand_callers": False,
            "expand_callees": False,
            "needs_example": False,
            "needs_implementation": False,
            "follow_up": False,
            "referenced_functions": []
        }

        # Detect input type requirements
        if any(w in query_lower for w in ["stdin", "standard input", "console", "cin", "scanf", "getchar", "fgets", "ввод"]):
            analysis["needs_stdin"] = True

        if any(w in query_lower for w in ["file", "fopen", "ifstream", "fstream", "read from file", "файл"]):
            analysis["needs_file"] = True

        if any(w in query_lower for w in ["api", "http", "request", "network", "curl", "socket", "сеть"]):
            analysis["needs_api"] = True

        # Detect parameter-related queries
        if any(w in query_lower for w in ["param", "argument", "arg", "input", "receive", "accept", "take", "переда", "вход", "параметр"]):
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

        # Detect function name mentions
        if self.symbols:
            for func_name in self.symbols.keys():
                if func_name.lower() in query_lower or func_name in query:
                    analysis["function_names"].append(func_name)
                    analysis["referenced_functions"].append(func_name)

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

        # Detect exclusion constraints
        if query_excludes_output(query_lower):
            analysis["exclude_output"] = True

        # Detect path/module filters
        analysis["path_filters"] = extract_path_filters_from_query(query, query_lower)

        # Detect novelty requests: "other/different/new functions"
        if any(w in query_lower for w in [
            "other", "another", "different", "new", "remaining", "else",
            "друг", "еще", "ещё", "остальн", "дополнительно"
        ]):
            analysis["exclude_previously_listed"] = True

        # Decide whether positive constraints are all-required or any-of
        positive_signals = 0
        positive_signals += int(analysis["needs_stdin"])
        positive_signals += int(analysis["needs_file"])
        positive_signals += int(analysis["needs_api"])
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

        # Detect example generation requests
        if any(w in query_lower for w in ["example", "пример", "как вызвать", "как использовать", "usage", "использовани"]):
            analysis["needs_example"] = True
            analysis["query_type"] = "example_generation"

        # Detect implementation questions
        if any(w in query_lower for w in ["implement", "реализ", "как работает", "how does", "algorithm", "алгоритм"]):
            analysis["needs_implementation"] = True
            analysis["query_type"] = "implementation_explanation"

        # Check for follow-up questions
        if context_history:
            if any(w in query_lower for w in ["this", "that", "these", "those", "эти", "этот", "такой", "так", "далее", "дальше", "продолж"]):
                analysis["follow_up"] = True
            # Check if referencing previous answer
            if any(w in query_lower for w in ["from the list", "из списка", "функци", "function a", "function b", "function c"]):
                analysis["follow_up"] = True
            if analysis["exclude_previously_listed"]:
                analysis["follow_up"] = True

        # Determine query type
        if analysis["needs_example"]:
            analysis["query_type"] = "example_generation"
        elif analysis["needs_implementation"]:
            analysis["query_type"] = "implementation_explanation"
        elif analysis["is_listing"]:
            analysis["query_type"] = "listing"
        elif analysis["needs_stdin"] or analysis["needs_file"] or analysis["needs_api"]:
            analysis["query_type"] = "input_specific"
        elif analysis["needs_type_info"]:
            analysis["query_type"] = "type_specific"
        elif len(analysis["function_names"]) > 0:
            analysis["query_type"] = "function_specific"

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


def build_prompt(frags, q, analysis=None, conversation_history=None, max_prompt_chars=20000):
    """Main prompt builder with total-character budget."""
    return _build_prompt_impl(
        frags,
        q,
        analysis=analysis,
        conversation_history=conversation_history,
        max_prompt_chars=max_prompt_chars,
    )


def call_llm(prompt, model, temperature=0.1):
    """Call LLM and return structured status dict."""
    return _call_llm_impl(prompt, model, OLLAMA_URL, temperature=temperature)


def verify_answer_with_context(answer, context_frags):
    """Verify that functions mentioned in answer exist in provided context."""
    return _verify_answer_with_context_impl(answer, context_frags)
