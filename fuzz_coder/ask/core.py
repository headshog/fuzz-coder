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
from .query_analysis import analyze_query as _analyze_query_impl
from .example_context import build_example_context as _build_example_context_impl
from .example_grounding import build_example_context_grounded as _build_example_context_grounded_impl
from .fallback_builders import build_example_answer_from_context as _build_example_answer_from_context_impl
from .fallback_builders import build_parameter_analysis_from_context as _build_parameter_analysis_from_context_impl
from .verification import verify_answer_with_context as _verify_answer_with_context_impl
from .verification import verify_example_answer_with_context as _verify_example_answer_with_context_impl
from fuzz_coder.languages.registry import get_query_language_adapter

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_QUERY_LANGUAGE = get_query_language_adapter("").name

TYPE_KEYWORDS = {}
TYPE_ALIASES = {}
TYPE_INDEX_ALIASES = {}
TYPE_PARAM_HINTS = {}
STDIN_QUERY_KEYWORDS = []
FILE_QUERY_KEYWORDS = []
API_QUERY_KEYWORDS = []
OUTPUT_QUERY_KEYWORDS = []
MEMORY_QUERY_KEYWORDS = []
ERROR_QUERY_KEYWORDS = []
PARAMS_QUERY_KEYWORDS = []
WRITE_LIKE_KEYWORDS = []
PARSE_LIKE_KEYWORDS = []
FUZZ_QUERY_KEYWORDS = []
FUZZ_TARGET_KEYWORDS = []
PATH_LIKE_TYPE_HINTS = []
SIMPLE_POINTER_TYPE_HINTS = []
FILE_HANDLE_TYPE_HINTS = []
PATH_PARAM_NAME_HINTS = []
VECTOR_LIKE_TYPE_HINTS = []
PATH_FILTER_PATTERNS = []
COMMON_QUERY_WORDS = set()
QUERY_SYMBOL_BLACKLIST = set()
FUZZ_PARAM_HINT_TOKENS = []
_CURRENT_QUERY_LANGUAGE = DEFAULT_QUERY_LANGUAGE


def _apply_query_language(language_name=None):
    global TYPE_KEYWORDS
    global TYPE_ALIASES
    global TYPE_INDEX_ALIASES
    global TYPE_PARAM_HINTS
    global STDIN_QUERY_KEYWORDS
    global FILE_QUERY_KEYWORDS
    global API_QUERY_KEYWORDS
    global OUTPUT_QUERY_KEYWORDS
    global MEMORY_QUERY_KEYWORDS
    global ERROR_QUERY_KEYWORDS
    global PARAMS_QUERY_KEYWORDS
    global WRITE_LIKE_KEYWORDS
    global PARSE_LIKE_KEYWORDS
    global FUZZ_QUERY_KEYWORDS
    global FUZZ_TARGET_KEYWORDS
    global PATH_LIKE_TYPE_HINTS
    global SIMPLE_POINTER_TYPE_HINTS
    global FILE_HANDLE_TYPE_HINTS
    global PATH_PARAM_NAME_HINTS
    global VECTOR_LIKE_TYPE_HINTS
    global PATH_FILTER_PATTERNS
    global COMMON_QUERY_WORDS
    global QUERY_SYMBOL_BLACKLIST
    global FUZZ_PARAM_HINT_TOKENS

    adapter = get_query_language_adapter(language_name or DEFAULT_QUERY_LANGUAGE)
    TYPE_KEYWORDS = dict(adapter.type_keywords)
    TYPE_ALIASES = dict(adapter.type_aliases)
    TYPE_INDEX_ALIASES = dict(adapter.type_index_aliases)
    TYPE_PARAM_HINTS = dict(adapter.type_param_hints)
    STDIN_QUERY_KEYWORDS = list(adapter.stdin_query_keywords)
    FILE_QUERY_KEYWORDS = list(adapter.file_query_keywords)
    API_QUERY_KEYWORDS = list(adapter.api_query_keywords)
    OUTPUT_QUERY_KEYWORDS = list(adapter.output_query_keywords)
    MEMORY_QUERY_KEYWORDS = list(adapter.memory_query_keywords)
    ERROR_QUERY_KEYWORDS = list(adapter.error_query_keywords)
    PARAMS_QUERY_KEYWORDS = list(adapter.params_query_keywords)
    WRITE_LIKE_KEYWORDS = list(adapter.write_like_keywords)
    PARSE_LIKE_KEYWORDS = list(adapter.parse_like_keywords)
    FUZZ_QUERY_KEYWORDS = list(adapter.fuzz_query_keywords)
    FUZZ_TARGET_KEYWORDS = list(adapter.fuzz_target_keywords)
    PATH_LIKE_TYPE_HINTS = list(adapter.path_like_type_hints)
    SIMPLE_POINTER_TYPE_HINTS = list(adapter.simple_pointer_type_hints)
    FILE_HANDLE_TYPE_HINTS = list(adapter.file_handle_type_hints)
    PATH_PARAM_NAME_HINTS = list(adapter.path_param_name_hints)
    VECTOR_LIKE_TYPE_HINTS = list(adapter.vector_like_type_hints)
    PATH_FILTER_PATTERNS = list(adapter.path_filter_patterns)
    COMMON_QUERY_WORDS = set(adapter.common_query_words)
    QUERY_SYMBOL_BLACKLIST = set(adapter.query_symbol_blacklist)
    FUZZ_PARAM_HINT_TOKENS = sorted({
        str(tok).lower()
        for values in TYPE_PARAM_HINTS.values()
        for tok in (values or [])
        if isinstance(tok, str) and str(tok).strip()
    })
    return adapter.name


def set_query_language(language_name=None):
    global _CURRENT_QUERY_LANGUAGE
    _CURRENT_QUERY_LANGUAGE = _apply_query_language(language_name)
    return _CURRENT_QUERY_LANGUAGE


_apply_query_language(DEFAULT_QUERY_LANGUAGE)


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

    neg_markers = ["not", "without", "exclude", "except", "excluding", "не", "без", "кроме", "исключая"]
    terms = set()
    terms.update([w for w in WRITE_LIKE_KEYWORDS if isinstance(w, str)])
    terms.update([w for w in OUTPUT_QUERY_KEYWORDS if isinstance(w, str)])
    terms.update(["write", "output", "print", "log", "serialize", "запис", "вывод", "печат", "лог"])
    terms = {t.strip().lower() for t in terms if t and len(t.strip()) >= 3}
    term_parts = [re.escape(t).replace(r"\ ", r"\s+") for t in sorted(terms, key=len, reverse=True)[:60]]
    if not term_parts:
        return False
    term_pat = "(?:" + "|".join(term_parts) + ")"
    neg_pat = (
        r"\b(?:"
        + "|".join(neg_markers)
        + r")\b[^.\n]{0,80}"
        + term_pat
    )
    return re.search(neg_pat, query_lower, flags=re.IGNORECASE) is not None


def parameter_matches_type(param, type_name):
    """Check if parameter likely matches a canonical type."""
    p_raw = f"{param.get('raw', '')} {param.get('type', '')}".lower()
    type_name = normalize_type_name(type_name)

    if type_name == "pointer":
        return "*" in p_raw
    if type_name == "reference":
        return "&" in p_raw

    return any(h in p_raw for h in TYPE_PARAM_HINTS.get(type_name, []))


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
        has_path_type = any(t in txt for t in PATH_LIKE_TYPE_HINTS)
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
        if any(token in p_text for token in FUZZ_PARAM_HINT_TOKENS):
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
    type_query_hints = []
    for t in ["byte_array", "template", "string"]:
        type_query_hints.extend(TYPE_KEYWORDS.get(t, []))
    param_keywords = list(PARAMS_QUERY_KEYWORDS) + list(type_query_hints)
    has_param_query = any(kw in query.lower() for kw in param_keywords)

    if not has_param_query:
        return 0.5  # Neutral if query doesn't care about params

    if chunk.get("parameters") and len(chunk["parameters"]) > 0:
        # Check if parameter types match query intent
        byteish_hints = (
            TYPE_PARAM_HINTS.get("byte_array", [])
            + TYPE_PARAM_HINTS.get("template", [])
            + TYPE_PARAM_HINTS.get("string", [])
        )
        for p in chunk["parameters"]:
            p_type = p["type"].lower()
            p_name = p["name"].lower()
            if any(kw in p_type or kw in p_name for kw in byteish_hints):
                return 1.0
        return 0.7  # Has params but not exact match
    return 0.0


def input_type_match_score(query, chunk):
    """Score based on input type matching"""
    query_lower = query.lower()

    # Check what input types the query is asking about
    wants_stdin = any(w in query_lower for w in STDIN_QUERY_KEYWORDS)
    wants_file = any(w in query_lower for w in FILE_QUERY_KEYWORDS)
    wants_api = any(w in query_lower for w in API_QUERY_KEYWORDS)
    wants_param = any(w in query_lower for w in PARAMS_QUERY_KEYWORDS + ["pass"])

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

def build_example_answer_from_context(frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
    return _build_example_answer_from_context_impl(
        frags,
        analysis=analysis,
        example_context=example_context,
        symbols=symbols,
        type_init_index=type_init_index,
    )


def build_parameter_analysis_from_context(frags, analysis=None, example_context=None, function_hints=None):
    return _build_parameter_analysis_from_context_impl(
        frags,
        analysis=analysis,
        example_context=example_context,
        function_hints=function_hints,
    )



class QueryPlanner:
    """Advanced query planner with thinking mode support"""

    def __init__(self, special_indices, symbols, call_graph, called_by, meta=None, language_name=None):
        self.special_indices = special_indices
        self.symbols = symbols
        self.call_graph = call_graph
        self.called_by = called_by
        self.meta = meta or []
        self.language_name = language_name or DEFAULT_QUERY_LANGUAGE
        set_query_language(self.language_name)
        self.symbols_by_lower = defaultdict(list)
        for fn in self.symbols.keys():
            self.symbols_by_lower[fn.lower()].append(fn)

    def analyze_query(self, query, context_history=None):
        """Analyze query using the single maintained query analyzer."""
        return _analyze_query_impl(
            query,
            context_history=context_history,
            symbols_by_lower=self.symbols_by_lower,
            language_name=self.language_name,
        )

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


def build_example_context(frags, analysis=None, language_name=None):
    """Build structured context facts for example-generation."""
    return _build_example_context_impl(frags, analysis=analysis, language_name=language_name)


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
    language_name=None,
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
        language_name=language_name or DEFAULT_QUERY_LANGUAGE,
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
    type_init_index=None,
):
    """Verify example-generation answer against file/signature/line evidence and context."""
    return _verify_example_answer_with_context_impl(
        answer,
        context_frags,
        target_function=target_function,
        known_functions=known_functions,
        example_context=example_context,
        type_init_index=type_init_index,
    )
