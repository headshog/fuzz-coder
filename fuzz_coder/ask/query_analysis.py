from __future__ import annotations

import re
from typing import Dict, List, Tuple

from fuzz_coder.languages.registry import get_query_language_adapter

_EXTRA_QUERY_SYMBOL_BLACKLIST = {
    "data", "line", "value", "values", "path", "file", "module", "directory",
    "input", "output", "request", "response", "process",
}


def _query_adapter(language_name: str | None):
    return get_query_language_adapter(language_name or "c_cpp")


def _qualified_symbol_pattern(language_name: str | None) -> str:
    return _query_adapter(language_name).qualified_symbol_pattern


def _token_looks_like_symbol(token: str, language_name: str | None) -> bool:
    if "_" in token:
        return True
    for sep in _query_adapter(language_name).qualifier_separators:
        if sep and sep in token:
            return True
    return any(ch.isupper() for ch in token[1:]) and any(ch.islower() for ch in token)


def extract_explicit_function_mentions(
    query: str,
    known_symbols_by_lower: Dict[str, List[str]] | None = None,
    query_symbol_blacklist=None,
    language_name: str | None = None,
) -> List[str]:
    """Extract only high-confidence function mentions (code-like/explicit)."""
    known_symbols_by_lower = known_symbols_by_lower or {}
    if query_symbol_blacklist is None:
        query_symbol_blacklist = set(_EXTRA_QUERY_SYMBOL_BLACKLIST)
    found: List[str] = []
    seen = set()

    symbol_pat = _qualified_symbol_pattern(language_name)
    patterns = [
        r"`([^`]+)`",  # explicit backticked symbol
        rf"\b({symbol_pat})\s*(?=\()",  # call-like mention
    ]
    for pat in patterns:
        for m in re.finditer(pat, query):
            token = (m.group(1) or "").strip().strip("`'\".,:;!?()[]{}")
            if not token:
                continue
            token_lower = token.lower()
            if token_lower in query_symbol_blacklist:
                continue
            resolved = known_symbols_by_lower.get(token_lower, [token])
            for fn in resolved:
                if fn not in seen:
                    seen.add(fn)
                    found.append(fn)
    return found


def query_contains_keyword(query_lower: str, keyword: str) -> bool:
    if len(keyword) <= 3 and keyword.isalpha():
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(keyword)}(?![A-Za-z0-9_])"
        return re.search(pattern, query_lower) is not None
    return keyword in query_lower


def query_has_any_keyword(query_lower: str, keywords: List[str]) -> bool:
    return any(query_contains_keyword(query_lower, kw) for kw in keywords)


def normalize_type_name(type_name: str, type_aliases=None) -> str:
    t = type_name.lower().strip()
    aliases = dict(type_aliases or {})
    return aliases.get(t, t)


def extract_requested_types(query_lower: str, type_keywords=None, type_aliases=None) -> List[str]:
    type_keywords = dict(type_keywords or {})
    matched = []
    for type_name, keywords in type_keywords.items():
        if any(query_contains_keyword(query_lower, kw) for kw in keywords):
            matched.append(type_name)

    if "*" in query_lower and "pointer" not in matched:
        matched.append("pointer")
    if "&" in query_lower and "reference" not in matched:
        matched.append("reference")

    return sorted(set(normalize_type_name(t, type_aliases=type_aliases) for t in matched))


def query_excludes_output(query_lower: str, write_like_keywords=None, output_query_keywords=None) -> bool:
    write_like_keywords = list(write_like_keywords or [])
    output_query_keywords = list(output_query_keywords or [])
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
    terms.update([w for w in write_like_keywords if isinstance(w, str)])
    terms.update([w for w in output_query_keywords if isinstance(w, str)])
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


def normalize_path_filter(path: str) -> str:
    p = (path or "").strip().strip("`'\"")
    p = p.replace("\\", "/")
    p = re.sub(r"/{2,}", "/", p)
    while p.startswith("./"):
        p = p[2:]
    p = p.strip().rstrip("/")
    return p.lower()


def clean_path_candidate(raw: str) -> str:
    p = (raw or "").strip()
    if not p:
        return p

    p = re.split(
        r"\s+(?:that|which|where|with|and|or|who|whose|function|functions|method|methods|котор(?:ый|ая|ые|ого|ому|ых)?|где|и|или|с|функц(?:ия|ии|ий|ию|иями)?)\b",
        p,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return p.strip(" \t\r\n.,:;!?")


def extract_path_filters_from_query(query: str, path_filter_patterns=None) -> List[str]:
    path_filter_patterns = list(path_filter_patterns or [])
    filters: List[str] = []
    invalid_filters = {"main", "main()", "function", "functions", "module", "directory"}

    for pattern in path_filter_patterns:
        for m in re.finditer(pattern, query, flags=re.IGNORECASE):
            raw = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
            norm = normalize_path_filter(clean_path_candidate(raw))
            if norm and norm not in invalid_filters:
                filters.append(norm)

    for p in re.findall(r"`([^`]+[/\\][^`]+)`", query):
        norm = normalize_path_filter(p)
        if norm and norm not in invalid_filters:
            filters.append(norm)
    for p in re.findall(r"['\"]([^'\"]+[/\\][^'\"]+)['\"]", query):
        norm = normalize_path_filter(p)
        if norm and norm not in invalid_filters:
            filters.append(norm)

    seen = set()
    out = []
    for f in filters:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def extract_max_param_count(query_lower: str) -> int | None:
    """Extract upper bound for function parameter count from query text."""
    patterns = [
        r"\b(?:at most|no more than|up to|max(?:imum)?|<=|less than or equal to)\s*(\d+)\s*(?:params?|parameters?|args?|arguments?)\b",
        r"\b(\d+)\s*(?:or fewer|or less)\s*(?:params?|parameters?|args?|arguments?)\b",
        r"\b(?:не более|максимум|до)\s*(\d+)\s*(?:параметр(?:а|ов)?|аргумент(?:а|ов)?)\b",
        r"\b(?:параметр(?:ов)?|аргумент(?:ов)?)\s*(?:не более|максимум|до)\s*(\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, query_lower, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if n >= 0:
            return n
    return None


def choose_primary_example_function(
    query: str,
    resolved_function_names: List[str],
    language_name: str | None = None,
) -> str | None:
    if not resolved_function_names:
        return None
    if len(resolved_function_names) == 1:
        return resolved_function_names[0]

    q = query or ""
    symbol_pat = _qualified_symbol_pattern(language_name)
    patterns = [
        rf"\b(?:calling|call|invoke|invoking|using|use)\s+({symbol_pat})\b",
        rf"\b(?:example|пример)\s+(?:of\s+)?({symbol_pat})\b",
        rf"\b(?:пример)\s+(?:вызова|использования)\s+({symbol_pat})\b",
    ]
    lowered_map = {fn.lower(): fn for fn in resolved_function_names}
    for pat in patterns:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            cand = (m.group(1) or "").strip().lower()
            if cand in lowered_map:
                return lowered_map[cand]

    helper_context = set()
    for m in re.finditer(
        rf"\bfrom\s+({symbol_pat})\s+function\b",
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


def extract_function_like_candidates(
    query: str,
    known_symbols_by_lower: Dict[str, List[str]] | None = None,
    common_query_words=None,
    query_symbol_blacklist=None,
    language_name: str | None = None,
) -> List[str]:
    known_symbols_by_lower = known_symbols_by_lower or {}
    if common_query_words is None:
        common_query_words = set()
    if query_symbol_blacklist is None:
        query_symbol_blacklist = set(_EXTRA_QUERY_SYMBOL_BLACKLIST)

    symbol_pat = _qualified_symbol_pattern(language_name)
    collected: List[Tuple[str, str, int]] = []
    order = 0
    for t in re.findall(r"`([^`]+)`", query):
        collected.append((t, "explicit", order))
        order += 1
    for t in re.findall(rf"\b({symbol_pat})\s*(?=\()", query):
        collected.append((t, "call_like", order))
        order += 1
    phrase_patterns = [
        rf"\b(?:of|for|from|in|using|use|invoke|invoking|calling|call)\s+({symbol_pat})\s+(?:function|method)\b",
        rf"\b({symbol_pat})\s+(?:function|method)\b",
    ]
    for pat in phrase_patterns:
        for t in re.findall(pat, query, flags=re.IGNORECASE):
            collected.append((t, "phrase", order))
            order += 1
    for t in re.findall(rf"\b({symbol_pat})\b", query):
        collected.append((t, "token", order))
        order += 1

    source_rank = {
        "explicit": 0,
        "call_like": 1,
        "phrase": 2,
        "token": 3,
    }

    best_by_token: Dict[str, Tuple[int, int, str]] = {}
    for raw, source, idx in collected:
        token = (raw or "").strip().strip("`'\".,:;!?()[]{}")
        if not token:
            continue
        token_lower = token.lower()
        known_symbol_match = token_lower in known_symbols_by_lower
        looks_like_symbol = _token_looks_like_symbol(token, language_name)

        if source == "token":
            if token_lower in query_symbol_blacklist:
                continue
            if token_lower in common_query_words and token_lower != "main":
                continue
            if len(token) < 3 and not known_symbol_match:
                continue
            if not (looks_like_symbol or known_symbol_match):
                continue
        elif source == "phrase":
            # Phrase captures are useful, but we keep only likely symbols.
            if token_lower in query_symbol_blacklist and token_lower != "main":
                continue
            if not (looks_like_symbol or known_symbol_match):
                continue
        elif source in {"explicit", "call_like"}:
            # User provided an explicit symbol-like mention. Keep it even if it
            # overlaps with common query words (e.g. function named "write").
            pass

        cur = best_by_token.get(token_lower)
        rank = source_rank[source]
        if cur is None or rank < cur[0] or (rank == cur[0] and idx < cur[1]):
            best_by_token[token_lower] = (rank, idx, token)

    chosen = sorted(best_by_token.values(), key=lambda x: (x[0], x[1]))
    return [x[2] for x in chosen]


def _base_analysis(query_lower: str) -> dict:
    return {
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


def analyze_query(
    query: str,
    context_history=None,
    symbols_by_lower: Dict[str, List[str]] | None = None,
    language_name: str | None = None,
) -> dict:
    query_lower = (query or "").lower()
    analysis = _base_analysis(query_lower)
    symbols_by_lower = symbols_by_lower or {}
    adapter = _query_adapter(language_name)
    type_keywords = dict(adapter.type_keywords)
    type_aliases = dict(adapter.type_aliases)
    fuzz_query_keywords = list(adapter.fuzz_query_keywords)
    parse_like_keywords = list(adapter.parse_like_keywords)
    stdin_query_keywords = list(adapter.stdin_query_keywords)
    file_query_keywords = list(adapter.file_query_keywords)
    api_query_keywords = list(adapter.api_query_keywords)
    output_query_keywords = list(adapter.output_query_keywords)
    memory_query_keywords = list(adapter.memory_query_keywords)
    error_query_keywords = list(adapter.error_query_keywords)
    params_query_keywords = list(adapter.params_query_keywords)
    path_filter_patterns = list(adapter.path_filter_patterns)
    common_query_words = set(adapter.common_query_words)
    query_symbol_blacklist = set(adapter.query_symbol_blacklist) | _EXTRA_QUERY_SYMBOL_BLACKLIST
    is_example_request = any(
        w in query_lower
        for w in ["example", "пример", "как вызвать", "как использовать", "usage", "использовани"]
    )

    if any(w in query_lower for w in stdin_query_keywords):
        analysis["needs_stdin"] = True
    if any(w in query_lower for w in file_query_keywords):
        analysis["needs_file"] = True
    if any(w in query_lower for w in api_query_keywords):
        analysis["needs_api"] = True

    if any(w in query_lower for w in output_query_keywords) or re.search(r"\bwrite(s|d|ing)?\s+(to|into)\b", query_lower):
        analysis["needs_output"] = True

    if any(w in query_lower for w in memory_query_keywords):
        analysis["needs_memory_mgmt"] = True

    if any(w in query_lower for w in error_query_keywords):
        analysis["needs_error_handling"] = True

    if any(w in query_lower for w in params_query_keywords):
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
        if analysis["function_names"] and not analysis.get("primary_function_name"):
            analysis["primary_function_name"] = analysis["function_names"][0]

    analysis["requested_types"] = extract_requested_types(
        query_lower,
        type_keywords=type_keywords,
        type_aliases=type_aliases,
    )
    if analysis["requested_types"]:
        analysis["needs_type_info"] = True
        analysis["needs_types"] = True

    if query_has_any_keyword(query_lower, parse_like_keywords):
        analysis["needs_parse_like"] = True

    if query_has_any_keyword(query_lower, fuzz_query_keywords):
        analysis["needs_fuzz_targets"] = True
        analysis["is_listing"] = True

    analysis["query_function_candidates"] = extract_function_like_candidates(
        query,
        known_symbols_by_lower=symbols_by_lower,
        common_query_words=common_query_words,
        query_symbol_blacklist=query_symbol_blacklist,
        language_name=language_name,
    )
    for cand in analysis["query_function_candidates"]:
        for resolved in symbols_by_lower.get(cand.lower(), []):
            if resolved not in analysis["function_names"]:
                analysis["function_names"].append(resolved)
                analysis["referenced_functions"].append(resolved)

    # Alias-friendly behavior: in prompts like
    # "define a standalone main() and call X from it", "main" is snippet context,
    # not a target function to retrieve from codebase.
    if ("standalone main" in query_lower or "define a standalone main" in query_lower) and len(analysis["function_names"]) > 1:
        analysis["function_names"] = [fn for fn in analysis["function_names"] if fn.lower() != "main"]
        analysis["referenced_functions"] = [fn for fn in analysis["referenced_functions"] if fn.lower() != "main"]

    # More conservative call-graph intent detection: avoid matching generic
    # "used for fuzzing" phrases as call-expansion signal.
    if any(w in query_lower for w in ["caller", "callee", "called by", "кто вызыва"]):
        analysis["expand_callers"] = True
    elif any(w in query_lower for w in ["invoked by", "calls ", "вызывает", "где вызывается"]):
        analysis["expand_callees"] = True

    if any(w in query_lower for w in ["list", "enumerate", "show all", "find all", "which functions", "какие функции", "перечисли", "покажи все"]):
        analysis["is_listing"] = True
        analysis["query_type"] = "listing"

    explicit_max_params = extract_max_param_count(query_lower)
    if explicit_max_params is not None:
        analysis["max_param_count"] = explicit_max_params

    if query_excludes_output(
        query_lower,
        write_like_keywords=adapter.write_like_keywords,
        output_query_keywords=output_query_keywords,
    ):
        analysis["exclude_output"] = True
        analysis["needs_output"] = False

    analysis["path_filters"] = extract_path_filters_from_query(
        query,
        path_filter_patterns=path_filter_patterns,
    )

    novelty_requested = any(w in query_lower for w in [
        "other", "another", "different", "new", "remaining", "else",
        "друг", "еще", "ещё", "остальн", "дополнительно",
    ])
    if novelty_requested and (analysis["is_listing"] or "function" in query_lower or "функц" in query_lower):
        analysis["exclude_previously_listed"] = True

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
            analysis["constraint_mode"] = "any"

    # Broad fuzz-list mode:
    # - enforce compact signatures (<= 4 params unless user specified another bound)
    # - require at least one practical fuzz-input surface:
    #   parse-like OR stdin/file source OR path/file-handle/simple-pointer/vector-like params
    if analysis["needs_fuzz_targets"] and analysis["is_listing"] and not is_example_request:
        analysis["needs_broad_fuzz_surface"] = True
        if analysis["max_param_count"] is None:
            analysis["max_param_count"] = 4
        analysis["constraint_mode"] = "any"
        if not analysis.get("listing_target_count"):
            analysis["listing_target_count"] = 25

        # For broad listing aliases/prompts, token extraction may pick common
        # words that coincide with symbol names (e.g. parse/split/array). Keep
        # only explicit code-like mentions, otherwise do not force symbol focus.
        explicit_mentions = extract_explicit_function_mentions(
            query,
            known_symbols_by_lower=symbols_by_lower,
            query_symbol_blacklist=query_symbol_blacklist,
            language_name=language_name,
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

    if is_example_request:
        analysis["needs_example"] = True
        analysis["query_type"] = "example_generation"
        analysis["is_listing"] = False
        analysis["exclude_previously_listed"] = False
        if analysis["function_names"]:
            analysis["needs_fuzz_targets"] = False
            analysis["primary_function_name"] = choose_primary_example_function(
                query,
                analysis["function_names"],
                language_name=language_name,
            )

    if any(w in query_lower for w in ["implement", "реализ", "как работает", "how does", "algorithm", "алгоритм"]):
        analysis["needs_implementation"] = True
        analysis["query_type"] = "implementation_explanation"

    if context_history:
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

    if analysis["needs_example"]:
        analysis["query_type"] = "example_generation"
    elif analysis["needs_implementation"]:
        analysis["query_type"] = "implementation_explanation"
    elif analysis["needs_param_semantics"] and len(analysis["function_names"]) > 0:
        analysis["query_type"] = "parameter_analysis"
        analysis["is_listing"] = False
        # Parameter semantics asks are not "file-input" or "output operation" filters.
        # Alias phrasing may contain words like "input/output" and "file:line" as schema hints.
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

