from __future__ import annotations

import re
from typing import Dict, List, Tuple


TYPE_KEYWORDS = {
    "byte_array": ["byte", "uint8", "char*", "buffer", "массив байт", "байт", "bytes", "qbytearray"],
    "string": ["string", "str", "char[]", "строка", "строку", "cstring", "wstring"],
    "integer": ["int", "integer", "число", "int32", "int64", "цел", "size_t", "ssize_t"],
    "float": ["float", "double", "веществен", "floating", "плавающ"],
    "template": ["vector", "array", "список", "массив", "std::vector", "template", "std::array"],
    "pointer": ["pointer", "указатель"],
    "reference": ["reference", "ссылка"],
}

TYPE_ALIASES = {
    "int": "integer",
    "vector": "template",
}

FUZZ_QUERY_KEYWORDS = [
    "fuzz", "fuzzer", "fuzzing", "libfuzzer", "afl", "afl++", "honggfuzz",
    "oss-fuzz", "mutation", "coverage-guided", "asan", "ubsan", "sanitizer",
    "фазз", "фаззинг", "фузз", "фуззинг",
]

PATH_FILTER_PATTERNS = [
    r"\b(?:from|in)\s+(?:the\s+)?(?:module|directory|subdirectory|folder|path)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1",
    r"\b(?:from|in)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1\s+(?:module|directory|subdirectory|folder|path)\b",
    r"\bunder\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1",
    r"\bиз\s+(?:модуля|директории|поддиректории|папки)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1",
    r"\bв\s+(?:модуле|директории|поддиректории|папке)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1",
    r"\b(?:из|в)\s+([`\"']?)([A-Za-z0-9_./\\:-]+)\1\s+(?:модуле|модуля|директории|поддиректории|папке)\b",
]

COMMON_QUERY_WORDS = {
    "write", "list", "of", "functions", "that", "can", "be", "used", "for", "fuzzing",
    "give", "an", "example", "from", "main", "function", "called", "call", "how",
    "to", "is", "in", "codebase", "show", "me", "the", "a", "and", "or", "with",
}

QUERY_SYMBOL_BLACKLIST = {
    "write", "list", "show", "find", "give", "make", "need", "example",
    "function", "functions", "called", "calling", "from", "for", "with",
    "that", "this", "these", "those", "can", "used", "use", "is", "are",
    "of", "in", "on", "to", "an", "a", "the",
    # High-frequency generic nouns/verbs that often appear in prompts and
    # should not become target symbols unless explicitly code-like.
    "data", "line", "value", "values", "path", "file", "module", "directory",
    "input", "output", "request", "response", "process", "format", "context", "range", "role",
    "construct", "constructed", "build", "built", "define", "generated", "snippet",
}


def extract_explicit_function_mentions(
    query: str,
    known_symbols_by_lower: Dict[str, List[str]] | None = None,
) -> List[str]:
    """Extract only high-confidence function mentions (code-like/explicit)."""
    known_symbols_by_lower = known_symbols_by_lower or {}
    found: List[str] = []
    seen = set()

    patterns = [
        r"`([^`]+)`",  # explicit backticked symbol
        r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*(?=\()",  # call-like mention
    ]
    for pat in patterns:
        for m in re.finditer(pat, query):
            token = (m.group(1) or "").strip().strip("`'\".,:;!?()[]{}")
            if not token:
                continue
            token_lower = token.lower()
            if token_lower in QUERY_SYMBOL_BLACKLIST:
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


def normalize_type_name(type_name: str) -> str:
    t = type_name.lower().strip()
    return TYPE_ALIASES.get(t, t)


def extract_requested_types(query_lower: str) -> List[str]:
    matched = []
    for type_name, keywords in TYPE_KEYWORDS.items():
        if any(query_contains_keyword(query_lower, kw) for kw in keywords):
            matched.append(type_name)

    if "*" in query_lower and "pointer" not in matched:
        matched.append("pointer")
    if "&" in query_lower and "reference" not in matched:
        matched.append("reference")

    return sorted(set(normalize_type_name(t) for t in matched))


def query_excludes_output(query_lower: str) -> bool:
    explicit_phrases = [
        "not write", "not write-like", "but not write", "without write",
        "exclude write", "except write", "not output", "exclude output",
        "without output", "не запис", "не вывод", "без вывода", "кроме write",
        "кроме вывода", "excluding write",
    ]
    if any(p in query_lower for p in explicit_phrases):
        return True

    neg_en = r"\b(not|without|exclude|except|excluding)\b[^.\n]{0,60}\b(write|output|print|printf|fprintf|cout|log|dump|serialize)\b"
    neg_ru = r"\b(не|без|кроме|исключая)\b[^.\n]{0,60}\b(запис\w*|вывод\w*|печат\w*|лог\w*)\b"
    return re.search(neg_en, query_lower) is not None or re.search(neg_ru, query_lower) is not None


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


def extract_path_filters_from_query(query: str) -> List[str]:
    filters: List[str] = []
    invalid_filters = {"main", "main()", "function", "functions", "module", "directory"}

    for pattern in PATH_FILTER_PATTERNS:
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


def choose_primary_example_function(query: str, resolved_function_names: List[str]) -> str | None:
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


def extract_function_like_candidates_v2(
    query: str,
    known_symbols_by_lower: Dict[str, List[str]] | None = None,
) -> List[str]:
    known_symbols_by_lower = known_symbols_by_lower or {}

    collected: List[Tuple[str, str, int]] = []
    order = 0
    for t in re.findall(r"`([^`]+)`", query):
        collected.append((t, "explicit", order))
        order += 1
    for t in re.findall(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*(?=\()", query):
        collected.append((t, "call_like", order))
        order += 1
    phrase_patterns = [
        r"\b(?:of|for|from|in|using|use|invoke|invoking|calling|call)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+(?:function|method)\b",
        r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+(?:function|method)\b",
    ]
    for pat in phrase_patterns:
        for t in re.findall(pat, query, flags=re.IGNORECASE):
            collected.append((t, "phrase", order))
            order += 1
    for t in re.findall(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\b", query):
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
        looks_like_symbol = (
            "_" in token
            or "::" in token
            or (any(ch.isupper() for ch in token[1:]) and any(ch.islower() for ch in token))
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
        elif source == "phrase":
            # Phrase captures are useful, but we keep only likely symbols.
            if token_lower in QUERY_SYMBOL_BLACKLIST and token_lower != "main":
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


def analyze_query_v2(
    query: str,
    context_history=None,
    symbols_by_lower: Dict[str, List[str]] | None = None,
) -> dict:
    query_lower = (query or "").lower()
    analysis = _base_analysis(query_lower)
    symbols_by_lower = symbols_by_lower or {}

    if any(w in query_lower for w in ["stdin", "standard input", "console", "cin", "scanf", "getchar", "fgets", "ввод"]):
        analysis["needs_stdin"] = True
    if any(w in query_lower for w in ["file", "fopen", "ifstream", "fstream", "read from file", "файл"]):
        analysis["needs_file"] = True
    if any(w in query_lower for w in ["api", "http", "request", "network", "curl", "socket", "сеть"]):
        analysis["needs_api"] = True

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
        if analysis["function_names"] and not analysis.get("primary_function_name"):
            analysis["primary_function_name"] = analysis["function_names"][0]

    analysis["requested_types"] = extract_requested_types(query_lower)
    if analysis["requested_types"]:
        analysis["needs_type_info"] = True
        analysis["needs_types"] = True

    if any(w in query_lower for w in [
        "parse", "parser", "parsing", "tokenize", "split", "decode", "deserialize",
        "scan", "lex", "grammar", "peg", "разбор", "парс",
    ]):
        analysis["needs_parse_like"] = True

    if query_has_any_keyword(query_lower, FUZZ_QUERY_KEYWORDS):
        analysis["needs_fuzz_targets"] = True
        analysis["is_listing"] = True

    analysis["query_function_candidates"] = extract_function_like_candidates_v2(
        query,
        known_symbols_by_lower=symbols_by_lower,
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

    if query_excludes_output(query_lower):
        analysis["exclude_output"] = True
        analysis["needs_output"] = False

    analysis["path_filters"] = extract_path_filters_from_query(query)

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
    if analysis["needs_fuzz_targets"] and analysis["is_listing"]:
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

    if any(w in query_lower for w in ["example", "пример", "как вызвать", "как использовать", "usage", "использовани"]):
        analysis["needs_example"] = True
        analysis["query_type"] = "example_generation"
        analysis["is_listing"] = False
        analysis["exclude_previously_listed"] = False
        if analysis["function_names"]:
            analysis["needs_fuzz_targets"] = False
            analysis["primary_function_name"] = choose_primary_example_function(
                query,
                analysis["function_names"],
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


def compare_analysis(legacy: dict, v2: dict) -> dict:
    """Return lightweight diff between two analysis dicts."""
    keys = sorted(set(legacy.keys()) | set(v2.keys()))
    diff = {}
    for k in keys:
        if legacy.get(k) != v2.get(k):
            diff[k] = {
                "legacy": legacy.get(k),
                "v2": v2.get(k),
            }
    return diff
