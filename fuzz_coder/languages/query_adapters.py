from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class QueryLanguageAdapter:
    name: str
    qualified_symbol_pattern: str
    qualifier_separators: List[str]
    type_keywords: Dict[str, List[str]]
    type_aliases: Dict[str, str]
    type_index_aliases: Dict[str, List[str]]
    type_param_hints: Dict[str, List[str]]
    stdin_query_keywords: List[str]
    file_query_keywords: List[str]
    api_query_keywords: List[str]
    output_query_keywords: List[str]
    memory_query_keywords: List[str]
    error_query_keywords: List[str]
    params_query_keywords: List[str]
    write_like_keywords: List[str]
    parse_like_keywords: List[str]
    fuzz_query_keywords: List[str]
    fuzz_target_keywords: List[str]
    path_like_type_hints: List[str]
    simple_pointer_type_hints: List[str]
    file_handle_type_hints: List[str]
    path_param_name_hints: List[str]
    vector_like_type_hints: List[str]
    path_filter_patterns: List[str]
    common_query_words: Set[str]
    query_symbol_blacklist: Set[str]


COMMON_PATH_FILTER_PATTERNS = [
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

COMMON_QUERY_SYMBOL_BLACKLIST = {
    "write", "list", "show", "find", "give", "make", "need", "example",
    "function", "functions", "called", "calling", "from", "for", "with",
    "that", "this", "these", "those", "can", "used", "use", "is", "are",
    "of", "in", "on", "to", "an", "a", "the",
    "construct", "constructed", "build", "built", "define", "generated", "snippet",
    "format", "context", "range", "role",
}

COMMON_FUZZ_QUERY_KEYWORDS = [
    "fuzz", "fuzzer", "fuzzing", "libfuzzer", "afl", "afl++", "honggfuzz",
    "oss-fuzz", "mutation", "coverage-guided", "asan", "ubsan", "sanitizer",
    "фазз", "фаззинг", "фузз", "фуззинг",
]

COMMON_FUZZ_TARGET_KEYWORDS = [
    "parse", "decode", "deserialize", "token", "grammar", "load", "read",
    "json", "xml", "yaml", "gguf", "tensor", "prompt", "chat", "template",
    "sample", "kv", "buffer", "memcpy", "memmove", "strncpy", "snprintf",
    "base64", "utf8", "utf-8", "binary", "header", "payload",
]


from .c_cpp.query_adapter import C_CPP_QUERY_ADAPTER
from .java.query_adapter import JAVA_QUERY_ADAPTER


_QUERY_ADAPTERS = {
    "c_cpp": C_CPP_QUERY_ADAPTER,
    "java": JAVA_QUERY_ADAPTER,
}


def get_query_language_adapter(language_name: str) -> QueryLanguageAdapter:
    return _QUERY_ADAPTERS.get(language_name or "", C_CPP_QUERY_ADAPTER)
