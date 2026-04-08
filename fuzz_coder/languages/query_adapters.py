from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class QueryLanguageAdapter:
    name: str
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

C_CPP_QUERY_ADAPTER = QueryLanguageAdapter(
    name="c_cpp",
    type_keywords={
        "byte_array": ["byte", "uint8", "char*", "buffer", "массив байт", "байт", "bytes", "qbytearray"],
        "string": ["string", "str", "char[]", "строка", "строку", "cstring", "wstring"],
        "integer": ["int", "integer", "число", "int32", "int64", "цел", "size_t", "ssize_t"],
        "float": ["float", "double", "веществен", "floating", "плавающ"],
        "template": ["vector", "array", "список", "массив", "std::vector", "template", "std::array"],
        "pointer": ["pointer", "указатель"],
        "reference": ["reference", "ссылка"],
    },
    type_aliases={
        "int": "integer",
        "vector": "template",
    },
    type_index_aliases={
        "integer": ["int"],
        "template": ["vector"],
    },
    type_param_hints={
        "byte_array": ["uint8", "unsigned char", "char *", "byte", "qbytearray", "vector<uint8"],
        "string": ["std::string", "string", "char *", "const char *", "qstring", "wstring", "char[]"],
        "integer": ["int", "long", "short", "int32", "int64", "size_t", "ssize_t", "uint32", "uint64"],
        "float": ["float", "double", "long double"],
        "template": ["std::vector", "vector<", "std::array", "array<", "std::map", "map<", "std::set", "set<", "unordered_map"],
    },
    stdin_query_keywords=["stdin", "standard input", "console", "cin", "scanf", "getchar", "fgets", "ввод"],
    file_query_keywords=["file", "fopen", "ifstream", "fstream", "read from file", "файл"],
    api_query_keywords=["api", "http", "request", "network", "curl", "socket", "сеть"],
    output_query_keywords=[
        "stdout", "stderr", "output", "print", "printf", "fprintf", "cout", "cerr",
        "clog", "logging", "logger", "log ", "log-", "вывод", "печать", "логг",
    ],
    memory_query_keywords=[
        "memory", "buffer", "malloc", "calloc", "realloc", "free",
        "new/delete", "memcpy", "memmove", "heap", "stack",
        "памят", "буфер", "переполн",
    ],
    error_query_keywords=[
        "error handling", "error", "errors", "exception", "exceptions",
        "throw", "catch", "assert", "errno", "validation",
        "ошиб", "исключен", "валидац",
    ],
    params_query_keywords=["param", "argument", "arg", "input", "receive", "accept", "take", "переда", "вход", "параметр"],
    write_like_keywords=[
        "write", "output", "print", "printf", "fprintf", "cout", "cerr", "clog",
        "log", "dump", "serialize", "save", "emit", "flush", "store",
        "запис", "вывод", "печат", "лог",
    ],
    parse_like_keywords=[
        "parse", "parser", "token", "tokenize", "split", "decode", "deserialize",
        "scan", "lex", "grammar", "peg", "readline", "from_string", "parse_",
        "парс", "разбор",
    ],
    fuzz_query_keywords=list(COMMON_FUZZ_QUERY_KEYWORDS),
    fuzz_target_keywords=list(COMMON_FUZZ_TARGET_KEYWORDS),
    path_like_type_hints=[
        "std::string", "string", "char *", "const char *", "filesystem::path", "path",
    ],
    simple_pointer_type_hints=[
        "char", "signed char", "unsigned char",
        "short", "unsigned short",
        "int", "unsigned int",
        "long", "unsigned long",
        "long long", "unsigned long long",
        "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t",
        "size_t", "ssize_t",
        "float", "double", "bool",
    ],
    file_handle_type_hints=[
        "file *", "file*", "std::file",
        "ifstream", "ofstream", "fstream",
        "istream", "ostream", "stream &", "stream&",
    ],
    path_param_name_hints=[
        "path", "file", "filename", "fname", "filepath", "dir", "directory",
    ],
    vector_like_type_hints=[
        "std::vector<", "vector<",
        "std::array<", "array<",
        "std::span<", "span<",
    ],
    path_filter_patterns=list(COMMON_PATH_FILTER_PATTERNS),
    common_query_words=set(COMMON_QUERY_WORDS),
    query_symbol_blacklist=set(COMMON_QUERY_SYMBOL_BLACKLIST),
)

JAVA_QUERY_ADAPTER = QueryLanguageAdapter(
    name="java",
    type_keywords={
        "byte_array": ["byte", "byte[]", "buffer", "bytes", "bytebuffer"],
        "string": ["string", "charsequence", "строка", "текст"],
        "integer": ["int", "integer", "long", "short", "size", "length", "число"],
        "float": ["float", "double", "decimal", "веществен"],
        "template": ["list", "arraylist", "map", "set", "collection", "массив", "список"],
        "pointer": ["reference", "object"],
        "reference": ["reference", "ref", "ссылка"],
    },
    type_aliases={
        "int": "integer",
        "list": "template",
        "array": "template",
    },
    type_index_aliases={
        "integer": ["int"],
        "template": ["list", "array"],
    },
    type_param_hints={
        "byte_array": ["byte[]", "bytebuffer", "buffer", "bytes"],
        "string": ["string", "charsequence", "char[]", "stringbuilder"],
        "integer": ["int", "integer", "long", "short", "size", "length"],
        "float": ["float", "double", "bigdecimal"],
        "template": ["list<", "arraylist<", "map<", "set<", "collection<", "stream<"],
    },
    stdin_query_keywords=["stdin", "standard input", "console", "system.in", "scanner", "bufferedreader", "ввод"],
    file_query_keywords=["file", "path", "files.", "fileinputstream", "bufferedreader", "read from file", "файл"],
    api_query_keywords=["api", "http", "request", "network", "urlconnection", "socket", "сеть"],
    output_query_keywords=[
        "stdout", "stderr", "output", "print", "println", "printf", "logger", "log",
        "write", "вывод", "печать", "логг",
    ],
    memory_query_keywords=[
        "memory", "buffer", "bytebuffer", "arraycopy", "heap", "stack", "allocate", "alloc",
        "памят", "буфер",
    ],
    error_query_keywords=[
        "error handling", "error", "errors", "exception", "exceptions",
        "throw", "catch", "assert", "validation",
        "ошиб", "исключен", "валидац",
    ],
    params_query_keywords=["param", "parameter", "argument", "arg", "input", "receive", "accept", "take", "вход", "параметр"],
    write_like_keywords=[
        "write", "output", "print", "println", "printf", "log", "logger",
        "save", "serialize", "emit", "flush", "store", "запис", "вывод", "лог",
    ],
    parse_like_keywords=[
        "parse", "parser", "token", "tokenize", "split", "decode", "deserialize",
        "scan", "grammar", "readline", "parse_", "парс", "разбор",
    ],
    fuzz_query_keywords=list(COMMON_FUZZ_QUERY_KEYWORDS),
    fuzz_target_keywords=list(COMMON_FUZZ_TARGET_KEYWORDS),
    path_like_type_hints=[
        "string", "path", "file", "filepath", "uri", "charsequence",
    ],
    simple_pointer_type_hints=[
        "byte[]", "int[]", "long[]", "short[]", "float[]", "double[]", "char[]", "boolean[]",
        "bytebuffer",
    ],
    file_handle_type_hints=[
        "inputstream", "outputstream", "reader", "writer", "bufferedreader", "file", "path",
    ],
    path_param_name_hints=[
        "path", "file", "filename", "filepath", "dir", "directory",
    ],
    vector_like_type_hints=[
        "list<", "arraylist<", "linkedlist<", "set<", "map<", "collection<", "stream<",
    ],
    path_filter_patterns=list(COMMON_PATH_FILTER_PATTERNS),
    common_query_words=set(COMMON_QUERY_WORDS),
    query_symbol_blacklist=set(COMMON_QUERY_SYMBOL_BLACKLIST),
)


_QUERY_ADAPTERS = {
    "c_cpp": C_CPP_QUERY_ADAPTER,
    "java": JAVA_QUERY_ADAPTER,
}


def get_query_language_adapter(language_name: str) -> QueryLanguageAdapter:
    return _QUERY_ADAPTERS.get(language_name or "", C_CPP_QUERY_ADAPTER)
