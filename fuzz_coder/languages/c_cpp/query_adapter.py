from __future__ import annotations

from ..query_adapters import (
    COMMON_FUZZ_QUERY_KEYWORDS,
    COMMON_FUZZ_TARGET_KEYWORDS,
    COMMON_PATH_FILTER_PATTERNS,
    COMMON_QUERY_SYMBOL_BLACKLIST,
    COMMON_QUERY_WORDS,
    QueryLanguageAdapter,
)


C_CPP_QUERY_ADAPTER = QueryLanguageAdapter(
    name="c_cpp",
    qualified_symbol_pattern=r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*",
    qualifier_separators=["::"],
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

