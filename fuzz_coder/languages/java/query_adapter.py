from __future__ import annotations

from ..query_adapters import (
    COMMON_FUZZ_QUERY_KEYWORDS,
    COMMON_FUZZ_TARGET_KEYWORDS,
    COMMON_PATH_FILTER_PATTERNS,
    COMMON_QUERY_SYMBOL_BLACKLIST,
    COMMON_QUERY_WORDS,
    QueryLanguageAdapter,
)


JAVA_QUERY_ADAPTER = QueryLanguageAdapter(
    name="java",
    qualified_symbol_pattern=r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*",
    qualifier_separators=["."],
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
