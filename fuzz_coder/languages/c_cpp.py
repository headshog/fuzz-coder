from __future__ import annotations

from typing import Any, Dict, List, Set

from .base import LanguageFrontend, LanguageProfile


C_CPP_PROFILE = LanguageProfile(
    name="c_cpp",
    supported_ext={".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"},
    control_keywords={
        "if", "for", "while", "switch", "return", "sizeof", "catch",
        "new", "delete", "throw", "else", "do", "class", "struct",
        "namespace", "template", "typedef", "using", "enum", "union"
    },
    stdin_patterns=[
        r"\b(cin)\s*>>",
        r"\b(scanf|getchar|getc|gets|gets_s)\s*\(",
        r"\b(fgets|fscanf)\s*\([^)]*(stdin)\b",
        r"\bstd::getline\s*\(\s*(std::)?cin\b",
        r"\bgetline\s*\(\s*(std::)?cin\b",
        r"\bread\s*\(\s*(0|STDIN_FILENO)\b",
        r"\b(std::)?cin\b",
        r"\b(System\.Console\.Read)\b",
        r"\b(Console\.Read|ReadLine|ReadKey)\b",
        r"\b(input|raw_input)\s*\(",
        r"\b(sys\.stdin|process\.stdin)\b",
    ],
    file_input_patterns=[
        r"\bf(open|fopen|ifstream|fstream)\s*\(",
        r"\b(fread|fgets|fscanf|fgetc|getc)\s*\(",
        r"\b(std::)?(ifstream|fstream|ofstream)\b",
        r"\b(File\.Open|File\.Read|StreamReader)\b",
        r"\b(fs\.readFileSync?|fs\.createReadStream)\b",
        r"\b(Path\.OpenText|File\.ReadAllText)\b",
    ],
    api_call_patterns=[
        r"\b(http_client|HttpClient|curl_easy|wget)\b",
        r"\b(requests\.(get|post|put|delete|patch))\b",
        r"\b(fetch|axios|XMLHttpRequest)\b",
        r"\b(urllib\.(request|urlopen))\b",
        r"\b(httplib::Client|boost::beast)\b",
    ],
    output_patterns=[
        r"\b(cout|printf|fprintf|sprintf)\b",
        r"\b(std::)?(cout|cerr|clog|print|writeln)\b",
        r"\b(Console\.Write|System\.out)\b",
    ],
    memory_management_patterns=[
        r"\b(malloc|calloc|realloc|free)\b",
        r"\b(new|delete)\b",
        r"\b(shared_ptr|unique_ptr|weak_ptr)\b",
    ],
    error_handling_patterns=[
        r"\b(throw|try|catch|finally)\b",
        r"\b(errno|perror|strerror)\b",
        r"\b(assert|static_assert)\b",
    ],
    type_patterns={
        "byte_array": [r"\b(uint8_t|unsigned\s+char|char\s*\*|std::vector<uint8_t>|QByteArray|ByteBuffer)\b"],
        "string": [r"\b(std::string|char\s*\*|const\s+char\s*\*|QString|std::wstring)\b"],
        "integer": [r"\b(int|long|short|int32_t|int64_t|size_t|ssize_t)\b"],
        "float": [r"\b(float|double|long\s+double)\b"],
        "pointer": [r"\w+\s*\*\s*\w+"],
        "reference": [r"\w+\s*&\s*\w+"],
        "template": [r"\b(std::vector|std::map|std::set|std::unordered_map|std::array)\b"],
    },
)


class CCppFrontend(LanguageFrontend):
    @property
    def name(self) -> str:
        return "c_cpp"

    def get_tree_sitter_raw_language(self) -> Any:
        try:
            import tree_sitter_cpp
            return tree_sitter_cpp.language()
        except Exception:
            return None

    def parse_tree_sitter_functions(
        self,
        *,
        parser_utils: Any,
        source_bytes: bytes,
        root: Any,
        control_keywords: Set[str],
    ) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []
        for func_node in parser_utils._iter_nodes_by_type(root, "function_definition"):
            name_node = parser_utils._extract_function_name_node(func_node)
            if name_node is None:
                continue

            func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode(
                "utf-8", errors="ignore"
            )
            if not func_name or func_name in control_keywords:
                continue

            params = []
            params_node = parser_utils._extract_function_params_node(func_node)
            if params_node:
                params_text = source_bytes[params_node.start_byte:params_node.end_byte].decode(
                    "utf-8", errors="ignore"
                )
                params = parser_utils._parse_parameters(params_text)

            body_node = func_node.child_by_field_name("body")
            if not body_node:
                continue

            func_code = source_bytes[func_node.start_byte:func_node.end_byte].decode(
                "utf-8", errors="ignore"
            )
            start_line = source_bytes[:func_node.start_byte].count(b"\n") + 1
            end_line = source_bytes[:func_node.end_byte].count(b"\n") + 1

            functions.append({
                "name": func_name,
                "signature": parser_utils._build_signature(func_name, params),
                "parameters": params,
                "code": func_code,
                "body": func_code,
                "start_line": start_line,
                "end_line": end_line,
                "parser": "tree-sitter",
            })
        return functions


C_CPP_FRONTEND = CCppFrontend()
