from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .base import LanguageFrontend, LanguageProfile
from .java_calls import detect_call_details as _detect_call_details_impl


JAVA_PROFILE = LanguageProfile(
    name="java",
    supported_ext={".java"},
    control_keywords={
        "if", "for", "while", "switch", "return", "catch",
        "new", "throw", "else", "do", "class", "interface",
        "enum", "record", "package", "import", "var",
    },
    stdin_patterns=[
        r"\b(System\.in)\b",
        r"\bScanner\s*\(\s*System\.in\s*\)",
        r"\bBufferedReader\s*\(\s*new\s+InputStreamReader\s*\(\s*System\.in\s*\)\s*\)",
        r"\b(console\s*\.\s*readLine|Console\s*\.\s*readLine)\s*\(",
    ],
    file_input_patterns=[
        r"\b(FileInputStream|FileReader|BufferedReader|DataInputStream)\b",
        r"\b(Files\.readAllBytes|Files\.readString|Files\.newBufferedReader)\s*\(",
        r"\b(Paths\.get|Path\.of)\s*\(",
    ],
    api_call_patterns=[
        r"\b(HttpClient|HttpRequest|HttpURLConnection|URLConnection)\b",
        r"\b(OkHttpClient|Retrofit|WebClient|RestTemplate)\b",
    ],
    output_patterns=[
        r"\b(System\.out\.(print|println|printf))\s*\(",
        r"\b(Logger|log4j|slf4j)\b",
    ],
    memory_management_patterns=[
        r"\b(ByteBuffer\.(allocate|allocateDirect)|Unsafe|DirectByteBuffer)\b",
        r"\b(new\s+byte\s*\[|ByteArray(Input|Output)Stream)\b",
    ],
    error_handling_patterns=[
        r"\b(try|catch|finally|throw|throws)\b",
        r"\b(assert)\b",
    ],
    type_patterns={
        "byte_array": [r"\b(byte\s*\[\]|ByteBuffer|ByteArray(Input|Output)Stream)\b"],
        "string": [r"\b(String|CharSequence|StringBuilder|StringBuffer)\b"],
        "integer": [r"\b(byte|short|int|long|Integer|Long|Short|BigInteger)\b"],
        "float": [r"\b(float|double|Float|Double|BigDecimal)\b"],
        "pointer": [],
        "reference": [],
        "template": [r"\b(List|Set|Map|ArrayList|HashMap|HashSet)\s*<"],
    },
)


class JavaFrontend(LanguageFrontend):
    @property
    def name(self) -> str:
        return "java"

    def get_tree_sitter_raw_language(self) -> Any:
        try:
            import tree_sitter_java
            return tree_sitter_java.language()
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
        for node_type in ("method_declaration", "constructor_declaration"):
            for func_node in parser_utils._iter_nodes_by_type(root, node_type):
                body_node = func_node.child_by_field_name("body")
                if not body_node:
                    continue

                name_node = func_node.child_by_field_name("name")
                if name_node is None:
                    continue

                func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore"
                )
                if not func_name or func_name in control_keywords:
                    continue

                params = []
                params_node = func_node.child_by_field_name("parameters")
                if params_node:
                    params_text = source_bytes[params_node.start_byte:params_node.end_byte].decode(
                        "utf-8", errors="ignore"
                    )
                    params = parser_utils._parse_parameters(params_text)

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

    def preprocess_regex_signature(self, compact_signature: str) -> str:
        return re.sub(r"^(@[A-Za-z_]\w*(?:\([^)]*\))?\s+)+", "", compact_signature)

    def extra_regex_bad_prefixes(self):
        return ("interface ", "enum ", "record ", "package ", "import ")

    def detect_call_details(self, body: str, control_keywords: Set[str]) -> List[Dict[str, Any]]:
        return _detect_call_details_impl(body or "", control_keywords)


JAVA_FRONTEND = JavaFrontend()
