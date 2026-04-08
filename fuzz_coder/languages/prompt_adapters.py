from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLanguageAdapter:
    name: str
    code_fence_lang: str
    cli_file_expr: str
    help_type_query_example: str
    alias_fuzz: str
    alias_fuzz_wide: str
    alias_more_fuzz_wide: str
    alias_more_fuzz: str
    alias_example_template: str
    alias_explain_template: str


_C_CPP_ALIAS_FUZZ = "Write a list of functions that can be used for fuzzing"

_C_CPP_ALIAS_FUZZ_WIDE = (
    "Write a large list (20-30) of functions that can be used for fuzzing. "
    "Only include functions with at most 4 parameters. "
    "A function is eligible if ANY of these is true: "
    "it parses input (or has parse/decode/split/tokenize in name/logic), "
    "OR it has path/filepath/file-name parameters, "
    "OR it has file-handle/stream parameters (FILE*, ifstream/fstream/istream), "
    "OR it has simple pointer-array parameters (char*, int*, uint8_t*, const variants), "
    "OR it reads stdin, "
    "OR it has std::vector/std::array/std::span-like parameters."
)

_C_CPP_ALIAS_MORE_FUZZ_WIDE = (
    "Write other functions in a large list (20-30) that can be used for fuzzing. "
    "Exclude functions already listed previously. "
    "Only include functions with at most 4 parameters. "
    "A function is eligible if ANY of these is true: "
    "it parses input (or has parse/decode/split/tokenize in name/logic), "
    "OR it has path/filepath/file-name parameters, "
    "OR it has file-handle/stream parameters (FILE*, ifstream/fstream/istream), "
    "OR it has simple pointer-array parameters (char*, int*, uint8_t*, const variants), "
    "OR it reads stdin, "
    "OR it has std::vector/std::array/std::span-like parameters."
)

_C_CPP_ALIAS_EXAMPLE = (
    "Write an example of {function_name} function. In the generated snippet, define a standalone main() "
    "and call {function_name} from it. Construct its parameters from data given from file in argv[1]"
)

_C_CPP_ALIAS_EXPLAIN = (
    "Analyze function parameter semantics for {function_name}: for each parameter, explain its role, "
    "expected data format/range, whether it is input/output/inout, where values usually come from in the codebase, "
    "and provide evidence from signature, call sites, and docs (file:line). "
    "If unknown, say explicitly \"unknown from provided context\"."
)


JAVA_ALIAS_FUZZ = _C_CPP_ALIAS_FUZZ

JAVA_ALIAS_FUZZ_WIDE = (
    "Write a large list (20-30) of functions that can be used for fuzzing. "
    "Only include functions with at most 4 parameters. "
    "A function is eligible if ANY of these is true: "
    "it parses input (or has parse/decode/split/tokenize in name/logic), "
    "OR it has path/filepath/file-name parameters, "
    "OR it has file-handle/stream parameters (File/InputStream/Reader/Path), "
    "OR it has simple array/buffer parameters (byte[], int[], char[], ByteBuffer), "
    "OR it reads stdin/System.in, "
    "OR it has List/Map/Set-like parameters."
)

JAVA_ALIAS_MORE_FUZZ_WIDE = (
    "Write other functions in a large list (20-30) that can be used for fuzzing. "
    "Exclude functions already listed previously. "
    "Only include functions with at most 4 parameters. "
    "A function is eligible if ANY of these is true: "
    "it parses input (or has parse/decode/split/tokenize in name/logic), "
    "OR it has path/filepath/file-name parameters, "
    "OR it has file-handle/stream parameters (File/InputStream/Reader/Path), "
    "OR it has simple array/buffer parameters (byte[], int[], char[], ByteBuffer), "
    "OR it reads stdin/System.in, "
    "OR it has List/Map/Set-like parameters."
)

JAVA_ALIAS_EXAMPLE = (
    "Write an example of {function_name} function. In the generated snippet, define a standalone main(String[] args) "
    "and call {function_name} from it. Construct its parameters from data given from file in args[0]"
)

JAVA_ALIAS_EXPLAIN = _C_CPP_ALIAS_EXPLAIN


_PROMPT_ADAPTERS = {
    "c_cpp": PromptLanguageAdapter(
        name="c_cpp",
        code_fence_lang="cpp",
        cli_file_expr="argv[1]",
        help_type_query_example="- Покажи функции с параметром std::string\n",
        alias_fuzz=_C_CPP_ALIAS_FUZZ,
        alias_fuzz_wide=_C_CPP_ALIAS_FUZZ_WIDE,
        alias_more_fuzz_wide=_C_CPP_ALIAS_MORE_FUZZ_WIDE,
        alias_more_fuzz="Write other functions that are good for fuzzing",
        alias_example_template=_C_CPP_ALIAS_EXAMPLE,
        alias_explain_template=_C_CPP_ALIAS_EXPLAIN,
    ),
    "java": PromptLanguageAdapter(
        name="java",
        code_fence_lang="java",
        cli_file_expr="args[0]",
        help_type_query_example="- Show functions with a String parameter\n",
        alias_fuzz=JAVA_ALIAS_FUZZ,
        alias_fuzz_wide=JAVA_ALIAS_FUZZ_WIDE,
        alias_more_fuzz_wide=JAVA_ALIAS_MORE_FUZZ_WIDE,
        alias_more_fuzz="Write other functions that are good for fuzzing",
        alias_example_template=JAVA_ALIAS_EXAMPLE,
        alias_explain_template=JAVA_ALIAS_EXPLAIN,
    ),
}


def get_prompt_language_adapter(language_name: str) -> PromptLanguageAdapter:
    return _PROMPT_ADAPTERS.get(language_name or "", _PROMPT_ADAPTERS["c_cpp"])
