from __future__ import annotations

from ..prompt_adapters import PromptLanguageAdapter


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


C_CPP_PROMPT_ADAPTER = PromptLanguageAdapter(
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
)
