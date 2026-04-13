from __future__ import annotations

from ..prompt_adapters import PromptLanguageAdapter


_JAVA_ALIAS_EXPLAIN = (
    "Analyze function parameter semantics for {function_name}: for each parameter, explain its role, "
    "expected data format/range, whether it is input/output/inout, where values usually come from in the codebase, "
    "and provide evidence from signature, call sites, and docs (file:line). "
    "If unknown, say explicitly \"unknown from provided context\"."
)


JAVA_PROMPT_ADAPTER = PromptLanguageAdapter(
    name="java",
    code_fence_lang="java",
    cli_file_expr="args[0]",
    help_type_query_example="- Show functions with a String parameter\n",
    alias_fuzz=(
        "Write a large list (20-30) of functions that can be used for fuzzing. "
        "Only include functions with at most 4 parameters. "
        "A function is eligible if ANY of these is true: "
        "it parses input (or has parse/decode/split/tokenize in name/logic), "
        "OR it has path/filepath/file-name parameters, "
        "OR it has file-handle/stream parameters (File/InputStream/Reader/Path), "
        "OR it has simple array/buffer parameters (byte[], int[], char[], ByteBuffer), "
        "OR it reads stdin/System.in, "
        "OR it has List/Map/Set-like parameters."
    ),
    alias_fuzz_wide="Write a list of functions that can be used for fuzzing without parameter count limit.",
    alias_more_fuzz_wide="Write other functions that are good for fuzzing without parameter count limit.",
    alias_more_fuzz=(
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
    ),
    alias_example_template=(
        "Write an example of {function_name} function. In the generated snippet, define a standalone main(String[] args) "
        "and call {function_name} from it. Construct its parameters from data given from file in args[0]"
    ),
    alias_explain_template=_JAVA_ALIAS_EXPLAIN,
)
