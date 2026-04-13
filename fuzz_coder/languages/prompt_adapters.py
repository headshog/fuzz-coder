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


from .c_cpp.prompt_adapter import C_CPP_PROMPT_ADAPTER
from .java.prompt_adapter import JAVA_PROMPT_ADAPTER


_PROMPT_ADAPTERS = {
    "c_cpp": C_CPP_PROMPT_ADAPTER,
    "java": JAVA_PROMPT_ADAPTER,
}


def get_prompt_language_adapter(language_name: str) -> PromptLanguageAdapter:
    return _PROMPT_ADAPTERS.get(language_name or "", _PROMPT_ADAPTERS["c_cpp"])
