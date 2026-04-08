from __future__ import annotations

from .base import LanguageFrontend, LanguageProfile
from .ask_adapters import (
    AskLanguageAdapter,
    get_ask_language_adapter as _get_ask_language_adapter_impl,
    infer_ask_language_from_fragments as _infer_ask_language_from_fragments_impl,
)
from .index_adapters import (
    IndexLanguageAdapter,
    get_index_language_adapter as _get_index_language_adapter_impl,
    infer_index_language_from_path as _infer_index_language_from_path_impl,
)
from .prompt_adapters import (
    PromptLanguageAdapter,
    get_prompt_language_adapter as _get_prompt_language_adapter_impl,
)
from .query_adapters import (
    QueryLanguageAdapter,
    get_query_language_adapter as _get_query_language_adapter_impl,
)
from .c_cpp import C_CPP_FRONTEND, C_CPP_PROFILE
from .java import JAVA_FRONTEND, JAVA_PROFILE


_PROFILES = {
    "c_cpp": C_CPP_PROFILE,
    "java": JAVA_PROFILE,
}

_FRONTENDS = {
    "c_cpp": C_CPP_FRONTEND,
    "java": JAVA_FRONTEND,
}


def get_language_profile(name: str) -> LanguageProfile:
    """Lookup language profile used by indexing pipeline."""
    if name not in _PROFILES:
        raise ValueError(f"Unsupported language profile: {name}")
    return _PROFILES[name]


def get_language_frontend(name: str) -> LanguageFrontend:
    """Lookup language frontend hooks used by tree-sitter/regex extraction."""
    if name not in _FRONTENDS:
        raise ValueError(f"Unsupported language frontend: {name}")
    return _FRONTENDS[name]


def get_supported_language_names():
    """Return supported language profile names."""
    return sorted(_PROFILES.keys())


def get_ask_language_adapter(name: str) -> AskLanguageAdapter:
    """Lookup ask-time language adapter (call/dataflow heuristics)."""
    return _get_ask_language_adapter_impl(name)


def infer_ask_language_from_fragments(frags, default: str = "c_cpp") -> str:
    """Infer language name from retrieved fragments."""
    return _infer_ask_language_from_fragments_impl(frags, default=default)


def get_index_language_adapter(name: str) -> IndexLanguageAdapter:
    """Lookup index-time adapter (regex parsing + structural extraction hooks)."""
    return _get_index_language_adapter_impl(name)


def infer_index_language_from_path(file_path: str, default: str = "c_cpp") -> str:
    """Infer language from file extension for index-time adapters."""
    return _infer_index_language_from_path_impl(file_path, default=default)


def get_prompt_language_adapter(name: str) -> PromptLanguageAdapter:
    """Lookup language-specific prompt/alias templates."""
    return _get_prompt_language_adapter_impl(name)


def get_query_language_adapter(name: str) -> QueryLanguageAdapter:
    """Lookup language-specific query-analysis heuristic adapter."""
    return _get_query_language_adapter_impl(name)
