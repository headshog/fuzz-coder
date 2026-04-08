from __future__ import annotations

"""Compatibility API for ask-side language adapter access.

Language-specific implementations live in `fuzz_coder.languages.ask_adapters`.
This module keeps stable imports for ask modules while exposing only shared API.
"""

from fuzz_coder.languages.ask_adapters import AskLanguageAdapter
from fuzz_coder.languages.registry import (
    get_ask_language_adapter,
    infer_ask_language_from_fragments,
)

__all__ = [
    "AskLanguageAdapter",
    "get_ask_language_adapter",
    "infer_ask_language_from_fragments",
]
