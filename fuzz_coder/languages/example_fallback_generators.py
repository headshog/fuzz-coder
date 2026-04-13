#!/usr/bin/env python3
"""Language-routed deterministic fallback example generators."""

from __future__ import annotations

from fuzz_coder.languages.registry import infer_ask_language_from_fragments

from .c_cpp.example_generator import CppExampleFallbackGenerator
from .java.example_generator import JavaExampleFallbackGenerator


_EXAMPLE_FALLBACK_GENERATORS = {
    "c_cpp": CppExampleFallbackGenerator(),
    "java": JavaExampleFallbackGenerator(),
}


def _resolve_language(frags, example_context=None):
    ctx_lang = str((example_context or {}).get("language", "") or "").strip().lower()
    if ctx_lang in _EXAMPLE_FALLBACK_GENERATORS:
        return ctx_lang
    return infer_ask_language_from_fragments(frags, default="c_cpp")


def build_example_answer_from_context(frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
    language_name = _resolve_language(frags, example_context=example_context)
    generator = _EXAMPLE_FALLBACK_GENERATORS.get(language_name, _EXAMPLE_FALLBACK_GENERATORS["c_cpp"])
    return generator.build(
        frags,
        analysis=analysis,
        example_context=example_context,
        symbols=symbols,
        type_init_index=type_init_index,
    )
