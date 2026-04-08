#!/usr/bin/env python3
"""Language-routed deterministic fallback example generators."""

from __future__ import annotations

from dataclasses import dataclass

from fuzz_coder.languages.registry import infer_ask_language_from_fragments

from .fallback_c_cpp import (
    build_example_answer_from_context_c_cpp,
    is_valid_function_chunk,
)
from .fallback_java import build_example_answer_from_context_java


@dataclass(frozen=True)
class ExampleFallbackGenerator:
    language: str

    def build(self, frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
        raise NotImplementedError


@dataclass(frozen=True)
class CppExampleFallbackGenerator(ExampleFallbackGenerator):
    language: str = "c_cpp"

    def build(self, frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
        return build_example_answer_from_context_c_cpp(
            frags,
            analysis=analysis,
            example_context=example_context,
            symbols=symbols,
            type_init_index=type_init_index,
        )


@dataclass(frozen=True)
class JavaExampleFallbackGenerator(ExampleFallbackGenerator):
    language: str = "java"

    def build(self, frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
        return build_example_answer_from_context_java(
            frags,
            analysis=analysis,
            example_context=example_context,
            symbols=symbols,
            type_init_index=type_init_index,
        )


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


def build_parameter_analysis_from_context(frags, analysis=None, example_context=None, function_hints=None):
    """Compatibility wrapper for legacy imports."""
    from fuzz_coder.ask.fallback_builders import (  # local import to avoid cycles
        build_parameter_analysis_from_context as _impl,
    )
    return _impl(
        frags,
        analysis=analysis,
        example_context=example_context,
        function_hints=function_hints,
    )
