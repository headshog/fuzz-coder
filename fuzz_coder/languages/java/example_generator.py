from __future__ import annotations

from .fallback import build_example_answer_from_context_java


class JavaExampleFallbackGenerator:
    language = "java"

    def build(self, frags, analysis=None, example_context=None, symbols=None, type_init_index=None):
        return build_example_answer_from_context_java(
            frags,
            analysis=analysis,
            example_context=example_context,
            symbols=symbols,
            type_init_index=type_init_index,
        )

