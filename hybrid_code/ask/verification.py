from __future__ import annotations

import re
from typing import Dict, List, Set


def verify_answer_with_context(answer, context_frags):
    """Verify that functions mentioned in the answer actually exist in the context."""
    actual_funcs = set(f["name"] for f in context_frags)

    mentioned_funcs = set()
    for fn in actual_funcs:
        if re.search(rf"\b{re.escape(fn)}\b", answer):
            mentioned_funcs.add(fn)

    func_pattern = re.compile(r"\b([A-Za-z_]\w*)\(", re.MULTILINE)
    call_like = set(func_pattern.findall(answer))
    code_ticks = set(re.findall(r"`([A-Za-z_]\w*)`", answer))
    mentioned_candidates = mentioned_funcs | call_like | code_ticks

    hallucinated = mentioned_candidates - actual_funcs

    common_keywords = {
        "if", "for", "while", "switch", "return", "sizeof", "catch",
        "new", "delete", "throw", "else", "do", "class", "struct",
        "namespace", "template", "typedef", "using", "enum", "union",
        "printf", "scanf", "malloc", "free", "memset", "memcpy",
        "std::vector", "std::string", "std::map", "std::set",
        "phase", "criteria", "console", "input", "output", "function",
    }

    hallucinated = {h for h in hallucinated if h.lower() not in common_keywords and len(h) > 2}

    return {
        "mentioned": mentioned_candidates,
        "actual": actual_funcs,
        "hallucinated": hallucinated,
        "is_valid": len(hallucinated) == 0,
    }
