from __future__ import annotations

import re
from typing import Set


COMMON_NON_FUNCTION_TOKENS = {
    "if", "for", "while", "switch", "return", "sizeof", "catch",
    "new", "delete", "throw", "else", "do", "class", "struct",
    "namespace", "template", "typedef", "using", "enum", "union",
    "printf", "scanf", "malloc", "free", "memset", "memcpy",
    "std", "vector", "string", "map", "set",
    "phase", "criteria", "console", "input", "output", "function",
    "void", "int", "float", "double", "char", "bool", "const", "size_t",
}


def _extract_name_from_signature(signature_line: str) -> str | None:
    sig = signature_line.strip().strip("`").strip()
    if not sig:
        return None

    lp = sig.find("(")
    if lp == -1:
        return None

    head = sig[:lp]
    tokens = re.findall(r"([A-Za-z_~]\w*(?:::[A-Za-z_~]\w*)*)", head)
    if not tokens:
        return None

    name = tokens[-1].split("::")[-1]
    if name.startswith("~"):
        name = name[1:]
    if not name:
        return None
    return name


def _extract_structured_function_candidates(answer: str) -> Set[str]:
    candidates: Set[str] = set()

    # Preferred output format: numbered entry header with bold/backticks.
    header_patterns = [
        r"^\s*\d+\.\s+\*\*(?:`)?([A-Za-z_]\w*)(?:`)?\*\*",
        r"^\s*\d+\.\s+`([A-Za-z_]\w*)`",
        r"^\s*\d+\.\s+\*\*([A-Za-z_]\w*)\*\*",
    ]
    for pat in header_patterns:
        candidates.update(re.findall(pat, answer, flags=re.MULTILINE))

    # Signature lines are highly reliable for function-name extraction.
    for sig_line in re.findall(r"^\s*-\s*Signature:\s*(.+)$", answer, flags=re.MULTILINE):
        fn = _extract_name_from_signature(sig_line)
        if fn:
            candidates.add(fn)

    # Some model outputs use "Function: <name>" style.
    for fn in re.findall(r"^\s*(?:-+\s*)?Function:\s*`?([A-Za-z_]\w*)`?\b", answer, flags=re.MULTILINE):
        candidates.add(fn)

    return candidates


def verify_answer_with_context(answer, context_frags, known_functions=None):
    """Verify mentioned functions with context and optional global known symbol set.

    Returns:
      - hallucinated: mentioned names unknown to the whole indexed codebase
      - out_of_context: known names that are not present in provided context_frags
    """
    actual_funcs = {f["name"] for f in context_frags if f.get("name")}
    known_funcs = set(known_functions) if known_functions is not None else set(actual_funcs)

    mentioned_actual = set()
    for fn in actual_funcs:
        if re.search(rf"\b{re.escape(fn)}\b", answer):
            mentioned_actual.add(fn)

    structured_candidates = _extract_structured_function_candidates(answer)

    # Fallback for non-structured answers.
    if not structured_candidates:
        call_like = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", answer, flags=re.MULTILINE))
        structured_candidates = {
            c for c in call_like
            if len(c) > 2 and c.lower() not in COMMON_NON_FUNCTION_TOKENS
        }

    mentioned_candidates = mentioned_actual | structured_candidates

    hallucinated = {
        h for h in structured_candidates
        if h not in known_funcs and h.lower() not in COMMON_NON_FUNCTION_TOKENS and len(h) > 2
    }
    out_of_context = {
        h for h in structured_candidates
        if h in known_funcs and h not in actual_funcs and h.lower() not in COMMON_NON_FUNCTION_TOKENS and len(h) > 2
    }

    return {
        "mentioned": mentioned_candidates,
        "structured_mentions": structured_candidates,
        "actual": actual_funcs,
        "known": known_funcs,
        "hallucinated": hallucinated,
        "out_of_context": out_of_context,
        "is_valid": len(hallucinated) == 0 and len(out_of_context) == 0,
    }
