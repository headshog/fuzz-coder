from __future__ import annotations

import re
from typing import Dict, List, Set


COMMON_NON_FUNCTION_TOKENS = {
    "if", "for", "while", "switch", "return", "sizeof", "catch",
    "new", "delete", "throw", "else", "do", "class", "struct",
    "namespace", "template", "typedef", "using", "enum", "union",
    "printf", "scanf", "malloc", "free", "memset", "memcpy",
    "std", "vector", "string", "map", "set",
    "phase", "criteria", "console", "input", "output", "function",
    "void", "int", "float", "double", "char", "bool", "const", "size_t",
}


ENTRY_HEADER_RE = re.compile(
    r"^\s*\d+\.\s+(?:\*\*)?(?:`)?([A-Za-z_]\w*)(?:`)?(?:\*\*)?\b.*$",
    flags=re.MULTILINE,
)

EVIDENCE_FILE_RE = re.compile(r"^\s*(?:[-*]\s*)?File:\s*(.+)$", flags=re.MULTILINE)
EVIDENCE_SIGNATURE_RE = re.compile(r"^\s*(?:[-*]\s*)?Signature:\s*(.+)$", flags=re.MULTILINE)


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


def _strip_inline_code(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("`") and t.endswith("`") and len(t) >= 2:
        t = t[1:-1].strip()
    return t


def _normalize_signature(signature: str) -> str:
    s = _strip_inline_code(signature)
    s = s.rstrip(";")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_arity(signature: str) -> int | None:
    sig = _normalize_signature(signature)
    lp = sig.find("(")
    rp = sig.rfind(")")
    if lp == -1 or rp == -1 or rp < lp:
        return None
    params = sig[lp + 1:rp].strip()
    if not params or params == "void":
        return 0
    depth = 0
    arity = 1
    for ch in params:
        if ch in "<({[":
            depth += 1
        elif ch in ">)}]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            arity += 1
    return arity


def _parse_file_ref(file_ref: str):
    raw = _strip_inline_code(file_ref).replace("\\", "/").strip().rstrip(".,:;")
    m = re.match(r"^(.*?)(?::(\d+)(?:-(\d+))?)?$", raw)
    if not m:
        return raw.lower(), None, None
    path = (m.group(1) or "").strip().lower()
    line_start = int(m.group(2)) if m.group(2) else None
    line_end = int(m.group(3)) if m.group(3) else line_start
    return path, line_start, line_end


def _file_matches_chunk(file_ref: str, chunk: Dict) -> bool:
    ref_path, ref_start, ref_end = _parse_file_ref(file_ref)
    if not ref_path:
        return False

    chunk_path = str(chunk.get("file", "")).replace("\\", "/").strip().lower()
    if not chunk_path:
        return False

    path_match = (
        chunk_path == ref_path
        or chunk_path.endswith(ref_path)
        or ref_path.endswith(chunk_path)
    )
    if not path_match:
        return False

    if ref_start is None:
        return True

    chunk_start = chunk.get("start_line")
    chunk_end = chunk.get("end_line")
    if not isinstance(chunk_start, int) or not isinstance(chunk_end, int):
        # Path is known but exact line window is unavailable in this chunk metadata.
        return True
    if ref_end is None:
        ref_end = ref_start

    return chunk_start <= ref_start <= chunk_end and chunk_start <= ref_end <= chunk_end


def _signature_matches_chunk(signature_ref: str, chunk: Dict) -> bool:
    ref_sig = _normalize_signature(signature_ref)
    if not ref_sig:
        return False

    chunk_sig = _normalize_signature(str(chunk.get("signature", "")))
    if not chunk_sig:
        return False

    if chunk_sig == ref_sig or chunk_sig in ref_sig or ref_sig in chunk_sig:
        return True

    ref_name = _extract_name_from_signature(ref_sig)
    chunk_name = _extract_name_from_signature(chunk_sig) or str(chunk.get("name", ""))
    if not ref_name or not chunk_name or ref_name != chunk_name:
        return False

    ref_arity = _extract_arity(ref_sig)
    chunk_arity = _extract_arity(chunk_sig)
    if ref_arity is not None and chunk_arity is not None and ref_arity != chunk_arity:
        return False

    return True


def _extract_structured_entries(answer: str) -> List[Dict[str, str | None]]:
    entries: List[Dict[str, str | None]] = []
    matches = list(ENTRY_HEADER_RE.finditer(answer))
    if not matches:
        return entries

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        block = answer[start:end]
        name = m.group(1)

        file_match = re.search(r"^\s*-\s*File:\s*(.+)$", block, flags=re.MULTILINE)
        sig_match = re.search(r"^\s*-\s*Signature:\s*(.+)$", block, flags=re.MULTILINE)

        entries.append({
            "name": name,
            "file": file_match.group(1).strip() if file_match else None,
            "signature": sig_match.group(1).strip() if sig_match else None,
        })

    return entries


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
    actual_by_name: Dict[str, List[Dict]] = {}
    for f in context_frags:
        name = f.get("name")
        if name:
            actual_by_name.setdefault(name, []).append(f)
    known_funcs = set(known_functions) if known_functions is not None else set(actual_funcs)

    mentioned_actual = set()
    for fn in actual_funcs:
        if re.search(rf"\b{re.escape(fn)}\b", answer):
            mentioned_actual.add(fn)

    structured_entries = _extract_structured_entries(answer)
    structured_candidates = _extract_structured_function_candidates(answer)
    structured_candidates.update({e["name"] for e in structured_entries if e.get("name")})

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

    file_mismatches = set()
    signature_mismatches = set()
    context_mismatches = set()

    for entry in structured_entries:
        fn = entry.get("name")
        if not fn or fn not in known_funcs or fn not in actual_funcs:
            continue

        candidates = actual_by_name.get(fn, [])
        if not candidates:
            continue

        has_file = bool(entry.get("file"))
        has_signature = bool(entry.get("signature"))
        file_candidates = [c for c in candidates if c.get("file")]
        signature_candidates = [c for c in candidates if c.get("signature")]
        pair_candidates = [c for c in candidates if c.get("file") and c.get("signature")]

        file_match_any = True
        sig_match_any = True
        if has_file and file_candidates:
            file_match_any = any(_file_matches_chunk(entry["file"], c) for c in file_candidates)
            if not file_match_any:
                file_mismatches.add(fn)
        if has_signature and signature_candidates:
            sig_match_any = any(_signature_matches_chunk(entry["signature"], c) for c in signature_candidates)
            if not sig_match_any:
                signature_mismatches.add(fn)

        if has_file and has_signature and pair_candidates:
            pair_match_any = any(
                _file_matches_chunk(entry["file"], c) and _signature_matches_chunk(entry["signature"], c)
                for c in pair_candidates
            )
            if not pair_match_any:
                context_mismatches.add(fn)

    return {
        "mentioned": mentioned_candidates,
        "structured_mentions": structured_candidates,
        "structured_entries": structured_entries,
        "actual": actual_funcs,
        "known": known_funcs,
        "hallucinated": hallucinated,
        "out_of_context": out_of_context,
        "file_mismatches": file_mismatches,
        "signature_mismatches": signature_mismatches,
        "context_mismatches": context_mismatches,
        "is_valid": (
            len(hallucinated) == 0
            and len(out_of_context) == 0
            and len(file_mismatches) == 0
            and len(signature_mismatches) == 0
            and len(context_mismatches) == 0
        ),
    }


def verify_example_answer_with_context(
    answer,
    context_frags,
    target_function=None,
    known_functions=None,
):
    """Verify example-generation answer against current context.

    Strict requirements for a valid example answer:
    - Must mention target function (if provided).
    - Must provide at least one `File:` reference with line information.
    - Must provide at least one `Signature:` reference.
    - File/signature references must match current context.
    """
    base = verify_answer_with_context(
        answer,
        context_frags,
        known_functions=known_functions,
    )

    file_refs = [m.strip() for m in EVIDENCE_FILE_RE.findall(answer or "")]
    signature_refs = [m.strip() for m in EVIDENCE_SIGNATURE_RE.findall(answer or "")]

    file_matches = 0
    has_line_reference = False
    for ref in file_refs:
        _, line_start, _ = _parse_file_ref(ref)
        if line_start is not None:
            has_line_reference = True
        if any(_file_matches_chunk(ref, c) for c in context_frags):
            file_matches += 1

    signature_matches = 0
    for sig in signature_refs:
        if any(_signature_matches_chunk(sig, c) for c in context_frags):
            signature_matches += 1

    target_mentioned = True
    target_signature_present = True
    if target_function:
        target_mentioned = re.search(rf"\b{re.escape(target_function)}\b", answer or "") is not None
        target_signature_present = any(
            (_extract_name_from_signature(sig) or "") == target_function
            for sig in signature_refs
        )

    missing_requirements = set()
    if not file_refs:
        missing_requirements.add("file_reference")
    if not signature_refs:
        missing_requirements.add("signature_reference")
    if not has_line_reference:
        missing_requirements.add("line_reference")
    if target_function and not target_mentioned:
        missing_requirements.add("target_mention")
    if target_function and signature_refs and not target_signature_present:
        missing_requirements.add("target_signature")
    if file_refs and file_matches == 0:
        missing_requirements.add("file_not_in_context")
    if signature_refs and signature_matches == 0:
        missing_requirements.add("signature_not_in_context")

    is_valid = (
        base.get("is_valid", False)
        and len(missing_requirements) == 0
    )

    out = dict(base)
    out.update({
        "file_references": file_refs,
        "signature_references": signature_refs,
        "has_line_reference": has_line_reference,
        "file_matches": file_matches,
        "signature_matches": signature_matches,
        "target_function": target_function,
        "target_mentioned": target_mentioned,
        "target_signature_present": target_signature_present,
        "missing_requirements": missing_requirements,
        "is_valid": is_valid,
    })
    return out
