from __future__ import annotations

import re
from typing import Dict, List, Set

from .language_adapter import (
    get_ask_language_adapter,
    infer_ask_language_from_fragments,
)


ENTRY_HEADER_RE = re.compile(
    r"^\s*\d+\.\s+(?:\*\*)?(?:`)?([A-Za-z_]\w*)(?:`)?(?:\*\*)?\b.*$",
    flags=re.MULTILINE,
)

EVIDENCE_FILE_RE = re.compile(r"^\s*(?:[-*]\s*)?File:\s*(.+)$", flags=re.MULTILINE)
EVIDENCE_SIGNATURE_RE = re.compile(r"^\s*(?:[-*]\s*)?Signature:\s*(.+)$", flags=re.MULTILINE)
EVIDENCE_OBSERVED_CALL_RE = re.compile(r"^\s*(?:[-*]\s*)?Observed call:\s*(.+)$", flags=re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+\-]*)\n(.*?)```", flags=re.DOTALL)


def _get_verification_adapter(context_frags):
    language_name = infer_ask_language_from_fragments(context_frags or [], default="c_cpp")
    return get_ask_language_adapter(language_name)


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


def _extract_arity(signature: str, adapter=None) -> int | None:
    sig = _normalize_signature(signature)
    lp = sig.find("(")
    rp = sig.rfind(")")
    if lp == -1 or rp == -1 or rp < lp:
        return None
    params = sig[lp + 1:rp].strip()
    if not params or params == "void":
        return 0
    ad = adapter or get_ask_language_adapter("c_cpp")
    return len(ad.split_top_level_arguments(params))


def _extract_call_arities(text: str, target_function: str, adapter=None) -> List[int]:
    if not text or not target_function:
        return []
    ad = adapter or get_ask_language_adapter("c_cpp")
    return ad.extract_call_arities(text, target_function, limit=64)


def _extract_call_arg_lists(text: str, target_function: str, adapter=None) -> List[List[str]]:
    if not text or not target_function:
        return []
    ad = adapter or get_ask_language_adapter("c_cpp")
    calls = ad.extract_call_argument_lists(text, target_function, limit=64)
    return [list(c.get("args", [])) for c in calls]


def _extract_file_source_vars(code_text: str, adapter=None) -> List[str]:
    ad = adapter or get_ask_language_adapter("c_cpp")
    return ad.extract_file_source_vars(code_text or "")


def _normalize_chain_token(token: str) -> str:
    ad = get_ask_language_adapter("c_cpp")
    return ad.normalize_chain_token(token)


def _lhs_base_name(lhs_expr: str) -> str:
    ad = get_ask_language_adapter("c_cpp")
    return ad.lhs_base_name(lhs_expr)


def _extract_simple_assignment_edges(code: str, adapter=None) -> List[tuple[str, str]]:
    ad = adapter or get_ask_language_adapter("c_cpp")
    return ad.extract_simple_assignment_edges(code or "")


def _extract_file_data_flow_symbols(code_text: str, source_vars: List[str], adapter=None) -> List[str]:
    ad = adapter or get_ask_language_adapter("c_cpp")
    return ad.extract_file_data_flow_symbols(code_text or "", source_vars or [])


def _arg_uses_any_var(arg_expr: str, names: List[str], adapter=None) -> bool:
    ad = adapter or get_ask_language_adapter("c_cpp")
    return ad.arg_uses_any_var(arg_expr or "", names or [])


def _extract_nominal_type_name(type_text: str, adapter=None) -> str:
    ad = adapter or get_ask_language_adapter("c_cpp")
    return ad.extract_nominal_type_name(type_text or "")


def _build_strong_recipe_expectations(type_init_index, target_function, target_params, adapter=None):
    raw = type_init_index or {}
    if not isinstance(raw, dict):
        return []
    by_type = raw.get("init_recipes_by_type")
    if not isinstance(by_type, dict):
        return []
    expected = []
    target = str(target_function or "").strip()
    ad = adapter or get_ask_language_adapter("c_cpp")
    for i, p in enumerate(target_params or []):
        ptype = str((p or {}).get("type", ""))
        nominal = _extract_nominal_type_name(ptype, adapter=ad)
        if not nominal:
            continue
        entries = list(by_type.get(nominal, [])) + list(by_type.get(nominal.lower(), []))
        if not entries:
            continue
        best = None
        best_key = None
        for e in entries:
            if not isinstance(e, dict):
                continue
            etarget = str(e.get("target_function", "")).strip()
            eidx = int(e.get("arg_index", -1))
            priority = 0
            if target and etarget == target and eidx == i:
                priority = 3
            elif target and etarget == target:
                priority = 2
            elif eidx == i:
                priority = 1
            score = int(e.get("score", 0))
            call_sites = int(e.get("call_sites", 0))
            has_alloc = bool(str(e.get("allocator_expr", "")).strip())
            key = (priority, score, call_sites, has_alloc)
            if best is None or key > best_key:
                best = e
                best_key = key
        if best is None:
            continue
        if best_key[0] <= 0:
            continue
        if int(best.get("score", 0)) < 4 and int(best.get("call_sites", 0)) < 1:
            continue
        expected.append({
            "arg_index": i,
            "nominal": nominal,
            "allocator_expr": str(best.get("allocator_expr", "")).strip(),
            "fields": list(best.get("fields") or []),
            "score": int(best.get("score", 0)),
            "call_sites": int(best.get("call_sites", 0)),
        })
    return expected


def _code_uses_default_init_for_nominal(code_text: str, nominal: str) -> bool:
    code = code_text or ""
    n = re.escape(str(nominal or "").strip())
    if not n:
        return False
    patterns = [
        rf"\b{n}\b\s+[A-Za-z_]\w*\s*\{{\s*\}}\s*;",
        rf"\b{n}\b\s+[A-Za-z_]\w*\s*=\s*\{{\s*\}}\s*;",
        rf"\b{n}\b\s*\*\s*[A-Za-z_]\w*\s*=\s*&\s*[A-Za-z_]\w+\s*;",
    ]
    return any(re.search(p, code) is not None for p in patterns)


def _code_contains_allocator_expr(code_text: str, allocator_expr: str) -> bool:
    expr = str(allocator_expr or "").strip()
    if not expr:
        return False
    call_name = _extract_name_from_signature(f"{expr};") or ""
    if call_name:
        return re.search(rf"\b{re.escape(call_name)}\s*\(", code_text or "") is not None
    m = re.match(r"\s*((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*\(", expr)
    if m:
        call_name = str(m.group(1) or "").split("::")[-1]
        return re.search(rf"\b{re.escape(call_name)}\s*\(", code_text or "") is not None
    return False


def _confidence_level(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


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


def _signature_matches_chunk(signature_ref: str, chunk: Dict, adapter=None) -> bool:
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

    ad = adapter or _get_verification_adapter([chunk])
    ref_arity = _extract_arity(ref_sig, adapter=ad)
    chunk_arity = _extract_arity(chunk_sig, adapter=ad)
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
    adapter = _get_verification_adapter(context_frags)
    non_function_tokens = adapter.non_function_tokens()
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
            if len(c) > 2 and c.lower() not in non_function_tokens
        }

    mentioned_candidates = mentioned_actual | structured_candidates

    hallucinated = {
        h for h in structured_candidates
        if h not in known_funcs and h.lower() not in non_function_tokens and len(h) > 2
    }
    out_of_context = {
        h for h in structured_candidates
        if h in known_funcs and h not in actual_funcs and h.lower() not in non_function_tokens and len(h) > 2
    }

    file_mismatches = set()
    signature_mismatches = set()
    context_mismatches = set()
    consistency_issues = set()

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

    duplicate_names = set()
    seen_names = set()
    for e in structured_entries:
        n = e.get("name")
        if not n:
            continue
        if n in seen_names:
            duplicate_names.add(n)
        seen_names.add(n)
    if duplicate_names:
        consistency_issues.add("duplicate_structured_entries")

    score = 1.0
    score -= 0.30 * len(hallucinated)
    score -= 0.22 * len(out_of_context)
    score -= 0.18 * len(context_mismatches)
    score -= 0.12 * len(file_mismatches)
    score -= 0.12 * len(signature_mismatches)
    score -= 0.08 * len(consistency_issues)
    if structured_entries and not (hallucinated or out_of_context or context_mismatches):
        score += 0.05
    score = max(0.0, min(1.0, score))

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
        "consistency_issues": consistency_issues,
        "confidence_score": score,
        "confidence_level": _confidence_level(score),
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
    example_context=None,
    type_init_index=None,
):
    """Verify example-generation answer against current context.

    Strict requirements for a valid example answer:
    - Must mention target function (if provided).
    - Must provide at least one `File:` reference with line information.
    - Must provide at least one `Signature:` reference.
    - File/signature references must match current context.
    - If structured example_context contains caller/call-site facts, caller evidence is required.
    - If structured example_context requires file-based data flow, target call must use data derived from the CLI file argument.
    """
    base = verify_answer_with_context(
        answer,
        context_frags,
        known_functions=known_functions,
    )
    adapter = _get_verification_adapter(context_frags)

    file_refs = [m.strip() for m in EVIDENCE_FILE_RE.findall(answer or "")]
    signature_refs = [m.strip() for m in EVIDENCE_SIGNATURE_RE.findall(answer or "")]
    observed_call_refs = [m.strip() for m in EVIDENCE_OBSERVED_CALL_RE.findall(answer or "")]
    code_blocks = CODE_BLOCK_RE.findall(answer or "")
    code_block_present = len(code_blocks) > 0
    text_for_calls = "\n".join(code_blocks) if code_blocks else (answer or "")
    expected_target = (example_context or {}).get("target")
    expected_caller = (example_context or {}).get("caller")
    expected_observed_call = (example_context or {}).get("observed_call")
    flow_hints = (example_context or {}).get("file_data_flow_hints") or {}
    requires_file_data = bool(flow_hints.get("requires_file_data"))

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
        if any(_signature_matches_chunk(sig, c, adapter=adapter) for c in context_frags):
            signature_matches += 1

    target_mentioned = True
    target_signature_present = True
    target_call_arities: List[int] = []
    target_call_arg_lists: List[List[str]] = []
    target_call_present = True
    target_call_arity_match = True
    if target_function:
        target_mentioned = re.search(rf"\b{re.escape(target_function)}\b", answer or "") is not None
        target_signature_present = any(
            (_extract_name_from_signature(sig) or "") == target_function
            for sig in signature_refs
        )
        target_call_arities = _extract_call_arities(text_for_calls, target_function, adapter=adapter)
        target_call_arg_lists = _extract_call_arg_lists(text_for_calls, target_function, adapter=adapter)
        target_call_present = len(target_call_arities) > 0

        expected_arities = set()
        for c in context_frags:
            if c.get("name") != target_function:
                continue
            ar = _extract_arity(str(c.get("signature", "")), adapter=adapter)
            if ar is not None:
                expected_arities.add(ar)
        if expected_arities and target_call_arities:
            target_call_arity_match = any(a in expected_arities for a in target_call_arities)

    target_params = []
    if target_function:
        for c in context_frags:
            if c.get("name") == target_function and isinstance(c.get("parameters"), list):
                target_params = list(c.get("parameters") or [])
                break
    recipe_expectations = _build_strong_recipe_expectations(
        type_init_index=type_init_index,
        target_function=target_function,
        target_params=target_params,
        adapter=adapter,
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

    if expected_target and target_function and expected_target.get("name") == target_function:
        target_file_ok = any(_file_matches_chunk(ref, expected_target) for ref in file_refs)
        target_sig_ok = any(_signature_matches_chunk(sig, expected_target) for sig in signature_refs)
        if not target_file_ok:
            missing_requirements.add("target_file_reference")
        if not target_sig_ok:
            missing_requirements.add("target_signature_reference")

    if expected_caller:
        caller_file_ok = any(_file_matches_chunk(ref, expected_caller) for ref in file_refs)
        caller_sig_ok = any(_signature_matches_chunk(sig, expected_caller) for sig in signature_refs)
        if not caller_file_ok:
            missing_requirements.add("caller_file_reference")
        if not caller_sig_ok:
            missing_requirements.add("caller_signature_reference")
        if not observed_call_refs:
            missing_requirements.add("observed_call_reference")

    consistency_issues = set(base.get("consistency_issues", set()))
    if not code_block_present:
        consistency_issues.add("code_block_missing")
    enforce_target_call_presence = bool(expected_observed_call) or requires_file_data
    if target_function and enforce_target_call_presence and not target_call_present:
        consistency_issues.add("target_call_not_found")
    if target_function and target_call_present and not target_call_arity_match:
        consistency_issues.add("target_call_arity_mismatch")

    if target_function and observed_call_refs and (expected_observed_call or expected_caller):
        observed_mentions_target = any(
            re.search(
                rf"\b(?:[A-Za-z_]\w*::)*{re.escape(target_function)}\s*\(",
                _strip_inline_code(ref),
            ) is not None
            for ref in observed_call_refs
        )
        if not observed_mentions_target:
            consistency_issues.add("observed_call_not_target")

    if target_function and expected_observed_call and observed_call_refs:
        expected_arity = expected_observed_call.get("arity")
        if isinstance(expected_arity, int):
            observed_ref_arities = []
            for ref in observed_call_refs:
                txt = _strip_inline_code(ref)
                observed_ref_arities.extend(
                    _extract_call_arities(txt, target_function, adapter=adapter)
                )
            if observed_ref_arities and all(a != expected_arity for a in observed_ref_arities):
                consistency_issues.add("observed_call_arity_mismatch")

    has_argv1 = adapter.is_cli_file_expr(text_for_calls)
    file_source_vars = _extract_file_source_vars(text_for_calls, adapter=adapter)
    file_flow_vars = _extract_file_data_flow_symbols(text_for_calls, file_source_vars, adapter=adapter)
    target_uses_file_data = False
    for arg_list in target_call_arg_lists:
        for arg in arg_list:
            if adapter.is_cli_file_expr(arg):
                target_uses_file_data = True
                break
            if _arg_uses_any_var(arg, file_flow_vars, adapter=adapter):
                target_uses_file_data = True
                break
        if target_uses_file_data:
            break

    if requires_file_data:
        if not has_argv1:
            missing_requirements.add("argv1_usage")
        if not target_uses_file_data:
            missing_requirements.add("file_data_flow_to_target")
            consistency_issues.add("file_data_not_used_in_target_call")

    default_init_penalized_nominals = []
    if recipe_expectations and code_block_present:
        for exp in recipe_expectations:
            nominal = str(exp.get("nominal", "")).strip()
            if not nominal:
                continue
            alloc_expr = str(exp.get("allocator_expr", "")).strip()
            if alloc_expr and _code_contains_allocator_expr(text_for_calls, alloc_expr):
                continue
            if _code_uses_default_init_for_nominal(text_for_calls, nominal):
                default_init_penalized_nominals.append(nominal)
        if default_init_penalized_nominals:
            consistency_issues.add("default_init_used_despite_recipe")

    score = float(base.get("confidence_score", 1.0))
    score -= 0.16 * len(missing_requirements)
    score -= 0.10 * len(consistency_issues)
    if file_matches > 0:
        score += 0.04
    if signature_matches > 0:
        score += 0.04
    if target_signature_present:
        score += 0.03
    if target_function and target_call_present:
        score += 0.04
    if target_function and target_call_present and target_call_arity_match:
        score += 0.04
    if expected_caller and observed_call_refs:
        score += 0.03
    if requires_file_data and target_uses_file_data:
        score += 0.05
    if default_init_penalized_nominals:
        score -= min(0.20, 0.08 * len(set(default_init_penalized_nominals)))

    # Keep confidence conservative when consistency checks fail.
    if consistency_issues:
        score = min(score, 0.74)
    if "target_call_arity_mismatch" in consistency_issues:
        # Arity mismatch is a strong signal that generated code likely does not
        # match real usage, so confidence must not be "high".
        score = min(score, 0.54)
    if "file_data_not_used_in_target_call" in consistency_issues:
        score = min(score, 0.45)
    if "default_init_used_despite_recipe" in consistency_issues:
        score = min(score, 0.60)

    score = max(0.0, min(1.0, score))

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
        "code_block_present": code_block_present,
        "target_call_present": target_call_present,
        "target_call_arities": target_call_arities,
        "target_call_arg_lists": target_call_arg_lists,
        "target_call_arity_match": target_call_arity_match,
        "observed_call_references": observed_call_refs,
        "requires_file_data": requires_file_data,
        "target_uses_file_data": target_uses_file_data,
        "has_argv1_in_code": has_argv1,
        "file_source_vars": file_source_vars,
        "file_flow_vars": file_flow_vars,
        "default_init_penalized_nominals": sorted(set(default_init_penalized_nominals)),
        "recipe_expectations_count": len(recipe_expectations),
        "consistency_issues": consistency_issues,
        "missing_requirements": missing_requirements,
        "confidence_score": score,
        "confidence_level": _confidence_level(score),
        "is_valid": is_valid,
    })
    return out
