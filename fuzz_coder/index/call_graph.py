from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from fuzz_coder.languages.registry import get_language_frontend


def _signature_arity(signature: str) -> Optional[int]:
    if not signature:
        return None
    sig = str(signature)
    lp = sig.find("(")
    rp = sig.rfind(")")
    if lp == -1 or rp == -1 or rp < lp:
        return None
    params = sig[lp + 1:rp].strip()
    if not params or params == "void":
        return 0

    depth_paren = 0
    depth_angle = 0
    depth_brace = 0
    depth_bracket = 0
    arity = 1
    i = 0
    n = len(params)
    while i < n:
        ch = params[i]
        if ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle = max(0, depth_angle - 1)
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == "," and depth_paren == 0 and depth_angle == 0 and depth_brace == 0 and depth_bracket == 0:
            arity += 1
        i += 1
    return arity


def _resolve_call_ids(
    call_info: Dict,
    caller_chunk: Dict,
    name_to_ids: Dict[str, List[int]],
    chunks_by_id: Dict[int, Dict],
) -> List[int]:
    call_name = call_info.get("name")
    call_arity = call_info.get("arity")
    if not call_name:
        return []

    candidates = name_to_ids.get(call_name, [])
    if not candidates:
        return []

    if call_arity is not None:
        arity_filtered = []
        for cid in candidates:
            target_arity = _signature_arity(chunks_by_id[cid].get("signature", ""))
            if target_arity is None or target_arity == call_arity:
                arity_filtered.append(cid)
        if arity_filtered:
            candidates = arity_filtered

    if len(candidates) == 1:
        return list(candidates)

    caller_file = caller_chunk.get("file")
    same_file = [cid for cid in candidates if chunks_by_id[cid].get("file") == caller_file]

    # Prefer unique same-file target (helps with namespaces/modules split across files).
    if len(same_file) == 1:
        return same_file

    # If same-file candidates share one signature, keep them (template instantiations/duplicates).
    if len(same_file) > 1:
        sigs = {chunks_by_id[cid].get("signature", "") for cid in same_file}
        if len(sigs) == 1:
            return sorted(set(same_file))
        return []

    # Ambiguous cross-file call: skip to reduce false positives for overloads/common names.
    return []


def _normalize_call_details(details: Iterable[Dict]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for d in details or []:
        if not isinstance(d, dict):
            continue
        name = d.get("name")
        if not name:
            continue
        entry = {
            "name": name,
            "qualified": d.get("qualified"),
            "is_member": bool(d.get("is_member", False)),
            "arity": d.get("arity"),
        }
        key = (entry["name"], entry["qualified"], entry["is_member"], entry["arity"])
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def detect_call_details(body, control_keywords, language_name: str = "c_cpp"):
    frontend = get_language_frontend(language_name)
    details = frontend.detect_call_details(body or "", set(control_keywords or []))
    return _normalize_call_details(details)


def detect_calls(body, control_keywords, language_name: str = "c_cpp"):
    return [d["name"] for d in detect_call_details(body, control_keywords, language_name=language_name)]


def build_call_graph(chunks, control_keywords, language_name: str = "c_cpp"):
    """Build a call graph with conservative name+file+signature resolution."""
    name_to_ids = defaultdict(list)
    chunks_by_id = {}

    for c in chunks:
        name_to_ids[c["name"]].append(c["id"])
        chunks_by_id[c["id"]] = c

    call_graph = {}
    reverse_by_id = defaultdict(set)
    calls_by_id = {}
    call_details_by_id = {}
    resolved_calls_by_id = {}

    for c in chunks:
        caller_id = c["id"]
        call_details = detect_call_details(c.get("body", ""), control_keywords, language_name=language_name)
        calls = list(dict.fromkeys(d["name"] for d in call_details))
        resolved_calls = []

        for call in call_details:
            target_ids = _resolve_call_ids(call, c, name_to_ids, chunks_by_id)
            resolved_calls.extend(target_ids)

        resolved_calls = sorted(set(resolved_calls))
        calls_by_id[caller_id] = calls
        call_details_by_id[caller_id] = call_details
        resolved_calls_by_id[caller_id] = resolved_calls
        for tid in resolved_calls:
            reverse_by_id[tid].add(caller_id)

    for c in chunks:
        cid = c["id"]
        call_graph[cid] = {
            "calls": calls_by_id.get(cid, []),
            "call_details": call_details_by_id.get(cid, []),
            "resolved_calls": resolved_calls_by_id.get(cid, []),
            "called_by": sorted(reverse_by_id.get(cid, set())),
        }

    # Keep "called_by.json" useful for both new id-based and legacy name-based lookups.
    called_by_out = {}
    for c in chunks:
        cid = c["id"]
        callers = sorted(reverse_by_id.get(cid, set()))
        called_by_out[str(cid)] = callers

    called_by_name = defaultdict(set)
    for c in chunks:
        cid = c["id"]
        called_by_name[c["name"]].update(reverse_by_id.get(cid, set()))
    for name, callers in called_by_name.items():
        called_by_out[name] = sorted(callers)

    return call_graph, called_by_out
