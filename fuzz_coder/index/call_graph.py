from __future__ import annotations

import re
from collections import defaultdict


CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def detect_calls(body, control_keywords):
    """Detect call-like tokens in function body."""
    calls = []
    for m in CALL_RE.finditer(body):
        n = m.group(1)
        if n not in control_keywords:
            calls.append(n)
    return list(set(calls))


def _resolve_call_ids(call_name, caller_chunk, name_to_ids, chunks_by_id):
    candidates = name_to_ids.get(call_name, [])
    if not candidates:
        return []

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

    # Cross-file overloaded call: avoid linking to reduce false positives.
    sigs = {chunks_by_id[cid].get("signature", "") for cid in candidates}
    if len(sigs) == 1:
        return sorted(set(candidates))

    return []


def build_call_graph(chunks, control_keywords):
    """Build a call graph with conservative name+file+signature resolution."""
    name_to_ids = defaultdict(list)
    chunks_by_id = {}

    for c in chunks:
        name_to_ids[c["name"]].append(c["id"])
        chunks_by_id[c["id"]] = c

    call_graph = {}
    called_by = defaultdict(list)

    for c in chunks:
        calls = detect_calls(c.get("body", ""), control_keywords)
        resolved_calls = []

        for call in calls:
            target_ids = _resolve_call_ids(call, c, name_to_ids, chunks_by_id)
            resolved_calls.extend(target_ids)

        resolved_calls = sorted(set(resolved_calls))
        for tid in resolved_calls:
            target_name = chunks_by_id[tid].get("name", "")
            if target_name:
                called_by[target_name].append(c["id"])

        call_graph[c["id"]] = {
            "calls": calls,
            "resolved_calls": resolved_calls,
            "called_by": sorted(set(called_by.get(c["name"], []))),
        }

    return call_graph, called_by
