from __future__ import annotations

from typing import Dict, List, Optional, Set

from fuzz_coder.languages.registry import get_ask_language_adapter, infer_ask_language_from_fragments

from .example_context import (
    _arg_uses_any_var,
    _classify_param_shape,
    _extract_call_argument_lists,
    _extract_file_data_flow_symbols,
    _extract_file_source_vars,
    _file_matches_any_filter,
    _target_quality_score,
)


def _to_int_ids(values) -> List[int]:
    out: List[int] = []
    for v in values or []:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _dedup(seq: List[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _normalize_id_to_chunk(meta: List[Dict]) -> Dict[int, Dict]:
    by_id: Dict[int, Dict] = {}
    for idx, chunk in enumerate(meta):
        cid = chunk.get("id", idx)
        try:
            by_id[int(cid)] = chunk
        except Exception:
            by_id[idx] = chunk
    return by_id


def _get_call_graph_entry(call_graph: Dict, cid: int) -> Dict:
    return (
        call_graph.get(str(cid))
        or call_graph.get(cid)
        or {}
    )


def _extract_target_arity(target: Dict) -> Optional[int]:
    params = list(target.get("parameters") or [])
    return len(params) if params is not None else None


def _observed_call_for_target(
    caller: Dict,
    target_name: str,
    target_arity: Optional[int],
    *,
    language_name: str,
) -> Optional[Dict]:
    calls = _extract_call_argument_lists(
        str(caller.get("code", "")),
        target_name,
        limit=8,
        language_name=language_name,
    )
    if not calls:
        return None
    if target_arity is None:
        return calls[0]
    for c in calls:
        if len(c.get("args", [])) == target_arity:
            return c
    return calls[0]


class ExampleGroundingResolver:
    """Resolve richer example grounding using broader candidate pool and call graph."""

    def __init__(
        self,
        *,
        meta: List[Dict],
        symbols: Dict[str, List[int]],
        call_graph: Dict,
        called_by: Optional[Dict] = None,
    ):
        self.meta = meta or []
        self.symbols = symbols or {}
        self.call_graph = call_graph or {}
        self.called_by = called_by or {}
        self.id_to_chunk = _normalize_id_to_chunk(self.meta)
        self.language_name = infer_ask_language_from_fragments(self.meta, default="c_cpp")
        self.adapter = get_ask_language_adapter(self.language_name)

    def _resolve_target(self, frags: List[Dict], analysis: Dict) -> Optional[Dict]:
        function_names = list(analysis.get("function_names") or [])
        primary = analysis.get("primary_function_name")
        path_filters = list(analysis.get("path_filters") or [])

        target_candidates: List[Dict] = []
        wanted_names: List[str] = []
        if primary:
            wanted_names.append(primary)
        for n in function_names:
            if n not in wanted_names:
                wanted_names.append(n)

        for name in wanted_names:
            for cid in _to_int_ids(self.symbols.get(name, [])):
                chunk = self.id_to_chunk.get(cid)
                if chunk is not None:
                    target_candidates.append(chunk)

        if not target_candidates:
            target_candidates = list(frags or [])

        if path_filters:
            filtered = [
                c for c in target_candidates
                if _file_matches_any_filter(str(c.get("file", "")), path_filters)
            ]
            if filtered:
                target_candidates = filtered

        if not target_candidates:
            return None

        needs_file = bool(analysis.get("needs_file"))

        def _target_score(chunk: Dict):
            cid = int(chunk.get("id", -1))
            cg = _get_call_graph_entry(self.call_graph, cid) if cid >= 0 else {}
            callers = _dedup(_to_int_ids(cg.get("called_by", [])))
            caller_chunks = [self.id_to_chunk.get(x) for x in callers if x in self.id_to_chunk]
            caller_chunks = [c for c in caller_chunks if c is not None]
            has_main_caller = any(str(c.get("name", "")).lower() == "main" for c in caller_chunks)
            has_file_reader_caller = any(self.adapter.caller_reads_cli_file_data(str(c.get("code", ""))) for c in caller_chunks)
            quality = _target_quality_score(chunk)
            return (
                1 if has_main_caller else 0,
                1 if (needs_file and has_file_reader_caller) else 0,
                len(callers),
                quality,
            )

        return sorted(target_candidates, key=_target_score, reverse=True)[0]

    def _collect_expanded_pool_ids(
        self,
        *,
        frags: List[Dict],
        target: Dict,
        analysis: Dict,
        candidate_ids: Optional[List[int]],
        top_k: int,
    ) -> List[int]:
        out: List[int] = []

        for f in frags or []:
            try:
                out.append(int(f.get("id")))
            except Exception:
                continue

        for cid in _to_int_ids(candidate_ids)[: max(0, top_k)]:
            out.append(cid)

        target_id = int(target.get("id", -1))
        if target_id >= 0:
            out.append(target_id)
            cg = _get_call_graph_entry(self.call_graph, target_id)
            out.extend(_to_int_ids(cg.get("called_by", [])))
            out.extend(_to_int_ids(cg.get("resolved_calls", [])))
            out.extend(_to_int_ids(self.called_by.get(str(target_id), [])))

        for fn in list(analysis.get("function_names") or []):
            for cid in _to_int_ids(self.symbols.get(fn, [])):
                out.append(cid)

        out = _dedup(out)
        return [cid for cid in out if cid in self.id_to_chunk][: max(0, top_k)]

    def _pick_caller(self, *, pool_ids: List[int], target: Dict, analysis: Dict) -> tuple[Optional[Dict], Optional[Dict]]:
        target_name = str(target.get("name", "")).strip()
        if not target_name:
            return None, None

        target_id = int(target.get("id", -1))
        target_arity = _extract_target_arity(target)
        edge_callers = set()
        if target_id >= 0:
            cg = _get_call_graph_entry(self.call_graph, target_id)
            edge_callers.update(_to_int_ids(cg.get("called_by", [])))
            edge_callers.update(_to_int_ids(self.called_by.get(str(target_id), [])))
            edge_callers.update(_to_int_ids(self.called_by.get(target_id, [])))

        secondary_names = {
            n for n in list(analysis.get("function_names") or [])
            if n and n != target_name
        }
        needs_file = bool(analysis.get("needs_file"))

        primary_candidates = []
        fallback_same_name_candidates = []
        for cid in pool_ids:
            if cid == target_id:
                continue
            chunk = self.id_to_chunk.get(cid)
            if chunk is None:
                continue

            observed = _observed_call_for_target(
                chunk,
                target_name,
                target_arity,
                language_name=self.language_name,
            )
            if observed is None:
                continue

            chunk_name = str(chunk.get("name", ""))
            arity_match = False
            if target_arity is not None:
                arity_match = len(observed.get("args", [])) == target_arity
            cg = _get_call_graph_entry(self.call_graph, cid)
            resolved = set(_to_int_ids(cg.get("resolved_calls", [])))
            edge_to_target = target_id in resolved
            caller_code = str(chunk.get("code", ""))
            caller_reads_file = self.adapter.caller_reads_cli_file_data(caller_code)
            source_vars = _extract_file_source_vars(caller_code, language_name=self.language_name)
            flow_vars = _extract_file_data_flow_symbols(caller_code, source_vars, language_name=self.language_name)
            observed_args = list(observed.get("args", []))
            target_uses_file_data = False
            for a in observed_args:
                if self.adapter.is_cli_file_expr(a or ""):
                    target_uses_file_data = True
                    break
                if _arg_uses_any_var(a, flow_vars, language_name=self.language_name):
                    target_uses_file_data = True
                    break

            # Higher is better.
            score = (
                1 if edge_to_target else 0,
                1 if cid in edge_callers else 0,
                1 if chunk_name in secondary_names else 0,
                1 if chunk_name.lower() == "main" else 0,
                1 if (needs_file and target_uses_file_data) else 0,
                1 if (needs_file and caller_reads_file) else 0,
                1 if arity_match else 0,
                _target_quality_score(chunk),
            )
            candidate = {
                "score": score,
                "chunk": chunk,
                "observed": observed,
                "arity_match": arity_match,
                "caller_reads_file": caller_reads_file,
                "target_uses_file_data": target_uses_file_data,
            }
            # Prefer non-self callers for clearer real-usage grounding.
            # Same-name callers (overloads/recursive-like patterns) are only a fallback.
            if chunk_name == target_name:
                fallback_same_name_candidates.append(candidate)
            else:
                primary_candidates.append(candidate)

        candidates = primary_candidates if primary_candidates else fallback_same_name_candidates
        if not candidates:
            return None, None

        # Prefer exact-arity call-site when at least one exists.
        if any(c["arity_match"] for c in candidates):
            candidates = [c for c in candidates if c["arity_match"]]

        # For file-driven examples, prefer callers that actually read argv[1]
        # and route file-derived vars into target-call arguments.
        if needs_file:
            if any(c["caller_reads_file"] for c in candidates):
                candidates = [c for c in candidates if c["caller_reads_file"]]
            if any(c["target_uses_file_data"] for c in candidates):
                candidates = [c for c in candidates if c["target_uses_file_data"]]

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        return best["chunk"], best["observed"]

    def resolve(
        self,
        *,
        frags: List[Dict],
        analysis: Optional[Dict] = None,
        candidate_ids: Optional[List[int]] = None,
        top_k: int = 40,
    ) -> Dict:
        analysis = analysis or {}
        frags = frags or []

        target = self._resolve_target(frags, analysis)
        if target is None:
            return {
                "target": None,
                "target_id": None,
                "caller": None,
                "caller_id": None,
                "observed_call": None,
                "arg_shapes": [],
                "file_data_flow_hints": {
                    "requires_file_data": False,
                    "caller_reads_argv1": False,
                    "source_vars": [],
                    "target_uses_file_data": False,
                    "argv1_direct_to_target": False,
                    "evidence": [],
                },
            }

        target_id = int(target.get("id", -1))
        pool_ids = self._collect_expanded_pool_ids(
            frags=frags,
            target=target,
            analysis=analysis,
            candidate_ids=candidate_ids,
            top_k=max(8, int(top_k)),
        )
        caller, observed_call = self._pick_caller(pool_ids=pool_ids, target=target, analysis=analysis)
        caller_id = int(caller.get("id", -1)) if caller is not None else None

        params = list(target.get("parameters") or [])
        observed_args = list((observed_call or {}).get("args", []))
        arg_shapes: List[Dict] = []
        for i, p in enumerate(params):
            p_name = str(p.get("name", "")).strip() or f"arg{i}"
            p_type = str(p.get("type", "")).strip()
            arg_shapes.append({
                "index": i,
                "name": p_name,
                "type": p_type,
                "shape": _classify_param_shape(p_type, p_name, language_name=self.language_name),
                "observed_arg": observed_args[i] if i < len(observed_args) else None,
            })

        requires_file_data = bool(analysis.get("needs_file"))
        caller_code = str((caller or {}).get("code", ""))
        source_vars = _extract_file_source_vars(caller_code, language_name=self.language_name)
        flow_vars = _extract_file_data_flow_symbols(caller_code, source_vars, language_name=self.language_name)
        caller_reads_argv1 = self.adapter.caller_reads_cli_file_data(caller_code)

        target_uses_file_data = False
        argv1_direct_to_target = False
        for a in observed_args:
            if self.adapter.is_cli_file_expr(a or ""):
                argv1_direct_to_target = True
                target_uses_file_data = True
                break
            if _arg_uses_any_var(a, flow_vars, language_name=self.language_name):
                target_uses_file_data = True
                break

        evidence = [
            f"target={target.get('name')} @ {target.get('file')}:{target.get('start_line', '?')}-{target.get('end_line', '?')}",
            f"pool_size={len(pool_ids)}",
        ]
        if caller is not None:
            evidence.append(
                f"caller={caller.get('name')} @ {caller.get('file')}:{caller.get('start_line', '?')}-{caller.get('end_line', '?')}"
            )
        if observed_call is not None:
            evidence.append(f"observed_call={observed_call.get('expr')}")
        if source_vars:
            evidence.append(f"file_source_vars={', '.join(source_vars)}")
        if flow_vars:
            evidence.append(f"file_flow_vars={', '.join(flow_vars)}")

        return {
            "target": target,
            "target_id": target_id if target_id >= 0 else None,
            "caller": caller,
            "caller_id": caller_id if caller_id is not None and caller_id >= 0 else None,
            "observed_call": observed_call,
            "arg_shapes": arg_shapes,
            "file_data_flow_hints": {
                "requires_file_data": requires_file_data,
                "caller_reads_argv1": caller_reads_argv1,
                "source_vars": source_vars,
                "flow_vars": flow_vars,
                "target_uses_file_data": target_uses_file_data,
                "argv1_direct_to_target": argv1_direct_to_target,
                "evidence": evidence,
            },
            "language": self.language_name,
        }


def build_example_context_grounded(
    *,
    frags: List[Dict],
    analysis: Optional[Dict],
    meta: List[Dict],
    symbols: Dict[str, List[int]],
    call_graph: Dict,
    called_by: Optional[Dict] = None,
    candidate_ids: Optional[List[int]] = None,
    top_k: int = 40,
) -> Dict:
    resolver = ExampleGroundingResolver(
        meta=meta,
        symbols=symbols,
        call_graph=call_graph,
        called_by=called_by,
    )
    return resolver.resolve(
        frags=frags,
        analysis=analysis,
        candidate_ids=candidate_ids,
        top_k=top_k,
    )
