from __future__ import annotations

from typing import Dict, List, Optional

from fuzz_coder.languages.registry import get_ask_language_adapter, infer_ask_language_from_fragments


def _extract_call_argument_lists(code: str, target_name: str, limit: int = 5, language_name: str = "c_cpp") -> List[Dict]:
    adapter = get_ask_language_adapter(language_name)
    return adapter.extract_call_argument_lists(code, target_name, limit=limit)


def _classify_param_shape(param_type: str, param_name: str, language_name: str = "c_cpp") -> str:
    adapter = get_ask_language_adapter(language_name)
    return adapter.classify_param_shape(param_type, param_name)


def _extract_file_source_vars(code: str, language_name: str = "c_cpp") -> List[str]:
    adapter = get_ask_language_adapter(language_name)
    return adapter.extract_file_source_vars(code)


def _extract_file_data_flow_symbols(code: str, source_vars: List[str], language_name: str = "c_cpp") -> List[str]:
    adapter = get_ask_language_adapter(language_name)
    return adapter.extract_file_data_flow_symbols(code, source_vars)


def _arg_uses_any_var(arg_expr: str, names: List[str], language_name: str = "c_cpp") -> bool:
    adapter = get_ask_language_adapter(language_name)
    return adapter.arg_uses_any_var(arg_expr, names)


def _format_file_loc(chunk: Dict) -> str:
    return f"{chunk.get('file', '')}:{chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}"


def _normalize_path(path: str) -> str:
    p = (path or "").replace("\\", "/").lower().strip()
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def _file_matches_any_filter(file_path: str, path_filters: List[str]) -> bool:
    if not path_filters:
        return True
    p = _normalize_path(file_path)
    if not p:
        return False
    for flt in path_filters:
        f = _normalize_path(flt)
        if not f:
            continue
        if f in p:
            return True
    return False


def _target_quality_score(chunk: Dict) -> tuple:
    span = 0
    try:
        span = int(chunk.get("end_line", 0)) - int(chunk.get("start_line", 0))
    except Exception:
        span = 0
    code = str(chunk.get("code", "") or "")
    has_body = ("{" in code and "}" in code) or span >= 5
    params_count = len(list(chunk.get("parameters") or []))
    sig_len = len(str(chunk.get("signature", "") or ""))
    code_len = len(code)
    return (
        1 if has_body else 0,
        params_count,
        span,
        sig_len,
        code_len,
    )


def _choose_target(frags: List[Dict], analysis: Optional[Dict]) -> Optional[Dict]:
    if not frags:
        return None
    primary = (analysis or {}).get("primary_function_name")
    candidates = list(frags)
    if primary:
        primary_candidates = [f for f in frags if f.get("name") == primary]
        if primary_candidates:
            candidates = primary_candidates

    path_filters = list((analysis or {}).get("path_filters") or [])
    filtered = [c for c in candidates if _file_matches_any_filter(str(c.get("file", "")), path_filters)]
    if filtered:
        candidates = filtered

    if not candidates:
        return frags[0]
    return sorted(candidates, key=_target_quality_score, reverse=True)[0]


def _choose_caller(
    frags: List[Dict],
    target: Dict,
    analysis: Optional[Dict],
    *,
    language_name: str,
) -> tuple[Optional[Dict], List[Dict]]:
    target_name = str(target.get("name", "")).strip()
    if not target_name:
        return None, []

    caller_candidates: List[tuple[Dict, List[Dict]]] = []
    for f in frags:
        if f.get("name") == target_name:
            continue
        calls = _extract_call_argument_lists(
            str(f.get("code", "")),
            target_name,
            limit=5,
            language_name=language_name,
        )
        if calls:
            caller_candidates.append((f, calls))

    if not caller_candidates:
        return None, []

    secondary_names = [
        n for n in (analysis or {}).get("function_names", [])
        if n and n != target_name
    ]
    for wanted in secondary_names:
        for c, calls in caller_candidates:
            if c.get("name") == wanted:
                return c, calls

    for c, calls in caller_candidates:
        if str(c.get("name", "")).lower() == "main":
            return c, calls

    return caller_candidates[0]


class ExampleContextBuilder:
    def __init__(
        self,
        frags: List[Dict],
        analysis: Optional[Dict] = None,
        *,
        language_name: Optional[str] = None,
    ):
        self.frags = frags or []
        self.analysis = analysis or {}
        self.language_name = language_name or infer_ask_language_from_fragments(self.frags)
        self.adapter = get_ask_language_adapter(self.language_name)

    def build(self) -> Dict:
        target = _choose_target(self.frags, self.analysis)
        if target is None:
            return {
                "target": None,
                "caller": None,
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

        caller, caller_calls = _choose_caller(
            self.frags,
            target,
            self.analysis,
            language_name=self.language_name,
        )
        observed_call = None
        target_arity = len(list(target.get("parameters") or []))
        if caller_calls:
            preferred = None
            for c in caller_calls:
                if c.get("arity") == target_arity:
                    preferred = c
                    break
            if preferred is None:
                preferred = caller_calls[0]
            observed_call = preferred

        params = list(target.get("parameters") or [])
        observed_args = list((observed_call or {}).get("args", []))
        arg_shapes = []
        for i, p in enumerate(params):
            p_name = str(p.get("name", "")).strip() or f"arg{i}"
            p_type = str(p.get("type", "")).strip()
            arg_shapes.append({
                "index": i,
                "name": p_name,
                "type": p_type,
                "shape": self.adapter.classify_param_shape(p_type, p_name),
                "observed_arg": observed_args[i] if i < len(observed_args) else None,
            })

        requires_file_data = bool(self.analysis.get("needs_file"))
        caller_code = str((caller or {}).get("code", ""))
        source_vars = self.adapter.extract_file_source_vars(caller_code)
        flow_vars = self.adapter.extract_file_data_flow_symbols(caller_code, source_vars)
        caller_reads_argv1 = self.adapter.caller_reads_cli_file_data(caller_code)

        target_uses_file_data = False
        argv1_direct_to_target = False
        for a in observed_args:
            if self.adapter.is_cli_file_expr(a or ""):
                argv1_direct_to_target = True
                target_uses_file_data = True
                break
            if self.adapter.arg_uses_any_var(a, flow_vars):
                target_uses_file_data = True
                break

        evidence = [
            f"target={target.get('name')} @ {_format_file_loc(target)}",
        ]
        if caller is not None:
            evidence.append(f"caller={caller.get('name')} @ {_format_file_loc(caller)}")
        if observed_call is not None:
            evidence.append(f"observed_call={observed_call.get('expr')}")
        if source_vars:
            evidence.append(f"file_source_vars={', '.join(source_vars)}")
        if flow_vars:
            evidence.append(f"file_flow_vars={', '.join(flow_vars)}")

        return {
            "target": target,
            "caller": caller,
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


def build_example_context(
    frags: List[Dict],
    analysis: Optional[Dict] = None,
    *,
    language_name: Optional[str] = None,
) -> Dict:
    return ExampleContextBuilder(frags, analysis=analysis, language_name=language_name).build()
