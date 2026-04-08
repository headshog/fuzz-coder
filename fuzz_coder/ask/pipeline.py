from __future__ import annotations

import time
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from .consistency_runner import (
    CandidateEval,
    build_candidate_prompts,
    is_good_candidate,
    pick_best_candidate,
    score_example_verification,
    should_run_consistency_runner,
)


LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass
class PipelineConfig:
    top_k: int = 10
    rerank_top_k: int = 5
    max_prompt_chars: int = 20000
    model: str = "qwen2.5-coder:32b"
    verbose: bool = False
    shadow_mode: bool = False


@dataclass
class PipelineResult:
    answer: str
    history_answer: str
    elapsed: float
    confidence: float
    verification_flags: Dict[str, Any]
    used_fallback: bool
    used_second_pass: bool
    analysis: Dict[str, Any]
    prompt: str
    llm_ok: bool
    reasoning_path: str = "single_pass"
    show_response_time: bool = True


def _print_query_analysis(analysis):
    print(f"\n[Query Analysis]")
    print(f"  Type: {analysis['query_type']}")
    print(f"  Is listing: {analysis['is_listing']}")
    print(f"  Follow-up: {analysis['follow_up']}")
    print(f"  Exclude previously listed: {analysis['exclude_previously_listed']}")
    print(f"  Needs stdin: {analysis['needs_stdin']}")
    print(f"  Needs file: {analysis['needs_file']}")
    print(f"  Needs output: {analysis['needs_output']}")
    print(f"  Needs memory mgmt: {analysis['needs_memory_mgmt']}")
    print(f"  Needs error handling: {analysis['needs_error_handling']}")
    print(f"  Needs params: {analysis['needs_params']}")
    print(f"  Needs param semantics: {analysis.get('needs_param_semantics', False)}")
    print(f"  Needs parse-like: {analysis['needs_parse_like']}")
    print(f"  Needs fuzz-targets: {analysis['needs_fuzz_targets']}")
    print(f"  Path filters: {analysis['path_filters']}")
    print(f"  Constraint mode: {analysis['constraint_mode']}")
    print(f"  Max params: {analysis.get('max_param_count')}")
    print(f"  Broad fuzz surface mode: {analysis.get('needs_broad_fuzz_surface', False)}")
    print(f"  Listing target count: {analysis.get('listing_target_count')}")
    print(f"  Exclude output/write-like: {analysis['exclude_output']}")
    print(f"  Requested types: {analysis['requested_types']}")
    print(f"  Mentioned functions: {analysis['function_names']}")
    print(f"  Primary function: {analysis.get('primary_function_name')}")
    print(f"  Keywords: {analysis['keywords'][:5]}...")
    if analysis.get("_shadow_diff"):
        print(f"  Shadow diff keys: {sorted(list(analysis['_shadow_diff'].keys()))[:8]}")


def _compute_retrieval(
    core,
    q,
    analysis,
    planner,
    idx,
    lex,
    meta,
    embed_model,
    top_k,
    rerank_top_k,
    excluded_prev_ids,
):
    candidate_ids = planner.get_search_candidates(analysis, k=top_k)
    if excluded_prev_ids:
        candidate_ids = [cid for cid in candidate_ids if cid not in excluded_prev_ids]

    is_listing = analysis.get("query_type") == "listing"
    desired_listing_count = int(analysis.get("listing_target_count") or 0) if is_listing else 0
    semantic_k = top_k * (4 if is_listing else 2)
    lexical_k = top_k * (6 if is_listing else 3)
    retrieval_budget = max(top_k * (12 if is_listing else 6), rerank_top_k * 4)
    if is_listing and desired_listing_count > 0:
        semantic_k = max(semantic_k, desired_listing_count * 4)
        lexical_k = max(lexical_k, desired_listing_count * 6)
        retrieval_budget = max(retrieval_budget, desired_listing_count * 6)
    if is_listing:
        retrieval_budget = max(retrieval_budget, 150)
    retrieval_budget = min(len(meta), retrieval_budget)

    emb = core.embed(q, embed_model)
    sem_ids, _ = core.semantic_search(idx, emb, semantic_k)
    lex_ids, _ = core.lexical_search(q, lex)
    lex_ranked_ids = lex_ids[:lexical_k]

    fused_ids, _ = core.reciprocal_rank_fusion(
        [candidate_ids, sem_ids.tolist(), lex_ranked_ids],
        rrf_k=50,
    )
    all_candidates = fused_ids[:retrieval_budget]
    if excluded_prev_ids:
        all_candidates = [cid for cid in all_candidates if cid not in excluded_prev_ids]

    return {
        "candidate_ids": candidate_ids,
        "sem_ids": sem_ids,
        "lex_ranked_ids": lex_ranked_ids,
        "all_candidates": all_candidates,
        "retrieval_budget": retrieval_budget,
    }


def _print_retrieval_stats(stats):
    print(f"\n[Retrieval]")
    print(f"  Special index candidates: {len(stats['candidate_ids'])}")
    print(f"  Semantic search candidates: {len(stats['sem_ids'])}")
    print(f"  Lexical search candidates: {len(stats['lex_ranked_ids'])}")
    print(f"  Fused candidates (RRF): {len(stats['all_candidates'])}")
    print(f"  Retrieval budget: {stats['retrieval_budget']}")


def _rerank_candidates(core, q, analysis, all_candidates, meta, reranker, rerank_top_k, excluded_prev_ids):
    effective_rerank_top_k = rerank_top_k
    if analysis.get("query_type") == "listing":
        desired_listing_count = int(analysis.get("listing_target_count") or 0)
        if desired_listing_count > 0:
            effective_rerank_top_k = max(rerank_top_k, min(desired_listing_count, 40))
        else:
            effective_rerank_top_k = 10
    elif analysis.get("query_type") == "example_generation":
        effective_rerank_top_k = max(rerank_top_k, 8)
    elif analysis.get("query_type") == "parameter_analysis":
        effective_rerank_top_k = max(rerank_top_k, 8)

    rerank_pool_size = max(effective_rerank_top_k * 4, effective_rerank_top_k + 10)

    if reranker and all_candidates:
        texts = []
        for cid in all_candidates:
            c = meta[cid]
            texts.append(f"{c['name']} {c['signature']} {c['code'][:500]}")

        pairs = [[q, t] for t in texts]
        rerank_scores = reranker.predict(pairs)
        cross_sorted = sorted(
            [(all_candidates[i], rerank_scores[i]) for i in range(len(all_candidates))],
            key=lambda x: x[1],
            reverse=True,
        )
        cross_ranked_ids = [x[0] for x in cross_sorted]

        heuristic_ranked = core.rerank_chunks(q, all_candidates, meta, analysis=analysis)
        heuristic_ranked_ids = [x["idx"] for x in heuristic_ranked]

        rerank_fused_ids, _ = core.reciprocal_rank_fusion(
            [cross_ranked_ids, heuristic_ranked_ids],
            rrf_k=20,
        )
        ranked_pool_ids = rerank_fused_ids[:rerank_pool_size]
    else:
        reranked = core.rerank_chunks(q, all_candidates, meta, analysis=analysis)
        ranked_pool_ids = [x["idx"] for x in reranked[:rerank_pool_size]]

    if excluded_prev_ids:
        ranked_pool_ids = [cid for cid in ranked_pool_ids if cid not in excluded_prev_ids]

    return ranked_pool_ids, effective_rerank_top_k


def _select_ranked_ids(core, analysis, ranked_pool_ids, effective_rerank_top_k, meta, symbols, call_graph):
    fuzz_fallback_applied = False

    if analysis.get("query_type") in {"example_generation", "parameter_analysis", "function_specific", "implementation_explanation"}:
        mentioned_ids = []
        primary_name = analysis.get("primary_function_name")
        if analysis.get("query_type") == "example_generation" and primary_name:
            focus_names = [primary_name]
        else:
            focus_names = list(analysis.get("function_names", []))

        for fn in focus_names:
            mentioned_ids.extend(symbols.get(fn, []))

        caller_ids = []
        callee_ids = []
        for mid in mentioned_ids:
            cg = call_graph.get(str(mid), {})
            caller_ids.extend(cg.get("called_by", []))
            callee_ids.extend(cg.get("resolved_calls", []))

        ranked_ids = []
        secondary_ids = []
        for fn in analysis.get("function_names", []):
            if fn in focus_names:
                continue
            secondary_ids.extend(symbols.get(fn, []))

        for cid in mentioned_ids + caller_ids + callee_ids + ranked_pool_ids:
            if cid not in ranked_ids:
                ranked_ids.append(cid)

        for cid in secondary_ids:
            if cid in ranked_ids:
                ranked_ids.remove(cid)
        ranked_ids = mentioned_ids + [cid for cid in secondary_ids if cid not in mentioned_ids] + [
            cid for cid in ranked_ids if cid not in mentioned_ids and cid not in secondary_ids
        ]

        ranked_ids = ranked_ids[:effective_rerank_top_k]
    elif analysis.get("query_type") == "listing":
        ranked_ids = core.prefilter_listing_candidates(ranked_pool_ids, meta, analysis)[:effective_rerank_top_k]
        if not ranked_ids and analysis.get("needs_fuzz_targets") and ranked_pool_ids:
            fuzz_fallback_applied = True
            ranked_ids = sorted(
                ranked_pool_ids,
                key=lambda cid: core.fuzz_target_score(meta[cid]),
                reverse=True,
            )[:effective_rerank_top_k]
    else:
        ranked_ids = ranked_pool_ids[:effective_rerank_top_k]

    ranked_ids = [cid for cid in ranked_ids if core.is_valid_function_chunk(meta[cid])]
    return ranked_ids, fuzz_fallback_applied


def _print_rerank_stats(effective_rerank_top_k, ranked_pool_ids, ranked_ids, excluded_prev_ids, analysis, fuzz_fallback_applied):
    print(f"\n[Reranking]")
    print(f"  Effective rerank_top_k: {effective_rerank_top_k}")
    print(f"  Rerank pool size: {len(ranked_pool_ids)}")
    print(f"  Selected top {len(ranked_ids)} chunks")
    if excluded_prev_ids:
        print(f"  Excluded previously listed ids: {len(excluded_prev_ids)}")
    if analysis.get("query_type") == "listing":
        print(f"  Strict pre-filter applied: yes")
    if fuzz_fallback_applied:
        print(f"  Fuzz-target fallback applied: yes")


def _build_prompt_and_call_llm(
    core,
    q,
    analysis,
    frags,
    conversation_history,
    max_prompt_chars,
    model,
    verbose,
    example_context=None,
    function_hints=None,
):
    prompt_history = conversation_history if (conversation_history and analysis.get("follow_up")) else None
    prompt = core.build_prompt(
        frags,
        q,
        analysis,
        prompt_history,
        max_prompt_chars=max_prompt_chars,
        example_context=example_context,
        function_hints=function_hints,
    )

    if verbose:
        print(f"\n[Generation]")
        print(f"  Context size: {len(frags)} functions")
        print(f"  Prompt length: {len(prompt)} chars")

    llm_result = core.call_llm(prompt, model)
    if llm_result.get("ok"):
        ans = llm_result.get("response", "")
    else:
        ans = f"Error calling LLM: {llm_result.get('error', 'unknown error')}"
        if verbose:
            print(f"\n[LLM ERROR]")
            print(f"  {llm_result.get('error', 'unknown error')}")

    return prompt, llm_result, ans


def _print_confidence(verification, verbose, prefix):
    if not verbose:
        return
    print(
        f"  {prefix} confidence: {verification.get('confidence_level', 'n/a')}"
        f" ({verification.get('confidence_score', 0.0):.2f})"
    )
    if verification.get("consistency_issues"):
        print(f"  Consistency issues: {verification['consistency_issues']}")


def _verify_answer(
    core,
    ans,
    llm_result,
    analysis,
    frags,
    symbols,
    verbose,
    model,
    prompt,
    question,
    example_context=None,
    function_hints=None,
    type_init_index=None,
):
    verification = {
        "is_valid": True,
        "hallucinated": set(),
        "out_of_context": set(),
        "file_mismatches": set(),
        "signature_mismatches": set(),
        "context_mismatches": set(),
        "consistency_issues": set(),
        "missing_requirements": set(),
        "confidence_score": 1.0,
        "confidence_level": "high",
    }
    used_fallback = False
    used_second_pass = False
    reasoning_path = "single_pass"

    if not llm_result.get("ok"):
        return ans, verification, used_fallback, used_second_pass, reasoning_path

    query_type = analysis.get("query_type")
    if query_type == "listing":
        verification = core.verify_answer_with_context(
            ans,
            frags,
            known_functions=set(symbols.keys()),
        )
        _print_confidence(verification, verbose, "Listing")
        has_verification_issues = any([
            verification.get("hallucinated"),
            verification.get("out_of_context"),
            verification.get("file_mismatches"),
            verification.get("signature_mismatches"),
            verification.get("context_mismatches"),
        ])
        low_confidence = float(verification.get("confidence_score", 0.0)) < LOW_CONFIDENCE_THRESHOLD
        if has_verification_issues or low_confidence:
            if verbose:
                print(f"\n[⚠️  VERIFICATION WARNING]")
                if verification.get("hallucinated"):
                    print(f"  Unknown functions (not found in index): {verification['hallucinated']}")
                if verification.get("out_of_context"):
                    print(f"  Mentioned but not in current context: {verification['out_of_context']}")
                if verification.get("file_mismatches"):
                    print(f"  File mismatch vs indexed context: {verification['file_mismatches']}")
                if verification.get("signature_mismatches"):
                    print(f"  Signature mismatch vs indexed context: {verification['signature_mismatches']}")
                if verification.get("context_mismatches"):
                    print(f"  Function/File/Signature tuple mismatch: {verification['context_mismatches']}")
                if verification.get("consistency_issues"):
                    print(f"  Consistency issues: {verification['consistency_issues']}")
                if low_confidence:
                    print(
                        f"  Low confidence route activated: "
                        f"{verification.get('confidence_level')} ({verification.get('confidence_score', 0.0):.2f})"
                    )
                print(f"  Consider re-querying with path/module filter or increasing context")
            ans = core.build_listing_answer_from_context(frags, analysis=analysis)
            used_fallback = True
            if verbose:
                print("  Replaced model output with deterministic context-based listing")
    elif query_type == "example_generation":
        verification = core.verify_example_answer_with_context(
            ans,
            frags,
            target_function=analysis.get("primary_function_name"),
            known_functions=set(symbols.keys()),
            example_context=example_context,
            type_init_index=type_init_index,
        )
        _print_confidence(verification, verbose, "Example")
        force_fallback = False
        should_second_pass, second_pass_reasons = should_run_consistency_runner(verification)
        if should_second_pass:
            used_second_pass = True
            if verbose:
                print("\n[Example Second Pass]")
                print("  Running consistency repair pass on model output...")
                if second_pass_reasons:
                    print(f"  Trigger reason(s): {second_pass_reasons}")

            candidate_prompts = build_candidate_prompts(
                original_prompt=prompt,
                question=question,
                first_answer=ans,
                verification=verification,
            )
            candidate_evals: List[CandidateEval] = []
            for idx, cprompt in enumerate(candidate_prompts, 1):
                cand_result = core.call_llm(cprompt, model, temperature=0.0)
                if not cand_result.get("ok"):
                    if verbose:
                        print(f"  Candidate pass {idx} failed: {cand_result.get('error', 'unknown error')}")
                    continue
                cand_ans = cand_result.get("response", "")
                cand_ver = core.verify_example_answer_with_context(
                    cand_ans,
                    frags,
                    target_function=analysis.get("primary_function_name"),
                    known_functions=set(symbols.keys()),
                    example_context=example_context,
                    type_init_index=type_init_index,
                )
                _print_confidence(cand_ver, verbose, f"Example(candidate {idx})")
                candidate_evals.append(
                    CandidateEval(
                        answer=cand_ans,
                        verification=cand_ver,
                        score=score_example_verification(cand_ver),
                        is_good=is_good_candidate(cand_ver),
                        label=f"candidate_{idx}",
                    )
                )

            best = pick_best_candidate(candidate_evals)
            if best is not None and best.is_good:
                ans = best.answer
                verification = best.verification
                reasoning_path = "dual_pass_consistent"
                if verbose:
                    print("  Accepted second-pass repaired answer")
                    print(f"  Winner: {best.label}, score={best.score:.2f}")
            else:
                force_fallback = True
                if verbose:
                    print("  No consistent candidate from dual-pass runner; fallback will be used")

        has_verification_issues = (not verification.get("is_valid", False)) or force_fallback
        if has_verification_issues:
            if verbose:
                print(f"\n[⚠️  EXAMPLE VERIFICATION WARNING]")
                if verification.get("missing_requirements"):
                    print(f"  Missing required evidence: {verification['missing_requirements']}")
                if verification.get("hallucinated"):
                    print(f"  Unknown functions (not found in index): {verification['hallucinated']}")
                if verification.get("out_of_context"):
                    print(f"  Mentioned but not in current context: {verification['out_of_context']}")
                if verification.get("file_mismatches"):
                    print(f"  File mismatch vs indexed context: {verification['file_mismatches']}")
                if verification.get("signature_mismatches"):
                    print(f"  Signature mismatch vs indexed context: {verification['signature_mismatches']}")
                if verification.get("context_mismatches"):
                    print(f"  Function/File/Signature tuple mismatch: {verification['context_mismatches']}")
                if verification.get("consistency_issues"):
                    print(f"  Consistency issues: {verification['consistency_issues']}")
                print("  Replacing with deterministic context-grounded example.")
            ans = core.build_example_answer_from_context(
                frags,
                analysis=analysis,
                example_context=example_context,
                symbols=symbols,
                type_init_index=type_init_index or {},
            )
            used_fallback = True
            if used_second_pass:
                reasoning_path = "dual_pass_fallback"
            if verbose:
                print("  Replaced model output with deterministic context-based example")
    elif query_type == "parameter_analysis":
        verification = core.verify_answer_with_context(
            ans,
            frags,
            known_functions=set(symbols.keys()),
        )
        _print_confidence(verification, verbose, "Parameter analysis")
        target_fn = analysis.get("primary_function_name")
        target_missing = bool(target_fn) and (target_fn not in (ans or ""))
        has_signature_line = re.search(r"^\s*Signature:\s*`?.+`?\s*$", ans or "", flags=re.MULTILINE) is not None
        has_function_line = re.search(r"^\s*Function:\s*`?.+`?\s*$", ans or "", flags=re.MULTILINE) is not None
        has_verification_issues = any([
            verification.get("hallucinated"),
            verification.get("out_of_context"),
            verification.get("file_mismatches"),
            verification.get("signature_mismatches"),
            verification.get("context_mismatches"),
        ])
        low_confidence = float(verification.get("confidence_score", 0.0)) < LOW_CONFIDENCE_THRESHOLD
        if has_verification_issues or low_confidence or target_missing or not has_signature_line or not has_function_line:
            if verbose:
                print(f"\n[⚠️  PARAMETER ANALYSIS WARNING]")
                if verification.get("hallucinated"):
                    print(f"  Unknown functions (not found in index): {verification['hallucinated']}")
                if verification.get("out_of_context"):
                    print(f"  Mentioned but not in current context: {verification['out_of_context']}")
                if verification.get("file_mismatches"):
                    print(f"  File mismatch vs indexed context: {verification['file_mismatches']}")
                if verification.get("signature_mismatches"):
                    print(f"  Signature mismatch vs indexed context: {verification['signature_mismatches']}")
                if verification.get("context_mismatches"):
                    print(f"  Function/File/Signature tuple mismatch: {verification['context_mismatches']}")
                if target_missing:
                    print(f"  Target function `{target_fn}` is missing in model output")
                if not has_function_line or not has_signature_line:
                    print("  Missing required structured header: Function/Signature")
                print("  Replacing with deterministic context-grounded parameter analysis.")
            ans = core.build_parameter_analysis_from_context(
                frags,
                analysis=analysis,
                example_context=example_context,
                function_hints=function_hints,
            )
            used_fallback = True
            if verbose:
                print("  Replaced model output with deterministic context-based parameter analysis")

    return ans, verification, used_fallback, used_second_pass, reasoning_path


def _append_verification_note(ans, verification):
    if any([
        verification.get("hallucinated"),
        verification.get("out_of_context"),
        verification.get("file_mismatches"),
        verification.get("signature_mismatches"),
        verification.get("context_mismatches"),
        verification.get("consistency_issues"),
        verification.get("missing_requirements"),
        verification.get("confidence_level") == "low",
    ]):
        ans += (
            f"\n\n[Note: Verification flags -> unknown: {verification.get('hallucinated', set())}; "
            f"out_of_context: {verification.get('out_of_context', set())}; "
            f"file_mismatch: {verification.get('file_mismatches', set())}; "
            f"signature_mismatch: {verification.get('signature_mismatches', set())}; "
            f"context_mismatch: {verification.get('context_mismatches', set())}; "
            f"consistency_issues: {verification.get('consistency_issues', set())}; "
            f"missing_requirements: {verification.get('missing_requirements', set())}; "
            f"confidence: {verification.get('confidence_level', 'n/a')}:{verification.get('confidence_score', 0.0):.2f}]"
        )
    return ans


class QueryPipeline:
    def __init__(
        self,
        *,
        core_module,
        planner,
        idx,
        lex,
        meta,
        embed_model,
        reranker,
        symbols,
        call_graph,
        config: PipelineConfig,
        called_by=None,
        function_hints=None,
        type_init_index=None,
        shadow_runner: Optional[Callable[[str, List[Any]], PipelineResult]] = None,
    ):
        self.core = core_module
        self.planner = planner
        self.idx = idx
        self.lex = lex
        self.meta = meta
        self.embed_model = embed_model
        self.reranker = reranker
        self.symbols = symbols
        self.call_graph = call_graph
        self.called_by = called_by or {}
        self.function_hints = function_hints or {}
        self.type_init_index = type_init_index or {}
        self.config = config
        self._shadow_runner = shadow_runner

    def _run_once(self, q, conversation_history):
        start_time = time.time()
        analysis = self.planner.analyze_query(q, conversation_history if conversation_history else None)

        excluded_prev_ids = set()
        if analysis.get("exclude_previously_listed"):
            excluded_prev_ids = self.planner.collect_previously_listed_ids(conversation_history)

        if (
            analysis.get("query_type") in {"example_generation", "function_specific", "implementation_explanation"}
            and analysis.get("query_function_candidates")
            and not analysis.get("function_names")
        ):
            ans = f"Target function(s) not found in index: {', '.join(analysis['query_function_candidates'][:5])}"
            elapsed = time.time() - start_time
            verification = {
                "confidence_score": 1.0,
                "confidence_level": "high",
            }
            return PipelineResult(
                answer=ans,
                history_answer=ans,
                elapsed=elapsed,
                confidence=1.0,
                verification_flags=verification,
                used_fallback=False,
                used_second_pass=False,
                analysis=analysis,
                prompt="",
                llm_ok=False,
                show_response_time=False,
            )

        if self.config.verbose:
            _print_query_analysis(analysis)

        retrieval = _compute_retrieval(
            self.core,
            q=q,
            analysis=analysis,
            planner=self.planner,
            idx=self.idx,
            lex=self.lex,
            meta=self.meta,
            embed_model=self.embed_model,
            top_k=self.config.top_k,
            rerank_top_k=self.config.rerank_top_k,
            excluded_prev_ids=excluded_prev_ids,
        )
        if self.config.verbose:
            _print_retrieval_stats(retrieval)

        ranked_pool_ids, effective_rerank_top_k = _rerank_candidates(
            self.core,
            q=q,
            analysis=analysis,
            all_candidates=retrieval["all_candidates"],
            meta=self.meta,
            reranker=self.reranker,
            rerank_top_k=self.config.rerank_top_k,
            excluded_prev_ids=excluded_prev_ids,
        )
        ranked_ids, fuzz_fallback_applied = _select_ranked_ids(
            self.core,
            analysis=analysis,
            ranked_pool_ids=ranked_pool_ids,
            effective_rerank_top_k=effective_rerank_top_k,
            meta=self.meta,
            symbols=self.symbols,
            call_graph=self.call_graph,
        )

        if self.config.verbose:
            _print_rerank_stats(
                effective_rerank_top_k=effective_rerank_top_k,
                ranked_pool_ids=ranked_pool_ids,
                ranked_ids=ranked_ids,
                excluded_prev_ids=excluded_prev_ids,
                analysis=analysis,
                fuzz_fallback_applied=fuzz_fallback_applied,
            )

        if analysis.get("query_type") == "listing" and not ranked_ids:
            if analysis.get("exclude_previously_listed"):
                ans = "No additional matching functions found beyond those already listed."
            else:
                ans = "No matching functions found in the indexed codebase for the specified constraints."
            elapsed = time.time() - start_time
            verification = {
                "confidence_score": 1.0,
                "confidence_level": "high",
            }
            return PipelineResult(
                answer=ans,
                history_answer=ans,
                elapsed=elapsed,
                confidence=1.0,
                verification_flags=verification,
                used_fallback=False,
                used_second_pass=False,
                analysis=analysis,
                prompt="",
                llm_ok=False,
                show_response_time=True,
            )

        example_context = None
        frags = [self.meta[i] for i in ranked_ids]
        if analysis.get("query_type") in {"example_generation", "parameter_analysis"}:
            legacy_grounding = os.getenv("FC_EXAMPLE_GROUNDING_LEGACY", "0") == "1"
            use_grounding_v2 = not legacy_grounding
            grounding_shadow = os.getenv("FC_EXAMPLE_GROUNDING_SHADOW", "0") == "1"
            grounding_budget = max(40, effective_rerank_top_k * 5)
            grounding_candidate_ids = []
            for cid in (ranked_pool_ids + retrieval["all_candidates"]):
                if cid not in grounding_candidate_ids:
                    grounding_candidate_ids.append(cid)
                if len(grounding_candidate_ids) >= grounding_budget:
                    break

            v1_context = None
            if not use_grounding_v2 or grounding_shadow:
                v1_context = self.core.build_example_context(frags, analysis=analysis)

            v2_context = None
            if use_grounding_v2 or grounding_shadow:
                v2_context = self.core.build_example_context_grounded(
                    frags=frags,
                    analysis=analysis,
                    meta=self.meta,
                    symbols=self.symbols,
                    call_graph=self.call_graph,
                    called_by=self.called_by,
                    candidate_ids=grounding_candidate_ids,
                    top_k=grounding_budget,
                )

            example_context = v2_context if use_grounding_v2 else (v1_context or v2_context or {})

            target_id = example_context.get("target_id")
            caller_id = example_context.get("caller_id")
            if target_id is not None or caller_id is not None:
                with_grounding = []
                if target_id is not None:
                    with_grounding.append(int(target_id))
                if caller_id is not None:
                    with_grounding.append(int(caller_id))
                with_grounding.extend(ranked_ids)
                deduped = []
                for cid in with_grounding:
                    if cid not in deduped:
                        deduped.append(cid)
                ranked_ids = deduped[:effective_rerank_top_k]
                frags = [self.meta[i] for i in ranked_ids]

            if grounding_shadow and self.config.verbose:
                def _ctx_name(ctx, key):
                    node = (ctx or {}).get(key) or {}
                    return node.get("name")

                def _ctx_call(ctx):
                    return ((ctx or {}).get("observed_call") or {}).get("expr")

                diffs = []
                if _ctx_name(v1_context, "target") != _ctx_name(v2_context, "target"):
                    diffs.append("target")
                if _ctx_name(v1_context, "caller") != _ctx_name(v2_context, "caller"):
                    diffs.append("caller")
                if _ctx_call(v1_context) != _ctx_call(v2_context):
                    diffs.append("observed_call")
                if diffs:
                    print("\n[Example Grounding Shadow]")
                    print(f"  Diff keys: {sorted(diffs)}")

            if self.config.verbose and example_context.get("target"):
                caller_name = (example_context.get("caller") or {}).get("name")
                observed = (example_context.get("observed_call") or {}).get("expr")
                print("  Example grounding:")
                print(f"    target: {example_context['target'].get('name')}")
                print(f"    caller: {caller_name or 'not found'}")
                print(f"    observed_call: {observed or 'not found'}")

        function_hints = None
        if analysis.get("query_type") == "parameter_analysis":
            primary = analysis.get("primary_function_name")
            targets = []
            if primary:
                targets.append(primary)
            for fn in analysis.get("function_names", []):
                if fn not in targets:
                    targets.append(fn)
            function_hints = {}
            for fn in targets:
                hints = list(self.function_hints.get(fn, []))
                if hints:
                    function_hints[fn] = hints

        prompt, llm_result, ans = _build_prompt_and_call_llm(
            self.core,
            q=q,
            analysis=analysis,
            frags=frags,
            conversation_history=conversation_history,
            max_prompt_chars=self.config.max_prompt_chars,
            model=self.config.model,
            verbose=self.config.verbose,
            example_context=example_context,
            function_hints=function_hints,
        )

        ans, verification, used_fallback, used_second_pass, reasoning_path = _verify_answer(
            self.core,
            ans=ans,
            llm_result=llm_result,
            analysis=analysis,
            frags=frags,
            symbols=self.symbols,
            verbose=self.config.verbose,
            model=self.config.model,
            prompt=prompt,
            question=q,
            example_context=example_context,
            function_hints=function_hints,
            type_init_index=self.type_init_index,
        )

        elapsed = time.time() - start_time
        history_ans = _append_verification_note(ans, verification)
        confidence = float(verification.get("confidence_score", 1.0))

        return PipelineResult(
            answer=ans,
            history_answer=history_ans,
            elapsed=elapsed,
            confidence=confidence,
            verification_flags=verification,
            used_fallback=bool(used_fallback or fuzz_fallback_applied),
            used_second_pass=used_second_pass,
            analysis=analysis,
            prompt=prompt,
            llm_ok=bool(llm_result.get("ok", False)),
            reasoning_path=reasoning_path,
            show_response_time=True,
        )

    def run(self, q, conversation_history):
        if not self.config.shadow_mode:
            return self._run_once(q, conversation_history)

        # Shadow mode: run both paths and compare key outputs, return main path result.
        # If shadow_runner is provided, it can execute an alternate orchestration path
        # (e.g., legacy pipeline) for contract validation without affecting user output.
        primary = self._run_once(q, conversation_history)
        shadow = self._shadow_runner(q, conversation_history) if self._shadow_runner else self._run_once(q, conversation_history)
        diffs = {}
        if primary.answer != shadow.answer:
            diffs["answer"] = {"primary": primary.answer, "shadow": shadow.answer}
        if primary.confidence != shadow.confidence:
            diffs["confidence"] = {"primary": primary.confidence, "shadow": shadow.confidence}
        if primary.used_fallback != shadow.used_fallback:
            diffs["used_fallback"] = {"primary": primary.used_fallback, "shadow": shadow.used_fallback}
        if primary.used_second_pass != shadow.used_second_pass:
            diffs["used_second_pass"] = {"primary": primary.used_second_pass, "shadow": shadow.used_second_pass}

        if diffs:
            primary.verification_flags = dict(primary.verification_flags)
            primary.verification_flags["_pipeline_shadow_diff"] = diffs
            if self.config.verbose:
                print(f"\n[Pipeline Shadow]")
                print(f"  Diff keys: {sorted(diffs.keys())}")
        return primary
