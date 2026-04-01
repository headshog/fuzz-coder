from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder

from fuzz_coder.embeddings.registry import get_embedding_backend

from . import core


HELP_QUERIES = {
    "help",
    "h",
    "?",
    "what can you do",
    "what can you do?",
    "help me",
    "помощь",
    "справка",
    "что ты умеешь",
    "что умеешь",
}

MAX_HISTORY = 5


def is_help_query(q: str) -> bool:
    qn = q.lower().strip()
    if qn in HELP_QUERIES:
        return True
    return any(p in qn for p in [
        "what can you do",
        "show examples",
        "как пользоваться",
        "покажи примеры",
        "что ты умеешь",
    ])


def render_help_text() -> str:
    return (
        "I can analyze indexed code and answer questions about functions, types, call flows, and fuzz targets.\n\n"
        "What I can do:\n"
        "1. List functions by criteria (stdin/file/API/types/parse/fuzz).\n"
        "2. Filter by module/subdirectory (path filter).\n"
        "3. Give examples of function usage.\n"
        "4. Explain implementation details from indexed code.\n"
        "5. Handle follow-ups (including 'other functions' without repeats).\n\n"
        "Example queries:\n"
        "- List functions good for fuzzing from module src/parsers\n"
        "- Какие функции читают из stdin?\n"
        "- Покажи функции с параметром std::string\n"
        "- Дай список других функций для фаззинга из директории src/parsers\n"
        "- Explain how decode_binary_blob works\n"
        "- Give an example calling parse_json_payload\n"
    )


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--model", default="qwen2.5-coder:32b")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--rerank_top_k", type=int, default=5)
    ap.add_argument("--max_prompt_chars", type=int, default=20000)
    ap.add_argument("--embedding_backend", default="sentence_transformers")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _load_indices(index_dir: Path):
    print("Loading indices...")
    idx = faiss.read_index(str(index_dir / "semantic.faiss"))
    meta = core.load_meta(index_dir)
    lex = core.load_json(index_dir / "lexical_index.json")
    special_indices = core.load_json(index_dir / "special_indices.json")
    symbols = core.load_json(index_dir / "symbols.json")
    call_graph = core.load_json(index_dir / "call_graph.json")
    called_by = core.load_json(index_dir / "called_by.json")
    return idx, meta, lex, special_indices, symbols, call_graph, called_by


def _load_models(args):
    print("Loading embedding model...")
    embed_model = get_embedding_backend(args.embed_model, backend=args.embedding_backend)

    reranker = None
    try:
        print("Loading reranker model...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        print("Reranker not available, using heuristic reranking")

    return embed_model, reranker


def _trim_history(history):
    while len(history) > MAX_HISTORY:
        history.pop(0)


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
    print(f"  Needs parse-like: {analysis['needs_parse_like']}")
    print(f"  Needs fuzz-targets: {analysis['needs_fuzz_targets']}")
    print(f"  Path filters: {analysis['path_filters']}")
    print(f"  Constraint mode: {analysis['constraint_mode']}")
    print(f"  Exclude output/write-like: {analysis['exclude_output']}")
    print(f"  Requested types: {analysis['requested_types']}")
    print(f"  Mentioned functions: {analysis['function_names']}")
    print(f"  Primary function: {analysis.get('primary_function_name')}")
    print(f"  Keywords: {analysis['keywords'][:5]}...")


def _compute_retrieval(
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
    semantic_k = top_k * (4 if is_listing else 2)
    lexical_k = top_k * (6 if is_listing else 3)
    retrieval_budget = max(top_k * (12 if is_listing else 6), rerank_top_k * 4)
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


def _rerank_candidates(q, analysis, all_candidates, meta, reranker, rerank_top_k, excluded_prev_ids):
    effective_rerank_top_k = rerank_top_k
    if analysis.get("query_type") == "listing":
        effective_rerank_top_k = 10
    elif analysis.get("query_type") == "example_generation":
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


def _select_ranked_ids(analysis, ranked_pool_ids, effective_rerank_top_k, meta, symbols, call_graph):
    fuzz_fallback_applied = False

    if analysis.get("query_type") in {"example_generation", "function_specific", "implementation_explanation"}:
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

        # Keep explicitly requested companion functions (e.g., "from main")
        # close to the target so they are not truncated away from context.
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


def _build_prompt_and_call_llm(q, analysis, frags, conversation_history, max_prompt_chars, model, verbose):
    prompt_history = conversation_history if (conversation_history and analysis.get("follow_up")) else None
    prompt = core.build_prompt(
        frags,
        q,
        analysis,
        prompt_history,
        max_prompt_chars=max_prompt_chars,
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


def _verify_answer(ans, llm_result, analysis, frags, symbols, verbose):
    verification = {
        "is_valid": True,
        "hallucinated": set(),
        "out_of_context": set(),
        "file_mismatches": set(),
        "signature_mismatches": set(),
        "context_mismatches": set(),
        "missing_requirements": set(),
    }

    if not llm_result.get("ok"):
        return ans, verification

    query_type = analysis.get("query_type")
    if query_type == "listing":
        verification = core.verify_answer_with_context(
            ans,
            frags,
            known_functions=set(symbols.keys()),
        )
        has_verification_issues = any([
            verification.get("hallucinated"),
            verification.get("out_of_context"),
            verification.get("file_mismatches"),
            verification.get("signature_mismatches"),
            verification.get("context_mismatches"),
        ])
        if has_verification_issues:
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
                print(f"  Consider re-querying with path/module filter or increasing context")
            ans = core.build_listing_answer_from_context(frags, analysis=analysis)
            if verbose:
                print("  Replaced model output with deterministic context-based listing")
    elif query_type == "example_generation":
        verification = core.verify_example_answer_with_context(
            ans,
            frags,
            target_function=analysis.get("primary_function_name"),
            known_functions=set(symbols.keys()),
        )
        has_verification_issues = not verification.get("is_valid", False)
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
                print("  Replacing with deterministic context-grounded example.")
            ans = core.build_example_answer_from_context(frags, analysis=analysis)
            if verbose:
                print("  Replaced model output with deterministic context-based example")

    return ans, verification


def _append_verification_note(ans, verification):
    if any([
        verification.get("hallucinated"),
        verification.get("out_of_context"),
        verification.get("file_mismatches"),
        verification.get("signature_mismatches"),
        verification.get("context_mismatches"),
        verification.get("missing_requirements"),
    ]):
        ans += (
            f"\n\n[Note: Verification flags -> unknown: {verification.get('hallucinated', set())}; "
            f"out_of_context: {verification.get('out_of_context', set())}; "
            f"file_mismatch: {verification.get('file_mismatches', set())}; "
            f"signature_mismatch: {verification.get('signature_mismatches', set())}; "
            f"context_mismatch: {verification.get('context_mismatches', set())}; "
            f"missing_requirements: {verification.get('missing_requirements', set())}]"
        )
    return ans


def main():
    args = _parse_args()
    index_dir = Path(args.index_dir)

    idx, meta, lex, special_indices, symbols, call_graph, called_by = _load_indices(index_dir)
    embed_model, reranker = _load_models(args)
    planner = core.QueryPlanner(special_indices, symbols, call_graph, called_by, meta=meta)

    conversation_history = []

    print(f"\nReady! Using model: {args.model}")
    print(f"Index contains {len(meta)} functions")
    print("Type 'quit' to exit, or 'help' to see examples\n")

    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break

        if q.lower() in ["quit", "exit", "q"]:
            break
        if not q:
            continue
        if is_help_query(q):
            print("\n" + render_help_text())
            continue

        start_time = time.time()
        analysis = planner.analyze_query(q, conversation_history if conversation_history else None)

        excluded_prev_ids = set()
        if analysis.get("exclude_previously_listed"):
            excluded_prev_ids = planner.collect_previously_listed_ids(conversation_history)

        if (
            analysis.get("query_type") in {"example_generation", "function_specific", "implementation_explanation"}
            and analysis.get("query_function_candidates")
            and not analysis.get("function_names")
        ):
            missing = ", ".join(analysis["query_function_candidates"][:5])
            ans = f"Target function(s) not found in index: {missing}"
            print(f"\n{ans}")
            conversation_history.append((q, ans))
            _trim_history(conversation_history)
            continue

        if args.verbose:
            _print_query_analysis(analysis)

        retrieval = _compute_retrieval(
            q=q,
            analysis=analysis,
            planner=planner,
            idx=idx,
            lex=lex,
            meta=meta,
            embed_model=embed_model,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k,
            excluded_prev_ids=excluded_prev_ids,
        )
        if args.verbose:
            _print_retrieval_stats(retrieval)

        ranked_pool_ids, effective_rerank_top_k = _rerank_candidates(
            q=q,
            analysis=analysis,
            all_candidates=retrieval["all_candidates"],
            meta=meta,
            reranker=reranker,
            rerank_top_k=args.rerank_top_k,
            excluded_prev_ids=excluded_prev_ids,
        )
        ranked_ids, fuzz_fallback_applied = _select_ranked_ids(
            analysis=analysis,
            ranked_pool_ids=ranked_pool_ids,
            effective_rerank_top_k=effective_rerank_top_k,
            meta=meta,
            symbols=symbols,
            call_graph=call_graph,
        )

        if args.verbose:
            _print_rerank_stats(
                effective_rerank_top_k=effective_rerank_top_k,
                ranked_pool_ids=ranked_pool_ids,
                ranked_ids=ranked_ids,
                excluded_prev_ids=excluded_prev_ids,
                analysis=analysis,
                fuzz_fallback_applied=fuzz_fallback_applied,
            )

        if analysis.get("query_type") == "listing" and not ranked_ids:
            elapsed = time.time() - start_time
            if analysis.get("exclude_previously_listed"):
                ans = "No additional matching functions found beyond those already listed."
            else:
                ans = "No matching functions found in the indexed codebase for the specified constraints."
            print(f"\n{ans}")
            print(f"\n[Response time: {elapsed:.2f}s]")
            conversation_history.append((q, ans))
            _trim_history(conversation_history)
            continue

        frags = [meta[i] for i in ranked_ids]
        _, llm_result, ans = _build_prompt_and_call_llm(
            q=q,
            analysis=analysis,
            frags=frags,
            conversation_history=conversation_history,
            max_prompt_chars=args.max_prompt_chars,
            model=args.model,
            verbose=args.verbose,
        )

        ans, verification = _verify_answer(
            ans=ans,
            llm_result=llm_result,
            analysis=analysis,
            frags=frags,
            symbols=symbols,
            verbose=args.verbose,
        )

        elapsed = time.time() - start_time
        print(f"\n{ans}")
        print(f"\n[Response time: {elapsed:.2f}s]")

        ans = _append_verification_note(ans, verification)
        conversation_history.append((q, ans))
        _trim_history(conversation_history)


if __name__ == "__main__":
    main()
