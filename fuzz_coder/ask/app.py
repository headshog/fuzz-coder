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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--model", default="qwen2.5-coder:32b")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--rerank_top_k", type=int, default=5)
    ap.add_argument("--max_prompt_chars", type=int, default=20000)
    ap.add_argument("--embedding_backend", default="sentence_transformers")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)

    # Load indices
    print("Loading indices...")
    idx = faiss.read_index(str(index_dir/"semantic.faiss"))
    meta = core.load_meta(index_dir)
    lex = core.load_json(index_dir/"lexical_index.json")
    special_indices = core.load_json(index_dir/"special_indices.json")
    symbols = core.load_json(index_dir/"symbols.json")
    call_graph = core.load_json(index_dir/"call_graph.json")
    called_by = core.load_json(index_dir/"called_by.json")

    # Initialize models
    print("Loading embedding model...")
    embed_model = get_embedding_backend(args.embed_model, backend=args.embedding_backend)

    # Try to load cross-encoder for reranking (optional, improves quality)
    reranker = None
    try:
        print("Loading reranker model...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except:
        print("Reranker not available, using heuristic reranking")

    # Initialize query planner
    planner = core.QueryPlanner(special_indices, symbols, call_graph, called_by, meta=meta)

    # Conversation history for follow-up questions
    conversation_history = []  # List of (question, answer) tuples

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

        # Step 1: Analyze query with conversation context
        analysis = planner.analyze_query(q, conversation_history if conversation_history else None)
        excluded_prev_ids = set()
        if analysis.get("exclude_previously_listed"):
            excluded_prev_ids = planner.collect_previously_listed_ids(conversation_history)

        if args.verbose:
            print(f"\n[Query Analysis]")
            print(f"  Type: {analysis['query_type']}")
            print(f"  Is listing: {analysis['is_listing']}")
            print(f"  Follow-up: {analysis['follow_up']}")
            print(f"  Exclude previously listed: {analysis['exclude_previously_listed']}")
            print(f"  Needs stdin: {analysis['needs_stdin']}")
            print(f"  Needs file: {analysis['needs_file']}")
            print(f"  Needs params: {analysis['needs_params']}")
            print(f"  Needs parse-like: {analysis['needs_parse_like']}")
            print(f"  Needs fuzz-targets: {analysis['needs_fuzz_targets']}")
            print(f"  Path filters: {analysis['path_filters']}")
            print(f"  Constraint mode: {analysis['constraint_mode']}")
            print(f"  Exclude output/write-like: {analysis['exclude_output']}")
            print(f"  Requested types: {analysis['requested_types']}")
            print(f"  Keywords: {analysis['keywords'][:5]}...")

        # Step 2: Get candidates
        candidate_ids = planner.get_search_candidates(analysis, k=args.top_k)
        if excluded_prev_ids:
            candidate_ids = [cid for cid in candidate_ids if cid not in excluded_prev_ids]

        # Step 3: Semantic + lexical retrieval with RRF fusion
        is_listing = analysis.get("is_listing", False)
        semantic_k = args.top_k * (4 if is_listing else 2)
        lexical_k = args.top_k * (6 if is_listing else 3)
        retrieval_budget = max(args.top_k * (12 if is_listing else 6), args.rerank_top_k * 4)
        if is_listing:
            retrieval_budget = max(retrieval_budget, 150)
        retrieval_budget = min(len(meta), retrieval_budget)

        emb = core.embed(q, embed_model)
        sem_ids, sem_scores = core.semantic_search(idx, emb, semantic_k)
        lex_ids, lex_scores = core.lexical_search(q, lex, meta)
        lex_ranked_ids = lex_ids[:lexical_k]

        fused_ids, fused_scores = core.reciprocal_rank_fusion(
            [candidate_ids, sem_ids.tolist(), lex_ranked_ids],
            rrf_k=50
        )
        all_candidates = fused_ids[:retrieval_budget]
        if excluded_prev_ids:
            all_candidates = [cid for cid in all_candidates if cid not in excluded_prev_ids]

        if args.verbose:
            print(f"\n[Retrieval]")
            print(f"  Special index candidates: {len(candidate_ids)}")
            print(f"  Semantic search candidates: {len(sem_ids)}")
            print(f"  Lexical search candidates: {len(lex_ranked_ids)}")
            print(f"  Fused candidates (RRF): {len(all_candidates)}")
            print(f"  Retrieval budget: {retrieval_budget}")

        # Step 4: Rerank
        effective_rerank_top_k = args.rerank_top_k
        if analysis.get("is_listing"):
            # Keep listing prompts compact to reduce context bloat.
            effective_rerank_top_k = 10
        rerank_pool_size = max(effective_rerank_top_k * 4, effective_rerank_top_k + 10)

        if reranker and all_candidates:
            # Use cross-encoder + heuristic reranking fusion
            texts = []
            for cid in all_candidates:
                c = meta[cid]
                text = f"{c['name']} {c['signature']} {c['code'][:500]}"
                texts.append(text)

            pairs = [[q, t] for t in texts]
            rerank_scores = reranker.predict(pairs)
            cross_sorted = sorted(
                [(all_candidates[i], rerank_scores[i]) for i in range(len(all_candidates))],
                key=lambda x: x[1],
                reverse=True
            )
            cross_ranked_ids = [x[0] for x in cross_sorted]

            heuristic_ranked = core.rerank_chunks(q, all_candidates, meta, call_graph, analysis)
            heuristic_ranked_ids = [x["idx"] for x in heuristic_ranked]

            rerank_fused_ids, _ = core.reciprocal_rank_fusion(
                [cross_ranked_ids, heuristic_ranked_ids],
                rrf_k=20
            )
            ranked_pool_ids = rerank_fused_ids[:rerank_pool_size]
        else:
            # Use heuristic reranking
            reranked = core.rerank_chunks(q, all_candidates, meta, call_graph, analysis)
            ranked_pool_ids = [x["idx"] for x in reranked[:rerank_pool_size]]
        if excluded_prev_ids:
            ranked_pool_ids = [cid for cid in ranked_pool_ids if cid not in excluded_prev_ids]

        # Step 4.1: strict listing pre-filter before LLM
        fuzz_fallback_applied = False
        if analysis.get("is_listing"):
            ranked_ids = core.prefilter_listing_candidates(ranked_pool_ids, meta, analysis)[:effective_rerank_top_k]
            if not ranked_ids and analysis.get("needs_fuzz_targets") and ranked_pool_ids:
                fuzz_fallback_applied = True
                ranked_ids = sorted(
                    ranked_pool_ids,
                    key=lambda cid: core.fuzz_target_score(meta[cid]),
                    reverse=True
                )[:effective_rerank_top_k]
        else:
            ranked_ids = ranked_pool_ids[:effective_rerank_top_k]

        if args.verbose:
            print(f"\n[Reranking]")
            print(f"  Effective rerank_top_k: {effective_rerank_top_k}")
            print(f"  Rerank pool size: {len(ranked_pool_ids)}")
            print(f"  Selected top {len(ranked_ids)} chunks")
            if excluded_prev_ids:
                print(f"  Excluded previously listed ids: {len(excluded_prev_ids)}")
            if analysis.get("is_listing"):
                print(f"  Strict pre-filter applied: yes")
            if fuzz_fallback_applied:
                print(f"  Fuzz-target fallback applied: yes")

        # If strict listing constraints removed everything, return deterministic answer
        if analysis.get("is_listing") and not ranked_ids:
            elapsed = time.time() - start_time
            if analysis.get("exclude_previously_listed"):
                ans = "No additional matching functions found beyond those already listed."
            else:
                ans = "No matching functions found in the indexed codebase for the specified constraints."
            print(f"\n{ans}")
            print(f"\n[Response time: {elapsed:.2f}s]")
            conversation_history.append((q, ans))
            if len(conversation_history) > 5:
                conversation_history.pop(0)
            continue

        # Step 5: Build context and generate answer
        frags = [meta[i] for i in ranked_ids]
        prompt = core.build_prompt(
            frags,
            q,
            analysis,
            conversation_history if conversation_history else None,
            max_prompt_chars=args.max_prompt_chars,
        )

        if args.verbose:
            print(f"\n[Generation]")
            print(f"  Context size: {len(frags)} functions")
            print(f"  Prompt length: {len(prompt)} chars")

        llm_result = core.call_llm(prompt, args.model)
        if llm_result.get("ok"):
            ans = llm_result.get("response", "")
        else:
            ans = f"Error calling LLM: {llm_result.get('error', 'unknown error')}"
            if args.verbose:
                print(f"\n[LLM ERROR]")
                print(f"  {llm_result.get('error', 'unknown error')}")

        # Step 6: Verify answer for hallucinations
        verification = {"is_valid": True, "hallucinated": set()}
        if args.verbose and llm_result.get("ok"):
            verification = core.verify_answer_with_context(ans, frags)
            if not verification["is_valid"]:
                print(f"\n[⚠️  VERIFICATION WARNING]")
                print(f"  Potentially hallucinated functions: {verification['hallucinated']}")
                print(f"  Consider asking for clarification or re-querying with more context")

        elapsed = time.time() - start_time
        print(f"\n{ans}")
        print(f"\n[Response time: {elapsed:.2f}s]")

        # Add verification note to conversation history if hallucinations detected
        if not verification.get("is_valid", True):
            ans += f"\n\n[Note: Answer may contain unverified function names: {verification['hallucinated']}]"

        # Update conversation history
        conversation_history.append((q, ans))
        # Keep only last 5 exchanges to avoid context explosion
        if len(conversation_history) > 5:
            conversation_history.pop(0)


if __name__ == "__main__":
    main()
