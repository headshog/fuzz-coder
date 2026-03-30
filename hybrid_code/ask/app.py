from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder

from hybrid_code.embeddings.registry import get_embedding_backend

from . import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--model", default="qwen2.5-coder:32b")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--rerank_top_k", type=int, default=5)
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
    planner = core.QueryPlanner(special_indices, symbols, call_graph, called_by)

    # Conversation history for follow-up questions
    conversation_history = []  # List of (question, answer) tuples

    print(f"\nReady! Using model: {args.model}")
    print(f"Index contains {len(meta)} functions")
    print("Type 'quit' to exit\n")

    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break

        if q.lower() in ["quit", "exit", "q"]:
            break

        if not q:
            continue

        start_time = time.time()

        # Step 1: Analyze query with conversation context
        analysis = planner.analyze_query(q, conversation_history if conversation_history else None)

        if args.verbose:
            print(f"\n[Query Analysis]")
            print(f"  Type: {analysis['query_type']}")
            print(f"  Is listing: {analysis['is_listing']}")
            print(f"  Follow-up: {analysis['follow_up']}")
            print(f"  Needs stdin: {analysis['needs_stdin']}")
            print(f"  Needs file: {analysis['needs_file']}")
            print(f"  Needs params: {analysis['needs_params']}")
            print(f"  Needs parse-like: {analysis['needs_parse_like']}")
            print(f"  Needs fuzz-targets: {analysis['needs_fuzz_targets']}")
            print(f"  Constraint mode: {analysis['constraint_mode']}")
            print(f"  Exclude output/write-like: {analysis['exclude_output']}")
            print(f"  Requested types: {analysis['requested_types']}")
            print(f"  Keywords: {analysis['keywords'][:5]}...")

        # Step 2: Get candidates
        candidate_ids = planner.get_search_candidates(analysis, k=args.top_k)

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
            if analysis.get("is_listing"):
                print(f"  Strict pre-filter applied: yes")
            if fuzz_fallback_applied:
                print(f"  Fuzz-target fallback applied: yes")

        # If strict listing constraints removed everything, return deterministic answer
        if analysis.get("is_listing") and not ranked_ids:
            elapsed = time.time() - start_time
            ans = "No matching functions found in the indexed codebase for the specified constraints."
            print(f"\n{ans}")
            print(f"\n[Response time: {elapsed:.2f}s]")
            conversation_history.append((q, ans))
            if len(conversation_history) > 5:
                conversation_history.pop(0)
            continue

        # Step 5: Build context and generate answer
        frags = [meta[i] for i in ranked_ids]
        prompt = core.build_prompt(frags, q, analysis, conversation_history if conversation_history else None)

        if args.verbose:
            print(f"\n[Generation]")
            print(f"  Context size: {len(frags)} functions")
            print(f"  Prompt length: {len(prompt)} chars")

        ans = core.call_llm(prompt, args.model)

        # Step 6: Verify answer for hallucinations
        verification = {"is_valid": True, "hallucinated": set()}
        if args.verbose:
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
