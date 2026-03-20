#!/usr/bin/env python3
"""
Hybrid Code Query System with Query Planning, Reranking, and Precise Retrieval
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from pathlib import Path
import requests
import re
import json
import argparse
import os
from collections import defaultdict
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

OLLAMA_URL = "http://localhost:11434/api/generate"


def load_meta(path):
    """Load metadata from index"""
    meta = []
    with open(path/"meta.jsonl") as f:
        for l in f:
            meta.append(json.loads(l))
    return meta


def load_json(path):
    """Load JSON file if exists"""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def embed(queries, model):
    """Encode queries with normalization"""
    if isinstance(queries, str):
        queries = [queries]
    v = model.encode(queries, convert_to_numpy=True)
    v = v/np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype("float32")


def semantic_search(idx, emb, k):
    """Search in FAISS index"""
    D, I = idx.search(emb, k)
    return I[0], D[0]


def lexical_search(q, lex, meta):
    """Keyword-based search with scoring"""
    tokens = re.findall(r"[A-Za-z_]\w+", q.lower())
    scores = defaultdict(float)
    
    for t in tokens:
        if t in lex:
            for doc_id in lex[t]:
                scores[doc_id] += 1.0
    
    # Sort by score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids, scores


def keyword_match_score(query, chunk):
    """Calculate keyword overlap score"""
    query_words = set(re.findall(r"[a-z_]\w+", query.lower()))
    
    # Extract keywords from chunk
    chunk_words = set()
    chunk_words.update(re.findall(r"[a-z_]\w+", chunk["name"].lower()))
    chunk_words.update(re.findall(r"[a-z_]\w+", chunk["code"][:500].lower()))
    
    if chunk.get("parameters"):
        for p in chunk["parameters"]:
            chunk_words.add(p["name"].lower())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words & chunk_words)
    return overlap / len(query_words)


def parameter_match_score(query, chunk):
    """Check if query mentions parameters and if chunk has them"""
    param_keywords = ["param", "argument", "arg", "input", "receive", "accept", "take"]
    has_param_query = any(kw in query.lower() for kw in param_keywords)
    
    if not has_param_query:
        return 0.5  # Neutral if query doesn't care about params
    
    if chunk.get("parameters") and len(chunk["parameters"]) > 0:
        return 1.0
    return 0.0


def input_type_match_score(query, chunk):
    """Score based on input type matching"""
    query_lower = query.lower()
    
    # Check what input types the query is asking about
    wants_stdin = any(w in query_lower for w in ["stdin", "standard input", "console input", "cin", "scanf"])
    wants_file = any(w in query_lower for w in ["file", "fopen", "ifstream", "read from file"])
    wants_api = any(w in query_lower for w in ["api", "http", "request", "network"])
    
    if not (wants_stdin or wants_file or wants_api):
        return 0.5  # No specific input type requested
    
    score = 0.0
    if wants_stdin and chunk.get("has_stdin"):
        score += 0.5
    if wants_file and chunk.get("has_file_input"):
        score += 0.5
    if wants_api and chunk.get("has_api_call"):
        score += 0.5
    
    # Penalize mismatches
    if wants_stdin and not chunk.get("has_stdin") and not chunk.get("has_file_input"):
        score -= 0.3
    if wants_file and not chunk.get("has_file_input") and not chunk.get("has_stdin"):
        score -= 0.3
    
    return max(0.0, min(1.0, score))


def rerank_chunks(query, chunks, meta, call_graph=None):
    """Rerank chunks using multiple signals"""
    if not chunks:
        return []
    
    scored_chunks = []
    
    for idx in chunks:
        chunk = meta[idx]
        
        # Multiple scoring signals
        kw_score = keyword_match_score(query, chunk)
        param_score = parameter_match_score(query, chunk)
        input_score = input_type_match_score(query, chunk)
        
        # Combine scores with weights
        total_score = (
            0.4 * kw_score +
            0.3 * param_score +
            0.3 * input_score
        )
        
        scored_chunks.append({
            "idx": idx,
            "chunk": chunk,
            "score": total_score,
            "breakdown": {
                "keyword": kw_score,
                "parameter": param_score,
                "input_type": input_score
            }
        })
    
    # Sort by total score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks


class QueryPlanner:
    """Plan query execution strategy"""
    
    def __init__(self, special_indices, symbols, call_graph, called_by):
        self.special_indices = special_indices
        self.symbols = symbols
        self.call_graph = call_graph
        self.called_by = called_by
    
    def analyze_query(self, query):
        """Analyze query to determine search strategy"""
        query_lower = query.lower()
        
        analysis = {
            "query_type": "general",
            "keywords": re.findall(r"[a-z_]\w+", query_lower),
            "needs_stdin": False,
            "needs_file": False,
            "needs_api": False,
            "needs_params": False,
            "function_names": [],
            "expand_callers": False,
            "expand_callees": False
        }
        
        # Detect input type requirements
        if any(w in query_lower for w in ["stdin", "standard input", "console", "cin", "scanf", "getchar"]):
            analysis["needs_stdin"] = True
            analysis["query_type"] = "input_specific"
        
        if any(w in query_lower for w in ["file", "fopen", "ifstream", "fstream", "read from file"]):
            analysis["needs_file"] = True
            analysis["query_type"] = "input_specific"
        
        if any(w in query_lower for w in ["api", "http", "request", "network", "curl", "socket"]):
            analysis["needs_api"] = True
            analysis["query_type"] = "input_specific"
        
        # Detect parameter-related queries
        if any(w in query_lower for w in ["param", "argument", "input", "receive", "accept", "take"]):
            analysis["needs_params"] = True
        
        # Detect function name mentions
        if self.symbols:
            for func_name in self.symbols.keys():
                if func_name.lower() in query_lower:
                    analysis["function_names"].append(func_name)
        
        # Detect call graph expansion needs
        if any(w in query_lower for w in ["call", "invoke", "use", "caller", "callee", "called by"]):
            if "caller" in query_lower or "called by" in query_lower:
                analysis["expand_callers"] = True
            else:
                analysis["expand_callees"] = True
        
        # Detect listing/enumeration queries
        if any(w in query_lower for w in ["list", "enumerate", "show all", "find all", "which functions"]):
            analysis["query_type"] = "listing"
        
        return analysis
    
    def get_search_candidates(self, analysis, k=20):
        """Get candidate indices based on query analysis"""
        candidates = set()
        
        # Use special indices for input-specific queries
        if analysis["needs_stdin"] and "stdin" in self.special_indices:
            candidates.update(self.special_indices["stdin"])
        
        if analysis["needs_file"] and "file_input" in self.special_indices:
            candidates.update(self.special_indices["file_input"])
        
        if analysis["needs_api"] and "api_calls" in self.special_indices:
            candidates.update(self.special_indices["api_calls"])
        
        # Add function-specific candidates
        for func_name in analysis["function_names"]:
            if func_name in self.symbols:
                candidates.update(self.symbols[func_name])
        
        # Expand via call graph if needed
        if analysis["expand_callers"] or analysis["expand_callees"]:
            expanded = set(candidates)
            for idx in list(candidates):
                if str(idx) in self.call_graph:
                    if analysis["expand_callers"]:
                        expanded.update(self.call_graph[str(idx)].get("called_by", []))
                    if analysis["expand_callees"]:
                        expanded.update(self.call_graph[str(idx)].get("resolved_calls", []))
            candidates = expanded
        
        return list(candidates)[:k*2]  # Return more candidates for reranking


def build_prompt(frags, q, analysis=None):
    """Build enhanced prompt with structured context"""
    ctx = ""
    
    for i, f in enumerate(frags, 1):
        param_info = ""
        if f.get("parameters"):
            params = f["parameters"]
            param_strs = [f"{p['name']}: {p['type']}" for p in params]
            param_info = f"\nParameters: {', '.join(param_strs)}"
        
        input_info = ""
        if f.get("has_stdin"):
            input_info = "\n[Reads from stdin]"
        elif f.get("has_file_input"):
            input_info = "\n[Reads from files]"
        elif f.get("has_api_call"):
            input_info = "\n[Makes API calls]"
        
        ctx += f"""
--- Function {i} ---
File: {f["file"]}
Function: {f["name"]}{param_info}{input_info}
Signature: {f["signature"]}
Code:
{f["code"][:2000]}  # Truncate very long functions

"""
    
    # Add instructions based on query type
    instructions = ""
    if analysis:
        if analysis["query_type"] == "listing":
            instructions = """
IMPORTANT: Provide a clear list of functions that match the criteria.
For each function, mention:
1. Function name
2. File location
3. Key parameters (if relevant to the question)
4. How it matches the criteria (e.g., uses stdin, reads files, etc.)

Be precise and accurate - only include functions that truly match."""
        
        if analysis["needs_params"]:
            instructions += """
Pay special attention to function parameters. Only include functions that have input parameters
if the question asks about receiving data through parameters."""
    
    return f"""You are an expert code analyst. Answer the question based on the provided code context.

{ctx}

Question: {q}
{instructions}

Answer:"""


def call_llm(prompt, model, temperature=0.1):
    """Call LLM with retry logic"""
    try:
        r = requests.post(
            OLLAMA_URL,
            json=dict(
                model=model,
                prompt=prompt,
                stream=False,
                options=dict(temperature=temperature)
            ),
            timeout=120
        )
        return r.json()["response"]
    except Exception as e:
        return f"Error calling LLM: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--model", default="qwen2.5-coder")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--rerank_top_k", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    
    index_dir = Path(args.index_dir)
    
    # Load indices
    print("Loading indices...")
    idx = faiss.read_index(str(index_dir/"semantic.faiss"))
    meta = load_meta(index_dir)
    lex = load_json(index_dir/"lexical_index.json")
    special_indices = load_json(index_dir/"special_indices.json")
    symbols = load_json(index_dir/"symbols.json")
    call_graph = load_json(index_dir/"call_graph.json")
    called_by = load_json(index_dir/"called_by.json")
    
    # Initialize models
    print("Loading embedding model...")
    embed_model = SentenceTransformer(args.embed_model)
    
    # Try to load cross-encoder for reranking (optional, improves quality)
    reranker = None
    try:
        print("Loading reranker model...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except:
        print("Reranker not available, using heuristic reranking")
    
    # Initialize query planner
    planner = QueryPlanner(special_indices, symbols, call_graph, called_by)
    
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
        
        # Step 1: Analyze query
        analysis = planner.analyze_query(q)
        
        if args.verbose:
            print(f"\n[Query Analysis]")
            print(f"  Type: {analysis['query_type']}")
            print(f"  Needs stdin: {analysis['needs_stdin']}")
            print(f"  Needs file: {analysis['needs_file']}")
            print(f"  Needs params: {analysis['needs_params']}")
            print(f"  Keywords: {analysis['keywords'][:5]}...")
        
        # Step 2: Get candidates
        candidate_ids = planner.get_search_candidates(analysis, k=args.top_k)
        
        # Step 3: Semantic search
        emb = embed(q, embed_model)
        sem_ids, sem_scores = semantic_search(idx, emb, args.top_k * 2)
        
        # Combine candidates
        all_candidates = list(set(candidate_ids + sem_ids.tolist()))
        
        if args.verbose:
            print(f"\n[Retrieval]")
            print(f"  Special index candidates: {len(candidate_ids)}")
            print(f"  Semantic search candidates: {len(sem_ids)}")
            print(f"  Total unique candidates: {len(all_candidates)}")
        
        # Step 4: Rerank
        if reranker:
            # Use cross-encoder reranking
            texts = []
            for cid in all_candidates:
                c = meta[cid]
                text = f"{c['name']} {c['signature']} {c['code'][:500]}"
                texts.append(text)
            
            pairs = [[q, t] for t in texts]
            rerank_scores = reranker.predict(pairs)
            
            scored = [(all_candidates[i], rerank_scores[i]) for i in range(len(all_candidates))]
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [x[0] for x in scored[:args.rerank_top_k]]
        else:
            # Use heuristic reranking
            reranked = rerank_chunks(q, all_candidates, meta, call_graph)
            ranked_ids = [x["idx"] for x in reranked[:args.rerank_top_k]]
        
        if args.verbose:
            print(f"\n[Reranking]")
            print(f"  Selected top {len(ranked_ids)} chunks")
        
        # Step 5: Build context and generate answer
        frags = [meta[i] for i in ranked_ids]
        prompt = build_prompt(frags, q, analysis)
        
        if args.verbose:
            print(f"\n[Generation]")
            print(f"  Context size: {len(frags)} functions")
        
        ans = call_llm(prompt, args.model)
        
        elapsed = time.time() - start_time
        print(f"\n{ans}")
        print(f"\n[Response time: {elapsed:.2f}s]")


if __name__ == "__main__":
    main()
