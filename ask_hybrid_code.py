#!/usr/bin/env python3
"""
Advanced Code Query System with Thinking Mode, Context Memory, and Multi-Step Reasoning
Supports follow-up questions, code generation examples, and deep codebase understanding
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
from typing import List, Dict, Any, Optional

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
            chunk_words.add(p["type"].lower())

    if not query_words:
        return 0.0

    overlap = len(query_words & chunk_words)
    return overlap / max(len(query_words), 1)


def parameter_match_score(query, chunk):
    """Check if query mentions parameters and if chunk has them"""
    param_keywords = ["param", "argument", "arg", "input", "receive", "accept", "take", "байт", "массив", "array", "byte"]
    has_param_query = any(kw in query.lower() for kw in param_keywords)

    if not has_param_query:
        return 0.5  # Neutral if query doesn't care about params

    if chunk.get("parameters") and len(chunk["parameters"]) > 0:
        # Check if parameter types match query intent
        query_lower = query.lower()
        for p in chunk["parameters"]:
            p_type = p["type"].lower()
            p_name = p["name"].lower()
            if any(kw in p_type or kw in p_name for kw in ["byte", "uint8", "char", "buffer", "data", "array", "vector", "string"]):
                return 1.0
        return 0.7  # Has params but not exact match
    return 0.0


def input_type_match_score(query, chunk):
    """Score based on input type matching"""
    query_lower = query.lower()

    # Check what input types the query is asking about
    wants_stdin = any(w in query_lower for w in ["stdin", "standard input", "console input", "cin", "scanf", "getchar", "fgets"])
    wants_file = any(w in query_lower for w in ["file", "fopen", "ifstream", "fstream", "read from file"])
    wants_api = any(w in query_lower for w in ["api", "http", "request", "network", "curl", "socket"])
    wants_param = any(w in query_lower for w in ["param", "argument", "input", "receive", "accept", "pass", "переда", "вход"])

    if not (wants_stdin or wants_file or wants_api or wants_param):
        return 0.5  # No specific input type requested

    score = 0.0
    if wants_stdin and chunk.get("has_stdin"):
        score += 0.5
    if wants_file and chunk.get("has_file_input"):
        score += 0.5
    if wants_api and chunk.get("has_api_call"):
        score += 0.5
    if wants_param and chunk.get("parameters") and len(chunk["parameters"]) > 0:
        score += 0.5

    return max(0.0, min(1.0, score))


def type_match_score(query, chunk):
    """Check if query asks about specific types and chunk has them"""
    query_lower = query.lower()

    # Type keywords in Russian and English
    type_patterns = {
        "byte_array": ["byte", "uint8", "char*", "buffer", "массив байт", "байт"],
        "string": ["string", "str", "char[]", "строка"],
        "int": ["int", "integer", "число", "int32", "int64"],
        "float": ["float", "double", "веществен", "floating"],
        "vector": ["vector", "array", "список", "массив"],
        "pointer": ["pointer", "*", "указатель"],
        "reference": ["reference", "&", "ссылка"],
    }

    matched_types = []
    for type_name, keywords in type_patterns.items():
        if any(kw in query_lower for kw in keywords):
            matched_types.append(type_name)

    if not matched_types:
        return 0.5  # No specific type requested

    if not chunk.get("parameters"):
        return 0.0

    # Check if chunk parameters match requested types
    chunk_types = []
    for p in chunk["parameters"]:
        p_type = p["type"].lower()
        p_raw = p["raw"].lower()
        for type_name, keywords in type_patterns.items():
            if any(kw in p_type or kw in p_raw for kw in keywords):
                chunk_types.append(type_name)

    # Calculate overlap
    overlap = len(set(matched_types) & set(chunk_types))
    return min(1.0, overlap / max(len(matched_types), 1))


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
        type_score = type_match_score(query, chunk)

        # Combine scores with weights
        total_score = (
            0.3 * kw_score +
            0.25 * param_score +
            0.25 * input_score +
            0.2 * type_score
        )

        scored_chunks.append({
            "idx": idx,
            "chunk": chunk,
            "score": total_score,
            "breakdown": {
                "keyword": kw_score,
                "parameter": param_score,
                "input_type": input_score,
                "type_match": type_score
            }
        })

    # Sort by total score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks


class QueryPlanner:
    """Advanced query planner with thinking mode support"""

    def __init__(self, special_indices, symbols, call_graph, called_by):
        self.special_indices = special_indices
        self.symbols = symbols
        self.call_graph = call_graph
        self.called_by = called_by

    def analyze_query(self, query, context_history=None):
        """Analyze query to determine search strategy with thinking mode"""
        query_lower = query.lower()

        analysis = {
            "query_type": "general",
            "keywords": re.findall(r"[a-z_а-яё]\w+", query_lower),
            "needs_stdin": False,
            "needs_file": False,
            "needs_api": False,
            "needs_params": False,
            "needs_type_info": False,
            "requested_types": [],
            "function_names": [],
            "expand_callers": False,
            "expand_callees": False,
            "needs_example": False,
            "needs_implementation": False,
            "follow_up": False,
            "referenced_functions": []
        }

        # Detect input type requirements
        if any(w in query_lower for w in ["stdin", "standard input", "console", "cin", "scanf", "getchar", "fgets", "ввод"]):
            analysis["needs_stdin"] = True

        if any(w in query_lower for w in ["file", "fopen", "ifstream", "fstream", "read from file", "файл"]):
            analysis["needs_file"] = True

        if any(w in query_lower for w in ["api", "http", "request", "network", "curl", "socket", "сеть"]):
            analysis["needs_api"] = True

        # Detect parameter-related queries
        if any(w in query_lower for w in ["param", "argument", "arg", "input", "receive", "accept", "take", "переда", "вход", "параметр"]):
            analysis["needs_params"] = True

        # Detect type-specific queries
        type_keywords = {
            "byte_array": ["byte", "uint8", "char*", "buffer", "массив байт", "байтов", "байты"],
            "string": ["string", "str", "char[]", "строка", "строку"],
            "int": ["int", "integer", "число", "int32", "int64", "цел"],
            "float": ["float", "double", "веществен", "floating", "плавающ"],
            "vector": ["vector", "array", "список", "массив", "std::vector"],
            "pointer": ["pointer", "*", "указатель"],
            "reference": ["reference", "&", "ссылка"],
        }

        for type_name, keywords in type_keywords.items():
            if any(kw in query_lower for kw in keywords):
                analysis["requested_types"].append(type_name)
                analysis["needs_type_info"] = True

        # Detect function name mentions
        if self.symbols:
            for func_name in self.symbols.keys():
                if func_name.lower() in query_lower or func_name in query:
                    analysis["function_names"].append(func_name)
                    analysis["referenced_functions"].append(func_name)

        # Detect call graph expansion needs
        if any(w in query_lower for w in ["call", "invoke", "use", "caller", "callee", "called by", "вызыва", "использу"]):
            if "caller" in query_lower or "called by" in query_lower or "кто вызыва" in query_lower:
                analysis["expand_callers"] = True
            else:
                analysis["expand_callees"] = True

        # Detect listing/enumeration queries
        if any(w in query_lower for w in ["list", "enumerate", "show all", "find all", "which functions", "какие функции", "перечисли", "покажи все"]):
            analysis["query_type"] = "listing"

        # Detect example generation requests
        if any(w in query_lower for w in ["example", "пример", "как вызвать", "как использовать", "usage", "использовани"]):
            analysis["needs_example"] = True
            analysis["query_type"] = "example_generation"

        # Detect implementation questions
        if any(w in query_lower for w in ["implement", "реализ", "как работает", "how does", "algorithm", "алгоритм"]):
            analysis["needs_implementation"] = True
            analysis["query_type"] = "implementation_explanation"

        # Check for follow-up questions
        if context_history:
            if any(w in query_lower for w in ["this", "that", "these", "those", "эти", "этот", "такой", "так", "далее", "дальше", "продолж"]):
                analysis["follow_up"] = True
            # Check if referencing previous answer
            if any(w in query_lower for w in ["from the list", "из списка", "функци", "function a", "function b", "function c"]):
                analysis["follow_up"] = True

        # Determine query type
        if analysis["needs_example"]:
            analysis["query_type"] = "example_generation"
        elif analysis["needs_implementation"]:
            analysis["query_type"] = "implementation_explanation"
        elif analysis["needs_stdin"] or analysis["needs_file"] or analysis["needs_api"]:
            analysis["query_type"] = "input_specific"
        elif analysis["needs_type_info"]:
            analysis["query_type"] = "type_specific"
        elif len(analysis["function_names"]) > 0:
            analysis["query_type"] = "function_specific"

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

        return list(candidates)[:k*3]  # Return more candidates for better reranking


def build_thinking_prompt(frags, q, analysis=None, conversation_history=None):
    """Build prompt with thinking mode for complex reasoning"""

    # Build function context
    ctx = ""
    for i, f in enumerate(frags, 1):
        param_info = ""
        if f.get("parameters"):
            params = f["parameters"]
            param_strs = [f"{p['name']}: {p['type']}" for p in params]
            param_info = f"\n  Parameters: {', '.join(param_strs)}"

        input_info = ""
        if f.get("has_stdin"):
            input_info = " [STDIN]"
        elif f.get("has_file_input"):
            input_info = " [FILE]"
        elif f.get("has_api_call"):
            input_info = " [API]"

        ctx += f"""
[Function {i}] {f["name"]}{input_info}
  File: {f["file"]}:{f["start_line"]}-{f["end_line"]}
  Signature: {f["signature"]}{param_info}
  Code:
```cpp
{f["code"][:1500]}
```
"""

    # Build conversation history context
    history_ctx = ""
    if conversation_history:
        history_ctx = "\n### Conversation History:\n"
        for idx, (prev_q, prev_a) in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
            history_ctx += f"User: {prev_q}\nAssistant: {prev_a[:300]}...\n\n"

    # Determine instructions based on query type
    instructions = ""
    if analysis:
        if analysis["query_type"] == "listing":
            instructions = """
### Instructions:
1. Carefully analyze EACH function in the context above
2. Check if it matches ALL criteria from the question
3. Create a numbered list of matching functions
4. For each function include:
   - Name and file location
   - Relevant parameters with types
   - Brief explanation why it matches
5. Be precise - only include functions that truly match the criteria"""

        elif analysis["query_type"] == "example_generation":
            instructions = """
### Instructions:
1. Identify the specific function(s) mentioned or implied
2. Write a complete, compilable code example showing how to call this function
3. Include:
   - Necessary #include statements
   - Proper variable declarations with correct types
   - The function call with appropriate arguments
   - Error handling if relevant
4. Add comments explaining key parts
5. Make sure the example is realistic and follows the codebase patterns"""

        elif analysis["query_type"] == "implementation_explanation":
            instructions = """
### Instructions:
1. Explain the algorithm/logic step by step
2. Reference specific code sections from the context
3. Describe:
   - What the function does
   - How it processes inputs
   - Key operations and their purpose
   - Return value meaning
4. Use simple language but be technically accurate"""

        elif analysis["needs_type_info"]:
            instructions = """
### Instructions:
1. Focus on parameter types mentioned in the question
2. Check each function's signature carefully
3. List only functions whose parameters match the requested type
4. Include the exact type signature for verification
5. If Russian terms used (e.g., "массив байтов"), match to C++ types like:
   - byte array → uint8_t*, char*, std::vector<uint8_t>, QByteArray
   - string → std::string, char*, const char*
   - integer → int, int32_t, int64_t, size_t"""

        elif analysis["follow_up"]:
            instructions = """
### Instructions:
1. This is a follow-up question - consider the conversation context
2. If user references "these functions" or "from the list", use previous answer
3. Maintain consistency with earlier responses
4. Build upon previous information rather than repeating"""

    # Thinking prompt structure
    thinking_instructions = """
### Thinking Process (think step-by-step before answering):
1. UNDERSTAND: What is the user really asking? Identify key requirements.
2. ANALYZE: Look at each function in the context - what are their signatures?
3. MATCH: Which functions satisfy the criteria? Why or why not?
4. VERIFY: Double-check parameter types, input methods, etc.
5. FORMULATE: Structure your answer clearly based on the question type.

Now provide your answer:"""

    return f"""You are an expert C/C++ code analyst with deep understanding of codebases.

{history_ctx}
### Current Question: {q}

### Available Functions from Codebase:
{ctx}
{instructions}
{thinking_instructions}
"""


def build_prompt(frags, q, analysis=None, conversation_history=None):
    """Main prompt builder - uses thinking mode for complex queries"""
    # Use thinking mode for complex queries
    complex_types = ["listing", "example_generation", "implementation_explanation", "type_specific"]
    use_thinking = (analysis and analysis["query_type"] in complex_types) or (conversation_history and len(conversation_history) > 0)

    if use_thinking:
        return build_thinking_prompt(frags, q, analysis, conversation_history)

    # Simple prompt for straightforward queries
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
{f["code"][:2000]}

"""

    return f"""You are an expert code analyst. Answer the question based on the provided code context.

{ctx}

Question: {q}

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
            timeout=600
        )
        return r.json()["response"]
    except Exception as e:
        return f"Error calling LLM: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--model", default="qwen2.5-coder:32b")
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
            print(f"  Follow-up: {analysis['follow_up']}")
            print(f"  Needs stdin: {analysis['needs_stdin']}")
            print(f"  Needs file: {analysis['needs_file']}")
            print(f"  Needs params: {analysis['needs_params']}")
            print(f"  Requested types: {analysis['requested_types']}")
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
        prompt = build_prompt(frags, q, analysis, conversation_history if conversation_history else None)

        if args.verbose:
            print(f"\n[Generation]")
            print(f"  Context size: {len(frags)} functions")
            print(f"  Prompt length: {len(prompt)} chars")

        ans = call_llm(prompt, args.model)

        elapsed = time.time() - start_time
        print(f"\n{ans}")
        print(f"\n[Response time: {elapsed:.2f}s]")

        # Update conversation history
        conversation_history.append((q, ans))
        # Keep only last 5 exchanges to avoid context explosion
        if len(conversation_history) > 5:
            conversation_history.pop(0)


if __name__ == "__main__":
    main()
