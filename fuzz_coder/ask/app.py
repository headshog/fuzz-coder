from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder

from fuzz_coder.embeddings.registry import get_embedding_backend
from fuzz_coder.languages.registry import get_prompt_language_adapter

from . import core
from .pipeline import PipelineConfig, QueryPipeline


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
DEFAULT_CHAT_LANGUAGE = "c_cpp"


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


def expand_chat_alias(q: str, language_name: str = DEFAULT_CHAT_LANGUAGE):
    """Expand short chat aliases into full natural-language queries."""
    raw = (q or "").strip()
    if not raw:
        return raw, False

    adapter = get_prompt_language_adapter(language_name or DEFAULT_CHAT_LANGUAGE)
    normalized = " ".join(raw.split()).lower()
    if normalized == "fuzz":
        return adapter.alias_fuzz, True
    if normalized in {"fuzz wide", "wide fuzz"}:
        return adapter.alias_fuzz_wide, True
    if normalized in {"more fuzz wide", "wide more fuzz", "more wide fuzz"}:
        return adapter.alias_more_fuzz_wide, True
    if normalized == "more fuzz":
        return adapter.alias_more_fuzz, True
    if normalized.startswith("explain "):
        parts = raw.split(None, 1)
        if len(parts) == 2:
            rest = parts[1].strip()
            m = re.match(r"^([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)(?:\s+(.*))?$", rest)
            if m:
                fn = (m.group(1) or "").strip("`'\"")
                tail = (m.group(2) or "").strip()
                if fn:
                    tail = re.sub(
                        r"\b(from|in)\s+([^\n,;]+?)\s+(module|directory|subdirectory|folder|path)\b",
                        r"\1 \3 \2",
                        tail,
                        flags=re.IGNORECASE,
                    )
                    tail = re.sub(
                        r"\b(из|в)\s+([^\n,;]+?)\s+(модуле|модуля|директории|поддиректории|папке)\b",
                        r"\1 \3 \2",
                        tail,
                        flags=re.IGNORECASE,
                    )
                    expanded = adapter.alias_explain_template.format(function_name=fn)
                    if tail:
                        expanded = f"{expanded} {tail}"
                    return expanded, True
    if normalized.startswith("example "):
        parts = raw.split(None, 1)
        if len(parts) == 2:
            rest = parts[1].strip()
            m = re.match(r"^([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)(?:\s+(.*))?$", rest)
            if m:
                fn = (m.group(1) or "").strip("`'\"")
                tail = (m.group(2) or "").strip()
                if fn:
                    tail = re.sub(
                        r"\b(from|in)\s+([^\n,;]+?)\s+(module|directory|subdirectory|folder|path)\b",
                        r"\1 \3 \2",
                        tail,
                        flags=re.IGNORECASE,
                    )
                    tail = re.sub(
                        r"\b(из|в)\s+([^\n,;]+?)\s+(модуле|модуля|директории|поддиректории|папке)\b",
                        r"\1 \3 \2",
                        tail,
                        flags=re.IGNORECASE,
                    )
                    expanded = adapter.alias_example_template.format(function_name=fn)
                    if tail:
                        expanded = f"{expanded} {tail}"
                    return expanded, True
            fn = rest.strip("`'\"")
            if fn:
                return adapter.alias_example_template.format(function_name=fn), True
    return raw, False


def render_help_text(language_name: str = DEFAULT_CHAT_LANGUAGE) -> str:
    adapter = get_prompt_language_adapter(language_name or DEFAULT_CHAT_LANGUAGE)
    cli_expr = adapter.cli_file_expr
    return (
        "I can analyze indexed code and answer questions about functions, types, call flows, and fuzz targets.\n\n"
        "What I can do:\n"
        "1. List functions by criteria (stdin/file/API/types/parse/fuzz).\n"
        "2. Filter by module/subdirectory (path filter).\n"
        "3. Give examples of function usage.\n"
        "4. Explain parameter semantics (format/role/source) for a function.\n"
        "5. Explain implementation details from indexed code.\n"
        "6. Handle follow-ups (including 'other functions' without repeats).\n\n"
        "Chat aliases:\n"
        "- fuzz -> Write a list of functions that can be used for fuzzing\n"
        "- fuzz wide -> Large fuzz-target list (20-30) with <=4 params and broad OR input-surface constraints\n"
        "- more fuzz -> Write other functions that are good for fuzzing\n"
        "- more fuzz wide -> Like fuzz wide, but exclude previously listed functions\n"
        f"- example FUNCTION_NAME -> Write an example of FUNCTION_NAME with a standalone main() and {cli_expr}-based params\n\n"
        "- explain FUNCTION_NAME -> Analyze parameter semantics/format/source for FUNCTION_NAME with evidence\n\n"
        "Example queries:\n"
        "- List functions good for fuzzing from module src/parsers\n"
        "- Какие функции читают из stdin?\n"
        f"{adapter.help_type_query_example}"
        "- Дай список других функций для фаззинга из директории src/parsers\n"
        "- Explain how decode_binary_blob works\n"
        "- Give an example calling parse_json_payload\n"
        "- What parameters does llama_params_fit take and what does each mean?\n"
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
    function_hints_path = index_dir / "function_hints.json"
    function_hints = core.load_json(function_hints_path) if function_hints_path.exists() else {}
    type_init_index_path = index_dir / "type_init_index.json"
    type_init_index = core.load_json(type_init_index_path) if type_init_index_path.exists() else {}
    index_meta_path = index_dir / "index_meta.json"
    index_meta = core.load_json(index_meta_path) if index_meta_path.exists() else {}
    language_name = (index_meta or {}).get("language")
    return idx, meta, lex, special_indices, symbols, call_graph, called_by, function_hints, type_init_index, language_name


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


def main():
    args = _parse_args()
    index_dir = Path(args.index_dir)
    if not index_dir.exists() or not index_dir.is_dir():
        print(f"Error: index directory does not exist: {index_dir}")
        print("Run index_fuzz_coder.py first to build the project index.")
        return

    idx, meta, lex, special_indices, symbols, call_graph, called_by, function_hints, type_init_index, language_name = _load_indices(index_dir)
    embed_model, reranker = _load_models(args)
    planner = core.QueryPlanner(special_indices, symbols, call_graph, called_by, meta=meta)

    pipeline = QueryPipeline(
        core_module=core,
        planner=planner,
        idx=idx,
        lex=lex,
        meta=meta,
        embed_model=embed_model,
        reranker=reranker,
        symbols=symbols,
        call_graph=call_graph,
        called_by=called_by,
        function_hints=function_hints,
        type_init_index=type_init_index,
        language_name=language_name,
        config=PipelineConfig(
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k,
            max_prompt_chars=args.max_prompt_chars,
            model=args.model,
            verbose=args.verbose,
            shadow_mode=os.getenv("FC_PIPELINE_SHADOW", "0") == "1",
        ),
    )

    conversation_history = []

    print(f"\nReady! Using model: {args.model}")
    print(f"Index contains {len(meta)} functions")
    print("Type 'quit' to exit, or 'help' to see examples and aliases\n")

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
            print("\n" + render_help_text(language_name=language_name or DEFAULT_CHAT_LANGUAGE))
            continue
        q, alias_used = expand_chat_alias(q, language_name=language_name or DEFAULT_CHAT_LANGUAGE)
        if alias_used and args.verbose:
            print(f"\n[Alias]")
            print(f"  Expanded query: {q}")

        result = pipeline.run(q, conversation_history)

        print(f"\n{result.answer}")
        if result.show_response_time:
            print(f"\n[Response time: {result.elapsed:.2f}s]")

        conversation_history.append((q, result.history_answer))
        _trim_history(conversation_history)


if __name__ == "__main__":
    main()
