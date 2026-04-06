#!/usr/bin/env python3
"""Browser chat UI for fuzz-coder using Gradio."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import faiss
import gradio as gr
from sentence_transformers import CrossEncoder

from fuzz_coder.ask import app as ask_app
from fuzz_coder.ask import core
from fuzz_coder.ask.pipeline import PipelineConfig, QueryPipeline
from fuzz_coder.embeddings.registry import get_embedding_backend


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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    return ap.parse_args()


def _load_indices(index_dir: Path):
    idx = faiss.read_index(str(index_dir / "semantic.faiss"))
    meta = core.load_meta(index_dir)
    lex = core.load_json(index_dir / "lexical_index.json")
    special_indices = core.load_json(index_dir / "special_indices.json")
    symbols = core.load_json(index_dir / "symbols.json")
    call_graph = core.load_json(index_dir / "call_graph.json")
    called_by = core.load_json(index_dir / "called_by.json")
    function_hints_path = index_dir / "function_hints.json"
    function_hints = core.load_json(function_hints_path) if function_hints_path.exists() else {}
    return idx, meta, lex, special_indices, symbols, call_graph, called_by, function_hints


def _load_models(args):
    embed_model = get_embedding_backend(args.embed_model, backend=args.embedding_backend)
    reranker = None
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        reranker = None
    return embed_model, reranker


def _history_to_conversation(history):
    if not history:
        return []

    if isinstance(history, list) and history and isinstance(history[0], (list, tuple)):
        out = []
        for item in history:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            u, a = item[0], item[1]
            if u is None or a is None:
                continue
            out.append((str(u), str(a)))
        return out

    if isinstance(history, list) and history and isinstance(history[0], dict):
        out = []
        pending_user = None
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                pending_user = str(content)
            elif role == "assistant" and pending_user is not None:
                out.append((pending_user, str(content)))
                pending_user = None
        return out

    return []


def _build_pipeline(args):
    index_dir = Path(args.index_dir)
    if not index_dir.exists() or not index_dir.is_dir():
        raise FileNotFoundError(f"Index directory does not exist: {index_dir}")

    idx, meta, lex, special_indices, symbols, call_graph, called_by, function_hints = _load_indices(index_dir)
    embed_model, reranker = _load_models(args)
    planner = core.QueryPlanner(special_indices, symbols, call_graph, called_by, meta=meta)

    return QueryPipeline(
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
        config=PipelineConfig(
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k,
            max_prompt_chars=args.max_prompt_chars,
            model=args.model,
            verbose=args.verbose,
            shadow_mode=os.getenv("FC_PIPELINE_SHADOW", "0") == "1",
        ),
    )


def main():
    args = _parse_args()
    pipeline = _build_pipeline(args)

    def _chat_fn(message, history):
        q = (message or "").strip()
        if not q:
            return "Please enter a question."
        if ask_app.is_help_query(q):
            return ask_app.render_help_text()

        expanded_q, _alias_used = ask_app.expand_chat_alias(q)
        conversation_history = _history_to_conversation(history)
        result = pipeline.run(expanded_q, conversation_history)
        return result.answer

    chat_kwargs = {
        "fn": _chat_fn,
        "title": "Fuzz Coder",
        "description": (
            "Codebase analysis chat for fuzzing targets, examples, parameter semantics, and implementation details.\n"
            "Aliases: `fuzz`, `fuzz wide`, `more fuzz`, `more fuzz wide`, `example FUNCTION`, `explain FUNCTION`."
        ),
        "chatbot": gr.Chatbot(height=650),
        "textbox": gr.Textbox(placeholder="Ask about functions, fuzz targets, or type 'help'"),
        "submit_btn": "Send",
        "clear_btn": "Clear",
    }

    # Keep compatibility with older gradio releases where some kwargs do not exist.
    supported = set(inspect.signature(gr.ChatInterface.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in chat_kwargs.items() if k in supported}
    demo = gr.ChatInterface(**filtered_kwargs)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
