#!/usr/bin/env python3
"""Browser chat UI for fuzz-coder using Gradio."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import faiss
import gradio as gr
from sentence_transformers import CrossEncoder

from fuzz_coder.ask import app as ask_app
from fuzz_coder.ask import core
from fuzz_coder.ask.pipeline import PipelineConfig, QueryPipeline
from fuzz_coder.embeddings.registry import get_embedding_backend


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default=None, help="Optional: use a single explicit index directory")
    ap.add_argument(
        "--index_base_dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory where index_data_* projects are discovered",
    )
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


def _looks_like_index_dir(path: Path):
    required = [
        "semantic.faiss",
        "meta.jsonl",
        "lexical_index.json",
        "special_indices.json",
        "symbols.json",
        "call_graph.json",
        "called_by.json",
    ]
    return path.is_dir() and all((path / f).exists() for f in required)


def _discover_projects(args):
    projects: Dict[str, Path] = {}

    if args.index_dir:
        explicit = Path(args.index_dir).resolve()
        if not _looks_like_index_dir(explicit):
            raise FileNotFoundError(f"Invalid index directory (missing required files): {explicit}")
        name = explicit.name
        if name.startswith("index_data_"):
            name = name[len("index_data_"):]
        projects[name] = explicit
        return projects

    base_dir = Path(args.index_base_dir).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Index base directory does not exist: {base_dir}")

    for p in sorted(base_dir.glob("index_data_*")):
        if not _looks_like_index_dir(p):
            continue
        name = p.name[len("index_data_"):] or p.name
        # Guard against duplicate suffixes.
        if name in projects:
            name = p.name
        projects[name] = p.resolve()

    return projects


def _build_pipeline_for_index(index_dir: Path, args, embed_model, reranker):
    idx, meta, lex, special_indices, symbols, call_graph, called_by, function_hints = _load_indices(index_dir)
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


def _create_chatbot():
    """Create Chatbot and infer its effective history mode from the instance itself."""
    try:
        chatbot = gr.Chatbot(height=650, type="messages")
    except TypeError:
        chatbot = gr.Chatbot(height=650)

    mode = getattr(chatbot, "type", None)
    if isinstance(mode, str):
        mode = mode.lower().strip()
    if mode not in {"messages", "tuples"}:
        # Most modern gradio builds use messages by default.
        mode = "messages"
    return chatbot, mode


def _normalize_history_for_mode(history, mode: str):
    history = list(history or [])
    if mode == "messages":
        if history and isinstance(history[0], dict):
            out = []
            for msg in history:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip()
                if role not in {"user", "assistant", "system"}:
                    continue
                out.append({"role": role, "content": str(msg.get("content", ""))})
            return out
        # Convert ChatMessage-like objects -> messages
        if history and hasattr(history[0], "role") and hasattr(history[0], "content"):
            out = []
            for msg in history:
                role = str(getattr(msg, "role", "")).strip()
                if role not in {"user", "assistant", "system"}:
                    continue
                out.append({"role": role, "content": str(getattr(msg, "content", ""))})
            return out
        # Convert tuple/list turns -> messages
        out = []
        for item in history:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            u, a = item[0], item[1]
            if u is None:
                u = ""
            if a is None:
                a = ""
            out.append({"role": "user", "content": str(u)})
            out.append({"role": "assistant", "content": str(a)})
        return out

    # tuples mode
    if history and isinstance(history[0], (list, tuple)):
        return history
    # Convert messages -> tuple turns
    out = []
    pending_user = None
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            pending_user = str(content)
        elif role == "assistant":
            if pending_user is None:
                pending_user = ""
            out.append((pending_user, str(content)))
            pending_user = None
    return out


def _append_turn(history, user_text: str, answer_text: str, mode: str):
    history = list(history or [])
    if mode == "messages":
        history.append({"role": "user", "content": str(user_text)})
        history.append({"role": "assistant", "content": str(answer_text)})
        return history
    history.append((str(user_text), str(answer_text)))
    return history


def main():
    args = _parse_args()
    projects = _discover_projects(args)
    if not projects:
        raise FileNotFoundError(
            "No index_data_* directories found. "
            "Place index_data_PROJECT folders next to web_fuzz_coder.py or pass --index_dir."
        )

    embed_model, reranker = _load_models(args)
    pipeline_cache: Dict[str, QueryPipeline] = {}

    def _get_pipeline(project_name: str):
        if project_name not in projects:
            raise ValueError(f"Unknown project: {project_name}")
        if project_name not in pipeline_cache:
            pipeline_cache[project_name] = _build_pipeline_for_index(
                projects[project_name], args, embed_model, reranker
            )
        return pipeline_cache[project_name]

    chatbot_mode_holder = {"mode": "messages"}

    def _on_project_change(project_name, histories):
        histories = dict(histories or {})
        mode = chatbot_mode_holder["mode"]
        project_hist = _normalize_history_for_mode(histories.get(project_name, []), mode)
        histories[project_name] = project_hist
        return project_hist, histories

    def _chat_submit(message, chat_history, project_name, histories):
        histories = dict(histories or {})
        mode = chatbot_mode_holder["mode"]
        chat_history = _normalize_history_for_mode(chat_history, mode)
        q = (message or "").strip()
        if not q:
            return "", chat_history, histories

        if ask_app.is_help_query(q):
            ans = ask_app.render_help_text()
        else:
            expanded_q, _alias_used = ask_app.expand_chat_alias(q)
            pipeline = _get_pipeline(project_name)
            conversation_history = _history_to_conversation(chat_history)
            result = pipeline.run(expanded_q, conversation_history)
            ans = result.answer

        chat_history = _append_turn(chat_history, q, ans, mode)
        histories[project_name] = chat_history
        return "", chat_history, histories

    def _clear_project_chat(project_name, histories):
        histories = dict(histories or {})
        histories[project_name] = []
        return [], histories

    project_names = list(projects.keys())
    default_project = project_names[0]

    with gr.Blocks(title="Fuzz Coder") as demo:
        gr.Markdown(
            "## Fuzz Coder\n"
            "Codebase analysis chat for fuzzing targets, examples, parameter semantics, and implementation details.\n"
            "Aliases: `fuzz`, `fuzz wide`, `more fuzz`, `more fuzz wide`, `example FUNCTION`, `explain FUNCTION`."
        )
        project = gr.Dropdown(
            choices=project_names,
            value=default_project,
            label="Project (index_data_PROJECT)",
        )
        chatbot, chatbot_mode = _create_chatbot()
        chatbot_mode_holder["mode"] = chatbot_mode
        if args.verbose:
            print(f"[Web UI] Chatbot mode: {chatbot_mode}")
        msg = gr.Textbox(placeholder="Ask about functions, fuzz targets, or type 'help'", label="Message")
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Current Project Chat")

        histories_state = gr.State({default_project: []})

        project.change(
            fn=_on_project_change,
            inputs=[project, histories_state],
            outputs=[chatbot, histories_state],
        )
        msg.submit(
            fn=_chat_submit,
            inputs=[msg, chatbot, project, histories_state],
            outputs=[msg, chatbot, histories_state],
        )
        send_btn.click(
            fn=_chat_submit,
            inputs=[msg, chatbot, project, histories_state],
            outputs=[msg, chatbot, histories_state],
        )
        clear_btn.click(
            fn=_clear_project_chat,
            inputs=[project, histories_state],
            outputs=[chatbot, histories_state],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
