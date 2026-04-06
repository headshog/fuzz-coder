#!/usr/bin/env python3
"""Browser chat UI for fuzz-coder using Gradio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List

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
    ap.add_argument(
        "--auth_users",
        default="",
        help="Optional basic auth users: 'alice:pass,bob:pass2'",
    )
    ap.add_argument(
        "--auth_users_file",
        default=None,
        help="Optional file with users for basic auth (.txt with user:pass per line, or .json)",
    )
    ap.add_argument(
        "--history_dir",
        default=str(Path(__file__).resolve().parent / ".web_fuzz_histories"),
        help="Directory for per-user persisted chat histories",
    )
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


def _conversation_to_history(conversation, mode: str):
    conversation = _normalize_conversation(conversation)
    if mode == "messages":
        out = []
        for user_text, assistant_text in conversation:
            out.append({"role": "user", "content": str(user_text)})
            out.append({"role": "assistant", "content": str(assistant_text)})
        return out
    return [(str(user_text), str(assistant_text)) for user_text, assistant_text in conversation]


def _normalize_conversation(conversation):
    out = []
    if not isinstance(conversation, list):
        return out
    for item in conversation:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text, assistant_text = item[0], item[1]
        elif isinstance(item, dict):
            user_text = item.get("user")
            assistant_text = item.get("assistant")
        else:
            continue
        if user_text is None or assistant_text is None:
            continue
        out.append((str(user_text), str(assistant_text)))
    return out


def _parse_auth_credentials(args):
    creds: Dict[str, str] = {}

    def _add_cred(raw_user: str, raw_pass: str, src: str):
        user = str(raw_user or "").strip()
        password = str(raw_pass or "").strip()
        if not user or not password:
            raise ValueError(f"Invalid auth entry from {src}: empty username/password")
        creds[user] = password

    if args.auth_users_file:
        p = Path(args.auth_users_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Auth users file not found: {p}")
        content = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            data = json.loads(content)
            if isinstance(data, dict):
                for user, password in data.items():
                    _add_cred(user, password, f"{p}")
            elif isinstance(data, list):
                for idx, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        user = item.get("username", item.get("user"))
                        password = item.get("password", item.get("pass"))
                        _add_cred(user, password, f"{p}:{idx}")
                    elif isinstance(item, str) and ":" in item:
                        user, password = item.split(":", 1)
                        _add_cred(user, password, f"{p}:{idx}")
                    else:
                        raise ValueError(f"Invalid JSON auth entry at {p}:{idx}")
            else:
                raise ValueError(f"Unsupported JSON auth format in {p}")
        else:
            for idx, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    raise ValueError(f"Invalid auth line (expected user:pass) at {p}:{idx}")
                user, password = line.split(":", 1)
                _add_cred(user, password, f"{p}:{idx}")

    raw_inline = str(args.auth_users or "").strip()
    if raw_inline:
        for idx, item in enumerate(raw_inline.split(","), 1):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid --auth_users entry #{idx} (expected user:pass)")
            user, password = item.split(":", 1)
            _add_cred(user, password, f"--auth_users#{idx}")

    if not creds:
        return None
    return list(creds.items())


def _user_from_request(request):
    if request is not None:
        username = getattr(request, "username", None)
        if username:
            return str(username)
    return "anonymous"


def _safe_user_slug(username: str):
    normalized = str(username or "anonymous")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("._")
    if not slug:
        slug = "user"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _history_file_for_user(history_dir: Path, username: str):
    return history_dir / f"{_safe_user_slug(username)}.json"


def _load_user_histories(history_dir: Path, username: str, project_names: List[str]):
    result = {name: [] for name in project_names}
    p = _history_file_for_user(history_dir, username)
    if not p.exists():
        return result
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return result

    projects_blob = data.get("projects") if isinstance(data, dict) else None
    if not isinstance(projects_blob, dict):
        return result

    for project_name in project_names:
        result[project_name] = _normalize_conversation(projects_blob.get(project_name, []))
    return result


def _save_user_histories(history_dir: Path, username: str, user_histories, project_names: List[str]):
    history_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "username": username,
        "projects": {},
    }
    for project_name in project_names:
        conv = _normalize_conversation(user_histories.get(project_name, []))
        if conv:
            payload["projects"][project_name] = [[u, a] for u, a in conv]

    p = _history_file_for_user(history_dir, username)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


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


def main():
    args = _parse_args()
    projects = _discover_projects(args)
    if not projects:
        raise FileNotFoundError(
            "No index_data_* directories found. "
            "Place index_data_PROJECT folders next to web_fuzz_coder.py or pass --index_dir."
        )

    embed_model, reranker = _load_models(args)
    auth_credentials = _parse_auth_credentials(args)
    history_dir = Path(args.history_dir).expanduser().resolve()
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

    def _init_history_state(state):
        state = dict(state or {})
        users = state.get("users")
        if not isinstance(users, dict):
            users = {}
        state["users"] = users
        return state

    def _ensure_user_histories(state, username: str):
        state = _init_history_state(state)
        users = state["users"]
        if username not in users:
            users[username] = _load_user_histories(history_dir, username, project_names)
        user_histories = users[username]
        for pname in project_names:
            if pname not in user_histories:
                user_histories[pname] = []
            else:
                user_histories[pname] = _normalize_conversation(user_histories[pname])
        return user_histories, state

    def _on_project_change(project_name, histories, request: gr.Request = None):
        username = _user_from_request(request)
        user_histories, histories = _ensure_user_histories(histories, username)
        mode = chatbot_mode_holder["mode"]
        project_hist = _conversation_to_history(user_histories.get(project_name, []), mode)
        return project_hist, histories

    def _chat_submit(message, chat_history, project_name, histories, request: gr.Request = None):
        username = _user_from_request(request)
        user_histories, histories = _ensure_user_histories(histories, username)
        mode = chatbot_mode_holder["mode"]
        chat_history = _normalize_history_for_mode(chat_history, mode)
        conversation_history = _history_to_conversation(chat_history)
        q = (message or "").strip()
        if not q:
            return "", chat_history, histories

        if ask_app.is_help_query(q):
            ans = ask_app.render_help_text()
        else:
            expanded_q, _alias_used = ask_app.expand_chat_alias(q)
            pipeline = _get_pipeline(project_name)
            result = pipeline.run(expanded_q, conversation_history)
            ans = result.answer

        conversation_history.append((q, ans))
        user_histories[project_name] = conversation_history
        _save_user_histories(history_dir, username, user_histories, project_names)

        return "", _conversation_to_history(conversation_history, mode), histories

    def _clear_project_chat(project_name, histories, request: gr.Request = None):
        username = _user_from_request(request)
        user_histories, histories = _ensure_user_histories(histories, username)
        user_histories[project_name] = []
        _save_user_histories(history_dir, username, user_histories, project_names)
        return _conversation_to_history([], chatbot_mode_holder["mode"]), histories

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
            print(f"[Web UI] Auth enabled: {'yes' if auth_credentials else 'no'}")
            print(f"[Web UI] History dir: {history_dir}")
        msg = gr.Textbox(placeholder="Ask about functions, fuzz targets, or type 'help'", label="Message")
        logout_btn = None
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Current Project Chat")
            if auth_credentials:
                logout_btn = gr.Button("Logout")

        histories_state = gr.State({"users": {}})

        project.change(
            fn=_on_project_change,
            inputs=[project, histories_state],
            outputs=[chatbot, histories_state],
        )
        demo.load(
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
        if logout_btn is not None:
            logout_btn.click(
                fn=lambda: None,
                js="() => { window.location.href = '/logout'; }",
            )

    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
    }
    if auth_credentials:
        launch_kwargs["auth"] = auth_credentials
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
