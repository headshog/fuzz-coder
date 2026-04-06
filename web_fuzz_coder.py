#!/usr/bin/env python3
"""Browser chat UI for fuzz-coder using Gradio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import faiss
import gradio as gr
from sentence_transformers import CrossEncoder

from fuzz_coder.ask import app as ask_app
from fuzz_coder.ask import core
from fuzz_coder.ask.pipeline import PipelineConfig, QueryPipeline
from fuzz_coder.embeddings.registry import get_embedding_backend
from fuzz_coder.languages.registry import get_supported_language_names


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
            out.append((_content_to_text(u), _content_to_text(a)))
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
                pending_user = _content_to_text(content)
            elif role == "assistant" and pending_user is not None:
                out.append((pending_user, _content_to_text(content)))
                pending_user = None
        return out

    return []


def _conversation_to_history(conversation, mode: str):
    conversation = _normalize_conversation(conversation)
    if mode == "messages":
        out = []
        for user_text, assistant_text in conversation:
            out.append({"role": "user", "content": _content_to_text(user_text)})
            out.append({"role": "assistant", "content": _content_to_text(assistant_text)})
        return out
    return [(_content_to_text(user_text), _content_to_text(assistant_text)) for user_text, assistant_text in conversation]


def _content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, dict):
        # Gradio content block style: {"type": "text", "text": "..."}
        if content.get("type") == "text" and "text" in content:
            return str(content.get("text", ""))
        if "content" in content:
            return _content_to_text(content.get("content"))
        if "value" in content:
            return _content_to_text(content.get("value"))
        # Fallback: join key textual fields if present.
        parts = []
        for key in ("text", "caption", "name", "label"):
            if key in content and content.get(key) is not None:
                parts.append(str(content.get(key)))
        if parts:
            return " ".join(parts)
        return str(content)
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            txt = _content_to_text(item).strip()
            if txt:
                parts.append(txt)
        return "\n".join(parts)
    # ChatMessage-like objects or unknown payloads
    if hasattr(content, "text"):
        return _content_to_text(getattr(content, "text"))
    if hasattr(content, "content"):
        return _content_to_text(getattr(content, "content"))
    return str(content)


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
        out.append((_content_to_text(user_text), _content_to_text(assistant_text)))
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


def _sanitize_project_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return cleaned


def _derive_project_name(zip_path: Path, preferred_name: str) -> str:
    preferred = _sanitize_project_name(preferred_name)
    if preferred:
        return preferred

    stem = zip_path.stem
    stem = re.sub(r"\.(tar|gz|tgz|bz2|xz)$", "", stem, flags=re.IGNORECASE)
    stem = _sanitize_project_name(stem)
    return stem or "project"


def _pick_unique_project_name(base_name: str, existing_names: List[str]) -> str:
    if base_name not in existing_names:
        return base_name
    i = 2
    while True:
        candidate = f"{base_name}_{i}"
        if candidate not in existing_names:
            return candidate
        i += 1


def _normalize_uploaded_zip_path(zip_file) -> str:
    if zip_file is None:
        return ""
    if isinstance(zip_file, (list, tuple)):
        if not zip_file:
            return ""
        zip_file = zip_file[0]
    return str(zip_file).strip()


def _run_index_build(
    zip_path: Path,
    out_dir: Path,
    language: str,
    args,
    *,
    active_builds: dict | None = None,
    build_key: str | None = None,
) -> tuple[bool, str]:
    index_entry = Path(__file__).resolve().parent / "index_fuzz_coder.py"
    cmd = [
        sys.executable,
        str(index_entry),
        "--src",
        str(zip_path),
        "--out",
        str(out_dir),
        "--embed_model",
        str(args.embed_model),
        "--embedding_backend",
        str(args.embedding_backend),
        "--language",
        str(language),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if active_builds is not None and build_key:
        active_builds[build_key] = {"proc": proc, "out_dir": out_dir}
    stdout, stderr = proc.communicate()
    if active_builds is not None and build_key:
        active_builds.pop(build_key, None)
    stdout = (stdout or "").strip()
    stderr = (stderr or "").strip()
    if proc.returncode != 0:
        details = stderr or stdout or f"indexer exited with code {proc.returncode}"
        return False, details
    details = stdout or "index build completed"
    return True, details


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
        chatbot = gr.Chatbot(height=820, type="messages")
    except TypeError:
        chatbot = gr.Chatbot(height=820)

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
                out.append({"role": role, "content": _content_to_text(msg.get("content", ""))})
            return out
        # Convert ChatMessage-like objects -> messages
        if history and hasattr(history[0], "role") and hasattr(history[0], "content"):
            out = []
            for msg in history:
                role = str(getattr(msg, "role", "")).strip()
                if role not in {"user", "assistant", "system"}:
                    continue
                out.append({"role": role, "content": _content_to_text(getattr(msg, "content", ""))})
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
            out.append({"role": "user", "content": _content_to_text(u)})
            out.append({"role": "assistant", "content": _content_to_text(a)})
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
            pending_user = _content_to_text(content)
        elif role == "assistant":
            if pending_user is None:
                pending_user = ""
            out.append((pending_user, _content_to_text(content)))
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
    index_base_dir = Path(args.index_base_dir).expanduser().resolve()
    pipeline_cache: Dict[str, QueryPipeline] = {}
    active_builds: Dict[str, dict] = {}

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

    def _chat_request_started(message):
        if not str(message or "").strip():
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    def _chat_request_finished():
        return gr.update(visible=True), gr.update(visible=False)

    def _clear_project_chat(project_name, histories, request: gr.Request = None):
        username = _user_from_request(request)
        user_histories, histories = _ensure_user_histories(histories, username)
        user_histories[project_name] = []
        _save_user_histories(history_dir, username, user_histories, project_names)
        return _conversation_to_history([], chatbot_mode_holder["mode"]), histories

    def _add_project_from_zip(
        zip_file,
        project_name_input,
        language_name,
        current_project_name,
        histories,
        request: gr.Request = None,
    ):
        histories = _init_history_state(histories)
        username = _user_from_request(request)
        current_project_name = current_project_name or default_project
        zip_value = _normalize_uploaded_zip_path(zip_file)
        zip_ready = bool(zip_value)

        def _resp(
            status_text: str,
            *,
            keep_zip: bool = True,
            keep_ready: bool = True,
            next_project_name: str | None = None,
        ):
            zip_still_selected = bool(keep_zip and zip_value)
            return (
                (next_project_name or current_project_name),
                histories,
                status_text,
                gr.update(value=(zip_value if keep_zip else None)),
                gr.update(interactive=(zip_ready if keep_ready else False)),
                gr.update(visible=False),
                gr.update(visible=zip_still_selected),
            )

        if args.index_dir:
            return _resp("Adding projects is disabled when started with --index_dir (single-project mode).")

        if not zip_value:
            return _resp("Please select a .zip archive first.", keep_zip=False, keep_ready=False)

        zip_path = Path(zip_value).expanduser().resolve()
        if not zip_path.exists():
            return _resp(f"Uploaded file is not accessible on server: {zip_path}")

        if zip_path.suffix.lower() != ".zip":
            return _resp(f"Only .zip archives are supported, got: {zip_path.name}")

        base_name = _derive_project_name(zip_path, project_name_input or "")
        if not base_name:
            return _resp("Unable to derive project name. Please set Project Name explicitly.")
        if base_name in project_names or (index_base_dir / f"index_data_{base_name}").exists():
            return _resp(
                f"Project `{base_name}` already exists. Choose another project name.",
                keep_zip=True,
                keep_ready=True,
            )

        new_project_name = base_name
        new_index_dir = index_base_dir / f"index_data_{new_project_name}"

        ok, details = _run_index_build(
            zip_path,
            new_index_dir,
            language_name,
            args,
            active_builds=active_builds,
            build_key=username,
        )
        if not ok:
            short_details = (details or "").strip()
            short_details = short_details.splitlines()[-1] if short_details else "unknown indexer error"
            if args.verbose and details:
                print("[Add Project] Index build failed:")
                print(details[-4000:])
            return _resp(
                f"Failed to build index for `{zip_path.name}`.\n\nReason: `{short_details}`",
                keep_zip=True,
                keep_ready=True,
            )

        if not _looks_like_index_dir(new_index_dir):
            return _resp(
                f"Indexer finished but output is incomplete: {new_index_dir}\n"
                "Expected semantic.faiss/meta.jsonl/indices files.",
                keep_zip=True,
                keep_ready=True,
            )

        projects[new_project_name] = new_index_dir.resolve()
        if new_project_name not in project_names:
            project_names.append(new_project_name)
        pipeline_cache.pop(new_project_name, None)

        for user_key, user_map in histories.get("users", {}).items():
            if isinstance(user_map, dict) and new_project_name not in user_map:
                user_map[new_project_name] = []
                _save_user_histories(history_dir, user_key, user_map, project_names)

        if args.verbose and details:
            print("[Add Project] Index build output:")
            print(details[-2000:])
        return _resp(
            f"Added project `{new_project_name}`.",
            keep_zip=False,
            keep_ready=False,
            next_project_name=new_project_name,
        )

    def _on_zip_change(zip_file):
        zip_value = _normalize_uploaded_zip_path(zip_file)
        zip_selected = bool(zip_value)
        if not zip_value:
            # Keep the current status message (do not erase add-project logs/errors).
            return gr.update(), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=False)
        zip_path = Path(zip_value).expanduser().resolve()
        if not zip_path.exists():
            return (
                f"Uploaded file is not accessible on server: `{zip_path}`",
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=zip_selected),
            )
        if zip_path.suffix.lower() != ".zip":
            return (
                f"Only .zip archives are supported, got: `{zip_path.name}`",
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=zip_selected),
            )
        # Valid ZIP selected: enable Add button without overwriting status.
        return gr.update(), gr.update(interactive=True), gr.update(visible=False), gr.update(visible=True)

    def _cleanup_partial_index_dir(path: Path):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _cancel_project_upload(zip_file, project_name_input, current_project_name, request: gr.Request = None):
        username = _user_from_request(request)
        current_project_name = current_project_name or default_project
        removed_any = False

        active = active_builds.pop(username, None)
        if active:
            proc = active.get("proc")
            out_dir = active.get("out_dir")
            if proc is not None and getattr(proc, "poll", None) and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if isinstance(out_dir, Path):
                removed_any = _cleanup_partial_index_dir(out_dir.resolve()) or removed_any

        zip_value = _normalize_uploaded_zip_path(zip_file)
        if zip_value:
            try:
                zip_path = Path(zip_value).expanduser().resolve()
                base_name = _derive_project_name(zip_path, project_name_input or "")
                if base_name:
                    candidate = (index_base_dir / f"index_data_{base_name}").resolve()
                    protected_dirs = {p.resolve() for p in projects.values()}
                    if candidate not in protected_dirs:
                        removed_any = _cleanup_partial_index_dir(candidate) or removed_any
            except Exception:
                pass

        status = "Project upload canceled."
        if removed_any:
            status = "Project upload canceled. Partial index directory was removed."
        return (
            status,
            gr.update(value=zip_file),
            gr.update(interactive=bool(_normalize_uploaded_zip_path(zip_file))),
            gr.update(visible=False),
            current_project_name,
            gr.update(visible=bool(_normalize_uploaded_zip_path(zip_file))),
        )

    def _clear_zip_selection():
        return (
            "ZIP selection canceled.",
            gr.update(value=None),
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _refresh_project_dropdown(target_project_name: str):
        target = target_project_name or default_project
        return gr.update(choices=project_names, value=target)

    def _toggle_add_project_modal(is_open: bool):
        next_open = not bool(is_open)
        if next_open:
            return (
                gr.update(visible=True),
                True,
                gr.update(value="Close Add Project"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        # Closing: fully reset modal UI so nothing remains visible.
        return (
            gr.update(visible=False),
            False,
            gr.update(value="Add Project"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(interactive=False),
        )

    project_names = list(projects.keys())
    default_project = project_names[0]
    language_choices = list(get_supported_language_names()) or ["c_cpp"]

    ui_css = """
    .gradio-container {
      max-width: min(1680px, 98vw) !important;
      padding-left: 14px !important;
      padding-right: 14px !important;
    }
    #add_project_btn button {
      background: #f59e0b !important;
      border-color: #f59e0b !important;
      color: #111827 !important;
    }
    #add_project_btn button[disabled] {
      background: #9ca3af !important;
      border-color: #9ca3af !important;
      color: #111827 !important;
      opacity: 0.55 !important;
    }
    #cancel_zip_select_btn button {
      min-width: 42px !important;
      width: 42px !important;
      padding-left: 0 !important;
      padding-right: 0 !important;
      font-weight: 700 !important;
    }
    #chat_input_row {
      background: var(--block-background-fill, #1f2937);
      border: 1px solid var(--border-color-primary, #374151);
      border-radius: 12px;
      padding: 8px;
      align-items: center !important;
      gap: 8px !important;
    }
    #chat_input_row .gr-textbox,
    #chat_input_row .gr-textbox textarea,
    #chat_input_row textarea,
    #chat_input_row input {
      background: transparent !important;
      color: var(--body-text-color, #e5e7eb) !important;
    }
    #chat_input_row .gr-textbox,
    #chat_input_row .gr-textbox > div,
    #chat_input_row .gr-textbox .wrap,
    #chat_input_row .gr-textbox .container {
      border: 0 !important;
      box-shadow: none !important;
      background: transparent !important;
      padding: 0 !important;
    }
    #chat_send_btn button, #chat_cancel_btn button {
      min-width: 56px !important;
      width: 56px !important;
      height: 44px !important;
      padding: 0 !important;
      font-size: 20px !important;
      line-height: 1 !important;
    }
    #add_project_modal {
      width: 100%;
      margin-top: 12px;
      padding: 18px;
      border-radius: 16px;
      border: 1px solid var(--border-color-primary, #374151);
      background: var(--block-background-fill, #111827);
      box-shadow:
        0 16px 38px rgba(0, 0, 0, 0.30),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
      animation: fc-slide-down 0.22s ease-out;
    }
    #project_zip_upload,
    #project_zip_upload > div,
    #project_zip_upload .wrap,
    #project_zip_upload .container {
      background: var(--block-background-fill, #111827) !important;
      border: 0 !important;
      box-shadow: none !important;
      padding: 0 !important;
    }
    #project_zip_upload button {
      width: 100% !important;
      min-width: 0 !important;
      background: var(--block-background-fill, #111827) !important;
      border-color: var(--border-color-primary, #374151) !important;
      color: var(--body-text-color, #e5e7eb) !important;
    }
    #project_zip_upload button:hover,
    #project_zip_upload button:active {
      background: var(--block-background-fill, #111827) !important;
      border-color: var(--border-color-primary, #374151) !important;
    }
    #clear_zip_row {
      justify-content: flex-end !important;
      margin-top: -4px !important;
      margin-bottom: 4px !important;
    }
    #add_project_modal h3 {
      margin-top: 0 !important;
      margin-bottom: 10px !important;
      font-weight: 700 !important;
    }
    #add_project_modal .gr-button {
      border-radius: 10px !important;
    }
    @keyframes fc-slide-down {
      from {
        opacity: 0;
        transform: translateY(-14px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    #open_add_project_btn button {
      background: #2563eb !important;
      border-color: #2563eb !important;
      color: white !important;
    }
    """

    with gr.Blocks(title="Fuzz Coder", css=ui_css) as demo:
        gr.Markdown(
            "## Fuzz Coder\n"
            "Codebase analysis chat for fuzzing targets, examples, parameter semantics, and implementation details.\n"
            "Aliases: `fuzz`, `fuzz wide`, `more fuzz`, `more fuzz wide`, `example FUNCTION`, `explain FUNCTION`."
        )
        with gr.Row():
            project = gr.Dropdown(
                choices=project_names,
                value=default_project,
                label="Project (index_data_PROJECT)",
            )
        with gr.Row():
            open_add_project_btn = gr.Button(
                "Add Project",
                variant="secondary",
                elem_id="open_add_project_btn",
                min_width=150,
            )

        with gr.Row(visible=False) as add_project_modal_wrap:
            with gr.Group(elem_id="add_project_modal") as add_project_modal:
                gr.Markdown("### Add Project From ZIP")
                add_project_status = gr.Markdown("")
                with gr.Row():
                    zip_upload = gr.File(
                        label="Project ZIP Archive",
                        file_types=[".zip"],
                        type="filepath",
                        elem_id="project_zip_upload",
                    )
                with gr.Row(elem_id="clear_zip_row"):
                    clear_zip_select_btn = gr.Button(
                        "✕",
                        visible=False,
                        elem_id="cancel_zip_select_btn",
                        min_width=42,
                    )
                with gr.Row():
                    project_name_input = gr.Textbox(
                        label="Project Name (optional)",
                        placeholder="e.g. pytorch",
                    )
                    language_input = gr.Dropdown(
                        choices=language_choices,
                        value="c_cpp" if "c_cpp" in language_choices else language_choices[0],
                        label="Language",
                    )
                with gr.Row():
                    add_project_btn = gr.Button(
                        "Add Project from ZIP",
                        variant="primary",
                        interactive=False,
                        elem_id="add_project_btn",
                    )
                    cancel_add_project_btn = gr.Button("✕ Cancel Upload", variant="stop", visible=False)
        chatbot, chatbot_mode = _create_chatbot()
        chatbot_mode_holder["mode"] = chatbot_mode
        if args.verbose:
            print(f"[Web UI] Chatbot mode: {chatbot_mode}")
            print(f"[Web UI] Auth enabled: {'yes' if auth_credentials else 'no'}")
            print(f"[Web UI] History dir: {history_dir}")
        with gr.Row(elem_id="chat_input_row"):
            msg = gr.Textbox(
                placeholder="Ask about functions, fuzz targets, or type 'help'",
                show_label=False,
                scale=12,
                container=False,
            )
            send_btn = gr.Button("→", variant="primary", scale=1, min_width=56, elem_id="chat_send_btn")
            cancel_chat_btn = gr.Button("■", variant="stop", visible=False, scale=1, min_width=56, elem_id="chat_cancel_btn")
        logout_btn = None
        with gr.Row():
            clear_btn = gr.Button("Clear Current Project Chat")
            if auth_credentials:
                logout_btn = gr.Button("Logout")

        histories_state = gr.State({"users": {}})
        add_project_target_state = gr.State(default_project)
        add_project_modal_open_state = gr.State(False)

        open_add_project_btn.click(
            fn=_toggle_add_project_modal,
            inputs=[add_project_modal_open_state],
            outputs=[
                add_project_modal_wrap,
                add_project_modal_open_state,
                open_add_project_btn,
                cancel_add_project_btn,
                clear_zip_select_btn,
                add_project_status,
                zip_upload,
                add_project_btn,
            ],
            show_progress="hidden",
            queue=False,
        )
        project.change(
            fn=_on_project_change,
            inputs=[project, histories_state],
            outputs=[chatbot, histories_state],
        )
        zip_change_evt = zip_upload.change(
            fn=_on_zip_change,
            inputs=[zip_upload],
            outputs=[add_project_status, add_project_btn, cancel_add_project_btn, clear_zip_select_btn],
        )
        clear_zip_select_btn.click(
            fn=_clear_zip_selection,
            outputs=[add_project_status, zip_upload, add_project_btn, cancel_add_project_btn, clear_zip_select_btn],
            cancels=[zip_change_evt],
            show_progress="hidden",
            queue=False,
        )
        show_cancel_evt = add_project_btn.click(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(interactive=False),
            ),
            outputs=[cancel_add_project_btn, add_project_btn],
            show_progress="hidden",
        )
        add_project_evt = show_cancel_evt.then(
            fn=_add_project_from_zip,
            inputs=[zip_upload, project_name_input, language_input, project, histories_state],
            outputs=[
                add_project_target_state,
                histories_state,
                add_project_status,
                zip_upload,
                add_project_btn,
                cancel_add_project_btn,
                clear_zip_select_btn,
            ],
            show_progress="full",
        )
        add_project_evt.then(
            fn=_refresh_project_dropdown,
            inputs=[add_project_target_state],
            outputs=[project],
            show_progress="hidden",
        )
        cancel_add_project_btn.click(
            fn=_cancel_project_upload,
            inputs=[zip_upload, project_name_input, project],
            outputs=[
                add_project_status,
                zip_upload,
                add_project_btn,
                cancel_add_project_btn,
                add_project_target_state,
                clear_zip_select_btn,
            ],
            cancels=[add_project_evt],
            show_progress="hidden",
        )
        demo.load(
            fn=_on_project_change,
            inputs=[project, histories_state],
            outputs=[chatbot, histories_state],
        )
        send_start_evt = send_btn.click(
            fn=_chat_request_started,
            inputs=[msg],
            outputs=[send_btn, cancel_chat_btn],
            show_progress="hidden",
            queue=False,
        )
        send_chat_evt = send_start_evt.then(
            fn=_chat_submit,
            inputs=[msg, chatbot, project, histories_state],
            outputs=[msg, chatbot, histories_state],
        )
        send_chat_evt.then(
            fn=_chat_request_finished,
            outputs=[send_btn, cancel_chat_btn],
            show_progress="hidden",
            queue=False,
        )

        submit_start_evt = msg.submit(
            fn=_chat_request_started,
            inputs=[msg],
            outputs=[send_btn, cancel_chat_btn],
            show_progress="hidden",
            queue=False,
        )
        submit_chat_evt = submit_start_evt.then(
            fn=_chat_submit,
            inputs=[msg, chatbot, project, histories_state],
            outputs=[msg, chatbot, histories_state],
        )
        submit_chat_evt.then(
            fn=_chat_request_finished,
            outputs=[send_btn, cancel_chat_btn],
            show_progress="hidden",
            queue=False,
        )

        cancel_chat_btn.click(
            fn=_chat_request_finished,
            outputs=[send_btn, cancel_chat_btn],
            cancels=[send_chat_evt, submit_chat_evt],
            show_progress="hidden",
            queue=False,
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
