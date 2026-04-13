#!/usr/bin/env python3
"""Browser chat UI for fuzz-coder using Gradio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
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

TYPING_PLACEHOLDER_TEXT = "..."
TYPING_PLACEHOLDER_HTML = "<div class='fc-chat-typing fc-typing-dots'><span></span><span></span><span></span></div>"
PROJECT_ID_SEP = "::"


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
    type_init_index_path = index_dir / "type_init_index.json"
    type_init_index = core.load_json(type_init_index_path) if type_init_index_path.exists() else {}
    index_meta_path = index_dir / "index_meta.json"
    index_meta = core.load_json(index_meta_path) if index_meta_path.exists() else {}
    language_name = (index_meta or {}).get("language")
    return idx, meta, lex, special_indices, symbols, call_graph, called_by, function_hints, type_init_index, language_name


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
            if _is_typing_placeholder(a):
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
                if _is_typing_placeholder(content):
                    continue
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


def _is_typing_placeholder(content) -> bool:
    txt = _content_to_text(content).strip()
    return txt in {TYPING_PLACEHOLDER_TEXT, TYPING_PLACEHOLDER_HTML}


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


def _load_user_histories(history_dir: Path, username: str, project_ids: List[str]):
    result = {pid: [] for pid in project_ids}
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

    for pid in project_ids:
        conv = projects_blob.get(pid, [])
        if not conv:
            # Backward compatibility: older history schema used project name as key.
            pname, _ptag = _split_project_id(pid)
            conv = projects_blob.get(pname, [])
        result[pid] = _normalize_conversation(conv)
    return result


def _load_user_last_project(history_dir: Path, username: str, project_ids: List[str]):
    p = _history_file_for_user(history_dir, username)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    project_id = str(data.get("last_project", "") or "").strip()
    if project_id in project_ids:
        return project_id
    # Backward compatibility: older schema stored only project name.
    if project_id:
        matches = [pid for pid in project_ids if _split_project_id(pid)[0] == project_id]
        if matches:
            return sorted(matches, reverse=True)[0]
    return ""


def _save_user_histories(
    history_dir: Path,
    username: str,
    user_histories,
    project_ids: List[str],
    *,
    last_project: str | None = None,
):
    history_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "username": username,
        "projects": {},
    }
    if last_project:
        payload["last_project"] = last_project
    for pid in project_ids:
        conv = _normalize_conversation(user_histories.get(pid, []))
        if conv:
            payload["projects"][pid] = [[u, a] for u, a in conv]

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


def _sanitize_tag_name(tag: str) -> str:
    raw = str(tag or "").strip()
    if not raw:
        return ""
    # Preserve human-readable tags (including spaces/colon) and only block path separators.
    cleaned = raw.replace("/", "-").replace("\\", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _default_project_tag() -> str:
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


def _compose_project_id(project_name: str, tag: str) -> str:
    return f"{project_name}{PROJECT_ID_SEP}{tag}"


def _split_project_id(project_id: str) -> tuple[str, str]:
    raw = str(project_id or "")
    if PROJECT_ID_SEP not in raw:
        return raw, "default"
    project_name, tag = raw.split(PROJECT_ID_SEP, 1)
    return project_name, tag or "default"


def _parse_index_suffix_to_project_tag(suffix: str) -> tuple[str, str]:
    token = str(suffix or "").strip()
    if not token:
        return "project", "default"
    if "_" not in token:
        return token, "default"
    project_name, tag = token.rsplit("_", 1)
    project_name = project_name.strip() or token
    tag = str(tag or "").strip() or "default"
    return project_name, tag


def _build_project_tag_map(projects: Dict[str, Path]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for pid in projects.keys():
        pname, ptag = _split_project_id(pid)
        out.setdefault(pname, {})[ptag] = pid
    return out


def _get_project_names(projects: Dict[str, Path]) -> List[str]:
    return sorted(_build_project_tag_map(projects).keys())


def _get_tags_for_project(projects: Dict[str, Path], project_name: str) -> List[str]:
    mapping = _build_project_tag_map(projects)
    tags = list(mapping.get(project_name, {}).keys())
    return sorted(tags, reverse=True)


def _resolve_project_id(projects: Dict[str, Path], project_name: str, tag: str) -> str | None:
    mapping = _build_project_tag_map(projects)
    by_tag = mapping.get(project_name, {})
    if not by_tag:
        return None
    if tag in by_tag:
        return by_tag[tag]
    tags = sorted(by_tag.keys(), reverse=True)
    return by_tag[tags[0]] if tags else None


def _discover_projects(args):
    projects: Dict[str, Path] = {}

    if args.index_dir:
        explicit = Path(args.index_dir).resolve()
        if not _looks_like_index_dir(explicit):
            raise FileNotFoundError(f"Invalid index directory (missing required files): {explicit}")
        name = explicit.name
        if name.startswith("index_data_"):
            name = name[len("index_data_"):]
        project_name, tag = _parse_index_suffix_to_project_tag(name)
        projects[_compose_project_id(project_name, tag)] = explicit
        return projects

    base_dir = Path(args.index_base_dir).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Index base directory does not exist: {base_dir}")

    for p in sorted(base_dir.glob("index_data_*")):
        if not _looks_like_index_dir(p):
            continue
        suffix = p.name[len("index_data_"):] or p.name
        project_name, tag = _parse_index_suffix_to_project_tag(suffix)
        pid = _compose_project_id(project_name, tag)
        if pid in projects:
            # Keep deterministic and collision-safe behavior.
            i = 2
            while True:
                alt_tag = f"{tag}-{i}"
                alt_pid = _compose_project_id(project_name, alt_tag)
                if alt_pid not in projects:
                    pid = alt_pid
                    break
                i += 1
        projects[pid] = p.resolve()

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


def _normalize_uploaded_zip_path(zip_file) -> str:
    if zip_file is None:
        return ""
    if isinstance(zip_file, (list, tuple)):
        if not zip_file:
            return ""
        zip_file = zip_file[0]
    return str(zip_file).strip()


def _clean_index_log_line(line: str) -> str:
    txt = str(line or "").replace("\r", "").rstrip("\n")
    # Strip ANSI escape sequences.
    txt = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", txt)
    return txt.strip()


def _build_index_status_text(project_name: str, log_lines: List[str]) -> str:
    tail = list(log_lines[-120:])
    if not tail:
        return f"Indexing `{project_name}`..."
    return (
        f"Indexing `{project_name}`...\n\n"
        "```text\n"
        + "\n".join(tail)
        + "\n```"
    )


def _build_pipeline_for_index(index_dir: Path, args, embed_model, reranker):
    idx, meta, lex, special_indices, symbols, call_graph, called_by, function_hints, type_init_index, language_name = _load_indices(index_dir)
    planner = core.QueryPlanner(
        special_indices,
        symbols,
        call_graph,
        called_by,
        meta=meta,
        language_name=language_name or ask_app.DEFAULT_CHAT_LANGUAGE,
    )
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


def _create_chatbot():
    """Create Chatbot and infer its effective history mode from the instance itself."""
    try:
        chatbot = gr.Chatbot(type="messages", elem_id="main_chatbot", sanitize_html=False)
    except TypeError:
        try:
            chatbot = gr.Chatbot(type="messages", elem_id="main_chatbot")
        except TypeError:
            try:
                chatbot = gr.Chatbot(elem_id="main_chatbot", sanitize_html=False)
            except TypeError:
                chatbot = gr.Chatbot(elem_id="main_chatbot")

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

    project_ids = list(projects.keys())
    project_names = _get_project_names(projects)
    if not project_names:
        raise FileNotFoundError("No projects found after discovery.")
    default_project = project_names[0]
    default_tags = _get_tags_for_project(projects, default_project)
    default_tag = default_tags[0] if default_tags else "default"
    default_project_id = _resolve_project_id(projects, default_project, default_tag) or project_ids[0]

    def _get_pipeline(project_id: str):
        if project_id not in projects:
            raise ValueError(f"Unknown project id: {project_id}")
        if project_id not in pipeline_cache:
            pipeline_cache[project_id] = _build_pipeline_for_index(
                projects[project_id], args, embed_model, reranker
            )
        return pipeline_cache[project_id]

    def _sync_projects_from_disk():
        if args.index_dir:
            return
        try:
            discovered = _discover_projects(args)
        except Exception:
            return
        changed = False
        for pid, path in discovered.items():
            if pid not in projects:
                projects[pid] = path.resolve()
                changed = True
        if changed:
            project_ids[:] = list(projects.keys())
            project_names[:] = _get_project_names(projects)

    chatbot_mode_holder = {"mode": "messages"}

    def _init_history_state(state):
        state = dict(state or {})
        users = state.get("users")
        if not isinstance(users, dict):
            users = {}
        state["users"] = users
        last_project = state.get("last_project")
        if not isinstance(last_project, dict):
            last_project = {}
        state["last_project"] = last_project
        return state

    def _ensure_user_histories(state, username: str):
        state = _init_history_state(state)
        users = state["users"]
        if username not in users:
            users[username] = _load_user_histories(history_dir, username, project_ids)
        user_histories = users[username]
        user_last = state["last_project"]
        if username not in user_last:
            loaded_last = _load_user_last_project(history_dir, username, project_ids)
            if loaded_last:
                user_last[username] = loaded_last
        for pid in project_ids:
            if pid not in user_histories:
                user_histories[pid] = []
            else:
                user_histories[pid] = _normalize_conversation(user_histories[pid])
        return user_histories, state

    def _on_project_change(project_name, project_tag, histories, request: gr.Request = None):
        username = _user_from_request(request)
        user_histories, histories = _ensure_user_histories(histories, username)
        tags = _get_tags_for_project(projects, project_name)
        selected_tag = project_tag if project_tag in tags else (tags[0] if tags else "default")
        selected_project_id = _resolve_project_id(projects, project_name, selected_tag) or default_project_id
        histories["last_project"][username] = selected_project_id
        _save_user_histories(
            history_dir,
            username,
            user_histories,
            project_ids,
            last_project=selected_project_id,
        )
        mode = chatbot_mode_holder["mode"]
        project_hist = _conversation_to_history(user_histories.get(selected_project_id, []), mode)
        return gr.update(choices=tags, value=selected_tag), project_hist, histories

    def _on_page_load(project_name, project_tag, histories, request: gr.Request = None):
        _sync_projects_from_disk()
        username = _user_from_request(request)
        user_histories, histories = _ensure_user_histories(histories, username)
        remembered_project_id = str(histories.get("last_project", {}).get(username, "") or "").strip()

        current_project = project_name if project_name in project_names else default_project
        current_tag = project_tag or ""
        if remembered_project_id in project_ids:
            remembered_project, remembered_tag = _split_project_id(remembered_project_id)
            if remembered_project in project_names:
                current_project = remembered_project
                current_tag = remembered_tag

        tags = _get_tags_for_project(projects, current_project)
        if current_tag not in tags:
            current_tag = tags[0] if tags else "default"

        selected_project_id = _resolve_project_id(projects, current_project, current_tag) or default_project_id
        histories["last_project"][username] = selected_project_id
        _save_user_histories(
            history_dir,
            username,
            user_histories,
            project_ids,
            last_project=selected_project_id,
        )
        mode = chatbot_mode_holder["mode"]
        project_hist = _conversation_to_history(user_histories.get(selected_project_id, []), mode)
        return (
            gr.update(choices=project_names, value=current_project),
            gr.update(choices=tags, value=current_tag),
            project_hist,
            histories,
        )

    def _append_pending_turn(chat_history, q: str, mode: str):
        normalized = _normalize_history_for_mode(chat_history, mode)
        if mode == "messages":
            normalized.append({"role": "user", "content": q})
            normalized.append({"role": "assistant", "content": TYPING_PLACEHOLDER_HTML})
            return normalized
        normalized.append((q, TYPING_PLACEHOLDER_HTML))
        return normalized

    def _finalize_pending_turn(chat_history, q: str, ans: str, mode: str):
        normalized = _normalize_history_for_mode(chat_history, mode)
        if mode == "messages":
            if (
                len(normalized) >= 2
                and isinstance(normalized[-1], dict)
                and isinstance(normalized[-2], dict)
                and normalized[-1].get("role") == "assistant"
                and normalized[-2].get("role") == "user"
                and _content_to_text(normalized[-2].get("content")) == q
                and _is_typing_placeholder(normalized[-1].get("content"))
            ):
                normalized[-1] = {"role": "assistant", "content": ans}
                return normalized
            normalized.append({"role": "assistant", "content": ans})
            return normalized

        if normalized and isinstance(normalized[-1], (list, tuple)) and len(normalized[-1]) >= 2:
            last_user = _content_to_text(normalized[-1][0])
            last_assistant = normalized[-1][1]
            if last_user == q and (last_assistant is None or _is_typing_placeholder(last_assistant)):
                normalized[-1] = (last_user, ans)
                return normalized
        normalized.append((q, ans))
        return normalized

    def _drop_pending_typing(chat_history, mode: str):
        normalized = _normalize_history_for_mode(chat_history, mode)
        if mode == "messages":
            while (
                normalized
                and isinstance(normalized[-1], dict)
                and normalized[-1].get("role") == "assistant"
                and _is_typing_placeholder(normalized[-1].get("content"))
            ):
                normalized.pop()
            return normalized

        if normalized and isinstance(normalized[-1], (list, tuple)) and len(normalized[-1]) >= 2:
            last_user = _content_to_text(normalized[-1][0])
            last_assistant = normalized[-1][1]
            if _is_typing_placeholder(last_assistant):
                normalized[-1] = (last_user, None)
        return normalized

    def _chat_submit(message, chat_history, project_name, project_tag, histories, request: gr.Request = None):
        username = _user_from_request(request)
        user_histories, histories = _ensure_user_histories(histories, username)
        project_id = _resolve_project_id(projects, project_name, project_tag)
        if not project_id:
            raise ValueError(f"Unknown project/tag: {project_name} / {project_tag}")
        mode = chatbot_mode_holder["mode"]
        chat_history = _normalize_history_for_mode(chat_history, mode)
        conversation_history = _history_to_conversation(chat_history)
        q = (message or "").strip()
        if not q:
            return "", chat_history, histories

        pending_history = _append_pending_turn(chat_history, q, mode)
        yield "", pending_history, histories

        token_q = queue.Queue()
        done = threading.Event()
        run_state = {"answer": "", "error": None}

        def _on_token(token: str):
            if token:
                token_q.put(str(token))

        def _run_worker():
            try:
                if ask_app.is_help_query(q):
                    lang = getattr(_get_pipeline(project_id), "language_name", None) if project_id else None
                    run_state["answer"] = ask_app.render_help_text(language_name=lang or ask_app.DEFAULT_CHAT_LANGUAGE)
                else:
                    pipeline = _get_pipeline(project_id)
                    expanded_q, _alias_used = ask_app.expand_chat_alias(
                        q,
                        language_name=(getattr(pipeline, "language_name", None) or ask_app.DEFAULT_CHAT_LANGUAGE),
                    )
                    result = pipeline.run(expanded_q, conversation_history, token_callback=_on_token)
                    run_state["answer"] = result.answer
            except Exception as e:
                run_state["error"] = f"[ERROR] {type(e).__name__}: {e}"
            finally:
                done.set()

        worker = threading.Thread(target=_run_worker, daemon=True)
        worker.start()

        streamed_answer = ""
        while True:
            emitted = False
            while True:
                try:
                    streamed_answer += token_q.get_nowait()
                    emitted = True
                except queue.Empty:
                    break
            if emitted:
                partial_history = _finalize_pending_turn(pending_history, q, streamed_answer, mode)
                yield "", partial_history, histories

            if done.is_set():
                try:
                    streamed_answer += token_q.get_nowait()
                    continue
                except queue.Empty:
                    break
            done.wait(0.03)

        ans = run_state["error"] if run_state["error"] else run_state["answer"]
        final_history = _finalize_pending_turn(pending_history, q, ans, mode)
        if ans != streamed_answer:
            yield "", final_history, histories

        conversation_history = _history_to_conversation(final_history)
        user_histories[project_id] = conversation_history
        histories["last_project"][username] = project_id
        _save_user_histories(
            history_dir,
            username,
            user_histories,
            project_ids,
            last_project=project_id,
        )

    def _chat_request_started(message):
        if not str(message or "").strip():
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    def _chat_request_finished():
        return gr.update(visible=True), gr.update(visible=False)

    def _cancel_chat_request(chat_history):
        mode = chatbot_mode_holder["mode"]
        cleaned = _drop_pending_typing(chat_history, mode)
        return cleaned, gr.update(visible=True), gr.update(visible=False)

    def _clear_project_chat(project_name, project_tag, histories, request: gr.Request = None):
        username = _user_from_request(request)
        user_histories, histories = _ensure_user_histories(histories, username)
        project_id = _resolve_project_id(projects, project_name, project_tag) or default_project_id
        user_histories[project_id] = []
        _save_user_histories(
            history_dir,
            username,
            user_histories,
            project_ids,
            last_project=histories.get("last_project", {}).get(username) or project_id,
        )
        return _conversation_to_history([], chatbot_mode_holder["mode"]), histories

    def _add_project_from_zip(
        zip_file,
        project_name_input,
        project_tag_input,
        language_name,
        current_project_name,
        current_project_tag,
        histories,
        request: gr.Request = None,
    ):
        histories = _init_history_state(histories)
        username = _user_from_request(request)
        current_project_name = current_project_name or default_project
        current_project_tag = current_project_tag or default_tag
        zip_value = _normalize_uploaded_zip_path(zip_file)
        zip_ready = bool(zip_value)

        def _resp(
            status_text: str,
            *,
            keep_zip: bool = True,
            keep_ready: bool = True,
            next_project_name: str | None = None,
            next_project_tag: str | None = None,
            show_cancel: bool | None = None,
            show_add: bool | None = None,
            update_zip_value: bool = False,
        ):
            zip_still_selected = bool(keep_zip and zip_value)
            add_update = gr.update(interactive=(zip_ready if keep_ready else False))
            cancel_update = gr.update()
            zip_update = gr.update()
            if show_add is not None:
                add_update = gr.update(
                    interactive=(zip_ready if keep_ready else False),
                    visible=show_add,
                )
            if show_cancel is not None:
                cancel_update = gr.update(visible=show_cancel)
            if update_zip_value:
                zip_update = gr.update(value=(zip_value if keep_zip else None))
            return (
                (next_project_name or current_project_name),
                (next_project_tag or current_project_tag),
                histories,
                status_text,
                zip_update,
                add_update,
                cancel_update,
                gr.update(visible=zip_still_selected),
                gr.update(visible=zip_still_selected),
            )

        if args.index_dir:
            yield _resp("Adding projects is disabled when started with --index_dir (single-project mode).")
            return

        if not zip_value:
            yield _resp("Please select a .zip archive first.", keep_zip=False, keep_ready=False)
            return

        zip_path = Path(zip_value).expanduser().resolve()
        if not zip_path.exists():
            yield _resp(f"Uploaded file is not accessible on server: {zip_path}")
            return

        if zip_path.suffix.lower() != ".zip":
            yield _resp(f"Only .zip archives are supported, got: {zip_path.name}")
            return

        base_name = _derive_project_name(zip_path, project_name_input or "")
        if not base_name:
            yield _resp("Unable to derive project name. Please set Project Name explicitly.")
            return
        tag_name = _sanitize_tag_name(project_tag_input or "")
        if not tag_name:
            tag_name = _default_project_tag()
        new_project_id = _compose_project_id(base_name, tag_name)
        if new_project_id in project_ids or (index_base_dir / f"index_data_{base_name}_{tag_name}").exists():
            yield _resp(
                f"Project `{base_name}` with tag `{tag_name}` already exists. Choose another project/tag.",
                keep_zip=True,
                keep_ready=True,
            )
            return

        new_project_name = base_name
        new_project_tag = tag_name
        new_index_dir = index_base_dir / f"index_data_{new_project_name}_{new_project_tag}"

        index_entry = Path(__file__).resolve().parent / "index_fuzz_coder.py"
        cmd = [
            sys.executable,
            str(index_entry),
            "--src",
            str(zip_path),
            "--out",
            str(new_index_dir),
            "--embed_model",
            str(args.embed_model),
            "--embedding_backend",
            str(args.embedding_backend),
            "--language",
            str(language_name),
        ]

        log_lines: List[str] = []
        yield _resp(
            _build_index_status_text(f"{new_project_name}:{new_project_tag}", log_lines),
            keep_zip=True,
            keep_ready=True,
            show_cancel=True,
            show_add=False,
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        active_builds[username] = {"proc": proc, "out_dir": new_index_dir}
        try:
            if proc.stdout is not None:
                for raw in iter(proc.stdout.readline, ""):
                    clean = _clean_index_log_line(raw)
                    if not clean:
                        if proc.poll() is not None:
                            break
                        continue
                    log_lines.append(clean)
                    if len(log_lines) > 500:
                        log_lines = log_lines[-300:]
                    yield _resp(
                        _build_index_status_text(f"{new_project_name}:{new_project_tag}", log_lines),
                        keep_zip=True,
                        keep_ready=True,
                        show_cancel=True,
                        show_add=False,
                    )
            return_code = proc.wait()
        finally:
            active_builds.pop(username, None)

        details = "\n".join(log_lines).strip()
        ok = return_code == 0
        if not ok:
            short_details = (details or "").strip()
            short_details = short_details.splitlines()[-1] if short_details else "unknown indexer error"
            if args.verbose and details:
                print("[Add Project] Index build failed:")
                print(details[-4000:])
            yield _resp(
                f"Failed to build index for `{zip_path.name}`.\n\nReason: `{short_details}`",
                keep_zip=True,
                keep_ready=True,
                show_cancel=False,
                show_add=True,
            )
            return

        if not _looks_like_index_dir(new_index_dir):
            yield _resp(
                f"Indexer finished but output is incomplete: {new_index_dir}\n"
                "Expected semantic.faiss/meta.jsonl/indices files.",
                keep_zip=True,
                keep_ready=True,
                show_cancel=False,
                show_add=True,
            )
            return

        projects[new_project_id] = new_index_dir.resolve()
        if new_project_id not in project_ids:
            project_ids.append(new_project_id)
        project_names[:] = _get_project_names(projects)
        pipeline_cache.pop(new_project_id, None)

        for user_key, user_map in histories.get("users", {}).items():
            if isinstance(user_map, dict) and new_project_id not in user_map:
                user_map[new_project_id] = []
                _save_user_histories(
                    history_dir,
                    user_key,
                    user_map,
                    project_ids,
                    last_project=histories.get("last_project", {}).get(user_key) or new_project_id,
                )

        if args.verbose and details:
            print("[Add Project] Index build output:")
            print(details[-2000:])
        yield _resp(
            f"Added project `{new_project_name}` with tag `{new_project_tag}`.",
            keep_zip=False,
            keep_ready=False,
            next_project_name=new_project_name,
            next_project_tag=new_project_tag,
            show_cancel=False,
            show_add=True,
            update_zip_value=True,
        )
        return

    def _on_zip_change(zip_file):
        zip_value = _normalize_uploaded_zip_path(zip_file)
        zip_selected = bool(zip_value)
        if not zip_value:
            # Keep the current status message (do not erase add-project logs/errors).
            return (
                gr.update(),
                gr.update(interactive=False, visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        zip_path = Path(zip_value).expanduser().resolve()
        if not zip_path.exists():
            return (
                f"Uploaded file is not accessible on server: `{zip_path}`",
                gr.update(interactive=False, visible=True),
                gr.update(visible=False),
                gr.update(visible=zip_selected),
                gr.update(visible=zip_selected),
            )
        if zip_path.suffix.lower() != ".zip":
            return (
                f"Only .zip archives are supported, got: `{zip_path.name}`",
                gr.update(interactive=False, visible=True),
                gr.update(visible=False),
                gr.update(visible=zip_selected),
                gr.update(visible=zip_selected),
            )
        # Valid ZIP selected: enable Add button without overwriting status.
        return (
            gr.update(),
            gr.update(interactive=True, visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )

    def _cleanup_partial_index_dir(path: Path):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _cancel_project_upload(
        zip_file,
        project_name_input,
        project_tag_input,
        current_project_name,
        current_project_tag,
        request: gr.Request = None,
    ):
        username = _user_from_request(request)
        current_project_name = current_project_name or default_project
        current_project_tag = current_project_tag or default_tag
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
                    tag_name = _sanitize_tag_name(project_tag_input or "")
                    if not tag_name:
                        tag_name = _default_project_tag()
                    candidate = (index_base_dir / f"index_data_{base_name}_{tag_name}").resolve()
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
            gr.update(interactive=bool(_normalize_uploaded_zip_path(zip_file)), visible=True),
            gr.update(visible=False),
            current_project_name,
            current_project_tag,
            gr.update(visible=bool(_normalize_uploaded_zip_path(zip_file))),
            gr.update(visible=bool(_normalize_uploaded_zip_path(zip_file))),
        )

    def _clear_zip_selection():
        return (
            "ZIP selection canceled.",
            gr.update(value=None),
            gr.update(interactive=False, visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _refresh_project_dropdown(target_project_name: str):
        target = target_project_name if target_project_name in project_names else default_project
        return gr.update(choices=project_names, value=target)

    def _refresh_tag_dropdown(project_name: str, target_tag: str | None = None):
        pname = project_name if project_name in project_names else default_project
        tags = _get_tags_for_project(projects, pname)
        value = target_tag if target_tag in tags else (tags[0] if tags else "default")
        return gr.update(choices=tags, value=value)

    def _toggle_add_project_modal(is_open: bool):
        next_open = not bool(is_open)
        if next_open:
            return (
                gr.update(visible=True),
                True,
                gr.update(value="Close Add Project"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(visible=True),
            )
        # Closing: fully reset modal UI so nothing remains visible.
        return (
            gr.update(visible=False),
            False,
            gr.update(value="Add Project"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(interactive=False, visible=True),
        )

    def _finalize_add_project_ui(status_text):
        text = _content_to_text(status_text).strip().lower()
        if text.startswith("added project"):
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False, visible=True),
                gr.update(value=None),
            )
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    project_names[:] = _get_project_names(projects)
    default_project = project_names[0]
    default_tag = (_get_tags_for_project(projects, default_project) or ["default"])[0]
    language_choices = list(get_supported_language_names()) or ["c_cpp"]

    ui_css = """
    *, *::before, *::after {
      box-sizing: border-box;
    }
    .gradio-container {
      width: 100% !important;
      max-width: 100% !important;
      padding-left: 0 !important;
      padding-right: 0 !important;
    }
    #fc_layout {
      width: 100% !important;
      max-width: min(1500px, 96vw) !important;
      margin-left: auto !important;
      margin-right: auto !important;
      padding-left: 6px !important;
      padding-right: 6px !important;
      overflow-x: hidden !important;
    }
    #fc_layout > * {
      width: 100% !important;
    }
    #fc_layout .gr-row {
      width: 100% !important;
      margin-left: 0 !important;
      margin-right: 0 !important;
      flex-wrap: wrap !important;
      gap: 8px !important;
    }
    #fc_layout .gr-row > * {
      min-width: 0 !important;
    }
    #fc_top_project_row > * {
      flex: 1 1 320px !important;
    }
    #fc_add_project_row > * {
      flex: 0 1 auto !important;
    }
    #fc_actions_row { }
    #main_chatbot {
      height: clamp(320px, 60dvh, 760px) !important;
      min-height: 320px !important;
    }
    @media (min-width: 981px) and (orientation: portrait) {
      #main_chatbot {
        height: clamp(820px, 160dvh, 2000px) !important;
        min-height: 420px !important;
      }
    }
    @media (min-width: 981px) and (orientation: landscape) {
      #main_chatbot {
        height: clamp(380px, 65dvh, 900px) !important;
        min-height: 320px !important;
      }
    }
    #chat_input_row > * {
      min-width: 0 !important;
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
      margin-top: 8px !important;
      margin-bottom: 8px !important;
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
    .fc-chat-typing {
      margin-top: 2px !important;
      margin-bottom: 2px !important;
      min-height: 16px;
    }
    .fc-typing-dots {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      height: 14px;
    }
    .fc-typing-dots span {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: #9ca3af;
      display: inline-block;
      animation: fc-bounce 1s infinite ease-in-out;
    }
    .fc-typing-dots span:nth-child(2) {
      animation-delay: 0.12s;
    }
    .fc-typing-dots span:nth-child(3) {
      animation-delay: 0.24s;
    }
    @media (max-width: 980px) {
      #fc_layout {
        max-width: 100vw !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
      }
      #main_chatbot {
        height: clamp(260px, 52dvh, 620px) !important;
        min-height: 260px !important;
      }
      #chat_input_row {
        margin-bottom: max(8px, env(safe-area-inset-bottom)) !important;
      }
      #chat_send_btn button, #chat_cancel_btn button {
        min-width: 48px !important;
        width: 48px !important;
      }
    }
    @keyframes fc-bounce {
      0%, 80%, 100% {
        transform: translateY(0);
        opacity: 0.35;
      }
      40% {
        transform: translateY(-4px);
        opacity: 1;
      }
    }
    """

    with gr.Blocks(title="Fuzz Coder", css=ui_css) as demo:
        with gr.Column(elem_id="fc_layout"):
            gr.Markdown(
                "## Fuzz Coder\n"
                "Codebase analysis chat for fuzzing targets, examples, parameter semantics, and implementation details.\n"
                "Aliases: `help`, `fuzz`, `fuzz wide`, `more fuzz`, `more fuzz wide`, "
                "`example FUNCTION`, `explain function FUNCTION`, `explain struct STRUCT`."
            )
            with gr.Row(elem_id="fc_top_project_row"):
                project = gr.Dropdown(
                    choices=project_names,
                    value=default_project,
                    label="Project",
                )
                project_tag = gr.Dropdown(
                    choices=_get_tags_for_project(projects, default_project),
                    value=default_tag,
                    label="Tag",
                )
            with gr.Row(elem_id="fc_add_project_row"):
                open_add_project_btn = gr.Button(
                    "Add Project",
                    variant="secondary",
                    elem_id="open_add_project_btn",
                    min_width=150,
                )

            with gr.Row(visible=False, elem_id="add_project_modal_wrap") as add_project_modal_wrap:
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
                    with gr.Row(elem_id="clear_zip_row", visible=False) as clear_zip_row:
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
                        project_tag_input = gr.Textbox(
                            label="Tag (optional)",
                            placeholder="e.g. v1 (default: current datetime)",
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
                        cancel_add_project_btn = gr.Button("✕", variant="stop", visible=False, min_width=56)
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
            with gr.Row(elem_id="fc_actions_row"):
                clear_btn = gr.Button("Clear Current Project Chat")
                if auth_credentials:
                    logout_btn = gr.Button("Logout")

        histories_state = gr.State({"users": {}})
        add_project_target_state = gr.State(default_project)
        add_project_target_tag_state = gr.State(default_tag)
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
                clear_zip_row,
                add_project_status,
                zip_upload,
                add_project_btn,
            ],
            show_progress="hidden",
            queue=False,
        )
        project.change(
            fn=_on_project_change,
            inputs=[project, project_tag, histories_state],
            outputs=[project_tag, chatbot, histories_state],
        )
        project_tag.change(
            fn=_on_project_change,
            inputs=[project, project_tag, histories_state],
            outputs=[project_tag, chatbot, histories_state],
        )
        zip_change_evt = zip_upload.change(
            fn=_on_zip_change,
            inputs=[zip_upload],
            outputs=[add_project_status, add_project_btn, cancel_add_project_btn, clear_zip_select_btn, clear_zip_row],
        )
        clear_zip_select_btn.click(
            fn=_clear_zip_selection,
            outputs=[add_project_status, zip_upload, add_project_btn, cancel_add_project_btn, clear_zip_select_btn, clear_zip_row],
            cancels=[zip_change_evt],
            show_progress="hidden",
            queue=False,
        )
        add_project_btn.click(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
            ),
            outputs=[cancel_add_project_btn, add_project_btn],
            show_progress="hidden",
            queue=False,
        )
        add_project_evt = add_project_btn.click(
            fn=_add_project_from_zip,
            inputs=[zip_upload, project_name_input, project_tag_input, language_input, project, project_tag, histories_state],
            outputs=[
                add_project_target_state,
                add_project_target_tag_state,
                histories_state,
                add_project_status,
                zip_upload,
                add_project_btn,
                cancel_add_project_btn,
                clear_zip_select_btn,
                clear_zip_row,
            ],
            show_progress="minimal",
        )
        add_project_evt.then(
            fn=_finalize_add_project_ui,
            inputs=[add_project_status],
            outputs=[clear_zip_select_btn, clear_zip_row, cancel_add_project_btn, add_project_btn, zip_upload],
            show_progress="hidden",
        )
        add_project_evt.then(
            fn=_refresh_project_dropdown,
            inputs=[add_project_target_state],
            outputs=[project],
            show_progress="hidden",
        )
        add_project_evt.then(
            fn=lambda pname, tag_value: _refresh_tag_dropdown(pname, tag_value),
            inputs=[add_project_target_state, add_project_target_tag_state],
            outputs=[project_tag],
            show_progress="hidden",
        )
        cancel_add_project_btn.click(
            fn=_cancel_project_upload,
            inputs=[zip_upload, project_name_input, project_tag_input, project, project_tag],
            outputs=[
                add_project_status,
                zip_upload,
                add_project_btn,
                cancel_add_project_btn,
                add_project_target_state,
                add_project_target_tag_state,
                clear_zip_select_btn,
                clear_zip_row,
            ],
            cancels=[add_project_evt],
            show_progress="hidden",
        )
        cancel_add_project_btn.click(
            fn=lambda pname, ptag: _refresh_tag_dropdown(pname, ptag),
            inputs=[add_project_target_state, add_project_target_tag_state],
            outputs=[project_tag],
            show_progress="hidden",
            queue=False,
        )
        demo.load(
            fn=_on_page_load,
            inputs=[project, project_tag, histories_state],
            outputs=[project, project_tag, chatbot, histories_state],
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
            inputs=[msg, chatbot, project, project_tag, histories_state],
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
            inputs=[msg, chatbot, project, project_tag, histories_state],
            outputs=[msg, chatbot, histories_state],
        )
        submit_chat_evt.then(
            fn=_chat_request_finished,
            outputs=[send_btn, cancel_chat_btn],
            show_progress="hidden",
            queue=False,
        )

        cancel_chat_btn.click(
            fn=_cancel_chat_request,
            inputs=[chatbot],
            outputs=[chatbot, send_btn, cancel_chat_btn],
            cancels=[send_chat_evt, submit_chat_evt],
            show_progress="hidden",
            queue=False,
        )
        clear_btn.click(
            fn=_clear_project_chat,
            inputs=[project, project_tag, histories_state],
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
