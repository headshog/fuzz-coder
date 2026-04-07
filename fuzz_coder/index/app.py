from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import faiss
from tqdm import tqdm

from fuzz_coder.embeddings.registry import get_embedding_backend
from fuzz_coder.languages.registry import get_language_profile, get_supported_language_names

from . import core


DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".txt", ".adoc"}
DOC_SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "third_party", "vendor", "build", "dist",
    ".venv", "venv", "__pycache__", "index_data", ".idea", ".vscode",
}


def collect_doc_files(src: Path):
    files = []
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if any(part in DOC_SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in DOC_EXTENSIONS:
            continue
        files.append(p)
    return files


def build_function_hints(symbols, doc_files, max_hints_per_function=8):
    hints = defaultdict(list)
    if not symbols or not doc_files:
        return {}

    symbol_names = set(symbols.keys())
    symbol_names_lower = {s.lower(): s for s in symbol_names}

    for doc_file in doc_files:
        try:
            text = core.read_text(doc_file)
        except Exception:
            continue
        lines = text.splitlines()
        if not lines:
            continue

        for i, line in enumerate(lines):
            tokens = set(re.findall(r"[A-Za-z_]\w*", line))
            if not tokens:
                continue
            matched_symbols = []
            for tok in tokens:
                key = symbol_names_lower.get(tok.lower())
                if key is not None:
                    matched_symbols.append(key)
            if not matched_symbols:
                continue

            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippet = "\n".join(lines[start:end]).strip()[:500]
            for fn in matched_symbols:
                if len(hints[fn]) >= max_hints_per_function:
                    continue
                hints[fn].append({
                    "source": "docs",
                    "file": str(doc_file),
                    "line": i + 1,
                    "snippet": snippet,
                })

    return dict(hints)


def apply_language_profile(language_name: str) -> None:
    """Apply language profile to legacy global constants without behavior changes."""
    profile = get_language_profile(language_name)
    core.SUPPORTED_EXT = set(profile.supported_ext)
    core.CONTROL_KEYWORDS = set(profile.control_keywords)
    core.STDIN_PATTERNS = list(profile.stdin_patterns)
    core.FILE_INPUT_PATTERNS = list(profile.file_input_patterns)
    core.API_CALL_PATTERNS = list(profile.api_call_patterns)
    core.OUTPUT_PATTERNS = list(profile.output_patterns)
    core.MEMORY_MANAGEMENT_PATTERNS = list(profile.memory_management_patterns)
    core.ERROR_HANDLING_PATTERNS = list(profile.error_handling_patterns)
    core.TYPE_PATTERNS = dict(profile.type_patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="index_data")
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--language", default="c_cpp", choices=get_supported_language_names())
    ap.add_argument("--embedding_backend", default="sentence_transformers")
    args = ap.parse_args()

    apply_language_profile(args.language)

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    src = core.unpack_if_zip(args.src, out/"src")
    files = core.collect_files(src)

    print(f"Found {len(files)} source files")

    chunks = []
    parser = core.TreeSitterParser(language_name=args.language)

    for f in tqdm(files, desc="Parsing files"):
        txt = core.read_text(f)

        if core.HAS_TREE_SITTER and parser.parser:
            funcs = parser.parse_functions(str(f), txt)
        else:
            funcs = core.extract_functions_regex(str(f), txt, language_name=args.language)

        for fn in funcs:
            fn["file"] = str(f)
            fn["calls"] = core.detect_calls(fn["body"])

            # Detect input type
            input_info = core.detect_input_type(fn["code"])
            fn.update(input_info)

            # Detect additional code features
            code_features = core.detect_code_features(fn["code"])
            fn.update(code_features)

            chunks.append(fn)

    print(f"Extracted {len(chunks)} functions")

    # Assign IDs
    for i, c in enumerate(chunks):
        c["id"] = i

    # Build call graph
    print("Building call graph...")
    call_graph, called_by = core.build_call_graph(chunks)

    # Build embeddings
    print("Building semantic embeddings...")
    backend = get_embedding_backend(args.embed_model, backend=args.embedding_backend)
    emb = core.build_embeddings(chunks, args.embed_model, embedding_backend=backend)

    # Create FAISS index
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))

    faiss.write_index(index, str(out/"semantic.faiss"))

    # Save metadata
    with open(out/"meta.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c)+"\n")

    # Save lexical index
    lex = core.build_lexical_index(chunks)
    with open(out/"lexical_index.json", "w") as f:
        json.dump(lex, f)

    # Save symbols index
    symbols = defaultdict(list)
    for c in chunks:
        symbols[c["name"]].append(c["id"])
    with open(out/"symbols.json", "w") as f:
        json.dump(dict(symbols), f)

    # Save doc/guide hints linked to function names.
    doc_files = collect_doc_files(src)
    function_hints = build_function_hints(dict(symbols), doc_files)
    with open(out/"function_hints.json", "w") as f:
        json.dump(function_hints, f)

    # Save type initialization patterns (for better example generation fallback).
    type_init_index = core.build_type_init_index(chunks)
    with open(out/"type_init_index.json", "w") as f:
        json.dump(type_init_index, f)

    # Save call graph
    with open(out/"call_graph.json", "w") as f:
        json.dump(call_graph, f)

    # Save reverse call index (called_by)
    with open(out/"called_by.json", "w") as f:
        json.dump(dict(called_by), f)

    # Create special indices for different query types
    stdin_indices = [i for i, c in enumerate(chunks) if c.get("has_stdin")]
    file_indices = [i for i, c in enumerate(chunks) if c.get("has_file_input")]
    api_indices = [i for i, c in enumerate(chunks) if c.get("has_api_call")]

    # Additional special indices for enhanced features
    output_indices = [i for i, c in enumerate(chunks) if c.get("has_output")]
    memory_mgmt_indices = [i for i, c in enumerate(chunks) if c.get("uses_memory_management")]
    error_handling_indices = [i for i, c in enumerate(chunks) if c.get("has_error_handling")]

    # Type-based indices
    type_indices = {}
    for type_name in ["byte_array", "string", "integer", "float", "pointer", "reference", "template"]:
        type_indices[type_name] = [i for i, c in enumerate(chunks)
                                   if type_name in c.get("types_used", [])]

    # Backward/forward-compatible aliases for query-side type normalization
    type_indices["int"] = list(type_indices["integer"])
    type_indices["vector"] = list(type_indices["template"])

    with open(out/"special_indices.json", "w") as f:
        json.dump({
            "stdin": stdin_indices,
            "file_input": file_indices,
            "api_calls": api_indices,
            "output": output_indices,
            "memory_management": memory_mgmt_indices,
            "error_handling": error_handling_indices,
            "by_type": type_indices
        }, f)

    print(f"Index ready: {out}")
    print(f"  - Functions with stdin: {len(stdin_indices)}")
    print(f"  - Functions with file input: {len(file_indices)}")
    print(f"  - Functions with API calls: {len(api_indices)}")
    print(f"  - Functions with output: {len(output_indices)}")
    print(f"  - Functions with memory management: {len(memory_mgmt_indices)}")
    print(f"  - Functions with error handling: {len(error_handling_indices)}")
    print(f"  - Docs/guide hints linked to functions: {sum(len(v) for v in function_hints.values())}")
    types_count = len(type_init_index.get("types", {})) if isinstance(type_init_index, dict) else 0
    struct_types_count = len(type_init_index.get("struct_field_writes", {})) if isinstance(type_init_index, dict) else 0
    effects_functions_count = len(type_init_index.get("function_effects", {})) if isinstance(type_init_index, dict) else 0
    callsite_flow_count = sum(
        len(v) for v in type_init_index.get("callsite_arg_flow", {}).values()
    ) if isinstance(type_init_index, dict) else 0
    print(f"  - Types with observed init patterns: {types_count}")
    print(f"  - Struct types with field-write evidence: {struct_types_count}")
    print(f"  - Functions with parameter effects: {effects_functions_count}")
    print(f"  - Callsites with arg-flow traces: {callsite_flow_count}")


if __name__ == "__main__":
    main()
