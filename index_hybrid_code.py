#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import numpy as np
from collections import defaultdict
from pathlib import Path
import zipfile
import re
import json
import argparse
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


SUPPORTED_EXT = {
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"
}

CONTROL_KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "catch",
    "new", "delete", "throw", "else", "do"
}

CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def unpack_if_zip(src, dst):
    src = Path(src)
    if src.suffix == ".zip":
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(dst)
        return dst
    return src


def collect_files(src):
    files = []
    for root, dirs, names in os.walk(src):
        for n in names:
            p = Path(root)/n
            if p.suffix.lower() in SUPPORTED_EXT:
                files.append(p)
    return files


def read_text(path):
    try:
        return path.read_text(encoding="utf8")
    except:
        return path.read_text(errors="ignore")


def find_matching_brace(text, pos):
    depth = 0
    i = pos
    n = len(text)

    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_str:
            if not escape and ch == '"':
                in_str = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if in_char:
            if not escape and ch == "'":
                in_char = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if ch == '"':
            in_str = True
            escape = False
            i += 1
            continue

        if ch == "'":
            in_char = True
            escape = False
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


def extract_functions(text):
    res = []
    lines = text.splitlines()

    # offsets[line_no] = смещение начала строки в тексте
    offsets = []
    cur = 0
    for line in lines:
        offsets.append(cur)
        cur += len(line) + 1

    i = 0
    n = len(lines)

    while i < n:
        # быстро отсеиваем строки, где даже нет "("
        if "(" not in lines[i]:
            i += 1
            continue

        # собираем возможную сигнатуру до "{"
        start_i = i
        sig_lines = []
        found_body = False
        j = i

        while j < n and j < i + 20:
            sig_lines.append(lines[j])
            joined = "\n".join(sig_lines)

            # если раньше встретили ;, то это скорее декларация, а не определение
            semi_pos = joined.find(";")
            brace_pos = joined.find("{")

            if brace_pos != -1 and (semi_pos == -1 or brace_pos < semi_pos):
                found_body = True
                break

            if semi_pos != -1:
                break

            j += 1

        if not found_body:
            i += 1
            continue

        joined = "\n".join(sig_lines)
        brace_pos = joined.find("{")
        signature = joined[:brace_pos].strip()

        # отсекаем очевидно нефункциональные конструкции
        if not signature or ")" not in signature:
            i += 1
            continue

        compact = " ".join(signature.split())

        bad_prefixes = ("if ", "for ", "while ", "switch ",
                        "catch ", "#", "typedef ", "return ")
        if compact.startswith(bad_prefixes):
            i += 1
            continue

        lp = compact.find("(")
        head = compact[:lp].strip()
        if not head:
            i += 1
            continue

        m = re.search(r"([A-Za-z_]\w*)\s*$", head)
        if not m:
            i += 1
            continue

        name = m.group(1)
        if name in CONTROL_KEYWORDS:
            i += 1
            continue

        start_offset = offsets[start_i]

        # ищем "{" начиная с начала предполагаемой сигнатуры
        open_brace = text.find("{", start_offset)
        if open_brace == -1:
            i += 1
            continue

        end = find_matching_brace(text, open_brace)
        if end == -1:
            i += 1
            continue

        code = text[start_offset:end + 1]
        res.append(
            dict(
                name=name,
                signature=compact,
                code=code,
                body=code,
            )
        )
        # перепрыгиваем к строке после конца функции
        i = text.count("\n", 0, end) + 1

    return res


def detect_calls(body):
    calls = []
    for m in CALL_RE.finditer(body):
        n = m.group(1)
        if n not in CONTROL_KEYWORDS:
            calls.append(n)
    return list(set(calls))


def build_embeddings(chunks, model_name):
    model = SentenceTransformer(model_name)

    texts = []
    for c in chunks:
        t = f"""
name: {c["name"]}
signature: {c["signature"]}
file: {c["file"]}
code:
{c["code"]}
"""
        texts.append(t)

    emb = model.encode(texts, convert_to_numpy=True)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, 1e-9, None)
    return emb


def build_lexical_index(chunks):
    idx = defaultdict(list)
    for i, c in enumerate(chunks):
        words = re.findall(r"[A-Za-z_]\w+", c["code"].lower())
        for w in set(words):
            idx[w].append(i)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="index_data")
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    src = unpack_if_zip(args.src, out/"src")
    files = collect_files(src)

    print("Files:", len(files))

    chunks = []
    for f in tqdm(files):
        txt = read_text(f)
        funcs = extract_functions(txt)
        for fn in funcs:
            fn["file"] = str(f)
            fn["calls"] = detect_calls(fn["body"])
            chunks.append(fn)

    print("Functions:", len(chunks))

    for i, c in enumerate(chunks):
        c["id"] = i

    emb = build_embeddings(chunks, args.embed_model)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))

    faiss.write_index(index, str(out/"semantic.faiss"))

    with open(out/"meta.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c)+"\n")

    lex = build_lexical_index(chunks)
    with open(out/"lexical_index.json", "w") as f:
        json.dump(lex, f)

    symbols = defaultdict(list)

    for c in chunks:
        symbols[c["name"]].append(c["id"])
    with open(out/"symbols.json", "w") as f:
        json.dump(symbols, f)

    print("Index ready")


if __name__ == "__main__":
    main()
