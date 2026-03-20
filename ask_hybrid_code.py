#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import requests
import re
import json
import argparse
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


OLLAMA_URL = "http://localhost:11434/api/generate"


def load_meta(path):
    meta = []
    with open(path/"meta.jsonl") as f:
        for l in f:
            meta.append(json.loads(l))
    return meta


def embed(q, model):
    v = model.encode([q], convert_to_numpy=True)
    v = v/np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype("float32")


def semantic_search(idx, emb, k):
    D, I = idx.search(emb, k)
    return I[0]


def lexical_search(q, lex, meta):
    tokens = re.findall(r"[A-Za-z_]\w+", q.lower())
    ids = set()
    for t in tokens:
        if t in lex:
            ids.update(lex[t])
    return list(ids)


def call_llm(prompt, model):
    r = requests.post(
        OLLAMA_URL,
        json=dict(
            model=model,
            prompt=prompt,
            stream=False
        )
    )
    return r.json()["response"]


def build_prompt(frags, q):
    ctx = ""
    for f in frags:
        ctx += f"""
file: {f["file"]}
function: {f["name"]}
code:
{f["code"]}
"""
    return f"""
Используй код чтобы ответить на вопрос.

{ctx}

Вопрос:
{q}

Ответ:
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--model", default="qwen2.5-coder")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    idx = faiss.read_index(str(Path(args.index_dir)/"semantic.faiss"))
    meta = load_meta(Path(args.index_dir))
    with open(Path(args.index_dir)/"lexical_index.json") as f:
        lex = json.load(f)

    model = SentenceTransformer(args.embed_model)

    while True:
        q = input("> ")
        emb = embed(q, model)
        sem = semantic_search(idx, emb, args.top_k)
        lex_ids = lexical_search(q, lex, meta)
        ids = list(set(sem.tolist()+lex_ids))[:args.top_k]
        frags = [meta[i] for i in ids]
        prompt = build_prompt(frags, q)
        ans = call_llm(prompt, args.model)
        print(ans)


if __name__ == "__main__":
    main()
