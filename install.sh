#!/bin/bash -eu

# 1) Ollama (см. https://ollama.com для актуальной команды)
curl -fsSL https://ollama.com/install.sh | sh

# Затем скачай модель (пример):
ollama pull qwen2.5-coder

# 2) Python env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3) Python deps
# если хочешь faiss-gpu — ставь через conda for CUDA compatibility; для быстрого старта можно faiss-cpu
pip install sentence-transformers transformers torch tqdm python-magic regex requests

# Tree-sitter (опционально, рекомендуется)
pip install tree_sitter tree_sitter_languages

# Если планируешь faiss-gpu, лучше ставить через conda:
# conda install -c pytorch faiss-gpu cudatoolkit=12.1
# либо pip install faiss-cpu  (быстрый старт)
pip install faiss-cpu

# Example:

# убедиться, что Ollama запущен (обычно сам запускает сервер)
ollama ls

#python index_qwen_rag.py --src ./my_project.zip --out ./index_data --embed_model intfloat/multilingual-e5-base

#HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python ask_qwen_rag.py --index_dir ./index_data --embed_model intfloat/multilingual-e5-base --top_k 8 --model qwen2.5-coder
