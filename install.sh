#!/bin/bash -eu

# 1) Ollama installation
#curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended model (fits in 24GB VRAM with context for code)
#ollama pull qwen3-coder:30b

# 2) Python env
#python3 -m venv .venv
#source .venv/bin/activate
#pip install --upgrade pip

# 3) Core dependencies
pip install sentence-transformers transformers torch tqdm python-magic regex requests pytest
# Optional browser chat UI
pip install gradio

# Tree-sitter for accurate AST parsing (highly recommended)
pip install tree_sitter tree_sitter_cpp tree_sitter_java

# FAISS for semantic search
# For GPU support (recommended if you have CUDA):
# conda install -c pytorch faiss-gpu cudatoolkit=11.8
# Or use CPU version:
pip install faiss-cpu

# Verify Ollama is running
ollama ls
