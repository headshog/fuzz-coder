#!/bin/bash -eu

# 1) Ollama installation
#curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended model (fits in 24GB VRAM with context for code)
#ollama pull qwen2.5-coder:7b

# 2) Python env
#python3 -m venv .venv
#source .venv/bin/activate
#pip install --upgrade pip

# 3) Core dependencies
pip install sentence-transformers transformers torch tqdm python-magic regex requests pytest

# Tree-sitter for accurate AST parsing (highly recommended)
pip install tree_sitter tree_sitter_cpp

# FAISS for semantic search
# For GPU support (recommended if you have CUDA):
# conda install -c pytorch faiss-gpu cudatoolkit=11.8
# Or use CPU version:
pip install faiss-cpu

# Verify Ollama is running
ollama ls

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Usage:"
echo "1. Index your code:"
echo "   python index_hybrid_code.py --src /path/to/your/code.zip --out ./index_data"
echo ""
echo "2. Ask questions:"
echo "   python ask_hybrid_code.py --index_dir ./index_data --model qwen2.5-coder:7b --verbose"
echo ""
echo "For best results on questions like 'list functions that read from stdin/file':"
echo "  - The indexer pre-detects input types (stdin, file, API)"
echo "  - Query planner routes to appropriate indices"
echo "  - Reranker filters by parameter presence and input type"
echo "  - Enhanced prompts instruct the LLM to be precise"
echo ""
