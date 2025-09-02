#!/bin/bash

# Persist model caches to avoid re-downloading on subsequent runs
# Use repo-local caches so everything stays self-contained
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export WHISPER_CACHE_DIR="$PWD/.cache/whisper"

# Create cache directories if missing
mkdir -p "$HF_HOME/transformers" "$TORCH_HOME" "$WHISPER_CACHE_DIR"

# Optional: speed up tokenizers (harmless if already set)
export TOKENIZERS_PARALLELISM=true

python src/train.py
