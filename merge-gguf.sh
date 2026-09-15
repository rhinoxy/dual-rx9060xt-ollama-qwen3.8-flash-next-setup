#!/usr/bin/env bash
# ==============================================================================
# merge-gguf.sh
# Merges multi-part GGUF files (e.g. Qwen 3.8 Flash Next split into 4 parts)
# into a single GGUF file suitable for Ollama importing.
# ==============================================================================

set -euo pipefail

INPUT_PREFIX="${1:-Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
OUTPUT_FILE="${2:-Qwen3.8-Flash-Next-merged.gguf}"

echo ">>> Merging GGUF parts into ${OUTPUT_FILE}..."

if command -v llama-gguf-split &>/dev/null; then
    echo "Found llama-gguf-split, using it to merge cleanly..."
    llama-gguf-split --merge "${INPUT_PREFIX}" "${OUTPUT_FILE}"
else
    echo "llama-gguf-split not found in PATH."
    echo "Attempting direct binary concatenation (works for standard GGUF shards)..."
    cat Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf > "${OUTPUT_FILE}"
fi

echo ">>> Done! Merged file created at ${OUTPUT_FILE}"
ls -lh "${OUTPUT_FILE}"
