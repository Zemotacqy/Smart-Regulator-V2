#!/bin/bash
# scripts/convert_to_gguf.sh
# Merges LoRA adapters with base model, converts to GGUF, and quantizes to Q4_K_M.

set -e

# Default values
BASE=""
ADAPTERS=""
OUTPUT=""

# Parse long options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base) BASE="$2"; shift ;;
        --adapters) ADAPTERS="$2"; shift ;;
        --output) OUTPUT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate inputs
if [ -z "$BASE" ] || [ -z "$ADAPTERS" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 --base <base_model> --adapters <adapters_dir> --output <output_gguf_path>"
    exit 1
fi

# Map friendly base name to the actual full-precision Hugging Face repo
if [ "$BASE" = "saullm-7b-instruct" ] || [ "$BASE" = "Equall/Saul-7B-Instruct-v1" ]; then
    BASE="Equall/Saul-7B-Instruct-v1"
fi

echo "=== GGUF Conversion Pipeline ==="
echo "Base Model: $BASE"
echo "Adapters:   $ADAPTERS"
echo "Output:     $OUTPUT"
echo "================================"

# Create output parent directory if it does not exist
mkdir -p "$(dirname "$OUTPUT")"

# Temporary directories for fusion
FUSED_DIR="models/fused_temp"
TEMP_GGUF="models/fused_temp.gguf"

# Clean up any leftover files from previous failed runs
rm -rf "$FUSED_DIR" "$TEMP_GGUF" convert_hf_to_gguf.py

# 1. Fuse adapters with the full-precision base model using mlx_lm
echo "Fusing LoRA adapters with base model..."
uv run mlx_lm fuse \
  --model "$BASE" \
  --adapter-path "$ADAPTERS" \
  --save-path "$FUSED_DIR"


# 2. Install gguf and sentencepiece dependencies
echo "Ensuring python conversion dependencies are installed..."
uv pip install gguf sentencepiece

# 3. Clone llama.cpp repository for conversion scripts
echo "Cloning llama.cpp repository..."
rm -rf llama.cpp_temp
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git llama.cpp_temp

# 4. Convert the fused model to a full-precision GGUF file
echo "Converting fused model to GGUF format..."
uv run python llama.cpp_temp/convert_hf_to_gguf.py "$FUSED_DIR" --outfile "$TEMP_GGUF"

# 5. Locate llama-quantize binary
echo "Locating llama-quantize tool..."
QUANTIZE_BIN="llama-quantize"
if ! command -v llama-quantize &> /dev/null; then
    if [ -f "/opt/homebrew/bin/llama-quantize" ]; then
        QUANTIZE_BIN="/opt/homebrew/bin/llama-quantize"
    else
        echo "Error: llama-quantize not found in PATH or /opt/homebrew/bin. Please install llama.cpp."
        rm -rf llama.cpp_temp
        exit 1
    fi
fi

# 6. Quantize the GGUF file to Q4_K_M
echo "Quantizing GGUF model to Q4_K_M..."
"$QUANTIZE_BIN" "$TEMP_GGUF" "$OUTPUT" Q4_K_M

# 7. Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$FUSED_DIR" "$TEMP_GGUF" llama.cpp_temp


echo "GGUF conversion and quantization completed successfully!"
echo "Saved quantized model to: $OUTPUT"
