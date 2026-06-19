#!/usr/bin/env bash
# =============================================================================
# compile_modelfiles.sh
# Compiles all IFSCA Regulatory Assistant Modelfiles into named Ollama models.
#
# Usage:
#   bash scripts/compile_modelfiles.sh              # Compile all models
#   bash scripts/compile_modelfiles.sh --check      # Check which are already compiled
#   bash scripts/compile_modelfiles.sh --skip-saul  # Skip the SaulLM fine-tuned model
#
# Requirements:
#   - Ollama must be running (check: ollama list)
#   - llama3.2:3b must be pulled: ollama pull llama3.2:3b
#   - For SaulLM: fine-tuned GGUF must exist at models/ifsca-saullm-7b-ft.Q4_K_M.gguf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELFILES_DIR="${REPO_ROOT}/Modelfiles"

# ANSI colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No colour

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Model definitions: parallel arrays of (model_name, modelfile_path)
# ---------------------------------------------------------------------------
MODEL_NAMES=("ifsca-classifier-3b" "ifsca-boundary-3b" "ifsca-extractor-3b" "ifsca-expander-3b" "ifsca-reranker-3b")
MODEL_FILES=(
    "${MODELFILES_DIR}/Modelfile.classifier"
    "${MODELFILES_DIR}/Modelfile.boundary"
    "${MODELFILES_DIR}/Modelfile.extractor"
    "${MODELFILES_DIR}/Modelfile.expander"
    "${MODELFILES_DIR}/Modelfile.reranker"
)

SAUL_MODEL_NAME="ifsca-saullm-7b-ft"
SAUL_MODELFILE="${MODELFILES_DIR}/Modelfile.saullm"
SAUL_GGUF_PATH="${REPO_ROOT}/models/ifsca-saullm-7b-ft.Q4_K_M.gguf"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
CHECK_ONLY=false
SKIP_SAUL=false

for arg in "$@"; do
    case $arg in
        --check)      CHECK_ONLY=true ;;
        --skip-saul)  SKIP_SAUL=true ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# //'
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " IFSCA Smart Regulator — Ollama Modelfile Compiler"
echo "============================================================"
echo ""

log_info "Repo root: ${REPO_ROOT}"
log_info "Modelfiles dir: ${MODELFILES_DIR}"
echo ""

# Check ollama is running
if ! ollama list > /dev/null 2>&1; then
    log_error "Ollama is not running or not in PATH."
    log_error "Start Ollama and retry: 'ollama serve'"
    exit 1
fi
log_ok "Ollama is running."

# Check llama3.2:3b is pulled (base model for the 4 instruction models)
if ! ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
    log_warn "llama3.2:3b not found in Ollama. Attempting to pull..."
    if ! ollama pull llama3.2:3b; then
        log_error "Failed to pull llama3.2:3b. Ensure network access and retry."
        exit 1
    fi
fi
log_ok "llama3.2:3b base model available."
echo ""

# ---------------------------------------------------------------------------
# Check-only mode: list current compilation status
# ---------------------------------------------------------------------------
if $CHECK_ONLY; then
    log_info "=== Model Compilation Status ==="
    for i in "${!MODEL_NAMES[@]}"; do
        name="${MODEL_NAMES[$i]}"
        if ollama list 2>/dev/null | grep -q "${name}"; then
            log_ok  "${name} — compiled"
        else
            log_warn "${name} — NOT compiled"
        fi
    done
    if ollama list 2>/dev/null | grep -q "${SAUL_MODEL_NAME}"; then
        log_ok  "${SAUL_MODEL_NAME} — compiled"
    else
        log_warn "${SAUL_MODEL_NAME} — NOT compiled"
    fi
    echo ""
    exit 0
fi

# ---------------------------------------------------------------------------
# Compile the 4 instruction models (classifier, boundary, extractor, expander)
# ---------------------------------------------------------------------------
FAILED_MODELS=()

for i in "${!MODEL_NAMES[@]}"; do
    name="${MODEL_NAMES[$i]}"
    modelfile="${MODEL_FILES[$i]}"

    echo "------------------------------------------------------------"
    log_info "Compiling: ${name}"
    log_info "Modelfile: ${modelfile}"

    if [ ! -f "${modelfile}" ]; then
        log_error "Modelfile not found: ${modelfile}"
        FAILED_MODELS+=("${name}")
        continue
    fi

    if ollama list 2>/dev/null | grep -q "${name}"; then
        log_warn "${name} already exists in Ollama — removing stale version."
        ollama rm "${name}" 2>/dev/null || true
    fi

    if ollama create "${name}" -f "${modelfile}"; then
        log_ok "${name} compiled successfully."
    else
        log_error "Failed to compile ${name}."
        FAILED_MODELS+=("${name}")
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Compile the SaulLM fine-tuned model (optional — requires GGUF)
# ---------------------------------------------------------------------------
if $SKIP_SAUL; then
    log_warn "Skipping SaulLM fine-tuned model (--skip-saul flag set)."
else
    echo "------------------------------------------------------------"
    log_info "Compiling: ${SAUL_MODEL_NAME}"
    log_info "Modelfile: ${SAUL_MODELFILE}"
    log_info "GGUF path: ${SAUL_GGUF_PATH}"

    if [ ! -f "${SAUL_GGUF_PATH}" ]; then
        log_warn "SaulLM GGUF not found at: ${SAUL_GGUF_PATH}"
        log_warn "Run the fine-tuning pipeline first (scripts/fine_tune_mlx.py),"
        log_warn "then convert and quantise the adapter before running this script."
        log_warn "Skipping SaulLM compilation."
    elif [ ! -f "${SAUL_MODELFILE}" ]; then
        log_error "Modelfile.saullm not found: ${SAUL_MODELFILE}"
        FAILED_MODELS+=("${SAUL_MODEL_NAME}")
    else
        if ollama list 2>/dev/null | grep -q "${SAUL_MODEL_NAME}"; then
            log_warn "${SAUL_MODEL_NAME} already exists — removing stale version."
            ollama rm "${SAUL_MODEL_NAME}" 2>/dev/null || true
        fi
        if ollama create "${SAUL_MODEL_NAME}" -f "${SAUL_MODELFILE}"; then
            log_ok "${SAUL_MODEL_NAME} compiled successfully."
        else
            log_error "Failed to compile ${SAUL_MODEL_NAME}."
            FAILED_MODELS+=("${SAUL_MODEL_NAME}")
        fi
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Compilation Summary"
echo "============================================================"
ollama list 2>/dev/null | grep -E "ifsca-|llama3.2:3b" || true
echo ""

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    log_error "The following models failed to compile:"
    for m in "${FAILED_MODELS[@]}"; do
        echo "  - ${m}"
    done
    echo ""
    exit 1
else
    log_ok "All models compiled successfully."
    echo ""
fi
