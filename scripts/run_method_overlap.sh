#!/usr/bin/env bash
set -euo pipefail
# Compute MSO between saved direction vectors and the ActSVD weight-delta subspace.
# Environment overrides:
#   MODEL_PATH           HF model id. Defaults to meta-llama/Llama-3.1-8B-Instruct.
#   ACTSVD_MODEL_PATH    Path to ActSVD modified model. Defaults to code/methods/actsvd/out.
#   HF_HOME              HF cache dir. Defaults to code/llm_weights.
#   PYTHON_RUNNER        Command to run Python.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
ACTSVD_MODEL_PATH="${ACTSVD_MODEL_PATH:-$CODE_DIR/methods/actsvd/out}"
HF_HOME="${HF_HOME:-$CODE_DIR/llm_weights}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

export HF_HOME TRANSFORMERS_CACHE

if [[ -z "${PYTHON_RUNNER:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER="uv run python"
  else
    PYTHON_RUNNER="python"
  fi
fi

MODEL_ID="${MODEL_PATH##*/}"

# Find the base model snapshot directory (directory containing the first safetensor file)
# Exclude .no_exist dirs which are HF cache placeholders, not real weights.
BASE_MODEL_PATH="$(find "$HF_HOME" -name "*.safetensors" -path "*$MODEL_ID*" -not -path "*/.no_exist/*" 2>/dev/null | head -1 | xargs -r dirname)"

if [[ -z "$BASE_MODEL_PATH" ]]; then
  echo "Warning: could not find base model safetensors for $MODEL_ID under $HF_HOME"
  echo "Set --base_model_path explicitly or ensure the model is cached."
  BASE_MODEL_PATH=""
fi

# Build directions array
DIM_DIR_PT="$CODE_DIR/methods/dim/pipeline/runs/$MODEL_ID/direction.pt"
RCO_PT="$CODE_DIR/results/geometry_repind/rco_direction.pt"
DIRECTIONS=()
[[ -f "$DIM_DIR_PT" ]] && DIRECTIONS+=("DIM:$DIM_DIR_PT")
[[ -f "$RCO_PT" ]] && DIRECTIONS+=("RCO:$RCO_PT")

if [[ ${#DIRECTIONS[@]} -eq 0 ]]; then
  echo "No directions found, skipping method overlap computation"
  exit 0
fi

echo "Running method overlap computation"
echo "  model: $MODEL_PATH"
echo "  actsvd model: $ACTSVD_MODEL_PATH"
if [[ -n "$BASE_MODEL_PATH" ]]; then
  echo "  base model path: $BASE_MODEL_PATH"
fi
echo "  directions: ${DIRECTIONS[*]}"

cd "$CODE_DIR"

if [[ -n "$BASE_MODEL_PATH" ]]; then
  $PYTHON_RUNNER analysis/compute_method_overlap.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --actsvd_model_path "$ACTSVD_MODEL_PATH" \
    --directions "${DIRECTIONS[@]}"
else
  $PYTHON_RUNNER analysis/compute_method_overlap.py \
    --actsvd_model_path "$ACTSVD_MODEL_PATH" \
    --directions "${DIRECTIONS[@]}"
fi

$PYTHON_RUNNER analysis/plot_method_overlap.py
