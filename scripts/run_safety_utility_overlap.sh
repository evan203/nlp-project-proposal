#!/usr/bin/env bash
set -euo pipefail

# Compute direct safety-vs-utility activation overlap and create report figures.
#
# Environment overrides:
#   MODEL_PATH            Hugging Face model id or local model path.
#   N_UTILITY_SAMPLES     Utility activation samples. Defaults to 128.
#   N_SAFETY_SAMPLES      Safety mean-diff samples if recomputing. Defaults to 128.
#   UTILITY_RANKS         Comma-separated PCA ranks. Defaults to 1,2,4,8,16,32.
#   PRIMARY_RANK          Rank used for the per-layer bar chart. Defaults to 8.
#   OUTPUT_DIR            Defaults to code/results/safety_utility_overlap.
#   PYTHON_RUNNER         Command used to run Python.
#   HF_HOME               Hugging Face cache directory. Defaults to code/llm_weights.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
N_UTILITY_SAMPLES="${N_UTILITY_SAMPLES:-128}"
N_SAFETY_SAMPLES="${N_SAFETY_SAMPLES:-128}"
UTILITY_RANKS="${UTILITY_RANKS:-1,2,4,8,16,32}"
PRIMARY_RANK="${PRIMARY_RANK:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/results/safety_utility_overlap}"
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
DIM_DIR_PT="$CODE_DIR/methods/dim/pipeline/runs/$MODEL_ID/direction.pt"
RCO_PT="$CODE_DIR/results/geometry_repind/rco_direction.pt"
ACTSVD_OUT="${ACTSVD_OUT:-$CODE_DIR/results/actsvd/out}"
ACTSVD_DIRECTION_PT="$CODE_DIR/results/actsvd/activation_direction.pt"
ACTSVD_N_SAMPLES="${ACTSVD_N_SAMPLES:-64}"

EXTRA_DIRS=()
[[ -f "$DIM_DIR_PT" ]] && EXTRA_DIRS+=("DIM:$DIM_DIR_PT")
[[ -f "$RCO_PT" ]] && EXTRA_DIRS+=("RCO:$RCO_PT")

# Extract ActSVD activation direction if the modified model exists but direction not yet computed
if [[ -d "$ACTSVD_OUT" ]]; then
  if [[ ! -f "$ACTSVD_DIRECTION_PT" ]]; then
    echo "Extracting ActSVD activation direction (n=$ACTSVD_N_SAMPLES) ..."
    cd "$ROOT_DIR"
    $PYTHON_RUNNER scripts/extract_actsvd_direction.py \
      --base_model_path "$MODEL_PATH" \
      --modified_model_path "$ACTSVD_OUT" \
      --output_path "$ACTSVD_DIRECTION_PT" \
      --n_samples "$ACTSVD_N_SAMPLES"
    cd "$CODE_DIR"
  fi
  [[ -f "$ACTSVD_DIRECTION_PT" ]] && EXTRA_DIRS+=("ActSVD:$ACTSVD_DIRECTION_PT")
fi

echo "Running safety-vs-utility overlap analysis"
echo "  model: $MODEL_PATH"
echo "  utility samples: $N_UTILITY_SAMPLES"
echo "  output: $OUTPUT_DIR"
if [[ ${#EXTRA_DIRS[@]} -gt 0 ]]; then
  echo "  extra directions: ${EXTRA_DIRS[*]}"
fi

cd "$CODE_DIR"
$PYTHON_RUNNER analysis/safety_utility_overlap.py \
  --model_path "$MODEL_PATH" \
  --n_utility_samples "$N_UTILITY_SAMPLES" \
  --n_safety_samples "$N_SAFETY_SAMPLES" \
  --utility_ranks "$UTILITY_RANKS" \
  --primary_rank "$PRIMARY_RANK" \
  --output_dir "$OUTPUT_DIR" \
  ${EXTRA_DIRS[@]:+--extra_directions "${EXTRA_DIRS[@]}"}

$PYTHON_RUNNER analysis/plot_safety_utility_overlap.py \
  --results_dir "$OUTPUT_DIR" \
  --primary_rank "$PRIMARY_RANK"
