#!/usr/bin/env bash
set -euo pipefail

# Run a local Geometry-paper-style RepInd profile analysis.
#
# By default this derives a small cone basis from high-norm DIM candidate
# directions, so it can run without W&B artifacts. To evaluate trained RDO/RCO
# directions, pass DIRECTIONS_JSON=/path/to/directions.json.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
N_PROMPTS="${N_PROMPTS:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DERIVED_CONE_DIM="${DERIVED_CONE_DIM:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/results/geometry_repind}"
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

ARGS=(
  analysis/geometry_repind.py
  --model_path "$MODEL_PATH"
  --n_prompts "$N_PROMPTS"
  --batch_size "$BATCH_SIZE"
  --derived_cone_dim "$DERIVED_CONE_DIM"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "${DIRECTIONS_JSON:-}" ]]; then
  ARGS+=(--directions_json "$DIRECTIONS_JSON")
fi

echo "Running Geometry/RepInd profile analysis"
echo "  model: $MODEL_PATH"
echo "  prompts: $N_PROMPTS"
echo "  output: $OUTPUT_DIR"

cd "$CODE_DIR"
$PYTHON_RUNNER "${ARGS[@]}"
$PYTHON_RUNNER analysis/plot_geometry_repind.py --results_dir "$OUTPUT_DIR"
