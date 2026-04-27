#!/usr/bin/env bash
set -euo pipefail

# Run the Refusal Direction/Cone Optimization code.
#
# This code logs to Weights & Biases by default in the upstream implementation.
# Set WANDB_MODE=offline if you do not want network logging.
#
# Environment overrides:
#   MODEL            Model id. Defaults to google/gemma-2b-it for a cheaper run.
#   MODE             direction, independent, orthogonal, or cone. Defaults to cone.
#   CONE_DIM         Cone dimension when MODE=cone. Defaults to 2.
#   N_SAMPLE         Cone Monte Carlo samples when MODE=cone. Defaults to 16.
#   PYTHON_RUNNER    Command used to run Python.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code"

MODEL="${MODEL:-google/gemma-2b-it}"
MODE="${MODE:-cone}"
CONE_DIM="${CONE_DIM:-2}"
N_SAMPLE="${N_SAMPLE:-16}"
HF_HOME="${HF_HOME:-$CODE_DIR/llm_weights}"
HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-$HF_HOME}"
SAVE_DIR="${SAVE_DIR:-$CODE_DIR/methods/cones-repind/results}"
DIM_DIR="${DIM_DIR:-dim}"
WANDB_PROJECT="${WANDB_PROJECT:-refusal_directions}"
WANDB_MODE="${WANDB_MODE:-offline}"

export HF_HOME HUGGINGFACE_CACHE_DIR SAVE_DIR DIM_DIR WANDB_PROJECT WANDB_MODE

MODEL_ID="${MODEL##*/}"
EXPECTED_DIM_DIR="$SAVE_DIR/$DIM_DIR/$MODEL_ID"
ACTIVE_DIM_DIR="$CODE_DIR/methods/dim/pipeline/runs/$MODEL_ID"

# Mirror DIM artifacts into the location rdo.py expects, or fail early.
if [[ ! -f "$EXPECTED_DIM_DIR/direction.pt" ]]; then
  if [[ ! -f "$ACTIVE_DIM_DIR/direction.pt" ]]; then
    echo "ERROR: DIM direction not found."
    echo "  Looked in: $ACTIVE_DIM_DIR"
    echo "  Run ./scripts/run_dim.sh first, then retry."
    exit 1
  fi
  echo "Mirroring DIM artifacts for RCO: $ACTIVE_DIM_DIR -> $EXPECTED_DIM_DIR"
  mkdir -p "$EXPECTED_DIM_DIR/generate_directions"
  cp "$ACTIVE_DIM_DIR/direction.pt"             "$EXPECTED_DIM_DIR/direction.pt"
  cp "$ACTIVE_DIM_DIR/direction_metadata.json"  "$EXPECTED_DIM_DIR/direction_metadata.json"
  cp "$ACTIVE_DIM_DIR/generate_directions/mean_diffs.pt" \
     "$EXPECTED_DIM_DIR/generate_directions/mean_diffs.pt"
fi

if [[ -z "${PYTHON_RUNNER:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER="uv run python"
  else
    PYTHON_RUNNER="python"
  fi
fi

case "$MODE" in
  direction)
    MODE_ARGS=(--train_direction)
    ;;
  independent)
    MODE_ARGS=(--train_independent_direction)
    ;;
  orthogonal)
    MODE_ARGS=(--train_orthogonal_direction)
    ;;
  cone)
    MODE_ARGS=(--train_cone --min_cone_dim "$CONE_DIM" --max_cone_dim "$CONE_DIM" --n_sample "$N_SAMPLE")
    ;;
  *)
    echo "Unknown MODE=$MODE. Expected direction, independent, orthogonal, or cone." >&2
    exit 2
    ;;
esac

echo "Running RCO/RDO"
echo "  model: $MODEL"
echo "  mode: $MODE"
echo "  save dir: $SAVE_DIR"
echo "  wandb mode: $WANDB_MODE"

cd "$CODE_DIR/methods/cones-repind"
$PYTHON_RUNNER rdo.py --model "$MODEL" "${MODE_ARGS[@]}"
