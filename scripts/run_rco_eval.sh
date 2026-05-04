#!/usr/bin/env bash
set -euo pipefail

# Run RCO training on Llama-3.1-8B-Instruct, then evaluate the trained direction
# on JailbreakBench and harmless compliance and add results to benchmark_results.json.
#
# This is the full pipeline for the third method (RCO/RDO from the Geometry paper):
#   1. Train the direction with rdo.py (offline wandb, local artifacts)
#   2. Extract the direction to a standalone .pt file
#   3. Evaluate it with eval_direction_benchmark.py
#   4. (Optional) Run RepInd comparison against DIM
#   5. (Optional) Save a weight-edited modified model
#
# Environment overrides:
#   MODEL            Model id. Defaults to meta-llama/Llama-3.1-8B-Instruct.
#   MODE             direction, independent, orthogonal, or cone. Defaults to direction.
#   CONE_DIM         Cone dimension (only used when MODE=cone). Defaults to 1.
#   PYTHON_RUNNER    Command used to run Python.
#   HF_HOME          Hugging Face cache directory.
#   SKIP_TRAIN       Set to 1 to skip training (use existing local vectors).
#   SKIP_EVAL        Set to 1 to skip benchmark evaluation.
#   SKIP_REPIND      Set to 1 to skip RepInd comparison after RCO.
#   SAVE_MODIFIED    Set to 1 to also save a weight-edited modified model.
#   METHOD_NAME      Name used in benchmark_results.json. Defaults to RDO.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODE="${MODE:-direction}"
CONE_DIM="${CONE_DIM:-1}"
METHOD_NAME="${METHOD_NAME:-RDO}"
HF_HOME="${HF_HOME:-$CODE_DIR/llm_weights}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$CODE_DIR/methods/cones-repind/results}"
DIM_DIR_NAME="${DIM_DIR:-dim}"

export HF_HOME TRANSFORMERS_CACHE WANDB_MODE SAVE_DIR

if [[ -z "${PYTHON_RUNNER:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER="uv run python"
  else
    PYTHON_RUNNER="python"
  fi
fi

MODEL_ID="${MODEL##*/}"
RCO_ROOT="$SAVE_DIR/rdo"
DIRECTIONS_JSON="$CODE_DIR/results/geometry_repind/directions_rco.json"
DIRECTION_PT="$CODE_DIR/results/geometry_repind/rco_direction.pt"

echo "=== RCO Evaluation Pipeline ==="
echo "  model:       $MODEL"
echo "  mode:        $MODE"
echo "  method name: $METHOD_NAME"
echo "  wandb mode:  $WANDB_MODE"
echo ""

# ---- Step 1: Train --------------------------------------------------------
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "[1/4] Training RCO/RDO direction ..."
  MODEL="$MODEL" MODE="$MODE" CONE_DIM="$CONE_DIM" WANDB_MODE="$WANDB_MODE" \
    "$ROOT_DIR/scripts/run_rco.sh"
else
  echo "[1/4] Skipping training (SKIP_TRAIN=1)"
fi

# ---- Step 2: Extract direction --------------------------------------------
echo "[2/4] Extracting direction from local RCO artifacts ..."

# --standalone_output copies the latest vector to DIRECTION_PT as a plain .pt
# (no --output here; we skip the JSON until Step 4 which includes DIM)
$PYTHON_RUNNER "$ROOT_DIR/scripts/build_rco_directions_json.py" \
  --latest \
  --rco_root "$RCO_ROOT" \
  --standalone_output "$DIRECTION_PT" \
  --no_dim

if [[ ! -f "$DIRECTION_PT" ]]; then
  echo "ERROR: No trained direction found under $RCO_ROOT"
  echo "  Expected: $RCO_ROOT/*/local_vectors/*/*/lowest_loss_vector.pt"
  echo "  Run training first: SKIP_TRAIN=0 $0"
  exit 1
fi
echo "  Direction saved to: $DIRECTION_PT"

# ---- Step 3: Benchmark evaluation -----------------------------------------
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  echo "[3/4] Evaluating direction on JailbreakBench and harmless compliance ..."
  cd "$CODE_DIR"
  # NOTE: --llm_judge intentionally NOT passed here. The post-hoc
  # analysis/judge_completions.py grades all methods with the unmodified
  # base model in a single pass, avoiding the cross-method confound where
  # a method's own intervention biases its self-judgment.
  $PYTHON_RUNNER analysis/eval_direction_benchmark.py \
    --model_path "$MODEL" \
    --direction_path "$DIRECTION_PT" \
    --method_name "$METHOD_NAME" \
    --eval_ppl --n_ppl_samples 64 \
    --eval_truthfulqa --n_tqa_samples 64 \
    --bootstrap 1000 \
    --output_dir results/benchmark

  echo ""
  echo "Regenerating benchmark plots ..."
  $PYTHON_RUNNER analysis/plot_benchmarks.py
else
  echo "[3/4] Skipping benchmark evaluation (SKIP_EVAL=1)"
fi

# ---- Step 4: RepInd comparison --------------------------------------------
if [[ "${SKIP_REPIND:-0}" != "1" ]]; then
  echo "[4/4] Running RepInd comparison (DIM + RCO) ..."
  # Build directions JSON that includes both DIM and RCO
  $PYTHON_RUNNER "$ROOT_DIR/scripts/build_rco_directions_json.py" \
    --latest \
    --rco_root "$RCO_ROOT" \
    --output "$CODE_DIR/results/geometry_repind/directions_rco.json"

  DIRECTIONS_JSON="$CODE_DIR/results/geometry_repind/directions_rco.json" \
    MODEL_PATH="$MODEL" \
    OUTPUT_DIR="$CODE_DIR/results/geometry_repind_rco" \
    PYTHON_RUNNER="$PYTHON_RUNNER" \
    "$ROOT_DIR/scripts/run_geometry_repind.sh"
else
  echo "[4/4] Skipping RepInd comparison (SKIP_REPIND=1)"
fi

# ---- Optional: Save weight-edited model -----------------------------------
if [[ "${SAVE_MODIFIED:-0}" == "1" ]]; then
  echo "[+] Saving weight-edited model ..."
  MODIFIED_DIR="$CODE_DIR/methods/cones-repind/results/modified_model/${MODEL_ID}-RCO"
  cd "$CODE_DIR/methods/cones-repind"
  $PYTHON_RUNNER save_modified_model.py \
    --model_path "$MODEL" \
    --direction "$DIRECTION_PT" \
    --output_dir "$MODIFIED_DIR"
  echo "  Saved to: $MODIFIED_DIR"
fi

echo ""
echo "=== RCO evaluation complete ==="
echo "  benchmark: $CODE_DIR/results/benchmark/benchmark_results.json"
echo "  direction: $DIRECTION_PT"
echo "  repind:    $CODE_DIR/results/geometry_repind_rco/"
