#!/usr/bin/env bash
set -euo pipefail

# Reproduce the behavioral benchmark table used in the report.
#
# This evaluates the base model, DIM ablation, ActSVD modified model, optional
# RCO cone ablation if its direction artifact exists, and the two random
# baselines. It then runs the post-hoc Qwen3Guard judge and regenerates
# benchmark plots.
#
# Environment overrides:
#   MODEL_PATH          Hugging Face model id or local model path.
#   MODEL_ALIAS         Run directory alias. Defaults to basename of MODEL_PATH.
#   OUTPUT_DIR          Defaults to code/results/benchmark.
#   RUN_BASE=0          Skip base-model evaluation.
#   RUN_DIM=0           Skip DIM evaluation.
#   RUN_ACTSVD=0        Skip ActSVD evaluation.
#   RUN_RCO=0           Skip RCO evaluation.
#   RUN_RANDOM=0        Skip random 1-D and 2-D baselines.
#   RUN_JUDGE=0         Skip Qwen3Guard post-hoc judging.
#   PYTHON_RUNNER       Command used to run Python.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_ALIAS="${MODEL_ALIAS:-${MODEL_PATH##*/}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/benchmark}"

DIM_DIR="$CODE_DIR/methods/dim/pipeline/runs/$MODEL_ALIAS"
ACTSVD_OUT="$CODE_DIR/methods/actsvd/out"
RCO_PT="$CODE_DIR/results/geometry_repind/rco_direction.pt"

if [[ -z "${PYTHON_RUNNER:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER="uv run python"
  else
    PYTHON_RUNNER="python"
  fi
fi

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --eval_ppl --n_ppl_samples 64
  --eval_truthfulqa --n_tqa_samples 64
  --bootstrap 1000
  --output_dir "$OUTPUT_DIR"
)

cd "$CODE_DIR"

if [[ "${RUN_BASE:-1}" == "1" ]]; then
  $PYTHON_RUNNER analysis/eval_direction_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --no_ablation \
    --method_name "Base (Llama-3.1-8B-Instruct)"
fi

if [[ "${RUN_DIM:-1}" == "1" ]]; then
  if [[ -f "$DIM_DIR/direction.pt" ]]; then
    $PYTHON_RUNNER analysis/eval_direction_benchmark.py \
      "${COMMON_ARGS[@]}" \
      --direction_path "$DIM_DIR/direction.pt" \
      --direction_metadata "$DIM_DIR/direction_metadata.json" \
      --method_name DIM-Ablated
  else
    echo "WARN: skipping DIM; missing $DIM_DIR/direction.pt"
  fi
fi

if [[ "${RUN_ACTSVD:-1}" == "1" ]]; then
  if [[ -d "$ACTSVD_OUT" ]]; then
    $PYTHON_RUNNER analysis/eval_direction_benchmark.py \
      "${COMMON_ARGS[@]}" \
      --modified_model_path "$ACTSVD_OUT" \
      --method_name ActSVD-Modified
  else
    echo "WARN: skipping ActSVD; missing $ACTSVD_OUT"
  fi
fi

if [[ "${RUN_RCO:-1}" == "1" ]]; then
  if [[ -f "$RCO_PT" ]]; then
    $PYTHON_RUNNER analysis/eval_direction_benchmark.py \
      "${COMMON_ARGS[@]}" \
      --direction_path "$RCO_PT" \
      --method_name RCO-Cone-2
  else
    echo "WARN: skipping RCO; missing $RCO_PT"
  fi
fi

if [[ "${RUN_RANDOM:-1}" == "1" ]]; then
  $PYTHON_RUNNER analysis/eval_direction_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --random_direction --random_subspace_dim 1 --seed 7 \
    --method_name Random-Direction-7-1D

  $PYTHON_RUNNER analysis/eval_direction_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --random_direction --random_subspace_dim 2 --seed 7 \
    --method_name Random-Subspace-7-2D
fi

if [[ "${RUN_JUDGE:-1}" == "1" ]]; then
  $PYTHON_RUNNER analysis/judge_completions.py \
    --benchmark_dir "$OUTPUT_DIR" \
    --judge_model_path Qwen/Qwen3Guard-Gen-4B \
    --bootstrap 1000
fi

$PYTHON_RUNNER analysis/plot_benchmarks.py

echo "Behavioral benchmark complete: $CODE_DIR/$OUTPUT_DIR"
