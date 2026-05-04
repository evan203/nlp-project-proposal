#!/usr/bin/env bash
set -euo pipefail

# Run the Difference-in-Means refusal-direction pipeline.
#
# Environment overrides:
#   MODEL_PATH       Hugging Face model id or local model path.
#   PYTHON_RUNNER    Command used to run Python. Defaults to "uv run python" when
#                    uv is installed, otherwise "python".
#   HF_HOME          Hugging Face cache directory. Defaults to code/llm_weights.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
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

echo "Running DIM pipeline"
echo "  model: $MODEL_PATH"
echo "  HF cache: $HF_HOME"

cd "$CODE_DIR/methods/dim"
$PYTHON_RUNNER -m pipeline.run_self_consistency --model_path "$MODEL_PATH"
