#!/usr/bin/env bash
set -euo pipefail

# Run the ActSVD low-rank safety/utility disentanglement pipeline.
#
# Environment overrides:
#   MODEL_ALIAS      Internal alias supported by code/methods/actsvd/main_low_rank_diff.py.
#                    Defaults to Llama-3.1-8B-Instruct.
#   RANK_POS         Utility rank cutoff r^u. Defaults to 3000.
#   RANK_NEG         Safety rank cutoff r^s. Defaults to 4000.
#   NSAMPLES         Calibration samples. Defaults to 128.
#   SAVE_DIR         Output directory. Defaults to code/methods/actsvd/out.
#   EVAL_PPL         Set to 1 to run wikitext perplexity eval.
#   EVAL_ATTACK      Set to 1 to run attack eval. Requires vLLM and more VRAM.
#   PYTHON_RUNNER    Command used to run Python.
#   HF_HOME          Hugging Face cache directory. Defaults to code/llm_weights.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code"

MODEL_ALIAS="${MODEL_ALIAS:-Llama-3.1-8B-Instruct}"
RANK_POS="${RANK_POS:-3000}"
RANK_NEG="${RANK_NEG:-4000}"
NSAMPLES="${NSAMPLES:-128}"
SAVE_DIR="${SAVE_DIR:-$CODE_DIR/methods/actsvd/out}"
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
  main_low_rank_diff.py
  --model "$MODEL_ALIAS"
  --rank_pos "$RANK_POS"
  --rank_neg "$RANK_NEG"
  --nsamples "$NSAMPLES"
  --save "$SAVE_DIR"
)

if [[ "${EVAL_PPL:-0}" == "1" ]]; then
  ARGS+=(--eval_ppl)
fi

if [[ "${EVAL_ATTACK:-0}" == "1" ]]; then
  ARGS+=(--eval_attack)
fi

echo "Running ActSVD pipeline"
echo "  model alias: $MODEL_ALIAS"
echo "  ranks: utility=$RANK_POS safety=$RANK_NEG"
echo "  samples: $NSAMPLES"
echo "  save dir: $SAVE_DIR"
echo "  HF cache: $HF_HOME"

cd "$CODE_DIR/methods/actsvd"
$PYTHON_RUNNER "${ARGS[@]}"
