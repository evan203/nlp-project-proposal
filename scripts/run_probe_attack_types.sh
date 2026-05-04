#!/usr/bin/env bash
# Run the prompt-attack-type direction probe and generate all figures.
#
# Data sources:
#   Baseline:     HarmBench standard behaviors (local JSON, no download needed)
#   Adversarial:  WildJailbreak 'train' config, streamed — adversarial_harmful
#                 and adversarial_benign rows (requires HF_TOKEN)
#
# Env vars (all optional):
#   MODEL_PATH    — HF model path (default: meta-llama/Llama-3.1-8B-Instruct)
#   MODEL_ALIAS   — short name matching the DIM run directory
#   N_DIRECT      — HarmBench direct-request samples (default: 25)
#   N_PER_TYPE    — WildJailbreak samples per data_type group (default: 25)
#   BATCH_SIZE    — forward-pass batch size (default: 8)
#   MAX_NEW_TOKENS — generation length per prompt (default: 200)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/../code"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_ALIAS="${MODEL_ALIAS:-Llama-3.1-8B-Instruct}"
N_DIRECT="${N_DIRECT:-25}"
N_PER_TYPE="${N_PER_TYPE:-25}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
LAYER_SWEEP="${LAYER_SWEEP:-1}"        # 1 = run per-layer projection sweep
ABLATION_CROSS="${ABLATION_CROSS:-1}"  # 1 = run DIM-ablated cross-test
BOOTSTRAP="${BOOTSTRAP:-1000}"          # 0 = disable

DIM_RUN="$CODE_DIR/methods/dim/pipeline/runs/$MODEL_ALIAS"

echo "=== Probe: adversarial wrapping vs. refusal direction ==="
echo "  model:          $MODEL_PATH"
echo "  baseline:       $N_DIRECT HarmBench direct-request prompts (local)"
echo "  adversarial:    $N_PER_TYPE WildJailbreak adversarial_harmful prompts (streamed)"
echo "  control:        $N_PER_TYPE WildJailbreak adversarial_benign prompts (streamed)"
echo "  layer sweep:    $LAYER_SWEEP    ablation cross-test: $ABLATION_CROSS"
echo "  bootstrap:      $BOOTSTRAP samples"
echo ""

EXTRA_FLAGS=()
[[ "$LAYER_SWEEP"    == "1" ]] && EXTRA_FLAGS+=(--layer_sweep)
[[ "$ABLATION_CROSS" == "1" ]] && EXTRA_FLAGS+=(--ablation_cross_test)

python "$CODE_DIR/analysis/probe_attack_types.py" \
    --model_path         "$MODEL_PATH" \
    --direction_path     "$DIM_RUN/direction.pt" \
    --direction_metadata "$DIM_RUN/direction_metadata.json" \
    --n_direct           "$N_DIRECT" \
    --n_per_type         "$N_PER_TYPE" \
    --batch_size         "$BATCH_SIZE" \
    --max_new_tokens     "$MAX_NEW_TOKENS" \
    --bootstrap          "$BOOTSTRAP" \
    "${EXTRA_FLAGS[@]}" \
    --output_dir         "$CODE_DIR/results/probe_attack_types"

echo ""
echo "=== Plotting ==="
python "$CODE_DIR/analysis/plot_attack_types.py" \
    --results_dir "$CODE_DIR/results/probe_attack_types" \
    --output_dir  "$CODE_DIR/results/probe_attack_types"
