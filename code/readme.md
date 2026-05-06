# Code Directory

Use the root-level `readme.md` and `scripts/` directory for the canonical workflow. This directory contains the underlying implementations.

## Main Modules

- `methods/dim/`: DIM refusal-direction extraction, evaluation, and modified-model saving.
- `methods/actsvd/`: ActSVD low-rank safety/utility disentanglement and modified-model saving.
- `methods/cones-repind/`: RCO/RDO and representational-independence experiments.
- `analysis/`: project-owned analysis and plotting scripts.
- `results/benchmark/`: summarized benchmark outputs and plots.
- `results/method_overlap/`: MSO and cross-model cosine outputs and plots.
- `results/safety_utility_overlap/`: direct safety-vs-utility overlap outputs and plots.
- `tools/chat.py`: interactive chat helper for saved models.
- `data-exploration/`: dataset loading and EDA plots.
- `llm_weights/`: local Hugging Face model cache, ignored by git.

Archived copied/reference code lives under `archive/` and is ignored by git.

## Common Commands

From the repository root:

```bash
./scripts/run_dim.sh
./scripts/run_actsvd.sh
./scripts/run_safety_utility_overlap.sh
python scripts/inventory.py
python scripts/sync_figures.py
```

Interactive chat with a saved modified model:

```bash
cd code
uv run python tools/chat.py --model_path methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/modified_model
uv run python tools/chat.py --model_path methods/actsvd/out
```
