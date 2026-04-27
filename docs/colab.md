# Colab Runbook

Use this when local VRAM is not enough. The notebook version is `notebooks/colab_end_to_end.ipynb`; the commands below are the same workflow as plain Colab cells.

## What To Import Into Colab

Import/open only one file manually:

- `notebooks/colab_end_to_end.ipynb`

The notebook clones the repository itself. If you do not use the notebook, create a fresh Colab notebook and run the command cells below.

You need these external credentials:

- Hugging Face token with access to `meta-llama/Llama-3.1-8B-Instruct`.
- Optional Together API key if you want LlamaGuard-based jailbreak evaluation. Without it, the DIM pipeline uses substring matching.
- Optional Weights & Biases account for cones-repind/RCO logging. Set `WANDB_MODE=offline` to avoid online logging.

## Runtime

Use `Runtime -> Change runtime type -> GPU`.

- A100 or similar 24GB+ VRAM: use Llama-3.1-8B-Instruct.
- T4/free tier: run only smoke tests, preferably DIM on `google/gemma-2b-it`.

## Commands

Clone and authenticate:

```bash
git clone https://github.com/evan203/nlp-project-proposal.git
cd nlp-project-proposal
pip install -U huggingface_hub
huggingface-cli login
```

Install dependencies:

```bash
pip install -U torch torchvision transformers accelerate datasets sentencepiece protobuf tqdm jaxtyping matplotlib seaborn zstandard litellm nnsight==0.3.7 vllm
```

Run DIM:

```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PYTHON_RUNNER=python ./scripts/run_dim.sh
```

Run ActSVD:

```bash
MODEL_ALIAS=Llama-3.1-8B-Instruct PYTHON_RUNNER=python NSAMPLES=128 ./scripts/run_actsvd.sh
```

Run direct safety-vs-utility overlap:

```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PYTHON_RUNNER=python N_UTILITY_SAMPLES=128 ./scripts/run_safety_utility_overlap.sh
```

Run the lightweight Geometry/RepInd profile analysis:

```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PYTHON_RUNNER=python N_PROMPTS=32 ./scripts/run_geometry_repind.sh
```

Run full cones-repind/RCO training as the heavier extension:

```bash
WANDB_MODE=offline MODEL=google/gemma-2b-it MODE=cone CONE_DIM=2 PYTHON_RUNNER=python ./scripts/run_rco.sh
```

Evaluate the latest trained RCO/RepInd vector with the RepInd profile analysis:

```bash
python scripts/build_rco_directions_json.py --latest
DIRECTIONS_JSON=code/results/geometry_repind/directions.json PYTHON_RUNNER=python ./scripts/run_geometry_repind.sh
```

Regenerate/sync plots:

```bash
python code/analysis/plot_benchmarks.py
python code/analysis/plot_method_overlap.py
python code/analysis/plot_safety_utility_overlap.py
python code/analysis/plot_geometry_repind.py
python scripts/sync_figures.py
python scripts/inventory.py
```

## Smoke-Test Commands

Use these before committing to a long run:

```bash
MODEL_PATH=google/gemma-2b-it PYTHON_RUNNER=python ./scripts/run_dim.sh
MODEL_ALIAS=Llama-3.1-8B-Instruct PYTHON_RUNNER=python NSAMPLES=32 ./scripts/run_actsvd.sh
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PYTHON_RUNNER=python N_UTILITY_SAMPLES=32 ./scripts/run_safety_utility_overlap.sh
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PYTHON_RUNNER=python N_PROMPTS=4 ./scripts/run_geometry_repind.sh
WANDB_MODE=offline MODEL=google/gemma-2b-it MODE=direction PYTHON_RUNNER=python ./scripts/run_rco.sh
```

## Outputs To Keep

Keep small result files and figures:

- `code/methods/dim/pipeline/runs/*/direction.pt`
- `code/methods/dim/pipeline/runs/*/direction_metadata.json`
- `code/methods/dim/pipeline/runs/*/completions/*.json`
- `code/results/benchmark/*`
- `code/results/method_overlap/*`
- `code/results/safety_utility_overlap/*`
- `code/results/geometry_repind/*`
- `docs/figures/*`

Do not commit model checkpoints or caches:

- `code/llm_weights/`
- `code/methods/actsvd/out/`
- `code/methods/dim/pipeline/runs/*/modified_model/`
