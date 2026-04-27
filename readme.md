# Comparing Safety-Removal Subspaces in Aligned LLMs

Course project for COMP SCI 639 (Deep Learning for NLP). We compare three linear methods for removing safety behavior from Llama-3.1-8B-Instruct and ask whether they identify the same underlying refusal mechanism.

## Research Questions

1. **Cross-method agreement**: Do DIM, ActSVD, and RDO converge on the same safety-relevant subspace, or do they each capture a distinct mechanism?
2. **Safety–utility entanglement**: How much do DIM safety directions overlap with harmless-instruction utility activation subspaces?
3. **Causal independence**: Are the refusal directions identified by each method representationally independent under ablation (RepInd)?

## Methods

| Method | Paper | What it identifies |
|--------|-------|-------------------|
| **DIM** | Arditi et al. 2024 | Single difference-in-means direction in residual stream |
| **ActSVD** | Wei et al. 2024 | Low-rank safety/utility projection matrices via SVD |
| **RDO** | Wollschläger et al. 2025 | Gradient-optimized refusal direction(s) / cone |

## Current Results

| Model variant | JBB ASR | Harmless compliance |
|---|---|---|
| Base (Llama-3.1-8B-Instruct) | 0.16 | 1.00 |
| DIM-Ablated | **1.00** | 1.00 |
| ActSVD-Modified | 0.63 | 1.00 |
| RDO (run `./scripts/run_rco_eval.sh`) | — | — |

Key geometric findings:
- DIM vs ActSVD MSO is near the random baseline for most layers, with a mild hotspot around layer 10.
- Direct safety-vs-utility overlap is substantially above random: rank-8 mean MSO = 0.192 vs 0.00195 baseline.
- RepInd is asymmetric: ablating DIM strongly changes derived basis profiles, but ablating those bases barely changes DIM.

## Repository Layout

```
docs/
  report.typ                 Final report (Typst format).
  slides.typ                 Midterm presentation slides.
  figures/                   PNG figures copied from code/results/ for Typst.
  bibliography.bib           Citations.

code/
  methods/
    dim/                     DIM refusal-direction pipeline (Arditi et al.).
    actsvd/                  ActSVD low-rank modification pipeline (Wei et al.).
    cones-repind/            RDO/RCO training and RepInd code (Wollschläger et al.).
  analysis/
    plot_benchmarks.py       Safety-utility tradeoff plots.
    plot_method_overlap.py   DIM-vs-ActSVD MSO plots.
    plot_safety_utility_overlap.py  Direct safety-vs-utility overlap plots.
    plot_geometry_repind.py  RepInd heatmap and profile plots.
    safety_utility_overlap.py      Direct overlap analysis runner.
    geometry_repind.py       RepInd cosine-profile comparison runner.
    eval_direction_benchmark.py    Evaluate any direction .pt on JBB + harmless compliance.
  results/
    benchmark/               JBB ASR, harmless compliance, perplexity JSON + plots.
    method_overlap/          DIM-vs-ActSVD MSO JSON + plots.
    safety_utility_overlap/  Safety-vs-utility MSO JSON + plots.
    geometry_repind/         RepInd JSON + plots (DIM-derived basis).
    geometry_repind_rco/     RepInd JSON + plots (DIM + RDO directions, after run_rco_eval.sh).
  tools/
    chat.py                  Interactive chat with any saved modified model.
  data-exploration/          Alpaca and BeaverTails EDA scripts and plots.
  llm_weights/               Local HF cache (git-ignored).

Selected Papers/             Reference paper PDFs and extracted TeX.

scripts/
  run_dim.sh                 Run DIM pipeline.
  run_actsvd.sh              Run ActSVD pipeline.
  run_rco.sh                 Run RDO/RCO training.
  run_rco_eval.sh            Train RDO + extract direction + evaluate on benchmark + RepInd.
  run_safety_utility_overlap.sh  Run direct safety-vs-utility overlap analysis.
  run_geometry_repind.sh     Run lightweight RepInd analysis.
  run_all_experiments.sh     End-to-end runner (all methods + figures).
  build_rco_directions_json.py   Extract trained direction from local wandb artifacts.
  sync_figures.py            Copy result plots to docs/figures/.
  inventory.py               Report all local model artifacts.

notebooks/
  colab_end_to_end.ipynb     Complete Colab notebook for running all experiments.

archive/                     Old reference code (not part of the active workflow).
```

## Local Model Artifacts

| Path | Size | Description |
|------|------|-------------|
| `code/llm_weights/` | ~30 GB | Base Llama-3.1-8B-Instruct (HF cache) |
| `code/methods/actsvd/out/` | ~15 GB | ActSVD-modified model |
| `code/methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/modified_model/` | ~15 GB | DIM-modified model |
| `code/methods/cones-repind/results/modified_model/Llama-3.1-8B-Instruct/` | ~15 GB | DIM-modified model (cones pipeline) |

```bash
python scripts/inventory.py
```

## Setup

```bash
cd code
uv sync
```

For gated Llama models:

```bash
huggingface-cli login
```

The scripts default `HF_HOME` to `code/llm_weights`, so downloads stay in-project.

## Running Experiments

### DIM (Method 1)

```bash
./scripts/run_dim.sh
# Override model:
MODEL_PATH=google/gemma-2b-it ./scripts/run_dim.sh
```

### ActSVD (Method 2)

```bash
./scripts/run_actsvd.sh
# Quicker smoke test:
NSAMPLES=32 ./scripts/run_actsvd.sh
# Full eval:
EVAL_PPL=1 EVAL_ATTACK=1 ./scripts/run_actsvd.sh
```

### RDO / RCO (Method 3 — The Geometry Paper)

Train, extract, and evaluate in one command:

```bash
./scripts/run_rco_eval.sh
# Overrides:
MODEL=meta-llama/Llama-3.1-8B-Instruct MODE=direction METHOD_NAME=RDO ./scripts/run_rco_eval.sh
# Full 2D cone:
MODEL=meta-llama/Llama-3.1-8B-Instruct MODE=cone CONE_DIM=2 METHOD_NAME=RCO-cone ./scripts/run_rco_eval.sh
# Skip training if artifacts already exist:
SKIP_TRAIN=1 ./scripts/run_rco_eval.sh
```

Or run only the training step:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct MODE=direction ./scripts/run_rco.sh
```

### Subspace Analyses

```bash
# Direct safety-vs-utility overlap (how much do safety directions lie inside utility activations?)
./scripts/run_safety_utility_overlap.sh
UTILITY_RANKS=1,2,4,8,16,32 PRIMARY_RANK=8 ./scripts/run_safety_utility_overlap.sh

# RepInd profile analysis (DIM-derived basis, no training required)
./scripts/run_geometry_repind.sh

# RepInd with trained RDO direction (after run_rco_eval.sh)
DIRECTIONS_JSON=code/results/geometry_repind/directions_rco.json \
  OUTPUT_DIR=code/results/geometry_repind_rco \
  ./scripts/run_geometry_repind.sh
```

### Evaluate any saved direction

```bash
cd code
uv run python analysis/eval_direction_benchmark.py \
  --direction_path methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/direction.pt \
  --method_name DIM-verify
```

### Full end-to-end

```bash
# Default: DIM + ActSVD + overlap + RepInd (no RCO training)
./scripts/run_all_experiments.sh

# With RDO:
RUN_RCO=1 RCO_MODE=direction RCO_METHOD_NAME=RDO ./scripts/run_all_experiments.sh

# Skip expensive steps:
RUN_ACTSVD=0 RUN_FIGURES=1 ./scripts/run_all_experiments.sh
```

## Regenerating Figures

```bash
cd code
uv run python analysis/plot_benchmarks.py
uv run python analysis/plot_method_overlap.py
uv run python analysis/plot_safety_utility_overlap.py
uv run python analysis/plot_geometry_repind.py
cd ..
python scripts/sync_figures.py
```

## Google Colab (Recommended for GPU-limited users)

Open `notebooks/colab_end_to_end.ipynb` in Google Colab with an **A100 GPU runtime**.

Quick start:

```
Runtime → Change runtime type → GPU → A100 (or A100 80 GB for ActSVD)
```

The notebook covers all steps end-to-end, including RDO training on Llama-3.1-8B-Instruct.

If you don't have A100 access, use `MODEL_PATH = "google/gemma-2b-it"` and skip ActSVD — DIM + RDO + analyses fit on a T4.

Minimal command sequence for Colab:

```bash
git clone -b findings https://github.com/evan203/nlp-project-proposal.git
cd nlp-project-proposal

pip install -U torch transformers accelerate datasets sentencepiece protobuf \
    tqdm jaxtyping matplotlib seaborn zstandard litellm einops nnsight==0.3.7 \
    vllm python-dotenv wandb

huggingface-cli login

MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PYTHON_RUNNER=python ./scripts/run_dim.sh
MODEL_ALIAS=Llama-3.1-8B-Instruct PYTHON_RUNNER=python NSAMPLES=128 ./scripts/run_actsvd.sh
MODEL=meta-llama/Llama-3.1-8B-Instruct MODE=direction PYTHON_RUNNER=python ./scripts/run_rco_eval.sh
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PYTHON_RUNNER=python ./scripts/run_safety_utility_overlap.sh
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PYTHON_RUNNER=python ./scripts/run_geometry_repind.sh

python code/analysis/plot_benchmarks.py
python code/analysis/plot_method_overlap.py
python code/analysis/plot_safety_utility_overlap.py
python code/analysis/plot_geometry_repind.py
python scripts/sync_figures.py
python scripts/inventory.py
```

## Chat with Modified Models

```bash
cd code
uv run python tools/chat.py \
  --model_path methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/modified_model
```

## Reference Papers

| Paper | Key method | Location |
|-------|------------|----------|
| Arditi et al. 2024 | DIM | `Selected Papers/Refusal in Language Models/` |
| Wei et al. 2024 | ActSVD | `Selected Papers/Assessing the Brittlness/` |
| Wollschläger et al. 2025 | RDO/RCO/RepInd | `Selected Papers/The Geometry/` |
| Ponkshe et al. 2026 | MSO, safety subspace entanglement | `Selected Papers/Safety Subspaces/` |
