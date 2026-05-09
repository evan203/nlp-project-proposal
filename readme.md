# Comparing Safety-Removal Subspaces in Aligned LLMs

Course project for COMP SCI 639 (Deep Learning for NLP). We compare three
methods for removing refusal behavior from Llama-3.1-8B-Instruct and ask
whether they identify the same underlying refusal mechanism.

## Research Questions

1. **Cross-method agreement**: Do DIM, ActSVD, and RCO converge on the same
   safety-relevant mechanism, or do they each capture a distinct solution?
2. **Safety-utility entanglement**: How much do refusal directions overlap with
   harmless-instruction utility activation subspaces?
3. **Causal independence**: Are the directions identified by each method
   representationally independent under ablation (RepInd)?

## Methods

| Method | Paper | What it identifies |
|--------|-------|-------------------|
| **DIM** | Arditi et al. 2024 | Single difference-in-means direction in the residual stream |
| **ActSVD** | Wei et al. 2024 | Low-rank safety/utility projection in weight space |
| **RCO** | Wollschlaeger et al. 2025 | Gradient-optimized 2-D refusal cone in activation space |

## Current Report Results

The shared final-report integration target is `docs/report.typ`, with the
compiled PDF mirrored at `report.pdf`. The results section was transferred from
`docs/claude_report.typ`. Both reports read figure files from `docs/figures/`.

| Model variant | Substring ASR | Qwen3Guard ASR | Harmless compliance | Pile PPL | Alpaca PPL |
|---|---:|---:|---:|---:|---:|
| Base (Llama-3.1-8B-Instruct) | 0.15 | 0.00 | 1.00 | 13.93 | 8.60 |
| DIM-Ablated | 1.00 | 0.90 | 1.00 | 14.17 | 8.80 |
| ActSVD-Modified | 0.80 | 0.77 | 1.00 | 20.16 | 11.65 |
| RCO-Cone-2 | 1.00 | **0.93** | 1.00 | 14.08 | 8.76 |
| Random-Direction-7-1D | 0.16 | 0.00 | 0.98 | 14.65 | 8.86 |
| Random-Subspace-7-2D | 0.14 | 0.00 | 0.99 | 14.61 | 8.78 |

Key geometric findings:

- DIM-vs-ActSVD and RCO-vs-ActSVD have a layer-10 overlap hotspot, but the
  weight-space bridge is treated as exploratory.
- The full DIM mean-difference stack overlaps the rank-8 utility PCA basis at
  98x random, while the selected DIM direction is 40x at layer 11.
- RCO's optimized 2-D cone has normalized utility overlap of 0.0030, or 1.5x
  random, while preserving perplexity comparably to DIM.
- RepInd and the prompt-attack probe suggest DIM captures a dominant but
  non-exhaustive refusal mediator.

## Repository Layout

```text
docs/
  claude_report.typ          Active final report source.
  claude_report.pdf          Compiled active report.
  figures/                   Report-ready PNG figures used by claude_report.typ.
  bibliography.bib           Citations.

code/
  methods/
    dim/                     DIM refusal-direction pipeline (Arditi et al.).
    actsvd/                  ActSVD low-rank modification pipeline (Wei et al.).
    cones-repind/            RCO/RDO training and RepInd code (Wollschlaeger et al.).
  analysis/
    eval_direction_benchmark.py    Evaluate base, DIM, ActSVD, RCO, or random subspaces.
    judge_completions.py           Post-hoc Qwen3Guard judge over saved completions.
    compute_method_overlap.py      DIM/RCO-vs-ActSVD weight-delta MSO.
    safety_utility_overlap.py      Safety-vs-utility activation-overlap analysis.
    geometry_repind.py             RepInd cosine-profile comparison runner.
    probe_attack_types.py          HarmBench/WildJailbreak prompt probe.
    plot_*.py                      Plotting scripts for the analyses above.
  results/                  Local/generated analysis outputs. Not used directly by the report.
  tools/
    chat.py                  Interactive chat with a saved modified model.
  llm_weights/               Local HF cache (git-ignored).

colab_results_v3/            Latest extracted Colab run artifacts used for final figures/data.

scripts/
  run_dim.sh                 Run DIM pipeline.
  run_actsvd.sh              Run ActSVD pipeline (r^u=3950, r^s=4090 by default).
  run_rco.sh                 Run RCO/RDO training backend.
  run_rco_eval.sh            Train RCO, extract direction, benchmark, and run RepInd.
  run_behavioral_benchmark.sh  Reproduce the full behavioral benchmark table.
  run_safety_utility_overlap.sh  Run safety-vs-utility overlap analysis.
  run_method_overlap.sh      Run DIM/RCO-vs-ActSVD weight-delta MSO.
  run_geometry_repind.sh     Run RepInd profile analysis.
  run_probe_attack_types.sh  Run prompt-attack probe with layer sweep and ablation cross-test.
  run_all_experiments.sh     Convenience runner for the major script steps.
  build_rco_directions_json.py   Extract trained RCO direction artifacts.
  sync_figures.py            Copy generated plots to docs/figures/.
  inventory.py               Report local model artifacts.

notebooks/
  colab_end_to_end.ipynb     Guided Colab workflow for reproducing the experiments.
```

## Setup

```bash
cd code
uv sync
```

For gated Llama models and WildJailbreak access:

```bash
huggingface-cli login
export HF_TOKEN=...
```

The scripts default `HF_HOME` to `code/llm_weights`, so downloads stay
in-project. The full Llama-3.1-8B-Instruct workflow expects an A100-class GPU;
ActSVD can require an A100 80 GB runtime depending on environment settings.

## Guided Colab Workflow

The recommended reproduction path is `notebooks/colab_end_to_end.ipynb`. Open it
in Google Colab, select an A100 GPU runtime, authenticate Hugging Face, and run
the cells in order. The notebook runs the same scripts documented below and
packages the resulting JSON files, figures, and direction tensors into
`colab_results.zip`.

The notebook writes report-facing figures to `docs/figures/` and lightweight
artifacts to `code/results/`. Modified model weights are not included in the
zip because they are too large for direct notebook download.

## Script Workflow

Run commands from the repository root unless noted otherwise. Each script honors
`PYTHON_RUNNER`; if unset, scripts prefer `uv run python` when `uv` is available
and otherwise fall back to `python`.

### 1. Train or Load Method Artifacts

DIM:

```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct scripts/run_dim.sh
```

ActSVD with report hyperparameters:

```bash
MODEL_ALIAS=Llama-3.1-8B-Instruct \
RANK_POS=3950 RANK_NEG=4090 NSAMPLES=128 \
scripts/run_actsvd.sh
```

RCO 2-D cone:

```bash
MODE=cone CONE_DIM=2 METHOD_NAME=RCO-Cone-2 scripts/run_rco_eval.sh
```

`run_rco_eval.sh` trains the cone, extracts the trained 2-D basis to
`code/results/geometry_repind/rco_direction.pt`, evaluates RCO on the benchmark,
and runs the DIM-vs-RCO RepInd comparison.

### 2. Behavioral Benchmark

After the DIM, ActSVD, and optional RCO artifacts exist, reproduce the full
behavioral benchmark table with:

```bash
scripts/run_behavioral_benchmark.sh
```

This evaluates the base model, DIM, ActSVD, RCO when
`code/results/geometry_repind/rco_direction.pt` exists, both random baselines,
the post-hoc Qwen3Guard judge, and benchmark plots. The main output is
`code/results/benchmark/benchmark_results.json`.

The wrapper is equivalent to the explicit commands below. They are kept here so
each reported row can be rerun independently if needed:

```bash
cd code

python analysis/eval_direction_benchmark.py \
  --no_ablation \
  --method_name "Base (Llama-3.1-8B-Instruct)" \
  --eval_ppl --n_ppl_samples 64 \
  --eval_truthfulqa --n_tqa_samples 64 \
  --bootstrap 1000

python analysis/eval_direction_benchmark.py \
  --direction_path methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/direction.pt \
  --direction_metadata methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/direction_metadata.json \
  --method_name DIM-Ablated \
  --eval_ppl --n_ppl_samples 64 \
  --eval_truthfulqa --n_tqa_samples 64 \
  --bootstrap 1000

python analysis/eval_direction_benchmark.py \
  --modified_model_path methods/actsvd/out \
  --method_name ActSVD-Modified \
  --eval_ppl --n_ppl_samples 64 \
  --eval_truthfulqa --n_tqa_samples 64 \
  --bootstrap 1000

python analysis/eval_direction_benchmark.py \
  --random_direction --random_subspace_dim 1 --seed 7 \
  --method_name Random-Direction-7-1D \
  --eval_ppl --n_ppl_samples 64 \
  --eval_truthfulqa --n_tqa_samples 64 \
  --bootstrap 1000

python analysis/eval_direction_benchmark.py \
  --random_direction --random_subspace_dim 2 --seed 7 \
  --method_name Random-Subspace-7-2D \
  --eval_ppl --n_ppl_samples 64 \
  --eval_truthfulqa --n_tqa_samples 64 \
  --bootstrap 1000
```

After all methods have saved completions, run the external judge:

```bash
python analysis/judge_completions.py \
  --benchmark_dir results/benchmark \
  --judge_model_path Qwen/Qwen3Guard-Gen-4B \
  --bootstrap 1000

python analysis/plot_benchmarks.py
cd ..
```

`Controversial` Qwen3Guard outputs are counted as not jailbroken unless
`--include_controversial` is passed.

### 3. Geometry Analyses

```bash
scripts/run_safety_utility_overlap.sh
scripts/run_method_overlap.sh
scripts/run_geometry_repind.sh
scripts/run_probe_attack_types.sh
```

These run, respectively:

- safety-vs-utility activation PCA overlap,
- DIM/RCO overlap with the ActSVD weight-delta column space,
- RepInd cosine-profile analysis,
- prompt-attack probe with layer sweep and ablation cross-test.

The prompt probe streams WildJailbreak and requires `HF_TOKEN`.

Expected outputs:

| Report item | Command | Main output |
|---|---|---|
| Behavioral benchmark table and tradeoff plot | `scripts/run_behavioral_benchmark.sh` | `code/results/benchmark/benchmark_results.json`, `code/results/benchmark/*.png` |
| Figure 2 cross-method MSO | `scripts/run_method_overlap.sh` | `code/results/method_overlap/comparison_results.json`, `code/results/method_overlap/mso_per_layer.png` |
| Safety-utility overlap table/figure | `scripts/run_safety_utility_overlap.sh` | `code/results/safety_utility_overlap/safety_utility_overlap_results.json`, `safety_utility_overlap_per_layer.png` |
| RepInd heatmap | `scripts/run_geometry_repind.sh` or the RepInd step inside `scripts/run_rco_eval.sh` | `code/results/geometry_repind_rco/geometry_repind_results.json`, `repind_change_heatmap.png` |
| Prompt-attack probe | `scripts/run_probe_attack_types.sh` | `code/results/probe_attack_types/results.json`, probe figure PNGs |

### 4. Figures and Report

For a fresh local run, sync generated plots into the report figure directory:

```bash
scripts/sync_figures.py
```

For this final submission, `docs/figures/` has already been prepared from
`colab_results_v3/`. The report uses only `docs/figures/*`, not
`code/results/*`.

Compile the shared final report:

```bash
typst compile --root . docs/report.typ report.pdf
```

Compile the standalone results source:

```bash
cd docs
typst compile claude_report.typ
cp claude_report.pdf ../report.pdf
```

Typst may warn about missing template fonts (`inconsolata` and
`tex gyre termes`) on machines that do not have them installed; the report still
compiles successfully with fallback fonts.

## Local Model Artifacts

| Path | Size | Description |
|------|------|-------------|
| `code/llm_weights/` | ~30 GB | Base Llama-3.1-8B-Instruct HF cache |
| `code/methods/actsvd/out/` | ~15 GB | ActSVD-modified model |
| `code/methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/` | varies | DIM direction, metadata, completions, optional modified model |
| `code/methods/cones-repind/results/` | varies | RCO/RDO training artifacts |

```bash
python scripts/inventory.py
```

## Lightweight Validation

These checks do not run the expensive model experiments, but they catch syntax,
import, shell, and Typst errors in the submitted code paths:

```bash
bash -n scripts/*.sh
python -m compileall -q code/analysis scripts code/tools
typst compile --root . docs/report.typ /tmp/report-check.pdf
```

## Reference Papers

| Paper | Key method | Location |
|-------|------------|----------|
| Arditi et al. 2024 | DIM | `Selected Papers/Refusal in Language Models/` |
| Wei et al. 2024 | ActSVD | `Selected Papers/Assessing the Brittlness/` |
| Wollschlaeger et al. 2025 | RCO/RDO/RepInd | `Selected Papers/The Geometry/` |
| Ponkshe et al. 2026 | MSO, safety subspace entanglement | `Selected Papers/Safety Subspaces/` |
