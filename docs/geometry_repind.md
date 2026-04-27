# Geometry / Cones / RepInd Comparisons

This project keeps `code/methods/cones-repind/` as the implementation path for **The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**.

The Geometry paper contributes two comparison types that are different from the MSO plots in the main report.

## 1. Cone/RDO Behavioral Comparison

The code trains vectors or vector sets with `rdo.py`:

- `--train_direction`: train one optimized refusal direction.
- `--train_orthogonal_direction`: train a direction constrained to be orthogonal to the DIM direction.
- `--train_independent_direction`: train a direction intended to be representationally independent from DIM.
- `--train_cone`: train a multi-vector refusal cone.

Those vectors are then evaluated with `refusal_direction/pipeline/run_rdo_pipeline.py`:

- Ablate a trained vector/basis vector and measure harmful-prompt bypass.
- Add a trained vector/basis vector and measure induced refusal.
- For cones, evaluate individual basis vectors and optionally sampled cone directions with `run_rdo_samples.py`.

In report terms, this asks:

> Does a multi-direction refusal region remove safety more reliably than a single DIM direction, and with fewer side effects?

## 2. Representational Independence Comparison

The key comparison is not just cosine similarity between vectors. It is a causal profile test:

1. Measure each direction's cosine-similarity profile across model layers on harmful prompts.
2. Ablate another direction.
3. Re-measure the first direction's cosine-similarity profile.
4. If the profile changes little, the directions are treated as representationally independent.

The upstream `cosinesim_analysis.py` compares:

- DIM baseline vs DIM after RDO-orthogonal ablation.
- RDO-orthogonal baseline vs RDO-orthogonal after DIM ablation.
- RepInd baseline vs RepInd after DIM ablation.
- DIM baseline vs DIM after RepInd ablation.

This is stronger than MSO/cosine because it tests whether one direction's functional representation depends on another direction.

## Current Project Status

The active report currently has completed code/results for:

- DIM behavioral evaluation.
- ActSVD behavioral evaluation.
- DIM-vs-ActSVD method-overlap MSO.
- Direct DIM safety-vs-utility activation overlap.
- Geometry-paper-style RepInd profile comparisons through a local project script.

Run the RepInd profile analysis with the default DIM-derived directions:

```bash
./scripts/run_geometry_repind.sh
```

This produces:

- `code/results/geometry_repind/geometry_repind_results.json`
- `code/results/geometry_repind/repind_change_heatmap.png`
- `code/results/geometry_repind/repind_dim_pair_profiles.png`

Without trained RCO artifacts, the default run derives a small cone-like basis
from high-norm DIM mean-difference candidates. This is enough to apply the
RepInd methodology, but it should be described as a DIM-derived profile
comparison, not as a fully optimized RCO cone.

The full cones-repind training code is runnable through:

```bash
WANDB_MODE=offline MODEL=google/gemma-2b-it MODE=cone CONE_DIM=2 ./scripts/run_rco.sh
```

After that, build a directions file and evaluate the trained vectors:

```bash
python scripts/build_rco_directions_json.py --latest
DIRECTIONS_JSON=code/results/geometry_repind/directions.json ./scripts/run_geometry_repind.sh
```

Only make full optimized cone/RCO claims after those trained-vector results are
present in `code/results/geometry_repind/`.
