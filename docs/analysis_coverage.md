# Analysis Coverage

This file maps report claims to code and outputs.

| Report claim | Code | Output |
|---|---|---|
| DIM can remove refusal behavior on Llama-3.1-Instruct. | `scripts/run_dim.sh`, `code/methods/dim/pipeline/run_pipeline.py` | `code/methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/` |
| ActSVD can partially remove refusal behavior with more utility degradation. | `scripts/run_actsvd.sh`, `code/methods/actsvd/main_low_rank_diff.py` | `code/methods/actsvd/out/`, summarized in `code/results/benchmark/` |
| Base, DIM, and ActSVD differ in safety/utility behavior. | `code/analysis/plot_benchmarks.py` | `code/results/benchmark/*.png`, `benchmark_results.json` |
| DIM and ActSVD target mostly different subspaces. | `code/analysis/plot_method_overlap.py` | `code/results/method_overlap/*.png`, `comparison_results.json` |
| DIM safety directions overlap utility activation subspaces above random. | `scripts/run_safety_utility_overlap.sh`, `code/analysis/safety_utility_overlap.py` | `code/results/safety_utility_overlap/*.json`, `*.png` |
| Geometry-paper RepInd asks whether directions remain causally independent after ablation. | `scripts/run_geometry_repind.sh`, `code/analysis/geometry_repind.py`, `code/analysis/plot_geometry_repind.py` | `code/results/geometry_repind/*.json`, `*.png` |
| Full RCO/cone training can be run as a heavier extension. | `scripts/run_rco.sh`, `code/methods/cones-repind/rdo.py` | Local vectors under `code/methods/cones-repind/results/rdo/*/local_vectors/` when run. |

The report should distinguish between the lightweight RepInd profile analysis and a full optimized RCO/cone result. Full cone claims require running `scripts/run_rco.sh` and then passing the saved vectors into `scripts/run_geometry_repind.sh` with `DIRECTIONS_JSON=...`.
