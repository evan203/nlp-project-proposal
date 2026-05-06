# Analysis Scripts

Project-owned scripts live here. Each script has one job and reads/writes files under `code/results/`.

| Script | Purpose | Main output |
|---|---|---|
| `plot_benchmarks.py` | Plot behavioral safety/utility benchmark results from JSON. | `code/results/benchmark/*.png` |
| `plot_method_overlap.py` | Plot DIM-vs-ActSVD method-overlap MSO and cross-model DIM cosine. | `code/results/method_overlap/*.png` |
| `safety_utility_overlap.py` | Compute direct safety-vs-utility activation-overlap metrics. | `code/results/safety_utility_overlap/safety_utility_overlap_results.json` |
| `plot_safety_utility_overlap.py` | Plot direct safety-vs-utility overlap figures from JSON. | `code/results/safety_utility_overlap/*.png` |
| `geometry_repind.py` | Compute Geometry-paper-style RepInd cosine-profile comparisons for saved directions. | `code/results/geometry_repind/geometry_repind_results.json` |
| `plot_geometry_repind.py` | Plot RepInd profile-change heatmaps and DIM pair profiles. | `code/results/geometry_repind/*.png` |

Method implementations are intentionally separate under `code/methods/`.
