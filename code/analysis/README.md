# Analysis Scripts

Project-owned scripts live here. Each script has one job and reads/writes files under `code/results/`.

| Script | Purpose | Main output |
|---|---|---|
| `eval_direction_benchmark.py` | Evaluate base, DIM, ActSVD, RCO, or random-subspace interventions on JailbreakBench, harmless compliance, perplexity, and TruthfulQA. | `code/results/benchmark/benchmark_results.json` |
| `judge_completions.py` | Re-grade saved JailbreakBench completions with Qwen3Guard-Gen-4B. | Updates `code/results/benchmark/benchmark_results.json` |
| `plot_benchmarks.py` | Plot behavioral safety/utility benchmark results from JSON. | `code/results/benchmark/*.png` |
| `compute_method_overlap.py` | Compute MSO between activation-space directions and the ActSVD weight-delta column space. | `code/results/method_overlap/comparison_results.json` |
| `plot_method_overlap.py` | Plot DIM-vs-ActSVD method-overlap MSO and cross-model DIM cosine. | `code/results/method_overlap/*.png` |
| `safety_utility_overlap.py` | Compute direct safety-vs-utility activation-overlap metrics. | `code/results/safety_utility_overlap/safety_utility_overlap_results.json` |
| `plot_safety_utility_overlap.py` | Plot direct safety-vs-utility overlap figures from JSON. | `code/results/safety_utility_overlap/*.png` |
| `geometry_repind.py` | Compute Wollschläger et al.-style RepInd cosine-profile comparisons for saved directions. | `code/results/geometry_repind/geometry_repind_results.json` |
| `plot_geometry_repind.py` | Plot RepInd profile-change heatmaps and DIM pair profiles. | `code/results/geometry_repind/*.png` |
| `probe_attack_types.py` | Run the HarmBench/WildJailbreak prompt probe, layer sweep, and ablation cross-test. | `code/results/probe_attack_types/results.json` |
| `plot_attack_types.py` | Plot prompt-probe projection, ASR, layer-sweep, and ablation-cross-test figures. | `code/results/probe_attack_types/*.png` |

Method implementations are intentionally separate under `code/methods/`.
