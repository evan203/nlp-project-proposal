#!/usr/bin/env bash
set -euo pipefail

# End-to-end project runner for the current implementation.
#
# Default behavior runs DIM, ActSVD, direct overlap, the lightweight
# Geometry/RepInd profile analysis (DIM-derived basis), and figure sync.
# RCO/RDO training and evaluation are opt-in with RUN_RCO=1 because they
# require a separate training run and additional GPU time.
#
# Environment overrides:
#   RUN_DIM=0        Skip DIM.
#   RUN_ACTSVD=0     Skip ActSVD.
#   RUN_OVERLAP=0    Skip direct safety-vs-utility overlap.
#   RUN_REPIND=0     Skip lightweight Geometry/RepInd profile analysis.
#   RUN_RCO=1        Include full RCO/RDO training + evaluation.
#   RCO_MODE         RCO training mode: direction, independent, orthogonal, cone. Default: direction.
#   RCO_METHOD_NAME  Name used in benchmark_results.json. Default: RDO.
#   RUN_FIGURES=0    Skip regenerating/syncing figures.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PYTHON_RUNNER:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON_RUNNER="uv run python"
  else
    PYTHON_RUNNER="python"
  fi
fi

if [[ "${RUN_DIM:-1}" == "1" ]]; then
  "$ROOT_DIR/scripts/run_dim.sh"
fi

if [[ "${RUN_ACTSVD:-1}" == "1" ]]; then
  "$ROOT_DIR/scripts/run_actsvd.sh"
fi

if [[ "${RUN_OVERLAP:-1}" == "1" ]]; then
  "$ROOT_DIR/scripts/run_safety_utility_overlap.sh"
fi

if [[ "${RUN_REPIND:-1}" == "1" ]]; then
  "$ROOT_DIR/scripts/run_geometry_repind.sh"
fi

if [[ "${RUN_RCO:-0}" == "1" ]]; then
  # Full RCO/RDO pipeline: train + extract direction + eval on benchmark + RepInd comparison
  MODE="${RCO_MODE:-direction}" METHOD_NAME="${RCO_METHOD_NAME:-RDO}" \
    "$ROOT_DIR/scripts/run_rco_eval.sh"
fi

if [[ "${RUN_FIGURES:-1}" == "1" ]]; then
  (cd "$ROOT_DIR/code" && $PYTHON_RUNNER analysis/plot_benchmarks.py)
  (cd "$ROOT_DIR/code" && $PYTHON_RUNNER analysis/plot_method_overlap.py)
  (cd "$ROOT_DIR/code" && $PYTHON_RUNNER analysis/plot_safety_utility_overlap.py)
  if [[ -f "$ROOT_DIR/code/results/geometry_repind/geometry_repind_results.json" ]]; then
    (cd "$ROOT_DIR/code" && $PYTHON_RUNNER analysis/plot_geometry_repind.py)
  fi
  "$ROOT_DIR/scripts/sync_figures.py"
fi
