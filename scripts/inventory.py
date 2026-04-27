#!/usr/bin/env python3
"""Print a compact inventory of project code, runs, results, and local model files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def size_gib(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024**3
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / 1024**3


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def list_path(path: Path, label: str) -> None:
    if path.exists():
        print(f"{label}: {path.relative_to(ROOT)} ({size_gib(path):.2f} GiB)")
    else:
        print(f"{label}: missing ({path.relative_to(ROOT)})")


def main() -> None:
    print_section("Code Modules")
    for rel, desc in [
        ("code/methods/dim", "DIM refusal-direction pipeline"),
        ("code/methods/actsvd", "ActSVD low-rank modification pipeline"),
        ("code/methods/cones-repind", "RCO/RDO and representational-independence code"),
        ("code/analysis", "project-owned analysis and plotting scripts"),
        ("code/results", "generated JSON summaries and figures"),
        ("code/data-exploration", "Dataset EDA"),
        ("archive/code/safety-subspaces", "archived safety-subspaces reference implementation"),
    ]:
        path = ROOT / rel
        status = "present" if path.exists() else "missing"
        print(f"{rel}: {status} - {desc}")

    print_section("Local Model Artifacts")
    list_path(ROOT / "code/llm_weights", "HF cache")
    list_path(ROOT / "code/methods/actsvd/out", "ActSVD modified model")
    list_path(
        ROOT / "code/methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/modified_model",
        "DIM modified Llama-3.1 model",
    )

    print_section("DIM Runs")
    runs = ROOT / "code/methods/dim/pipeline/runs"
    if runs.exists():
        for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
            direction = run_dir / "direction.pt"
            completions = run_dir / "completions"
            modified = run_dir / "modified_model"
            flags = []
            if direction.exists():
                flags.append("direction")
            if completions.exists():
                flags.append("completions")
            if modified.exists():
                flags.append("modified_model")
            print(f"{run_dir.name}: {', '.join(flags) if flags else 'no artifacts'}")
    else:
        print("No DIM run directory found.")

    print_section("Published Result Summaries")
    for rel in [
        "code/results/benchmark/benchmark_results.json",
        "code/results/method_overlap/comparison_results.json",
        "code/results/safety_utility_overlap/safety_utility_overlap_results.json",
        "code/results/geometry_repind/geometry_repind_results.json",
    ]:
        path = ROOT / rel
        if not path.exists():
            print(f"{rel}: missing")
            continue
        try:
            data = json.loads(path.read_text())
            print(f"{rel}: present, top-level keys={list(data)[:8]}")
        except json.JSONDecodeError:
            print(f"{rel}: present")


if __name__ == "__main__":
    main()
