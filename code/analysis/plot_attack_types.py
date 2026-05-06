#!/usr/bin/env python3
"""Plot results from probe_attack_types.py.

Produces three figures saved to both results/probe_attack_types/ and docs/figures/:
  1. scatter_projection_vs_outcome.png  — per-prompt direction projection vs. refusal/jailbreak
  2. boxplot_projection_by_attack_type.png — projection distributions split by outcome per category
  3. asr_and_projection_by_attack_type.png — side-by-side bar charts of ASR and mean projection
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent
FIGURES_DIR = ROOT / "docs" / "figures"

_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]


def _ordered_tactics(results: list[dict]) -> list[str]:
    """Return tactic names: direct_request first, then rest sorted by frequency."""
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r["tactic"]] += 1
    others = sorted((t for t in counts if t != "direct_request"), key=lambda t: -counts[t])
    return (["direct_request"] if "direct_request" in counts else []) + others


def _label(tactic: str) -> str:
    return tactic.replace("_", " ").title()


def _color_map(tactics: list[str]) -> dict[str, str]:
    return {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(tactics)}


def _save(fig: plt.Figure, path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / path.name, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path,
                   default=CODE_ROOT / "results/probe_attack_types")
    p.add_argument("--output_dir", type=Path,
                   default=CODE_ROOT / "results/probe_attack_types")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = json.loads((args.results_dir / "results.json").read_text())
    for r in results:
        if "tactic" in r and "attack_type" not in r:
            r["attack_type"] = r["tactic"]
    present_types = _ordered_tactics(results)
    colors = _color_map(present_types)

    # Discover which direction projections are available (proj_DIM, proj_RCO, ...)
    proj_keys = sorted({k for r in results for k in r if k.startswith("proj_")})

    rng = np.random.default_rng(0)

    # -----------------------------------------------------------------------
    # Figure 1: Scatter — direction projection vs jailbreak outcome
    # One panel per direction (DIM, RCO if available)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(proj_keys), figsize=(9 * len(proj_keys), 5), squeeze=False)
    for ax, proj_key in zip(axes[0], proj_keys):
        dir_name = proj_key.replace("proj_", "")
        for atype in present_types:
            entries = [r for r in results if r["attack_type"] == atype]
            projs = [e.get(proj_key, float("nan")) for e in entries]
            outcomes = [e["is_jailbreak"] + rng.uniform(-0.06, 0.06) for e in entries]
            ax.scatter(projs, outcomes, alpha=0.6, s=45,
                       color=colors[atype], label=_label(atype))
        ax.set_xlabel(f"{dir_name} direction projection  (EOI activation)")
        ax.set_ylabel("Outcome")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Refused", "Jailbroken"])
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.legend(loc="center right", fontsize=7, framealpha=0.9)
        ax.set_title(f"{dir_name} direction: projection vs. outcome\n(base model, no modification)")
    fig.tight_layout()
    _save(fig, args.output_dir / "scatter_projection_vs_outcome.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 2: Box plot — projection per attack type, split by outcome.
    # One row per direction (DIM, RCO if available).
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(len(proj_keys), 1,
                             figsize=(11, 4.5 * len(proj_keys)),
                             squeeze=False)
    for ax, proj_key in zip(axes[:, 0], proj_keys):
        dir_name = proj_key.replace("proj_", "")
        tick_centers: list[float] = []
        tick_labels: list[str] = []
        gap = 1.5

        for i, atype in enumerate(present_types):
            entries = [r for r in results if r["attack_type"] == atype]
            refused_projs = [e[proj_key] for e in entries if e["is_refusal"] and proj_key in e]
            jb_projs = [e[proj_key] for e in entries if e["is_jailbreak"] and proj_key in e]
            base = i * gap
            color = colors[atype]
            w = 0.35

            def _bp(data, pos, alpha):
                if not data:
                    return
                ax.boxplot(
                    data, positions=[pos], widths=w, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=alpha),
                    medianprops=dict(color="white" if alpha > 0.6 else "black", linewidth=1.5),
                    whiskerprops=dict(color=color, alpha=alpha),
                    capprops=dict(color=color, alpha=alpha),
                    flierprops=dict(marker="o", markerfacecolor=color, alpha=alpha, markersize=3),
                )

            _bp(refused_projs, base - w / 2, alpha=0.35)
            _bp(jb_projs, base + w / 2, alpha=0.85)
            tick_centers.append(base)
            tick_labels.append(_label(atype))

        ax.set_xticks(tick_centers)
        ax.set_xticklabels(tick_labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(f"{dir_name} direction projection")
        ax.set_title(f"{dir_name} direction projection by attack type\n(light = refused, dark = jailbroken)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        refused_patch = mpatches.Patch(color="gray", alpha=0.35, label="Refused")
        jb_patch = mpatches.Patch(color="gray", alpha=0.85, label="Jailbroken")
        ax.legend(handles=[refused_patch, jb_patch], fontsize=9)

    fig.tight_layout()
    _save(fig, args.output_dir / "boxplot_projection_by_attack_type.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 3: Paired bars — ASR and mean projection per direction per tactic
    # -----------------------------------------------------------------------
    ncols = 1 + len(proj_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5))
    x = np.arange(len(present_types))
    bar_colors = [colors[a] for a in present_types]
    bar_labels = [_label(a) for a in present_types]

    asrs = [sum(r["is_jailbreak"] for r in results if r["attack_type"] == a)
            / max(sum(1 for r in results if r["attack_type"] == a), 1)
            for a in present_types]

    ax1 = axes[0]
    ax1.bar(x, asrs, color=bar_colors, alpha=0.85, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(bar_labels, rotation=22, ha="right", fontsize=8)
    ax1.set_ylabel("Attack success rate (ASR)"); ax1.set_ylim(0, 1.05)
    ax1.axhline(asrs[0], color="gray", linewidth=1, linestyle="--", alpha=0.6,
                label=f"Direct baseline = {asrs[0]:.2f}")
    ax1.legend(fontsize=8); ax1.set_title("ASR by tactic")

    for ax2, proj_key in zip(axes[1:], proj_keys):
        dir_name = proj_key.replace("proj_", "")
        mean_projs = [
            sum(r.get(proj_key, 0) for r in results if r["attack_type"] == a)
            / max(sum(1 for r in results if r["attack_type"] == a), 1)
            for a in present_types
        ]
        ax2.bar(x, mean_projs, color=bar_colors, alpha=0.85, edgecolor="white")
        ax2.set_xticks(x); ax2.set_xticklabels(bar_labels, rotation=22, ha="right", fontsize=8)
        ax2.set_ylabel(f"Mean {dir_name} projection")
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.axhline(mean_projs[0], color="gray", linewidth=1, linestyle=":",
                    label=f"Direct = {mean_projs[0]:.3f}")
        ax2.legend(fontsize=8)
        ax2.set_title(f"Mean {dir_name} projection by tactic\n(lower = more suppression)")

    fig.suptitle("Prompt-based jailbreak: ASR and direction suppression by tactic", fontsize=11)
    fig.tight_layout()
    _save(fig, args.output_dir / "asr_and_projection_by_attack_type.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 4: DIM vs RCO projection scatter — one point per prompt
    # Shows whether attacks that suppress DIM also suppress RCO
    # (only produced when both directions are available)
    # -----------------------------------------------------------------------
    if "proj_DIM" in {k for r in results for k in r} and \
       "proj_RCO" in {k for r in results for k in r}:
        fig, ax = plt.subplots(figsize=(7, 6))
        for atype in present_types:
            entries = [r for r in results if r["attack_type"] == atype]
            dim_ps = [e.get("proj_DIM", float("nan")) for e in entries]
            rco_ps = [e.get("proj_RCO", float("nan")) for e in entries]
            markers = ["o" if e["is_jailbreak"] else "x" for e in entries]
            for d, r_, m in zip(dim_ps, rco_ps, markers):
                ax.scatter([d], [r_], color=colors[atype], marker=m, s=40, alpha=0.6)
            ax.scatter([], [], color=colors[atype], label=_label(atype), s=40)
        ax.set_xlabel("DIM direction projection  (utility MSO = 0.078)")
        ax.set_ylabel("RCO direction projection  (utility MSO = 0.004)")
        ax.set_title("DIM vs RCO projection per prompt\n(○ = jailbroken, × = refused)")
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7, loc="upper left")
        _save(fig, args.output_dir / "dim_vs_rco_projection_scatter.png")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 5: Correlation scatter — mean projection vs ASR per tactic
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(proj_keys), figsize=(6 * len(proj_keys), 5), squeeze=False)
    for ax, proj_key in zip(axes[0], proj_keys):
        dir_name = proj_key.replace("proj_", "")
        mean_projs = [
            sum(r.get(proj_key, 0) for r in results if r["attack_type"] == a)
            / max(sum(1 for r in results if r["attack_type"] == a), 1)
            for a in present_types
        ]
        for atype, asr, mp in zip(present_types, asrs, mean_projs):
            ax.scatter([mp], [asr], color=colors[atype], s=120, zorder=3)
            ax.annotate(_label(atype), (mp, asr), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=colors[atype])
        ax.set_xlabel(f"Mean {dir_name} projection")
        ax.set_ylabel("ASR")
        ax.set_title(f"ASR vs mean {dir_name} projection\n(per tactic — does lower projection → higher ASR?)")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, args.output_dir / "correlation_projection_vs_asr.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 6: Layer sweep — projection per layer per attack group
    # -----------------------------------------------------------------------
    sweep_path = args.results_dir / "layer_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        n_layers = sweep["n_layers"]
        per_dir = sweep["per_direction"]  # name -> [n_prompts][n_layers]
        sweep_tactics = sweep["tactics"]

        sweep_proj_keys = sorted(per_dir.keys())
        fig, axes = plt.subplots(1, len(sweep_proj_keys),
                                 figsize=(7 * len(sweep_proj_keys), 4.5),
                                 squeeze=False)
        layers = list(range(n_layers))
        for ax, dname in zip(axes[0], sweep_proj_keys):
            data = per_dir[dname]  # [n_prompts][n_layers]
            for atype in present_types:
                idxs = [i for i, t in enumerate(sweep_tactics) if t == atype]
                if not idxs:
                    continue
                arr = np.asarray([data[i] for i in idxs])  # [n_atype, n_layers]
                mean = arr.mean(axis=0)
                # Bootstrap-style 1-sigma band over the prompt dimension
                if arr.shape[0] > 1:
                    sd = arr.std(axis=0, ddof=1) / max(np.sqrt(arr.shape[0]), 1.0)
                    ax.fill_between(layers, mean - sd, mean + sd,
                                    color=colors[atype], alpha=0.15)
                ax.plot(layers, mean, color=colors[atype], linewidth=1.6,
                        label=_label(atype))
            ax.set_xlabel("Layer")
            ax.set_ylabel(f"Mean {dname} projection")
            ax.set_title(f"{dname} projection vs layer\n(group means with ±SE band)")
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        _save(fig, args.output_dir / "layer_sweep_projection.png")
        plt.close(fig)
    else:
        print(f"  (skipping layer-sweep figure; no {sweep_path.name})")

    # -----------------------------------------------------------------------
    # Figure 7: Ablation cross-test — base ASR vs DIM-ablated ASR per group
    # -----------------------------------------------------------------------
    if any("ablated_is_jailbreak" in r for r in results):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(present_types))
        bar_labels = [_label(a) for a in present_types]
        bar_colors = [colors[a] for a in present_types]

        base_asr = []
        abl_asr = []
        for a in present_types:
            entries = [r for r in results if r["attack_type"] == a]
            n = max(len(entries), 1)
            base_asr.append(sum(r["is_jailbreak"] for r in entries) / n)
            abl_asr.append(sum(r.get("ablated_is_jailbreak", 0) for r in entries) / n)

        w = 0.4
        ax.bar(x - w / 2, base_asr, width=w, color=bar_colors, alpha=0.6,
               edgecolor="white", label="Base model")
        ax.bar(x + w / 2, abl_asr, width=w, color=bar_colors, alpha=0.95,
               edgecolor="white", label="DIM-ablated")
        ax.set_xticks(x); ax.set_xticklabels(bar_labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("ASR")
        ax.set_ylim(0, 1.05)
        ax.set_title("Probe × ablation cross-test:\n"
                     "for the same prompts, does DIM ablation help where prompt attacks fail?")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        _save(fig, args.output_dir / "ablation_cross_test.png")
        plt.close(fig)
    else:
        print("  (skipping ablation cross-test figure; no ablated_is_jailbreak in results)")

    print("\nAll figures written.")


if __name__ == "__main__":
    main()
