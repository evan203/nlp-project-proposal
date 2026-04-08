#!/usr/bin/env python
"""
Jailbreak Comparison: Compute Representational Independence Statistics

Compares activation spaces of two jailbroken models:
1. ActSVD jailbroken model (via activation subspace modification)
2. Diff-in-Means jailbroken model (via orthogonalization)

Computes:
- Layer-wise cosine similarities between DIM directions
- RepInd statistic: MSE of cosine similarity differences across layers
- Per-layer statistics and visualizations
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformer_utils import (
    load_model_and_tokenizer,
    apply_chat_template,
    get_residual_stream_activations,
    compute_mean_diff_direction,
    compute_dim_directions,
    get_cosine_sims_for_prompts,
    get_model_config,
)


def unload_model(model, tokenizer):
    """Safely unload model and clear GPU memory."""
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


sns.set_style("whitegrid")


def load_saladbench_data(
    data_dir: str = "data/saladbench_splits",
    n_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Load harmful and harmless prompts from SaladBench splits."""
    harmful_path = os.path.join(data_dir, "harmful_train.json")
    harmless_path = os.path.join(data_dir, "harmless_train.json")

    harmful_data = json.load(open(harmful_path))
    harmless_data = json.load(open(harmless_path))

    harmful_instructions = [d["instruction"] for d in harmful_data]
    harmless_instructions = [d["instruction"] for d in harmless_data]

    if n_samples is not None:
        harmful_instructions = harmful_instructions[:n_samples]
        harmless_instructions = harmless_instructions[:n_samples]

    return harmful_instructions, harmless_instructions


def compute_refusal_score(
    logits: torch.Tensor,
    refusal_tokens: List[int],
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute refusal score from logits."""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_tokens].sum(dim=-1)
    nonrefusal_probs = 1 - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)


@torch.no_grad()
def evaluate_model_jailbreak_effectiveness(
    model,
    tokenizer,
    prompts: List[str],
    refusal_tokens: List[int],
    batch_size: int = 8,
) -> Dict[str, float]:
    """Evaluate how well a model has been jailbroken."""
    config = get_model_config(
        model.name_or_path if hasattr(model, "name_or_path") else str(model)
    )

    all_scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.cuda() for k, v in enc.items()}

        outputs = model(**enc)
        logits = outputs.logits[:, -1, :]
        scores = compute_refusal_score(logits, refusal_tokens)
        all_scores.extend(scores.cpu().tolist())

    all_scores = torch.tensor(all_scores)
    return {
        "mean_score": all_scores.mean().item(),
        "std_score": all_scores.std().item(),
        "bypass_rate": (all_scores < 0).float().mean().item(),
        "refusal_rate": (all_scores > 0).float().mean().item(),
    }


def compute_layer_wise_cosine_similarities(
    model_a_activations: List[Dict[int, torch.Tensor]],
    model_b_activations: List[Dict[int, torch.Tensor]],
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    layer_range: Tuple[int, int] = (None, None),
) -> Dict[str, np.ndarray]:
    """
    Compute cosine similarities between directions and activations across layers.

    Returns dict with:
    - 'model_a_cosine_sims': Cosine sims of direction_a with model_a activations
    - 'model_b_cosine_sims': Cosine sims of direction_b with model_b activations
    - 'cross_cosine_sims': Cosine sims of direction_a with model_b activations and vice versa
    """
    n_layers = len(model_a_activations[0])
    start = layer_range[0] if layer_range[0] is not None else 0
    end = layer_range[1] if layer_range[1] is not None else n_layers

    direction_a = direction_a / direction_a.norm()
    direction_b = direction_b / direction_b.norm()

    model_a_cosine_sims = []
    model_b_cosine_sims = []

    for layer_idx in range(start, end):
        model_a_layer_acts = torch.stack(
            [act[layer_idx] for act in model_a_activations]
        )
        model_b_layer_acts = torch.stack(
            [act[layer_idx] for act in model_b_activations]
        )

        cos_a = torch.nn.functional.cosine_similarity(
            model_a_layer_acts, direction_a.unsqueeze(0), dim=-1
        )
        cos_b = torch.nn.functional.cosine_similarity(
            model_b_layer_acts, direction_b.unsqueeze(0), dim=-1
        )

        model_a_cosine_sims.append(cos_a.mean().item())
        model_b_cosine_sims.append(cos_b.mean().item())

    return {
        "model_a_cosine_sims": np.array(model_a_cosine_sims),
        "model_b_cosine_sims": np.array(model_b_cosine_sims),
        "layers": np.arange(start, end),
    }


def compute_cross_model_cosine_sims(
    model_a_activations: List[Dict[int, torch.Tensor]],
    model_b_activations: List[Dict[int, torch.Tensor]],
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    layer_range: Tuple[int, int] = (None, None),
) -> Dict[str, np.ndarray]:
    """
    Compute cross-model cosine similarities.

    Computes cosine similarity between:
    - direction_a with model_b activations
    - direction_b with model_a activations
    """
    n_layers = len(model_a_activations[0])
    start = layer_range[0] if layer_range[0] is not None else 0
    end = layer_range[1] if layer_range[1] is not None else n_layers

    direction_a = direction_a / direction_a.norm()
    direction_b = direction_b / direction_b.norm()

    a_on_b_sims = []
    b_on_a_sims = []

    for layer_idx in range(start, end):
        model_a_layer_acts = torch.stack(
            [act[layer_idx] for act in model_a_activations]
        )
        model_b_layer_acts = torch.stack(
            [act[layer_idx] for act in model_b_activations]
        )

        cos_a_on_b = torch.nn.functional.cosine_similarity(
            model_b_layer_acts, direction_a.unsqueeze(0), dim=-1
        )
        cos_b_on_a = torch.nn.functional.cosine_similarity(
            model_a_layer_acts, direction_b.unsqueeze(0), dim=-1
        )

        a_on_b_sims.append(cos_a_on_b.mean().item())
        b_on_a_sims.append(cos_b_on_a.mean().item())

    return {
        "direction_a_on_model_b": np.array(a_on_b_sims),
        "direction_b_on_model_a": np.array(b_on_a_sims),
        "layers": np.arange(start, end),
    }


def compute_repind_statistic(
    model_a_activations: List[Dict[int, torch.Tensor]],
    model_b_activations: List[Dict[int, torch.Tensor]],
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    layer_range: Tuple[int, int] = (None, None),
) -> Dict[str, float]:
    """
    Compute the Representational Independence (RepInd) statistic.

    Based on the cones paper, this measures how independent two jailbreak
    subspaces are by comparing cosine similarity patterns across layers.

    RepInd = MSE[(cos_sim_A(A) - cos_sim_A(B)) + (cos_sim_B(B) - cos_sim_B(A))]

    Lower values indicate more independent subspaces.
    """
    n_layers = len(model_a_activations[0])
    start = layer_range[0] if layer_range[0] is not None else 0
    end = layer_range[1] if layer_range[1] is not None else n_layers

    direction_a = direction_a / direction_a.norm()
    direction_b = direction_b / direction_b.norm()

    cos_a_on_a = []
    cos_a_on_b = []
    cos_b_on_b = []
    cos_b_on_a = []

    for layer_idx in range(start, end):
        model_a_layer_acts = torch.stack(
            [act[layer_idx] for act in model_a_activations]
        )
        model_b_layer_acts = torch.stack(
            [act[layer_idx] for act in model_b_activations]
        )

        cos_a_on_a.append(
            torch.nn.functional.cosine_similarity(
                model_a_layer_acts, direction_a.unsqueeze(0), dim=-1
            )
            .mean()
            .item()
        )

        cos_a_on_b.append(
            torch.nn.functional.cosine_similarity(
                model_b_layer_acts, direction_a.unsqueeze(0), dim=-1
            )
            .mean()
            .item()
        )

        cos_b_on_b.append(
            torch.nn.functional.cosine_similarity(
                model_b_layer_acts, direction_b.unsqueeze(0), dim=-1
            )
            .mean()
            .item()
        )

        cos_b_on_a.append(
            torch.nn.functional.cosine_similarity(
                model_a_layer_acts, direction_b.unsqueeze(0), dim=-1
            )
            .mean()
            .item()
        )

    cos_a_on_a = np.array(cos_a_on_a)
    cos_a_on_b = np.array(cos_a_on_b)
    cos_b_on_b = np.array(cos_b_on_b)
    cos_b_on_a = np.array(cos_b_on_a)

    within_a_diff = cos_a_on_a - cos_a_on_b
    within_b_diff = cos_b_on_b - cos_b_on_a

    combined_diff = within_a_diff + within_b_diff

    repind_mse = (combined_diff**2).mean()
    repind_mae = np.abs(combined_diff).mean()

    return {
        "repind_mse": repind_mse,
        "repind_mae": repind_mae,
        "within_a_mse": (within_a_diff**2).mean(),
        "within_b_mse": (within_b_diff**2).mean(),
        "layer_diffs": combined_diff.tolist(),
        "cos_a_on_a": cos_a_on_a.tolist(),
        "cos_a_on_b": cos_a_on_b.tolist(),
        "cos_b_on_b": cos_b_on_b.tolist(),
        "cos_b_on_a": cos_b_on_a.tolist(),
    }


def compute_direction_similarity(
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
) -> Dict[str, float]:
    """Compute similarity metrics between two direction vectors."""
    direction_a = direction_a / direction_a.norm()
    direction_b = direction_b / direction_b.norm()

    cosine_sim = torch.nn.functional.cosine_similarity(
        direction_a.unsqueeze(0), direction_b.unsqueeze(0), dim=-1
    ).item()

    dot_product = (direction_a * direction_b).sum().item()

    projection_a_on_b = (direction_a @ direction_b) * direction_b
    orthogonal_component = direction_a - projection_a_on_b
    orthogonality = orthogonal_component.norm().item()

    return {
        "cosine_similarity": cosine_sim,
        "dot_product": dot_product,
        "orthogonality": orthogonality,
        "projection_magnitude": projection_a_on_b.norm().item(),
    }


def plot_repind_analysis(
    results: Dict,
    save_path: str,
    model_a_name: str,
    model_b_name: str,
):
    """Create visualizations for the RepInd analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = results["layer_diffs"]
    x = np.arange(len(layers))

    ax1 = axes[0, 0]
    ax1.plot(
        x,
        results["cos_a_on_a"],
        "b-",
        label=f"{model_a_name} dir on {model_a_name[:10]}",
    )
    ax1.plot(
        x,
        results["cos_a_on_b"],
        "b--",
        label=f"{model_a_name} dir on {model_b_name[:10]}",
    )
    ax1.plot(
        x,
        results["cos_b_on_b"],
        "r-",
        label=f"{model_b_name} dir on {model_b_name[:10]}",
    )
    ax1.plot(
        x,
        results["cos_b_on_a"],
        "r--",
        label=f"{model_b_name} dir on {model_a_name[:10]}",
    )
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Within-Model vs Cross-Model Cosine Similarities")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.bar(x, results["layer_diffs"], alpha=0.7)
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Combined Difference")
    ax2.set_title("Layer-wise RepInd Component")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    metrics = ["RepInd MSE", "RepInd MAE", "Within-A MSE", "Within-B MSE"]
    values = [
        results["repind_mse"],
        results["repind_mae"],
        results["within_a_mse"],
        results["within_b_mse"],
    ]
    bars = ax3.bar(
        metrics, values, alpha=0.7, color=["#2ecc71", "#27ae60", "#3498db", "#2980b9"]
    )
    ax3.set_ylabel("Value")
    ax3.set_title("RepInd Statistics (Lower = More Independent)")
    for bar, val in zip(bars, values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = axes[1, 1]
    dir_metrics = ["Cosine Sim", "Dot Product", "Orthogonality"]
    dir_values = [
        results["direction_similarity"]["cosine_similarity"],
        results["direction_similarity"]["dot_product"],
        results["direction_similarity"]["orthogonality"],
    ]
    ax4.bar(dir_metrics, dir_values, alpha=0.7, color=["#9b59b6", "#8e44ad", "#7d3c98"])
    ax4.set_ylabel("Value")
    ax4.set_title("Direction Vector Similarity")
    for i, val in enumerate(dir_values):
        ax4.text(i, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax4.set_ylim(0, max(dir_values) * 1.2)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Representational Independence Analysis: {model_a_name} vs {model_b_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Representational Independence Statistics Between Jailbroken Models"
    )
    parser.add_argument(
        "--model_a_path",
        type=str,
        required=True,
        help="Path to ActSVD jailbroken model",
    )
    parser.add_argument(
        "--model_b_path",
        type=str,
        required=True,
        help="Path to Diff-in-Means jailbroken model",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model (for computing original DIM directions)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/saladbench_splits",
        help="Path to SaladBench data directory",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples to use for analysis",
    )
    parser.add_argument(
        "--best_layer_a",
        type=int,
        default=None,
        help="Best layer for model A direction extraction (default: auto-detect)",
    )
    parser.add_argument(
        "--best_layer_b",
        type=int,
        default=None,
        help="Best layer for model B direction extraction (default: auto-detect)",
    )
    parser.add_argument(
        "--layer_cutoff",
        type=float,
        default=0.9,
        help="Use layers 0 to layer_cutoff * n_layers for RepInd computation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/jailbreak_comparison",
        help="Directory to save results",
    )
    parser.add_argument(
        "--save_activations",
        action="store_true",
        help="Save extracted activations for reuse",
    )
    parser.add_argument(
        "--load_activations",
        action="store_true",
        help="Load pre-extracted activations if available",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Jailbreak Representational Independence Analysis")
    print("(Sequential model loading to avoid OOM)")
    print("=" * 60)

    # Step 1: Load prompts
    print(f"\n[1/7] Loading SaladBench data...")
    harmful_instructions, harmless_instructions = load_saladbench_data(
        args.data_dir, n_samples=args.n_samples
    )
    print(f"  - Harmful prompts: {len(harmful_instructions)}")
    print(f"  - Harmless prompts: {len(harmless_instructions)}")

    model_a_name = os.path.basename(args.model_a_path.rstrip("/"))
    model_b_name = os.path.basename(args.model_b_path.rstrip("/"))

    activations_path_a = os.path.join(
        args.output_dir, f"activations_a_{model_a_name}.pt"
    )
    activations_path_b = os.path.join(
        args.output_dir, f"activations_b_{model_b_name}.pt"
    )
    direction_path_a = os.path.join(args.output_dir, f"direction_a_{model_a_name}.pt")
    direction_path_b = os.path.join(args.output_dir, f"direction_b_{model_b_name}.pt")
    metadata_path_a = os.path.join(args.output_dir, f"metadata_a_{model_a_name}.json")
    metadata_path_b = os.path.join(args.output_dir, f"metadata_b_{model_b_name}.json")

    # Step 2: Model A - load, extract, unload
    print(f"\n[2/7] Loading and processing Model A ({model_a_name})...")

    if (
        args.load_activations
        and os.path.exists(activations_path_a)
        and os.path.exists(metadata_path_a)
    ):
        print(f"  Loading pre-extracted activations from {activations_path_a}")
        activations_a = torch.load(activations_path_a, map_location="cpu")
        metadata_a = json.load(open(metadata_path_a))
        n_layers_a = metadata_a["n_layers"]
        best_layer_a = metadata_a["best_layer"]
        direction_a = torch.load(direction_path_a, map_location="cpu")
        print(
            f"  - Loaded activations, n_layers={n_layers_a}, best_layer={best_layer_a}"
        )
    else:
        print(f"  Loading model from {args.model_a_path}...")
        model_a, tokenizer_a = load_model_and_tokenizer(args.model_a_path)
        model_a.eval()
        n_layers_a = model_a.config.num_hidden_layers
        print(f"  - Model A layers: {n_layers_a}")

        prompts_a = apply_chat_template(
            tokenizer_a, harmful_instructions, args.model_a_path
        )

        print(f"  Extracting activations from Model A...")
        activations_a = get_residual_stream_activations(
            model_a, tokenizer_a, prompts_a, batch_size=8
        )

        best_layer_a = args.best_layer_a if args.best_layer_a else int(0.7 * n_layers_a)
        print(f"  Computing DIM direction from layer {best_layer_a}...")
        direction_a = compute_mean_diff_direction(
            activations_a, activations_a, best_layer_a
        )

        print(f"  Saving activations and direction...")
        torch.save(activations_a, activations_path_a)
        torch.save(direction_a, direction_path_a)
        metadata_a = {"n_layers": n_layers_a, "best_layer": best_layer_a}
        with open(metadata_path_a, "w") as f:
            json.dump(metadata_a, f)
        print(f"  - Saved to {args.output_dir}")

        print(f"  Unloading Model A from GPU memory...")
        unload_model(model_a, tokenizer_a)

    # Step 3: Model B - load, extract, unload
    print(f"\n[3/7] Loading and processing Model B ({model_b_name})...")

    if (
        args.load_activations
        and os.path.exists(activations_path_b)
        and os.path.exists(metadata_path_b)
    ):
        print(f"  Loading pre-extracted activations from {activations_path_b}")
        activations_b = torch.load(activations_path_b, map_location="cpu")
        metadata_b = json.load(open(metadata_path_b))
        n_layers_b = metadata_b["n_layers"]
        best_layer_b = metadata_b["best_layer"]
        direction_b = torch.load(direction_path_b, map_location="cpu")
        print(
            f"  - Loaded activations, n_layers={n_layers_b}, best_layer={best_layer_b}"
        )
    else:
        print(f"  Loading model from {args.model_b_path}...")
        model_b, tokenizer_b = load_model_and_tokenizer(args.model_b_path)
        model_b.eval()
        n_layers_b = model_b.config.num_hidden_layers
        print(f"  - Model B layers: {n_layers_b}")

        prompts_b = apply_chat_template(
            tokenizer_b, harmful_instructions, args.model_b_path
        )

        print(f"  Extracting activations from Model B...")
        activations_b = get_residual_stream_activations(
            model_b, tokenizer_b, prompts_b, batch_size=8
        )

        best_layer_b = args.best_layer_b if args.best_layer_b else int(0.7 * n_layers_b)
        print(f"  Computing DIM direction from layer {best_layer_b}...")
        direction_b = compute_mean_diff_direction(
            activations_b, activations_b, best_layer_b
        )

        print(f"  Saving activations and direction...")
        torch.save(activations_b, activations_path_b)
        torch.save(direction_b, direction_path_b)
        metadata_b = {"n_layers": n_layers_b, "best_layer": best_layer_b}
        with open(metadata_path_b, "w") as f:
            json.dump(metadata_b, f)
        print(f"  - Saved to {args.output_dir}")

        print(f"  Unloading Model B from GPU memory...")
        unload_model(model_b, tokenizer_b)

    # Step 4: Compute RepInd statistics (CPU only now)
    print(f"\n[4/7] Computing Representational Independence statistics...")

    repind_layers = (0, int(min(n_layers_a, n_layers_b) * args.layer_cutoff))

    repind_results = compute_repind_statistic(
        activations_a,
        activations_b,
        direction_a,
        direction_b,
        layer_range=repind_layers,
    )

    direction_similarity = compute_direction_similarity(direction_a, direction_b)

    layer_wise_cos = compute_layer_wise_cosine_similarities(
        activations_a,
        activations_b,
        direction_a,
        direction_b,
    )

    cross_cos = compute_cross_model_cosine_sims(
        activations_a,
        activations_b,
        direction_a,
        direction_b,
    )

    results = {
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "model_a_path": args.model_a_path,
        "model_b_path": args.model_b_path,
        "best_layer_a": best_layer_a,
        "best_layer_b": best_layer_b,
        "n_layers_a": n_layers_a,
        "n_layers_b": n_layers_b,
        "layer_range": repind_layers,
        "n_samples": len(harmful_instructions),
        "repind_mse": repind_results["repind_mse"],
        "repind_mae": repind_results["repind_mae"],
        "within_a_mse": repind_results["within_a_mse"],
        "within_b_mse": repind_results["within_b_mse"],
        "layer_diffs": repind_results["layer_diffs"],
        "cos_a_on_a": repind_results["cos_a_on_a"],
        "cos_a_on_b": repind_results["cos_a_on_b"],
        "cos_b_on_b": repind_results["cos_b_on_b"],
        "cos_b_on_a": repind_results["cos_b_on_a"],
        "direction_similarity": direction_similarity,
        "model_a_cosine_sims": layer_wise_cos["model_a_cosine_sims"].tolist(),
        "model_b_cosine_sims": layer_wise_cos["model_b_cosine_sims"].tolist(),
        "cross_direction_a_on_b": cross_cos["direction_a_on_model_b"].tolist(),
        "cross_direction_b_on_a": cross_cos["direction_b_on_model_a"].tolist(),
    }

    print(f"\n[5/7] Saving results...")

    results_path = os.path.join(args.output_dir, "repind_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  - Results saved to {results_path}")

    print(f"\n[6/7] Creating visualization...")
    viz_path = os.path.join(args.output_dir, "repind_visualization.png")
    plot_repind_analysis(results, viz_path, model_a_name, model_b_name)

    print(f"\n[7/7] Cleanup...")
    del activations_a, activations_b, direction_a, direction_b
    gc.collect()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"RepInd MSE: {results['repind_mse']:.6f}")
    print(f"RepInd MAE: {results['repind_mae']:.6f}")
    print(
        f"Direction Cosine Similarity: {results['direction_similarity']['cosine_similarity']:.6f}"
    )
    print(
        f"Direction Orthogonality: {results['direction_similarity']['orthogonality']:.6f}"
    )
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
