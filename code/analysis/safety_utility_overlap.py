#!/usr/bin/env python3
"""Compute and plot direct safety-vs-utility activation-subspace overlap.

Safety subspace:
    Per-layer harmful-minus-harmless mean activation differences from the DIM
    pipeline, averaged across end-of-instruction positions.

Utility subspace:
    Per-layer PCA bases of harmless instruction activations at the same
    end-of-instruction positions.

The main metric is 1-D-vs-k MSO: the fraction of each safety direction's energy
that lies in the top-k utility activation subspace. The random baseline is k/d.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DIM_DIR = CODE_ROOT / "methods" / "dim"
sys.path.insert(0, str(DIM_DIR))

from dataset.load_dataset import load_dataset, load_dataset_split  # noqa: E402
from pipeline.model_utils.model_factory import construct_model_base  # noqa: E402
from pipeline.submodules.generate_directions import get_mean_diff  # noqa: E402
from pipeline.utils.hook_utils import add_hooks  # noqa: E402


def parse_ranks(value: str) -> list[int]:
    ranks = sorted({int(x) for x in value.split(",") if x.strip()})
    if not ranks or min(ranks) < 1:
        raise argparse.ArgumentTypeError("ranks must be positive comma-separated integers")
    return ranks


def get_activation_collection_pre_hook(layer: int, cache: list[list[torch.Tensor]], positions: list[int]):
    def hook_fn(module, inputs):
        activation = inputs[0].detach()
        selected = activation[:, positions, :].reshape(-1, activation.shape[-1])
        cache[layer].append(selected.float().cpu())

    return hook_fn


def collect_utility_activations(
    model_base,
    instructions: list[str],
    positions: list[int],
    batch_size: int,
) -> list[torch.Tensor]:
    n_layers = model_base.model.config.num_hidden_layers
    cache: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    hooks = [
        (
            model_base.model_block_modules[layer],
            get_activation_collection_pre_hook(layer, cache, positions),
        )
        for layer in range(n_layers)
    ]

    for start in tqdm(range(0, len(instructions), batch_size), desc="utility activations"):
        batch = instructions[start : start + batch_size]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with torch.no_grad(), add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
            model_base.model(
                input_ids=inputs.input_ids.to(model_base.model.device),
                attention_mask=inputs.attention_mask.to(model_base.model.device),
            )

    return [torch.cat(layer_cache, dim=0) for layer_cache in cache]


def load_utility_instructions(dataset_name: str, n_samples: int, seed: int) -> list[str]:
    random.seed(seed)

    if dataset_name == "harmless_train":
        instructions = load_dataset_split("harmless", "train", instructions_only=True)
    elif dataset_name == "harmless_test":
        instructions = load_dataset_split("harmless", "test", instructions_only=True)
    else:
        dataset = load_dataset(dataset_name, instructions_only=True)
        instructions = dataset

    if n_samples > len(instructions):
        raise ValueError(f"Requested {n_samples} samples, but {dataset_name} only has {len(instructions)}")
    return random.sample(instructions, n_samples)


def load_or_compute_safety_mean_diffs(args, model_base, positions: list[int]) -> torch.Tensor:
    if args.safety_mean_diffs_path and args.safety_mean_diffs_path.exists():
        print(f"Loading safety mean diffs from {args.safety_mean_diffs_path}")
        return torch.load(args.safety_mean_diffs_path, map_location="cpu", weights_only=True).float()

    print("Computing safety mean diffs from harmful/harmless train splits")
    random.seed(args.seed)
    harmful = random.sample(
        load_dataset_split("harmful", "train", instructions_only=True),
        args.n_safety_samples,
    )
    harmless = random.sample(
        load_dataset_split("harmless", "train", instructions_only=True),
        args.n_safety_samples,
    )
    mean_diffs = get_mean_diff(
        model_base.model,
        model_base.tokenizer,
        harmful,
        harmless,
        model_base.tokenize_instructions_fn,
        model_base.model_block_modules,
        batch_size=args.batch_size,
        positions=positions,
    )
    return mean_diffs.detach().float().cpu()


def compute_layer_bases(activations_by_layer: list[torch.Tensor], max_rank: int) -> torch.Tensor:
    bases = []
    for layer, activations in enumerate(tqdm(activations_by_layer, desc="utility PCA")):
        centered = activations - activations.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = min(max_rank, vh.shape[0])
        basis = vh[:rank].T.contiguous()
        bases.append(basis)
        print(f"layer {layer:02d}: utility activation matrix={tuple(activations.shape)}, basis_rank={rank}")
    return torch.stack(bases, dim=0)


def projection_scores(safety_vectors: torch.Tensor, utility_bases: torch.Tensor, ranks: list[int]) -> dict:
    n_layers, d_model = safety_vectors.shape
    max_rank = utility_bases.shape[-1]
    rank_rows = {}

    safety_unit = safety_vectors / safety_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    for rank in ranks:
        rank = min(rank, max_rank)
        scores = []
        for layer in range(n_layers):
            basis = utility_bases[layer, :, :rank]
            score = torch.square(basis.T @ safety_unit[layer]).sum().item()
            scores.append(score)
        rank_rows[str(rank)] = {
            "mso": scores,
            "random_baseline": rank / d_model,
            "mean_mso": float(np.mean(scores)),
            "mean_over_random": float(np.mean(scores) / (rank / d_model)),
        }

    return rank_rows


def load_named_direction(spec: str, n_layers: int) -> tuple[str, torch.Tensor, str]:
    """Load a named direction and tag it with its kind.

    Returns (name, tensor, kind) where `kind` is one of:
      - "single": 1-D `[d_model]` direction (e.g. DIM selected vector)
      - "per_layer": 2-D `[n_layers, d_model]` per-layer directions
        (e.g. ActSVD activation delta, where row i is the layer-i direction)
      - "subspace": 2-D `[k, d_model]` k-direction subspace basis
        (e.g. RCO 2-D refusal cone, where row j is the j-th cone basis vector)

    Disambiguation between per_layer and subspace is by first dim:
    if it equals `n_layers` we treat it as per-layer; otherwise as a
    subspace basis. If the file name contains `cone` or `rco`, we force
    the subspace interpretation even when the dims happen to coincide.
    """
    name, path_str = spec.split(":", 1)
    path = Path(path_str)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    if isinstance(raw, list):
        raw = torch.stack([torch.as_tensor(x) for x in raw])
    direction = torch.as_tensor(raw).float()

    if direction.ndim == 1:
        norm = direction.norm()
        if norm < 1e-12:
            raise ValueError(f"Direction from {path} has near-zero norm.")
        return name, direction / norm, "single"

    if direction.ndim == 2:
        norms = direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
        normed = direction / norms
        # File-name hint forces the subspace interpretation.
        path_hint = path.name.lower() + name.lower()
        is_cone_by_name = ("cone" in path_hint) or ("rco" in path_hint) or ("rdo" in path_hint)
        if direction.shape[0] == n_layers and not is_cone_by_name:
            kind = "per_layer"
        else:
            kind = "subspace"
        return name, normed, kind

    raise ValueError(f"Direction from {path} has unexpected shape {tuple(direction.shape)}.")


def named_direction_overlap(
    direction: torch.Tensor,
    utility_bases: torch.Tensor,
    ranks: list[int],
    kind: str,
) -> dict:
    """Compute per-layer MSO between a named direction and the utility PCA bases.

    - kind="single": single 1-D direction projected onto each layer's utility basis.
        score(layer, k) = ||U_k^T s||^2 in [0, 1]. Random baseline k/d.
    - kind="per_layer": per-layer 1-D directions, one row per layer, each
        projected onto its own layer's utility basis. Random baseline k/d.
    - kind="subspace": k_S-dim subspace (orthonormalized via QR), projected
        onto each layer's rank-k_U utility basis using normalized subspace MSO:
            MSO = ||U_k^T B||_F^2 / min(k_S, k_U)   (in [0, 1])
        Random baseline = max(k_S, k_U) / d (expected normalized MSO of two
        random orthonormal subspaces).
    """
    n_layers = utility_bases.shape[0]
    d = utility_bases.shape[1]
    max_rank = utility_bases.shape[-1]

    # Pre-compute orthonormal column basis for the subspace case.
    Q_subspace = None
    k_S = 0
    if kind == "subspace":
        # `direction` is [k, d]; columns of Q span the subspace.
        cols = direction.t().contiguous()  # [d, k]
        Q_subspace, _ = torch.linalg.qr(cols, mode="reduced")  # [d, k]
        k_S = Q_subspace.shape[1]

    result = {}
    for rank in ranks:
        r = min(rank, max_rank)
        scores = []
        for layer in range(n_layers):
            basis = utility_bases[layer, :, :r]  # [d, r]
            if kind == "subspace":
                proj = basis.t() @ Q_subspace  # [r, k_S]
                num = float(torch.square(proj).sum().item())
                denom = float(min(k_S, r))
                score = num / denom if denom > 0 else 0.0
            elif kind == "per_layer":
                dir_l = direction[layer] if layer < direction.shape[0] else direction[-1]
                score = float(torch.square(basis.t() @ dir_l).sum().item())
            else:  # single
                score = float(torch.square(basis.t() @ direction).sum().item())
            scores.append(score)
        if kind == "subspace":
            random_baseline = float(max(k_S, r)) / float(d)
        else:
            random_baseline = float(r) / float(d)
        result[str(rank)] = {
            "mso": scores,
            "random_baseline": random_baseline,
            "kind": kind,
        }
        if kind == "subspace":
            result[str(rank)]["k_subspace"] = k_S
    return result


def maybe_selected_dim_overlap(args, utility_bases: torch.Tensor, ranks: list[int]) -> dict | None:
    if not args.dim_direction_path or not args.dim_metadata_path:
        return None
    if not args.dim_direction_path.exists() or not args.dim_metadata_path.exists():
        return None

    direction = torch.load(args.dim_direction_path, map_location="cpu", weights_only=True).float()
    metadata = json.loads(args.dim_metadata_path.read_text())
    layer = int(metadata["layer"])
    direction = direction / direction.norm().clamp_min(1e-12)

    out = {"layer": layer, "rank_scores": {}}
    for rank in ranks:
        rank = min(rank, utility_bases.shape[-1])
        basis = utility_bases[layer, :, :rank]
        out["rank_scores"][str(rank)] = float(torch.square(basis.T @ direction).sum().item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute direct safety-vs-utility activation overlap.")
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=Path, default=CODE_ROOT / "results" / "safety_utility_overlap")
    parser.add_argument("--utility_dataset", default="harmless_train")
    parser.add_argument("--n_utility_samples", type=int, default=128)
    parser.add_argument("--n_safety_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positions", type=str, default=None, help="Comma-separated positions. Default: model EOI token positions.")
    parser.add_argument("--utility_ranks", type=parse_ranks, default=parse_ranks("1,2,4,8,16,32"))
    parser.add_argument("--primary_rank", type=int, default=8)
    parser.add_argument("--save_tensors", action="store_true")
    parser.add_argument(
        "--safety_mean_diffs_path",
        type=Path,
        default=CODE_ROOT / "methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/generate_directions/mean_diffs.pt",
    )
    parser.add_argument(
        "--dim_direction_path",
        type=Path,
        default=CODE_ROOT / "methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/direction.pt",
    )
    parser.add_argument(
        "--dim_metadata_path",
        type=Path,
        default=CODE_ROOT / "methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/direction_metadata.json",
    )
    parser.add_argument(
        "--extra_directions",
        nargs="*",
        default=[],
        help="Extra directions as NAME:PATH pairs (e.g. RCO:results/geometry_repind/rco_direction.pt). Each gets a per-layer overlap curve.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.primary_rank not in args.utility_ranks:
        args.utility_ranks = sorted({*args.utility_ranks, args.primary_rank})

    print(f"Loading model: {args.model_path}")
    model_base = construct_model_base(args.model_path)
    positions = (
        [int(x) for x in args.positions.split(",")]
        if args.positions
        else list(range(-len(model_base.eoi_toks), 0))
    )
    print(f"Using positions: {positions}")

    safety_mean_diffs = load_or_compute_safety_mean_diffs(args, model_base, positions)
    safety_vectors = safety_mean_diffs.mean(dim=0).float()

    utility_instructions = load_utility_instructions(args.utility_dataset, args.n_utility_samples, args.seed)
    utility_activations = collect_utility_activations(
        model_base,
        utility_instructions,
        positions=positions,
        batch_size=args.batch_size,
    )
    utility_bases = compute_layer_bases(utility_activations, max(args.utility_ranks))

    rank_results = projection_scores(safety_vectors, utility_bases, args.utility_ranks)
    selected_dim = maybe_selected_dim_overlap(args, utility_bases, args.utility_ranks)

    n_layers_total = utility_bases.shape[0]
    named_directions: dict = {}
    for spec in args.extra_directions:
        try:
            name, direction, kind = load_named_direction(spec, n_layers=n_layers_total)
            entry = named_direction_overlap(direction, utility_bases, args.utility_ranks, kind=kind)
            entry = {"kind": kind, **entry}
            named_directions[name] = entry
            print(f"Computed named direction overlap: {name} (kind={kind}, shape={tuple(direction.shape)})")
        except Exception as e:
            print(f"Warning: skipping extra direction {spec!r}: {e}")

    results = {
        "model_path": args.model_path,
        "utility_dataset": args.utility_dataset,
        "n_utility_samples": args.n_utility_samples,
        "n_safety_samples": args.n_safety_samples,
        "positions": positions,
        "safety_definition": "harmful-minus-harmless DIM mean difference, averaged over positions",
        "utility_definition": "top PCA directions of harmless instruction activations at matching positions",
        "rank_results": rank_results,
        "selected_dim_direction": selected_dim,
        "named_directions": named_directions,
    }

    (args.output_dir / "safety_utility_overlap_results.json").write_text(json.dumps(results, indent=2))
    if args.save_tensors:
        torch.save(safety_vectors, args.output_dir / "safety_vectors_by_layer.pt")
        torch.save(utility_bases, args.output_dir / "utility_pca_bases_by_layer.pt")

    print(f"Saved results to {args.output_dir / 'safety_utility_overlap_results.json'}")


if __name__ == "__main__":
    main()
