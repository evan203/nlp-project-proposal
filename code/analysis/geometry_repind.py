#!/usr/bin/env python3
"""Compute Geometry-paper-style RepInd profiles for saved refusal directions.

This script does not train RCO vectors. It evaluates the representational
independence test from the Geometry paper:

1. Measure each direction's layerwise cosine profile on harmful prompts.
2. Ablate another direction.
3. Re-measure the first direction's layerwise cosine profile.
4. Summarize how much the profile changed.

If no external direction JSON is supplied, the script derives a small local
"cone basis" from high-norm DIM candidate directions so the analysis can run
without W&B artifacts. When trained RDO/RCO/RepInd vectors are available, pass
them with --directions_json and the same code evaluates those directions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
DIM_DIR = CODE_ROOT / "methods" / "dim"
sys.path.insert(0, str(DIM_DIR))

from dataset.load_dataset import load_dataset_split  # noqa: E402
from pipeline.model_utils.model_factory import construct_model_base  # noqa: E402
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks  # noqa: E402


DEFAULT_RUN = CODE_ROOT / "methods/dim/pipeline/runs/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT = CODE_ROOT / "results/geometry_repind"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RepInd cosine-profile comparisons.")
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dim_run_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--directions_json", type=Path, default=None)
    parser.add_argument("--n_prompts", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--derived_cone_dim", type=int, default=3)
    parser.add_argument("--max_candidate_scan", type=int, default=64)
    return parser.parse_args()


def normalize(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.detach().float().cpu().flatten()
    return vector / vector.norm().clamp_min(1e-12)


def orthogonalize(vector: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    out = normalize(vector)
    for base in basis:
        base = normalize(base)
        out = out - torch.dot(out, base) * base
    return normalize(out)


def load_tensor_directions(path: Path, name: str) -> list[dict]:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(tensor, dict):
        raise TypeError(f"{path} contains a dict; save a tensor or list of tensors instead")
    if isinstance(tensor, (list, tuple)):
        tensor = torch.stack([torch.as_tensor(x) for x in tensor])
    tensor = torch.as_tensor(tensor).float()

    if tensor.ndim == 1:
        return [{"name": name, "vector": normalize(tensor), "source": str(path)}]

    flat = tensor.reshape(-1, tensor.shape[-1])
    directions = []
    for idx, vector in enumerate(flat):
        directions.append({"name": f"{name}_{idx + 1}", "vector": normalize(vector), "source": str(path)})
    return directions


def load_external_directions(path: Path) -> list[dict]:
    spec = json.loads(path.read_text())
    directions = []
    for item in spec:
        loaded = load_tensor_directions(Path(item["path"]), item["name"])
        if item.get("index") is not None:
            loaded = [loaded[int(item["index"])]]
            loaded[0]["name"] = item["name"]
        directions.extend(loaded)
    return directions


def derive_dim_geometry_directions(dim_run_dir: Path, cone_dim: int, max_candidate_scan: int) -> list[dict]:
    dim_direction = normalize(torch.load(dim_run_dir / "direction.pt", map_location="cpu", weights_only=True))
    mean_diffs = torch.load(dim_run_dir / "generate_directions/mean_diffs.pt", map_location="cpu", weights_only=True).float()
    candidates = mean_diffs.reshape(-1, mean_diffs.shape[-1])
    norms = candidates.norm(dim=-1)
    top_indices = torch.argsort(norms, descending=True)[:max_candidate_scan].tolist()

    basis = [dim_direction]
    directions = [{"name": "DIM", "vector": dim_direction, "source": str(dim_run_dir / "direction.pt")}]

    for idx in top_indices:
        candidate = orthogonalize(candidates[idx], basis)
        if candidate.norm().item() < 1e-6:
            continue
        basis.append(candidate)
        directions.append(
            {
                "name": f"DIM-cone-basis-{len(directions)}",
                "vector": candidate,
                "source": f"{dim_run_dir}/generate_directions/mean_diffs.pt[{idx}]",
            }
        )
        if len(directions) >= cone_dim:
            break

    return directions


def collect_block_inputs(model_base, prompts: list[str], fwd_pre_hooks, fwd_hooks, batch_size: int) -> torch.Tensor:
    n_layers = model_base.model.config.num_hidden_layers
    hidden = model_base.model.config.hidden_size
    collected: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    def make_hook(layer: int):
        def hook_fn(module, inputs):
            activation = inputs[0].detach()
            collected[layer].append(activation[:, -1, :].float().cpu())

        return hook_fn

    collection_hooks = [(model_base.model_block_modules[layer], make_hook(layer)) for layer in range(n_layers)]
    all_pre_hooks = [*fwd_pre_hooks, *collection_hooks]

    for start in tqdm(range(0, len(prompts), batch_size), desc="cosine profiles"):
        batch = prompts[start : start + batch_size]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with torch.no_grad(), add_hooks(module_forward_pre_hooks=all_pre_hooks, module_forward_hooks=fwd_hooks):
            model_base.model(
                input_ids=inputs.input_ids.to(model_base.model.device),
                attention_mask=inputs.attention_mask.to(model_base.model.device),
            )

    layers = []
    for layer_cache in collected:
        if not layer_cache:
            layers.append(torch.empty((0, hidden)))
        else:
            layers.append(torch.cat(layer_cache, dim=0))
    return torch.stack(layers, dim=1)


def cosine_profiles(activations: torch.Tensor, directions: list[dict]) -> dict[str, list[float]]:
    out = {}
    for item in directions:
        vector = item["vector"].to(activations)
        scores = torch.nn.functional.cosine_similarity(activations, vector[None, None, :], dim=-1)
        out[item["name"]] = scores.mean(dim=0).cpu().tolist()
    return out


def summarize_changes(baseline_profiles: dict, ablated_profiles: dict) -> dict:
    summary = {}
    for intervention, measured_profiles in ablated_profiles.items():
        summary[intervention] = {}
        for measured_name, profile in measured_profiles.items():
            baseline = np.array(baseline_profiles[measured_name])
            changed = np.array(profile)
            delta = changed - baseline
            summary[intervention][measured_name] = {
                "mean_abs_change": float(np.mean(np.abs(delta))),
                "max_abs_change": float(np.max(np.abs(delta))),
                "signed_mean_change": float(np.mean(delta)),
            }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    if args.directions_json:
        directions = load_external_directions(args.directions_json)
    else:
        directions = derive_dim_geometry_directions(args.dim_run_dir, args.derived_cone_dim, args.max_candidate_scan)

    if len(directions) < 2:
        raise ValueError("Need at least two directions for RepInd comparisons")

    for item in directions:
        item["vector"] = normalize(item["vector"])

    print("Directions:")
    for item in directions:
        print(f"  {item['name']}: {item['source']}")

    model_base = construct_model_base(args.model_path)
    harmful = load_dataset_split("harmful", "test", instructions_only=True)
    prompts = random.sample(harmful, min(args.n_prompts, len(harmful)))

    baseline_activations = collect_block_inputs(model_base, prompts, [], [], args.batch_size)
    baseline_profiles = cosine_profiles(baseline_activations, directions)

    ablated_profiles = {}
    for intervention in directions:
        print(f"Ablating {intervention['name']}")
        fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, intervention["vector"].to(model_base.model.device))
        activations = collect_block_inputs(model_base, prompts, fwd_pre_hooks, fwd_hooks, args.batch_size)
        ablated_profiles[intervention["name"]] = cosine_profiles(activations, directions)

    summary = summarize_changes(baseline_profiles, ablated_profiles)

    serializable_directions = [
        {"name": item["name"], "source": item["source"], "norm": float(item["vector"].norm().item())}
        for item in directions
    ]
    results = {
        "model_path": args.model_path,
        "n_prompts": len(prompts),
        "directions": serializable_directions,
        "baseline_profiles": baseline_profiles,
        "ablated_profiles": ablated_profiles,
        "summary": summary,
        "interpretation": "Lower mean_abs_change means the measured direction is more representationally independent of the ablated direction.",
    }

    (args.output_dir / "geometry_repind_results.json").write_text(json.dumps(results, indent=2))
    torch.save({item["name"]: item["vector"] for item in directions}, args.output_dir / "geometry_directions.pt")
    print(f"Saved RepInd results to {args.output_dir}")


if __name__ == "__main__":
    main()
