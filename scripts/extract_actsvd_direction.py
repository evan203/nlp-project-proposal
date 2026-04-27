#!/usr/bin/env python3
"""Extract a per-layer activation-delta direction from an ActSVD modified model.

Computes mean(activations_actsvd - activations_base) per layer over a sample of
harmless prompts, normalizes each layer's delta, and saves a [n_layers, d_model]
tensor suitable for safety-utility overlap analysis.

Usage:
    python scripts/extract_actsvd_direction.py \
        --base_model_path meta-llama/Llama-3.1-8B-Instruct \
        --modified_model_path code/results/actsvd/out \
        --output_path code/results/actsvd/activation_direction.pt
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = REPO_ROOT / "code"
DIM_DIR = CODE_ROOT / "methods" / "dim"
sys.path.insert(0, str(DIM_DIR))

from dataset.load_dataset import load_dataset_split  # noqa: E402
from pipeline.model_utils.model_factory import construct_model_base  # noqa: E402
from pipeline.utils.hook_utils import add_hooks  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract per-layer activation delta from ActSVD model.")
    p.add_argument("--base_model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--modified_model_path", type=Path, required=True)
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_mean_activations(model_base, instructions: list[str], batch_size: int) -> torch.Tensor:
    """Collect mean activations at EOI positions per layer.

    Returns [n_layers, d_model].
    """
    n_layers = model_base.model.config.num_hidden_layers
    positions = list(range(-len(model_base.eoi_toks), 0))

    cache: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    def make_hook(layer_idx: int):
        def hook_fn(module, inputs):
            activation = inputs[0].detach().float().cpu()
            selected = activation[:, positions, :].mean(dim=1)  # [batch, d]
            cache[layer_idx].append(selected)
        return hook_fn

    hooks = [(model_base.model_block_modules[l], make_hook(l)) for l in range(n_layers)]

    for start in tqdm(range(0, len(instructions), batch_size), desc="collecting activations"):
        batch = instructions[start : start + batch_size]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with torch.no_grad(), add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
            model_base.model(
                input_ids=inputs.input_ids.to(model_base.model.device),
                attention_mask=inputs.attention_mask.to(model_base.model.device),
            )

    return torch.stack([torch.cat(cache[l], dim=0).mean(dim=0) for l in range(n_layers)])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    all_harmless = load_dataset_split("harmless", "train", instructions_only=True)
    sample = random.sample(all_harmless, min(args.n_samples, len(all_harmless)))
    print(f"Using {len(sample)} harmless instructions")

    print(f"Loading base model: {args.base_model_path}")
    base_model = construct_model_base(args.base_model_path)
    base_means = collect_mean_activations(base_model, sample, args.batch_size)
    print(f"Base activations: {base_means.shape}")

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading modified model: {args.modified_model_path}")
    mod_model = construct_model_base(str(args.modified_model_path))
    mod_means = collect_mean_activations(mod_model, sample, args.batch_size)
    print(f"Modified activations: {mod_means.shape}")

    del mod_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    delta = mod_means - base_means  # [n_layers, d_model]
    norms = delta.norm(dim=1, keepdim=True)
    print(
        f"Delta norms per layer  min={norms.min().item():.4e}  "
        f"mean={norms.mean().item():.4e}  max={norms.max().item():.4e}"
    )
    delta_normalized = delta / norms.clamp_min(1e-12)

    torch.save(delta_normalized, args.output_path)
    print(f"Saved per-layer activation direction to {args.output_path}")


if __name__ == "__main__":
    main()
