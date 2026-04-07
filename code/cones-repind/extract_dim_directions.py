#!/usr/bin/env python
"""
Extract DIM directions from pre-saved jailbroken models.

This script extracts Difference-in-Means (DIM) refusal directions from 
jailbroken models that have already been saved to disk.

Usage:
    python extract_dim_directions.py \
        --model_path /path/to/jailbroken/model \
        --prompts_file data/saladbench_splits/harmful_train.json \
        --output_dir results/directions \
        --n_samples 500
"""

import argparse
import json
import os
from typing import List, Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm

from transformer_utils import (
    load_model_and_tokenizer,
    apply_chat_template,
    get_residual_stream_activations,
    compute_dim_directions,
    get_model_config,
)


def load_prompts_from_file(prompts_file: str, n_samples: int = None) -> List[str]:
    """Load prompts from JSON file."""
    with open(prompts_file, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            instructions = [d.get("instruction", d.get("prompt", str(d))) for d in data]
        else:
            instructions = data
    else:
        instructions = [data.get("instruction", data.get("prompt", str(data)))]

    if n_samples is not None:
        instructions = instructions[:n_samples]

    return instructions


def find_best_layer_for_direction(
    harmful_activations: List[Dict[int, torch.Tensor]],
    harmless_activations: List[Dict[int, torch.Tensor]],
    n_layers: int,
    layer_range: Tuple[int, int] = (None, None),
) -> Tuple[int, Dict]:
    """
    Find the layer that gives the best separation between harmful and harmless.

    Returns the layer index with maximum mean difference magnitude.
    """
    start = layer_range[0] if layer_range[0] is not None else 0
    end = layer_range[1] if layer_range[1] is not None else n_layers

    layer_scores = []

    for layer_idx in range(start, end):
        harmful_layer_acts = torch.stack(
            [act[layer_idx] for act in harmful_activations]
        )
        harmless_layer_acts = torch.stack(
            [act[layer_idx] for act in harmless_activations]
        )

        harmful_mean = harmful_layer_acts.mean(dim=0)
        harmless_mean = harmless_layer_acts.mean(dim=0)

        diff = (harmful_mean - harmless_mean).norm().item()
        layer_scores.append((layer_idx, diff))

    layer_scores.sort(key=lambda x: x[1], reverse=True)
    best_layer, best_score = layer_scores[0]

    return best_layer, {"best_layer": best_layer, "layer_scores": layer_scores[:10]}


def main():
    parser = argparse.ArgumentParser(
        description="Extract DIM refusal directions from jailbroken models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the jailbroken model",
    )
    parser.add_argument(
        "--harmful_prompts",
        type=str,
        default="data/saladbench_splits/harmful_train.json",
        help="Path to harmful prompts JSON file",
    )
    parser.add_argument(
        "--harmless_prompts",
        type=str,
        default="data/saladbench_splits/harmless_train.json",
        help="Path to harmless prompts JSON file",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/directions",
        help="Directory to save results",
    )
    parser.add_argument(
        "--layer_range",
        type=str,
        default="0.5,1.0",
        help="Layer range as fraction of total layers (e.g., '0.5,1.0')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers")

    print(f"Loading prompts...")
    harmful_instructions = load_prompts_from_file(args.harmful_prompts, args.n_samples)
    harmless_instructions = load_prompts_from_file(
        args.harmless_prompts, args.n_samples
    )
    print(
        f"Loaded {len(harmful_instructions)} harmful and {len(harmless_instructions)} harmless prompts"
    )

    prompts_harmful = apply_chat_template(
        tokenizer, harmful_instructions, args.model_path
    )
    prompts_harmless = apply_chat_template(
        tokenizer, harmless_instructions, args.model_path
    )

    print("Extracting activations for harmful prompts...")
    activations_harmful = get_residual_stream_activations(
        model, tokenizer, prompts_harmful, batch_size=args.batch_size
    )

    print("Extracting activations for harmless prompts...")
    activations_harmless = get_residual_stream_activations(
        model, tokenizer, prompts_harmless, batch_size=args.batch_size
    )

    layer_start = int(n_layers * float(args.layer_range.split(",")[0]))
    layer_end = int(n_layers * float(args.layer_range.split(",")[1]))
    layer_range = (layer_start, layer_end)

    print(f"Finding best layer in range {layer_range}...")
    best_layer, layer_info = find_best_layer_for_direction(
        activations_harmful, activations_harmless, n_layers, layer_range
    )
    print(f"Best layer: {best_layer}")

    print("Computing DIM directions...")
    dim_directions = compute_dim_directions(
        activations_harmful, activations_harmless, layer_range
    )

    best_direction = dim_directions[best_layer]

    print("Saving results...")
    torch.save(
        best_direction, os.path.join(args.output_dir, f"{model_name}_dim_direction.pt")
    )
    torch.save(
        dim_directions,
        os.path.join(args.output_dir, f"{model_name}_all_dim_directions.pt"),
    )

    with open(os.path.join(args.output_dir, f"{model_name}_metadata.json"), "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "model_path": args.model_path,
                "best_layer": best_layer,
                "n_layers": n_layers,
                "n_samples": args.n_samples,
                "layer_range": {"start": layer_start, "end": layer_end},
                "top_layers_by_separation": layer_info["layer_scores"],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {args.output_dir}/")
    print(f"  - Best direction: {model_name}_dim_direction.pt")
    print(f"  - All directions: {model_name}_all_dim_directions.pt")
    print(f"  - Metadata: {model_name}_metadata.json")


if __name__ == "__main__":
    main()
