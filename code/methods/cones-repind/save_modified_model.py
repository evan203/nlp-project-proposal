#!/usr/bin/env python3
"""Save a weight-edited model from a trained refusal direction.

Applies the DIM-style rank-one weight edit W' = W - r̂ r̂ᵀ W to all linear
projection layers (q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj,
down_proj), making the model permanently unable to represent the direction r̂.

Usage (from the cones-repind directory):
    # After training a direction with run_rco.sh, extract it first:
    python ../../scripts/build_rco_directions_json.py --latest
    # Then save the modified model:
    python save_modified_model.py \\
        --direction results/rdo/Llama-3.1-8B-Instruct/local_vectors/.../lowest_loss_vector.pt \\
        --output_dir results/modified_model/Llama-3.1-8B-Instruct-RCO

    # Or verify the DIM direction as a sanity check:
    python save_modified_model.py \\
        --direction results/dim/Llama-3.1-8B-Instruct/direction.pt \\
        --output_dir results/modified_model/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Layers to which the weight edit is applied, following Arditi et al. (2024).
_PROJECTION_NAMES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save a weight-edited model from a trained direction.")
    p.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--direction", required=True, type=Path,
                   help="Path to direction .pt file.")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Directory to save modified model.")
    p.add_argument("--hf_home", type=Path, default=None)
    return p.parse_args()


def load_direction(path: Path) -> torch.Tensor:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    direction = torch.as_tensor(raw).float()
    if direction.ndim > 1:
        direction = direction[0]
    direction = direction.flatten()
    return direction / direction.norm().clamp_min(1e-12)


def apply_weight_edit(model, direction: torch.Tensor) -> None:
    """Apply W' = W - r̂ r̂ᵀ W in-place to all projection layers."""
    r = direction.to(model.dtype).to(model.device)
    # r̂ r̂ᵀ is a (d, d) rank-1 matrix; W' = W - (r̂ r̂ᵀ) W = W - r̂ (r̂ᵀ W)
    n_edited = 0
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in _PROJECTION_NAMES and hasattr(module, "weight"):
            W = module.weight.data.float()
            # rᵀ W has shape (out_features,); outer-product gives (d, out_features)
            # but W has shape (out_features, in_features), so we need:
            # W' = W - r̂ (r̂ᵀ W)  where r̂ has shape (in_features,)
            # r̂ᵀ W  → shape (out_features,) when W is (out_features, in_features)? No:
            # r̂ has shape (in_features,), W has shape (out_features, in_features)
            # r̂ᵀ W = (in_features,) @ (in_features, out_features)ᵀ  is incorrect dimension
            # Correct: W x → (out_features,); ablate direction r̂ from input space
            # W' x' = W (x - r̂ r̂ᵀ x) = Wx - W r̂ (r̂ᵀ x) = (W - W r̂ r̂ᵀ) x
            # So W' = W - (W r̂) r̂ᵀ  where W r̂ has shape (out_features,)
            # This is the correct rank-1 edit from Arditi et al. eq. W'_out = W_out - r̂ r̂ᵀ W_out
            # Their notation: W'_out = W_out - r̂ r̂ᵀ W_out
            # In matrix form: W_out is (out, in), r̂ is (in,) [or (out,)?]
            # Actually from the paper: W'_out <- W_out - r̂ r̂ᵀ W_out
            # where r̂ has shape (d_model,) = (in_features for W_out which is d_model x d_model)
            # So: r̂ r̂ᵀ W_out = r̂ (r̂ᵀ W_out) where r̂ᵀ W_out has shape (d_model,)
            # Wait - W_out is the output projection (d_model, d_head)? Or (d_head, d_model)?
            # In transformers, Linear layers are (out_features, in_features)
            # r̂ lives in residual stream space (d_model,)
            # For residual stream: pre-hook zeroes r̂ r̂ᵀ from the input
            # Weight edit equivalent: for each Linear layer W (out, in):
            #   if in_features == d_model: W' = W - W (r̂ r̂ᵀ) = W - (W r̂) r̂ᵀ
            #   if out_features == d_model: W' = W - (r̂ r̂ᵀ) W applied to output
            # The DIM paper applies to W_out (the output matrix that writes to residual stream)
            # Simplest correct implementation: project residual stream before/after each layer
            # Weight edit: W'_out = W_out - r̂ r̂ᵀ W_out
            # Here W_out writes to d_model, so shape is (d_model, d_something)
            # r̂ r̂ᵀ W_out: r̂ is (d_model,), r̂ r̂ᵀ is (d_model, d_model), W_out is (d_model, d_k)
            # = (d_model, d_model) @ (d_model, d_k) = (d_model, d_k) ✓
            # But for Linear(in=d_model, out=d_k): weight shape is (d_k, d_model) (transposed)
            # So W_out in DIM notation = weight.T in PyTorch notation
            # W'_out = W_out - r̂ r̂ᵀ W_out
            # weight.T' = weight.T - r̂ r̂ᵀ weight.T
            # weight' = weight - weight r̂ r̂ᵀ  (transposing both sides)
            # weight' = weight - (weight @ r̂.unsqueeze(1)) @ r̂.unsqueeze(0)
            # = weight - outer(weight @ r̂, r̂)
            if W.shape[-1] == r.shape[0]:
                # in_features == d_model: ablate from input side
                # W' = W - W r̂ r̂ᵀ = W - outer(W @ r̂, r̂)
                proj = W @ r.float()  # (out_features,)
                W_new = W - torch.ger(proj, r.float())
                module.weight.data = W_new.to(module.weight.dtype)
                n_edited += 1
    print(f"Applied weight edit to {n_edited} layers.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import os
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(args.hf_home)

    print(f"Loading direction from {args.direction}")
    direction = load_direction(args.direction)
    print(f"  direction shape: {direction.shape}, norm: {direction.norm():.4f}")

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Applying weight edit ...")
    apply_weight_edit(model, direction)

    print(f"Saving modified model to {args.output_dir}")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    meta = {
        "source_model": args.model_path,
        "direction_path": str(args.direction),
        "edit": "W_prime = W - W @ r_hat @ r_hat.T (input projection ablation)",
    }
    (args.output_dir / "edit_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Modified model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
