#!/usr/bin/env python3
"""Compute MSO between saved direction vectors and the ActSVD weight-delta subspace.

Appends results to comparison_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute MSO between directions and ActSVD weight-delta subspace.")
    p.add_argument("--base_model_path", type=Path, default=None,
                   help="Path to base model safetensors directory. If not given, auto-detected from HF cache.")
    p.add_argument("--actsvd_model_path", type=Path,
                   default=CODE_ROOT / "methods" / "actsvd" / "out")
    p.add_argument("--directions", nargs="*", default=[],
                   help="NAME:PATH pairs of direction vectors.")
    p.add_argument("--output", type=Path,
                   default=CODE_ROOT / "results" / "method_overlap" / "comparison_results.json")
    p.add_argument("--weight_types", type=str,
                   default="self_attn.q_proj,self_attn.o_proj,mlp.down_proj")
    p.add_argument("--svd_threshold", type=float, default=0.01,
                   help="Fraction of max singular value for effective rank cutoff.")
    return p.parse_args()


def load_named_direction(spec: str) -> tuple[str, torch.Tensor]:
    name, path_str = spec.split(":", 1)
    path = Path(path_str)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    if isinstance(raw, list):
        raw = raw[0]
    direction = torch.as_tensor(raw).float()
    if direction.ndim > 1:
        direction = direction[0]
    direction = direction.flatten()
    norm = direction.norm()
    if norm < 1e-12:
        raise ValueError(f"Direction from {path} has near-zero norm.")
    return name, direction / norm


def find_safetensors_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    return files


def load_tensor_from_shards(key: str, shard_files: list[Path]) -> torch.Tensor | None:
    try:
        from safetensors.torch import safe_open
    except ImportError:
        raise ImportError("safetensors library is required. Install with: pip install safetensors")
    for shard in shard_files:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def auto_detect_base_model_path() -> Path | None:
    hf_home = CODE_ROOT / "llm_weights"
    pattern = str(hf_home / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots" / "*")
    snapshots = glob(pattern)
    for snap in sorted(snapshots):
        snap_path = Path(snap)
        if list(snap_path.glob("*.safetensors")):
            return snap_path
    return None


def compute_mso_for_direction(
    direction: torch.Tensor,
    base_shards: list[Path],
    actsvd_shards: list[Path],
    weight_types: list[str],
    svd_threshold: float,
) -> dict[str, dict]:
    results = {}
    # Detect number of layers from base model index or by iterating
    # Find all layer indices by scanning shard keys for one weight type
    from safetensors.torch import safe_open
    all_keys: set[str] = set()
    for shard in base_shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            all_keys.update(f.keys())

    layer_indices: set[int] = set()
    for key in all_keys:
        if key.startswith("model.layers.") and key.endswith(".weight"):
            parts = key.split(".")
            try:
                layer_indices.add(int(parts[2]))
            except (IndexError, ValueError):
                pass

    for layer_idx in sorted(layer_indices):
        for wtype in weight_types:
            tensor_key = f"model.layers.{layer_idx}.{wtype}.weight"
            w_base = load_tensor_from_shards(tensor_key, base_shards)
            w_actsvd = load_tensor_from_shards(tensor_key, actsvd_shards)
            if w_base is None or w_actsvd is None:
                continue
            w_base = w_base.float()
            w_actsvd = w_actsvd.float()
            delta_w = w_actsvd - w_base  # [out, in]

            # SVD of delta_W
            try:
                U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
            except Exception as e:
                print(f"  SVD failed for {tensor_key}: {e}")
                continue

            threshold = svd_threshold * S[0].item()
            mask = S > threshold
            k_B = int(mask.sum().item())
            if k_B == 0:
                k_B = 1
                mask[0] = True

            U_B = U[:, mask]  # [out, k_B]
            d = direction.shape[0]

            # Project direction onto U_B columns (U_B has shape [out, k_B])
            # direction has shape [d]; if d != out, skip
            if d != U_B.shape[0]:
                print(f"  Skipping {tensor_key}: direction dim {d} != weight out dim {U_B.shape[0]}")
                continue

            mso = float(torch.square(U_B.T @ direction).sum().item())
            random_baseline = k_B / d

            results[tensor_key] = {
                "mso": mso,
                "k_A": 1,
                "k_B": k_B,
                "d": d,
                "random_baseline": random_baseline,
            }
            print(f"  {tensor_key}: MSO={mso:.4f}, k_B={k_B}, baseline={random_baseline:.4f}")

    return results


def main() -> None:
    args = parse_args()

    weight_types = [w.strip() for w in args.weight_types.split(",")]

    # Load directions
    named_directions: list[tuple[str, torch.Tensor]] = []
    for spec in args.directions:
        try:
            name, direction = load_named_direction(spec)
            named_directions.append((name, direction))
            print(f"Loaded direction: {name}, shape={direction.shape}")
        except Exception as e:
            print(f"Warning: skipping direction {spec!r}: {e}")

    # Load or initialize output JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if args.output.exists():
        data = json.loads(args.output.read_text())
    data.setdefault("mso", {})
    data.setdefault("direction_cosine", {})

    # Compute direction-pair cosine similarities
    if len(named_directions) >= 2:
        print("\nComputing direction cosine similarities...")
        for i in range(len(named_directions)):
            for j in range(i + 1, len(named_directions)):
                name_i, dir_i = named_directions[i]
                name_j, dir_j = named_directions[j]
                if dir_i.shape != dir_j.shape:
                    print(f"  Skipping {name_i} vs {name_j}: shape mismatch {dir_i.shape} vs {dir_j.shape}")
                    continue
                cos_sim = float(torch.dot(dir_i, dir_j).item())
                pair_key = f"{name_i}_vs_{name_j}"
                data["direction_cosine"][pair_key] = cos_sim
                print(f"  {pair_key}: cosine={cos_sim:.4f}")

    # Compute MSO for each direction vs ActSVD
    actsvd_path = args.actsvd_model_path
    if not actsvd_path.exists():
        print(f"Warning: ActSVD model path does not exist: {actsvd_path}")
        print("Skipping MSO computation. Only direction cosine similarities computed.")
    elif not named_directions:
        print("No valid directions loaded. Nothing to compute.")
    else:
        # Resolve base model path
        base_model_path = args.base_model_path
        if base_model_path is None:
            base_model_path = auto_detect_base_model_path()
            if base_model_path is None:
                print("Error: could not auto-detect base model path. Use --base_model_path.")
                sys.exit(1)
            print(f"Auto-detected base model path: {base_model_path}")

        try:
            base_shards = find_safetensors_files(base_model_path)
            actsvd_shards = find_safetensors_files(actsvd_path)
            print(f"Base model: {len(base_shards)} shard(s) in {base_model_path}")
            print(f"ActSVD model: {len(actsvd_shards)} shard(s) in {actsvd_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        for name, direction in named_directions:
            print(f"\nComputing MSO for direction: {name}")
            mso_results = compute_mso_for_direction(
                direction, base_shards, actsvd_shards, weight_types, args.svd_threshold
            )
            pair_key = f"{name}_vs_ActSVD_per_layer"
            data["mso"][pair_key] = mso_results
            print(f"Stored {len(mso_results)} weight matrix results under mso[{pair_key!r}]")

    args.output.write_text(json.dumps(data, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
