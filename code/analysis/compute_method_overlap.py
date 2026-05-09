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
    """Load a direction or cone from a .pt file.

    Returns a `[d]` 1-D direction or a `[k, d]` cone basis. Each row is
    unit-norm; the caller orthonormalizes via QR for true subspace MSO.
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
        return name, direction / norm
    if direction.ndim == 2:
        norms = direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return name, direction / norms
    raise ValueError(f"Direction from {path} has unexpected shape {tuple(direction.shape)}.")


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
    """Compute per-(layer, weight) MSO between `direction` and the
    ActSVD weight-delta column space.

    `direction` is `[d]` (1-D direction) or `[k, d]` (k-direction subspace).
    For a 1-D direction, MSO = ||U_B^T s||^2 in [0, 1]; random baseline k_B/d.
    For a k-D subspace, MSO = ||U_B^T Q_S||_F^2 / min(k_S, k_B), where Q_S is
    an orthonormal basis for the subspace; random baseline = max(k_S, k_B)/d.
    """
    is_subspace = direction.ndim == 2
    if is_subspace:
        cols = direction.t().contiguous()  # [d, k_S]
        Q_S, _ = torch.linalg.qr(cols, mode="reduced")  # [d, k_S]
        k_S = Q_S.shape[1]
    else:
        Q_S = None
        k_S = 1
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
        print(f"Layer {layer_idx}/{max(layer_indices)} ...", flush=True)
        for wtype in weight_types:
            tensor_key = f"model.layers.{layer_idx}.{wtype}.weight"
            w_base = load_tensor_from_shards(tensor_key, base_shards)
            w_actsvd = load_tensor_from_shards(tensor_key, actsvd_shards)
            if w_base is None or w_actsvd is None:
                continue
            w_base = w_base.float()
            w_actsvd = w_actsvd.float()
            delta_w = w_actsvd - w_base  # [out, in]

            # Use randomized SVD for speed — we only need enough singular
            # vectors to cover the threshold-based effective rank.
            # Start with q=64; if threshold selects all of them, retry with
            # full SVD (rare in practice).
            q = min(64, *delta_w.shape)
            try:
                U, S, V = torch.svd_lowrank(delta_w, q=q, niter=4)
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
            d = direction.shape[-1]

            if d != U_B.shape[0]:
                print(f"  Skipping {tensor_key}: direction dim {d} != weight out dim {U_B.shape[0]}")
                continue

            if is_subspace:
                proj = U_B.t() @ Q_S  # [k_B, k_S]
                mso = float(torch.square(proj).sum().item()) / float(min(k_S, k_B))
                random_baseline = float(max(k_S, k_B)) / d
            else:
                mso = float(torch.square(U_B.t() @ direction).sum().item())
                random_baseline = k_B / d

            results[tensor_key] = {
                "mso": mso,
                "k_A": k_S,
                "k_B": k_B,
                "d": d,
                "random_baseline": random_baseline,
            }
            print(f"  {tensor_key}: MSO={mso:.4f}, k_A={k_S}, k_B={k_B}, baseline={random_baseline:.4f}")

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
    data.setdefault("principal_cosines", {})

    # Compute direction-pair similarities. For 1-D vs 1-D this is a true cosine.
    # For 1-D vs k-D or k-D vs k-D we report the maximum singular value of
    # the cross-projection (= cosine of the smallest principal angle), which
    # reduces to the cosine when both sides are 1-D.
    def _orthonormal_cols(d: torch.Tensor) -> torch.Tensor:
        if d.ndim == 1:
            return (d / d.norm().clamp_min(1e-12)).unsqueeze(-1)  # [d, 1]
        Q, _ = torch.linalg.qr(d.t().contiguous(), mode="reduced")  # [d, k]
        return Q

    if len(named_directions) >= 2:
        print("\nComputing direction-pair principal-angle cosines ...")
        for i in range(len(named_directions)):
            for j in range(i + 1, len(named_directions)):
                name_i, dir_i = named_directions[i]
                name_j, dir_j = named_directions[j]
                if dir_i.shape[-1] != dir_j.shape[-1]:
                    print(f"  Skipping {name_i} vs {name_j}: hidden-dim mismatch "
                          f"{dir_i.shape[-1]} vs {dir_j.shape[-1]}")
                    continue
                Qi = _orthonormal_cols(dir_i)
                Qj = _orthonormal_cols(dir_j)
                # Principal-angle cosine between the two subspaces.
                M = Qi.t() @ Qj  # [k_i, k_j]
                singular = torch.linalg.svdvals(M).tolist()
                top_cos = float(singular[0]) if singular else 0.0
                pair_key = f"{name_i}_vs_{name_j}"
                data["direction_cosine"][pair_key] = top_cos
                data["principal_cosines"][pair_key] = singular
                print(f"  {pair_key}: top principal cosine={top_cos:.4f}, "
                      f"all={['%.3f' % s for s in singular]}")

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
