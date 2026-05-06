#!/usr/bin/env python3
"""Compute geometric overlap between two saved vector/subspace tensors.

This is the lightweight analysis helper for the report extension: once safety
and utility directions/subspaces are saved as tensors, this script reports MSO
and a random-subspace baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_tensor(path: Path, key: str | None = None) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if key:
        for part in key.split("."):
            if isinstance(obj, dict):
                obj = obj[part]
            elif isinstance(obj, (list, tuple)):
                obj = obj[int(part)]
            else:
                raise TypeError(f"Cannot index {type(obj).__name__} with {part!r}")
    if isinstance(obj, dict):
        raise TypeError(f"{path} contains a dict; pass --left-key/--right-key")
    if isinstance(obj, (list, tuple)):
        obj = torch.stack([torch.as_tensor(x) for x in obj])
    return torch.as_tensor(obj, dtype=torch.float32)


def basis_from_tensor(tensor: torch.Tensor, rank: int | None = None, energy: float = 0.99) -> torch.Tensor:
    """Return an orthonormal column basis with shape [hidden_dim, k]."""
    tensor = tensor.detach().float().cpu()

    if tensor.ndim == 1:
        vector = tensor / tensor.norm().clamp_min(1e-12)
        return vector[:, None]

    matrix = tensor.reshape(-1, tensor.shape[-1])
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)

    if rank is None:
        energy_curve = torch.cumsum(singular_values.square(), dim=0) / singular_values.square().sum().clamp_min(1e-12)
        rank = int(torch.searchsorted(energy_curve, torch.tensor(energy)).item()) + 1

    rank = max(1, min(rank, vh.shape[0]))
    return vh[:rank].T.contiguous()


def mso(left_basis: torch.Tensor, right_basis: torch.Tensor) -> float:
    overlap = left_basis.T @ right_basis
    return overlap.square().sum().item() / min(left_basis.shape[1], right_basis.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two saved vector/subspace tensors with MSO.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-key", default=None)
    parser.add_argument("--right-key", default=None)
    parser.add_argument("--left-rank", type=int, default=None)
    parser.add_argument("--right-rank", type=int, default=None)
    parser.add_argument("--energy", type=float, default=0.99)
    args = parser.parse_args()

    left = basis_from_tensor(load_tensor(args.left, args.left_key), args.left_rank, args.energy)
    right = basis_from_tensor(load_tensor(args.right, args.right_key), args.right_rank, args.energy)

    if left.shape[0] != right.shape[0]:
        raise ValueError(f"Hidden dimensions differ: {left.shape[0]} vs {right.shape[0]}")

    score = mso(left, right)
    random_baseline = max(left.shape[1], right.shape[1]) / left.shape[0]

    print(f"left_dim: {left.shape[1]}")
    print(f"right_dim: {right.shape[1]}")
    print(f"hidden_dim: {left.shape[0]}")
    print(f"mso: {score:.6f}")
    print(f"random_baseline: {random_baseline:.6f}")
    print(f"mso_over_random: {score / random_baseline:.3f}")


if __name__ == "__main__":
    main()
