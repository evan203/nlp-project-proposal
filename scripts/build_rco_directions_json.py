#!/usr/bin/env python3
"""Build a directions JSON file for Geometry/RepInd analysis.

Use this after `scripts/run_rco.sh` creates local vector artifacts. The output
can be passed to `scripts/run_geometry_repind.sh` as `DIRECTIONS_JSON=...`.

Pass `--standalone_output PATH` to also copy the latest RCO vector to PATH
as a plain .pt file (for use with eval_direction_benchmark.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIM_DIRECTION = ROOT / "code/methods/dim/pipeline/runs/Llama-3.1-8B-Instruct/direction.pt"
DEFAULT_RCO_ROOT = ROOT / "code/methods/cones-repind/results/rdo"
DEFAULT_OUTPUT = ROOT / "code/results/geometry_repind/directions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build directions JSON for RepInd analysis.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dim_direction", type=Path, default=DEFAULT_DIM_DIRECTION)
    parser.add_argument("--rco_root", type=Path, default=DEFAULT_RCO_ROOT)
    parser.add_argument("--vector", action="append", type=Path, default=[], help="Explicit vector .pt file to include.")
    parser.add_argument("--latest", action="store_true", help="Include the latest local RCO/RepInd vector artifact.")
    parser.add_argument("--name", action="append", default=[], help="Name for each explicit --vector, in order.")
    parser.add_argument("--no_dim", action="store_true", help="Do not include the DIM baseline direction.")
    parser.add_argument(
        "--standalone_output", type=Path, default=None,
        help="Also copy the latest RCO vector to this .pt path (for eval_direction_benchmark.py).",
    )
    return parser.parse_args()


def latest_local_vector(rco_root: Path) -> Path | None:
    candidates = list(rco_root.glob("*/local_vectors/*/*/lowest_loss_vector.pt"))
    if not candidates:
        candidates = list(rco_root.glob("*/local_vectors/*/*/vectors.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    args = parse_args()
    spec = []

    if not args.no_dim:
        if not args.dim_direction.exists():
            raise FileNotFoundError(f"DIM direction not found: {args.dim_direction}")
        spec.append({"name": "DIM", "path": str(args.dim_direction)})

    explicit_names = args.name or []
    for idx, vector_path in enumerate(args.vector):
        if not vector_path.exists():
            raise FileNotFoundError(f"Vector not found: {vector_path}")
        name = explicit_names[idx] if idx < len(explicit_names) else f"RCO-{idx + 1}"
        spec.append({"name": name, "path": str(vector_path)})

    rco_vector_path: Path | None = None
    if args.latest:
        rco_vector_path = latest_local_vector(args.rco_root)
        if rco_vector_path is None:
            raise FileNotFoundError(f"No local RCO/RepInd vectors found under {args.rco_root}")
        spec.append({"name": "Latest-RCO", "path": str(rco_vector_path)})

    # Copy standalone .pt if requested (does not require 2 directions in spec)
    if args.standalone_output is not None:
        src = rco_vector_path or (args.vector[-1] if args.vector else None)
        if src is None:
            raise ValueError("--standalone_output requires --latest or at least one --vector")
        args.standalone_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, args.standalone_output)
        print(f"Standalone direction saved to {args.standalone_output}")

    # Write directions JSON only when we have enough entries for a comparison
    if len(spec) >= 2:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(spec, indent=2))
        print(f"Wrote {args.output}")
        for item in spec:
            print(f"  {item['name']}: {item['path']}")
    elif args.standalone_output is None:
        raise ValueError("Need at least two directions for a directions JSON. Add --latest or one or more --vector paths.")


if __name__ == "__main__":
    main()
