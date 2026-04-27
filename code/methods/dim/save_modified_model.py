"""
Load the saved refusal direction from a previous pipeline run,
apply orthogonalization to the model weights, and save the modified model.

Usage:
    uv run save_modified_model.py --model_path 'meta-llama/Llama-3.1-8B-Instruct'
"""

import os
import argparse
import torch

from pipeline.model_utils.model_factory import construct_model_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None, help="Path to pipeline run directory (default: pipeline/runs/<model_alias>)")
    args = parser.parse_args()

    model_alias = os.path.basename(args.model_path)
    run_dir = args.run_dir or os.path.join(os.path.dirname(__file__), "pipeline", "runs", model_alias)

    direction_path = os.path.join(run_dir, "direction.pt")
    assert os.path.exists(direction_path), f"No saved direction found at {direction_path}. Run the pipeline first."

    print(f"Loading direction from {direction_path}...")
    direction = torch.load(direction_path, weights_only=True)

    print(f"Loading model {args.model_path}...")
    model_base = construct_model_base(args.model_path)

    print("Applying orthogonalization to model weights...")
    orthogonalization_fn = model_base._get_orthogonalization_mod_fn(direction)
    orthogonalization_fn(model_base.model)

    save_path = os.path.join(run_dir, "modified_model")
    print(f"Saving modified model to {save_path}...")
    model_base.model.save_pretrained(save_path)
    model_base.tokenizer.save_pretrained(save_path)
    print("Done!")


if __name__ == "__main__":
    main()
