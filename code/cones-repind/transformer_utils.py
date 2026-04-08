import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Dict, Tuple, Optional
import json
import os
from jaxtyping import Float
from torch import Tensor
import numpy as np


def get_model_config(model_name: str) -> dict:
    """Get chat template and refusal tokens for supported models."""
    configs = {
        "llama-3": {
            "template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "refusal_tokens": [40],
            "eos_tokens": [128001, 128009],
        },
        "llama-3.1": {
            "template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "refusal_tokens": [40],
            "eos_tokens": [128001, 128009],
        },
        "llama-2": {
            "template": "[INST] {instruction} [/INST]",
            "refusal_tokens": [29896, 25580, 29991],  # "I" "I'm" "I'"
            "eos_tokens": [2],
        },
        "gemma": {
            "template": "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
            "refusal_tokens": [235285],
            "eos_tokens": [107, 108],
        },
        "qwen": {
            "template": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
            "refusal_tokens": [40, 2121],
            "eos_tokens": [151643, 151645],
        },
    }

    model_lower = model_name.lower()
    for key, config in configs.items():
        if key in model_lower:
            return config

    # Fallback: load from config.json for local paths
    if os.path.exists(model_name):
        config_path = os.path.join(model_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type", "").lower()
            architectures = config.get("architectures", [])

            if "llama" in model_type or any(
                "llama" in str(a).lower() for a in architectures
            ):
                return configs["llama-3.1"]
            elif "gemma" in model_type:
                return configs["gemma"]
            elif "qwen" in model_type:
                return configs["qwen"]

    raise ValueError(f"Model {model_name} not supported. Please add configuration.")


def load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def apply_chat_template(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    model_path: str,
) -> List[str]:
    """Apply chat template to instructions."""
    config = get_model_config(model_path)
    template = config["template"]
    return [template.format(instruction=inst) for inst in instructions]


class ActivationExtractor:
    """Extract activations from transformer layers."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layer_type: str = "last_token",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_type = layer_type
        self.activations = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self.hooks = []

        def get_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output

                if self.layer_type == "last_token":
                    act = activation[:, -1, :]
                elif self.layer_type == "all_tokens":
                    act = activation
                elif self.layer_type == "mean_pool":
                    act = activation.mean(dim=1)
                else:
                    raise ValueError(f"Unknown layer_type: {self.layer_type}")

                self.activations[name] = act.detach().cpu()

            return hook

        for name, module in self.model.named_modules():
            if "model.layers" in name or "transformer.h" in name:
                if ".mlp." in name and ("down_proj" in name or name.count(".") == 4):
                    handle = module.register_forward_hook(get_hook(name))
                    self.hooks.append(handle)

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def get_layer_names(self) -> List[str]:
        """Get sorted list of layer names."""
        return sorted(self.activations.keys())

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


@torch.no_grad()
def get_residual_stream_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    layer_type: str = "last_token",
) -> List[Dict[int, Float[Tensor, "batch d_model"]]]:
    """
    Get residual stream activations for each layer.

    Returns:
        List of dicts, one per prompt, mapping layer_idx -> activations
    """
    all_layer_activations = []
    n_layers = model.config.num_hidden_layers

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        for j in range(len(batch_prompts)):
            prompt_activations = {}
            for layer_idx in range(n_layers):
                h = hidden_states[layer_idx][j]
                if layer_type == "last_token":
                    prompt_activations[layer_idx] = h[-1].cpu()
                elif layer_type == "all_tokens":
                    prompt_activations[layer_idx] = h.cpu()
                elif layer_type == "mean_pool":
                    prompt_activations[layer_idx] = h.mean(0).cpu()
            all_layer_activations.append(prompt_activations)

        del outputs, hidden_states, enc
        torch.cuda.empty_cache()

    return all_layer_activations


@torch.no_grad()
def compute_mean_diff_direction(
    harmful_activations: List[Dict[int, Float[Tensor, "batch d_model"]]],
    harmless_activations: List[Dict[int, Float[Tensor, "batch d_model"]]],
    layer_idx: int,
) -> Float[Tensor, "d_model"]:
    """
    Compute difference of means direction for a given layer.

    Args:
        harmful_activations: List of activation dicts for harmful prompts
        harmless_activations: List of activation dicts for harmless prompts
        layer_idx: Which layer to extract directions from

    Returns:
        Unit-normalized direction vector
    """
    harmful_layer_acts = torch.stack([act[layer_idx] for act in harmful_activations])
    harmless_layer_acts = torch.stack([act[layer_idx] for act in harmless_activations])

    harmful_mean = harmful_layer_acts.mean(dim=0)
    harmless_mean = harmless_layer_acts.mean(dim=0)

    direction = harmful_mean - harmless_mean
    direction = direction / direction.norm()

    return direction


def compute_dim_directions(
    harmful_activations: List[Dict[int, Float[Tensor, "batch d_model"]]],
    harmless_activations: List[Dict[int, Float[Tensor, "batch d_model"]]],
    layer_range: Tuple[int, int] = (None, None),
) -> Dict[int, Float[Tensor, "d_model"]]:
    """
    Compute DIM directions for all layers in range.

    Args:
        harmful_activations: List of activation dicts for harmful prompts
        harmless_activations: List of activation dicts for harmless prompts
        layer_range: (start, end) layer indices, None for all

    Returns:
        Dict mapping layer_idx -> direction vector
    """
    n_layers = len(harmful_activations[0])
    start = layer_range[0] if layer_range[0] is not None else 0
    end = layer_range[1] if layer_range[1] is not None else n_layers

    directions = {}
    for layer_idx in range(start, end):
        directions[layer_idx] = compute_mean_diff_direction(
            harmful_activations, harmless_activations, layer_idx
        )

    return directions


def get_cosine_sims_for_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    direction: Float[Tensor, "d_model"],
    batch_size: int = 8,
    device: str = "cuda",
) -> Float[Tensor, "n_layers batch"]:
    """
    Compute cosine similarity between direction and activations across layers.

    Returns:
        Tensor of shape (n_layers, batch) with cosine similarities
    """
    n_layers = model.config.num_hidden_layers
    direction = direction / direction.norm()

    all_cosine_sims = [[] for _ in range(n_layers)]

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        batch_size_actual = enc["input_ids"].shape[0]

        for layer_idx in range(n_layers):
            h = hidden_states[layer_idx][:, -1, :]
            cos_sim = torch.nn.functional.cosine_similarity(
                h, direction.to(device).unsqueeze(0), dim=-1
            )
            all_cosine_sims[layer_idx].append(cos_sim.cpu())

        del outputs, hidden_states, enc
        torch.cuda.empty_cache()

    result = []
    for layer_idx in range(n_layers):
        layer_sims = torch.cat(all_cosine_sims[layer_idx], dim=0)
        result.append(layer_sims)

    return torch.stack(result, dim=0)
