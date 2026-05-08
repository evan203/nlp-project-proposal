
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def _orthonormal_basis(direction: Tensor) -> Tensor:
    """Return an orthonormal column basis [d, k] from a [k, d] or [d] tensor.

    For multi-D inputs, QR is computed in float32 because CUDA's `geqrf`
    backend does not implement bf16/fp16. The returned basis is cast back to
    the input's original dtype.
    """
    if direction.ndim == 1:
        return (direction / (direction.norm() + 1e-8)).unsqueeze(-1)
    orig_dtype = direction.dtype
    # [d, k] in fp32 for QR
    cols_fp32 = direction.t().contiguous().to(torch.float32)
    Q_fp32, _ = torch.linalg.qr(cols_fp32, mode="reduced")  # [d, k]
    return Q_fp32.to(orig_dtype)


def get_direction_ablation_input_pre_hook(direction: Tensor):
    # Precompute the orthonormal basis once at hook construction so we don't
    # redo QR on every forward pass.
    basis_cache = {"Q": _orthonormal_basis(direction)}

    def hook_fn(module, input):
        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        Q = basis_cache["Q"]
        if Q.device != activation.device or Q.dtype != activation.dtype:
            Q = Q.to(device=activation.device, dtype=activation.dtype)
            basis_cache["Q"] = Q
        proj = activation @ Q  # [..., k]
        activation = activation - proj @ Q.t()

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    basis_cache = {"Q": _orthonormal_basis(direction)}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        Q = basis_cache["Q"]
        if Q.device != activation.device or Q.dtype != activation.dtype:
            Q = Q.to(device=activation.device, dtype=activation.dtype)
            basis_cache["Q"] = Q
        proj = activation @ Q
        activation = activation - proj @ Q.t()

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_direction_ablation_hooks(
    model_base,
    direction: Tensor,
):
    """Build per-layer hooks that ablate `direction` from residual stream and
    attn/MLP outputs. `direction` may be 1D `[d_model]` (single-direction
    ablation) or 2D `[k, d_model]` (k-direction subspace ablation)."""
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn