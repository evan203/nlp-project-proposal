
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
    """Return an orthonormal column basis [d, k] from a [k, d] or [d] tensor."""
    if direction.ndim == 1:
        return (direction / (direction.norm() + 1e-8)).unsqueeze(-1)
    # direction: [k, d]. Stack as columns and QR-orthonormalize.
    cols = direction.t().contiguous()  # [d, k]
    Q, _ = torch.linalg.qr(cols, mode="reduced")  # [d, k] orthonormal
    return Q


def _ablate_subspace(activation: Tensor, direction: Tensor) -> Tensor:
    """Project activation onto the orthogonal complement of the subspace
    spanned by `direction` (1D or 2D)."""
    Q = _orthonormal_basis(direction).to(activation)  # [d, k]
    proj = activation @ Q  # [..., k]
    return activation - proj @ Q.t()


def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        activation = _ablate_subspace(activation, direction)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        activation = _ablate_subspace(activation, direction)

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