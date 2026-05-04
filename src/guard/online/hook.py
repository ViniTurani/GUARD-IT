"""Forward hook registration for additive / rotation-only steering."""

from typing import Any

import torch
from loguru import logger

__all__ = ["register_steering_hook"]


def register_steering_hook(
    model: Any,
    sv: torch.Tensor,
    coeff_holder: list[float],
    layer_idx: int,
    module_name: str = "residual",
    rotation_only: bool = True,
) -> Any:
    """Register an additive steering hook on a transformer layer.

    The hook reads ``coeff_holder[0]`` on **every** forward pass, so the
    caller can update the steering coefficient between passes without
    re-registering — enabling fast coefficient sweeps with a single model load.

    The steering vector ``sv`` is cast to the hidden state's dtype and device
    inside the hook.

    Args:
        model: HuggingFace causal LM (modified in place via the hook).
        sv: Steering vector tensor ``[hidden_dim]``.
        coeff_holder: Single-element mutable list.  ``coeff_holder[0]`` is
            read on each forward pass.
        layer_idx: 0-based transformer layer index.
        module_name: ``'residual'`` or sub-module name (``'mlp'``, ``'self_attn'``).
        rotation_only: If ``True`` (default), apply rotation-only steering (Eq. 11)::

                h' = (h - α·sv) · (‖h‖ / ‖h - α·sv‖)

            The hidden-state magnitude is preserved; only its direction changes.

    Returns:
        A ``RemovableHandle``.  Call ``.remove()`` to detach the hook.
    """
    from guard.online.registry import get_target_module

    target = get_target_module(model, layer_idx, module_name)

    def _hook(
        _module: Any,
        _inputs: Any,
        output: Any,
    ) -> Any:
        coeff = coeff_holder[0]
        is_tuple = isinstance(output, tuple)
        h: torch.Tensor = output[0] if is_tuple else output

        sv_dev = sv.to(dtype=h.dtype, device=h.device)

        if rotation_only:
            orig_norm = h.norm(dim=-1, keepdim=True)  # [B, S, 1]
            h_new = h - coeff * sv_dev
            new_norm = h_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h_new = h_new * (orig_norm / new_norm)
        else:
            h_new = h - coeff * sv_dev

        return (h_new,) + output[1:] if is_tuple else h_new

    handle = target.register_forward_hook(_hook)
    logger.debug(
        "Registered steering hook: layer={}  module={}  rotation_only={}",
        layer_idx, module_name, rotation_only,
    )
    return handle
