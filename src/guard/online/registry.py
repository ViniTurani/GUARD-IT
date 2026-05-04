"""Target-module registry — resolves transformer layer + submodule for hook registration."""


from typing import Any

__all__ = ["get_target_module"]


def get_target_module(model: Any, layer_idx: int, module_name: str) -> Any:
    """Return the ``nn.Module`` to attach a forward hook to.

    Supports Llama-style (``model.model.layers``) and GPT-style
    (``model.transformer.h``) architectures.

    Args:
        model: HuggingFace causal LM.
        layer_idx: 0-based transformer layer index.
        module_name: ``'residual'`` → the full decoder layer;
            any other string → ``getattr(layer, module_name)``.

    Returns:
        The target ``nn.Module``.

    Raises:
        ValueError: If the architecture is not supported or the index is out of
            range or the sub-module does not exist.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError(
            "Unsupported model architecture: cannot find transformer layers at "
            "model.model.layers (Llama-style) or model.transformer.h (GPT-style)."
        )

    try:
        layer = layers[layer_idx]
    except IndexError:
        raise ValueError(
            f"layer_idx={layer_idx} is out of range for this model "
            f"({len(layers)} layers total)."
        ) from None

    if module_name == "residual":
        return layer

    if not hasattr(layer, module_name):
        raise ValueError(
            f"Layer {layer_idx} has no sub-module '{module_name}'. "
            f"Available attributes: {[n for n, _ in layer.named_children()]}"
        )
    return getattr(layer, module_name)
