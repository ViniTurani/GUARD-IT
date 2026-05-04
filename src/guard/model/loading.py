"""Utilities for consistent HuggingFace model loading options.

Supports regular bf16/fp32 loading and optional bitsandbytes quantization.
"""


from typing import Any, Literal

import torch

__all__ = ["build_model_load_kwargs"]


def _dtype_from_name(name: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported quant_compute_dtype='{name}'.")
    return mapping[name]


def build_model_load_kwargs(
    *,
    device: str,
    quantization: Literal["none", "4bit", "8bit"] = "none",
    quant_4bit_type: Literal["nf4", "fp4"] = "nf4",
    quant_4bit_double_quant: bool = True,
    quant_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16",
) -> dict[str, Any]:
    """Build kwargs for ``AutoModelForCausalLM.from_pretrained``.

    Args:
        device: Runtime device string (``'cuda'`` or ``'cpu'``).
        quantization: Quantization mode (none/4bit/8bit).
        quant_4bit_type: bitsandbytes 4-bit quant type.
        quant_4bit_double_quant: Whether to enable double quantization in 4-bit mode.
        quant_compute_dtype: Compute dtype used by bitsandbytes quantized matmuls.
    """
    if quantization == "none" or device != "cuda":
        return {
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "device_map": device,
        }

    try:
        from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Quantization requested but BitsAndBytesConfig is unavailable. "
            "Install optional dependencies: bitsandbytes and accelerate."
        ) from exc

    if quantization == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_4bit_type,
            bnb_4bit_use_double_quant=quant_4bit_double_quant,
            bnb_4bit_compute_dtype=_dtype_from_name(quant_compute_dtype),
        )
    elif quantization == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported quantization='{quantization}'.")

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "quantization_config": bnb_cfg,
        "low_cpu_mem_usage": True,
    }
    # bitsandbytes 8-bit int8 matmul doesn't support bfloat16 inputs; force float16
    if quantization == "8bit":
        kwargs["torch_dtype"] = torch.float16
    return kwargs
