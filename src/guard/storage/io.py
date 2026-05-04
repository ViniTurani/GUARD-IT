"""Steering vector persistence — save, load, and canonical path resolution."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from guard._version import __version__

__all__ = [
    "sv_path",
    "save_sv",
    "load_sv",
    "load_sv_with_meta",
]


def _sanitize(name: str) -> str:
    """Replace path-unsafe characters in a model name."""
    return name.replace("/", "_").replace("\\", "_")


def sv_path(
    output_dir: Path | str,
    model_name: str,
    behavior: str,
    module_name: str,
    layer_idx: int,
    method: str,
    extension: str = "pt",
) -> Path:
    """Return the canonical path for a steering vector file.

    Layout::

        {output_dir}/{model_name_sanitized}/{method}/{behavior}/{module_name}/layer_{idx}/sv.{ext}

    Args:
        output_dir: Root output directory.
        model_name: HuggingFace model ID or local path (``/`` replaced with ``_``).
        behavior: Experiment label (e.g. ``'forget01'``).
        module_name: Module hooked (e.g. ``'residual'``).
        layer_idx: Layer index.
        method: SUV method (``'orthogonal'`` or ``'diff_means'``).
        extension: File extension (default ``'pt'``).

    Returns:
        :class:`pathlib.Path` to the SV file.
    """
    return (
        Path(output_dir)
        / _sanitize(model_name)
        / method
        / behavior
        / module_name
        / f"layer_{layer_idx}"
        / f"sv.{extension}"
    )


def save_sv(
    sv: torch.Tensor,
    output_dir: Path | str,
    model_name: str,
    behavior: str,
    module_name: str,
    layer_idx: int,
    method: str,
    metadata: dict[str, Any],
    config_source: Path | str | None = None,
) -> Path:
    """Save a steering vector with its metadata and the generating config.

    Creates the output directory if it does not exist, then writes:

    * ``sv.pt``         — the tensor (``float32``, ``weights_only``-safe)
    * ``sv.json``       — JSON metadata for reproducibility
    * ``generate.yaml`` — verbatim copy of the config file (if ``config_source`` given)

    Args:
        sv: Steering vector ``[hidden_dim]``.  Saved as ``float32``.
        output_dir: Root output directory.
        model_name: HuggingFace model ID or local path.
        behavior: Experiment label.
        module_name: Module hooked.
        layer_idx: Layer index.
        method: SUV method name.
        metadata: Arbitrary key/value pairs merged into ``sv.json``.  Should
            include at minimum: ``token_position``, ``sv_scaling``,
            ``rotation_only``, ``activation_norm``, ``projection_eps``.
        config_source: Path to the ``generate.yaml`` file to copy alongside
            the saved SV.  ``None`` = no copy.

    Returns:
        Path to the saved ``sv.pt`` file.
    """
    pt_path = sv_path(output_dir, model_name, behavior, module_name, layer_idx, method)
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure the vector is float32 and 1-D.
    sv_f32 = sv.float().detach().cpu()
    if sv_f32.dim() != 1:
        raise ValueError(f"Expected a 1-D steering vector, got shape {sv_f32.shape}.")
    torch.save(sv_f32, pt_path)

    # Assemble full metadata.
    full_meta: dict[str, Any] = {
        "guard_version": __version__,
        "torch_version": torch.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "behavior": behavior,
        "layer_idx": layer_idx,
        "module_name": module_name,
        "method": method,
        "hidden_dim": sv_f32.shape[0],
        **metadata,
    }
    json_path = pt_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(full_meta, fh, indent=2)

    # Copy the generating config for full reproducibility.
    if config_source is not None:
        dest = pt_path.parent / "generate.yaml"
        shutil.copy2(config_source, dest)

    return pt_path


def load_sv(path: Path | str) -> torch.Tensor:
    """Load a steering vector tensor.

    Args:
        path: Path to ``sv.pt``.

    Returns:
        1-D ``float32`` tensor on CPU.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded tensor is not 1-D.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Steering vector not found: {path}")
    sv: torch.Tensor = torch.load(path, map_location="cpu", weights_only=True)
    if sv.dim() != 1:
        raise ValueError(f"Expected a 1-D steering vector, got shape {sv.shape}.")
    return sv.float()


def load_sv_with_meta(path: Path | str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a steering vector and its JSON sidecar metadata.

    Args:
        path: Path to ``sv.pt``.  The sidecar is expected at the same path
            with ``.json`` extension.

    Returns:
        ``(sv_tensor, metadata_dict)``.

    Raises:
        FileNotFoundError: If either ``sv.pt`` or ``sv.json`` is missing.
    """
    sv = load_sv(path)
    json_path = Path(path).with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Steering vector metadata not found: {json_path}")
    with json_path.open(encoding="utf-8") as fh:
        meta: dict[str, Any] = json.load(fh)
    return sv, meta


def documents_hash(documents: list[str]) -> str:
    """Return a short SHA-256 hex digest of the document list.

    Used in ``sv.json`` to record which corpus produced the vector.
    """
    h = hashlib.sha256()
    for doc in documents:
        h.update(doc.encode("utf-8"))
    return h.hexdigest()[:16]
