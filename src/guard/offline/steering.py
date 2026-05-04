"""Steering vector computation — diff-means and orthogonalisation."""

from typing import Literal

import torch
from loguru import logger

from guard.config.normalization import NormalizationConfig, SvScaling

__all__ = ["compute_steering_vector"]


def compute_steering_vector(
    forget_mean: torch.Tensor,
    forget_norm: float,
    retain_mean: torch.Tensor | None,
    retain_norm: float | None,
    method: Literal["diff_means", "orthogonal", "orth_diff_means"],
    norm_cfg: NormalizationConfig,
) -> tuple[torch.Tensor, float]:
    """Compute the steering vector (SV) from forget/retain activation means.

    All operations are performed in ``float32`` regardless of input dtype.

    Applies a two-step normalisation pipeline controlled by ``norm_cfg.sv_scaling``:

    1. ``'unit'`` or ``'activation_norm'``: divide the raw vector by its L2 norm.
    2. ``'activation_norm'`` only: rescale so ``‖sv‖ = (ρ_f + ρ_r) / 2``, making
       the steering coefficient α model-agnostic across architectures.

    Args:
        forget_mean: Mean activation of the forget corpus, shape ``[hidden_dim]``.
        forget_norm: Mean L2 norm ρ_f of forget activations.
        retain_mean: Mean activation of the retain corpus, shape ``[hidden_dim]``,
            or ``None`` when retain data is unavailable.
        retain_norm: Mean L2 norm ρ_r of retain activations, or ``None``.
        method: SV computation method.  ``'diff_means'`` computes ``forget − retain``.
            ``'orthogonal'`` projects forget orthogonal to retain.
            ``'orth_diff_means'`` computes diff-means then projects orthogonal to retain.
        norm_cfg: Normalisation settings from :class:`~guard.config.NormalizationConfig`.

    Returns:
        ``(sv, activation_norm)`` — the steering vector and the target norm used for
        activation-norm scaling (``0.0`` when ``sv_scaling != 'activation_norm'``).

    Raises:
        ValueError: If ``method`` requires ``retain_mean`` but it is ``None``.
    """
    eps = norm_cfg.projection_eps
    f = forget_mean.float()

    if method == "diff_means":
        if retain_mean is None:
            raise ValueError("method='diff_means' requires retain_mean.")
        r = retain_mean.float()
        steering = f - r

    elif method == "orthogonal":
        if retain_mean is None:
            raise ValueError("method='orthogonal' requires retain_mean.")
        r = retain_mean.float()
        # sv = forget − (forget·retain / ‖retain‖²) · retain
        retain_norm_sq = torch.dot(r, r) + eps
        projection_coeff = torch.dot(f, r) / retain_norm_sq
        steering = f - projection_coeff * r
        logger.debug("Orthogonal projection_coeff={:.6f}", float(projection_coeff.item()))

    elif method == "orth_diff_means":
        if retain_mean is None:
            raise ValueError("method='orth_diff_means' requires retain_mean.")
        r = retain_mean.float()
        # dm = forget − retain; sv = dm − (dm·retain / ‖retain‖²) · retain
        dm = f - r
        retain_norm_sq = torch.dot(r, r) + eps
        projection_coeff = torch.dot(dm, r) / retain_norm_sq
        steering = dm - projection_coeff * r
        logger.debug("Orth_diff_means projection_coeff={:.6f}", float(projection_coeff.item()))

    else:
        raise ValueError(
            f"Unknown method: '{method}'.  Expected one of: "
            "'diff_means', 'orthogonal', 'orth_diff_means'."
        )

    raw_norm = float(torch.norm(steering).item())
    logger.debug(
        "Pre-scaling: forget_norm={:.4f}  retain_norm={}  steering_norm={:.4f}",
        forget_norm,
        f"{retain_norm:.4f}" if retain_norm is not None else "N/A",
        raw_norm,
    )

    activation_norm = 0.0

    if norm_cfg.sv_scaling in (SvScaling.UNIT, SvScaling.ACTIVATION_NORM):
        steering = steering / (torch.norm(steering) + eps)

    if norm_cfg.sv_scaling == SvScaling.ACTIVATION_NORM:
        target_norm = (forget_norm + retain_norm) / 2.0 if retain_norm is not None else forget_norm
        sv_norm = float(torch.norm(steering).item())
        steering = steering * (target_norm / (sv_norm + eps))
        activation_norm = target_norm
        logger.debug(
            "Post-scaling: target_norm={:.4f}  final_norm={:.4f}",
            target_norm,
            float(torch.norm(steering).item()),
        )

    return steering.detach(), activation_norm
