"""Unit tests for steering vector computation and normalization math."""

import math

import pytest
import torch

from guard.compute.steering import compute_steering_vector
from guard.config.normalization import NormalizationConfig, SvScaling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vecs() -> tuple[torch.Tensor, torch.Tensor]:
    """Two non-parallel unit vectors for deterministic tests."""
    torch.manual_seed(0)
    forget = torch.randn(64)
    retain = torch.randn(64)
    # Make them non-zero and non-parallel.
    forget = forget / forget.norm()
    retain = retain / retain.norm()
    return forget, retain


# ---------------------------------------------------------------------------
# diff_means tests
# ---------------------------------------------------------------------------


def test_diff_means_basic(vecs: tuple[torch.Tensor, torch.Tensor]) -> None:
    f, r = vecs
    norm_cfg = NormalizationConfig(sv_scaling=SvScaling.NONE)
    sv, act_norm = compute_steering_vector(
        forget_mean=f, forget_norm=1.0,
        retain_mean=r, retain_norm=1.0,
        method="diff_means", norm_cfg=norm_cfg,
    )
    expected = f - r
    assert torch.allclose(sv, expected, atol=1e-6), "diff_means should return forget − retain"
    assert act_norm == 0.0, "sv_scaling=none should return activation_norm=0.0"


def test_diff_means_requires_retain(vecs: tuple[torch.Tensor, torch.Tensor]) -> None:
    f, _ = vecs
    norm_cfg = NormalizationConfig(sv_scaling=SvScaling.NONE)
    with pytest.raises(ValueError, match="retain_mean"):
        compute_steering_vector(
            forget_mean=f, forget_norm=1.0,
            retain_mean=None, retain_norm=None,
            method="diff_means", norm_cfg=norm_cfg,
        )


# ---------------------------------------------------------------------------
# orthogonal tests
# ---------------------------------------------------------------------------


def test_orthogonal_perpendicular_to_retain(vecs: tuple[torch.Tensor, torch.Tensor]) -> None:
    f, r = vecs
    norm_cfg = NormalizationConfig(sv_scaling=SvScaling.NONE)
    sv, _ = compute_steering_vector(
        forget_mean=f, forget_norm=1.0,
        retain_mean=r, retain_norm=1.0,
        method="orthogonal", norm_cfg=norm_cfg,
    )
    # sv · retain should be ~0 (up to projection_eps).
    dot = float(torch.dot(sv, r).item())
    assert abs(dot) < 1e-4, f"Orthogonal SV should be perpendicular to retain, got dot={dot:.6f}"


def test_orthogonal_same_as_suv_formula(vecs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Verify the exact formula from SUV.py is preserved."""
    f, r = vecs
    eps = 1e-6
    retain_norm_sq = torch.dot(r, r) + eps
    projection_coeff = torch.dot(f, r) / retain_norm_sq
    expected = f - projection_coeff * r

    norm_cfg = NormalizationConfig(sv_scaling=SvScaling.NONE, projection_eps=eps)
    sv, _ = compute_steering_vector(
        forget_mean=f, forget_norm=1.0,
        retain_mean=r, retain_norm=1.0,
        method="orthogonal", norm_cfg=norm_cfg,
    )
    assert torch.allclose(sv, expected, atol=1e-5), "Orthogonal formula should match SUV.py"


# ---------------------------------------------------------------------------
# SvScaling tests
# ---------------------------------------------------------------------------


def test_unit_scaling_produces_unit_vector(vecs: tuple[torch.Tensor, torch.Tensor]) -> None:
    f, r = vecs
    norm_cfg = NormalizationConfig(sv_scaling=SvScaling.UNIT)
    sv, act_norm = compute_steering_vector(
        forget_mean=f * 5.0, forget_norm=5.0,
        retain_mean=r * 5.0, retain_norm=5.0,
        method="diff_means", norm_cfg=norm_cfg,
    )
    sv_norm = float(sv.norm().item())
    assert abs(sv_norm - 1.0) < 1e-5, f"SvScaling.UNIT should yield unit vector, got norm={sv_norm}"
    assert act_norm == 0.0


def test_activation_norm_scaling(vecs: tuple[torch.Tensor, torch.Tensor]) -> None:
    f, r = vecs
    forget_norm = 7.5
    retain_norm = 8.5
    expected_target_norm = (forget_norm + retain_norm) / 2.0

    norm_cfg = NormalizationConfig(sv_scaling=SvScaling.ACTIVATION_NORM)
    sv, act_norm = compute_steering_vector(
        forget_mean=f, forget_norm=forget_norm,
        retain_mean=r, retain_norm=retain_norm,
        method="diff_means", norm_cfg=norm_cfg,
    )
    sv_norm = float(sv.norm().item())
    assert abs(sv_norm - expected_target_norm) < 1e-4, (
        f"SvScaling.ACTIVATION_NORM should yield norm≈{expected_target_norm}, got {sv_norm}"
    )
    assert abs(act_norm - expected_target_norm) < 1e-6


def test_activation_norm_forget_only(vecs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """When retain is None (forget-only), activation_norm = forget_norm."""
    f, _ = vecs
    forget_norm = 6.0
    norm_cfg = NormalizationConfig(sv_scaling=SvScaling.ACTIVATION_NORM)
    # diff_means without retain raises; use orthogonal with retain for this test.
    # Actually for this test we pass retain to diff_means with a specific norm.
    sv, act_norm = compute_steering_vector(
        forget_mean=f, forget_norm=forget_norm,
        retain_mean=None, retain_norm=None,
        method="orthogonal", norm_cfg=norm_cfg,
    ) if False else (None, None)  # orthogonal needs retain too.

    # Use a tiny retain vector for the no-retain case — test activation_norm fallback.
    retain = torch.zeros_like(f)
    retain[0] = 1e-8  # nearly zero — projection is negligible
    sv, act_norm = compute_steering_vector(
        forget_mean=f, forget_norm=forget_norm,
        retain_mean=retain, retain_norm=None,  # retain_norm=None → uses forget_norm only
        method="orthogonal", norm_cfg=norm_cfg,
    )
    assert abs(act_norm - forget_norm) < 1e-4, (
        f"When retain_norm is None, activation_norm should equal forget_norm ({forget_norm}), "
        f"got {act_norm}"
    )


# ---------------------------------------------------------------------------
# Unknown method guard
# ---------------------------------------------------------------------------


def test_unknown_method_raises(vecs: tuple[torch.Tensor, torch.Tensor]) -> None:
    f, r = vecs
    with pytest.raises(ValueError, match="Unknown method"):
        compute_steering_vector(
            forget_mean=f, forget_norm=1.0,
            retain_mean=r, retain_norm=1.0,
            method="bad_method",  # type: ignore[arg-type]
            norm_cfg=NormalizationConfig(),
        )
