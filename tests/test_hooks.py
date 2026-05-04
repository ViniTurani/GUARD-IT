"""Tests for hook registration and SteeredModel lifecycle."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal stub model (CPU, no GPU needed)
# ---------------------------------------------------------------------------


class _TinyLayer(nn.Module):
    """One-layer MLP that returns the input unchanged (identity)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyModelInner(nn.Module):
    """Inner object with a `.layers` attribute — mirrors Llama's `model.model.layers`."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer() for _ in range(4)])


class _TinyModel(nn.Module):
    """Bare-minimum model that matches the Llama-style `model.model.layers[i]` pattern."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.model = _TinyModelInner()
        self._dummy_param = nn.Parameter(torch.zeros(hidden))

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        yield self._dummy_param

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        for layer in self.model.layers:
            x = layer(x)
        return (x,)


# ---------------------------------------------------------------------------
# get_target_module
# ---------------------------------------------------------------------------


def test_get_target_module_residual() -> None:
    from guard.hooks.registry import get_target_module

    model = _TinyModel()
    mod = get_target_module(model, layer_idx=2, module_name="residual")
    assert mod is model.model.layers[2]


def test_get_target_module_out_of_range() -> None:
    from guard.hooks.registry import get_target_module

    model = _TinyModel()
    with pytest.raises(ValueError, match="out of range"):
        get_target_module(model, layer_idx=99, module_name="residual")


def test_get_target_module_unsupported_arch() -> None:
    from guard.hooks.registry import get_target_module

    class BadModel(nn.Module):
        pass

    with pytest.raises(ValueError, match="Unsupported"):
        get_target_module(BadModel(), layer_idx=0, module_name="residual")


# ---------------------------------------------------------------------------
# register_steering_hook
# ---------------------------------------------------------------------------


def test_steering_hook_adds_sv() -> None:
    """The hook should add -coeff * sv to the layer output."""
    from guard.hooks.steering_hook import register_steering_hook

    model = _TinyModel(hidden=4)
    sv = torch.ones(4)
    coeff_holder = [2.0]

    handle = register_steering_hook(
        model, sv, coeff_holder, layer_idx=0,
        module_name="residual", rotation_only=False,
    )

    x = torch.zeros(1, 3, 4)  # [batch, seq, hidden]
    with torch.no_grad():
        out = model(x)
    h = out[0]
    # Every element should be -2.0 (-coeff * 1.0)
    assert torch.allclose(h, torch.full_like(h, -2.0)), f"Expected -2.0 everywhere, got {h}"

    handle.remove()


def test_steering_hook_coeff_update() -> None:
    """Changing coeff_holder[0] should take effect immediately without re-registering."""
    from guard.hooks.steering_hook import register_steering_hook

    model = _TinyModel(hidden=4)
    sv = torch.ones(4)
    coeff_holder = [1.0]

    handle = register_steering_hook(
        model, sv, coeff_holder, layer_idx=0,
        module_name="residual", rotation_only=False,
    )

    x = torch.zeros(1, 1, 4)
    with torch.no_grad():
        out1 = model(x)
    assert torch.allclose(out1[0], -torch.ones(1, 1, 4))

    # Update coefficient in place.
    coeff_holder[0] = 5.0
    with torch.no_grad():
        out2 = model(x)
    assert torch.allclose(out2[0], torch.full((1, 1, 4), -5.0))

    handle.remove()


def test_steering_hook_rotation_only() -> None:
    """rotation_only=True should preserve the original hidden-state magnitude."""
    from guard.hooks.steering_hook import register_steering_hook

    model = _TinyModel(hidden=4)
    # Set input so its norm is known.
    coeff_holder = [10.0]  # large coeff to make the effect visible
    sv = torch.randn(4)

    handle = register_steering_hook(
        model, sv, coeff_holder, layer_idx=0,
        module_name="residual", rotation_only=True,
    )

    x_val = torch.randn(1, 2, 4)
    orig_norms = x_val.norm(dim=-1)  # [1, 2]
    with torch.no_grad():
        out = model(x_val)
    steered_norms = out[0].norm(dim=-1)
    assert torch.allclose(orig_norms, steered_norms, atol=1e-5), (
        "rotation_only=True should preserve each token's L2 norm"
    )

    handle.remove()


def test_hook_removed_after_handle_remove() -> None:
    """After handle.remove(), the hook should no longer modify outputs."""
    from guard.hooks.steering_hook import register_steering_hook

    model = _TinyModel(hidden=4)
    sv = torch.ones(4)
    coeff_holder = [3.0]

    handle = register_steering_hook(
        model, sv, coeff_holder, layer_idx=0,
        module_name="residual", rotation_only=False,
    )
    handle.remove()

    x = torch.zeros(1, 1, 4)
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out[0], torch.zeros(1, 1, 4)), "Output should be zero after hook removal"


# ---------------------------------------------------------------------------
# SteeredModel context manager
# ---------------------------------------------------------------------------


def test_steered_model_context_manager(tmp_path: Path) -> None:
    from guard.model.steered import SteeredModel

    model = _TinyModel(hidden=4)
    sv = torch.ones(4)
    pt_path = tmp_path / "sv.pt"
    torch.save(sv, pt_path)

    x = torch.zeros(1, 1, 4)

    with SteeredModel.from_sv_path(
        model, sv_path=pt_path, coeff=2.0,
        layer_idx=0, rotation_only=False,
    ) as steered:
        assert steered.coeff == 2.0
        with torch.no_grad():
            out_inside = model(x)
        assert torch.allclose(out_inside[0], torch.full((1, 1, 4), -2.0))

    # After __exit__, hooks are removed.
    with torch.no_grad():
        out_outside = model(x)
    assert torch.allclose(out_outside[0], torch.zeros(1, 1, 4))


def test_steered_model_coeff_property(tmp_path: Path) -> None:
    from guard.model.steered import SteeredModel

    model = _TinyModel(hidden=4)
    sv = torch.ones(4)
    pt_path = tmp_path / "sv.pt"
    torch.save(sv, pt_path)

    with SteeredModel.from_sv_path(
        model, sv_path=pt_path, coeff=1.0,
        layer_idx=0, rotation_only=False,
    ) as steered:
        steered.coeff = 7.0
        x = torch.zeros(1, 1, 4)
        with torch.no_grad():
            out = model(x)
        assert torch.allclose(out[0], torch.full((1, 1, 4), -7.0))


def test_steered_model_getattr_delegation(tmp_path: Path) -> None:
    """SteeredModel should delegate unknown attributes to the wrapped model."""
    from guard.model.steered import SteeredModel

    model = _TinyModel(hidden=4)
    model.custom_attr = "hello"  # type: ignore[attr-defined]

    sv = torch.ones(4)
    pt_path = tmp_path / "sv.pt"
    torch.save(sv, pt_path)

    with SteeredModel.from_sv_path(
        model, sv_path=pt_path, coeff=0.0, layer_idx=0,
    ) as steered:
        assert steered.custom_attr == "hello"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------


def test_save_load_sv(tmp_path: Path) -> None:
    from guard.storage.io import load_sv, load_sv_with_meta, save_sv

    sv = torch.randn(32)
    metadata = {
        "token_position": -1,
        "sv_scaling": "activation_norm",
        "rotation_only": True,
        "projection_eps": 1e-6,
        "activation_norm": 5.0,
    }
    pt_path = save_sv(
        sv, output_dir=tmp_path, model_name="test/model",
        behavior="forget01", module_name="residual",
        layer_idx=8, method="orthogonal", metadata=metadata,
    )
    assert pt_path.exists()
    assert pt_path.with_suffix(".json").exists()

    loaded = load_sv(pt_path)
    assert loaded.dtype == torch.float32
    assert torch.allclose(sv.float(), loaded, atol=1e-6)

    loaded2, meta2 = load_sv_with_meta(pt_path)
    assert meta2["layer_idx"] == 8
    assert meta2["method"] == "orthogonal"
    assert meta2["sv_scaling"] == "activation_norm"


def test_save_sv_copies_config(tmp_path: Path) -> None:
    from guard.storage.io import save_sv

    sv = torch.randn(16)
    config_file = tmp_path / "gen.yaml"
    config_file.write_text("model_name: test\nlayers: [8]\n")

    pt_path = save_sv(
        sv, output_dir=tmp_path, model_name="test/m",
        behavior="forget01", module_name="residual",
        layer_idx=8, method="orthogonal",
        metadata={}, config_source=config_file,
    )
    yaml_copy = pt_path.parent / "generate.yaml"
    assert yaml_copy.exists()
    assert "model_name" in yaml_copy.read_text()

