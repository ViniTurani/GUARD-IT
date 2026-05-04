"""Tests for config loading, validation, and override."""

import textwrap
from pathlib import Path

import pytest
import yaml

from guard.config.generation import (
    GenerationConfig,
    LocalJSONLDatasetConfig,
    TofuDatasetConfig,
)
from guard.config.loader import load_generation_config, override_config
from guard.config.normalization import NormalizationConfig, SvScaling


# ---------------------------------------------------------------------------
# NormalizationConfig
# ---------------------------------------------------------------------------


def test_normalization_defaults() -> None:
    cfg = NormalizationConfig()
    assert cfg.sv_scaling == SvScaling.ACTIVATION_NORM
    assert cfg.rotation_only is True
    assert cfg.projection_eps == 1e-6


def test_normalization_legacy_key_remapped() -> None:
    cfg = NormalizationConfig.model_validate({"norm_activation_coeff": False})
    assert cfg.rotation_only is False


def test_normalization_frozen() -> None:
    cfg = NormalizationConfig()
    with pytest.raises(Exception):  # ValidationError or AttributeError depending on Pydantic
        cfg.sv_scaling = SvScaling.NONE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GenerationConfig round-trip
# ---------------------------------------------------------------------------


def test_generation_config_minimal() -> None:
    cfg = GenerationConfig.model_validate({
        "model_name": "test/model",
        "layers": [8],
    })
    assert cfg.model_name == "test/model"
    assert cfg.layers == [8]
    assert cfg.method == "orthogonal"
    assert cfg.normalizations.sv_scaling == SvScaling.ACTIVATION_NORM


def test_generation_config_tofu_dataset() -> None:
    cfg = GenerationConfig.model_validate({
        "model_name": "test/model",
        "layers": [8],
        "dataset": {"type": "tofu", "forget_split": "forget10"},
    })
    assert isinstance(cfg.dataset, TofuDatasetConfig)
    assert cfg.dataset.forget_split == "forget10"
    assert cfg.dataset.inferred_retain_split() == "retain90"


def test_generation_config_local_jsonl() -> None:
    cfg = GenerationConfig.model_validate({
        "model_name": "test/model",
        "layers": [8],
        "dataset": {
            "type": "local_jsonl",
            "forget_jsonl": "/tmp/forget.jsonl",
            "retain_jsonl": "/tmp/retain.jsonl",
        },
    })
    assert isinstance(cfg.dataset, LocalJSONLDatasetConfig)
    assert cfg.dataset.forget_jsonl == "/tmp/forget.jsonl"


def test_generation_config_effective_behavior_default() -> None:
    cfg = GenerationConfig.model_validate({
        "model_name": "test/model",
        "layers": [4],
        "dataset": {"type": "tofu", "forget_split": "forget05"},
    })
    assert cfg.effective_behavior() == "forget05"


def test_generation_config_effective_behavior_override() -> None:
    cfg = GenerationConfig.model_validate({
        "model_name": "test/model",
        "layers": [4],
        "behavior": "custom_label",
    })
    assert cfg.effective_behavior() == "custom_label"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def test_load_generation_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        model_name: test/model
        layers: [8, 16]
        method: diff_means
        dataset:
          type: tofu
          forget_split: forget01
        normalizations:
          sv_scaling: unit
          rotation_only: false
    """)
    config_file = tmp_path / "gen.yaml"
    config_file.write_text(yaml_content)

    cfg = load_generation_config(config_file)
    assert cfg.layers == [8, 16]
    assert cfg.method == "diff_means"
    assert cfg.normalizations.sv_scaling == SvScaling.UNIT
    assert cfg.normalizations.rotation_only is False


def test_load_generation_config_missing_required(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("method: orthogonal\n")  # missing model_name and layers
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        load_generation_config(config_file)


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------


def test_override_scalar() -> None:
    base = GenerationConfig.model_validate({"model_name": "test/m", "layers": [8]})
    new = override_config(base, ["batch_size=32"])
    assert new.batch_size == 32
    assert base.batch_size == 128


def test_override_nested() -> None:
    base = GenerationConfig.model_validate({"model_name": "test/m", "layers": [8]})
    new = override_config(base, ["normalizations.sv_scaling=none"])
    assert new.normalizations.sv_scaling == SvScaling.NONE


def test_override_bad_format() -> None:
    base = GenerationConfig.model_validate({"model_name": "test/m", "layers": [8]})
    with pytest.raises(ValueError, match="key=value"):
        override_config(base, ["batch_size32"])  # missing '='


# ---------------------------------------------------------------------------
# TofuDatasetConfig retain inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forget,expected_retain", [
    ("forget01", "retain99"),
    ("forget05", "retain95"),
    ("forget10", "retain90"),
])
def test_retain_split_inference(forget: str, expected_retain: str) -> None:
    cfg = TofuDatasetConfig(forget_split=forget)
    assert cfg.inferred_retain_split() == expected_retain


def test_retain_split_inference_custom() -> None:
    cfg = TofuDatasetConfig(forget_split="forget01", retain_split="retain50")
    assert cfg.inferred_retain_split() == "retain50"
