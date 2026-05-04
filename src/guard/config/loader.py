"""YAML → Pydantic config loaders."""

from pathlib import Path
from typing import Any

import yaml

from guard.config.gate import GateConfig
from guard.config.generation import GenerationConfig

__all__ = ["load_generation_config", "load_gate_config", "override_config"]


def _read_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level, got {type(data)}: {path}")
    return data


def load_generation_config(path: Path | str) -> GenerationConfig:
    """Load and validate a ``generate.yaml`` file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A fully-validated :class:`GenerationConfig` instance.

    Raises:
        pydantic.ValidationError: If the YAML is missing required fields or
            contains invalid values.
        FileNotFoundError: If ``path`` does not exist.
    """
    return GenerationConfig.model_validate(_read_yaml(path))


def load_gate_config(path: Path | str) -> GateConfig:
    """Load and validate a Similarity Gate config YAML.

    Args:
        path: Path to a YAML file whose top-level keys map to :class:`GateConfig` fields.

    Returns:
        A validated :class:`GateConfig` instance.
    """
    return GateConfig.model_validate(_read_yaml(path))


def override_config(
    cfg: GenerationConfig,
    overrides: list[str],
) -> GenerationConfig:
    """Apply ``key=value`` override strings to a :class:`GenerationConfig`.

    Supports nested keys via dot notation, e.g. ``normalizations.sv_scaling=unit``.
    Values are parsed as YAML scalars (so ``true``, ``42``, ``1e-6`` all work).

    Args:
        cfg: The base config to override.
        overrides: List of ``"key=value"`` strings.

    Returns:
        A new :class:`GenerationConfig` with the overrides applied.

    Raises:
        ValueError: If an override string is not in ``key=value`` format.
    """
    if not overrides:
        return cfg

    raw = cfg.model_dump()

    for ov in overrides:
        if "=" not in ov:
            raise ValueError(
                f"Override '{ov}' is not in 'key=value' format.  "
                "Example: --override batch_size=32"
            )
        key, _, value_str = ov.partition("=")
        parsed_value = yaml.safe_load(value_str)

        parts = key.strip().split(".")
        target: dict[str, Any] = raw
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(
                    f"Override key '{key}': '{part}' is not a nested mapping in the config."
                )
            target = target[part]  # type: ignore[assignment]
        target[parts[-1]] = parsed_value

    return GenerationConfig.model_validate(raw)
