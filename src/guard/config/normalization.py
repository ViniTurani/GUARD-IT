"""Normalization configuration for GUARD steering vectors."""

from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["SvScaling", "NormalizationConfig"]


class SvScaling(str, Enum):
    """How to scale the steering vector after computation.

    - ``none``: Keep the raw vector from the diff/orthogonalization step.
    - ``unit``: L2-normalize to unit norm.
    - ``activation_norm``: L2-normalize first, then rescale so the SV norm equals
      the mean activation norm of the forget (and retain) corpus.  This makes
      the effective coefficient model-agnostic across architectures with different
      hidden-state scales.
    """

    NONE = "none"
    UNIT = "unit"
    ACTIVATION_NORM = "activation_norm"


class NormalizationConfig(BaseModel):
    """Full normalization strategy — declared once, applied consistently.

    Generation-time flag (``sv_scaling``) shapes the saved SV.
    Eval-time flag (``rotation_only``) shapes how the SV is applied at inference.
    Both live here so a single YAML block captures the complete normalization
    strategy for a run.

    Defaults match the best-performing configuration from experiments.
    """

    model_config = ConfigDict(frozen=True)

    sv_scaling: SvScaling = Field(
        default=SvScaling.ACTIVATION_NORM,
        description=(
            "Scaling applied to the SV after computation. "
            "'activation_norm' (default) normalises to unit then rescales to "
            "match the mean corpus activation norm, making α model-agnostic."
        ),
    )
    rotation_only: bool = Field(
        default=True,
        description=(
            "If True (default), apply rotation-only steering at inference: "
            "h' = (h − α·sv) · (‖h‖ / ‖h − α·sv‖).  The hidden-state "
            "magnitude is preserved; only its direction changes (Eq. 8)."
        ),
    )
    projection_eps: float = Field(
        default=1e-6,
        gt=0.0,
        description="Epsilon added to all denominators to avoid division by zero.",
    )

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data: Any) -> Any:
        """Accept the old ``norm_activation_coeff`` key and remap it to ``rotation_only``."""
        if isinstance(data, dict) and "norm_activation_coeff" in data:
            logger.warning(
                "'norm_activation_coeff' is deprecated — rename to 'rotation_only' in your config."
            )
            data = {**data, "rotation_only": data.pop("norm_activation_coeff")}
        return data
