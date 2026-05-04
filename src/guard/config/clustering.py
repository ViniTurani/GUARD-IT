"""Clustering configuration — embedded in GenerationConfig for `guard cluster`."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ClusteringConfig"]


class ClusteringConfig(BaseModel):
    """Clustering parameters for forget-doc partitioning.

    Embedded in :class:`~guard.config.GenerationConfig` under the ``clustering:`` key.

    Forget docs are clustered via KMeans on MiniLM embeddings into K_f clusters.
    Each cluster gets its own steering vector computed against the full retain corpus.
    The inference-time routing threshold (min cosim to activate a cluster's SV)
    lives in ``GateConfig.threshold``, not here.

    Example YAML::

        clustering:
          n_clusters: auto
          k_min: 2
          k_max: 10
          embed_model: all-MiniLM-L6-v2
    """

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------
    # Forget clustering
    # ------------------------------------------------------------------
    n_clusters: int | Literal["auto"] = Field(
        default="auto",
        description=(
            "Number of forget clusters.  'auto' selects K with the highest silhouette "
            "score over [k_min, k_max]."
        ),
    )
    k_min: int = Field(default=2, ge=2, description="Min K for forget silhouette sweep.")
    k_max: int = Field(default=20, ge=2, description="Max K for forget silhouette sweep.")
    k_candidates: list[int] | None = Field(
        default=None,
        description=(
            "When set, restricts the forget silhouette sweep to this explicit list "
            "(ignoring k_min/k_max).  Only used when n_clusters='auto'."
        ),
    )

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------
    calibration_alpha: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight on retain false-positive rate when calibrating routing threshold. "
            "Objective: recall_forget - alpha * fpr_retain. "
            "alpha=1.0 is the Youden index (balanced). "
            "alpha>1 is more conservative: penalises activating on retain inputs more. "
            "Ignored when calibration_method='retain_percentile'."
        ),
    )
    calibration_method: str = Field(
        default="youden_alpha",
        description=(
            "'youden_alpha' (default): threshold argmax of recall_forget - alpha*fpr_retain. "
            "'retain_percentile': threshold = quantile of retain_scores such that at most "
            "calibration_retain_fpr fraction of retain docs activate — guarantees a fixed "
            "false-positive rate on the retain corpus, independent of forget distribution."
        ),
    )
    calibration_retain_fpr: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Target FPR on retain when calibration_method='retain_percentile'. "
            "threshold = quantile(retain_scores, 1 - calibration_retain_fpr)."
        ),
    )

    # ------------------------------------------------------------------
    # Clustering algorithm
    # ------------------------------------------------------------------
    use_kmedoids: bool = Field(
        default=False,
        description=(
            "Use k-medoids instead of k-means.  Medoids are actual data points, "
            "which can produce more representative clusters for skewed distributions."
        ),
    )
    normalize_embeddings_for_clustering: bool = Field(
        default=True,
        description=(
            "L2-normalise MiniLM embeddings before clustering.  Set to False to cluster "
            "in raw embedding space (Euclidean distances on unnormalised vectors). "
            "Routing centroids are always normalised regardless of this setting."
        ),
    )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    embed_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name used for text embedding.",
    )
