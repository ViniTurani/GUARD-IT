"""Similarity Gate configuration — cosine-similarity routing for GUARD."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["GateConfig"]


class GateConfig(BaseModel):
    """Configuration for the Similarity Gate that routes inputs to cluster PSVs.

    When ``enabled=True``, each forward pass embeds the input (text or activation)
    and computes cosine similarity to pre-computed cluster centroids.  The gate
    decides whether to steer and, if so, which cluster PSVs to combine.

    Text routing (``routing_source='text'``) is the recommended mode:
    sentence-transformer embeddings are computed once per ``generate()`` call
    and cached across all autoregressive steps.

    Activation routing (``routing_source='activation'``) uses the hidden state
    at ``token_position`` inside the hook — no extra dependencies, but computed
    on every forward pass.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="Enable the Similarity Gate.  False = apply SV to all inputs unconditionally.",
    )
    routing_source: Literal["text", "activation", "activation_euclidean", "llama_mean"] = Field(
        default="text",
        description=(
            "'text': embed the input with a SentenceTransformer model and compare "
            "to pre-computed text centroids. "
            "'activation': compare the hidden state at token_position to "
            "activation-space centroids (no extra dependencies)."
        ),
    )
    threshold: float = Field(
        default=0.5,
        description=(
            "Minimum cosine similarity to any cluster centroid to trigger steering (T). "
            "Inputs below this threshold are passed through unchanged. "
            "For Euclidean routing: negative distance (unbounded)."
        ),
    )
    routing_mode: Literal[
        "best",
        "threshold",
        "weighted_threshold",
        "bm25",
        "soft",
        "ratio",
        "dissimilar_threshold",
    ] = Field(
        default="threshold",
        description=(
            "'threshold' (default): average all cluster PSVs whose cosim exceeds "
            "``threshold``; skip if none qualify. "
            "'weighted_threshold': like threshold, but PSVs are weighted by cosine similarity. "
            "'best': always steer using the single closest cluster's PSV. "
            "'bm25': use BM25 retrieval over cluster texts to score clusters (text routing). "
            "'ratio': route when sim_forget/sim_retain > threshold (requires retain_centroid). "
            "'dissimilar_threshold': use up to `dissimilar_top_k` clusters with sim <= threshold; "
            "if none match, fallback to `dissimilar_fallback_k` farthest clusters."
        ),
    )
    dissimilar_top_k: int = Field(
        default=4,
        ge=1,
        description=(
            "For routing_mode='dissimilar_threshold': max number of dissimilar clusters "
            "(lowest similarity) to average when sim <= threshold."
        ),
    )
    dissimilar_fallback_k: int = Field(
        default=4,
        ge=1,
        description=(
            "For routing_mode='dissimilar_threshold': if no cluster satisfies sim <= threshold, "
            "average the K farthest clusters (lowest similarity)."
        ),
    )
    st_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model used for text-space routing.",
    )
    retain_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "If > 0, skip steering when the input's cosim to the retain centroid "
            "exceeds this value.  0.0 disables the retain gate."
        ),
    )
    token_position: int = Field(
        default=-1,
        description="Token index used for activation-space routing (negative = from end).",
    )
    log_routing: bool = Field(
        default=False,
        description="Print per-batch routing diagnostics (useful for debugging).",
    )
