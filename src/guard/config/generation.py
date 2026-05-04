"""Generation config — drives `guard generate`."""

import os
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from guard.config.clustering import ClusteringConfig
from guard.config.normalization import NormalizationConfig

__all__ = [
    "TofuDatasetConfig",
    "LocalJSONLDatasetConfig",
    "DatasetConfig",
    "GenerationConfig",
    "ClusteringConfig",
]


# ---------------------------------------------------------------------------
# Dataset configs — discriminated by `type`
# ---------------------------------------------------------------------------


class TofuDatasetConfig(BaseModel):
    """TOFU benchmark dataset (locuslab/TOFU on HuggingFace).

    The retain split is inferred automatically from the forget split
    (e.g. ``forget01`` → ``retain99``) unless ``retain_split`` is set.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tofu"] = "tofu"
    forget_split: str = Field(
        default="forget01",
        description="TOFU forget split name: forget01, forget05, or forget10.",
    )
    retain_split: str | None = Field(
        default=None,
        description="Retain split name.  Inferred from forget_split when None.",
    )
    limit: int | None = Field(
        default=None,
        gt=0,
        description="Truncate both forget and retain lists to this many samples.",
    )

    def inferred_retain_split(self) -> str:
        """Return the retain split, inferring from forget_split if needed."""
        if self.retain_split is not None:
            return self.retain_split
        try:
            pct = int(self.forget_split.replace("forget", ""))
            return f"retain{100 - pct}"
        except ValueError as exc:
            raise ValueError(
                f"Cannot infer retain split from forget_split='{self.forget_split}'. "
                "Provide retain_split explicitly."
            ) from exc


class LocalJSONLDatasetConfig(BaseModel):
    """Forget/retain documents loaded from local JSONL files.

    Used for per-cluster steering vector generation, where each cluster
    has its own forget JSONL.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["local_jsonl"] = "local_jsonl"
    forget_jsonl: str = Field(description="Path to the forget JSONL file.")
    retain_jsonl: str = Field(description="Path to the retain JSONL file.")
    text_key: str = Field(
        default="text",
        description="JSON field to use as the document text.",
    )
    limit: int | None = Field(
        default=None,
        gt=0,
        description="Truncate both lists to this many samples.",
    )


#: Discriminated union — Pydantic selects the correct subtype via the ``type`` field.
DatasetConfig = Annotated[
    TofuDatasetConfig | LocalJSONLDatasetConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Generation config — the top-level YAML schema
# ---------------------------------------------------------------------------


class GenerationConfig(BaseModel):
    """Complete configuration for one steering vector generation run.

    This is the schema validated when loading a ``generate.yaml`` file.
    Every field that can have a sensible default does, so a minimal config
    only needs to specify ``model_name``, ``layers``, and ``dataset``.

    Example YAML::

        model_name: open-unlearning/tofu_Llama-3.2-1B-Instruct_full
        layers: [8]
        method: orthogonal
        dataset:
          type: tofu
          forget_split: forget01
    """

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------
    model_name: str = Field(
        description="HuggingFace model ID or local path to load with AutoModelForCausalLM."
    )
    layers: list[int] = Field(
        min_length=1,
        description=(
            "Transformer layer indices to compute steering vectors for. "
            "All layers are captured in a single forward pass."
        ),
    )

    # ------------------------------------------------------------------
    # SUV method
    # ------------------------------------------------------------------
    method: Literal[
        "diff_means",
        "orthogonal",
        "orth_diff_means",
    ] = Field(
        default="orthogonal",
        description=(
            "'orthogonal' (default): project forget mean orthogonal to retain mean. "
            "'diff_means': simple forget_mean − retain_mean. "
            "'orth_diff_means': diff_means then projected orthogonal to retain mean."
        ),
    )
    layer_methods: dict[int, str] | None = Field(
        default=None,
        description=(
            "Per-layer method overrides. Maps layer index (int) to method name. "
            "Layers not listed here use the global `method`. "
            "SVs are saved under the global method directory; effective method is "
            "recorded in sv.json metadata."
        ),
    )
    module_name: Literal["residual", "mlp", "self_attn"] = Field(
        default="residual",
        description=(
            "Which module to hook inside each transformer layer. "
            "'residual' hooks the full decoder layer output."
        ),
    )
    token_position: int | Literal["mean", "pooling", "max", "last_content", "last_non_special"] = Field(
        default=-1,
        description=(
            "Token position to extract activations from. "
            "Negative integers count from the end (−1 = last token). "
            "'mean' / 'pooling' average across all non-padding tokens. "
            "'max' takes the element-wise max across non-padding tokens."
        ),
    )
    add_special_tokens: bool = Field(
        default=True,
        description="Whether to add special tokens when tokenising documents.",
    )
    apply_chat_template: bool = Field(
        default=False,
        description=(
            "When True, wrap each (question, answer) pair with the model's chat template "
            "before extracting activations.  Aligns the SV activation distribution with "
            "inference-time activations (which always see the chat template).  "
            "Only meaningful for instruction-tuned models with a chat template."
        ),
    )
    cluster_source: str = Field(
        default="text",
        description=(
            "Source of embeddings used for clustering forget/retain docs. "
            "'text': MiniLM sentence embeddings (default). "
            "'activation': LLM hidden states at token_position, unnormalized (Euclidean KMeans). "
            "'llama_mean': LLM mean-pooled hidden states, L2-normalized (cosine KMeans)."
        ),
    )
    cluster_layer: int | str | None = Field(
        default=None,
        description=(
            "Layer index to extract clustering activations from when "
            "cluster_source ∈ {activation, llama_mean}.  Defaults to layers[0] when None. "
            "Special values: 'mean_all' averages across all transformer layers; "
            "'embed' uses the input embedding layer (model.model.embed_tokens) directly."
        ),
    )

    cluster_with_template: bool = Field(
        default=True,
        description=(
            "Controls whether MiniLM embeddings (used for clustering) are computed on "
            "template-formatted text.  Only has effect when apply_chat_template=True.  "
            "When False, clustering uses plain 'question\\nanswer' text while activation "
            "capture for SV computation still uses the chat template.  This separates the "
            "semantic clustering signal (content) from the SV direction (inference context)."
        ),
    )

    # ------------------------------------------------------------------
    # Normalizations
    # ------------------------------------------------------------------
    normalizations: NormalizationConfig = Field(
        default_factory=NormalizationConfig,
        description="SV scaling and eval-time rotation settings.",
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset: DatasetConfig = Field(
        default_factory=TofuDatasetConfig,
        description="Dataset configuration (TOFU or local JSONL).",
    )

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig,
        description="Parameters for forget-doc clustering and per-cluster SV generation.",
    )

    # ------------------------------------------------------------------
    # Inference / batching
    # ------------------------------------------------------------------
    batch_size: int = Field(
        default=128,
        gt=0,
        description=(
            "Number of documents per forward pass during activation capture. "
            "Default 128 is tuned for A6000 49GB + 1B bfloat16 model."
        ),
    )
    max_length: int = Field(
        default=512,
        gt=0,
        description="Tokeniser truncation length.",
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="DataLoader workers for tokenisation.  0 = main process only.",
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = Field(
        default="steering_vectors/",
        description=(
            "Root directory for saved steering vectors.  "
            "Actual path: {output_dir}/{model_name_sanitized}/{method}/{behavior}/"
            "{module_name}/layer_{idx}/"
        ),
    )
    behavior: str | None = Field(
        default=None,
        description=(
            "Experiment label used in the output path.  "
            "Defaults to the dataset forget split name (e.g. 'forget01')."
        ),
    )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    cuda: str | None = Field(
        default=None,
        description="CUDA_VISIBLE_DEVICES override (e.g. '0', '2,3').",
    )

    quantization: Literal["none", "4bit", "8bit"] = Field(
        default="none",
        description=(
            "Optional model quantization mode for model loading. "
            "Use '4bit' or '8bit' (bitsandbytes) on CUDA to reduce VRAM."
        ),
    )
    quant_4bit_type: Literal["nf4", "fp4"] = Field(
        default="nf4",
        description="4-bit quantization format when quantization='4bit'.",
    )
    quant_4bit_double_quant: bool = Field(
        default=True,
        description="Enable nested/double quantization when quantization='4bit'.",
    )
    quant_compute_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="Compute dtype for quantized ops (4-bit mode).",
    )

    @field_validator("cuda", mode="before")
    @classmethod
    def _coerce_cuda(cls, v: object) -> str | None:
        """Accept cuda: 1 (int) or cuda: "1" (str) in YAML."""
        if v is None:
            return None
        return str(v)

    seed: int = Field(default=42, description="Random seed for reproducibility.")
    limit: int | None = Field(
        default=None,
        gt=0,
        description="Global document cap; overrides dataset-level limit.",
    )

    def effective_behavior(self) -> str:
        """Return the behavior label, falling back to the dataset forget split."""
        if self.behavior is not None:
            return self.behavior
        ds = self.dataset
        if isinstance(ds, TofuDatasetConfig):
            return ds.forget_split
        # LocalJSONL — derive from forget_jsonl filename
        return os.path.splitext(os.path.basename(ds.forget_jsonl))[0]
