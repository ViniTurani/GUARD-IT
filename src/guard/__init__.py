"""GUARD — Gated Unlearning via Activation Redirection.

Clustered steering vectors with cosine-similarity Similarity Gate routing,
full Pydantic v2 configuration, and a single HuggingFace-compatible
entry point for TOFU and MUSE benchmarks.
"""

from guard._version import __version__

# Config
from guard.config import (
    DatasetConfig,
    GateConfig,
    GenerationConfig,
    LocalJSONLDatasetConfig,
    NormalizationConfig,
    SvScaling,
    TofuDatasetConfig,
    load_gate_config,
    load_generation_config,
    override_config,
)

# Offline phase
from guard.offline import (
    compute_steering_vector,
    extract_mean_activations_multilayer,
)

# Storage
from guard.storage import (
    documents_hash,
    load_sv,
    load_sv_with_meta,
    save_sv,
    sv_path,
)

# Online phase
from guard.online import (
    SimilarityGate,
    TextEmbedder,
    get_target_module,
    register_steering_hook,
)

# Model
from guard.model import SteeredModel

__all__ = [
    "__version__",
    # Config
    "SvScaling",
    "NormalizationConfig",
    "TofuDatasetConfig",
    "LocalJSONLDatasetConfig",
    "DatasetConfig",
    "GenerationConfig",
    "GateConfig",
    "load_generation_config",
    "load_gate_config",
    "override_config",
    # Offline phase
    "extract_mean_activations_multilayer",
    "compute_steering_vector",
    # Storage
    "sv_path",
    "save_sv",
    "load_sv",
    "load_sv_with_meta",
    "documents_hash",
    # Online phase
    "TextEmbedder",
    "SimilarityGate",
    "get_target_module",
    "register_steering_hook",
    # Model
    "SteeredModel",
]
