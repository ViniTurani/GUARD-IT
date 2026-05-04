"""Offline phase — precomputed once per forget corpus."""

from guard.offline.activations import (
    extract_all_activations_multilayer,
    extract_embedding_layer,
    extract_mean_activations_multilayer,
)
from guard.offline.steering import compute_steering_vector

__all__ = [
    "extract_mean_activations_multilayer",
    "extract_all_activations_multilayer",
    "extract_embedding_layer",
    "compute_steering_vector",
]
