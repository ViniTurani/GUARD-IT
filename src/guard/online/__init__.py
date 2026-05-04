"""Online phase — per inference call."""

from guard.online.embed import TextEmbedder
from guard.online.gate import SimilarityGate
from guard.online.hook import register_steering_hook
from guard.online.registry import get_target_module

__all__ = [
    "TextEmbedder",
    "SimilarityGate",
    "register_steering_hook",
    "get_target_module",
]
