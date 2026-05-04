"""Cluster scorers — pluggable strategies for routing and ranking."""

from guard.online.scoring.activation import ActivationCosineScorer
from guard.online.scoring.base import (
    ActivationScorerBase,
    ActivationScorerProtocol,
    TextScorerBase,
    TextScorerProtocol,
)
from guard.online.scoring.bm25 import TextBM25Scorer
from guard.online.scoring.cosine import TextCosineScorer
from guard.online.scoring.euclidean import ActivationEuclideanScorer
from guard.online.scoring.llama_mean import LlamaMeanScorer

__all__ = [
    "ActivationCosineScorer",
    "ActivationEuclideanScorer",
    "LlamaMeanScorer",
    "ActivationScorerBase",
    "ActivationScorerProtocol",
    "TextBM25Scorer",
    "TextCosineScorer",
    "TextScorerBase",
    "TextScorerProtocol",
]
