"""Cosine similarity scorer — sentence-transformer embeddings vs. per-cluster text centroids."""


from typing import Any

import torch
import torch.nn.functional as F

from guard.online.scoring.base import TextScorerBase

__all__ = ["TextCosineScorer"]


class TextCosineScorer(TextScorerBase):
    """Score inputs by cosine similarity against pre-computed text-space centroids.

    Args:
        text_centroids: ``[K, emb_dim]`` per-cluster centroids (any dtype — cast to float32).
        st_model_name: Name of the sentence-transformer model to load lazily.
    """

    def __init__(self, text_centroids: torch.Tensor, st_model_name: str) -> None:
        from guard.online.embed import TextEmbedder

        self._centroids = text_centroids.float().cpu()
        self._embedder = TextEmbedder(st_model_name)

    def score(self, texts: list[str]) -> torch.Tensor:
        embs: Any = self._embedder.encode(texts)  # [B, E] — L2-normed, CPU
        sim: torch.Tensor = F.cosine_similarity(
            embs.unsqueeze(1),          # [B, 1, E]
            self._centroids.unsqueeze(0),  # [1, K, E]
            dim=-1,
        )
        return sim

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Expose the underlying embedder for callers that need raw embeddings
        (e.g. the retain gate re-uses them against a separate centroid)."""
        embs: Any = self._embedder.encode(texts)
        return embs  # type: ignore[no-any-return]
