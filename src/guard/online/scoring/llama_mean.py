"""Llama mean-pool scorer — L2-normalized mean of hidden states vs. centroids."""


import torch
import torch.nn.functional as F

from guard.online.scoring.base import ActivationScorerBase

__all__ = ["LlamaMeanScorer"]


class LlamaMeanScorer(ActivationScorerBase):
    """Score hidden states by cosine similarity after mean-pooling over all tokens.

    Computes the L2-normalized mean of the hidden states across the sequence
    dimension, then compares to pre-computed L2-normalized Llama centroids.
    Unlike :class:`ActivationCosineScorer`, this uses all tokens rather than
    a single probe position.

    Args:
        centroids: ``[K, H]`` L2-normalized Llama centroids (from clustering).
    """

    def __init__(self, centroids: torch.Tensor) -> None:
        self._centroids = F.normalize(centroids.float(), dim=-1).cpu()

    def score(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = hidden_states.float()
        if attention_mask is not None:
            m = attention_mask.to(h.device).unsqueeze(-1).float()  # [B, S, 1]
            mean_h = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            mean_h = h.mean(dim=1)
        mean_h = F.normalize(mean_h, dim=-1)
        centroids = self._centroids.to(hidden_states.device)
        return F.cosine_similarity(
            mean_h.unsqueeze(1),    # [B, 1, H]
            centroids.unsqueeze(0), # [1, K, H]
            dim=-1,
        )  # [B, K]
