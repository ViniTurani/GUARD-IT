"""Activation-space cosine scorer — hidden-state probe token vs. centroids."""


import torch
import torch.nn.functional as F

from guard.online.scoring.base import ActivationScorerBase

__all__ = ["ActivationCosineScorer"]


class ActivationCosineScorer(ActivationScorerBase):
    """Score hidden states by cosine similarity against activation centroids.

    Picks a single probe token at ``token_position`` from each sequence and
    compares it to ``K`` pre-computed cluster centroids. Used for both the
    main routing signal (``K`` cluster centroids) and the retain gate
    (``K=1`` retain centroid — the caller squeezes the last dim).

    Args:
        centroids: ``[K, H]`` centroids in activation space (any dtype —
            cast to float32). Kept on CPU until use; moved to device per
            forward pass.
        token_position: Sequence position to probe. Negative values index
            from the end (``-1`` = last token).
    """

    def __init__(self, centroids: torch.Tensor, token_position: int) -> None:
        self._centroids = centroids.float().cpu()
        self._token_position = token_position

    @staticmethod
    def _probe_index(seq_len: int, token_position: int) -> int:
        idx = seq_len + token_position if token_position < 0 else token_position
        return max(0, min(seq_len - 1, idx))

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = int(hidden_states.shape[1])
        tok_idx = self._probe_index(seq_len, self._token_position)
        h_probe = hidden_states[:, tok_idx, :].float()  # [B, H]
        centroids = self._centroids.to(hidden_states.device)
        return F.cosine_similarity(
            h_probe.unsqueeze(1),   # [B, 1, H]
            centroids.unsqueeze(0), # [1, K, H]
            dim=-1,
        )  # [B, K]
