"""Activation-space Euclidean scorer — negative L2 distance, no normalization."""


import torch

from guard.online.scoring.base import ActivationScorerBase

__all__ = ["ActivationEuclideanScorer"]


class ActivationEuclideanScorer(ActivationScorerBase):
    """Score hidden states by negative Euclidean distance to activation centroids.

    Vectors are NOT normalized — raw activation space distances are used.
    Higher score = closer centroid.

    Args:
        centroids: ``[K, H]`` centroids in activation space (unnormalized).
        token_position: Sequence position to probe. Negative values index from end.
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
        h_probe = hidden_states[:, tok_idx, :].float()  # [B, H] — unnormalized
        centroids = self._centroids.to(hidden_states.device)  # [K, H] — unnormalized
        dist = torch.cdist(h_probe, centroids, p=2)  # [B, K]
        return -dist  # higher = closer
