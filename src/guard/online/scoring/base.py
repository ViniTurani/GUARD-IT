"""Protocols and ABCs for scorers (text-space and activation-space)."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch

__all__ = [
    "ActivationScorerBase",
    "ActivationScorerProtocol",
    "TextScorerBase",
    "TextScorerProtocol",
]


class TextScorerBase(ABC):
    """Abstract base class for text scorers."""

    @abstractmethod
    def score(self, texts: list[str]) -> torch.Tensor:
        """Return ``[B, K]`` scores, higher = more relevant cluster."""
        ...


class ActivationScorerBase(ABC):
    """Abstract base class for activation-space scorers."""

    @abstractmethod
    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return ``[B, K]`` scores from hidden states ``[B, S, H]``."""
        ...


@runtime_checkable
class TextScorerProtocol(Protocol):
    """Scores each input **text** against each cluster.

    Given a batch of input texts, returns a ``[B, K]`` tensor where higher
    means more relevant. Values live on CPU; the router moves them to device.

    To plug in a new text-routing strategy, implement this protocol and pass
    the instance to :class:`~guard.online.gate.SimilarityGate` via
    ``scorer=``. Scores are compared against
    :attr:`GateConfig.threshold`, so most scorers normalize to ``[0, 1]``
    (e.g. cosine similarity, softmax).
    """

    def score(self, texts: list[str]) -> torch.Tensor:
        """Return ``[B, K]`` scores, higher = more relevant cluster."""
        ...


@runtime_checkable
class ActivationScorerProtocol(Protocol):
    """Scores each input's **hidden state** against each cluster.

    Given a hidden-state tensor ``[B, S, H]``, returns a ``[B, K]`` tensor
    where higher means more relevant. Runs inside the forward hook, so the
    output must live on the same device as the input.

    To plug in a new activation-routing strategy, implement this protocol and
    pass the instance to :class:`~guard.online.gate.SimilarityGate` via
    ``activation_scorer=``.
    """

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return ``[B, K]`` scores from hidden states ``[B, S, H]``."""
        ...
