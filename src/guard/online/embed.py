"""Text embedder — sentence-transformer wrapper with process-level cache."""


import logging
from typing import Any

import torch

__all__ = ["TextEmbedder"]

logger = logging.getLogger(__name__)

# Loading a SentenceTransformer model takes ~4 s; reuse the same instance
# across all SteeredModel objects in one process to avoid redundant I/O.
_ST_MODEL_CACHE: dict[str, Any] = {}


class TextEmbedder:
    """Thin wrapper around a ``SentenceTransformer`` model with process-level caching.

    Embeddings are computed on CPU and returned as ``float32`` tensors.

    Args:
        model_name: SentenceTransformer model identifier (default ``'all-MiniLM-L6-v2'``).

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Text routing requires 'sentence-transformers'.  "
                "Install with: pip install sentence-transformers  "
                "or: pip install guard[gateway]"
            ) from exc

        if model_name not in _ST_MODEL_CACHE:
            logger.info("Loading SentenceTransformer '%s'.", model_name)
            _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
        self._model = _ST_MODEL_CACHE[model_name]
        self.model_name = model_name

    def encode(self, texts: list[str], normalize: bool = True) -> torch.Tensor:
        """Encode a list of texts into embeddings.

        Args:
            texts: Input strings.
            normalize: L2-normalise output so cosine sim = dot product (default True).

        Returns:
            ``float32`` tensor of shape ``[len(texts), emb_dim]`` on CPU.
        """
        embs: torch.Tensor = self._model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return embs.float().cpu()
