"""BM25 scorer — lexical retrieval over per-cluster document bags."""

import math
import re
from collections import Counter
from typing import TypedDict

import numpy as np
import torch

from guard.online.scoring.base import TextScorerBase

__all__ = ["TextBM25Scorer"]


class _BM25Index(TypedDict):
    tf_list: list[dict[str, int]]
    lens: list[int]
    avgdl: float
    idf: dict[str, float]
    N: int
    k1: float
    b: float


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TextBM25Scorer(TextScorerBase):
    """BM25 lexical scorer with per-query softmax normalization.

    Each cluster is treated as a single concatenated document. The raw BM25
    scores for a query are softmax-normalized so values land in ``[0, 1]`` and
    become comparable against ``GateConfig.threshold``: a confident match
    produces a concentrated distribution (max near 1); ambiguous queries flatten
    out toward ``1/K`` and fail higher thresholds.

    Args:
        cluster_texts: Per-cluster documents. All docs for cluster ``k`` are
            concatenated into a single pseudo-document before indexing.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalization parameter.
    """

    def __init__(self, cluster_texts: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self._index = self._build_index(cluster_texts, k1=k1, b=b)

    @staticmethod
    def _build_index(cluster_texts: list[list[str]], *, k1: float, b: float) -> _BM25Index:
        docs = [" ".join(c) if c else "" for c in cluster_texts]
        N = len(docs)
        tf_list: list[dict[str, int]] = []
        lens: list[int] = []
        df: dict[str, int] = {}
        for doc in docs:
            toks = _tokenize(doc)
            tf = Counter(toks)
            tf_list.append(dict(tf))
            lens.append(sum(tf.values()))
            for term in tf:
                df[term] = df.get(term, 0) + 1

        avgdl = float(sum(lens)) / N if N > 0 else 0.0
        idf = {term: math.log((N - dfv + 0.5) / (dfv + 0.5) + 1.0) for term, dfv in df.items()}
        return {
            "tf_list": tf_list,
            "lens": lens,
            "avgdl": avgdl,
            "idf": idf,
            "N": N,
            "k1": float(k1),
            "b": float(b),
        }

    def _raw_scores(self, queries: list[str]) -> np.ndarray:
        tf_list = self._index["tf_list"]
        lens = self._index["lens"]
        avgdl = self._index["avgdl"]
        idf = self._index["idf"]
        k1 = self._index["k1"]
        b = self._index["b"]

        B = len(queries)
        K = len(tf_list)
        out = np.zeros((B, K), dtype=float)
        for i, q in enumerate(queries):
            for term in set(_tokenize(q)):
                idfv = idf.get(term, 0.0)
                if idfv == 0.0:
                    continue
                for j in range(K):
                    tf = tf_list[j].get(term, 0)
                    if tf == 0:
                        continue
                    norm = lens[j] / avgdl if avgdl > 0 else 0.0
                    denom = tf + k1 * (1.0 - b + b * norm)
                    out[i, j] += idfv * (tf * (k1 + 1.0)) / (denom if denom != 0 else 1.0)
        return out

    def score(self, texts: list[str]) -> torch.Tensor:
        raw = self._raw_scores(texts)
        # Softmax maps arbitrary BM25 magnitudes into [0, 1] with sum=1 per row
        # so threshold comparisons behave consistently across queries.
        shifted = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return torch.tensor(probs, dtype=torch.float32)
