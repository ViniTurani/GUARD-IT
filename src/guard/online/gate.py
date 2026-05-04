"""SimilarityGate — pluggable cluster routing for GUARD.

Text routing is delegated to a :class:`~guard.online.scoring.TextScorerProtocol`
(cosine, BM25, …) — add a new method by dropping in a scorer class.
Activation-space routing (hidden-state cosine vs. activation centroids) is
handled directly inside the hook.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F
from loguru import logger

from guard.online.scoring import (
    ActivationCosineScorer,
    ActivationEuclideanScorer,
    ActivationScorerProtocol,
    LlamaMeanScorer,
    TextBM25Scorer,
    TextCosineScorer,
    TextScorerProtocol,
)

__all__ = ["SimilarityGate"]


class SimilarityGate:
    """Routes each input to the most-similar cluster's PSV (Partial Steering Vector).

    Implements the online-phase Similarity Gate: embeds the input, computes cosine
    similarity to cluster centroids, and selects the active set K(x) according to
    the configured ``routing_mode`` and ``threshold`` T.

    Supports two routing sources:

    * ``'text'`` — delegates scoring to a :class:`TextScorerProtocol`
      (``cosine`` or ``bm25``), computed once per ``generate()`` call and cached
      for all autoregressive steps.
    * ``'activation'`` — cosine similarity between the hidden state at
      ``token_position`` and activation-space centroids, computed inside the
      hook on every forward pass.

    An optional **retain gate** skips steering entirely when the input's cosim to
    the retain centroid exceeds ``retain_threshold``.

    Args:
        centroids: ``[K, H]`` activation-space cluster centroids.
        psv_clusters: ``[K, H]`` one PSV per forget cluster.
        gate_cfg: :class:`~guard.config.GateConfig` instance.
        text_centroids: ``[K, E]`` per-cluster ST embeddings (cosine scoring).
        tokenizer: HF tokenizer used to decode ``input_ids`` for text routing.
            Required when ``routing_source='text'``.
        retain_centroid: Optional ``[H]`` activation-space retain centroid.
        retain_text_centroid: Optional ``[E]`` text-space retain centroid.
        cluster_texts: Per-cluster forget docs. Required for BM25 routing.
        scorer: Pre-built :class:`TextScorerProtocol`. Overrides the built-in
            text scorer — use this to plug in custom routing strategies.
        activation_scorer: Pre-built :class:`ActivationScorerProtocol`. Overrides
            the built-in activation cosine scorer.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        psv_clusters: torch.Tensor,
        gate_cfg: Any,  # GateConfig
        text_centroids: torch.Tensor | None = None,
        tokenizer: Any | None = None,
        retain_centroid: torch.Tensor | None = None,
        retain_text_centroid: torch.Tensor | None = None,
        cluster_texts: list[list[str]] | None = None,
        scorer: TextScorerProtocol | None = None,
        activation_scorer: ActivationScorerProtocol | None = None,
        llama_centroids: torch.Tensor | None = None,
        rotation_only: bool = True,
    ) -> None:
        self._cfg = gate_cfg
        self._rotation_only = rotation_only
        self._centroids_cpu = centroids.float().cpu()
        self._psv_clusters_cpu = psv_clusters.float().cpu()
        self._tokenizer = tokenizer

        self._scorer: TextScorerProtocol | None = None
        if gate_cfg.routing_source == "text":
            if tokenizer is None:
                raise ValueError("routing_source='text' requires tokenizer.")
            self._scorer = scorer or self._default_scorer(
                gate_cfg, text_centroids, cluster_texts
            )

        if activation_scorer is not None:
            self._activation_scorer = activation_scorer
        elif gate_cfg.routing_source == "activation_euclidean":
            self._activation_scorer = ActivationEuclideanScorer(
                self._centroids_cpu, gate_cfg.token_position
            )
        elif gate_cfg.routing_source == "llama_mean":
            if llama_centroids is None:
                raise ValueError("routing_source='llama_mean' requires llama_centroids.")
            self._activation_scorer = LlamaMeanScorer(llama_centroids.float().cpu())
        else:
            self._activation_scorer = ActivationCosineScorer(
                self._centroids_cpu, gate_cfg.token_position
            )

        self._retain_centroid_cpu = (
            retain_centroid.float().cpu() if retain_centroid is not None else None
        )
        self._retain_text_centroid_cpu = (
            retain_text_centroid.float().cpu() if retain_text_centroid is not None else None
        )
        self._retain_activation_scorer = (
            ActivationCosineScorer(
                self._retain_centroid_cpu.unsqueeze(0), gate_cfg.token_position
            )
            if self._retain_centroid_cpu is not None
            else None
        )

        self._cached_sim: torch.Tensor | None = None       # [B, K] — populated by cache_routing()
        self._cached_retain_sim: torch.Tensor | None = None  # [B]
        self._per_forward_sim: torch.Tensor | None = None  # [B, K] — from scoring hook
        self._attn_mask: torch.Tensor | None = None        # [B, S]
        self._mean_all_sum: torch.Tensor | None = None     # [B, H] — for mean_all routing
        self._mean_all_count: int = 0
        self._bypass_sv: bool = False  # set True during the pre-forward mean_all pass

    @staticmethod
    def _default_scorer(
        gate_cfg: Any,
        text_centroids: torch.Tensor | None,
        cluster_texts: list[list[str]] | None,
    ) -> TextScorerProtocol:
        mode = gate_cfg.routing_mode
        if mode == "bm25":
            if cluster_texts is None:
                raise ValueError("routing_mode='bm25' requires cluster_texts.")
            return TextBM25Scorer(cluster_texts)
        if text_centroids is None:
            raise ValueError("routing_source='text' requires text_centroids.")
        return TextCosineScorer(text_centroids, gate_cfg.st_model_name)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def cache_routing(self, input_ids: Any) -> None:
        """Pre-compute and cache routing similarities from input text.

        Called once at the start of ``generate()`` or ``__call__()`` so the
        hook reuses the result across all autoregressive steps.
        No-op when ``routing_source != 'text'``.
        """
        if self._cfg.routing_source != "text":
            return
        assert self._tokenizer is not None
        assert self._scorer is not None

        texts: list[str] = self._tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        self._cached_sim = self._scorer.score(texts)

        self._cached_retain_sim = None
        need_retain_sim = (
            (self._cfg.retain_threshold > 0 or self._cfg.routing_mode == "ratio")
            and self._retain_text_centroid_cpu is not None
            and isinstance(self._scorer, TextCosineScorer)
        )
        if need_retain_sim:
            embs = self._scorer.encode(texts)
            rc = self._retain_text_centroid_cpu
            self._cached_retain_sim = F.cosine_similarity(
                embs, rc.unsqueeze(0).expand_as(embs), dim=-1
            )

    def clear_cache(self) -> None:
        """Clear cached routing similarities (call after ``generate()`` completes)."""
        self._cached_sim = None
        self._cached_retain_sim = None

    def cache_mean_all_routing(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Pre-compute mean_all routing via a separate no-steering forward pass.

        Runs the model once with the SV hook bypassed so the per-layer accumulator
        hooks fill ``_mean_all_sum``.  Computes the global mean, scores against
        centroids, and caches the sim in ``_cached_sim`` for subsequent forward passes.
        """
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            self._attn_mask = attention_mask
        self._mean_all_sum = None
        self._mean_all_count = 0
        self._bypass_sv = True
        try:
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            self._bypass_sv = False
        if self._mean_all_sum is None or self._mean_all_count == 0:
            logger.warning("mean_all pre-forward produced no activations.")
            return
        global_mean = self._mean_all_sum / self._mean_all_count  # [B, H]
        try:
            fake = global_mean.unsqueeze(1)
            ones = torch.ones(fake.shape[0], 1, dtype=torch.long, device=device)
            sim = self._activation_scorer.score(fake, attention_mask=ones)
        except TypeError:
            sim = self._activation_scorer.score(global_mean.unsqueeze(1))
        self._cached_sim = sim.detach()
        self._mean_all_sum = None
        self._mean_all_count = 0

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def register_hook(
        self,
        model: Any,
        coeff_holder: list[float],
        layer_idx: int,
        module_name: str = "residual",
    ) -> Any:
        """Register the gated steering hook on the model.

        Args:
            model: HuggingFace causal LM.
            coeff_holder: Mutable ``[coeff]`` list (shared with :class:`SteeredModel`).
            layer_idx: Transformer layer index.
            module_name: Module to hook.

        Returns:
            A ``RemovableHandle``.
        """
        from guard.online.registry import get_target_module

        target = get_target_module(model, layer_idx, module_name)
        handle = target.register_forward_hook(self._make_hook(coeff_holder))
        return handle

    def register_mask_capture(self, model: Any) -> Any:
        """Register a pre-forward hook that captures ``attention_mask``.

        Needed for masked mean-pooling inside scorers — the attention_mask is
        only visible to the top-level model call, not to mid-model layer hooks.
        """
        gate = self

        def _pre_hook(_module: Any, args: tuple, kwargs: dict) -> None:
            mask = kwargs.get("attention_mask")
            if mask is None and len(args) > 1:
                maybe = args[1]
                if isinstance(maybe, torch.Tensor) and maybe.dtype in (
                    torch.int64,
                    torch.int32,
                    torch.bool,
                ):
                    mask = maybe
            gate._attn_mask = mask

        return model.register_forward_pre_hook(_pre_hook, with_kwargs=True)

    def register_scoring_hook(
        self,
        model: Any,
        cluster_layer: int | str,
        module_name: str = "residual",
    ) -> Any | list[Any]:
        """Register forward hook(s) that compute routing scores at cluster_layer.

        Stores the score in ``self._per_forward_sim`` each forward pass; the
        SV hook then reads this instead of recomputing from its own layer.

        Supports three forms:
          * int — single layer
          * ``'embed'`` — input embedding module (pre-transformer)
          * ``'mean_all'`` — hook every transformer layer, accumulate the
            token-mean-pooled activation, and score the global mean at the
            SV layer.

        Returns the handle(s) so the caller can remove them on cleanup.
        """
        gate = self

        def _scoring_hook(_module: Any, _inputs: Any, output: Any) -> None:
            h = output[0] if isinstance(output, tuple) else output
            try:
                sim = gate._activation_scorer.score(h, attention_mask=gate._attn_mask)
            except TypeError:
                sim = gate._activation_scorer.score(h)
            gate._per_forward_sim = sim

        if isinstance(cluster_layer, str) and cluster_layer == "embed":
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                target = model.model.embed_tokens
            elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                target = model.transformer.wte
            else:
                raise ValueError("Cannot locate embedding layer for scoring hook.")
            return target.register_forward_hook(_scoring_hook)

        if isinstance(cluster_layer, str) and cluster_layer == "mean_all":
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layers = model.model.layers
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layers = model.transformer.h
            else:
                raise ValueError("Cannot locate transformer layers for mean_all scoring.")

            def _accum_hook(_module: Any, _inputs: Any, output: Any) -> None:
                h = output[0] if isinstance(output, tuple) else output
                hf = h.float()
                mask = gate._attn_mask
                if mask is not None:
                    m = mask.to(hf.device).unsqueeze(-1).float()
                    pooled = (hf * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                else:
                    pooled = hf.mean(dim=1)
                if gate._mean_all_sum is None:
                    gate._mean_all_sum = pooled
                else:
                    gate._mean_all_sum = gate._mean_all_sum.to(pooled.device) + pooled
                gate._mean_all_count += 1

            handles = [layer.register_forward_hook(_accum_hook) for layer in layers]
            return handles

        from guard.online.registry import get_target_module

        target = get_target_module(model, int(cluster_layer), module_name)
        return target.register_forward_hook(_scoring_hook)

    def _make_hook(self, coeff_holder: list[float]) -> Any:
        """Return a forward hook closure that applies gated steering."""
        gate = self

        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if gate._bypass_sv:
                return output
            coeff = coeff_holder[0]
            is_tuple = isinstance(output, tuple)
            h: torch.Tensor = cast(torch.Tensor, output[0] if is_tuple else output)

            device = h.device
            psv_clusters = gate._psv_clusters_cpu.to(device)  # [K, H]
            cfg = gate._cfg

            if gate._cached_sim is not None:
                sim = gate._cached_sim.to(device)  # [B, K]
            elif gate._mean_all_sum is not None:
                hf = h.float()
                mask = gate._attn_mask
                if mask is not None:
                    m = mask.to(hf.device).unsqueeze(-1).float()
                    pooled = (hf * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                else:
                    pooled = hf.mean(dim=1)
                global_mean = (gate._mean_all_sum.to(device) + pooled) / (
                    gate._mean_all_count + 1
                )
                try:
                    fake = global_mean.unsqueeze(1)
                    ones = torch.ones(fake.shape[0], 1, dtype=torch.long, device=device)
                    sim = gate._activation_scorer.score(fake, attention_mask=ones)
                except TypeError:
                    sim = gate._activation_scorer.score(global_mean.unsqueeze(1))
                gate._mean_all_sum = None
                gate._mean_all_count = 0
            elif gate._per_forward_sim is not None:
                sim = gate._per_forward_sim.to(device)
                gate._per_forward_sim = None
            else:
                try:
                    sim = gate._activation_scorer.score(h, attention_mask=gate._attn_mask)
                except TypeError:
                    sim = gate._activation_scorer.score(h)

            retain_sim: torch.Tensor | None = (
                gate._cached_retain_sim.to(device)
                if gate._cached_retain_sim is not None
                else None
            )
            if (
                cfg.retain_threshold > 0
                and retain_sim is None
                and gate._retain_activation_scorer is not None
            ):
                retain_sim = gate._retain_activation_scorer.score(h).squeeze(-1)

            route_mode = "threshold" if cfg.routing_mode == "bm25" else cfg.routing_mode
            batch_size = int(h.shape[0])
            h = h.clone()

            for b in range(batch_size):
                if cfg.retain_threshold > 0 and retain_sim is not None:
                    rs = float(retain_sim[b].item())
                    if rs > cfg.retain_threshold:
                        if cfg.log_routing:
                            logger.debug(
                                "[gate] batch={} retain_sim={:.3f} > {:.3f} → skipped",
                                b, rs, cfg.retain_threshold,
                            )
                        continue

                rs_val = float(retain_sim[b].item()) if retain_sim is not None else None
                sv = _select_sv(
                    sim[b],
                    psv_clusters,
                    route_mode,
                    cfg.threshold,
                    b,
                    cfg.log_routing,
                    rs_val,
                    dissimilar_top_k=getattr(cfg, "dissimilar_top_k", 3),
                    dissimilar_fallback_k=getattr(cfg, "dissimilar_fallback_k", 3),
                )
                if sv is None:
                    continue

                sv = sv.to(dtype=h.dtype)
                h_b = h[b] - coeff * sv  # [S, H]

                if gate._rotation_only:
                    orig_norm = h[b].norm(dim=-1, keepdim=True)
                    steered_norm = h_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    h[b] = h_b * (orig_norm / steered_norm)
                else:
                    h[b] = h_b

            return (h, *output[1:]) if is_tuple else h

        return _hook


# ----------------------------------------------------------------------
# Helper — module-level so the hook closure stays small
# ----------------------------------------------------------------------


def _select_sv(
    sim_row: torch.Tensor,
    psv_clusters: torch.Tensor,
    route_mode: str,
    threshold: float,
    batch_idx: int,
    log_routing: bool,
    retain_sim: float | None = None,
    dissimilar_top_k: int = 3,
    dissimilar_fallback_k: int = 3,
) -> torch.Tensor | None:
    """Pick (or synthesize) the steering vector for a single batch row."""
    best_k = int(sim_row.argmax().item())

    if route_mode in ("threshold", "weighted_threshold"):
        mask = sim_row > threshold  # [K]
        if not bool(mask.any().item()):
            if log_routing:
                logger.debug(
                    "[gate] batch={} all_sims={} → none above {:.2f} → skipped",
                    batch_idx, sim_row.tolist(), threshold,
                )
            return None
        if log_routing:
            active_ks = mask.nonzero(as_tuple=True)[0].tolist()
            active_sims = sim_row[mask].tolist()
            logger.debug(
                "[gate] batch={} ACTIVATED clusters={} sims={}",
                batch_idx, active_ks, [f"{s:.3f}" for s in active_sims],
            )
        if route_mode == "weighted_threshold":
            weights = sim_row[mask]
            weights = weights / weights.sum()
            return (weights.unsqueeze(-1) * psv_clusters[mask]).sum(dim=0)
        return psv_clusters[mask].mean(dim=0)

    if route_mode == "dissimilar_threshold":
        k = max(1, min(int(dissimilar_top_k), int(sim_row.numel())))
        k_fb = max(1, min(int(dissimilar_fallback_k), int(sim_row.numel())))
        candidate_idx = (sim_row <= threshold).nonzero(as_tuple=True)[0]
        if int(candidate_idx.numel()) > 0:
            cand_sims = sim_row[candidate_idx]
            order = torch.argsort(cand_sims, descending=False)
            chosen = candidate_idx[order[: min(k, int(candidate_idx.numel()))]]
            if log_routing:
                logger.debug(
                    "[gate] batch={} DISSIM <= {:.3f} picked={} sims={}",
                    batch_idx, threshold, chosen.tolist(),
                    [f"{float(sim_row[i].item()):.3f}" for i in chosen.tolist()],
                )
            return psv_clusters[chosen].mean(dim=0)

        global_order = torch.argsort(sim_row, descending=False)
        chosen = global_order[:k_fb]
        if log_routing:
            logger.debug(
                "[gate] batch={} DISSIM no sim<={:.3f} fallback_k={} picked={} sims={}",
                batch_idx, threshold, k_fb, chosen.tolist(),
                [f"{float(sim_row[i].item()):.3f}" for i in chosen.tolist()],
            )
        return psv_clusters[chosen].mean(dim=0)

    if route_mode == "soft":
        weights = sim_row.clamp(min=0.0)
        total = weights.sum()
        if total < 1e-9:
            return None
        weights = weights / total
        sv = (weights.unsqueeze(-1) * psv_clusters).sum(dim=0)
        if log_routing:
            logger.debug(
                "[gate] batch={} SOFT weights={}",
                batch_idx, [f"{w:.3f}" for w in weights.tolist()],
            )
        return sv

    if route_mode == "ratio":
        best_sim = float(sim_row.max().item())
        denom = retain_sim if retain_sim is not None else 1.0
        ratio = best_sim / max(denom, 1e-6)
        if ratio <= threshold:
            if log_routing:
                logger.debug(
                    "[gate] batch={} RATIO {:.3f}/{:.3f}={:.3f} <= {:.2f} → skipped",
                    batch_idx, best_sim, denom, ratio, threshold,
                )
            return None
        mask = sim_row == sim_row.max()
        if log_routing:
            logger.debug(
                "[gate] batch={} RATIO {:.3f}/{:.3f}={:.3f} > {:.2f} → cluster {}",
                batch_idx, best_sim, denom, ratio, threshold,
                int(sim_row.argmax().item()),
            )
        return psv_clusters[mask][0]

    # "best" — always steer with the single closest cluster
    if log_routing:
        logger.debug(
            "[gate] batch={} best=c{}({:.3f})",
            batch_idx, best_k, float(sim_row[best_k].item()),
        )
    return psv_clusters[best_k]
