"""CLI entry point: `guard cluster <config.yaml>`.

Clusters forget documents via KMeans on MiniLM embeddings, then computes one
PSV (Partial Steering Vector) per forget-cluster paired against the full retain corpus.

Algorithm
---------
1. Embed forget docs with all-MiniLM-L6-v2 → L2-normalised [N_f, D]
2. Auto-select K_f via silhouette → KMeans forget → K_f clusters
3. For each forget cluster k:
   a. Capture activations: forget_cluster_k docs vs. full retain corpus
   b. PSV_k = compute_steering_vector(forget_cluster_k, retain_all)
4. Save psv_clusters.pt, centroids.pt, text_centroids.pt, text_embeddings.pt,
   labels.pt, forget_docs.json, cluster_meta.json per layer.

Output layout::

    steering_vectors/<model>/orthogonal_clustered/<behavior>_Kf<kf>_s<seed>/<module>/layer_<N>/
        psv_clusters.pt           — [Kf, H] stacked Partial Steering Vectors
        centroids.pt              — [Kf, H] per-cluster LLM activation centroids
        text_centroids.pt         — [Kf, D] forget MiniLM centroids (for routing)
    steering_vectors/<model>/orthogonal_clustered/<behavior>_Kf<kf>_s<seed>/
        text_embeddings.pt        — [N_f, D] all forget embeddings (for plotting)
        labels.pt                 — [N_f] forget cluster assignments
        forget_docs.json          — forget doc texts
        cluster_meta.json         — full reproducibility metadata

Usage
-----
    guard cluster configs/generate.yaml
    guard cluster configs/generate_forget05.yaml --n-clusters auto
    guard cluster configs/generate.yaml --n-clusters 4
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from guard.offline.activations import (
    extract_all_activations_multilayer,
    extract_embedding_layer,
    extract_mean_activations_multilayer,
)
from guard.offline.steering import compute_steering_vector
from guard.config.generation import LocalJSONLDatasetConfig, TofuDatasetConfig
from guard.config.loader import load_generation_config, override_config
from loguru import logger

from guard.online.embed import TextEmbedder
from guard.model.loading import build_model_load_kwargs
from guard.storage.io import documents_hash

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------


def _embed(docs: list[str], embedder: TextEmbedder) -> torch.Tensor:
    """Embed docs → [N, D] L2-normalised float32 on CPU."""
    return embedder.encode(docs)


def _kmedoids(embs: np.ndarray, k: int, seed: int = 42, max_iter: int = 100) -> np.ndarray:
    """K-medoids clustering (PAM-style) using Euclidean distance.

    Medoids are actual data points; suitable for unnormalised embeddings.
    """
    rng = np.random.RandomState(seed)
    n = len(embs)
    medoid_idx = rng.choice(n, k, replace=False)

    for _ in range(max_iter):
        diff = embs[:, None, :] - embs[medoid_idx][None, :, :]  # [n, k, D]
        dists = np.linalg.norm(diff, axis=2)  # [n, k]
        labels = dists.argmin(axis=1)  # [n]

        new_medoid_idx = medoid_idx.copy()
        for ki in range(k):
            mask = labels == ki
            if not mask.any():
                continue
            cluster_pts = embs[mask]
            inner_diff = cluster_pts[:, None, :] - cluster_pts[None, :, :]
            inner_dists = np.linalg.norm(inner_diff, axis=2)
            new_medoid_idx[ki] = np.where(mask)[0][inner_dists.sum(axis=1).argmin()]

        if np.array_equal(new_medoid_idx, medoid_idx):
            break
        medoid_idx = new_medoid_idx

    return labels


def _find_best_k(
    embs: np.ndarray,
    k_range: "range | list[int]",
    seed: int = 42,
    use_kmedoids: bool = False,
) -> int:
    """Pick K with highest silhouette score."""
    from sklearn.metrics import silhouette_score

    ks = list(k_range)
    best_k, best_score = ks[0], -1.0
    for k in ks:
        if k >= len(embs):
            break
        labels = _kmedoids(embs, k, seed=seed) if use_kmedoids else _kmeans(embs, k, seed=seed)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(embs, labels))
        algo = "kmedoids" if use_kmedoids else "kmeans"
        logger.info("  silhouette[{}] k={} → {:.4f}", algo, k, score)
        if score > best_score:
            best_score, best_k = score, k
    logger.info("Best K={} (silhouette={:.4f})", best_k, best_score)
    return best_k


def _kmeans(embs: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return km.fit_predict(embs)


def _calibrate_threshold(
    forget_scores: np.ndarray,
    retain_scores: np.ndarray,
    alpha: float = 1.0,
    n_steps: int = 200,
) -> dict:
    """Find routing threshold that maximises recall_forget - alpha * fpr_retain.

    Args:
        forget_scores: [N_f] max-cluster score for each forget doc.
        retain_scores: [N_r] max-cluster score for each retain doc.
        alpha: Weight on retain false-positive rate. Higher = more conservative.
        n_steps: Number of threshold candidates to evaluate.

    Returns:
        Dict with ``threshold``, ``recall_forget``, ``fpr_retain``, ``objective``.
    """
    all_scores = np.concatenate([forget_scores, retain_scores])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    thresholds = np.linspace(lo, hi, n_steps)

    best = {"threshold": lo, "recall_forget": 1.0, "fpr_retain": 1.0, "objective": -999.0}
    for t in thresholds:
        recall = float((forget_scores >= t).mean())
        fpr = float((retain_scores >= t).mean())
        obj = recall - alpha * fpr
        if obj > best["objective"]:
            best = {
                "threshold": float(t),
                "recall_forget": recall,
                "fpr_retain": fpr,
                "objective": obj,
            }

    logger.info(
        "Calibrated threshold=%.4f  recall_forget=%.3f  fpr_retain=%.3f  obj=%.3f  (alpha=%.1f)",
        best["threshold"],
        best["recall_forget"],
        best["fpr_retain"],
        best["objective"],
        alpha,
    )
    return best


def _calibrate_threshold_retain_percentile(
    forget_scores: np.ndarray,
    retain_scores: np.ndarray,
    target_fpr: float = 0.05,
) -> dict:
    """Pick threshold = (1 - target_fpr)-quantile of retain_scores.

    Guarantees at most ``target_fpr`` fraction of retain docs trigger steering,
    independent of the forget distribution.
    """
    q = float(np.quantile(retain_scores, 1.0 - target_fpr))
    recall = float((forget_scores >= q).mean())
    fpr = float((retain_scores >= q).mean())
    logger.info(
        "Calibrated threshold=%.4f (retain %.1fth pct)  recall_forget=%.3f  fpr_retain=%.3f  "
        "(method=retain_percentile, target_fpr=%.3f)",
        q,
        100.0 * (1.0 - target_fpr),
        recall,
        fpr,
        target_fpr,
    )
    return {
        "threshold": q,
        "recall_forget": recall,
        "fpr_retain": fpr,
        "target_fpr": target_fpr,
        "method": "retain_percentile",
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _apply_template(docs_qa: list[tuple[str, str]], tokenizer: Any) -> list[str]:
    """Wrap (question, answer) pairs with the model's chat template."""
    out: list[str] = []
    for question, answer in docs_qa:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        out.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )
    return out


def _load_documents(
    cfg: Any,
    tokenizer: Any | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load documents for clustering and SV extraction.

    Returns:
        embed_forget: docs used for MiniLM embedding / clustering
        embed_retain: docs used for MiniLM embedding (retain, for calibration)
        sv_forget:    docs used for LLM activation extraction (SV computation)
        sv_retain:    docs used for LLM activation extraction (SV computation, retain)
    """
    ds = cfg.dataset
    limit = cfg.limit if cfg.limit is not None else getattr(ds, "limit", None)
    use_template = getattr(cfg, "apply_chat_template", False) and tokenizer is not None
    cluster_with_template = getattr(cfg, "cluster_with_template", True)

    if isinstance(ds, TofuDatasetConfig):
        from datasets import load_dataset as hf_load  # type: ignore[import-untyped]

        forget_split = ds.forget_split
        retain_split = ds.inferred_retain_split()
        logger.info("Loading TOFU: forget={}  retain={}", forget_split, retain_split)
        forget_ds = hf_load("locuslab/TOFU", name=forget_split, split="train")
        retain_ds = hf_load("locuslab/TOFU", name=retain_split, split="train")

        plain_forget: list[str] = [f"{x['question']}\n{x['answer']}" for x in forget_ds]
        plain_retain: list[str] = [f"{x['question']}\n{x['answer']}" for x in retain_ds]

        if use_template:
            logger.info("Applying chat template to SV documents (apply_chat_template=True)")
            tmpl_forget: list[str] = _apply_template(
                [(x["question"], x["answer"]) for x in forget_ds], tokenizer
            )
            tmpl_retain: list[str] = _apply_template(
                [(x["question"], x["answer"]) for x in retain_ds], tokenizer
            )
            sv_forget: list[str] = tmpl_forget
            sv_retain: list[str] = tmpl_retain
            if cluster_with_template:
                logger.info("Clustering will also use chat-template-formatted text")
                embed_forget: list[str] = tmpl_forget
                embed_retain: list[str] = tmpl_retain
            else:
                logger.info("Clustering uses plain text; SVs use chat template")
                embed_forget = plain_forget
                embed_retain = plain_retain
        else:
            embed_forget = plain_forget
            embed_retain = plain_retain
            sv_forget = plain_forget
            sv_retain = plain_retain

    elif isinstance(ds, LocalJSONLDatasetConfig):

        def _read(path: str, key: str, lim: int | None) -> list[str]:
            out: list[str] = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        text = str(obj.get(key, ""))
                        if text:
                            out.append(text)
            return out[:lim] if lim else out

        plain_forget = _read(ds.forget_jsonl, ds.text_key, limit)
        plain_retain = _read(ds.retain_jsonl, ds.text_key, limit)
        embed_forget = plain_forget
        embed_retain = plain_retain
        sv_forget = plain_forget
        sv_retain = plain_retain
        logger.info("LocalJSONL: {} forget, {} retain", len(embed_forget), len(embed_retain))
        return embed_forget, embed_retain, sv_forget, sv_retain

    else:
        raise ValueError(f"Unsupported dataset type: {type(ds)}")

    if limit is not None:
        embed_forget = embed_forget[:limit]
        embed_retain = embed_retain[:limit]
        sv_forget = sv_forget[:limit]
        sv_retain = sv_retain[:limit]

    logger.info("Dataset: {} forget, {} retain", len(embed_forget), len(embed_retain))
    return embed_forget, embed_retain, sv_forget, sv_retain


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def run_cluster(
    cfg: Any,
    n_clusters: int | str = "auto",
    k_min: int = 2,
    k_max: int = 20,
    k_candidates: list[int] | None = None,
    config_path: Path | None = None,
) -> None:
    """Generate clustered steering vectors.

    Forget docs are clustered into K_f groups; each cluster's SV is computed
    against the full retain corpus.

    Args:
        cfg: Validated :class:`~guard.config.GenerationConfig`.
        n_clusters: Number of forget clusters, or ``'auto'`` for silhouette.
        k_min: Minimum K_f when ``n_clusters='auto'``.
        k_max: Maximum K_f when ``n_clusters='auto'``.
        k_candidates: Explicit K candidates for silhouette (overrides k_min/k_max).
        config_path: Path to the originating YAML (copied verbatim).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    behavior = cfg.effective_behavior()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
        logger.info("CUDA_VISIBLE_DEVICES={}", cfg.cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model '{}' on {} ...", cfg.model_name, device)
    load_kwargs = build_model_load_kwargs(
        device=device,
        quantization=getattr(cfg, "quantization", "none"),
        quant_4bit_type=getattr(cfg, "quant_4bit_type", "nf4"),
        quant_4bit_double_quant=getattr(cfg, "quant_4bit_double_quant", True),
        quant_compute_dtype=getattr(cfg, "quant_compute_dtype", "bfloat16"),
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.vocab_size < 100:
        # Some HF repos ship a stub tokenizer with only special tokens (vocab=3:
        # <s>, <unk>, </s>), which silently encodes every input as "<s><unk>" —
        # producing identical activations for all docs and meaningless SVs.
        # Fall back to the base Llama-2 tokenizer when we detect this.
        _fallback = getattr(cfg, "tokenizer", None) or "meta-llama/Llama-2-7b-hf"
        logger.warning(
            "Tokenizer vocab=%d for '%s' is broken — falling back to '%s'",
            tokenizer.vocab_size,
            cfg.model_name,
            _fallback,
        )
        tokenizer = AutoTokenizer.from_pretrained(_fallback)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # embed_* = used for MiniLM clustering; sv_* = used for LLM activation extraction
    embed_forget, embed_retain, sv_forget, sv_retain = _load_documents(cfg, tokenizer=tokenizer)

    cluster_source = getattr(cfg, "cluster_source", "text")

    kw_act = dict(
        model=model,
        tokenizer=tokenizer,
        layer_indices=cfg.layers,
        module_name=cfg.module_name,
        token_position=cfg.token_position,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        add_special_tokens=cfg.add_special_tokens,
        num_workers=cfg.num_workers,
    )

    _cl_cfg_obj = getattr(cfg, "clustering", None) or cfg
    _use_kmedoids = bool(getattr(_cl_cfg_obj, "use_kmedoids", False))
    _norm_for_cluster = bool(getattr(_cl_cfg_obj, "normalize_embeddings_for_clustering", True))
    if _use_kmedoids:
        logger.info(
            "Clustering algorithm: k-medoids (normalize_for_clustering=%s)", _norm_for_cluster
        )

    # MiniLM embeddings — needed for text-space clustering and routing centroids
    use_minilm = cluster_source == "text"
    forget_embs: torch.Tensor
    retain_embs: torch.Tensor
    forget_embs_np: np.ndarray
    retain_embs_np: np.ndarray

    if use_minilm:
        embedder = TextEmbedder()
        logger.info("Embedding {} forget docs with MiniLM ...", len(embed_forget))
        forget_embs = _embed(embed_forget, embedder)
        forget_embs_np = forget_embs.numpy()
        logger.info("Embedding {} retain docs with MiniLM ...", len(embed_retain))
        retain_embs = _embed(embed_retain, embedder)
        retain_embs_np = retain_embs.numpy()

        if not _norm_for_cluster:
            logger.info("Re-embedding without L2 normalisation for clustering ...")
            _forget_cluster_embs: np.ndarray = embedder.encode(
                embed_forget, normalize=False
            ).numpy()
        else:
            _forget_cluster_embs = forget_embs_np

    if cluster_source in ("activation", "llama_mean"):
        _cl = cfg.cluster_layer if cfg.cluster_layer is not None else cfg.layers[0]
        if isinstance(_cl, int) and _cl != cfg.layers[0]:
            raise ValueError(
                f"cluster_layer={_cl} must equal layers[0]={cfg.layers[0]} when "
                f"cluster_source='{cluster_source}' — routing and SV extraction "
                "have to share the same activation space."
            )
        mean_all_layers = isinstance(_cl, str) and _cl == "mean_all"
        use_embed_layer = isinstance(_cl, str) and _cl == "embed"
        tok_pos = cfg.token_position if cluster_source == "activation" else "mean"

        if use_embed_layer:
            cluster_layer = -2
            kw_embed = dict(
                model=model,
                tokenizer=tokenizer,
                token_position=tok_pos,
                batch_size=cfg.batch_size,
                max_length=cfg.max_length,
                add_special_tokens=cfg.add_special_tokens,
                num_workers=cfg.num_workers,
            )
            logger.info(
                "cluster_source=%s — extracting embedding layer for %d forget docs ...",
                cluster_source,
                len(sv_forget),
            )
            forget_embs_t = extract_embedding_layer(documents=sv_forget, **kw_embed)
            cluster_embs_np = forget_embs_t.numpy()
            logger.info("Extracting embeddings for {} retain docs ...", len(sv_retain))
            retain_embs_t = extract_embedding_layer(documents=sv_retain, **kw_embed)
            cluster_retain_embs_np = retain_embs_t.numpy()
        else:
            if mean_all_layers:
                n_layers = int(model.config.num_hidden_layers)
                layer_indices = list(range(n_layers))
                cluster_layer = -1
            else:
                cluster_layer = int(_cl)
                layer_indices = [cluster_layer]
            kw_llama = {**kw_act, "token_position": tok_pos, "layer_indices": layer_indices}
            logger.info(
                "cluster_source={} — extracting layers={} activations for {} forget docs ...",
                cluster_source,
                "mean_all" if mean_all_layers else str(layer_indices),
                len(sv_forget),
            )
            all_forget_acts = extract_all_activations_multilayer(documents=sv_forget, **kw_llama)
            if mean_all_layers:
                cluster_embs_np = np.stack(
                    [all_forget_acts[li].numpy() for li in layer_indices]
                ).mean(axis=0)
            else:
                cluster_embs_np = all_forget_acts[cluster_layer].numpy()
            logger.info("Extracting activations for {} retain docs ...", len(sv_retain))
            all_retain_acts = extract_all_activations_multilayer(documents=sv_retain, **kw_llama)
            if mean_all_layers:
                cluster_retain_embs_np = np.stack(
                    [all_retain_acts[li].numpy() for li in layer_indices]
                ).mean(axis=0)
            else:
                cluster_retain_embs_np = all_retain_acts[cluster_layer].numpy()

        if cluster_source == "llama_mean":
            norms = np.linalg.norm(cluster_embs_np, axis=1, keepdims=True)
            cluster_embs_np = cluster_embs_np / np.clip(norms, 1e-12, None)
            norms = np.linalg.norm(cluster_retain_embs_np, axis=1, keepdims=True)
            cluster_retain_embs_np = cluster_retain_embs_np / np.clip(norms, 1e-12, None)
    else:
        cluster_embs_np = _forget_cluster_embs if not _norm_for_cluster else forget_embs_np

    _cluster_fn = _kmedoids if _use_kmedoids else _kmeans
    _algo_label = "kmedoids" if _use_kmedoids else "kmeans"

    if n_clusters == "auto":
        if k_candidates is not None:
            logger.info(
                "Auto-selecting K_f via silhouette[%s] over candidates=%s ...",
                _algo_label,
                k_candidates,
            )
            k = _find_best_k(
                cluster_embs_np, k_candidates, seed=cfg.seed, use_kmedoids=_use_kmedoids
            )
        else:
            logger.info(
                "Auto-selecting K_f via silhouette[%s] (range %d–%d) ...",
                _algo_label,
                k_min,
                k_max,
            )
            k = _find_best_k(
                cluster_embs_np, range(k_min, k_max + 1), seed=cfg.seed, use_kmedoids=_use_kmedoids
            )
    else:
        k = int(n_clusters)
    k = min(k, len(embed_forget))

    logger.info(
        "%s forget: K_f=%d (source=%s, norm_for_cluster=%s)",
        _algo_label,
        k,
        cluster_source,
        _norm_for_cluster,
    )
    forget_labels_np = _cluster_fn(cluster_embs_np, k, seed=cfg.seed)

    # Guard against degenerate clusters
    unique_lbls = np.unique(forget_labels_np)
    if len(unique_lbls) < k:
        logger.warning(
            "%s produced %d non-empty clusters out of K=%d (degenerate). "
            "Compacting labels and reducing K to %d.",
            _algo_label,
            len(unique_lbls),
            k,
            len(unique_lbls),
        )
        remap = {int(old): new for new, old in enumerate(unique_lbls)}
        forget_labels_np = np.array(
            [remap[int(l)] for l in forget_labels_np], dtype=forget_labels_np.dtype
        )
        k = len(unique_lbls)

    forget_labels = torch.from_numpy(forget_labels_np)

    text_cents_list: list[torch.Tensor] = []
    if use_minilm:
        for ki in range(k):
            mask = forget_labels_np == ki
            cent = torch.tensor(forget_embs_np[mask].mean(axis=0), dtype=torch.float32)
            cent = cent / (cent.norm() + 1e-12)
            text_cents_list.append(cent)
    text_cents: torch.Tensor | None = torch.stack(text_cents_list) if text_cents_list else None

    cluster_forget_docs: list[list[str]] = [
        [doc for doc, lbl in zip(sv_forget, forget_labels_np.tolist()) if lbl == ki]
        for ki in range(k)
    ]
    for ki, cdocs in enumerate(cluster_forget_docs):
        logger.info("  forget cluster {}: {} docs", ki, len(cdocs))

    # All forget clusters use the full retain corpus
    logger.info("Using full retain corpus ({} docs) for all {} forget clusters", len(sv_retain), k)

    # LLM activation capture
    kw = dict(
        model=model,
        tokenizer=tokenizer,
        layer_indices=cfg.layers,
        module_name=cfg.module_name,
        token_position=cfg.token_position,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        add_special_tokens=cfg.add_special_tokens,
        num_workers=cfg.num_workers,
    )

    forget_results: list[dict[int, tuple[torch.Tensor, float]]] = []
    for ki, cdocs in enumerate(cluster_forget_docs):
        logger.info("Capturing forget cluster {} activations ({} docs) ...", ki, len(cdocs))
        forget_results.append(extract_mean_activations_multilayer(documents=cdocs, **kw))

    logger.info("Capturing retain activations ({} docs) ...", len(sv_retain))
    retain_result = extract_mean_activations_multilayer(documents=sv_retain, **kw)

    f_hash = documents_hash(sv_forget)
    r_hash = documents_hash(sv_retain)

    model_tag = cfg.model_name.replace("/", "_")
    behavior_dir = (
        Path(cfg.output_dir)
        / model_tag
        / f"{cfg.method}_clustered"
        / f"{behavior}_Kf{k}_s{cfg.seed}"
    )
    behavior_dir.mkdir(parents=True, exist_ok=True)

    # Calibrate routing threshold
    _cl_cfg = getattr(cfg, "clustering", None) or cfg
    calibration_alpha = float(getattr(_cl_cfg, "calibration_alpha", 1.0))
    calibration_method = str(getattr(_cl_cfg, "calibration_method", "youden_alpha"))
    calibration_retain_fpr = float(getattr(_cl_cfg, "calibration_retain_fpr", 0.05))

    def _run_calib(f_sc: np.ndarray, r_sc: np.ndarray) -> dict:
        if calibration_method == "retain_percentile":
            out = _calibrate_threshold_retain_percentile(
                f_sc, r_sc, target_fpr=calibration_retain_fpr
            )
        else:
            out = _calibrate_threshold(f_sc, r_sc, alpha=calibration_alpha)
            out["alpha"] = calibration_alpha
        out["method"] = calibration_method
        return out

    if cluster_source == "llama_mean":
        llama_cents_np = np.stack(
            [cluster_embs_np[forget_labels_np == ki].mean(axis=0) for ki in range(k)]
        )
        norms = np.linalg.norm(llama_cents_np, axis=1, keepdims=True)
        llama_cents_np = llama_cents_np / np.clip(norms, 1e-12, None)
        forget_max_scores = (cluster_embs_np @ llama_cents_np.T).max(axis=1)
        retain_max_scores = (cluster_retain_embs_np @ llama_cents_np.T).max(axis=1)
    elif cluster_source == "activation":
        act_cents_np = np.stack(
            [cluster_embs_np[forget_labels_np == ki].mean(axis=0) for ki in range(k)]
        )
        from sklearn.metrics import pairwise_distances

        f_dists = pairwise_distances(cluster_embs_np, act_cents_np, metric="euclidean")
        r_dists = pairwise_distances(cluster_retain_embs_np, act_cents_np, metric="euclidean")
        forget_max_scores = -f_dists.min(axis=1)
        retain_max_scores = -r_dists.min(axis=1)
        act_cents_normed = act_cents_np / np.clip(
            np.linalg.norm(act_cents_np, axis=1, keepdims=True), 1e-12, None
        )
        f_embs_normed = cluster_embs_np / np.clip(
            np.linalg.norm(cluster_embs_np, axis=1, keepdims=True), 1e-12, None
        )
        r_embs_normed = cluster_retain_embs_np / np.clip(
            np.linalg.norm(cluster_retain_embs_np, axis=1, keepdims=True), 1e-12, None
        )
        forget_cos_scores = (f_embs_normed @ act_cents_normed.T).max(axis=1)
        retain_cos_scores = (r_embs_normed @ act_cents_normed.T).max(axis=1)
        calib_cos = _run_calib(forget_cos_scores, retain_cos_scores)
        calib_cos["cluster_source"] = cluster_source
        calib_cos["metric"] = "cosine"
    else:
        assert text_cents is not None
        text_cents_np = text_cents.numpy()
        forget_max_scores = (forget_embs_np @ text_cents_np.T).max(axis=1)
        retain_max_scores = (retain_embs_np @ text_cents_np.T).max(axis=1)

    calib = _run_calib(forget_max_scores, retain_max_scores)
    calib["cluster_source"] = cluster_source
    if cluster_source == "activation":
        calib["metric"] = "euclidean"
    (behavior_dir / "routing_threshold.json").write_text(json.dumps(calib, indent=2))
    logger.info("Saved routing_threshold.json → threshold={:.4f}", calib["threshold"])
    if cluster_source == "activation":
        (behavior_dir / "routing_threshold_cosine.json").write_text(json.dumps(calib_cos, indent=2))
        logger.info("Saved routing_threshold_cosine.json → threshold={:.4f}", calib_cos["threshold"])

    if use_minilm:
        torch.save(forget_embs.float(), behavior_dir / "text_embeddings.pt")
        torch.save(text_cents.float(), behavior_dir / "text_centroids.pt")
        torch.save(retain_embs.float(), behavior_dir / "retain_embeddings.pt")
    cluster_forget_embs_t = torch.from_numpy(cluster_embs_np).float()
    forget_cluster_centroids = torch.stack(
        [
            torch.from_numpy(cluster_embs_np[forget_labels_np == ki].mean(axis=0)).float()
            for ki in range(k)
        ]
    )
    torch.save(cluster_forget_embs_t, behavior_dir / "cluster_embeddings.pt")
    torch.save(forget_cluster_centroids, behavior_dir / "cluster_centroids.pt")
    torch.save(forget_labels, behavior_dir / "labels.pt")
    (behavior_dir / "forget_docs.json").write_text(
        json.dumps(embed_forget, ensure_ascii=False, indent=2)
    )
    _cl_meta = (
        cfg.cluster_layer
        if cfg.cluster_layer is not None
        else (cfg.layers[0] if cluster_source in ("activation", "llama_mean") else None)
    )
    (behavior_dir / "cluster_meta.json").write_text(
        json.dumps(
            {
                "behavior": behavior,
                "n_clusters": k,
                "cluster_sizes": [len(d) for d in cluster_forget_docs],
                "k_auto": n_clusters == "auto",
                "model_name": cfg.model_name,
                "method": cfg.method,
                "seed": cfg.seed,
                "cluster_source": cluster_source,
                "cluster_layer": _cl_meta,
            },
            indent=2,
        )
    )

    _layer_methods: dict[int, str] = dict(getattr(cfg, "layer_methods", None) or {})

    for li in tqdm(cfg.layers, desc="Layers", unit="layer"):
        sv_list: list[torch.Tensor] = []
        act_cent_list: list[torch.Tensor] = []
        effective_method = _layer_methods.get(li, cfg.method)

        for ki in range(k):
            forget_mean, forget_norm = forget_results[ki][li]
            retain_mean, retain_norm = retain_result[li]

            sv, activation_norm = compute_steering_vector(
                forget_mean=forget_mean,
                forget_norm=forget_norm,
                retain_mean=retain_mean,
                retain_norm=retain_norm,
                method=effective_method,
                norm_cfg=cfg.normalizations,
            )
            sv_list.append(sv)
            act_cent_list.append(forget_mean.float().cpu())
            logger.info(
                "  Layer %d  cluster %d  activation_norm=%.4f",
                li,
                ki,
                activation_norm,
            )

        psv_clusters = torch.stack(sv_list)   # [K, H]
        act_centroids = torch.stack(act_cent_list)  # [K, H]

        out_dir = behavior_dir / cfg.module_name / f"layer_{li}"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(psv_clusters, out_dir / "psv_clusters.pt")
        torch.save(act_centroids, out_dir / "centroids.pt")
        if text_cents is not None:
            torch.save(text_cents.float(), out_dir / "text_centroids.pt")

        (out_dir / "routing_threshold.json").write_text(json.dumps(calib, indent=2))
        if cluster_source == "activation":
            (out_dir / "routing_threshold_cosine.json").write_text(
                json.dumps(calib_cos, indent=2)
            )
        if cluster_source == "llama_mean":
            llama_cents_list = [
                torch.tensor(
                    cluster_embs_np[forget_labels_np == ki].mean(axis=0), dtype=torch.float32
                )
                for ki in range(k)
            ]
            llama_cents = torch.stack(llama_cents_list)
            import torch.nn.functional as _F

            llama_cents = _F.normalize(llama_cents, dim=-1)
            torch.save(llama_cents, out_dir / "llama_centroids.pt")

        meta = {
            "guard_version": _guard_version(),
            "model_name": cfg.model_name,
            "behavior": behavior,
            "layer_idx": li,
            "module_name": cfg.module_name,
            "method": cfg.method,
            "effective_method": effective_method,
            "n_clusters": k,
            "k_auto": n_clusters == "auto",
            "cluster_forget_sizes": [len(d) for d in cluster_forget_docs],
            "n_retain_docs": len(sv_retain),
            "token_position": cfg.token_position,
            "apply_chat_template": getattr(cfg, "apply_chat_template", False),
            "cluster_with_template": getattr(cfg, "cluster_with_template", True),
            "routing_source": "text",
            "sv_scaling": cfg.normalizations.sv_scaling.value,
            "rotation_only": cfg.normalizations.rotation_only,
            "n_forget_docs": len(sv_forget),
            "forget_docs_hash": f_hash,
            "retain_docs_hash": r_hash,
            "seed": cfg.seed,
            "max_length": cfg.max_length,
            "batch_size": cfg.batch_size,
        }
        (out_dir / "cluster_meta.json").write_text(json.dumps(meta, indent=2))

        if config_path is not None:
            shutil.copy2(config_path, out_dir / "generate.yaml")

        logger.info("Saved: {}", out_dir)

    print(f"Done. Clustered SVs → {behavior_dir.resolve()}")


def _guard_version() -> str:
    try:
        from guard._version import __version__

        return __version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="guard cluster",
        description=(
            "Cluster forget docs with MiniLM, then compute "
            "per-cluster steering vectors against the full retain corpus."
        ),
    )
    parser.add_argument("config", help="Path to generate.yaml")
    parser.add_argument(
        "--n-clusters",
        default=None,
        metavar="K",
        help="Number of forget clusters, or 'auto' (silhouette). Overrides config.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=None,
        metavar="K_MIN",
        help="Min K_f for forget silhouette sweep. Overrides config.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        metavar="K_MAX",
        help="Max K_f for forget silhouette sweep. Overrides config.",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    from guard.cli._logging import setup_logging
    setup_logging(args.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    config_path = Path(args.config)
    cfg = load_generation_config(config_path)
    cfg = override_config(cfg, args.override)

    if cfg.cuda is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda

    cl = cfg.clustering

    def _parse_k(val: str | None, default: int | str) -> int | str:
        if val is None:
            return default
        return val if val == "auto" else int(val)

    n_clusters = _parse_k(args.n_clusters, cl.n_clusters)
    k_min = args.k_min if args.k_min is not None else cl.k_min
    k_max = args.k_max if args.k_max is not None else cl.k_max
    k_candidates = cl.k_candidates

    run_cluster(
        cfg,
        n_clusters=n_clusters,
        k_min=k_min,
        k_max=k_max,
        k_candidates=k_candidates,
        config_path=config_path,
    )
