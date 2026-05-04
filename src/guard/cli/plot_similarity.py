"""guard plot similarity — cosine-similarity distribution plot for threshold selection.

Embeds the forget and retain corpora plus an optional FineWeb-Edu reference sample,
computes max cosine similarity of each document to the forget cluster centroids, and
plots overlaid histograms.  The threshold line shows where the gateway will cut.

Usage
-----
    # From a guard generate/cluster YAML config:
    guard plot similarity configs/examples/tofu_1b_forget05.yaml --threshold 0.55

    # From any HuggingFace dataset:
    guard plot similarity \\
        --hf-dataset locuslab/TOFU --hf-forget-split forget05 --hf-retain-split retain95 \\
        --threshold 0.55

    # Local JSONL files (same config type as guard generate):
    guard plot similarity configs/my_dataset.yaml --threshold 0.4 --fineweb-n 0
"""


import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# ── embedding helpers ─────────────────────────────────────────────────────────

def _load_st(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "guard plot similarity requires 'sentence-transformers'.\n"
            "Install with: pip install sentence-transformers  or  pip install guard[gateway]"
        ) from exc
    return SentenceTransformer(model_name, device="cpu")


def _embed(texts: list[str], st_model: Any, batch_size: int = 256) -> "torch.Tensor":
    import torch
    embs = st_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embs.float().cpu()


def _kmeans_centroids(embs: "torch.Tensor", k: int, seed: int) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F
    from sklearn.cluster import KMeans  # type: ignore
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    km.fit(embs.numpy())
    cents = torch.from_numpy(km.cluster_centers_).float()
    return F.normalize(cents, dim=-1)


def _max_sim(embs: "torch.Tensor", centroids: "torch.Tensor") -> np.ndarray:
    sims = embs @ centroids.T  # [N, K]
    return sims.max(dim=1).values.numpy()


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_hf_dataset(dataset_id: str, split: str, text_key: str) -> list[str]:
    from datasets import load_dataset  # type: ignore
    logger.info("Loading %s / %s from HuggingFace …", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    if text_key not in ds.column_names:
        available = ", ".join(ds.column_names)
        raise ValueError(
            f"Text key '{text_key}' not found in dataset '{dataset_id}' split '{split}'. "
            f"Available columns: {available}"
        )
    texts = [str(ex[text_key]) for ex in ds]
    logger.info("  → %d documents", len(texts))
    return texts


def _load_tofu(split: str) -> list[str]:
    from datasets import load_dataset  # type: ignore
    logger.info("Loading TOFU %s …", split)
    ds = load_dataset("locuslab/TOFU", split, split="train")
    texts = [f"{ex['question']}\n{ex['answer']}" for ex in ds]
    logger.info("  → %d Q&A pairs", len(texts))
    return texts


def _load_jsonl(path: str, text_key: str) -> list[str]:
    texts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                texts.append(str(obj[text_key]))
    logger.info("Loaded %d documents from %s", len(texts), path)
    return texts


def _load_fineweb(n: int, seed: int) -> list[str]:
    from datasets import load_dataset  # type: ignore
    logger.info("Streaming %d examples from FineWeb-Edu …", n)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    rng = np.random.default_rng(seed)
    reservoir: list[str] = []
    for i, ex in enumerate(ds):
        text: str = ex["text"][:1024]
        if len(reservoir) < n:
            reservoir.append(text)
        else:
            j = int(rng.integers(0, i + 1))
            if j < n:
                reservoir[j] = text
        if i >= n * 20:
            break
    logger.info("  → %d texts collected", len(reservoir))
    return reservoir


# ── dataset resolution ────────────────────────────────────────────────────────

def _resolve_datasets(
    config_path: str | None,
    hf_dataset: str | None,
    hf_forget_split: str | None,
    hf_retain_split: str | None,
    hf_text_key: str,
    forget_split_override: str | None,
    retain_split_override: str | None,
) -> tuple[list[str], list[str]]:
    """Return (forget_texts, retain_texts) from the appropriate source."""

    if hf_dataset is not None:
        forget_split = hf_forget_split or forget_split_override or "train"
        retain_split = hf_retain_split or retain_split_override
        if retain_split is None:
            raise ValueError("--hf-retain-split is required when using --hf-dataset")
        forget_texts = _load_hf_dataset(hf_dataset, forget_split, hf_text_key)
        retain_texts = _load_hf_dataset(hf_dataset, retain_split, hf_text_key)
        return forget_texts, retain_texts

    if config_path is None:
        raise ValueError(
            "Provide either a config YAML (guard plot similarity config.yaml) "
            "or --hf-dataset with --hf-forget-split / --hf-retain-split."
        )

    # Load from guard generate/cluster YAML config
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parents[2]
    sys.path.insert(0, str(_ROOT / "src"))

    from guard.config.loader import load_generation_config  # type: ignore

    cfg = load_generation_config(config_path)
    ds = cfg.dataset

    if ds.type == "tofu":
        forget_split = forget_split_override or ds.forget_split
        retain_split = retain_split_override or ds.retain_split
        if retain_split is None:
            # Mirror the logic in cluster.py: forget05 → retain95
            n = forget_split.replace("forget", "")
            retain_split = f"retain{100 - int(n):02d}"
        forget_texts = _load_tofu(forget_split)
        retain_texts = _load_tofu(retain_split)
    elif ds.type == "local_jsonl":
        forget_texts = _load_jsonl(ds.forget_jsonl, ds.text_key)
        retain_texts = _load_jsonl(ds.retain_jsonl, ds.text_key)
    else:
        raise ValueError(f"Unknown dataset type: {ds.type!r}")

    return forget_texts, retain_texts


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_similarity_distributions(
    forget_texts: list[str],
    retain_texts: list[str],
    out_path: Path,
    n_clusters: int = 10,
    fineweb_n: int = 5000,
    threshold: float = 0.5,
    st_model_name: str = "all-MiniLM-L6-v2",
    seed: int = 42,
) -> None:
    import torch
    import torch.nn.functional as F

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("guard plot similarity requires matplotlib: pip install matplotlib") from exc

    st = _load_st(st_model_name)

    logger.info("Embedding forget corpus (%d docs) …", len(forget_texts))
    forget_embs = _embed(forget_texts, st)
    logger.info("Embedding retain corpus (%d docs) …", len(retain_texts))
    retain_embs = _embed(retain_texts, st)

    populations = []

    if fineweb_n > 0:
        fw_texts = _load_fineweb(fineweb_n, seed=seed)
        logger.info("Embedding FineWeb-Edu sample …")
        fw_embs = _embed(fw_texts, st)
        populations.append((fw_embs, f"FineWeb-Edu (n={len(fw_texts):,})", "#0173B2", "o"))

    k = min(n_clusters, len(forget_texts))
    logger.info("Computing %d forget centroids (k-means) …", k)
    if k == 1:
        centroids = F.normalize(forget_embs.mean(dim=0, keepdim=True), dim=-1)
    else:
        centroids = _kmeans_centroids(forget_embs, k=k, seed=seed)

    forget_sims = _max_sim(forget_embs, centroids)
    retain_sims = _max_sim(retain_embs, centroids)

    populations.append((retain_embs, f"Retain (n={len(retain_texts):,})", "#029E73", "s"))
    populations.append((forget_embs, f"Forget (n={len(forget_texts):,})", "#D55E00", "^"))

    all_sims = [_max_sim(embs, centroids) for embs, *_ in populations]

    lo = min(s.min() for s in all_sims) - 0.02
    hi = max(s.max() for s in all_sims) + 0.02
    bins = np.linspace(max(lo, 0.0), min(hi, 1.0), 40)
    alpha = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (embs, label, color, marker), sims in zip(populations, all_sims):
        counts, edges = np.histogram(sims, bins=bins)
        density = counts / counts.sum()
        centers = (edges[:-1] + edges[1:]) / 2
        ax.fill_between(centers, density, alpha=alpha, color=color, label=None)
        ax.plot(centers, density, color=color, linewidth=1.5,
                marker=marker, markersize=4, markevery=3, label=label)

    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.3,
               label=f"threshold = {threshold}")

    ax.set_xlabel("Max cosine similarity to any forget cluster", fontsize=10)
    ax.set_ylabel("Relative frequency", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, color="0.8")
    ax.set_xlim(max(lo, 0.0), min(hi, 1.0))

    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(populations) + 1,
        fontsize=9,
        framealpha=0.85,
        handlelength=2.0,
    )
    fig.subplots_adjust(bottom=0.18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)
    logger.info("Saved → %s", pdf_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="guard plot similarity",
        description=(
            "Plot cosine-similarity distributions to help choose a routing threshold.\n\n"
            "Embeds the forget and retain corpora, computes max cosine similarity of each\n"
            "document to the forget cluster centroids, and plots overlaid histograms."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "config", nargs="?", default=None,
        help="Path to a guard generate/cluster YAML config (sets dataset, model, splits).",
    )

    # HuggingFace direct mode
    hf_group = ap.add_argument_group("HuggingFace dataset (alternative to config)")
    hf_group.add_argument(
        "--hf-dataset", default=None, metavar="HF_ID",
        help="HuggingFace dataset ID (e.g. 'locuslab/TOFU').",
    )
    hf_group.add_argument(
        "--hf-forget-split", default=None, metavar="SPLIT",
        help="Dataset split to use as the forget set.",
    )
    hf_group.add_argument(
        "--hf-retain-split", default=None, metavar="SPLIT",
        help="Dataset split to use as the retain set.",
    )
    hf_group.add_argument(
        "--hf-text-key", default="text", metavar="KEY",
        help="Column name to use as document text (default: 'text').",
    )

    # Overrides that work with both modes
    ap.add_argument("--forget-split", default=None, metavar="SPLIT",
                    help="Override the forget split from config.")
    ap.add_argument("--retain-split", default=None, metavar="SPLIT",
                    help="Override the retain split from config.")

    # Plot settings
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Routing threshold to mark on the plot (default: 0.5).")
    ap.add_argument("--n-clusters", type=int, default=10, metavar="K",
                    help="K-means clusters for forget centroids (default: 10).")
    ap.add_argument("--out", default="experiments/plots/similarity.png", metavar="PATH",
                    help="Output PNG path (a PDF is also saved alongside).")
    ap.add_argument("--fineweb-n", type=int, default=5000, metavar="N",
                    help="FineWeb-Edu reference sample size, 0 to disable (default: 5000).")
    ap.add_argument("--st-model", default="all-MiniLM-L6-v2", metavar="MODEL",
                    help="SentenceTransformer model name (default: all-MiniLM-L6-v2).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    forget_texts, retain_texts = _resolve_datasets(
        config_path=args.config,
        hf_dataset=args.hf_dataset,
        hf_forget_split=args.hf_forget_split,
        hf_retain_split=args.hf_retain_split,
        hf_text_key=args.hf_text_key,
        forget_split_override=args.forget_split,
        retain_split_override=args.retain_split,
    )

    plot_similarity_distributions(
        forget_texts=forget_texts,
        retain_texts=retain_texts,
        out_path=Path(args.out),
        n_clusters=args.n_clusters,
        fineweb_n=args.fineweb_n,
        threshold=args.threshold,
        st_model_name=args.st_model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
