"""MUSE benchmark evaluation using GUARD SteeredModel.

Usage
-----
    python benchmarks/muse.py benchmarks/configs/muse_books.yaml
    python benchmarks/muse.py benchmarks/configs/muse_books.yaml --cuda 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

_HERE = Path(__file__).resolve().parent  # benchmarks/
_ROOT = _HERE.parent  # guard/
_MUSE = _ROOT.parent / "muse_bench"  # stearing_vectors/muse_bench/
_MUSE_PARENT = _MUSE.parent  # stearing_vectors/

# Honour GUARD_HF_CACHE (set in .env or shell) as the HuggingFace datasets cache dir.
if _hf_cache := os.environ.get("GUARD_HF_CACHE"):
    os.environ.setdefault("HF_DATASETS_CACHE", _hf_cache)

for _p in [str(_MUSE_PARENT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from guard import GateConfig, SteeredModel

if os.environ.get("GUARD_DEBUG"):
    logger.enable("guard")


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        cand = path.with_stem(f"{path.stem}_{i}")
        if not cand.exists():
            return cand
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="MUSE eval with GUARD SteeredModel.")
    parser.add_argument("config", help="Path to YAML eval config.")
    parser.add_argument(
        "--sv-dir",
        default=None,
        metavar="PATH",
        help="Path to clustered SV layer dir (overrides config variants).",
    )
    parser.add_argument(
        "--output-path", default=None, metavar="PATH", help="Override output_path from config."
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cuda", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.cuda:
        cfg["cuda"] = args.cuda
    if args.output_path:
        cfg["output_path"] = args.output_path
    if args.cuda or (cfg.get("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda"])

    # muse_bench data files are resolved relative to its own directory
    os.chdir(str(_MUSE))

    from muse_bench.eval import eval_model  # type: ignore[import-untyped]  # noqa: E402

    output_path = _ROOT / cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name: str = cfg["model"]
    logger.info("Loading model: {}", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    # MUSE target models (Llama-2 fine-tunes) don't ship tokenizer files.
    # Fall back to the base Llama-2 tokenizer if vocab is broken (<100 tokens).
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.vocab_size < 100:
        _fallback = cfg.get("tokenizer", "open-unlearning/tofu_Llama-2-7b-chat-hf_full")
        logger.warning("Tokenizer vocab={} — falling back to {}", tokenizer.vocab_size, _fallback)
        tokenizer = AutoTokenizer.from_pretrained(_fallback)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_effective = "effective_coeffs" in cfg
    coeffs: list[float] = [
        float(c) for c in cfg.get("effective_coeffs" if use_effective else "coeffs", [])
    ]
    corpus: str = cfg.get("corpus", "books")

    # --sv-dir overrides config variants with a single clustered variant
    if args.sv_dir:
        sv_path = Path(args.sv_dir)
        if not sv_path.is_absolute():
            sv_path = _ROOT / sv_path
        # Include the method (parent of cluster dir: 'orthogonal_clustered', etc.)
        # so per_example outputs from different methods don't overwrite each other.
        _clust_dir_name = sv_path.parent.parent.name
        _method = sv_path.parent.parent.parent.name.replace("_clustered", "")
        _variant_name = f"{_method}__{_clust_dir_name}"
        variants: list[dict[str, Any]] = [
            {"name": _variant_name, "clustered_sv_path": str(sv_path)}
        ]
        clustered = True
    else:
        clustered = "clustered_variants" in cfg
        variants = cfg.get("clustered_variants", []) if clustered else cfg.get("variants", [])

    all_results: dict[str, Any] = {}

    for variant in variants:
        name: str = variant.get("name", "unnamed")
        logger.info("Variant: {}", name)

        sv_dir = Path(variant.get("clustered_sv_path", variant.get("sv_path", "")))
        if not sv_dir.is_absolute():
            sv_dir = _ROOT / sv_dir

        if clustered:
            gw = GateConfig(
                enabled=True,
                routing_source=cfg.get("routing_source", "text"),
                threshold=float(cfg.get("routing_threshold", cfg.get("cluster_threshold", 0.5))),
                routing_mode=cfg.get("routing_mode", cfg.get("cluster_routing_mode", "threshold")),
                dissimilar_top_k=int(cfg.get("dissimilar_top_k", 3)),
                dissimilar_fallback_k=int(cfg.get("dissimilar_fallback_k", 3)),
                bilateral_farthest_top_k=int(cfg.get("bilateral_farthest_top_k", 4)),
                log_routing=bool(cfg.get("log_routing", False)),
            )
            steered_ctx = SteeredModel.from_cluster_dir(
                model=model,
                cluster_dir=sv_dir,
                coeff=0.0,
                gate_cfg=gw,
                layer_idx=int(cfg["layer"]),
                module_name=cfg.get("module_name", "residual"),
                tokenizer=tokenizer,
            )
        else:
            pt = sv_dir / "sv.pt"
            steered_ctx = SteeredModel.from_sv_path(
                model=model,
                sv_path=pt,
                coeff=0.0,
                layer_idx=int(cfg["layer"]),
                module_name=cfg.get("module_name", "residual"),
                rotation_only=bool(
                    cfg.get("rotation_only", cfg.get("norm_activation_coeff", True))
                ),
            )

        variant_results: dict[str, Any] = {}

        with steered_ctx as steered:
            for coeff in coeffs:
                steered.coeff = coeff
                logger.info("  coeff={:.4f}", coeff)
                from muse_bench.constants import SUPPORTED_METRICS  # noqa: E402

                # Pass knowmem paths explicitly — constants.py has None for books
                # even though the files exist.
                _data_root = Path("data") / corpus if corpus else None
                # Save per-example outputs (prompt/gt/generation) next to the
                # aggregated JSON so we can inspect what the steered model wrote.
                _coeff_tag = (
                    f"neg{abs(coeff):.2f}".replace(".", "p")
                    if coeff < 0
                    else f"{coeff:.2f}".replace(".", "p")
                )
                _per_ex_dir = output_path.parent / "per_example" / name / _coeff_tag
                _per_ex_dir.mkdir(parents=True, exist_ok=True)
                result = eval_model(
                    model=steered,
                    tokenizer=tokenizer,
                    corpus=corpus,
                    metrics=list(SUPPORTED_METRICS),
                    knowmem_forget_qa_file=str(_data_root / "knowmem/forget_qa.json")
                    if _data_root
                    else None,
                    knowmem_forget_qa_icl_file=str(_data_root / "knowmem/forget_qa_icl.json")
                    if _data_root
                    else None,
                    knowmem_retain_qa_file=str(_data_root / "knowmem/retain_qa.json")
                    if _data_root
                    else None,
                    knowmem_retain_qa_icl_file=str(_data_root / "knowmem/retain_qa_icl.json")
                    if _data_root
                    else None,
                    privleak_batch_size=int(cfg.get("privleak_batch_size", 8)),
                    gen_batch_size=int(cfg.get("gen_batch_size", 4)),
                    temp_dir=str(_per_ex_dir),
                )
                variant_results[str(coeff)] = result
                logger.info("  → {}", result)

        all_results[name] = variant_results

    out = _unique_path(output_path)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Results saved: {}", out)


if __name__ == "__main__":
    main()
