"""CLI entry point: `guard generate <config.yaml>`."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from guard.offline.activations import extract_mean_activations_multilayer
from guard.offline.steering import compute_steering_vector
from guard.config.generation import LocalJSONLDatasetConfig, TofuDatasetConfig
from guard.config.loader import load_generation_config, override_config
from guard.model.loading import build_model_load_kwargs
from guard.storage.io import documents_hash, save_sv

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_documents(cfg: Any) -> tuple[list[str], list[str]]:
    """Load forget and retain document lists from the dataset config."""
    ds = cfg.dataset
    limit = cfg.limit if cfg.limit is not None else getattr(ds, "limit", None)

    if isinstance(ds, TofuDatasetConfig):
        from datasets import load_dataset as hf_load  # type: ignore[import-untyped]

        forget_split = ds.forget_split
        retain_split = ds.inferred_retain_split()
        logger.info("Loading TOFU: forget={}  retain={}", forget_split, retain_split)
        forget_ds = hf_load("locuslab/TOFU", name=forget_split, split="train")
        retain_ds = hf_load("locuslab/TOFU", name=retain_split, split="train")
        forget_docs: list[str] = [f"{x['question']}\n{x['answer']}" for x in forget_ds]
        retain_docs: list[str] = [f"{x['question']}\n{x['answer']}" for x in retain_ds]

    elif isinstance(ds, LocalJSONLDatasetConfig):

        def _read_jsonl(path: str, key: str, lim: int | None) -> list[str]:
            docs = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get(key, "")
                    if text:
                        docs.append(str(text))
            return docs[:lim] if lim is not None else docs

        forget_docs = _read_jsonl(ds.forget_jsonl, ds.text_key, limit)
        retain_docs = _read_jsonl(ds.retain_jsonl, ds.text_key, limit)
        logger.info(
            "LocalJSONL: {} forget docs, {} retain docs",
            len(forget_docs),
            len(retain_docs),
        )
        return forget_docs, retain_docs  # limit already applied

    else:
        raise ValueError(f"Unsupported dataset type: {type(ds)}")

    if limit is not None:
        forget_docs = forget_docs[:limit]
        retain_docs = retain_docs[:limit]

    logger.info(
        "Dataset loaded: %d forget docs, %d retain docs",
        len(forget_docs),
        len(retain_docs),
    )
    return forget_docs, retain_docs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(cfg: Any) -> tuple[Any, Any]:
    """Load the model and tokenizer from the generation config."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        **load_kwargs,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_generate(cfg: Any, config_path: Path | None = None, dry_run: bool = False) -> None:
    """Generate steering vectors for all layers in the config.

    Args:
        cfg: A validated :class:`~guard.config.GenerationConfig`.
        config_path: Path to the originating YAML (copied into each SV folder).
        dry_run: If ``True``, validate and print output paths without running inference.
    """
    behavior = cfg.effective_behavior()
    output_paths = [
        str(
            Path(cfg.output_dir)
            / cfg.model_name.replace("/", "_")
            / cfg.method
            / behavior
            / cfg.module_name
            / f"layer_{li}"
            / "sv.pt"
        )
        for li in cfg.layers
    ]

    if dry_run:
        print("Dry run — would write:")
        for p in output_paths:
            print(f"  {p}")
        return

    _set_seed(cfg.seed)
    model, tokenizer = _load_model_and_tokenizer(cfg)
    forget_docs, retain_docs = _load_documents(cfg)

    logger.info(
        "Capturing activations: {} layers, {} forget docs, {} retain docs",
        len(cfg.layers),
        len(forget_docs),
        len(retain_docs),
    )

    forget_results = extract_mean_activations_multilayer(
        model=model,
        tokenizer=tokenizer,
        documents=forget_docs,
        layer_indices=cfg.layers,
        module_name=cfg.module_name,
        token_position=cfg.token_position,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        add_special_tokens=cfg.add_special_tokens,
    )

    retain_results = extract_mean_activations_multilayer(
        model=model,
        tokenizer=tokenizer,
        documents=retain_docs,
        layer_indices=cfg.layers,
        module_name=cfg.module_name,
        token_position=cfg.token_position,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        add_special_tokens=cfg.add_special_tokens,
    )

    f_hash = documents_hash(forget_docs)
    r_hash = documents_hash(retain_docs)

    layer_methods: dict[int, str] = dict(cfg.layer_methods or {})

    for li in tqdm(cfg.layers, desc="Computing steering vectors", unit="layer"):
        forget_mean, forget_norm = forget_results[li]
        retain_mean, retain_norm = retain_results[li]

        effective_method = layer_methods.get(li, cfg.method)

        sv, activation_norm = compute_steering_vector(
            forget_mean=forget_mean,
            forget_norm=forget_norm,
            retain_mean=retain_mean,
            retain_norm=retain_norm,
            method=effective_method,
            norm_cfg=cfg.normalizations,
        )

        metadata = {
            "token_position": cfg.token_position
            if isinstance(cfg.token_position, str)
            else int(cfg.token_position),
            "sv_scaling": cfg.normalizations.sv_scaling.value,
            "rotation_only": cfg.normalizations.rotation_only,
            "projection_eps": cfg.normalizations.projection_eps,
            "activation_norm": float(activation_norm),
            "forget_norm": float(forget_norm),
            "retain_norm": float(retain_norm),
            "n_forget_docs": len(forget_docs),
            "n_retain_docs": len(retain_docs),
            "forget_docs_hash": f_hash,
            "retain_docs_hash": r_hash,
            "dataset_type": cfg.dataset.type,
            "seed": cfg.seed,
            "max_length": cfg.max_length,
            "batch_size": cfg.batch_size,
            "effective_method": effective_method,
        }
        if effective_method != cfg.method:
            logger.info("Layer {}: using method '{}' (override)", li, effective_method)

        pt_path = save_sv(
            sv=sv,
            output_dir=cfg.output_dir,
            model_name=cfg.model_name,
            behavior=behavior,
            module_name=cfg.module_name,
            layer_idx=li,
            method=cfg.method,
            metadata=metadata,
            config_source=config_path,
        )
        logger.info("Saved: {}  (effective_method={}  activation_norm={:.4f})", pt_path, effective_method, activation_norm)

    print(f"Done. Steering vectors saved to: {Path(cfg.output_dir).resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="guard generate",
        description="Compute and save GUARD steering vectors from a YAML config.",
    )
    p.add_argument("config", type=Path, help="Path to generate.yaml")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config field (repeatable).  Supports dot notation for "
            "nested keys, e.g. --override normalizations.sv_scaling=unit"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print output paths without running inference.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for `guard generate`."""
    import logging as _logging
    from guard.cli._logging import setup_logging
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)

    cfg = load_generation_config(args.config)
    if args.override:
        cfg = override_config(cfg, args.override)

    run_generate(cfg, config_path=args.config, dry_run=args.dry_run)
