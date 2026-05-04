"""CLI entry point: `guard eval <sv_dir> [--model ...] [--coeffs ...]`

Runs a coefficient sweep: for each coeff, generates sample outputs from a
steered model and reports perplexity on the forget/retain docs.
Useful as a quick sanity check before running full benchmarks (TOFU, MUSE).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from tqdm import tqdm

from guard.model.loading import build_model_load_kwargs
from guard.model.steered import SteeredModel
from guard.storage.io import load_sv_with_meta

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Perplexity helper
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _perplexity(
    model: SteeredModel,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> float:
    """Compute mean perplexity of the steered model on a list of texts."""
    losses: list[float] = []
    device = model.device

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs: dict[str, Any] = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        losses.append(float(outputs.loss.item()))

    return float(torch.tensor(losses).exp().mean().item())


# ---------------------------------------------------------------------------
# Main eval logic
# ---------------------------------------------------------------------------


def run_eval(
    sv_path: Path,
    model_name: str,
    coeffs: list[float],
    forget_docs: list[str] | None,
    retain_docs: list[str] | None,
    batch_size: int,
    max_length: int,
    cuda: str | None,
    output_json: Path | None,
) -> None:
    """Run a coefficient sweep and report perplexity.

    Args:
        sv_path: Path to ``sv.pt``.
        model_name: HuggingFace model ID (overrides metadata if given).
        coeffs: List of steering coefficients to evaluate.
        forget_docs: Texts to compute forget perplexity on (optional).
        retain_docs: Texts to compute retain perplexity on (optional).
        batch_size: Inference batch size.
        max_length: Tokeniser truncation length.
        cuda: ``CUDA_VISIBLE_DEVICES`` override.
        output_json: Save results to this JSON file (optional).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    if cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    sv, meta = load_sv_with_meta(sv_path)
    resolved_model = model_name or meta.get("model_name", "")
    if not resolved_model:
        raise ValueError("model_name not found in sv.json metadata.  Provide --model explicitly.")
    layer_idx: int = int(meta["layer_idx"])
    module_name: str = str(meta.get("module_name", "residual"))
    rotation_only: bool = bool(
        meta.get("rotation_only", meta.get("norm_activation_coeff", True))
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model '{}' on {} ...", resolved_model, device)
    load_kwargs = build_model_load_kwargs(device=device)
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        **load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: dict[str, dict[str, float]] = {}

    with SteeredModel.from_sv_path(
        base_model,
        sv_path=sv_path,
        coeff=0.0,
        layer_idx=layer_idx,
        module_name=module_name,
        rotation_only=rotation_only,
    ) as steered:
        for coeff in tqdm(coeffs, desc="Coefficient sweep"):
            steered.coeff = coeff
            row: dict[str, float] = {}

            if forget_docs:
                row["forget_ppl"] = _perplexity(
                    steered, tokenizer, forget_docs, max_length, batch_size
                )
            if retain_docs:
                row["retain_ppl"] = _perplexity(
                    steered, tokenizer, retain_docs, max_length, batch_size
                )

            results[str(coeff)] = row
            parts = [f"coeff={coeff}"]
            for k, v in row.items():
                parts.append(f"{k}={v:.2f}")
            print("  ".join(parts))

    if output_json is not None:
        with output_json.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results saved to: {output_json}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="guard eval",
        description="Quick coefficient-sweep eval: perplexity on forget/retain docs.",
    )
    p.add_argument("sv_path", type=Path, help="Path to sv.pt (metadata from sv.json).")
    p.add_argument("--model", default="", help="HF model ID (overrides sv.json).")
    p.add_argument(
        "--coeffs",
        nargs="+",
        type=float,
        default=[-0.8, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0],
        metavar="COEFF",
        help="Steering coefficients to sweep (default: -0.8 -0.3 -0.2 -0.15 -0.1 -0.05 0.0).",
    )
    p.add_argument("--forget-docs", type=Path, help="JSONL file of forget texts (optional).")
    p.add_argument("--retain-docs", type=Path, help="JSONL file of retain texts (optional).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES override.")
    p.add_argument("--output", type=Path, default=None, help="Save JSON results here.")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _load_jsonl_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support 'text', 'question'+'answer', or first string value.
            if "text" in obj:
                texts.append(str(obj["text"]))
            elif "question" in obj and "answer" in obj:
                texts.append(f"{obj['question']}\n{obj['answer']}")
            else:
                first = next((v for v in obj.values() if isinstance(v, str)), None)
                if first:
                    texts.append(first)
    return texts


def main(argv: list[str] | None = None) -> None:
    """Entry point for `guard eval`."""
    parser = build_parser()
    args = parser.parse_args(argv)

    from guard.cli._logging import setup_logging
    setup_logging(args.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    forget_docs = _load_jsonl_texts(args.forget_docs) if args.forget_docs else None
    retain_docs = _load_jsonl_texts(args.retain_docs) if args.retain_docs else None

    run_eval(
        sv_path=args.sv_path,
        model_name=args.model,
        coeffs=args.coeffs,
        forget_docs=forget_docs,
        retain_docs=retain_docs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cuda=args.cuda,
        output_json=args.output,
    )
