"""TOFU benchmark evaluation using GUARD SteeredModel.

Supports both single-SV and clustered (gateway) routing modes.
Model is loaded once; coefficient is updated in-place between sweeps.

Usage
-----
    # single SV
    python benchmarks/tofu.py benchmarks/configs/tofu_forget01.yaml

    # clustered gateway
    python benchmarks/tofu.py benchmarks/configs/tofu_forget01_clustered.yaml

    # override batch size
    python benchmarks/tofu.py benchmarks/configs/tofu_forget01.yaml --batch-size 16
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # benchmarks/
_ROOT = _HERE.parent  # guard/
_OPEN_UNL = _HERE / "open-unlearning"  # benchmarks/open-unlearning/

# Honour GUARD_HF_CACHE (set in .env or shell) as the HuggingFace datasets cache dir.
if _hf_cache := os.environ.get("GUARD_HF_CACHE"):
    os.environ.setdefault("HF_DATASETS_CACHE", _hf_cache)

for _p in [str(_OPEN_UNL / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
import importlib.util as _ilu

# Stub evals.lm_eval so evals/__init__.py doesn't try to import the real lm_eval package.
# We only need TOFUEvaluator; LMEvalEvaluator is never used here.
import types as _types

from transformers import AutoModelForCausalLM, AutoTokenizer

from guard import GateConfig, SteeredModel
from guard.model.loading import build_model_load_kwargs

_evals_lm_eval_stub = _types.ModuleType("evals.lm_eval")
_evals_lm_eval_stub.LMEvalEvaluator = type("LMEvalEvaluator", (), {})  # type: ignore[attr-defined]
sys.modules["evals.lm_eval"] = _evals_lm_eval_stub

_spec = _ilu.spec_from_file_location("evals.tofu", _OPEN_UNL / "src" / "evals" / "tofu.py")
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
TOFUEvaluator = _mod.TOFUEvaluator

import os as _os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import OmegaConf

if _os.environ.get("GUARD_DEBUG") == "1":
    logger.enable("guard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_eval_cfg(cfg: dict[str, Any], output_base: Path) -> Any:
    retain_logs = cfg.get("retain_logs_path") or None
    eval_variant = "tofu"
    forget_split = cfg.get("forget_split", "forget01")
    holdout_split = cfg.get("holdout_split", forget_split.replace("forget", "holdout"))
    batch_size = cfg.get("batch_size", 8)
    gibberish_device = cfg.get("gibberish_device", "cpu")

    overrides = [
        "model=Llama-3.2-1B-Instruct",
        f"eval={eval_variant}",
        f"eval.tofu.forget_split={forget_split}",
        f"eval.tofu.holdout_split={holdout_split}",
        f"eval.tofu.batch_size={batch_size}",
        f"eval.tofu.metrics.forget_Q_A_gibberish.device={gibberish_device}",
        f"paths.output_dir={output_base}",
        "task_name=grave_steered",
    ]
    if retain_logs:
        overrides.append(f"eval.tofu.retain_logs_path={retain_logs}")
    else:
        overrides.append("eval.tofu.retain_logs_path=null")

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(_OPEN_UNL / "configs"), version_base=None)
    hydra_cfg = compose(config_name="eval.yaml", overrides=overrides)
    resolved = OmegaConf.to_container(hydra_cfg.eval.tofu, resolve=True, throw_on_missing=False)
    return OmegaConf.create(resolved)


def _template_args(cfg: dict[str, Any]) -> Any:
    ta = cfg.get("template_args", {})
    # Llama-3.2-1B-Instruct was fine-tuned WITH the chat template.
    # Defaults match open-unlearning/configs/model/Llama-3.2-1B-Instruct.yaml.
    defaults: dict[str, Any] = {
        "apply_chat_template": True,
        "system_prompt": "You are a helpful assistant.",
        "system_prompt_with_special_tokens": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
        ),
        "user_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end_tag": "<|eot_id|>",
        "asst_start_tag": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "asst_end_tag": "<|eot_id|>",
    }
    defaults.update(ta)
    return OmegaConf.create(defaults)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_stem(f"{path.stem}_{ts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="TOFU eval with GUARD SteeredModel.")
    parser.add_argument("config", help="Path to YAML eval config.")
    parser.add_argument(
        "--sv-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the layer_N/ clustered SV directory "
            "(e.g. steering_vectors/.../forget01_K3/residual/layer_8). "
            "Overrides any clustered_variants in the config."
        ),
    )
    parser.add_argument("--retain-logs", default=None)
    parser.add_argument("--output-path", default=None, metavar="PATH")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cuda", default=None)
    parser.add_argument(
        "--coeffs",
        default=None,
        metavar="C1,C2,...",
        help="Comma-separated list of coefficients to evaluate, overriding config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a flat config key, e.g. --override routing_threshold=0.3",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if args.retain_logs:
        cfg["retain_logs_path"] = args.retain_logs
    if args.output_path:
        cfg["output_path"] = args.output_path
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.cuda:
        cfg["cuda"] = args.cuda
    if args.coeffs:
        key = "effective_coeffs" if "effective_coeffs" in cfg else "coeffs"
        cfg[key] = [float(c) for c in args.coeffs.split(",")]
    for ov in args.override:
        k, _, v = ov.partition("=")
        # try yaml.safe_load first (handles lists, bools, ints, floats, null, strings)
        try:
            v = yaml.safe_load(v)
        except yaml.YAMLError:
            pass  # keep as string
        cfg[k] = v
    if cfg.get("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda"])

    # --sv-dir overrides any clustered_variants in the config
    if args.sv_dir:
        sv_dir = Path(args.sv_dir)
        cluster_name = sv_dir.parent.parent.name  # e.g. forget01_Kf2_Kr4
        method_dir = sv_dir.parent.parent.parent.name  # e.g. orthogonal_clustered
        method_tag = method_dir.replace("_clustered", "")  # e.g. orthogonal
        # include routing_threshold + timestamp in variant name for traceability
        _thresh = cfg.get("routing_threshold", cfg.get("cluster_threshold", 0.5))
        _thresh_tag = f"t{str(_thresh).replace('.', 'p')}"
        from datetime import datetime as _dt

        _ts = _dt.now().strftime("%Y%m%d_%H%M")
        sv_dir_name = f"{method_tag}_{cluster_name}_{_thresh_tag}_{_ts}"
        cfg["_sv_dir_override"] = args.sv_dir
        cfg["_sv_dir_name"] = sv_dir_name

    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load model once ---
    model_name: str = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model: {}", model_name)
    load_kwargs = build_model_load_kwargs(
        device=device,
        quantization=cfg.get("quantization", "none"),
        quant_4bit_type=cfg.get("quant_4bit_type", "nf4"),
        quant_4bit_double_quant=cfg.get("quant_4bit_double_quant", True),
        quant_compute_dtype=cfg.get("quant_compute_dtype", "bfloat16"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If forget_jsonl is set, monkey-patch datasets.load_dataset so the evaluator
    # loads paraphrased Q&A instead of the original TOFU split.
    _ds_patch_restore = None
    if cfg.get("forget_jsonl"):
        import datasets as _hf_ds

        _forget_jsonl = str(Path(cfg["forget_jsonl"]))
        _forget_split = cfg.get("forget_split", "forget01")
        _para_data = _hf_ds.load_dataset("json", data_files={"train": _forget_jsonl}, split="train")
        _original_load = _hf_ds.load_dataset

        def _patched_load(path, name=None, **kwargs):
            if path == "locuslab/TOFU" and name == _forget_split:
                return _para_data
            return _original_load(path, name=name, **kwargs)

        _hf_ds.load_dataset = _patched_load
        _ds_patch_restore = (_hf_ds, _original_load)
        logger.info(
            "forget_jsonl active: swapping forget split '%s' with %s", _forget_split, _forget_jsonl
        )

    eval_cfg = _build_eval_cfg(cfg, output_path.parent)
    template_args = _template_args(cfg)

    # --- coefficient list (effective = model-agnostic) ---
    use_effective = "effective_coeffs" in cfg
    coeffs: list[float] = [
        float(c) for c in cfg.get("effective_coeffs" if use_effective else "coeffs", [])
    ]

    if cfg.get("_sv_dir_override"):
        clustered = True
        variants = [{"name": cfg["_sv_dir_name"], "clustered_sv_path": cfg["_sv_dir_override"]}]
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

        # --- build SteeredModel ---
        # Multi-layer: cfg["layers"] = [4, 8, 9] and sv_dir points to the module dir
        #              (e.g. residual/) so layer dirs are sv_dir/layer_N/.
        # Single-layer: cfg["layer"] = 8 and sv_dir already points to the layer dir.
        _layer_cfg = cfg.get("layers", cfg.get("layer"))
        _layer_list: list[int] = (
            [int(x) for x in _layer_cfg] if isinstance(_layer_cfg, list) else [int(_layer_cfg)]
        )
        _multi_layer = len(_layer_list) > 1

        if clustered:
            # For calibration, load from the first layer dir.
            _first_layer_dir = sv_dir / f"layer_{_layer_list[0]}" if _multi_layer else sv_dir
            _routing_src = cfg.get("routing_source", "text")
            _calib_path = _first_layer_dir / "routing_threshold.json"
            if (
                _routing_src == "activation"
                and (_first_layer_dir / "routing_threshold_cosine.json").exists()
            ):
                _calib_path = _first_layer_dir / "routing_threshold_cosine.json"
            _calib_threshold: float | None = None
            if _calib_path.exists() and "routing_threshold" not in cfg:
                import json as _json

                _calib = _json.loads(_calib_path.read_text())
                _calib_threshold = float(_calib["threshold"])
                logger.info(
                    "Auto-loaded calibrated threshold=%.4f from %s "
                    "(recall_forget=%.3f fpr_retain=%.3f alpha=%.1f)",
                    _calib_threshold,
                    _calib_path,
                    _calib.get("recall_forget", float("nan")),
                    _calib.get("fpr_retain", float("nan")),
                    _calib.get("alpha", float("nan")),
                )
            _threshold = (
                _calib_threshold
                if _calib_threshold is not None
                else float(cfg.get("routing_threshold", cfg.get("cluster_threshold", 0.5)))
            )
            gw = GateConfig(
                enabled=True,
                routing_source=cfg.get("routing_source", "text"),
                threshold=_threshold,
                routing_mode=cfg.get("routing_mode", cfg.get("cluster_routing_mode", "threshold")),
                dissimilar_top_k=int(cfg.get("dissimilar_top_k", 3)),
                dissimilar_fallback_k=int(cfg.get("dissimilar_fallback_k", 3)),
                retain_threshold=float(cfg.get("retain_threshold", 0.0)),
                log_routing=_os.environ.get("GUARD_DEBUG") == "1",
            )
            steered_ctx = [
                SteeredModel.from_cluster_dir(
                    model=model,
                    cluster_dir=sv_dir / f"layer_{li}" if _multi_layer else sv_dir,
                    coeff=0.0,
                    gate_cfg=gw,
                    layer_idx=li,
                    module_name=cfg.get("module_name", "residual"),
                    tokenizer=tokenizer,
                )
                for li in _layer_list
            ]
        else:
            pt = sv_dir / "sv.pt"
            steered_ctx = [
                SteeredModel.from_sv_path(
                    model=model,
                    sv_path=pt,
                    coeff=0.0,
                    layer_idx=_layer_list[0],
                    module_name=cfg.get("module_name", "residual"),
                    rotation_only=bool(
                        cfg.get("rotation_only", cfg.get("norm_activation_coeff", True))
                    ),
                )
            ]

        variant_results: dict[str, Any] = {}
        evaluator = TOFUEvaluator(eval_cfg)

        with contextlib.ExitStack() as _stack:
            steered_list = [_stack.enter_context(ctx) for ctx in steered_ctx]
            steered = steered_list[0]  # used for eval + coeff display; all share the same model

            for coeff in coeffs:
                for _sm in steered_list:
                    _sm.coeff = coeff
                coeff_str = str(coeff).replace(".", "p").replace("-", "neg")
                coeff_dir = _unique_path(output_path.parent / "logs" / name / coeff_str)
                coeff_dir.mkdir(parents=True, exist_ok=True)

                # Save effective config so this run is self-documenting
                _cluster_meta: dict[str, Any] = {}
                _meta_path = sv_dir / "cluster_meta.json"
                if _meta_path.exists():
                    with _meta_path.open(encoding="utf-8") as _mf:
                        _cluster_meta = json.load(_mf)
                _run_meta = {
                    "timestamp": coeff_dir.stem.split("_", 1)[1] if "_" in coeff_dir.stem else "",
                    "config_file": args.config,
                    "sv_dir": str(sv_dir),
                    # model
                    "model": cfg.get("model", _cluster_meta.get("model_name", "")),
                    # steering
                    "coeff": coeff,
                    "layer": _layer_list if len(_layer_list) > 1 else _layer_list[0],
                    "module_name": cfg.get(
                        "module_name", _cluster_meta.get("module_name", "residual")
                    ),
                    "method": _cluster_meta.get("method", ""),
                    "rotation_only": cfg.get(
                        "rotation_only",
                        cfg.get(
                            "norm_activation_coeff",
                            _cluster_meta.get(
                                "rotation_only", _cluster_meta.get("norm_activation_coeff", True)
                            ),
                        ),
                    ),
                    # clustering
                    "n_forget_clusters": _cluster_meta.get("n_clusters", ""),
                    "token_position": _cluster_meta.get("token_position", ""),
                    # routing (inference-time)
                    "routing_source": cfg.get("routing_source", "text"),
                    "routing_threshold": cfg.get("routing_threshold", 0.5),
                    "routing_mode": cfg.get("routing_mode", "threshold"),
                    "retain_threshold": cfg.get("retain_threshold", 0.0),
                    # dataset
                    "forget_split": cfg.get("forget_split", _cluster_meta.get("behavior", "")),
                    # run
                    "overrides": getattr(args, "override", []),
                    "guard_version": _cluster_meta.get("grave_version", ""),
                }
                (coeff_dir / "run_config.json").write_text(
                    json.dumps(_run_meta, indent=2), encoding="utf-8"
                )

                logger.info("  coeff={:.4f}", coeff)
                result = evaluator.evaluate(
                    model=steered,
                    tokenizer=tokenizer,
                    template_args=template_args,
                    output_dir=str(coeff_dir),
                )
                # Augment summary with sub-metrics from TOFU_EVAL.json
                _eval_json = coeff_dir / "TOFU_EVAL.json"
                if _eval_json.exists():
                    with _eval_json.open(encoding="utf-8") as _ef:
                        _full = json.load(_ef)
                    for _key in (
                        "retain_Q_A_ROUGE",
                        "retain_Q_A_Prob",
                        "retain_Truth_Ratio",
                        "exact_memorization",
                        "forget_Q_A_PARA_Prob",
                        "forget_truth_ratio",
                        "forget_Q_A_gibberish",
                    ):
                        _entry = _full.get(_key)
                        if isinstance(_entry, dict) and "agg_value" in _entry:
                            result[_key] = _entry["agg_value"]

                # mem = harmonic_mean(1 - col) for the 4 forgetting components
                _mem_cols = [
                    "extraction_strength",
                    "exact_memorization",
                    "forget_Q_A_PARA_Prob",
                    "forget_truth_ratio",
                ]
                _mem_vals = [
                    1.0 - result[c] for c in _mem_cols if c in result and result[c] is not None
                ]
                if len(_mem_vals) == len(_mem_cols) and all(v > 0 for v in _mem_vals):
                    result["mem"] = len(_mem_vals) / sum(1.0 / v for v in _mem_vals)
                else:
                    result["mem"] = 0.0

                # overall = harmonic_mean(mem, model_utility)
                _mu = result.get("model_utility")
                _mem = result.get("mem")
                if _mu and _mem:
                    result["overall"] = 2.0 / (1.0 / _mu + 1.0 / _mem)
                variant_results[str(coeff)] = result
                logger.info("  → {}", result)

        all_results[name] = variant_results

    if _ds_patch_restore:
        _ds_patch_restore[0].load_dataset = _ds_patch_restore[1]

    out = _unique_path(output_path)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Results saved: {}", out)


if __name__ == "__main__":
    main()
