"""TOFU summary metrics reported in the paper.

    Mem  = HM(1 - ES, 1 - EM, 1 - Para.Prob, 1 - TR)
    Util = model_utility
    Gibb = forget_Q_A_gibberish
    Avg  = HM(Util, Mem)

``TR`` is open-unlearning's ``forget_truth_ratio``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["harmonic_mean", "summarize", "load_eval"]


def load_eval(path: str | Path) -> dict[str, Any]:
    """Load a TOFU_EVAL.json file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _agg(ev: dict[str, Any], key: str) -> float | None:
    entry = ev.get(key)
    if isinstance(entry, dict):
        entry = entry.get("agg_value")
    return None if entry is None else float(entry)


def harmonic_mean(vals: list[float | None]) -> float | None:
    """Plain harmonic mean. None if any value is missing; 0 if any value is <= 0
    (a component at 0 means that criterion is completely failed)."""
    if not vals or any(v is None for v in vals):
        return None
    if any(v <= 0 for v in vals):  # type: ignore[operator]
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)  # type: ignore[operator]


def summarize(ev: dict[str, Any]) -> dict[str, float | None]:
    """Util / Mem / Avg / Gibb plus the four Mem components for one run."""
    es = _agg(ev, "extraction_strength")
    em = _agg(ev, "exact_memorization")
    pp = _agg(ev, "forget_Q_A_PARA_Prob")
    tr = _agg(ev, "forget_truth_ratio")
    util = _agg(ev, "model_utility")
    mem = harmonic_mean([None if x is None else 1.0 - x for x in (es, em, pp, tr)])
    return {
        "Util": util,
        "Mem": mem,
        "Avg": harmonic_mean([util, mem]) if util is not None and mem is not None else None,
        "Gibb": _agg(ev, "forget_Q_A_gibberish"),
        "ES": es,
        "EM": em,
        "ParaProb": pp,
        "TR": tr,
    }
